"""Complete four-family correction testing and compare saved direct/correction results."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

import compare_stage2_correction as correction
import compare_stage2_direct as direct
import evaluate_stage2 as legacy
from audit_stage2_readiness import digest, solar_elevation
from evaluate_stage2_quantum import KEYS, aligned_inputs, scores_with_r2
from stage2_shared import preserve_plan

OUT = correction.OUT/"test"


def selected_bundles(hpo, ledger, reference):
    plan = hpo["plan"]
    candidates = {c["id"]: c for c in plan["candidates"]}
    if (ledger.id.duplicated().any() or set(ledger.id) != set(candidates)
            or not ledger.status.eq("complete").all() or not ledger.error.fillna("").eq("").all()
            or not ledger.daylight_n.eq(reference["daylight_n"]).all()
            or not ledger.all_day_n.eq(plan["counts"]["validation"]["eligible"]).all()
            or not np.isfinite(ledger.daylight_mae_kw).all()):
        raise ValueError("Incomplete/incompatible correction HPO trial ledger")
    baseline = joblib.load(correction.OUT/"ridge.joblib")
    fitted_identity = joblib.hash((baseline["processor"], baseline["scaler"]))
    selected, bundles = {}, {}
    for family in correction.MODELS:
        rows = ledger.loc[ledger.family.eq(family)].to_dict("records")
        if not rows:
            raise ValueError(f"Missing correction {family} trial records")
        for row in rows:
            candidate = candidates[row["id"]]
            if (candidate["family"] != family or correction.canonical(json.loads(row["config"]))
                    != correction.canonical(candidate["config"])):
                raise ValueError(f"Correction trial configuration differs: {row['id']}")
        winner = min(rows, key=lambda row: (row["daylight_mae_kw"], list(candidates).index(row["id"])))
        candidate = candidates[winner["id"]]
        recorded = hpo["best_per_family"][family]
        if (recorded["trial"] != winner["id"] or recorded["config"] != candidate["config"]
                or not np.isclose(recorded["tuned_daylight_mae_kw"], winner["daylight_mae_kw"], rtol=1e-12, atol=1e-9)):
            raise ValueError(f"Wrong validation-selected correction {family}")
        path = correction.OUT/"hpo"/f"best_{family}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; restore this exact winning bundle from backup. No automatic retraining.")
        bundle = joblib.load(path)
        model = bundle["model"]
        for obj in (model, bundle["processor"], bundle["scaler"]):
            check_is_fitted(obj)
        if (bundle["features"] != correction.FEATURES or bundle["trial"]["id"] != winner["id"]
                or bundle["trial"]["status"] != "complete"
                or correction.canonical(json.loads(bundle["trial"]["config"])) != correction.canonical(candidate["config"])
                or type(model) is not type(correction.estimator(family, correction.MODELS[family]))
                or correction.canonical(model.get_params()) != correction.canonical(candidate["effective_settings"])
                or joblib.hash((bundle["processor"], bundle["scaler"])) != fitted_identity
                or model.n_features_in_ != len(bundle["processor"].get_feature_names_out())):
            raise ValueError(f"Incompatible fitted correction bundle/preprocessing: {family}")
        for name in ("daylight_mae_kw", "daylight_rmse_kw", "daylight_bias_kw", "all_day_mae_kw"):
            np.testing.assert_allclose(bundle["trial"][name], winner[name], rtol=1e-12, atol=1e-9)
        selected[family] = dict(trial=winner["id"], configuration=candidate["config"],
                                effective_settings=candidate["effective_settings"])
        bundles[family] = bundle
    overall = min(ledger.to_dict("records"), key=lambda row: (row["daylight_mae_kw"], list(candidates).index(row["id"])))
    if hpo["selected_trial"] != overall["id"]:
        raise ValueError("Overall correction validation selection differs")
    return selected, bundles


def frozen_artifacts():
    path = correction.OUT/"hpo/settings.json"
    hpo = json.loads(path.read_text())
    plan = hpo["plan"]
    saved = json.loads((legacy.OUT/"settings.json").read_text())
    reference = saved["neural_reference"]
    if (plan["protocol"] != "classical-correction-hpo-v1" or plan["split"] != legacy.SPLIT
            or plan["features"] != correction.FEATURES or saved["protocol"] != "stage2-frozen-test-v1"
            or saved["split"] != plan["split"] or saved["selected_trial"] != hpo["selected_trial"]
            or plan["counts"] != reference["counts"] or plan["frame_fingerprints"] != reference["frame_fingerprints"]
            or correction.canonical(plan["candidates"]) != correction.canonical(correction.search_candidates())):
        raise ValueError("Correction training/validation protocol differs from frozen ML/DL reference")
    if {n: metadata.version(n) for n in plan["versions"]} != plan["versions"]:
        raise ValueError("Use the saved correction software versions")
    hashes = {}
    for mapping in (plan["source_hashes"], plan["baseline_hashes"], saved["source_hashes"]):
        direct.verify_hashes(mapping)
        hashes.update(mapping)
    if digest(correction.OUT/"ridge.joblib") != reference["preprocessing_sha256"]:
        raise ValueError("Original training-fitted preprocessing changed")
    selected, bundles = selected_bundles(hpo, pd.read_csv(correction.OUT/"hpo/trials.csv"), reference)
    paths = [path, correction.OUT/"hpo/trials.csv", Path(__file__), Path("scripts/compare_stage2_direct.py"),
             Path("scripts/evaluate_stage2_quantum.py"), *[correction.OUT/"hpo"/f"best_{f}.joblib" for f in correction.MODELS]]
    paths += [legacy.OUT/n for n in ("settings.json", "predictions.parquet", "metrics.csv")]
    paths += [legacy.WEATHER/n for n in
              ("forecasts_testing.parquet", "physics_testing.parquet", "testing_settings.json", "physics_testing_settings.json")]
    hashes.update({p.as_posix(): digest(p) for p in paths})
    result = dict(protocol="stage2-four-family-correction-test-v1", split=legacy.SPLIT, features=correction.FEATURES,
                  selected=selected, validation_preferred_trial=hpo["selected_trial"], versions=plan["versions"],
                  source_hashes=hashes, rf_reuse="Original ML/DL predictions.parquet:forest_05; no RF inference",
                  sign="correction=actual-physics; final=physics+predicted correction; bias=prediction-actual; no clipping",
                  scoring="Exact original ML/DL rows; target elevation>0 daylight; R2 undefined for constant targets",
                  disclosure=saved["disclosure"])
    return preserve_plan(OUT/"settings.json", result), bundles


def predictions_from_bundles(bundles, features, saved, day):
    if list(features.columns) != correction.FEATURES or not features.index.equals(saved.index):
        raise ValueError("Correction features or origins differ")
    prediction = saved[KEYS].copy()
    prediction["daylight"] = day
    physics = saved.physics_forecast_kw.to_numpy()
    for family, bundle in bundles.items():
        if family == "forest":
            final = saved.forest_05.to_numpy()
            delta = final-physics
        else:
            x = bundle["processor"].transform(features)
            if family == "ridge":
                x = bundle["scaler"].transform(x)
            delta = bundle["model"].predict(x)
            final = legacy.neural.final_power(physics, delta, 1.)
        prediction[f"predicted_correction_{family}_kw"] = delta
        prediction[f"correction_{family}"] = final
        np.testing.assert_allclose(final, physics+delta, rtol=1e-12, atol=1e-9)
    return prediction


def evaluate(plan, bundles):
    saved = pd.read_parquet(legacy.OUT/"predictions.parquet")
    features, table, day, _ = legacy.load_testing_examples()
    aligned_inputs(saved, features, table, day, pd.read_csv(legacy.OUT/"metrics.csv"))
    np.testing.assert_allclose(table.correction_kw, table.actual_target_power_kw-table.physics_forecast_kw, rtol=1e-12, atol=1e-9)
    prediction = predictions_from_bundles(bundles, features, saved, day)
    columns = ["correction_"+f for f in correction.MODELS]
    scores = scores_with_r2(prediction, columns, day)
    direct.verify_hashes(plan["source_hashes"])
    with (OUT/"predictions.parquet").open("xb") as stream:
        prediction.to_parquet(stream)
    with (OUT/"metrics.csv").open("x", newline="") as stream:
        scores.to_csv(stream)
    print(scores.to_string())
    print(plan["disclosure"])


def compare():
    destinations = [OUT/"comparison.csv", OUT/"comparison_settings.json"]
    if any(p.exists() for p in destinations):
        raise ValueError("Existing comparison; refusing overwrite")
    folders = dict(correction=OUT, direct=direct.OUT/"test", original=legacy.OUT)
    paths = [folder/n for folder in folders.values() for n in ("settings.json", "predictions.parquet", "metrics.csv")]
    paths.append(direct.OUT/"ml/selection.json")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved evaluation artifacts; run the corresponding --prepare/--prepare-test and --evaluate --allow-test commands first:\n"+"\n".join(missing))
    snapshots = {p.as_posix(): digest(p) for p in paths+[Path(__file__)]}
    settings = {name: json.loads((folder/"settings.json").read_text()) for name, folder in folders.items()}
    expected = dict(correction="stage2-four-family-correction-test-v1", direct="stage2-direct-frozen-test-v1", original="stage2-frozen-test-v1")
    for name, spec in settings.items():
        if spec["protocol"] != expected[name]:
            raise ValueError(f"Incompatible {name} evaluation protocol")
        direct.verify_hashes(spec["source_hashes"])
    if (settings["correction"]["split"] != settings["original"]["split"]
            or settings["direct"]["training_plan"]["split"] != settings["original"]["split"]):
        raise ValueError("Direct/correction evaluation splits differ")
    direct.verify_hashes(settings["direct"]["training_plan"]["source_hashes"])
    direct_selection = json.loads((direct.OUT/"ml/selection.json").read_text())
    frames = {name: pd.read_parquet(folder/"predictions.parquet") for name, folder in folders.items()}
    original = frames["original"]
    day = solar_elevation(pd.DatetimeIndex(original.target_timestamp)).to_numpy() > 0
    recorded = {name: pd.read_csv(folder/"metrics.csv", index_col=0) for name, folder in folders.items()}
    for name, frame in frames.items():
        if (not frame.index.is_unique or not frame.target_timestamp.is_unique
                or not frame.index.equals(original.index)
                or not pd.DatetimeIndex(frame.target_timestamp).equals(frame.index+pd.Timedelta(minutes=15))):
            raise ValueError(f"Different or duplicate {name} origin/target keys")
        pd.testing.assert_frame_equal(frame[KEYS], original[KEYS], check_dtype=False, check_exact=True, check_freq=False)
        if name != "original":
            np.testing.assert_array_equal(frame.daylight, day)
    np.testing.assert_array_equal(frames["correction"].correction_forest, original.forest_05)
    np.testing.assert_array_equal(frames["direct"].persistence_kw, original.persistence_kw)
    rows = []
    for scheme in ("correction", "direct"):
        columns = [scheme+"_"+f for f in correction.MODELS]
        if scheme == "correction":
            for family in correction.MODELS:
                np.testing.assert_allclose(frames[scheme][scheme+"_"+family], original.physics_forecast_kw+
                    frames[scheme][f"predicted_correction_{family}_kw"], rtol=1e-12, atol=1e-9)
        scores = scores_with_r2(frames[scheme], columns, day)
        for family, column in zip(correction.MODELS, columns):
            candidate = (settings[scheme]["selected"][family]["trial"] if scheme=="correction" else
                         direct_selection["best_per_family"][family])
            rows.append(dict(scheme=scheme, family=family, statistic="selected_model", trial=candidate, **scores.loc[column].to_dict()))
        check_scores(recorded[scheme], scores)
    for scheme, name, prefix in (("correction", "original", "classical_"), ("direct", "direct", "direct_classical_")):
        names = [prefix+str(seed) for seed in legacy.neural.SEEDS]
        scores = scores_with_r2(frames[name], names, day)
        check_scores(recorded[name], scores)
        summary = scores.drop(columns=["daylight_n", "all_day_n"]).agg(["mean", "std"])
        for statistic in summary.index:
            rows.append(dict(scheme=scheme, family="classical_neural", statistic="mean" if statistic=="mean" else "sample_std",
                             daylight_n=int(day.sum()), all_day_n=len(day), **summary.loc[statistic].to_dict()))
    reference = scores_with_r2(original, ["physics_forecast_kw", "persistence_kw", "forest_05"], day)
    check_scores(recorded["original"], reference)
    check_scores(recorded["direct"], reference.loc[["physics_forecast_kw", "persistence_kw"]])
    for name, row in reference.loc[["physics_forecast_kw", "persistence_kw"]].iterrows():
        rows.append(dict(scheme="reference", family=name, statistic="reference", **row.to_dict()))
    result = pd.DataFrame(rows)
    for family in correction.MODELS:
        a = result.scheme.eq("correction") & result.family.eq(family)
        b = result.scheme.eq("direct") & result.family.eq(family)
        result.loc[a, "daylight_mae_minus_direct_kw"] = result.loc[a,"daylight_mae_kw"].iloc[0]-result.loc[b,"daylight_mae_kw"].iloc[0]
    direct.verify_hashes(snapshots)
    with destinations[0].open("x", newline="") as stream:
        result.to_csv(stream, index=False)
    with destinations[1].open("x") as stream:
        json.dump(dict(source_hashes=snapshots, disclosure=settings["correction"]["disclosure"],
                       interpretation="Descriptive held-out comparisons; negative correction-minus-direct MAE favors correction; five-seed mean/sample std, no ensemble or new selection"), stream, indent=2)
    print(result.to_string(index=False))
    print(settings["correction"]["disclosure"])


def check_scores(recorded, calculated):
    for name, row in calculated.iterrows():
        if name not in recorded.index:
            raise ValueError(f"Missing recorded metric row: {name}")
        for column in calculated.columns.intersection(recorded.columns):
            np.testing.assert_allclose(row[column], recorded.loc[name,column], rtol=1e-12, atol=1e-9, equal_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "evaluate", "compare"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate and not args.allow_test:
        parser.error("Held-out inference requires --allow-test after reviewing --prepare")
    if args.compare:
        compare()
        return
    if args.evaluate and any((OUT/n).exists() for n in ("predictions.parquet", "metrics.csv")):
        raise ValueError("Existing four-family correction results; refusing repeated evaluation or overwrite")
    plan, bundles = frozen_artifacts()
    path = OUT/"settings.json"
    if args.prepare:
        OUT.mkdir(parents=True, exist_ok=True)
        legacy.neural.freeze(path, plan)
        print(json.dumps(plan["selected"], indent=2))
        print("Four fitted winners verified; RF predictions will be reused. No test rows loaded or predictions generated.")
        return
    if not path.exists() or correction.canonical(json.loads(path.read_text())) != correction.canonical(plan):
        raise ValueError("Run --prepare first; frozen artifacts must match exactly")
    evaluate(plan, bundles)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
