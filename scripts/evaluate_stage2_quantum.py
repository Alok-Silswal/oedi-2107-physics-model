"""Evaluate three frozen CPU quantum seeds on the saved ML/DL test rows."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import evaluate_stage2 as baseline
import train_stage2_quantum_cpu as native
from compare_stage2_correction import canonical, FEATURES
from audit_stage2_readiness import digest, solar_elevation

OUT = Path("outputs/2107_stage2_quantum_test")
KEYS = ["target_timestamp", "actual_target_power_kw", "physics_forecast_kw"]


def completed_model(seed, reference):
    directory = native.OUT/"quantum"/f"seed_{seed}"
    spec = dict(reference, model="quantum", seed=seed)
    if canonical(json.loads((directory/"settings.json").read_text())) != canonical(spec):
        raise ValueError(f"Incompatible quantum seed {seed} settings")
    best = native.read_checkpoint(directory/"best.pt", spec)
    last = native.read_checkpoint(directory/"last.pt", spec)
    result = json.loads((directory/"metrics.json").read_text())
    selected, score, stale = 0, float("inf"), 0
    for epoch, row in enumerate(last["history"], 1):
        value = row["daylight_mae_kw"]
        if row["epoch"] != epoch or not np.isfinite(value):
            raise ValueError(f"Invalid validation history for seed {seed}")
        if score-value >= native.SETTINGS["min_delta_kw"]:
            selected, score, stale = epoch, value, 0
        else:
            stale += 1
    if (last["epoch"] != len(last["history"]) or not 1 <= last["epoch"] <= native.SETTINGS["epochs"]
            or (last["epoch"] < native.SETTINGS["epochs"] and stale < native.SETTINGS["patience"])
            or last["stale"] != stale or best["epoch"] != selected
            or last["best_mae"] != score or best["best_mae"] != score
            or canonical(best["history"]) != canonical(last["history"][:selected])
            or result["stopping_epoch"] != last["epoch"] or result["best_epoch"] != selected
            or not np.isclose(result["daylight_mae_kw"], score, rtol=1e-12, atol=1e-9)):
        raise ValueError(f"Incomplete/inconsistent validation-selected quantum seed {seed}")
    model = native.make_model(reference["dimension"], seed, "default.qubit", batched=True)
    model.load_state_dict(best["model"], strict=True)
    return model


def frozen_artifacts():
    paths = [native.OUT/"settings.json", native.OUT/"preprocessing.joblib"]
    for seed in native.SEEDS:
        paths.extend(native.OUT/"quantum"/f"seed_{seed}"/name
                     for name in ("settings.json", "best.pt", "last.pt", "metrics.json"))
    paths.extend(baseline.OUT/name for name in ("settings.json", "metrics.csv", "predictions.parquet"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required returned runs or ML/DL artifacts missing:\n"+"\n".join(missing))
    saved = json.loads((baseline.OUT/"settings.json").read_text())
    if saved["protocol"] != "stage2-frozen-test-v1" or saved["split"] != baseline.SPLIT:
        raise ValueError("Incompatible saved ML/DL test protocol")
    reference = saved["neural_reference"]
    spec = json.loads((native.OUT/"settings.json").read_text())
    if (spec["architecture"] != "stage2-native-batched-cpu-v1" or spec["settings"] != native.SETTINGS
            or spec["seeds"] != native.SEEDS or spec["dimension"] != 14
            or spec["features"] != FEATURES or spec["split"] != baseline.SPLIT):
        raise ValueError("Incompatible CPU quantum architecture/settings/split")
    for key in ("features", "dimension", "split", "counts", "daylight_n", "target_scale_kw",
                "target_scale_ddof", "target_centered", "preprocessing", "frame_fingerprints",
                "preprocessing_sha256", "versions"):
        if canonical(spec[key]) != canonical(reference[key]):
            raise ValueError(f"Quantum/classical training compatibility differs: {key}")
    if (native.neural.dependencies() != spec["versions"]
            or spec["classical_reference_sha256"] != digest(native.neural.OUT/"settings.json")
            or spec["classical_reference_commit"] != reference["source_commit"]):
        raise ValueError("Incompatible software or classical reference provenance")
    expected_sources = {"scripts/"+n.replace("\\", "/").rsplit("/", 1)[-1]: h
                        for n, h in reference["source_hashes"].items()}
    expected_sources.update({str(p): digest(p) for p in
                            (Path("scripts/train_stage2_quantum_cpu.py"), Path("scripts/train_stage2_quantum_gpu.py"))})
    expected_sources = {Path(n).as_posix(): h for n, h in expected_sources.items()}
    if spec["source_hashes"] != expected_sources:
        raise ValueError("Quantum training source provenance differs")
    for name, expected in {**saved["source_hashes"], **expected_sources}.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen source/artifact changed: {name}")
        paths.append(Path(name))
    # Compare fitted objects, not serialization bytes that can differ across checkouts.
    if digest(baseline.CORRECTION/"ridge.joblib") != spec["preprocessing_sha256"]:
        raise ValueError("Training preprocessing source changed")
    fitted = joblib.load(baseline.CORRECTION/"ridge.joblib")
    preprocessing = joblib.load(native.OUT/"preprocessing.joblib")
    expected = dict(processor=fitted["processor"], scaler=fitted["scaler"],
                    features=FEATURES, continuous_columns=len(FEATURES), target_scale_kw=spec["target_scale_kw"])
    if joblib.hash(preprocessing) != joblib.hash(expected):
        raise ValueError("Returned training-fitted preprocessing/target scale differs")
    for seed in native.SEEDS:
        completed_model(seed, spec)
    paths += [Path(__file__), native.neural.OUT/"settings.json"]
    paths.extend(baseline.WEATHER/name for name in
                 ("forecasts_testing.parquet", "physics_testing.parquet", "testing_settings.json", "physics_testing_settings.json"))
    return dict(protocol="stage2-frozen-cpu-quantum-test-v1", quantum_reference=spec,
                source_hashes={p.as_posix(): digest(p) for p in paths},
                scoring="Saved ML/DL rows; target elevation>0; bias=prediction-actual; no clipping; R2 undefined for constant targets",
                disclosure=saved["disclosure"])


def aligned_inputs(saved, features, table, day, recorded_scores):
    if (not saved.index.is_unique or not saved.target_timestamp.is_unique
            or not saved.index.equals(table.index) or not saved.index.equals(features.index)
            or list(features.columns) != FEATURES
            or not pd.DatetimeIndex(saved.target_timestamp).equals(saved.index+pd.Timedelta(minutes=15))):
        raise ValueError("Test origin/target keys or feature alignment differ")
    pd.testing.assert_frame_equal(saved[KEYS], table[KEYS], check_dtype=False, check_exact=True, check_freq=False)
    np.testing.assert_array_equal(saved.persistence_kw, features.origin_power_kw)
    np.testing.assert_array_equal(saved.physics_forecast_kw, features.physics_forecast_kw)
    expected_day = solar_elevation(pd.DatetimeIndex(saved.target_timestamp)).to_numpy() > 0
    if (not np.array_equal(day, expected_day) or not recorded_scores.all_day_n.eq(len(saved)).all()
            or not recorded_scores.daylight_n.eq(int(expected_day.sum())).all()):
        raise ValueError("ML/DL eligible/daylight rows differ")


def scores_with_r2(predictions, columns, day):
    scores = baseline.score_predictions(predictions, columns, day)
    actual = predictions.actual_target_power_kw.to_numpy()
    for label, mask in (("daylight", day), ("all_day", np.ones(len(actual), dtype=bool))):
        denominator = np.sum((actual[mask]-actual[mask].mean())**2)
        for column in columns:
            numerator = np.sum((predictions[column].to_numpy()[mask]-actual[mask])**2)
            scores.loc[column, label+"_r2"] = 1-numerator/denominator if denominator > 0 else np.nan
    return scores


def evaluate(plan):
    saved = pd.read_parquet(baseline.OUT/"predictions.parquet")
    features, table, day, _ = baseline.load_testing_examples()
    aligned_inputs(saved, features, table, day, pd.read_csv(baseline.OUT/"metrics.csv"))
    preprocessing = joblib.load(native.OUT/"preprocessing.joblib")
    transformed = preprocessing["processor"].transform(features)
    continuous = preprocessing["continuous_columns"]
    x = np.ascontiguousarray(np.column_stack([
        preprocessing["scaler"].transform(transformed)[:, :continuous], transformed[:, continuous:]]), dtype=np.float64)
    spec = plan["quantum_reference"]
    if x.shape != (len(saved), spec["dimension"]) or not np.isfinite(x).all():
        raise ValueError("Invalid transformed quantum inputs")
    predictions = saved[KEYS].copy()
    predictions["daylight"] = day
    columns = []
    for seed in native.SEEDS:
        name = f"quantum_{seed}"
        correction = native.inference(completed_model(seed, spec), x)*preprocessing["target_scale_kw"]
        predictions[name+"_correction_kw"] = correction
        predictions[name] = predictions.physics_forecast_kw+correction
        columns.append(name)
    scores = scores_with_r2(predictions, columns, day)
    summary = scores.drop(columns=["daylight_n", "all_day_n"]).agg(["mean", "std"])
    if canonical(frozen_artifacts()) != canonical(plan):
        raise ValueError("Frozen artifacts changed during inference")
    # Exclusive creation also protects against another evaluation writing concurrently.
    with (OUT/"predictions.parquet").open("xb") as stream:
        predictions.to_parquet(stream)
    with (OUT/"metrics.csv").open("x", newline="") as stream:
        pd.concat([scores, summary]).to_csv(stream)
    print(scores.to_string())
    print("Across all three seeds (sample std, ddof=1; no ensemble):\n"+summary.to_string())
    print(plan["disclosure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate and not args.allow_test:
        parser.error("Evaluation requires --allow-test after reviewing preparation")
    if args.evaluate and any((OUT/name).exists() for name in ("predictions.parquet", "metrics.csv")):
        raise ValueError("Existing quantum test results; refusing repeated evaluation or overwrite")
    import torch
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    plan = frozen_artifacts()
    settings = OUT/"settings.json"
    if args.prepare:
        OUT.mkdir(parents=True, exist_ok=True)
        native.neural.freeze(settings, plan)
        print("Three completed CPU quantum seeds verified; no test rows loaded or scored. Frozen plan:", settings)
        print(plan["disclosure"])
        return
    if not settings.exists() or canonical(json.loads(settings.read_text())) != canonical(plan):
        raise ValueError("Run --prepare first; frozen artifacts must match exactly")
    evaluate(plan)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
