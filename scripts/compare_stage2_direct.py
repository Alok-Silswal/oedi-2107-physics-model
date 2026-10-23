"""Direct power baselines: frozen correction candidates, five neural seeds, gated testing."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import random
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import compare_stage2_correction as correction
import evaluate_stage2 as evaluation
import train_stage2_neural as neural
from audit_stage2_readiness import digest, solar_elevation
from evaluate_stage2_quantum import aligned_inputs, scores_with_r2
from forecast_stage2_weather import SPLIT, preprocess, estimator
from stage2_shared import read_checkpoint, verify_hashes, preserve_plan

OUT = Path("outputs/2107_stage2_direct")
FEATURES = [name for name in correction.FEATURES if name != "physics_forecast_kw"]
KEYS = ["target_timestamp", "actual_target_power_kw"]
SEEDS = list(range(2107, 2112))
TRAINING_KEYS = ["batch_size", "epochs", "patience", "min_delta_kw", "optimizer", "loss",
                 "learning_rate", "huber_delta", "threads", "dtype"]


def identity(value):
    return hashlib.sha256(correction.canonical(value).encode()).hexdigest()


def direct_features(frame):
    result = frame.loc[:, FEATURES].copy()
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise ValueError("Direct origins must be unique and ordered")
    return result


def development_data():
    reference_path = neural.OUT/"settings.json"
    search_path = correction.OUT/"hpo/settings.json"
    reference = json.loads(reference_path.read_text())
    search = json.loads(search_path.read_text())["plan"]
    verify_hashes(reference["source_hashes"])
    versions = neural.dependencies() | {n: metadata.version(n) for n in ("xgboost", "threadpoolctl")}
    if any(versions[n] != v for n, v in (reference["versions"] | search["versions"]).items()):
        raise ValueError("Use the completed correction environment versions")
    candidates = correction.search_candidates()
    if correction.canonical(candidates) != correction.canonical(search["candidates"]):
        raise ValueError("Candidate configurations differ from completed correction HPO")
    if (reference["split"] != SPLIT or search["split"] != SPLIT
            or reference["seeds"] != SEEDS or reference["settings"] != neural.SETTINGS):
        raise ValueError("Completed correction split/seeds/settings differ")
    features, tables, counts, paths = correction.load_examples()
    day = solar_elevation(pd.DatetimeIndex(tables["validation"].target_timestamp)).to_numpy() > 0
    if counts != reference["counts"] or counts != search["counts"] or int(day.sum()) != reference["daylight_n"]:
        raise ValueError(f"Correction eligibility differs: {counts}, daylight={day.sum()}")
    fingerprints = {}
    for split in ("training", "validation"):
        frame = features[split].join(tables[split][KEYS+["correction_kw"]])
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
        if fingerprint != search["frame_fingerprints"][split] or fingerprint != reference["frame_fingerprints"][split]:
            raise ValueError(f"Correction rows/labels/features changed: {split}")
        features[split] = direct_features(features[split])
        fingerprints[split] = hashlib.sha256(pd.util.hash_pandas_object(
            features[split].join(tables[split][KEYS]), index=True).to_numpy().tobytes()).hexdigest()
    if hashlib.sha256(day.tobytes()).hexdigest() != search["daylight_fingerprint"]:
        raise ValueError("Validation daylight rows changed")
    scale = neural.correction_scale(tables["training"].actual_target_power_kw.to_numpy())
    paths += [reference_path, search_path, Path(__file__), Path("scripts/train_stage2_neural.py"),
              Path("scripts/evaluate_stage2.py"), Path("scripts/evaluate_stage2_quantum.py"),
              Path("scripts/stage2_shared.py"), Path("scripts/compare_stage2_correction.py"),
              Path("scripts/forecast_stage2_weather.py"), Path("scripts/audit_stage2_readiness.py"),
              Path("scripts/calculate_stage2_physics.py"), Path("scripts/benchmark_short_term_forecasting.py")]
    commit = (json.loads((OUT/"settings.json").read_text())["source_commit"] if (OUT/"settings.json").exists()
              else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    plan = dict(protocol="stage2-direct-power-v1", architecture="classical-tanh-4-4-1-direct-v1",
                split=SPLIT, horizon_minutes=15, features=FEATURES, seeds=SEEDS, candidates=candidates,
                counts=counts, daylight_n=int(day.sum()), frame_fingerprints=fingerprints,
                settings={n: reference["settings"][n] for n in TRAINING_KEYS}, seed=2107, cpu_threads=4,
                target="actual_target_power_kw; direct prediction, no physics addition or clipping",
                target_scale_kw=scale, target_scale_ddof=0, target_centered=False,
                preprocessing="New training-only median/all-column indicators; Ridge scales all columns; neural scales continuous columns only",
                selection="Full validation daylight MAE; candidate-order ties; checkpoint improvement >=0.001 kW; no validation refit or seed selection",
                versions=versions, source_commit=commit, source_hashes={p.as_posix(): digest(p) for p in paths},
                disclosure="Direct baselines specified after correction results; parts of the test period appeared in earlier exploratory work, not entirely unseen")
    return features, tables, day, preserve_plan(OUT/"settings.json", plan)


def transformed(preprocessing, features, kind):
    if list(features.columns) != FEATURES or preprocessing["features"] != FEATURES:
        raise ValueError("Direct inputs must exclude physics and its indicators")
    raw = preprocessing["processor"].transform(features)
    scaled = preprocessing["scaler"].transform(raw)
    if kind == "neural":
        result = np.column_stack([scaled[:, :len(FEATURES)], raw[:, len(FEATURES):]])
    else:
        result = scaled if kind == "ridge" else raw
    if result.shape[1] != preprocessing["dimension"] or not np.isfinite(result).all():
        raise ValueError("Invalid direct transformed inputs")
    return np.ascontiguousarray(result, dtype=np.float64)


def preprocessing_for(features, plan, *, fit=False):
    path = OUT/"preprocessing.joblib"
    if path.exists():
        result = joblib.load(path)
        if (result["plan_sha256"] != identity(plan) or result["features"] != FEATURES
                or result["target_scale_kw"] != plan["target_scale_kw"]):
            raise ValueError("Incompatible direct preprocessing/scaling")
    elif fit:
        train, _, scaler, processor = preprocess(features["training"], features["validation"], True)
        result = dict(processor=processor, scaler=scaler, dimension=train.shape[1], features=FEATURES,
                      target_scale_kw=plan["target_scale_kw"], plan_sha256=identity(plan))
        temporary = path.with_suffix(".tmp")
        joblib.dump(result, temporary)
        temporary.replace(path)
    else:
        raise FileNotFoundError("Direct preprocessing missing; train direct models first")
    return result


def score(actual, prediction, day):
    frame = pd.DataFrame(dict(actual_target_power_kw=actual, prediction=prediction))
    return scores_with_r2(frame, ["prediction"], day).loc["prediction"].to_dict()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def ml_cache(plan):
    directory = OUT/"ml"
    trial_path = directory/"trials.csv"
    rows = pd.read_csv(trial_path).to_dict("records") if trial_path.exists() else []
    by_id = {c["id"]: c for c in plan["candidates"]}
    bundles = {}
    for family in correction.MODELS:
        path = directory/f"best_{family}.joblib"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        row = bundle["trial"]
        candidate = by_id.get(row["id"])
        if (candidate is None or candidate["family"] != family or bundle["plan_sha256"] != identity(plan)
                or bundle["preprocessing_sha256"] != digest(OUT/"preprocessing.joblib")
                or correction.canonical(bundle["model"].get_params()) != correction.canonical(candidate["effective_settings"])):
            raise ValueError(f"Incompatible direct ML bundle: {family}")
        bundles[family] = bundle
        if not any(r["id"] == row["id"] for r in rows):
            rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate direct ML trials")
    for row in rows:
        candidate = by_id.get(row["id"])
        if (candidate is None or row["family"] != candidate["family"]
                or correction.canonical(json.loads(row["config"])) != correction.canonical(candidate["config"])
                or row["status"] not in ("complete", "failed") or row["training_n"] != plan["counts"]["training"]["eligible"]):
            raise ValueError("Incompatible direct ML trial cache")
        if row["status"] == "complete" and (row["all_day_n"] != plan["counts"]["validation"]["eligible"]
                or row["daylight_n"] != plan["daylight_n"] or not np.isfinite(row["daylight_mae_kw"])):
            raise ValueError("Direct ML scoring rows differ")
    for family in correction.MODELS:
        successful = [r for r in rows if r["family"] == family and r["status"] == "complete"]
        if successful:
            winner = min(successful, key=lambda r: (r["daylight_mae_kw"], list(by_id).index(r["id"])))
            if family not in bundles or bundles[family]["trial"]["id"] != winner["id"]:
                raise ValueError(f"Missing validation-selected direct {family} bundle")
    return rows, bundles


def completed_ml(plan):
    rows, bundles = ml_cache(plan)
    if len(rows) != len(plan["candidates"]) or any(r["status"] != "complete" for r in rows):
        raise ValueError("All 43 direct candidates must finish successfully before testing; inspect ml/trials.csv")
    selection = json.loads((OUT/"ml/selection.json").read_text())
    expected = {family: bundle["trial"]["id"] for family, bundle in bundles.items()}
    winner = min(plan["candidates"], key=lambda c: next(r["daylight_mae_kw"] for r in rows if r["id"] == c["id"]))
    if selection["plan_sha256"] != identity(plan) or selection["best_per_family"] != expected or selection["preferred_trial"] != winner["id"]:
        raise ValueError("Incompatible direct ML validation selection")
    verify_hashes(selection["source_hashes"])
    return bundles


def train_ml(features, tables, day, plan, resume):
    directory = OUT/"ml"
    directory.mkdir(exist_ok=True)
    if not resume and any(directory.iterdir()):
        raise ValueError("Existing direct ML progress; use --resume")
    if (directory/"selection.json").exists():
        completed_ml(plan)
        print("All direct ML trials already complete; skipped")
        return
    preprocessing = preprocessing_for(features, plan, fit=not resume)
    rows, bundles = ml_cache(plan)
    done = {r["id"] for r in rows}
    actual = tables["validation"].actual_target_power_kw.to_numpy()
    x = {n: transformed(preprocessing, features[n], "tree") for n in features}
    ridge = {n: transformed(preprocessing, features[n], "ridge") for n in features}
    for candidate in plan["candidates"]:
        if candidate["id"] in done:
            continue
        family = candidate["family"]
        row = dict(id=candidate["id"], family=family, config=correction.canonical(candidate["config"]),
                   training_n=len(features["training"]), seed=2107)
        start = time.monotonic()
        print(candidate["id"], "fitting", row["training_n"], "rows", flush=True)
        try:
            model = estimator(family, correction.MODELS[family]).set_params(**candidate["config"])
            inputs = ridge if family == "ridge" else x
            model.fit(inputs["training"], tables["training"].actual_target_power_kw.to_numpy())
            row["fit_seconds"] = time.monotonic()-start
            prediction = model.predict(inputs["validation"])
            row.update(score(actual, prediction, day), status="complete", error="")
            previous = bundles.get(family)
            order = [c["id"] for c in plan["candidates"]]
            if previous is None or (row["daylight_mae_kw"], order.index(row["id"])) < (
                    previous["trial"]["daylight_mae_kw"], order.index(previous["trial"]["id"])):
                bundle = dict(model=model, trial=row.copy(), plan_sha256=identity(plan),
                              preprocessing_sha256=digest(OUT/"preprocessing.joblib"))
                path = directory/f"best_{family}.joblib"
                temporary = path.with_suffix(".tmp")
                joblib.dump(bundle, temporary)
                temporary.replace(path)
                bundles[family] = bundle
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            row.setdefault("fit_seconds", time.monotonic()-start)
        row["wall_seconds"] = time.monotonic()-start
        rows.append(row)
        correction.save_trials(directory/"trials.csv", rows)
        print(candidate["id"], row["status"], row["fit_seconds"], row.get("error", ""), flush=True)
    failures = [r["id"] for r in rows if r["status"] != "complete"]
    if failures:
        raise ValueError(f"Failed direct trials retained without replacement: {failures}")
    verify_hashes(plan["source_hashes"])
    prediction = tables["validation"][KEYS].copy()
    for family, bundle in bundles.items():
        prediction[family] = bundle["model"].predict(ridge["validation"] if family == "ridge" else x["validation"])
    prediction.to_parquet(directory/"validation_predictions.parquet")
    winner = min(plan["candidates"], key=lambda c: next(r["daylight_mae_kw"] for r in rows if r["id"] == c["id"]))
    paths = [directory/"trials.csv", directory/"validation_predictions.parquet",
             *[directory/f"best_{family}.joblib" for family in correction.MODELS]]
    atomic_json(directory/"selection.json", dict(plan_sha256=identity(plan),
        best_per_family={family: bundle["trial"]["id"] for family, bundle in bundles.items()},
        preferred_trial=winner["id"], source_hashes={p.as_posix(): digest(p) for p in paths},
        label="Validation model selection only; all families retained; no validation refit"))
    print((directory/"selection.json").read_text(), flush=True)


def make_neural(dimension, seed):
    import torch
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        return neural.make_model("classical", dimension, seed)
    finally:
        torch.set_default_dtype(previous)


def run_specification(plan, preprocessing, seed):
    return dict(plan, model="direct_classical", seed=seed, dimension=preprocessing["dimension"],
                preprocessing_sha256=digest(OUT/"preprocessing.joblib"))


def direct_power(normalized, scale):
    return np.asarray(normalized)*scale


def selected_history(history, settings):
    selected, best, stale = 0, float("inf"), 0
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or not np.isfinite(row["daylight_mae_kw"]):
            raise ValueError("Invalid direct neural history")
        if best-row["daylight_mae_kw"] >= settings["min_delta_kw"]:
            selected, best, stale = epoch, row["daylight_mae_kw"], 0
        else:
            stale += 1
    return selected, best, stale


def completed_neural(seed, plan, preprocessing):
    directory = OUT/"neural"/f"seed_{seed}"
    spec = run_specification(plan, preprocessing, seed)
    if correction.canonical(json.loads((directory/"settings.json").read_text())) != correction.canonical(spec):
        raise ValueError(f"Incompatible direct seed {seed} settings")
    parameters = 4*preprocessing["dimension"]+29
    best = read_checkpoint(directory/"best.pt", spec, parameters)
    last = read_checkpoint(directory/"last.pt", spec, parameters)
    result = json.loads((directory/"metrics.json").read_text())
    epoch, value, stale = selected_history(last["history"], plan["settings"])
    if (last["epoch"] != len(last["history"]) or not 1 <= last["epoch"] <= plan["settings"]["epochs"]
            or best["epoch"] != epoch or last["stale"] != stale
            or not (last["epoch"] == plan["settings"]["epochs"] or stale >= plan["settings"]["patience"])
            or best["best_mae"] != value or last["best_mae"] != value
            or correction.canonical(best["history"]) != correction.canonical(last["history"][:epoch])
            or result["best_epoch"] != epoch or result["stopping_epoch"] != last["epoch"]
            or not np.isclose(result["daylight_mae_kw"], value, rtol=1e-12, atol=1e-9)
            or digest(directory/"validation_predictions.parquet") != result["predictions_sha256"]):
        raise ValueError(f"Incomplete/incompatible direct seed {seed}")
    model = make_neural(preprocessing["dimension"], seed)
    model.load_state_dict(best["model"], strict=True)
    return model


def train_neural(seed, resume, features, tables, day, plan):
    import torch
    directory = OUT/"neural"/f"seed_{seed}"
    last, best = directory/"last.pt", directory/"best.pt"
    if resume != last.exists():
        raise ValueError("Use --resume for existing direct checkpoints; omit it for a new seed")
    preprocessing = preprocessing_for(features, plan, fit=not resume)
    spec = run_specification(plan, preprocessing, seed)
    directory.mkdir(parents=True, exist_ok=True)
    neural.freeze(directory/"settings.json", spec)
    if (directory/"metrics.json").exists():
        completed_neural(seed, plan, preprocessing)
        print(seed, "already complete; skipped")
        return
    settings = plan["settings"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_neural(preprocessing["dimension"], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    epoch, best_mae, stale, history = 0, float("inf"), 0, []
    if resume:
        state = read_checkpoint(last, spec, 4*preprocessing["dimension"]+29)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        epoch, best_mae, stale, history = state["epoch"], state["best_mae"], state["stale"], state["history"]
        generator.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    x = {name: transformed(preprocessing, frame, "neural") for name, frame in features.items()}
    xt = torch.from_numpy(x["training"])
    scale = preprocessing["target_scale_kw"]
    yt = torch.from_numpy(tables["training"].actual_target_power_kw.to_numpy()/scale)
    actual = tables["validation"].actual_target_power_kw.to_numpy()
    loss_fn = torch.nn.HuberLoss(delta=settings["huber_delta"])
    while epoch < settings["epochs"] and stale < settings["patience"]:
        start = time.monotonic()
        model.train()
        total_loss = 0.
        for batch in torch.randperm(len(xt), generator=generator).split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(batch)
        prediction = direct_power(neural.inference(model, x["validation"]), scale)
        scores = score(actual, prediction, day)
        improved = best_mae-scores["daylight_mae_kw"] >= settings["min_delta_kw"]
        best_mae = scores["daylight_mae_kw"] if improved else best_mae
        stale = 0 if improved else stale+1
        epoch += 1
        history.append(dict(epoch=epoch, normalized_huber=total_loss/len(xt), **scores, seconds=time.monotonic()-start))
        state = dict(specification=spec, model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     best_mae=best_mae, stale=stale, history=history, shuffle_rng=generator.get_state(),
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        if improved:
            neural.checkpoint(best, state)
        neural.checkpoint(last, state)
        pd.DataFrame(history).to_csv(directory/"history.csv", index=False)
        print(seed, history[-1], flush=True)
    best_state = read_checkpoint(best, spec, 4*preprocessing["dimension"]+29)
    model.load_state_dict(best_state["model"], strict=True)
    prediction = direct_power(neural.inference(model, x["validation"]), scale)
    verify_hashes(plan["source_hashes"])
    saved = tables["validation"][KEYS].copy()
    saved["direct_power_kw"] = prediction
    saved.to_parquet(directory/"validation_predictions.parquet")
    atomic_json(directory/"metrics.json", dict(**score(actual, prediction, day), seed=seed,
        stopping_epoch=epoch, best_epoch=best_state["epoch"], training_validation_seconds=sum(h["seconds"] for h in history),
        predictions_sha256=digest(directory/"validation_predictions.parquet"), label="Validation selection; no test or seed selection"))


def test_plan(plan):
    verify_hashes(plan["source_hashes"])
    preprocessing = preprocessing_for(None, plan)
    completed_ml(plan)
    for seed in SEEDS:
        completed_neural(seed, plan, preprocessing)
    reference = json.loads((evaluation.OUT/"settings.json").read_text())
    if reference["protocol"] != "stage2-frozen-test-v1" or reference["split"] != SPLIT:
        raise ValueError("Saved ML/DL test protocol differs")
    verify_hashes(reference["source_hashes"])
    paths = [OUT/"settings.json", OUT/"preprocessing.joblib", OUT/"ml/selection.json", OUT/"ml/trials.csv",
             *[OUT/"ml"/f"best_{family}.joblib" for family in correction.MODELS]]
    paths += [OUT/"neural"/f"seed_{seed}"/name for seed in SEEDS
              for name in ("settings.json", "best.pt", "last.pt", "metrics.json", "validation_predictions.parquet")]
    paths += [evaluation.OUT/name for name in ("settings.json", "metrics.csv", "predictions.parquet")]
    paths += [evaluation.WEATHER/name for name in
              ("forecasts_testing.parquet", "physics_testing.parquet", "testing_settings.json", "physics_testing_settings.json")]
    result = dict(protocol="stage2-direct-frozen-test-v1", training_plan=plan,
                source_hashes={p.as_posix(): digest(p) for p in paths}, disclosure=plan["disclosure"],
                scoring="Exact saved ML/DL keys/labels/daylight rows; bias=prediction-actual; no postprocessing; R2 undefined for constant targets")
    return preserve_plan(OUT/"test/settings.json", result)


def evaluate(frozen):
    plan = frozen["training_plan"]
    saved = pd.read_parquet(evaluation.OUT/"predictions.parquet")
    features, table, day, _ = evaluation.load_testing_examples()
    aligned_inputs(saved, features, table, day, pd.read_csv(evaluation.OUT/"metrics.csv"))
    features = direct_features(features)
    preprocessing = preprocessing_for(None, plan)
    bundles = completed_ml(plan)
    predictions = saved[KEYS+["physics_forecast_kw", "persistence_kw"]].copy()
    predictions["daylight"] = day
    columns = ["physics_forecast_kw", "persistence_kw"]
    for family, bundle in bundles.items():
        name = "direct_"+family
        predictions[name] = bundle["model"].predict(transformed(preprocessing, features, family))
        columns.append(name)
    x = transformed(preprocessing, features, "neural")
    names = []
    for seed in SEEDS:
        name = f"direct_classical_{seed}"
        predictions[name] = direct_power(neural.inference(completed_neural(seed, plan, preprocessing), x),
                                         preprocessing["target_scale_kw"])
        names.append(name)
    scores = scores_with_r2(predictions, columns+names, day)
    summary = scores.loc[names].drop(columns=["all_day_n", "daylight_n"]).agg(["mean", "std"])
    if correction.canonical(test_plan(plan)) != correction.canonical(frozen):
        raise ValueError("Frozen direct artifacts changed during inference")
    with (OUT/"test/predictions.parquet").open("xb") as stream:
        predictions.to_parquet(stream)
    with (OUT/"test/metrics.csv").open("x", newline="") as stream:
        pd.concat([scores, summary.rename(index={"mean":"direct_classical_mean", "std":"direct_classical_sample_std"})]).to_csv(stream)
    print(scores.to_string())
    print("Five seeds; mean/sample std, no ensemble:\n"+summary.to_string())
    print(frozen["disclosure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "train-ml", "train-neural", "prepare-test", "evaluate"):
        mode.add_argument("--"+name, action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate and not args.allow_test:
        parser.error("Held-out evaluation requires --allow-test after --prepare-test")
    if args.train_neural and args.seed is None:
        parser.error("--train-neural requires --seed")
    if args.resume and not (args.train_ml or args.train_neural):
        parser.error("--resume is only for training")
    import torch
    torch.set_num_threads(4)
    if args.prepare_test or args.evaluate:
        if args.evaluate and any((OUT/"test"/n).exists() for n in ("predictions.parquet", "metrics.csv")):
            raise ValueError("Existing direct test results; refusing overwrite or repeated evaluation")
        plan = json.loads((OUT/"settings.json").read_text())
        if plan["versions"] != neural.dependencies() | {n: metadata.version(n) for n in ("xgboost", "threadpoolctl")}:
            raise ValueError("Direct evaluation software versions differ")
        frozen = test_plan(plan)
        path = OUT/"test/settings.json"
        if args.prepare_test:
            path.parent.mkdir(exist_ok=True)
            neural.freeze(path, frozen)
            print("Four ML winners and all five neural seeds frozen; no test rows loaded/scored.")
            return
        if not path.exists() or correction.canonical(json.loads(path.read_text())) != correction.canonical(frozen):
            raise ValueError("Run --prepare-test first; frozen artifacts must match")
        evaluate(frozen)
        return
    features, tables, day, plan = development_data()
    OUT.mkdir(exist_ok=True)
    neural.freeze(OUT/"settings.json", plan)
    if args.prepare:
        print(json.dumps(dict(features=FEATURES, counts=plan["counts"], daylight_n=plan["daylight_n"],
            candidates={n:sum(c["family"] == n for c in plan["candidates"]) for n in correction.MODELS},
            new_ml_fits=len(plan["candidates"]), neural_seeds=SEEDS, settings=plan["settings"],
            target_power_sd_kw=plan["target_scale_kw"], transformed_dimension="Derived from new fitted processor during training",
            output=str(OUT)), indent=2))
        print("Preparation only: no preprocessing/model fitting, predictions or test rows.")
    elif args.train_ml:
        train_ml(features, tables, day, plan, args.resume)
    else:
        train_neural(args.seed, args.resume, features, tables, day, plan)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
