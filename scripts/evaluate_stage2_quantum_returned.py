"""Inference compatibility for returned CPU checkpoints; historical fingerprints stay intact."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import evaluate_stage2_quantum as original
from compare_stage2_direct import verify_hashes
from stage2_shared import current_source, historical_source_bytes, preserve_plan

native = original.native
baseline = original.baseline
OUT = original.OUT
AUDIT = Path("C:/Users/silsw/Documents/Codex/2026-10-16/sy/outputs/quantum_return_audit.md")
# Absolute bounds exceed the audit maxima while remaining negligible in physical units.
TOLERANCES = dict(target_scale_kw=1e-9, statistics_=1e-9, mean_=1e-9, scale_=1e-9,
                  var_=1e-7, transformed=1e-9, validation_physics_kw=3e-7, validation_correction_kw=1e-8)


def close_numeric(actual, expected, atol, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=0, atol=atol)):
        raise ValueError(f"Numerically incompatible {label}; absolute tolerance {atol}")
    return float(np.max(np.abs(actual-expected)))


def transform(preprocessing, features):
    if list(features.columns) != original.FEATURES:
        raise ValueError("Returned input feature order differs")
    raw = preprocessing["processor"].transform(features)
    scaled = preprocessing["scaler"].transform(raw)
    n = preprocessing["continuous_columns"]
    return np.ascontiguousarray(np.column_stack([scaled[:, :n], raw[:, n:]]), dtype=np.float64)


def preprocessing_compatibility(returned, local, spec):
    if (returned["features"] != original.FEATURES or returned["continuous_columns"] != len(original.FEATURES)
            or returned["target_scale_kw"] != spec["target_scale_kw"]):
        raise ValueError("Returned preprocessing/target scale does not match returned specification")
    for name in ("processor", "scaler"):
        a, b = returned[name], local[name]
        if type(a) is not type(b) or original.canonical(a.get_params(deep=False)) != original.canonical(b.get_params(deep=False)):
            raise ValueError(f"Returned preprocessing configuration differs: {name}")
    deltas = {}
    for (name, a), (other, b) in zip(returned["processor"].transformer_list, local["processor"].transformer_list):
        if name != other or type(a) is not type(b) or original.canonical(a.get_params()) != original.canonical(b.get_params()):
            raise ValueError("Returned transformer configuration differs")
        for attribute in ("n_features_in_", "feature_names_in_", "features_", "_fit_dtype", "indicator_", "_n_features", "_precomputed"):
            if hasattr(a, attribute) != hasattr(b, attribute) or (hasattr(a, attribute)
                    and not np.array_equal(getattr(a, attribute), getattr(b, attribute))):
                raise ValueError(f"Returned preprocessing schema differs: {name}.{attribute}")
        if hasattr(a, "statistics_"):
            deltas[name] = close_numeric(a.statistics_, b.statistics_, TOLERANCES["statistics_"], name+" medians")
    for attribute in ("n_features_in_", "n_samples_seen_"):
        if not np.array_equal(getattr(returned["scaler"], attribute), getattr(local["scaler"], attribute)):
            raise ValueError(f"Returned scaler training count/schema differs: {attribute}")
    for attribute in ("mean_", "var_", "scale_"):
        deltas[attribute] = close_numeric(getattr(returned["scaler"], attribute), getattr(local["scaler"], attribute),
                                        TOLERANCES[attribute], "scaler "+attribute)
    np.testing.assert_array_equal(returned["processor"].get_feature_names_out(), local["processor"].get_feature_names_out())
    # Check both value transformation and all missing indicators without any test data.
    n = len(original.FEATURES)
    probe = pd.DataFrame(np.vstack([np.zeros(n), np.ones(n), np.full(n, np.nan), np.eye(n)]), columns=original.FEATURES)
    local_neural = dict(local, continuous_columns=len(original.FEATURES))
    deltas["transformed"] = close_numeric(transform(returned, probe), transform(local_neural, probe),
                                           TOLERANCES["transformed"], "preprocessing probe")
    return deltas


def source_compatibility(recorded, reference):
    expected = {Path(n).resolve().relative_to(Path.cwd().resolve()).as_posix() for n in reference}
    helper = "scripts/train_stage2_quantum_gpu.py" if "scripts/train_stage2_quantum_gpu.py" in recorded else "scripts/stage2_shared.py"
    expected.update(("scripts/train_stage2_quantum_cpu.py", helper))
    if set(recorded) != expected:
        raise ValueError("Returned source identities differ")
    status = {}
    for name, sha in recorded.items():
        raw = historical_source_bytes(name)
        lf = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() == sha:
            status[name] = "exact bytes"
        elif sha in {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}:
            status[name] = "CRLF/LF equivalent; recorded byte hash remains different"
        else:
            raise ValueError(f"Returned source changed beyond line endings: {name}")
    return status


def readiness():
    saved = json.loads((baseline.OUT/"settings.json").read_text())
    reference = saved["neural_reference"]
    spec = json.loads((native.OUT/"settings.json").read_text())
    verify_hashes(saved["source_hashes"])
    verify_hashes(reference["source_hashes"])
    if (saved["protocol"] != "stage2-frozen-test-v1" or spec["architecture"] != "stage2-native-batched-cpu-v1"
            or spec["settings"] != native.SETTINGS or spec["seeds"] != native.SEEDS
            or spec["dimension"] != 2*len(original.FEATURES) or spec["features"] != original.FEATURES):
        raise ValueError("Returned architecture/settings/features differ")
    for key in ("features", "dimension", "split", "counts", "daylight_n", "target_scale_ddof", "target_centered", "preprocessing"):
        if original.canonical(spec[key]) != original.canonical(reference[key]):
            raise ValueError(f"Returned training specification differs: {key}")
    if spec["split"] != baseline.SPLIT or saved["split"] != baseline.SPLIT:
        raise ValueError("Returned/local split differs")
    scale_delta = close_numeric(spec["target_scale_kw"], reference["target_scale_kw"], TOLERANCES["target_scale_kw"], "target scale")
    runtime = native.neural.dependencies()
    if runtime != reference["versions"]:
        raise ValueError("Use the original local evaluation environment")
    if set(spec["versions"]) != set(runtime):
        raise ValueError("Returned software records incomplete")
    for name, version in spec["versions"].items():
        allowed = {runtime[name]}
        if name == "torch":
            allowed.add(runtime[name].removesuffix("+cpu"))
        if version not in allowed:
            raise ValueError(f"Unverified returned software version: {name}={version}")
    sources = source_compatibility(spec["source_hashes"], reference["source_hashes"])
    reference_path = native.neural.OUT/"settings.json"
    local_values = dict(reference, classical_reference_sha256=original.digest(reference_path),
                        classical_reference_commit=reference["source_commit"])
    keys = ("target_scale_kw", "frame_fingerprints", "preprocessing_sha256", "classical_reference_sha256",
            "classical_reference_commit", "versions", "python")
    differences = {k: dict(returned=spec.get(k), local=local_values.get(k)) for k in keys
                   if original.canonical(spec.get(k)) != original.canonical(local_values.get(k))}
    if original.digest(baseline.CORRECTION/"ridge.joblib") != reference["preprocessing_sha256"]:
        raise ValueError("Original training preprocessing changed")
    local = joblib.load(baseline.CORRECTION/"ridge.joblib")
    returned = joblib.load(native.OUT/"preprocessing.joblib")
    deltas = preprocessing_compatibility(returned, local, spec)
    # One existing QC/exact-match load, followed by the original fingerprint convention.
    features, tables, counts, data_paths = native.neural.load_examples()
    if counts != reference["counts"]:
        raise ValueError("Original development eligibility differs")
    for split in ("training", "validation"):
        frame = features[split].join(tables[split][["target_timestamp", "actual_target_power_kw", "correction_kw"]])
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
        if fingerprint != reference["frame_fingerprints"][split]:
            raise ValueError(f"Original development data fingerprint differs: {split}")
    valid = tables["validation"]
    sample = np.linspace(0, len(valid)-1, min(256, len(valid)), dtype=int)
    x = transform(returned, features["validation"].iloc[sample])
    if x.shape != (len(sample), spec["dimension"]) or not np.isfinite(x).all():
        raise ValueError("Invalid returned transformed features")
    checks = {}
    paths = [native.OUT/"settings.json", native.OUT/"preprocessing.joblib", reference_path,
             baseline.CORRECTION/"ridge.joblib", Path(__file__), Path("scripts/compare_stage2_direct.py"),
             Path("scripts/evaluate_stage2_quantum.py")]
    for seed in native.SEEDS:
        directory = native.OUT/"quantum"/f"seed_{seed}"
        model = original.completed_model(seed, spec)
        frame = pd.read_parquet(directory/"validation_predictions.parquet")
        if not frame.index.is_unique or not frame.target_timestamp.is_unique:
            raise ValueError(f"Repeated returned validation keys: {seed}")
        pd.testing.assert_frame_equal(frame[["target_timestamp", "actual_target_power_kw"]],
            valid[["target_timestamp", "actual_target_power_kw"]], check_dtype=False, check_exact=True, check_freq=False)
        physics_delta = close_numeric(frame.final_power_kw-frame.predicted_correction_kw, valid.physics_forecast_kw,
                                       TOLERANCES["validation_physics_kw"], "returned validation physics")
        prediction = native.inference(model, x)*spec["target_scale_kw"]
        correction_delta = close_numeric(prediction, frame.predicted_correction_kw.iloc[sample],
                                           TOLERANCES["validation_correction_kw"], f"seed {seed} validation replay")
        checks[str(seed)] = dict(samples=len(sample), max_correction_difference_kw=correction_delta,
                                 max_reconstructed_physics_difference_kw=physics_delta)
        paths.extend(directory/n for n in ("settings.json", "best.pt", "last.pt", "metrics.json", "validation_predictions.parquet"))
    paths.extend(current_source(n) for n in saved["source_hashes"])
    paths.extend(current_source(n) for n in reference["source_hashes"])
    paths.extend(current_source(n) for n in spec["source_hashes"])
    paths.extend(data_paths)
    paths.extend(baseline.OUT/n for n in ("settings.json", "predictions.parquet", "metrics.csv"))
    paths.extend(baseline.WEATHER/n for n in
                 ("forecasts_testing.parquet", "physics_testing.parquet", "testing_settings.json", "physics_testing_settings.json"))
    if AUDIT.is_file():
        paths.append(AUDIT)
    plan = dict(protocol="stage2-returned-cpu-quantum-test-v1", quantum_reference=spec,
        compatibility=dict(scope="Inference compatibility, not proof of identical historical training frames/reference file",
            unresolved={k: "Unmatched recorded provenance; exact friend source data/reference files not supplied; equality is not established"
                        for k in ("frame_fingerprints", "preprocessing_sha256", "classical_reference_sha256", "classical_reference_commit") if k in differences},
            differences=differences, source_equivalence=sources, tolerances=TOLERANCES,
            target_scale_difference_kw=scale_delta, preprocessing_max_differences=deltas, validation_replay=checks),
        source_hashes={p.as_posix(): original.digest(p) for p in paths}, disclosure=saved["disclosure"])
    return preserve_plan(OUT/"settings.json", plan), returned


def evaluate(plan, preprocessing):
    saved = pd.read_parquet(baseline.OUT/"predictions.parquet")
    features, table, day, _ = baseline.load_testing_examples()
    original.aligned_inputs(saved, features, table, day, pd.read_csv(baseline.OUT/"metrics.csv"))
    x = transform(preprocessing, features)
    spec = plan["quantum_reference"]
    if x.shape != (len(saved), spec["dimension"]) or not np.isfinite(x).all():
        raise ValueError("Invalid held-out transformed quantum features")
    predictions = saved[original.KEYS].copy()
    predictions["daylight"] = day
    names = []
    for seed in native.SEEDS:
        name = f"quantum_{seed}"
        delta = native.inference(original.completed_model(seed, spec), x)*spec["target_scale_kw"]
        predictions[name+"_correction_kw"] = delta
        predictions[name] = predictions.physics_forecast_kw+delta
        names.append(name)
    scores = original.scores_with_r2(predictions, names, day)
    summary = scores.drop(columns=["daylight_n", "all_day_n"]).agg(["mean", "std"])
    verify_hashes(plan["source_hashes"])
    with (OUT/"predictions.parquet").open("xb") as stream:
        predictions.to_parquet(stream)
    with (OUT/"metrics.csv").open("x", newline="") as stream:
        pd.concat([scores, summary]).to_csv(stream)
    print(scores.to_string())
    print("Mean/sample SD, no ensemble:\n"+summary.to_string())
    print(plan["disclosure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate and not args.allow_test:
        parser.error("Held-out inference requires --allow-test after --prepare")
    if args.evaluate and any((OUT/n).exists() for n in ("predictions.parquet", "metrics.csv")):
        raise ValueError("Existing quantum test results; refusing overwrite")
    import torch
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    plan, preprocessing = readiness()
    path = OUT/"settings.json"
    if args.prepare:
        OUT.mkdir(parents=True, exist_ok=True)
        native.neural.freeze(path, plan)
        print(json.dumps(plan["compatibility"], indent=2))
        print("Ready for local inference with recorded provenance differences. No held-out rows loaded or scored.")
        return
    if not path.exists() or original.canonical(json.loads(path.read_text())) != original.canonical(plan):
        raise ValueError("Run returned --prepare first; frozen artifacts must match")
    evaluate(plan, preprocessing)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
