"""Matched classical/quantum correction simulation; validation selection only."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ["OMP_NUM_THREADS"] = "4"
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_stage2_readiness import digest, solar_elevation
from compare_stage2_correction import FEATURES, OUT as CORRECTION, canonical, load_examples, metrics
from forecast_stage2_weather import SPLIT

OUT = Path("outputs/2107_stage2_neural")
SEEDS = list(range(2107, 2112))
SETTINGS = dict(batch_size=128, epochs=100, patience=10, min_delta_kw=0.001,
                optimizer="Adam", loss="Huber", learning_rate=0.001, huber_delta=1.0, threads=4, dtype="float64",
                backend="lightning.qubit", interface="torch", differentiation="adjoint",
                shots=None, qubits=4, layers=2, ranges=[1, 2], entangler="CNOT")
REFERENCE_MAE = 32.87024469806436


def dependencies():
    result = {}
    for name in ("torch", "pennylane", "pennylane-lightning", "numpy", "pandas",
                 "scikit-learn", "scipy", "joblib", "pvlib"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def make_model(kind, dimension, seed):
    import torch
    from torch import nn

    class Correction(nn.Module):
        def __init__(self):
            super().__init__()
            # Construct shared components before the different middle blocks.
            self.projection = nn.Linear(dimension, 4)
            self.head = nn.Linear(4, 1)
            if kind == "classical":
                self.middle = nn.Linear(4, 4)
            elif kind == "quantum":
                import pennylane as qml
                self.weights = nn.Parameter(torch.empty(2, 4, 3).uniform_(-0.1, 0.1))
                device = qml.device(SETTINGS["backend"], wires=4, shots=None)

                @qml.qnode(device, interface="torch", diff_method="adjoint")
                def circuit(angles, weights):
                    for wire in range(4):
                        qml.RY(angles[wire], wires=wire)
                    qml.StronglyEntanglingLayers(weights, wires=range(4),
                                                ranges=[1, 2], imprimitive=qml.CNOT)
                    return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(4))
                self.circuit = circuit
            else:
                raise ValueError("Unknown model")

        def forward(self, x):
            hidden = torch.tanh(self.projection(x))
            if kind == "classical":
                hidden = torch.tanh(self.middle(hidden))
            else:
                # Individual tapes avoid backend-dependent broadcast gradients.
                hidden = torch.stack([torch.stack(self.circuit(row * torch.pi, self.weights))
                                      for row in hidden])
            # PennyLane expectations can use a different dtype from the model.
            hidden = hidden.to(dtype=self.head.weight.dtype)
            return self.head(hidden).squeeze(-1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Correction().double().cpu()


def gradient_check(model, dimension):
    import torch
    x = torch.randn(3, dimension, generator=torch.Generator().manual_seed(17), dtype=torch.float64)
    prediction = model(x)
    if prediction.shape != (3,) or not torch.isfinite(prediction).all():
        raise ValueError("Invalid model output")
    prediction.square().mean().backward()
    names = ["projection.weight"] + (["weights"] if hasattr(model, "weights") else ["middle.weight"])
    for name, parameter in model.named_parameters():
        if name in names and (parameter.grad is None or not torch.isfinite(parameter.grad).all()
                              or not parameter.grad.abs().sum() > 0):
            raise ValueError(f"Missing/nonfinite/zero gradient: {name}")
    model.zero_grad(set_to_none=True)
    return sum(p.numel() for p in model.parameters())


def correction_scale(labels):
    scale = float(np.std(labels, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training correction standard deviation must be positive")
    return scale


def final_power(physics, normalized_correction, scale):
    return np.asarray(physics) + np.asarray(normalized_correction) * scale


def prepare_data():
    features, tables, counts, _ = load_examples()
    day = solar_elevation(pd.DatetimeIndex(tables["validation"].target_timestamp)).to_numpy() > 0
    actual_counts = (len(features["training"]), len(features["validation"]), int(day.sum()))
    if actual_counts != (221057, 70555, 43949):
        raise ValueError(f"Eligibility differs: {actual_counts}; expected (221057, 70555, 43949)")
    plan = json.loads((CORRECTION/"hpo/settings.json").read_text())["plan"]
    fingerprints = {}
    for split in ("training", "validation"):
        frame = features[split].join(tables[split][["target_timestamp", "actual_target_power_kw", "correction_kw"]])
        fingerprints[split] = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
    if fingerprints != plan["frame_fingerprints"]:
        raise ValueError("Correction examples differ from completed HPO")
    source = CORRECTION/"ridge.joblib"
    expected = {str(Path(p).resolve()): h for p, h in plan["baseline_hashes"].items()}
    if digest(source) != expected[str(source.resolve())]:
        raise ValueError("Saved training-fitted preprocessing changed")
    bundle = joblib.load(source)
    if bundle["features"] != FEATURES:
        raise ValueError("Preprocessing features differ")
    processor, scaler = bundle["processor"], bundle["scaler"]
    x = {}
    for split, frame in features.items():
        transformed = processor.transform(frame)
        continuous = scaler.transform(transformed)[:, :len(FEATURES)]
        x[split] = np.ascontiguousarray(np.column_stack([continuous, transformed[:, len(FEATURES):]]), dtype=np.float64)
    if any(not np.isfinite(v).all() for v in x.values()):
        raise ValueError("Nonfinite transformed input")
    scale = correction_scale(tables["training"].correction_kw.to_numpy())
    sources = [Path(__file__), Path("scripts/compare_stage2_correction.py"),
               Path("scripts/forecast_stage2_weather.py"), Path("scripts/audit_stage2_readiness.py"),
               Path("scripts/calculate_stage2_physics.py"), Path("scripts/benchmark_short_term_forecasting.py")]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    spec = dict(architecture="stage2-matched-4wire-v1", settings=SETTINGS, split=SPLIT,
                seeds=SEEDS, features=FEATURES, dimension=x["training"].shape[1], counts=counts,
                daylight_n=int(day.sum()), target_scale_kw=scale, target_scale_ddof=0, target_centered=False,
                preprocessing="Saved training-fitted median/indicators; continuous-only standardization; binary indicators",
                frame_fingerprints=fingerprints, preprocessing_sha256=digest(source),
                source_hashes={p.as_posix(): digest(p) for p in sources}, source_commit=commit,
                versions=dependencies(), python=sys.version.split()[0], reference_forest_05_mae_kw=REFERENCE_MAE)
    return x, tables, day, spec, dict(processor=processor, scaler=scaler, target_scale_kw=scale,
                                    features=FEATURES, continuous_columns=len(FEATURES))


def freeze(path, spec):
    if path.exists():
        if canonical(json.loads(path.read_text())) != canonical(spec):
            raise ValueError(f"Incompatible settings: {path}")
    else:
        path.write_text(json.dumps(spec, indent=2))


def checkpoint(path, state):
    import torch
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def read_checkpoint(path, spec):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    if canonical(state["specification"]) != canonical(spec):
        raise ValueError("Incompatible resume checkpoint")
    return state


def inference(model, x):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+128])).numpy()
                               for i in range(0, len(x), 128)])


def pilot(x, labels, spec):
    import torch
    indices = np.random.default_rng(2107).choice(len(x["training"]), 1024, replace=False)
    sample = torch.from_numpy(x["training"][indices])
    # Pilot labels exercise the real loss/scale, but accuracy is never scored.
    results = {}
    for kind in ("classical", "quantum"):
        model = make_model(kind, spec["dimension"], 2107)
        gradient_check(model, spec["dimension"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.HuberLoss(delta=1.0)
        target = torch.from_numpy(labels[indices])
        def update(a, b):
            optimizer.zero_grad(set_to_none=True)
            loss(model(a), b).backward()
            optimizer.step()
        update(sample[:128], target[:128])
        start = time.perf_counter()
        for i in range(0, 1024, 128):
            update(sample[i:i+128], target[i:i+128])
        fit_seconds = time.perf_counter()-start
        inference(model, sample[:128].numpy())
        start = time.perf_counter()
        inference(model, sample.numpy())
        infer_seconds = time.perf_counter()-start
        train_epoch = fit_seconds/8 * math.ceil(len(x["training"])/128)
        validation = infer_seconds/8 * math.ceil(len(x["validation"])/128)
        results[kind] = dict(eight_update_batches_seconds=fit_seconds, eight_inference_batches_seconds=infer_seconds,
                             approximate_training_epoch_seconds=train_epoch, approximate_validation_seconds=validation,
                             rough_100_epoch_upper_estimate_five_seeds_hours=5*100*(train_epoch+validation)/3600)
        print(kind, json.dumps(results[kind]), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT/"pilot.json").write_text(json.dumps(dict(specification=spec, sample_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        results=results, limitation="Rough epoch-ceiling extrapolation, not a guaranteed runtime bound; excludes checkpoint/I/O overhead; no early-stopping savings assumed"), indent=2))


def train(kind, seed, resume, x, tables, day, spec, preprocessing):
    import torch
    run_spec = dict(spec, model=kind, seed=seed)
    directory = OUT/kind/f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    freeze(directory/"settings.json", run_spec)
    if not (OUT/"preprocessing.joblib").exists():
        joblib.dump(preprocessing, OUT/"preprocessing.joblib")
    last, best = directory/"last.pt", directory/"best.pt"
    if not resume and last.exists():
        raise ValueError("Existing run; use --resume")
    if resume and not last.exists():
        raise ValueError("No checkpoint to resume")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_model(kind, spec["dimension"], seed)
    gradient_check(model, spec["dimension"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    history, epoch, best_mae, stale = [], 0, float("inf"), 0
    if resume:
        state = read_checkpoint(last, run_spec)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, best_mae, stale, history = state["epoch"], state["best_mae"], state["stale"], state["history"]
        generator.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    xt = torch.from_numpy(x["training"])
    yt = torch.from_numpy(tables["training"].correction_kw.to_numpy()/spec["target_scale_kw"])
    actual = tables["validation"].actual_target_power_kw.to_numpy()
    physics = tables["validation"].physics_forecast_kw.to_numpy()
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    while epoch < 100 and stale < 10:
        begin = time.perf_counter()
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        total_loss = 0.0
        for batch in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(batch)
        prediction = final_power(physics, inference(model, x["validation"]), spec["target_scale_kw"])
        score = metrics(actual, prediction, day)
        if not all(np.isfinite(v) for v in score.values()):
            raise ValueError("Nonfinite validation metric")
        improved = best_mae-score["daylight_mae_kw"] >= 0.001
        best_mae = score["daylight_mae_kw"] if improved else best_mae
        stale = 0 if improved else stale+1
        epoch += 1
        history.append(dict(epoch=epoch, normalized_huber=total_loss/len(xt), **score,
                            seconds=time.perf_counter()-begin))
        state = dict(specification=run_spec, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     epoch=epoch, best_mae=best_mae, stale=stale, history=history,
                     shuffle_rng=generator.get_state(), torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate())
        if improved:
            checkpoint(best, state)
        checkpoint(last, state)
        pd.DataFrame(history).to_csv(directory/"history.csv", index=False)
        print(kind, seed, history[-1], flush=True)
    best_state = read_checkpoint(best, run_spec)
    model.load_state_dict(best_state["model"])
    correction = inference(model, x["validation"])*spec["target_scale_kw"]
    prediction = physics+correction
    saved = tables["validation"][["target_timestamp", "actual_target_power_kw"]].copy()
    saved["predicted_correction_kw"] = correction
    saved["final_power_kw"] = prediction
    saved.to_parquet(directory/"validation_predictions.parquet")
    result = dict(**metrics(actual, prediction, day), seed=seed, model=kind, stopping_epoch=epoch,
                  best_epoch=best_state["epoch"], training_validation_seconds=sum(h["seconds"] for h in history),
                  label="Validation selection results; not final test performance")
    (directory/"metrics.json").write_text(json.dumps(result, indent=2))
    freeze(OUT/"settings.json", spec)


def summary():
    rows = []
    for path in OUT.glob("*/seed_*/metrics.json"):
        spec = json.loads((OUT/"settings.json").read_text())
        row = json.loads(path.read_text())
        saved = json.loads((path.parent/"settings.json").read_text())
        if canonical(saved) != canonical(dict(spec, model=row["model"], seed=row["seed"])):
            raise ValueError(f"Incompatible summary result: {path}")
        rows.append(row)
    if not rows:
        print("No completed seed results")
        return
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    names = ["daylight_mae_kw", "daylight_rmse_kw", "daylight_bias_kw", "all_day_mae_kw",
             "training_validation_seconds", "stopping_epoch"]
    print(table.groupby("model")[names].agg(["count", "mean", "std"]).to_string())
    print("Per-seed metrics; sample standard deviation (ddof=1); no seed selection or ensemble.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "pilot", "train", "summary"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--model", choices=["classical", "quantum"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accept-pilot", action="store_true", help="Confirm review of runtime estimates before training")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    x, tables, day, spec, preprocessing = prepare_data()
    d = spec["dimension"]
    print(json.dumps(dict(counts=spec["counts"], daylight_n=int(day.sum()), dimension=d,
                         target_scale_kw=spec["target_scale_kw"], parameter_counts=dict(classical=4*d+29, quantum=4*d+33),
                         dependencies=spec["versions"], settings=SETTINGS), indent=2), flush=True)
    if args.prepare:
        for kind in ("classical", "quantum"):
            required = ["torch"] + (["pennylane", "pennylane-lightning"] if kind=="quantum" else [])
            if any(spec["versions"][p] is None for p in required):
                print(kind, "backend/shape/gradient checks unavailable: missing dependencies")
            else:
                import torch
                torch.set_num_threads(4)
                print(kind, "verified parameter count:", gradient_check(make_model(kind,d,2107), d))
        return
    required = ["torch"] + (["pennylane", "pennylane-lightning"] if args.pilot or args.model=="quantum" else [])
    if any(spec["versions"][p] is None for p in required):
        parser.error("Missing required dependencies; run --prepare and install the constrained optional extra")
    import torch
    torch.set_num_threads(4)
    if args.pilot:
        pilot(x, tables["training"].correction_kw.to_numpy()/spec["target_scale_kw"], spec)
        return
    if args.model is None or args.seed is None:
        parser.error("--train requires --model and --seed")
    if not args.accept_pilot:
        parser.error("Review --pilot timings first; --accept-pilot acknowledges the projected cost")
    saved = json.loads((OUT/"pilot.json").read_text())
    if canonical(saved["specification"]) != canonical(spec):
        raise ValueError("Incompatible pilot; repeat pilot before training")
    print("Reviewed pilot estimate:", saved["results"][args.model], flush=True)
    freeze(OUT/"settings.json", spec)
    train(args.model, args.seed, args.resume, x, tables, day, spec, preprocessing)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
