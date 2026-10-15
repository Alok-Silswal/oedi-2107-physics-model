"""GPU simulator pilot and explicitly accepted five-seed correction training."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import train_stage2_neural as cpu

OUT = Path("outputs/2107_stage2_quantum_gpu")
CPU_FACTORY = cpu.make_model
TOLERANCES = dict(output_atol=1e-10, output_rtol=1e-8, gradient_atol=1e-9,
                  gradient_rtol=1e-6, finite_difference_atol=1e-7, finite_difference_rtol=1e-4)


def make_model(dimension, seed, backend="lightning.gpu"):
    import torch
    import pennylane as qml
    # Preserve the completed classical model's float32 initialization draws.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = CPU_FACTORY("quantum", dimension, seed)
    finally:
        torch.set_default_dtype(previous)
    device = qml.device(backend, wires=4, shots=None, c_dtype=np.complex128)
    model.circuit = qml.QNode(model.circuit.func, device, interface="torch", diff_method="adjoint")
    return model


def read_checkpoint(path, spec):
    import torch
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType,
               np._core.multiarray.scalar, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location="cpu", weights_only=True)
    if cpu.canonical(state["specification"]) != cpu.canonical(spec):
        raise ValueError("Incompatible GPU checkpoint")
    if sum(p.numel() for p in state["model"].values())!=89 or not all(
            p.dtype==torch.float64 and torch.isfinite(p).all() for p in state["model"].values()):
        raise ValueError("Invalid GPU checkpoint model parameters")
    return state


def gpu_evidence(model):
    device = model.circuit.device
    state = device._statevector
    if device.name != "lightning.gpu" or device.c_dtype != np.complex128 or state is None:
        raise ValueError("No executed complex128 GPU statevector")
    from pennylane_lightning.lightning_gpu_ops import DevPool, get_gpu_arch
    state_class = type(state).__module__+"."+type(state).__name__
    if "lightning_gpu" not in state_class or DevPool.getTotalDevices() < 1:
        raise ValueError("Actual Lightning GPU simulator use was not verified")
    return dict(backend=device.name, statevector=state_class, visible_gpus=DevPool.getTotalDevices(),
                architecture=str(get_gpu_arch()), c_dtype="complex128", verified=True)


def specification(original, hardware):
    spec = dict(original)
    spec["settings"] = dict(original["settings"], backend="lightning.gpu")
    spec["gpu_execution"] = dict(version=1, torch_device="cpu", torch_dtype="float64",
                                 initialization_dtype="float32", c_dtype="complex128", hardware=hardware,
                                 tolerances=TOLERANCES)
    spec["source_hashes"] = {"scripts/"+name.replace("\\", "/").rsplit("/", 1)[-1]: value
                             for name, value in original["source_hashes"].items()}
    spec["source_hashes"]["scripts/train_stage2_quantum_gpu.py"] = cpu.digest(Path(__file__))
    versions = dict(original["versions"])
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name.startswith(("nvidia-", "custatevec-")) or name=="pennylane-lightning-gpu":
            versions[name] = distribution.version
    spec["versions"] = versions
    return spec


def compare(cpu_model, gpu_model, x, y):
    import torch
    import pennylane as qml
    models = (cpu_model, gpu_model)
    predictions, gradients = [], []
    for model in models:
        with torch.no_grad():
            angles = torch.tanh(model.projection(x[0]))*torch.pi
            expectations = torch.stack(model.circuit(angles, model.weights))
            if expectations.dtype != torch.float64 or not torch.isfinite(expectations).all():
                raise ValueError("Simulator expectations must be finite float64")
        model.zero_grad(set_to_none=True)
        with qml.Tracker(model.circuit.device) as tracker:
            prediction = model(x)
            torch.nn.HuberLoss(delta=1.0)(prediction, y).backward()
        if tracker.totals.get("executions", 0) < 1:
            raise ValueError("Simulator execution not tracked")
        if prediction.dtype != torch.float64 or not torch.isfinite(prediction).all():
            raise ValueError("Invalid output precision/values")
        params = dict(model.named_parameters())
        if sum(p.numel() for p in params.values()) != 89:
            raise ValueError("Unexpected parameter count")
        for name in ("projection.weight", "weights"):
            gradient = params[name].grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum()==0:
                raise ValueError(f"No finite nonzero gradient: {name}")
        predictions.append(prediction.detach().numpy())
        gradients.append({name: p.grad.detach().numpy().copy() for name, p in params.items()})
    np.testing.assert_allclose(*predictions, atol=TOLERANCES["output_atol"], rtol=TOLERANCES["output_rtol"])
    for name in gradients[0]:
        np.testing.assert_allclose(gradients[0][name], gradients[1][name],
                                   atol=TOLERANCES["gradient_atol"], rtol=TOLERANCES["gradient_rtol"])
    finite_differences = {}
    for name in ("projection.weight", "weights"):
        parameter = dict(cpu_model.named_parameters())[name]
        position = np.unravel_index(np.argmax(np.abs(gradients[0][name])), tuple(parameter.shape))
        original = parameter[position].item()
        values = []
        with torch.no_grad():
            for shift in (1e-6, -1e-6):
                parameter[position] = original+shift
                values.append(torch.nn.HuberLoss(delta=1.0)(cpu_model(x), y).item())
            parameter[position] = original
        numeric = (values[0]-values[1])/2e-6
        np.testing.assert_allclose(numeric, gradients[0][name][position],
                                   atol=TOLERANCES["finite_difference_atol"], rtol=TOLERANCES["finite_difference_rtol"])
        finite_differences[name] = dict(numeric=numeric, analytic=float(gradients[0][name][position]))
    for model in models:
        model.zero_grad(set_to_none=True)
    return dict(passed=True, output_max_abs=float(np.max(np.abs(predictions[0]-predictions[1]))),
                expectation_dtype="float64",
                gradient_max_abs=max(float(np.max(np.abs(gradients[0][n]-gradients[1][n]))) for n in gradients[0]),
                finite_differences=finite_differences)


def timing(model, x, y, training_n, validation_n):
    import torch
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    def update(a, b):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.HuberLoss(delta=1.0)(model(a), b).backward()
        optimizer.step()
    update(x[:128], y[:128])
    begin = time.perf_counter()
    for i in range(0, 1024, 128):
        update(x[i:i+128], y[i:i+128])
    updates = time.perf_counter()-begin
    cpu.inference(model, x[:128].numpy())
    begin = time.perf_counter()
    cpu.inference(model, x.numpy())
    inference = time.perf_counter()-begin
    epoch = updates/8 * math.ceil(training_n/128)
    validation = inference/8 * math.ceil(validation_n/128)
    return dict(eight_update_batches_seconds=updates, eight_inference_batches_seconds=inference,
                training_epoch_seconds=epoch, full_validation_seconds=validation,
                rough_five_seeds_100_epochs_hours=500*(epoch+validation)/3600)


def require_pilot(path, spec, accepted):
    if not accepted:
        raise ValueError("Full training requires explicit --accept-pilot after manual review")
    saved = json.loads(Path(path).read_text())
    if saved.get("status")!="complete" or not saved.get("correctness", {}).get("passed") or not saved.get("gpu", {}).get("verified"):
        raise ValueError("GPU pilot did not complete successfully")
    if cpu.canonical(saved["specification"]) != cpu.canonical(spec):
        raise ValueError("GPU pilot is incompatible with current data/code/settings/software/hardware")
    return saved


def train_all(x, tables, day, spec, preprocessing):
    # Reuse the unchanged trainer with process-local factory/output bindings.
    previous = cpu.OUT, cpu.make_model, cpu.read_checkpoint
    cpu.OUT = OUT
    cpu.make_model = lambda kind, dimension, seed: make_model(dimension, seed)
    cpu.read_checkpoint = read_checkpoint
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        cpu.freeze(OUT/"settings.json", spec)
        for seed in cpu.SEEDS:
            directory = OUT/"quantum"/f"seed_{seed}"
            run_spec = dict(spec, model="quantum", seed=seed)
            if (directory/"metrics.json").exists():
                state = read_checkpoint(directory/"last.pt", run_spec)
                best = read_checkpoint(directory/"best.pt", run_spec)
                result = json.loads((directory/"metrics.json").read_text())
                if not (state["epoch"]>=100 or state["stale"]>=10) or result["stopping_epoch"]!=state["epoch"] or result["best_epoch"]!=best["epoch"]:
                    raise ValueError(f"Incomplete/inconsistent finished seed {seed}")
                frame = pd.read_parquet(directory/"validation_predictions.parquet")
                if not frame.index.equals(tables["validation"].index) or not np.array_equal(frame.target_timestamp.to_numpy(), tables["validation"].target_timestamp.to_numpy()):
                    raise ValueError("Completed seed validation keys differ")
                np.testing.assert_array_equal(frame.actual_target_power_kw, tables["validation"].actual_target_power_kw)
                np.testing.assert_allclose(frame.final_power_kw,
                    tables["validation"].physics_forecast_kw+frame.predicted_correction_kw, rtol=1e-12, atol=1e-9)
                if not np.isclose(result["daylight_mae_kw"], best["best_mae"], rtol=1e-12, atol=1e-9):
                    raise ValueError("Completed seed best checkpoint metric differs")
                for key, value in cpu.metrics(tables["validation"].actual_target_power_kw.to_numpy(), frame.final_power_kw.to_numpy(), day).items():
                    if not np.isclose(value, result[key], rtol=1e-12, atol=1e-9):
                        raise ValueError("Completed seed metrics differ")
                print(f"Seed {seed}: compatible completed run; skipped", flush=True)
                continue
            cpu.train("quantum", seed, (directory/"last.pt").exists(), x, tables, day, spec, preprocessing)
        cpu.summary()
    finally:
        cpu.OUT, cpu.make_model, cpu.read_checkpoint = previous


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--train-all", action="store_true")
    parser.add_argument("--pilot-file", type=Path, default=OUT/"gpu_pilot.json")
    parser.add_argument("--accept-pilot", action="store_true")
    args = parser.parse_args()
    if args.train_all and not args.accept_pilot:
        parser.error("--train-all requires explicit --accept-pilot after manual review")
    torch.set_num_threads(4)
    x, tables, day, original, preprocessing = cpu.prepare_data()
    reference = json.loads(Path("classical_reference_settings.json").read_text()) if Path("classical_reference_settings.json").exists() else json.loads((cpu.OUT/"settings.json").read_text())
    for key in ("settings", "split", "seeds", "features", "dimension", "counts", "daylight_n", "target_scale_kw", "frame_fingerprints", "preprocessing_sha256", "versions"):
        if cpu.canonical(original[key]) != cpu.canonical(reference[key]):
            raise ValueError(f"Differs from completed classical reference: {key}")
    for name, expected in reference["source_hashes"].items():
        source = Path("scripts")/name.replace("\\", "/").rsplit("/", 1)[-1]
        if cpu.digest(source) != expected:
            raise ValueError(f"Completed classical source provenance changed: {source}")
    original["classical_reference_commit"] = reference["source_commit"]
    torch.set_default_dtype(torch.float64)
    d = original["dimension"]
    if args.prepare:
        try:
            version = metadata.version("pennylane-lightning-gpu")
        except metadata.PackageNotFoundError:
            version = None
        print(json.dumps(dict(dimension=d, parameters=89, counts=original["counts"],
            gpu_plugin_version=version, actual_gpu_verified=False,
            execution="lightning.gpu/complex128; Torch float64 on CPU", tolerances=TOLERANCES)))
        return
    gpu = make_model(d, 2107)
    probe = torch.from_numpy(x["training"][:3])
    with torch.no_grad():
        gpu(probe)
    hardware = gpu_evidence(gpu)
    spec = specification(original, hardware)
    if args.train_all:
        require_pilot(args.pilot_file, spec, args.accept_pilot)
        train_all(x, tables, day, spec, preprocessing)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/"gpu_pilot.json"
    if path.exists():
        raise ValueError("Existing GPU pilot will not be overwritten")
    record = dict(status="failed", specification=spec, gpu=hardware)
    try:
        indices = np.random.default_rng(2107).choice(len(x["training"]), 1024, replace=False)
        sample = torch.from_numpy(x["training"][indices])
        target = torch.from_numpy(tables["training"].correction_kw.to_numpy()[indices]/spec["target_scale_kw"])
        reference_model = make_model(d, 2107, "lightning.qubit")
        record["correctness"] = compare(reference_model, gpu, sample[:8], target[:8])
        record["sample_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
        record["timings"] = {name: timing(model, sample, target, len(x["training"]), len(x["validation"]))
                             for name, model in [("cpu", reference_model), ("gpu", gpu)]}
        record["status"] = "complete"
        record["limitations"] = "Four qubits may be slower on GPU. Epoch-ceiling extrapolation excludes setup/I/O overhead; no early-stopping savings assumed. Torch layers stay on CPU."
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
