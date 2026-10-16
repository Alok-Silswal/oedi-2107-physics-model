"""GPU simulator pilot and explicitly accepted five-seed correction training."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import time
from types import MethodType

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import train_stage2_neural as cpu

OUT = Path("outputs/2107_stage2_quantum_gpu")
CPU_FACTORY = cpu.make_model
TOLERANCES = dict(output_atol=1e-10, output_rtol=1e-8, gradient_atol=1e-9,
                  gradient_rtol=1e-6, finite_difference_atol=1e-7, finite_difference_rtol=1e-4)


def make_model(dimension, seed, backend="lightning.gpu", *, batched=False):
    import torch
    import pennylane as qml
    # Preserve the completed classical model's float32 initialization draws.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = CPU_FACTORY("quantum", dimension, seed)
    finally:
        torch.set_default_dtype(previous)
    if backend == "default.qubit":
        device = qml.device(backend, wires=4, shots=None, max_workers=None)
        differentiation = "backprop"
    else:
        device = qml.device(backend, wires=4, shots=None, c_dtype=np.complex128)
        differentiation = "adjoint"
    model.circuit = qml.QNode(model.circuit.func, device, interface="torch", diff_method=differentiation)
    if batched:
        def forward(self, x):
            angles = torch.tanh(self.projection(x))*torch.pi
            # Reuse angles[wire] in the original circuit without changing its gates.
            hidden = torch.stack(self.circuit(angles.T, self.weights), dim=-1)
            return self.head(hidden.to(dtype=self.head.weight.dtype)).squeeze(-1)
        model.forward = MethodType(forward, model)
    return model


def read_checkpoint(path, spec, parameters=89):
    import torch
    allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType,
               np._core.multiarray.scalar, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(path, map_location="cpu", weights_only=True)
    if cpu.canonical(state["specification"]) != cpu.canonical(spec):
        raise ValueError("Incompatible GPU checkpoint")
    if sum(p.numel() for p in state["model"].values())!=parameters or not all(
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
    predictions, gradients, losses = [], [], []
    for model in models:
        device = next(model.parameters()).device
        a, b = x.to(device), y.to(device)
        with torch.no_grad():
            angles = torch.tanh(model.projection(a[0]))*torch.pi
            expectations = torch.stack(model.circuit(angles, model.weights))
            if expectations.dtype != torch.float64 or not torch.isfinite(expectations).all():
                raise ValueError("Simulator expectations must be finite float64")
        model.zero_grad(set_to_none=True)
        with qml.Tracker(model.circuit.device) as tracker:
            prediction = model(a)
            loss = torch.nn.HuberLoss(delta=1.0)(prediction, b)
            loss.backward()
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
        predictions.append(prediction.detach().cpu().numpy())
        gradients.append({name: p.grad.detach().cpu().numpy().copy() for name, p in params.items()})
        losses.append(loss.item())
    np.testing.assert_allclose(*predictions, atol=TOLERANCES["output_atol"], rtol=TOLERANCES["output_rtol"])
    np.testing.assert_allclose(*losses, atol=TOLERANCES["output_atol"], rtol=TOLERANCES["output_rtol"])
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
                device = next(cpu_model.parameters()).device
                values.append(torch.nn.HuberLoss(delta=1.0)(cpu_model(x.to(device)), y.to(device)).item())
            parameter[position] = original
        numeric = (values[0]-values[1])/2e-6
        np.testing.assert_allclose(numeric, gradients[0][name][position],
                                   atol=TOLERANCES["finite_difference_atol"], rtol=TOLERANCES["finite_difference_rtol"])
        finite_differences[name] = dict(numeric=numeric, analytic=float(gradients[0][name][position]))
    for model in models:
        model.zero_grad(set_to_none=True)
    return dict(passed=True, output_max_abs=float(np.max(np.abs(predictions[0]-predictions[1]))),
                loss_max_abs=abs(losses[0]-losses[1]),
                expectation_dtype="float64",
                gradient_max_abs=max(float(np.max(np.abs(gradients[0][n]-gradients[1][n]))) for n in gradients[0]),
                finite_differences=finite_differences)


def synchronize(model):
    import torch
    if model.circuit.device.name == "default.qubit":
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return
    if model.circuit.device.name == "lightning.qubit":
        return
    if model.circuit.device.name != "lightning.gpu":
        raise ValueError("Unsupported timing backend")
    from pennylane_lightning.lightning_gpu_ops import DevPool
    if DevPool.getTotalDevices() != 1:
        raise ValueError("Timing requires exactly one visible GPU")
    DevPool.setDeviceID(0)
    DevPool().syncDevice()  # Native cudaDeviceSynchronize; Torch is CPU-only.


def timing(model, x, y, training_n, validation_n, batch_size=128):
    import torch
    device = next(model.parameters()).device
    x, y = x.to(device), y.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    def update(a, b):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.HuberLoss(delta=1.0)(model(a), b)
        loss.backward()
        optimizer.step()
        loss.item()  # Include the existing trainer's per-batch loss logging transfer.
    def infer(a):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(batch).cpu().numpy() for batch in a.split(batch_size)])
    update(x[:batch_size], y[:batch_size])
    synchronize(model)
    begin = time.perf_counter()
    for i in range(0, len(x), batch_size):
        update(x[i:i+batch_size], y[i:i+batch_size])
    synchronize(model)
    updates = time.perf_counter()-begin
    infer(x[:batch_size])
    synchronize(model)
    begin = time.perf_counter()
    infer(x)
    synchronize(model)
    inference = time.perf_counter()-begin
    batches = math.ceil(len(x)/batch_size)
    epoch = updates/batches * math.ceil(training_n/batch_size)
    validation = inference/batches * math.ceil(validation_n/batch_size)
    return dict(update_batches_seconds=updates, inference_batches_seconds=inference,
                training_epoch_seconds=epoch, full_validation_seconds=validation,
                rough_three_seeds_100_epochs_hours=300*(epoch+validation)/3600,
                rough_five_seeds_100_epochs_hours=500*(epoch+validation)/3600)


def summarize_timings(values):
    summary = {}
    for key in values[0]:
        numbers = [value[key] for value in values]
        median = float(np.median(numbers))
        summary[key] = dict(median=median, minimum=min(numbers), maximum=max(numbers),
                            relative_range=(max(numbers)-min(numbers))/median)
    return summary


def batch_diagnostic(reference, current, candidate, x, y, training_n, validation_n):
    import pennylane as qml
    if metadata.version("pennylane-lightning-gpu") != "0.45.0":
        raise ValueError("Diagnostic requires pinned Lightning GPU 0.45.0")
    angles = candidate.projection(x[:128]).tanh()*np.pi
    tape = candidate.circuit.construct((angles.T, candidate.weights), {})
    if not candidate.circuit.device.supports_derivatives(
            qml.devices.ExecutionConfig(gradient_method="adjoint"), tape):
        raise ValueError("Broadcast QNode adjoint gradients unsupported; no fallback or timing")
    # Check one complete 128-example batch before any optimizer update or timing.
    checks = dict(cpu_to_row=compare(reference, current, x[:128], y[:128]),
                  row_to_broadcast=compare(current, candidate, x[:128], y[:128]))
    hardware = {name:gpu_evidence(model) for name,model in (("row",current), ("broadcast",candidate))}
    models = dict(cpu=reference, row=current, broadcast=candidate)
    initial = {name:p.clone() for name,p in current.state_dict().items()}
    runs = {name:[] for name in models}
    names = list(models)
    for repeat in range(5):
        # Rotate timing order and reset parameters/Adam for matched repetitions.
        for name in names[repeat % 3:]+names[:repeat % 3]:
            model = models[name]
            model.load_state_dict(initial)
            runs[name].append(timing(model, x, y, training_n, validation_n))
            print(f"Diagnostic repetition {repeat+1}/5: {name} complete", flush=True)
    summaries = {name:summarize_timings(values) for name,values in runs.items()}
    ratios = {key:dict(row_over_broadcast=summaries["row"][key]["median"]/summaries["broadcast"][key]["median"],
                       cpu_over_row=summaries["cpu"][key]["median"]/summaries["row"][key]["median"])
              for key in ("update_batches_seconds", "inference_batches_seconds",
                          "rough_five_seeds_100_epochs_hours")}
    return dict(equivalence=checks, gpu_paths=hardware, repetitions=runs, timing_summary=summaries,
                speed_ratios=ratios, synchronization="DevPool.setDeviceID(0); DevPool.syncDevice (cudaDeviceSynchronize)",
                batching="One broadcast QNode call per batch; backend expands into individual circuits, not native vectorized statevectors",
                limitations="Median/min/max over five warmed repetitions; inspect spread before concluding speedup. Extrapolation excludes setup/I/O and assumes no early stopping. Diagnostic cannot authorize training.")


def broadcast_evidence(model, x):
    import torch
    import pennylane as qml
    device = next(model.parameters()).device
    with torch.no_grad():
        a = x.to(device)
        with qml.Tracker(model.circuit.device) as tracker:
            prediction = model(a)
        angles = model.projection(a).tanh().T*torch.pi
        tape = model.circuit.construct((angles, model.weights), {})
        state = qml.execute((tape.copy(measurements=[qml.state()]),), model.circuit.device,
                            interface="torch", diff_method="backprop")[0]
    if (model.circuit.device.name!="default.qubit" or model.circuit.diff_method!="backprop"
            or tracker.totals.get("simulations")!=1 or tuple(state.shape)!=(len(x),16)
            or state.dtype!=torch.complex128 or state.device!=device
            or prediction.dtype!=torch.float64 or not torch.isfinite(state).all()):
        raise ValueError("Native batched complex128 simulation on the requested device was not verified")
    return dict(backend="default.qubit", differentiation="backprop", torch_device=str(device),
                state_dtype=str(state.dtype), state_shape=list(state.shape), tracker=tracker.totals)


def broadcast_benchmark(x, tables, spec, device, cpu_reference=None):
    import torch
    if device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA-enabled Torch and a GPU are required; no CPU fallback or installation")
    path = OUT.parent/"2107_stage2_quantum_broadcast"/f"benchmark_{device}.json"
    if path.exists():
        raise ValueError(f"Existing benchmark will not be overwritten: {path}")
    indices = np.random.default_rng(2107).choice(len(x["training"]),1024,replace=False)
    sample = torch.from_numpy(x["training"][indices])
    target = torch.from_numpy(tables["training"].correction_kw.to_numpy()[indices]/spec["target_scale_kw"])
    reference = make_model(spec["dimension"],2107,"lightning.qubit")
    candidate = make_model(spec["dimension"],2107,"default.qubit",batched=True).to(device)
    record = dict(status="failed", workload="native_broadcast_benchmark", reference_specification=spec,
        execution=dict(backend="default.qubit", differentiation="backprop", shots=None,
                       torch_dtype="float64", state_dtype="complex128", torch_device=device),
        benchmark_source_sha256=cpu.digest(Path(__file__)), device=device,
        hardware=torch.cuda.get_device_name(0) if device=="cuda" else "CPU (four threads)",
        cuda_version=torch.version.cuda, seed=2107, batch_sizes=[128,256] if cpu_reference else [64,128,256], repetitions=5,
        sample_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), seeds_estimated=[2107,2108,2109],
        parameters=sum(p.numel() for p in candidate.parameters()),
        gpu_budget=dict(weekly_hours=30, planning_reserve_hours=5, compute_hours=25),
        limitations="Timing only, not model selection or training approval. Validation time is extrapolated from training-input inference. Estimates assume all 100 epochs, exclude setup/data transfer/checkpoint/final-evaluation overhead, and are not guaranteed bounds. Batch 64/256 is diagnostic only; agreed training batch remains 128.")
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if cpu_reference:
            saved = json.loads(Path(cpu_reference).read_text())
            if (saved.get("status")!="complete" or saved.get("device")!="cpu"
                    or saved.get("execution")!=dict(record["execution"],torch_device="cpu")
                    or any(saved.get(key)!=record[key] for key in ("sample_sha256","parameters","seed","repetitions"))
                    or not all(saved.get("equivalence",{}).get(str(n),{}).get("passed") for n in (128,256))):
                raise ValueError("Incompatible saved CPU benchmark")
            for key in ("architecture","settings","split","features","dimension","counts","daylight_n",
                        "target_scale_kw","target_scale_ddof","target_centered","frame_fingerprints","preprocessing_sha256"):
                if cpu.canonical(saved["reference_specification"][key])!=cpu.canonical(spec[key]):
                    raise ValueError(f"CPU benchmark differs: {key}")
            record["cpu_reference"] = dict(sha256=cpu.digest(Path(cpu_reference)),
                versions=saved["reference_specification"]["versions"], source_sha256=saved["benchmark_source_sha256"],
                summary={name:saved["summary"][name] for name in ("broadcast_128","broadcast_256")},
                limitation="Different machines and potentially Torch releases; this is not an isolated hardware-speed comparison.")
        record["equivalence"] = {str(size):compare(reference,candidate,sample[:size],target[:size])
                                 for size in record["batch_sizes"]}
        record["native_broadcast"] = broadcast_evidence(candidate,sample[:256])
        initial = {name:p.clone() for name,p in reference.state_dict().items()}
        configurations = {} if cpu_reference else dict(reference_row_128=(reference,128))
        configurations.update({f"broadcast_{size}":(candidate,size) for size in record["batch_sizes"]})
        record["timings"] = {name:[] for name in configurations}
        names = list(configurations)
        for repeat in range(5):
            for name in names[repeat % len(names):]+names[:repeat % len(names)]:
                model,size = configurations[name]
                model.load_state_dict(initial)
                record["timings"][name].append(timing(model,sample,target,len(x["training"]),len(x["validation"]),size))
                print(f"Broadcast benchmark {repeat+1}/5: {name} complete",flush=True)
        record["summary"] = {name:summarize_timings(values) for name,values in record["timings"].items()}
        metric = "rough_three_seeds_100_epochs_hours"
        if cpu_reference:
            record["cpu_to_gpu_speed_ratios"] = {name:saved["summary"][name][metric]["median"]/values[metric]["median"]
                                                 for name,values in record["summary"].items()}
        else:
            baseline = record["summary"]["reference_row_128"][metric]["median"]
            record["speed_ratios"] = {name:baseline/values[metric]["median"] for name,values in record["summary"].items()}
        if device=="cuda":
            record["within_25_hour_compute_budget"] = {
                name:values[metric]["maximum"]<=25 for name,values in record["summary"].items() if name!="reference_row_128"}
        record["status"] = "complete"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        path.write_text(json.dumps(record,indent=2))
        print(json.dumps(record,indent=2),flush=True)
        print("Saved:",path,flush=True)


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
    mode.add_argument("--diagnostic", action="store_true", help="Optional synchronized row/broadcast comparison; no training approval")
    mode.add_argument("--broadcast-benchmark", action="store_true", help="Bounded native default.qubit/backprop benchmark only")
    mode.add_argument("--train-all", action="store_true")
    parser.add_argument("--pilot-file", type=Path, default=OUT/"gpu_pilot.json")
    parser.add_argument("--accept-pilot", action="store_true")
    parser.add_argument("--device", choices=["cpu","cuda"], help="Required only with --broadcast-benchmark")
    parser.add_argument("--cpu-reference", type=Path, help="GPU benchmark only: compare saved CPU timings; time only batched 128/256")
    parser.add_argument("--benchmark-torch-release", help="Explicit benchmark-only Torch release exception, recorded in results")
    args = parser.parse_args()
    if args.broadcast_benchmark != (args.device is not None):
        parser.error("--broadcast-benchmark requires --device cpu or cuda; --device is only for this benchmark")
    if (args.cpu_reference or args.benchmark_torch_release) and not (args.broadcast_benchmark and args.device=="cuda"):
        parser.error("CPU reference and Torch release exception are only for the CUDA broadcast benchmark")
    if args.broadcast_benchmark and args.device=="cuda" and not torch.cuda.is_available():
        parser.error("CUDA-enabled Torch and a GPU are required; no fallback or installation")
    if args.train_all and not args.accept_pilot:
        parser.error("--train-all requires explicit --accept-pilot after manual review")
    torch.set_num_threads(4)
    x, tables, day, original, preprocessing = cpu.prepare_data()
    reference = json.loads(Path("classical_reference_settings.json").read_text()) if Path("classical_reference_settings.json").exists() else json.loads((cpu.OUT/"settings.json").read_text())
    for key in ("settings", "split", "seeds", "features", "dimension", "counts", "daylight_n", "target_scale_kw", "frame_fingerprints", "preprocessing_sha256", "versions"):
        actual, expected = original[key], reference[key]
        if key=="versions" and args.broadcast_benchmark:
            # A CUDA build of the same Torch release is isolated to this benchmark.
            actual, expected = dict(actual), dict(expected)
            actual["torch"], expected["torch"] = actual["torch"].split("+")[0], expected["torch"].split("+")[0]
            if args.benchmark_torch_release:
                expected["torch"] = args.benchmark_torch_release
        if cpu.canonical(actual) != cpu.canonical(expected):
            raise ValueError(f"Differs from completed classical reference: {key}")
    for name, expected in reference["source_hashes"].items():
        source = Path("scripts")/name.replace("\\", "/").rsplit("/", 1)[-1]
        if cpu.digest(source) != expected:
            raise ValueError(f"Completed classical source provenance changed: {source}")
    original["classical_reference_commit"] = reference["source_commit"]
    torch.set_default_dtype(torch.float64)
    d = original["dimension"]
    if args.broadcast_benchmark:
        if args.cpu_reference:
            broadcast_benchmark(x,tables,original,args.device,cpu_reference=args.cpu_reference)
        else:
            broadcast_benchmark(x,tables,original,args.device)
        return
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
    path = OUT/("gpu_batch_diagnostic.json" if args.diagnostic else "gpu_pilot.json")
    if path.exists():
        raise ValueError("Existing GPU pilot will not be overwritten")
    record = dict(status="failed", specification=spec, gpu=hardware)
    try:
        indices = np.random.default_rng(2107).choice(len(x["training"]), 1024, replace=False)
        sample = torch.from_numpy(x["training"][indices])
        target = torch.from_numpy(tables["training"].correction_kw.to_numpy()[indices]/spec["target_scale_kw"])
        reference_model = make_model(d, 2107, "lightning.qubit")
        record["sample_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
        if args.diagnostic:
            record.update(batch_diagnostic(reference_model, gpu, make_model(d, 2107, batched=True),
                                           sample, target, len(x["training"]), len(x["validation"])))
            record["status"] = "complete"
            return
        record["correctness"] = compare(reference_model, gpu, sample[:8], target[:8])
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
