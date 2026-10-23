"""Shared CPU quantum construction, checkpoint checks and historical-source compatibility."""
import hashlib
import json
from pathlib import Path
from types import MethodType
from zipfile import ZipFile

import numpy as np
import train_stage2_neural as cpu

CPU_FACTORY = cpu.make_model
TOLERANCES = dict(output_atol=1e-10, output_rtol=1e-8, gradient_atol=1e-9,
                  gradient_rtol=1e-6, finite_difference_atol=1e-7, finite_difference_rtol=1e-4)


def make_model(dimension, seed, backend="lightning.qubit", *, batched=False):
    if backend not in ("lightning.qubit", "default.qubit"):
        raise ValueError("Only CPU quantum backends are supported")
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
        raise ValueError("Incompatible checkpoint")
    if sum(p.numel() for p in state["model"].values())!=parameters or not all(
            p.dtype==torch.float64 and torch.isfinite(p).all() for p in state["model"].values()):
        raise ValueError("Invalid checkpoint model parameters")
    return state


def compare(reference, candidate, x, y):
    import torch
    import pennylane as qml
    models = (reference, candidate)
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
        parameter = dict(reference.named_parameters())[name]
        position = np.unravel_index(np.argmax(np.abs(gradients[0][name])), tuple(parameter.shape))
        original = parameter[position].item()
        values = []
        with torch.no_grad():
            for shift in (1e-6, -1e-6):
                parameter[position] = original+shift
                device = next(reference.parameters()).device
                values.append(torch.nn.HuberLoss(delta=1.0)(reference(x.to(device)), y.to(device)).item())
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


ARCHIVE = Path(__file__).resolve().parents[1]/"docs/historical_sources.zip"


def _registry():
    with ZipFile(ARCHIVE) as archive:
        return json.loads(archive.read("cleanup.json"))


def _identity(path):
    path = Path(path).resolve()
    root = ARCHIVE.parent.parent.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def verify_runtime():
    """Only the reviewed cleanup sources may consume archived training provenance."""
    registry = _registry()
    for name, expected in registry["current_sources"].items():
        path = ARCHIVE.parent.parent/name
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"Unapproved cleanup source change: {name}")
    return registry


def current_source(path):
    name = _identity(path)
    if name == "scripts/train_stage2_quantum_gpu.py" and not Path(path).exists():
        return Path(path).with_name("stage2_shared.py")
    return Path(path)


def historical_source_bytes(path):
    """Read original source for provenance comparison, never for execution."""
    name = _identity(path)
    registry = _registry()
    if name in registry["original_sources"]:
        verify_runtime()
        with ZipFile(ARCHIVE) as archive:
            raw = archive.read(name)
        if hashlib.sha256(raw).hexdigest() != registry["original_sources"][name]:
            raise ValueError(f"Historical source archive changed: {name}")
        return raw
    return Path(path).read_bytes()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_file() and cpu.digest(path) == expected:
            continue
        identity = _identity(path)
        registry = _registry()
        if identity in registry["retired_diagnostics"]:
            raise ValueError(f"Deliberately retired diagnostic is not verified: {name}")
        if identity in registry["original_sources"]:
            raw = historical_source_bytes(path)
            if hashlib.sha256(raw).hexdigest() == expected:
                continue
        raise ValueError(f"Frozen source/artifact changed: {name}")


def preserve_plan(path, current):
    """Keep original scientific plans; admit only verified cleanup source changes."""
    path = Path(path)
    if not path.is_file():
        return current
    saved = json.loads(path.read_text())
    def compare(a, b, key=""):
        if key == "source_hashes":
            # Unchanged returned provenance retains its dedicated line-ending checks.
            if a == b:
                return
            # Older plans contain slash-style aliases; retain and verify every key.
            moves = {n: n.replace("train_stage2_quantum_gpu.py", "stage2_shared.py")
                     for n in [*a, *b] if _identity(n) == "scripts/train_stage2_quantum_gpu.py"}
            old = {moves.get(n, n): h for n, h in a.items()}
            new = {moves.get(n, n): h for n, h in b.items()}
            if len(old) != len(a) or len(new) != len(b) or old.keys() != new.keys():
                raise ValueError("Incompatible source identities")
            verify_hashes(a)
            verify_hashes(b)
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for name in a:
                compare(a[name], b[name], name)
        elif cpu.canonical(a) != cpu.canonical(b):
            raise ValueError(f"Incompatible scientific plan: {path}: {key}")
    compare(saved, current)
    if cpu.canonical(saved) != cpu.canonical(current):
        verify_runtime()
        print("Original plan retained; archived training source and approved current CPU runtime verified.", flush=True)
    return saved
