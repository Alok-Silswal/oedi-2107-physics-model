import json
from pathlib import Path
import sys
import subprocess
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_stage2_quantum_gpu as gpu
import kaggle_stage2_pilot as batch


def test_reused_cpu_circuit_float64_gradients_and_matched_initialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    old = torch.get_default_dtype()
    try:
        torch.set_num_threads(4)
        for seed in gpu.cpu.SEEDS:
            torch.set_default_dtype(torch.float32)
            classical = gpu.CPU_FACTORY("classical", 14, seed)
            torch.set_default_dtype(torch.float64)
            quantum = gpu.make_model(14, seed, "lightning.qubit")
            assert sum(p.numel() for p in quantum.parameters())==89
            for name in ("projection", "head"):
                for a, b in zip(getattr(classical,name).parameters(), getattr(quantum,name).parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = gpu.make_model(14, 2107, "lightning.qubit")
        second = gpu.make_model(14, 2107, "lightning.qubit")
        x = torch.randn(3,14,generator=torch.Generator().manual_seed(11),dtype=torch.float64)
        result = gpu.compare(first,second,x,torch.tensor([0.4,-0.3,1.2]))
        assert result["passed"] and result["expectation_dtype"]=="float64"
        with pytest.raises(ValueError, match="GPU statevector"):
            gpu.gpu_evidence(first)
    finally:
        torch.set_default_dtype(old)


def test_gpu_pilot_requires_acceptance_success_and_exact_compatibility(tmp_path):
    spec = dict(features=["past_only"], seed=2107, settings=dict(backend="lightning.gpu"))
    path = tmp_path/"gpu_pilot.json"
    saved = dict(status="complete", specification=spec, correctness=dict(passed=True), gpu=dict(verified=True))
    path.write_text(json.dumps(saved))
    assert gpu.require_pilot(path,spec,True)==saved
    with pytest.raises(ValueError, match="explicit"):
        gpu.require_pilot(path,spec,False)
    with pytest.raises(ValueError, match="incompatible"):
        gpu.require_pilot(path,dict(spec,features=["changed"]),True)
    saved["gpu"]["verified"] = False
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="successfully"):
        gpu.require_pilot(path,spec,True)


def test_safe_recovery_extract_rejects_traversal_and_overwriting(tmp_path):
    archive = tmp_path/"resume.zip"
    target = tmp_path/"working"
    target.mkdir()
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        batch.extract(archive,target)
    assert not (tmp_path/"outside").exists()
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("last.pt", "checkpoint")
    batch.extract(archive,target)
    with pytest.raises(ValueError, match="overwrite"):
        batch.extract(archive,target)
    assert (target/"last.pt").read_text()=="checkpoint"


def test_restricted_checkpoint_roundtrip_and_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    model = gpu.make_model(14,2107,"lightning.qubit")
    path = tmp_path/"last.pt"
    spec = dict(backend="lightning.gpu", seed=2107)
    gpu.cpu.checkpoint(path,dict(specification=spec,model=model.state_dict(),numpy_rng=np.random.get_state()))
    state = gpu.read_checkpoint(path,spec)
    torch.testing.assert_close(state["model"]["weights"],model.weights,rtol=0,atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        gpu.read_checkpoint(path,dict(spec,seed=2108))


@pytest.mark.parametrize("exit_code", [0, 7])
def test_finite_gpu_pilot_has_no_transition_retry_or_timeout(monkeypatch, tmp_path, exit_code):
    project, work, stage = (tmp_path/name for name in ("input", "working", "stage"))
    work.mkdir()
    for name in ("scripts", "src", "data", "outputs/2107_stage2_weather", "outputs/2107_stage2_correction"):
        (project/name).mkdir(parents=True)
    (project/"pyproject.toml").write_text('[project.optional-dependencies]\nneural-quantum=[]\nquantum-gpu=[]\n')
    (project/"classical_reference_settings.json").write_text(json.dumps(dict(versions=dict(torch="2.14.1+cpu"))))
    monkeypatch.setattr(Path, "symlink_to", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch.venv.EnvBuilder, "create", lambda *args, **kwargs: None)
    commands = []
    def child(command, **kwargs):
        assert "timeout" not in kwargs
        commands.append(command)
        return subprocess.CompletedProcess(command, exit_code if command[-1]=="--pilot" else 0)
    monkeypatch.setattr(batch.subprocess, "run", child)
    assert batch.run(project,work,stage)==exit_code
    assert len(commands)==3 and commands[-1][-1]=="--pilot"
    assert not any("--train-all" in command for command in commands)
    status = json.loads((work/"quantum_gpu_batch_status.json").read_text())
    assert status["status"]==("complete" if exit_code==0 else "failed")
    assert (work/"quantum_gpu_batch.log").exists()


def test_full_batch_requires_manual_acceptance_before_setup(monkeypatch, tmp_path):
    project, work, stage = (tmp_path/name for name in ("input", "working", "stage"))
    work.mkdir()
    for name in ("scripts", "src", "data", "outputs/2107_stage2_weather", "outputs/2107_stage2_correction"):
        (project/name).mkdir(parents=True)
    monkeypatch.setattr(batch, "JOB_MODE", "training")
    monkeypatch.setattr(batch, "ACCEPTED_PILOT_SHA256", None)
    monkeypatch.setattr(Path, "symlink_to", lambda *args, **kwargs: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unaccepted job must not install or train")
    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    assert batch.run(project,work,stage)==1
    assert "gated" in json.loads((work/"quantum_gpu_batch_status.json").read_text())["error"]


@pytest.mark.parametrize("fail_seed", [None, 2109])
def test_five_seed_driver_preserves_examples_resumes_and_stops_on_failure(monkeypatch, tmp_path, fail_seed):
    monkeypatch.setattr(gpu, "OUT", tmp_path)
    monkeypatch.setattr(gpu.cpu, "summary", lambda: None)
    prior = gpu.cpu.OUT, gpu.cpu.make_model, gpu.cpu.read_checkpoint
    directory = tmp_path/"quantum/seed_2108"
    directory.mkdir(parents=True)
    (directory/"last.pt").write_bytes(b"synthetic pending checkpoint")
    x, tables, day, spec, preprocessing = object(), object(), object(), {}, object()
    calls = []
    def training(kind, seed, resume, actual_x, actual_tables, actual_day, actual_spec, actual_preprocessing):
        assert kind=="quantum" and actual_x is x and actual_tables is tables
        assert actual_day is day and actual_preprocessing is preprocessing
        assert actual_spec is spec and resume==(seed==2108)
        calls.append(seed)
        if seed==fail_seed:
            raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(gpu.cpu, "train", training)
    if fail_seed is None:
        gpu.train_all(x,tables,day,spec,preprocessing)
        assert calls==gpu.cpu.SEEDS
    else:
        with pytest.raises(RuntimeError, match="interruption"):
            gpu.train_all(x,tables,day,spec,preprocessing)
        assert calls==[2107,2108,2109]
    assert (gpu.cpu.OUT,gpu.cpu.make_model,gpu.cpu.read_checkpoint)==prior
