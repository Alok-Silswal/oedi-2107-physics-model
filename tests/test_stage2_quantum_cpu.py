import copy
import hashlib
import json
import zipfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import train_stage2_quantum_cpu as native
import stage2_shared as shared


def test_historical_plan_keeps_identity_but_rejects_unapproved_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path/"docs/historical_sources.zip"
    archive.parent.mkdir()
    source = tmp_path/"scripts/model.py"
    source.parent.mkdir()
    old, current = b"original model\r\n", b"approved import migration\n"
    source.write_bytes(current)
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    registry = dict(original_sources={"scripts/model.py": sha(old)},
                    current_sources={"scripts/model.py": sha(current)},
                    retired_diagnostics={"outputs/retired.parquet": "old"})
    with zipfile.ZipFile(archive, "w") as saved:
        saved.writestr("scripts/model.py", old)
        saved.writestr("cleanup.json", json.dumps(registry))
    monkeypatch.setattr(shared, "ARCHIVE", archive)
    source.write_bytes(current.replace(b"\n", b"\r\n"))
    shared.verify_runtime()
    source.write_bytes(current)
    original = dict(dimension=14, split="fixed", source_hashes={"scripts/model.py": sha(old)},
                    reference=dict(source_hashes={"scripts/model.py": sha(old.replace(b"\r\n", b"\n"))}))
    plan = tmp_path/"settings.json"
    plan.write_text(json.dumps(original))
    before = plan.read_bytes()
    updated = dict(original, source_hashes={"scripts/model.py": sha(current)})
    assert shared.preserve_plan(plan, updated) == original
    assert plan.read_bytes() == before
    with pytest.raises(ValueError, match="Frozen source/artifact"):
        shared.verify_hashes(original["reference"]["source_hashes"])
    with pytest.raises(ValueError, match="scientific plan"):
        shared.preserve_plan(plan, dict(updated, dimension=12))
    aliases = dict(original, source_hashes={"scripts/model.py": sha(old), str(source): sha(old)})
    if Path("scripts\\model.py").resolve() == source.resolve():
        aliases["source_hashes"]["scripts\\model.py"] = sha(old)
    plan.write_text(json.dumps(aliases))
    current_aliases = dict(aliases, source_hashes={n: sha(current) for n in aliases["source_hashes"]})
    assert shared.preserve_plan(plan, current_aliases) == aliases
    with pytest.raises(ValueError, match="source identities"):
        shared.preserve_plan(plan, dict(current_aliases, source_hashes={"scripts/model.py": sha(current)}))
    with pytest.raises(ValueError, match="retired diagnostic is not verified"):
        shared.verify_hashes({"outputs/retired.parquet": "old"})
    data = tmp_path/"data.parquet"
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen source/artifact"):
        shared.verify_hashes({str(data): sha(b"original")})
    source.write_bytes(b"arbitrary model change")
    with pytest.raises(ValueError, match="Unapproved cleanup source"):
        shared.verify_hashes(original["source_hashes"])


def test_restricted_checkpoint_rejects_changed_spec_and_nonfinite_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    model = shared.make_model(14, 2107, "default.qubit", batched=True)
    path = tmp_path/"best.pt"
    spec = dict(architecture="fixture", features=list(range(14)))
    torch.save(dict(specification=spec, model=model.state_dict()), path)
    loaded = shared.read_checkpoint(path, spec)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded["model"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        shared.read_checkpoint(path, dict(spec, architecture="different"))
    loaded["model"]["head.bias"][0] = float("nan")
    torch.save(loaded, path)
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        shared.read_checkpoint(path, spec)


@pytest.fixture(autouse=True)
def restore_default_precision():
    torch = pytest.importorskip("torch")
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


def test_native_resume_matches_uninterrupted_updates_and_best_checkpoint(monkeypatch,tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    torch.set_num_threads(4)
    rng = np.random.default_rng(123)
    x = dict(training=rng.normal(size=(17,14)),validation=rng.normal(size=(5,14)))
    tables = dict(training=pd.DataFrame(dict(correction_kw=rng.normal(size=17))),
                  validation=pd.DataFrame(dict(actual_target_power_kw=[0.,1.,2.,0.,3.],physics_forecast_kw=1.,
                                              target_timestamp=pd.date_range("2022-03-01",periods=5,freq="5min"))))
    day = np.array([False,True,True,False,True])
    spec = dict(dimension=14,target_scale_kw=2.,settings=dict(native.SETTINGS,epochs=3),seeds=native.SEEDS)
    checkpoint = native.neural.checkpoint
    uninterrupted = tmp_path/"uninterrupted"
    monkeypatch.setattr(native,"OUT",uninterrupted)
    native.train(2107,False,x,tables,day,spec,{})
    split = tmp_path/"interrupted"
    monkeypatch.setattr(native,"OUT",split)
    def interrupt(path,state):
        checkpoint(path,state)
        if path.name=="last.pt" and state["epoch"]==1:
            raise KeyboardInterrupt("Synthetic interruption after an atomic checkpoint")
    monkeypatch.setattr(native.neural,"checkpoint",interrupt)
    with pytest.raises(KeyboardInterrupt):
        native.train(2107,False,x,tables,day,spec,{})
    monkeypatch.setattr(native.neural,"checkpoint",checkpoint)
    native.train(2107,True,x,tables,day,spec,{})
    run_spec = dict(spec,model="quantum",seed=2107)
    for filename in ("last.pt","best.pt"):
        first = native.read_checkpoint(uninterrupted/"quantum/seed_2107"/filename,run_spec)
        resumed = native.read_checkpoint(split/"quantum/seed_2107"/filename,run_spec)
        assert first["epoch"]==resumed["epoch"]
        for name in first["model"]:
            torch.testing.assert_close(first["model"][name],resumed["model"][name],rtol=0,atol=0)
        torch.testing.assert_close(first["shuffle_rng"],resumed["shuffle_rng"],rtol=0,atol=0)
        for a,b in zip(first["history"],resumed["history"]):
            assert {k:v for k,v in a.items() if k!="seconds"}=={k:v for k,v in b.items() if k!="seconds"}
    pd.testing.assert_frame_equal(pd.read_parquet(uninterrupted/"quantum/seed_2107/validation_predictions.parquet"),
                                  pd.read_parquet(split/"quantum/seed_2107/validation_predictions.parquet"))
    before = (split/"quantum/seed_2107/metrics.json").read_bytes()
    native.train(2107,True,x,tables,day,spec,{})
    assert (split/"quantum/seed_2107/metrics.json").read_bytes()==before
    changed = copy.deepcopy(spec)
    changed["settings"]["batch_size"] = 128
    with pytest.raises(ValueError,match="Incompatible"):
        native.train(2107,True,x,tables,day,changed,{})


def test_training_requires_explicit_manual_acceptance(monkeypatch):
    pytest.importorskip("torch")
    def forbidden(): raise AssertionError("Do not load data before training authorization")
    monkeypatch.setattr(native,"prepare_data",forbidden)
    monkeypatch.setattr(sys,"argv",["native","--train","--seed","2107"])
    with pytest.raises(SystemExit): native.main()


def test_prepare_checks_native_equivalence_without_training_or_writes(monkeypatch,tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    monkeypatch.chdir(tmp_path)
    spec = dict(dimension=14,target_scale_kw=2.,counts={},daylight_n=2,change="batch 256")
    x = dict(training=np.random.default_rng(1).normal(size=(3,14)))
    tables = dict(training=pd.DataFrame(dict(correction_kw=[1.,2.,3.])))
    monkeypatch.setattr(native,"prepare_data",lambda:(x,tables,None,spec,{}))
    def forbidden(*args): raise AssertionError("Preparation must not train or save checkpoints")
    monkeypatch.setattr(native,"train",forbidden)
    monkeypatch.setattr(native.neural,"checkpoint",forbidden)
    monkeypatch.setattr(sys,"argv",["native","--prepare"])
    native.main()
    assert not list(tmp_path.iterdir())


def test_reused_cpu_circuit_float64_gradients_and_matched_initialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    old = torch.get_default_dtype()
    try:
        torch.set_num_threads(4)
        for seed in shared.cpu.SEEDS:
            torch.set_default_dtype(torch.float32)
            classical = shared.CPU_FACTORY("classical", 14, seed)
            torch.set_default_dtype(torch.float64)
            quantum = shared.make_model(14, seed, "lightning.qubit")
            assert sum(p.numel() for p in quantum.parameters())==89
            for name in ("projection", "head"):
                for a, b in zip(getattr(classical,name).parameters(), getattr(quantum,name).parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = shared.make_model(14, 2107, "lightning.qubit")
        second = shared.make_model(14, 2107, "lightning.qubit")
        x = torch.randn(3,14,generator=torch.Generator().manual_seed(11),dtype=torch.float64)
        result = shared.compare(first,second,x,torch.tensor([0.4,-0.3,1.2]))
        assert result["passed"] and result["expectation_dtype"]=="float64"
    finally:
        torch.set_default_dtype(old)



@pytest.mark.parametrize("size", [1, 128])
@pytest.mark.parametrize("backend", ["lightning.qubit", "default.qubit"])
def test_broadcast_qnode_preserves_circuit_outputs_and_all_gradients(size,backend):
    torch = pytest.importorskip("torch")
    qml = pytest.importorskip("pennylane")
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(4)
        current = shared.make_model(14,2107,"lightning.qubit")
        candidate = shared.make_model(14,2107,backend,batched=True)
        assert sum(p.numel() for p in candidate.parameters())==89
        x = torch.randn(size,14,generator=torch.Generator().manual_seed(11))
        y = torch.linspace(-1,1,size)
        result = shared.compare(current,candidate,x,y)
        assert result["passed"] and result["loss_max_abs"] < shared.TOLERANCES["output_atol"]
        if backend=="default.qubit":
            evidence = shared.broadcast_evidence(candidate,x)
            assert evidence["tracker"]["simulations"]==1 and evidence["state_dtype"]=="torch.complex128"
        angles = candidate.projection(x).tanh()*torch.pi
        tape = candidate.circuit.construct((angles.T,candidate.weights),{})
        assert candidate.circuit.device.supports_derivatives(qml.devices.ExecutionConfig(gradient_method=candidate.circuit.diff_method),tape)
        expanded,_ = qml.transforms.broadcast_expand(tape)
        for i, item in enumerate(expanded):
            row = current.circuit.construct((angles[i],current.weights),{})
            qml.assert_equal(item,row)
    finally:
        torch.set_default_dtype(old)
