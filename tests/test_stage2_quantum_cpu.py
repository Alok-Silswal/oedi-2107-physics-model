import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import train_stage2_quantum_cpu as native


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
