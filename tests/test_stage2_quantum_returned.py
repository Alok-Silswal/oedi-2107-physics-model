import copy
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import evaluate_stage2_quantum_returned as returned
from test_stage2_quantum_evaluation import runs


def test_numeric_preprocessing_tolerance_rejects_meaningful_changes(runs):
    spec, _, _, _ = runs
    local = joblib.load(returned.baseline.CORRECTION/"ridge.joblib")
    preprocessing = joblib.load(returned.native.OUT/"preprocessing.joblib")
    preprocessing["scaler"].mean_[0] += 5e-11
    preprocessing["scaler"].scale_[0] += 7e-11
    preprocessing["scaler"].var_[0] += 4e-8
    deltas = returned.preprocessing_compatibility(preprocessing, local, spec)
    assert deltas["mean_"] > 0 and deltas["var_"] > 0
    assert returned.close_numeric(95.7946774290642,95.79467742904654,1e-9,"scale") > 0
    with pytest.raises(ValueError,match="scale"):
        returned.close_numeric(95.795,95.79467742904654,1e-9,"scale")
    for attribute in ("mean_", "scale_", "var_"):
        changed = copy.deepcopy(preprocessing)
        getattr(changed["scaler"],attribute)[0] += .001
        with pytest.raises(ValueError,match="scaler"):
            returned.preprocessing_compatibility(changed,local,spec)
    changed = copy.deepcopy(preprocessing)
    changed["processor"].transformer_list[1][1].features_ = np.arange(7)[::-1]
    with pytest.raises(ValueError,match="schema"):
        returned.preprocessing_compatibility(changed,local,spec)
    changed = copy.deepcopy(preprocessing)
    changed["processor"].transformer_list[0][1].strategy = "most_frequent"
    with pytest.raises(ValueError,match="configuration"):
        returned.preprocessing_compatibility(changed,local,spec)
    changed = copy.deepcopy(preprocessing)
    changed["scaler"].n_samples_seen_ += 1
    with pytest.raises(ValueError,match="training count"):
        returned.preprocessing_compatibility(changed,local,spec)
    changed = copy.deepcopy(preprocessing)
    changed["target_scale_kw"] += 1e-11
    with pytest.raises(ValueError,match="returned specification"):
        returned.preprocessing_compatibility(changed,local,spec)


def test_source_check_accepts_only_proven_line_endings(runs):
    spec, _, _, _ = runs
    path = Path("scripts/train_stage2_neural.py")
    data = b"value = 1\r\n# source\r\n"
    path.write_bytes(data)
    recorded = dict(spec["source_hashes"])
    recorded[path.as_posix()] = hashlib.sha256(data.replace(b"\r\n",b"\n")).hexdigest()
    reference = {path.as_posix(): hashlib.sha256(data).hexdigest()}
    checks = returned.source_compatibility(recorded,reference)
    assert checks[path.as_posix()].startswith("CRLF/LF equivalent")
    path.write_bytes(data.replace(b"1",b"2"))
    with pytest.raises(ValueError,match="beyond line endings"):
        returned.source_compatibility(recorded,reference)
    recorded["other/train_stage2_neural.py"] = recorded.pop(path.as_posix())
    with pytest.raises(ValueError,match="identities"):
        returned.source_compatibility(recorded,reference)


@pytest.fixture
def compatible(runs, monkeypatch):
    import torch
    old_spec, _, features, _ = runs
    monkeypatch.setattr(returned,"OUT",returned.original.OUT)
    monkeypatch.setattr(returned,"AUDIT",Path("absent_audit.md"))
    for name in ("compare_stage2_direct.py", "evaluate_stage2_quantum.py"):
        (Path("scripts")/name).write_text("synthetic helper provenance")
    tables,inputs = {},{}
    for split,start in (("training","2021-03-01 12:00"),("validation","2022-03-01 12:00")):
        index = pd.date_range(start,periods=3,freq="5min",name="origin")
        inputs[split] = features.set_axis(index)
        table = pd.DataFrame(dict(target_timestamp=index+pd.Timedelta(minutes=15),actual_target_power_kw=[0.,10.,20.],
                                  physics_forecast_kw=features.physics_forecast_kw.to_numpy()),index=index)
        table["correction_kw"] = table.actual_target_power_kw-table.physics_forecast_kw
        tables[split] = table
    reference_path = returned.native.neural.OUT/"settings.json"
    reference = json.loads(reference_path.read_text())
    reference["frame_fingerprints"] = {name:hashlib.sha256(pd.util.hash_pandas_object(inputs[name].join(
        tables[name][["target_timestamp","actual_target_power_kw","correction_kw"]]),index=True).to_numpy().tobytes()).hexdigest()
        for name in tables}
    reference_path.write_text(json.dumps(reference))
    old = json.loads((returned.baseline.OUT/"settings.json").read_text())
    old["neural_reference"] = reference
    (returned.baseline.OUT/"settings.json").write_text(json.dumps(old))
    spec = dict(old_spec,target_scale_kw=old_spec["target_scale_kw"]+1.8e-11,
                frame_fingerprints={"training":"a"*64,"validation":"b"*64},preprocessing_sha256="c"*64,
                classical_reference_sha256="d"*64,classical_reference_commit=None)
    preprocessing_path = returned.native.OUT/"preprocessing.joblib"
    preprocessing = joblib.load(preprocessing_path)
    preprocessing["target_scale_kw"] = spec["target_scale_kw"]
    preprocessing["scaler"].mean_[0] += 5e-11
    joblib.dump(preprocessing,preprocessing_path)
    (returned.native.OUT/"settings.json").write_text(json.dumps(spec))
    for seed in returned.native.SEEDS:
        directory = returned.native.OUT/"quantum"/f"seed_{seed}"
        for name in ("best.pt","last.pt"):
            state = returned.native.read_checkpoint(directory/name,dict(old_spec,model="quantum",seed=seed))
            state["specification"] = dict(spec,model="quantum",seed=seed)
            torch.save(state,directory/name)
        (directory/"settings.json").write_text(json.dumps(dict(spec,model="quantum",seed=seed)))
        frame = tables["validation"][["target_timestamp","actual_target_power_kw"]].copy()
        frame["predicted_correction_kw"] = (seed-2106)*spec["target_scale_kw"]
        frame["final_power_kw"] = tables["validation"].physics_forecast_kw+frame.predicted_correction_kw
        frame.to_parquet(directory/"validation_predictions.parquet")
    monkeypatch.setattr(returned.native.neural,"load_examples",lambda:(inputs,tables,{},[]))
    return spec,preprocessing


def test_returned_readiness_keeps_unmatched_provenance_and_never_reads_test_rows(compatible,monkeypatch):
    spec,_ = compatible
    before = {p:p.read_bytes() for p in returned.native.OUT.rglob("*") if p.is_file()}
    before.update({p:p.read_bytes() for p in returned.baseline.OUT.iterdir()})
    read = pd.read_parquet
    def guarded(path,*args,**kwargs):
        assert Path(path)!=returned.baseline.OUT/"predictions.parquet","Readiness must not load test rows"
        return read(path,*args,**kwargs)
    monkeypatch.setattr(pd,"read_parquet",guarded)
    def forbidden(*args,**kwargs): raise AssertionError("No held-out evaluation or fitting")
    monkeypatch.setattr(returned,"evaluate",forbidden)
    monkeypatch.setattr(returned.baseline,"load_testing_examples",forbidden)
    monkeypatch.setattr(sys,"argv",["returned","--prepare"])
    returned.main()
    plan=json.loads((returned.OUT/"settings.json").read_text())
    assert plan["quantum_reference"]==spec
    assert plan["compatibility"]["differences"]["frame_fingerprints"]["returned"]==spec["frame_fingerprints"]
    assert "frame_fingerprints" in plan["compatibility"]["unresolved"]
    assert "classical_reference_sha256" in plan["compatibility"]["unresolved"]
    assert set(plan["compatibility"]["validation_replay"])=={"2107","2108","2109"}
    assert not (returned.OUT/"predictions.parquet").exists()
    for path,data in before.items(): assert path.read_bytes()==data


def test_readiness_rejects_unjustified_validation_or_software_differences(compatible):
    spec,_ = compatible
    path=returned.native.OUT/"quantum/seed_2107/validation_predictions.parquet"
    original=pd.read_parquet(path)
    changed=original.copy()
    changed.predicted_correction_kw+=1e-4
    changed.final_power_kw+=1e-4
    changed.to_parquet(path)
    with pytest.raises(ValueError,match="validation replay"):
        returned.readiness()
    changed=original.copy()
    changed.actual_target_power_kw+=1e-8
    changed.to_parquet(path)
    with pytest.raises(AssertionError): returned.readiness()
    original.to_parquet(path)
    root=returned.native.OUT/"settings.json"
    root.write_text(json.dumps(dict(spec,versions={"torch":"999.0"})))
    with pytest.raises(ValueError,match="software"):
        returned.readiness()


def test_returned_evaluation_has_explicit_gate_and_does_not_overwrite(monkeypatch,tmp_path):
    monkeypatch.setattr(returned,"OUT",tmp_path)
    def forbidden(): raise AssertionError("Readiness should not run before the gate")
    monkeypatch.setattr(returned,"readiness",forbidden)
    monkeypatch.setattr(sys,"argv",["returned","--evaluate"])
    with pytest.raises(SystemExit): returned.main()
    (tmp_path/"predictions.parquet").write_bytes(b"existing")
    monkeypatch.setattr(sys,"argv",["returned","--evaluate","--allow-test"])
    with pytest.raises(ValueError,match="overwrite"): returned.main()
    assert (tmp_path/"predictions.parquet").read_bytes()==b"existing"


def test_synthetic_inference_uses_returned_scale_and_keeps_original_results(compatible,monkeypatch):
    spec,preprocessing = compatible
    plan,_ = returned.readiness()
    returned.OUT.mkdir()
    before = {p:p.read_bytes() for p in returned.baseline.OUT.iterdir()}
    actual_transform = returned.transform
    def checked_transform(p,features):
        assert p is preprocessing
        return actual_transform(p,features)
    monkeypatch.setattr(returned,"transform",checked_transform)
    returned.evaluate(plan,preprocessing)
    saved = pd.read_parquet(returned.OUT/"predictions.parquet")
    for seed in returned.native.SEEDS:
        np.testing.assert_array_equal(saved[f"quantum_{seed}_correction_kw"],
                                       np.full(len(saved),(seed-2106)*spec["target_scale_kw"]))
        np.testing.assert_array_equal(saved[f"quantum_{seed}"],
            saved.physics_forecast_kw+saved[f"quantum_{seed}_correction_kw"])
    for path,data in before.items(): assert path.read_bytes()==data
