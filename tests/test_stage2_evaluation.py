import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import evaluate_stage2 as evaluation


def test_prepare_freezes_models_without_test_access(monkeypatch,tmp_path):
    plan = dict(selected_trial="forest_05",seeds=list(range(2107,2112)),disclosure="Previously explored")
    monkeypatch.setattr(evaluation,"OUT",tmp_path/"test")
    monkeypatch.setattr(evaluation,"frozen_artifacts",lambda:plan)
    def forbidden(*args): raise AssertionError("Preparation must not read test examples or evaluate")
    monkeypatch.setattr(evaluation,"load_testing_examples",forbidden)
    monkeypatch.setattr(evaluation,"evaluate",forbidden)
    monkeypatch.setattr(sys,"argv",["evaluate","--prepare"])
    evaluation.main()
    assert json.loads((tmp_path/"test/settings.json").read_text())==plan
    plan["selected_trial"] = "other"
    with pytest.raises(ValueError,match="Incompatible"):
        evaluation.main()


def test_evaluation_requires_manual_gate_and_frozen_plan(monkeypatch,tmp_path):
    monkeypatch.setattr(evaluation,"OUT",tmp_path)
    def forbidden(): raise AssertionError("Not allowed to access test data")
    monkeypatch.setattr(evaluation,"load_testing_examples",forbidden)
    monkeypatch.setattr(evaluation,"frozen_artifacts",lambda:{})
    monkeypatch.setattr(sys,"argv",["evaluate","--evaluate"])
    with pytest.raises(SystemExit): evaluation.main()
    monkeypatch.setattr(sys,"argv",["evaluate","--evaluate","--allow-test"])
    with pytest.raises(ValueError,match="prepare first"): evaluation.main()


def test_metrics_share_rows_preserve_zeros_and_reject_missing_predictions():
    data = pd.DataFrame(dict(actual_target_power_kw=[0.,10.,20.],physics=[-2.,12.,21.],model=[1.,9.,22.]))
    day = np.array([False,True,True])
    result = evaluation.score_predictions(data,["physics","model"],day)
    assert result.all_day_n.eq(3).all() and result.daylight_n.eq(2).all()
    assert result.loc["model","daylight_mae_kw"]==1.5
    assert result.loc["model","daylight_bias_kw"]==.5
    assert result.loc["model","all_day_mae_kw"]==pytest.approx(4/3)
    assert result.loc["physics","all_day_bias_kw"]==pytest.approx(1/3)
    data.loc[0,"model"] = np.nan
    with pytest.raises(ValueError,match="same eligible rows"):
        evaluation.score_predictions(data,["physics","model"],day)


def test_test_loader_enforces_alignment_and_complete_fleet_labels(monkeypatch,tmp_path):
    weather,data = tmp_path/"weather",tmp_path/"data"
    weather.mkdir()
    data.mkdir()
    monkeypatch.setattr(evaluation,"WEATHER",weather)
    monkeypatch.setattr(evaluation,"DATA",data)
    boundary = pd.Timestamp(evaluation.SPLIT["test"])
    origins = pd.date_range(boundary-pd.Timedelta(minutes=15),periods=3,freq="5min",name="origin")
    targets = origins+pd.Timedelta(minutes=15)
    times = pd.date_range(boundary-pd.Timedelta(minutes=30),targets[-1],freq="5min",name="measured_on")
    channels = [f"ac_{n}" for n in range(24)]
    ac = pd.DataFrame(1.,index=times,columns=channels)
    ac.loc[origins[1],channels[0]] = np.nan
    ac.loc[targets[0],:] = 0.
    ac.loc[targets[2],channels[0]] = np.nan
    ac.reset_index().assign(measured_on=times.strftime("%Y-%m-%d %H:%M:%S")).to_parquet(data/"2107_electrical_clean.parquet")
    pd.DataFrame(dict(measured_on=times.strftime("%Y-%m-%d %H:%M:%S"),poa=500.)).to_parquet(data/"2107_irradiance_clean.parquet")
    for name in ("electrical_qc_events","irradiance_qc_events"):
        pd.DataFrame(dict(measured_on=pd.Series([],dtype="str"))).to_parquet(data/f"2107_{name}.parquet")
    monkeypatch.setattr(evaluation,"discover_channels",lambda names:SimpleNamespace(ac_power=channels))
    table = pd.DataFrame(dict(target_timestamp=targets,training_cutoff=pd.Timestamp(evaluation.SPLIT["validation"]),
        forecast_poa=500.,forecast_ambient=20.,forecast_wind=2.,physics_forecast_kw=100.,
        actual_target_power_kw=[0.,24.,np.nan],correction_kw=[-100.,-76.,np.nan],split="testing"),index=origins)
    table.to_parquet(weather/"physics_testing.parquet")
    table.to_parquet(weather/"forecasts_testing.parquet")
    spec = dict(split=evaluation.SPLIT,protocol="elapsed-70-15-15-v1",step="testing",source_hashes={})
    for name in ("testing_settings.json","physics_testing_settings.json"):
        (weather/name).write_text(json.dumps(spec))
    features,paired,_,counts = evaluation.load_testing_examples()
    assert len(features)==1 and paired.actual_target_power_kw.iloc[0]==0
    assert features.origin_power_kw.iloc[0]==24
    assert counts["missing_label_or_physics"]==1 and counts["missing_current_power"]==1
    changed = table.copy()
    changed.loc[origins[0],"target_timestamp"] += pd.Timedelta(minutes=5)
    changed.to_parquet(weather/"forecasts_testing.parquet")
    with pytest.raises(AssertionError): evaluation.load_testing_examples()
    table.to_parquet(weather/"forecasts_testing.parquet")
    changed = table.copy()
    changed.loc[origins[0],"actual_target_power_kw"] = 1.
    changed.to_parquet(weather/"physics_testing.parquet")
    with pytest.raises(AssertionError): evaluation.load_testing_examples()
