import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from forecast_stage2_weather import (weather_features, recorded_targets, period_mask,
    constrained, preprocess, forecast_block)
import forecast_stage2_weather as weather


def test_unavailable_or_future_measurements_cannot_change_history():
    index = pd.date_range("2018-06-01 08:00",periods=60,freq="5min")
    poa = pd.Series(np.arange(len(index),dtype=float),index=index)
    env = pd.DataFrame({"ambient_temperature_c":20.,"wind_speed_m_s":2.},index=index[::3])
    origin = pd.DatetimeIndex([index[36]])
    before = weather_features(poa,env,origin)
    changed = poa.copy(); changed.loc[changed.index>origin[0]] = 1e6
    changed_env = env.copy()
    changed_env.loc[changed_env.index>origin[0]-pd.Timedelta(minutes=15)] = 1e6
    pd.testing.assert_frame_equal(before,weather_features(changed,changed_env,origin))
    older_missing = poa.drop(origin[0]-pd.Timedelta(minutes=120))
    features = weather_features(older_missing,env,origin)
    assert np.isnan(features.iloc[0].poa_lag120)
    assert features.iloc[0].poa_lag0 == poa.loc[origin[0]]


def test_targets_use_exact_future_labels_and_preserve_zero():
    index = pd.date_range("2018-06-01 10:00",periods=12,freq="5min")
    poa = pd.Series(0.,index=index).drop(index[4])
    env = pd.DataFrame({"ambient_temperature_c":-2.,"wind_speed_m_s":0.},index=index[::3])
    result = recorded_targets(poa,env,index[:2])
    assert result.iloc[0].poa == 0
    assert result.iloc[0].ambient == -2
    assert result.iloc[0].wind == 0
    assert result.iloc[1].isna().all()


def test_target_geometry_and_qc_labels_are_aligned_to_origin(monkeypatch):
    index = pd.date_range("2018-06-01 10:00",periods=6,freq="5min")
    poa = pd.DataFrame({"poa":0.},index=index)
    env = pd.DataFrame({"ambient_temperature_c":20.,"wind_speed_m_s":2.},
                       index=pd.date_range("2018-06-01 09:00",periods=10,freq="15min"))
    monkeypatch.setattr(weather,"indexed",lambda path,columns=None:poa if "irradiance" in path.name else env)
    monkeypatch.setattr(weather.pd,"read_parquet",lambda path:pd.DataFrame({"measured_on":[index[4]]}))
    monkeypatch.setattr(weather,"solar_elevation",lambda times:pd.Series(times.minute,dtype=float,index=times))
    features,targets,elevation,_ = weather.load_data()
    assert elevation.loc[index[1]]==20  # Origin 10:05, target geometry at 10:20.
    assert pd.isna(targets.loc[index[1],"poa"])
    assert features.loc[index[0],"poa_lag0"]==0


def test_training_cutoff_purges_labels_not_legitimate_past_features():
    index = pd.to_datetime(["2018-06-30 23:40","2018-06-30 23:45","2018-07-01 00:00"])
    assert period_mask(index,"2018-01-01","2018-07-01").tolist()==[True,False,False]
    assert period_mask(index,"2018-07-01","2018-10-01").tolist()==[False,True,True]


def test_imputation_and_scaling_do_not_fit_validation_values():
    train = pd.DataFrame({"a":[1.,np.nan,3.],"empty":[np.nan]*3})
    first = pd.DataFrame({"a":[100.,np.nan],"empty":[1000.,np.nan]})
    second = first*100
    x,xv,scale = preprocess(train,first)
    other,_,other_scale = preprocess(train,second)
    np.testing.assert_array_equal(x,other)
    np.testing.assert_array_equal(scale.mean_,other_scale.mean_)
    assert xv[1,0]==2 and xv[1,1]==0
    assert xv[1,2:].tolist()==[1,1]


def test_forecast_constraints_do_not_clip_temperature_or_upper_values():
    np.testing.assert_array_equal(constrained([-1,2000,500],"poa",[30,30,0]),[0,2000,0])
    np.testing.assert_array_equal(constrained([-2,100],"wind",[0,0]),[0,100])
    np.testing.assert_array_equal(constrained([-20,50],"ambient",[0,0]),[-20,50])


def test_forecast_block_uses_only_earlier_labels_and_keeps_missing_targets(monkeypatch):
    index = pd.date_range("2020-01-01 00:00", periods=20, freq="5min")
    features = pd.DataFrame({"poa_lag0":100., "ambient_lag0":20., "wind_lag0":2.}, index=index)
    targets = pd.DataFrame({"poa":120., "ambient":21., "wind":3.}, index=index)
    cutoff, end = index[10], index[-1]
    targets.loc[index[9]] = np.nan
    elevation = pd.Series(30.,index=index)
    fitted_labels = []

    class DummyModel:
        def fit(self, x, y):
            fitted_labels.append(y.index)
            self.value = y.mean()
        def predict(self, x):
            return np.full(len(x), self.value)

    monkeypatch.setattr(weather,"estimator",lambda *args:DummyModel())
    winners = {v:dict(family="forest",config={}) for v in targets}
    result,models = forecast_block(features,targets,elevation,winners,cutoff,end)
    assert all((i+pd.Timedelta(minutes=15)<cutoff).all() for i in fitted_labels)
    assert result.loc[index[9],["actual_poa","actual_ambient","actual_wind"]].isna().all()
    assert result.loc[index[9],["forecast_poa","forecast_ambient","forecast_wind"]].notna().all()
    changed=targets.copy(); changed.loc[index[7]:] *= 1000
    again,_=forecast_block(features,changed,elevation,winners,cutoff,end)
    pd.testing.assert_frame_equal(result.filter(like="forecast_"),again.filter(like="forecast_"))
    fitted_labels.clear()
    forecast_block(features,changed,elevation,winners,cutoff,end,models)
    assert not fitted_labels


def test_shared_splits_assign_boundary_targets_once_and_keep_earlier_inputs():
    edges=[pd.Timestamp(t) for t in weather.SPLIT.values()]
    start, validation, test, end=edges
    assert validation==(start+(end-start)*.7).ceil("5min")
    assert test==(start+(end-start)*.85).ceil("5min")
    targets=pd.DatetimeIndex([start-pd.Timedelta(minutes=5),start,validation-pd.Timedelta(minutes=5),validation,test-pd.Timedelta(minutes=5),test,end-pd.Timedelta(minutes=5),end])
    masks=weather.split_masks(targets-pd.Timedelta(minutes=15))
    assert masks["training"].tolist()==[False,True,True,False,False,False,False,False]
    assert masks["validation"].tolist()==[False,False,False,True,True,False,False,False]
    assert masks["testing"].tolist()==[False,False,False,False,False,True,True,False]
    assert (sum(masks.values())<=1).all()
    blocks=weather.training_blocks()
    assert blocks[-1][2]==validation
    assert all(a<b<=validation for _,a,b in blocks)
    assert blocks[0][2]==blocks[1][1] and blocks[1][2]==blocks[2][1]


def test_preparation_entry_points_cannot_fit_write_or_calculate_physics(monkeypatch,tmp_path):
    import calculate_stage2_physics as physics
    data=(pd.DataFrame(),pd.DataFrame(),pd.Series(dtype=float),[])
    monkeypatch.setattr(weather,"load_data",lambda:data)
    calls=[]
    monkeypatch.setattr(weather,"prepare",lambda *args:calls.append("prepared"))
    def forbidden(*args,**kwargs):
        raise AssertionError("Scientific work in preparation mode")
    monkeypatch.setattr(weather,"select_models",forbidden)
    monkeypatch.setattr(weather,"estimator",forbidden)
    monkeypatch.setattr(physics,"physics_labels",forbidden)
    monkeypatch.setattr(weather,"OUT",tmp_path/"weather")
    monkeypatch.setattr(physics,"OUT",tmp_path/"physics")
    monkeypatch.setattr(sys,"argv",["script","--prepare"])
    weather.main()
    physics.main()
    assert calls==["prepared","prepared"]
    assert not list(tmp_path.iterdir())


def test_interrupted_selection_reuses_completed_candidates(monkeypatch,tmp_path):
    index=pd.date_range("2020-01-01",periods=20,freq="5min")
    features=pd.DataFrame({"poa_lag0":100.},index=index)
    targets=pd.DataFrame({"poa":120.},index=index)
    elevation=pd.Series(30.,index=index)
    monkeypatch.setattr(weather,"VARIABLES",{"poa":"W/m2"})
    completed=[]
    attempts=[]
    class DummyModel:
        def fit(self,x,y):
            attempts.append(1)
            if len(attempts)==3:
                raise RuntimeError("interrupted")
            completed.append(1)
        def predict(self,x):
            return np.zeros(len(x))
    monkeypatch.setattr(weather,"estimator",lambda *args:DummyModel())
    path=tmp_path/"selection.csv"
    folds=[(index[0],index[10],index[-1])]
    with pytest.raises(RuntimeError,match="interrupted"):
        weather.select_models(features,targets,elevation,folds,path)
    assert len(pd.read_csv(path))==3  # Persistence and two completed candidates.
    attempts.extend([1])
    result,_=weather.select_models(features,targets,elevation,folds,path)
    assert len(completed)==11
    assert len(result)==24


def test_test_execution_requires_explicit_frozen_choices(monkeypatch):
    import calculate_stage2_physics as physics
    def forbidden():
        raise AssertionError("Data loaded before test gate")
    monkeypatch.setattr(weather,"load_data",forbidden)
    monkeypatch.setattr(sys,"argv",["script","--step","testing"])
    for module in (weather,physics):
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code==2


def test_later_steps_reuse_initial_winners_without_selection(monkeypatch,tmp_path):
    import json
    monkeypatch.setattr(weather,"OUT",tmp_path)
    monkeypatch.setattr(weather,"digest",lambda p:"hash")
    monkeypatch.setattr(weather.metadata,"version",lambda p:"1")
    data=(pd.DataFrame(),pd.DataFrame(),pd.Series(dtype=float),[])
    monkeypatch.setattr(weather,"load_data",lambda:data)
    versions={p:"1" for p in ["numpy","pandas","scikit-learn","scipy","pvlib","pyarrow","xgboost","joblib","threadpoolctl"]}
    winners={v:{"family":"ridge","config":{"alpha":10.}} for v in weather.VARIABLES}
    initial=dict(specification=dict(protocol="elapsed-70-15-15-v1",split=weather.SPLIT,
        source_hashes={},versions=versions,seed=weather.SEED,search=weather.SEARCH,code_hash="hash",
        tuning="Select once in initial training period: first 80% fit / last 20% select; freeze configurations for all expanding refits and outer validation/test"),winners=winners)
    (tmp_path/"training_1_settings.json").write_text(json.dumps(initial))
    def forbidden(*args,**kwargs):
        raise AssertionError("Repeated weather selection")
    monkeypatch.setattr(weather,"select_models",forbidden)
    seen=[]
    def forecast(features,targets,elevation,choices,cutoff,end,models):
        seen.append(choices)
        index=pd.DatetimeIndex([pd.Timestamp(cutoff)-pd.Timedelta(minutes=15)],name="origin")
        part=pd.DataFrame({"target_timestamp":index+pd.Timedelta(minutes=15)},index=index)
        for variable in choices:
            part[f"forecast_{variable}"]=1.
        return part,{}
    monkeypatch.setattr(weather,"forecast_block",forecast)
    for step in ("training_2","training_3","validation"):
        monkeypatch.setattr(sys,"argv",["script","--step",step])
        weather.main()
        saved=json.loads((tmp_path/f"{step}_settings.json").read_text())
        assert saved["winners"]==winners
        assert saved["specification"]["frozen_winners"]==winners
    assert len(seen)==9
    assert all(choice[v]==winners[v] for choice in seen for v in choice)
    assert not list(tmp_path.glob("*_selection.csv"))
