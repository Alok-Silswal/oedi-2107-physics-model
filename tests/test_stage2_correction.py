import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_stage2_correction import correction_features, examples, FEATURES
from calculate_stage2_physics import complete_fleet_power
from forecast_stage2_weather import preprocess


def sample():
    origins=pd.date_range("2020-01-01 10:00",periods=3,freq="5min",name="origin")
    table=pd.DataFrame({"target_timestamp":origins+pd.Timedelta(minutes=15),
        "forecast_poa":500.,"forecast_ambient":20.,"forecast_wind":2.,
        "physics_forecast_kw":100.,"actual_target_power_kw":[0.,200.,90.],
        "correction_kw":[-100.,100.,-10.]},index=origins)
    plant=pd.Series([0.,18.,19.],index=origins)
    poa=pd.Series(500.,index=origins)
    return table,plant,poa


def test_features_are_exact_causal_and_exclude_target_outcomes():
    table,plant,poa=sample()
    f=correction_features(table,plant,poa)
    assert f.columns.tolist()==FEATURES
    assert f.power_change_15_kw.isna().all()
    shifted=plant.copy(); shifted.index-=pd.Timedelta(minutes=15)
    with_history=pd.concat([shifted,plant]).sort_index()
    assert correction_features(table,with_history,poa).power_change_15_kw.eq(0).all()
    table['actual_target_power_kw']*=1e6
    table['correction_kw']*=1e6
    table['actual_poa']=1e6
    future=table.target_timestamp
    changed_plant=pd.concat([plant,pd.Series(1e9,index=future)]).sort_index()
    changed_poa=pd.concat([poa,pd.Series(1e9,index=future)]).sort_index()
    pd.testing.assert_frame_equal(f,correction_features(table,changed_plant,changed_poa))
    bad=table.copy(); bad.target_timestamp+=pd.Timedelta(minutes=5)
    with pytest.raises(ValueError,match='15'):
        correction_features(bad,plant,poa)


def test_missing_current_fleet_is_excluded_but_old_gap_zero_and_signed_labels_remain():
    table,plant,poa=sample()
    ac=pd.DataFrame(0.,index=plant.index,columns=range(24))
    ac.iloc[1,0]=np.nan
    current=complete_fleet_power(ac,[])
    f,paired,counts=examples(table,current,poa)
    assert len(f)==2 and counts['missing_current_power']==1
    assert counts['older_power_missing_retained']==2
    assert f.origin_power_kw.eq(0).all()
    assert paired.actual_target_power_kw.iloc[0]==0
    assert paired.correction_kw.iloc[0]==-100
    assert examples(table,plant,poa)[1].correction_kw.iloc[1]==100
    missing=table.copy(); missing.loc[table.index[2],['actual_target_power_kw','correction_kw']]=np.nan
    assert len(examples(missing,plant,poa)[0])==2
    bad=table.copy(); bad.correction_kw*=-1
    with pytest.raises(AssertionError): examples(bad,plant,poa)


def test_imputer_and_scaler_fit_training_only_with_missing_indicator():
    train=pd.DataFrame({'origin_power_kw':[0.,10.,20.],'power_change_15_kw':[np.nan,2.,4.]})
    valid=pd.DataFrame({'origin_power_kw':[1000.],'power_change_15_kw':[np.nan]})
    x,xv,scaler,processor=preprocess(train,valid,True)
    changed=valid.copy(); changed.origin_power_kw*=1e6
    again,_,other,_=preprocess(train,changed,True)
    np.testing.assert_array_equal(x,again)
    np.testing.assert_array_equal(scaler.mean_,other.mean_)
    assert xv[0,1]==3 and xv[0,-1]==1
    np.testing.assert_array_equal(xv,processor.transform(valid))


def test_hpo_candidates_are_deterministic_unique_bounded_and_include_baselines():
    import compare_stage2_correction as correction
    from sklearn.model_selection import ParameterGrid
    first=correction.search_candidates()
    assert correction.canonical(first)==correction.canonical(correction.search_candidates())
    assert len(first)==43
    for family,budget in [("ridge",7),("forest",12),("histogram",12),("xgboost",12)]:
        rows=[c for c in first if c["family"]==family]
        assert len(rows)==budget and sum(c["baseline"] for c in rows)==1
        assert len({correction.canonical(c["config"]) for c in rows})==budget
        if family!="ridge":
            space=list(ParameterGrid(correction.HPO_SPACES[family]))
            assert rows[0]["baseline"]
            assert all(c["config"] in space for c in rows)
        assert all(c["effective_settings"]["random_state"]==2107 for c in rows)
    forests=[c for c in first if c["family"]=="forest"]
    assert all(c["effective_settings"]["n_estimators"]==200 and c["effective_settings"]["max_depth"] in (6,12,18) and c["effective_settings"]["n_jobs"]==4 for c in forests)
    assert all(not c["effective_settings"]["early_stopping"] for c in first if c["family"]=="histogram")


def test_hpo_cache_requires_identical_candidates_data_and_code(tmp_path):
    import compare_stage2_correction as correction
    import copy
    path=tmp_path/'settings.json'
    plan=dict(candidates=correction.search_candidates(),source_hashes={'data':'a','code':'b'})
    correction.freeze_search(path,plan)
    original=path.read_bytes()
    correction.freeze_search(path,plan)
    for key in ('data','code'):
        changed=copy.deepcopy(plan); changed['source_hashes'][key]='changed'
        with pytest.raises(ValueError,match='Incompatible'):
            correction.freeze_search(path,changed)
    changed=copy.deepcopy(plan); changed['candidates'].reverse()
    with pytest.raises(ValueError,match='Incompatible'):
        correction.freeze_search(path,changed)
    assert path.read_bytes()==original


def test_hpo_preparation_freezes_plan_without_fitting_or_predicting(monkeypatch,tmp_path):
    import compare_stage2_correction as correction
    import json
    table,plant,poa=sample()
    x=correction_features(table,plant,poa)
    monkeypatch.setattr(correction,'OUT',tmp_path)
    monkeypatch.setattr(correction,'load_examples',lambda:({'training':x,'validation':x},{'training':table,'validation':table},{},[]))
    monkeypatch.setattr(correction,'baseline_reuse',lambda *args:({}, {'ridge':'not compatible'},None))
    def forbidden(*args,**kwargs):
        raise AssertionError('Fit or prediction during preparation')
    monkeypatch.setattr(correction,'preprocess',forbidden)
    original=correction.estimator
    def guarded(family,config):
        model=original(family,config)
        model.fit=forbidden
        model.predict=forbidden
        return model
    monkeypatch.setattr(correction,'estimator',guarded)
    correction.run_search(prepare=True)
    saved=json.loads((tmp_path/'hpo'/'settings.json').read_text())
    assert len(saved['plan']['candidates'])==43
    assert sorted(p.name for p in (tmp_path/'hpo').iterdir())==['settings.json']


def test_hpo_cache_accepts_relative_absolute_alias_but_preserves_code_hash_check(tmp_path,monkeypatch):
    import compare_stage2_correction as correction
    import json
    monkeypatch.chdir(tmp_path)
    source=tmp_path/"correction.py"
    source.write_text("source")
    path=tmp_path/"settings.json"
    saved={"source_hashes":{"correction.py":"unchanged"},"candidates":[{"id":"ridge_00"}]}
    path.write_text(json.dumps({"plan":saved}))
    before=path.read_bytes()
    current={**saved,"source_hashes":{str(source):"unchanged"}}
    correction.freeze_search(path,current)
    assert path.read_bytes()==before
    current["source_hashes"][str(source)]="different code"
    with pytest.raises(ValueError,match="Incompatible"):
        correction.freeze_search(path,current)
    assert path.read_bytes()==before
