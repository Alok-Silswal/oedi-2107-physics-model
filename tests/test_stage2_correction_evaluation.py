import copy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import evaluate_stage2_correction as evaluation
import evaluate_stage2_quantum as shared


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    root = tmp_path/"correction"
    root.mkdir()
    (root/"hpo").mkdir()
    monkeypatch.setattr(evaluation.correction, "OUT", root)
    monkeypatch.setattr(evaluation, "OUT", root/"test")
    monkeypatch.setattr(evaluation.legacy, "OUT", tmp_path/"original")
    monkeypatch.setattr(evaluation.direct, "OUT", tmp_path/"direct")
    evaluation.legacy.OUT.mkdir()
    (evaluation.direct.OUT/"test").mkdir(parents=True)
    (evaluation.direct.OUT/"ml").mkdir()
    features = pd.DataFrame(np.arange(21.).reshape(3,7), columns=evaluation.correction.FEATURES,
                            index=pd.date_range("2023-03-01 12:00", periods=3, freq="5min", name="origin"))
    features.iloc[0,-1] = np.nan
    training = pd.DataFrame(np.arange(28.).reshape(4,7), columns=features.columns)
    x, _, scaler, processor = evaluation.direct.preprocess(training, features, True)
    joblib.dump(dict(processor=processor,scaler=scaler), root/"ridge.joblib")
    candidates,rows,best = [],[],{}
    for i,family in enumerate(evaluation.correction.MODELS):
        model = Ridge(alpha=i+1,random_state=2107)
        model.coef_ = np.zeros(x.shape[1])
        model.intercept_ = -5. if family=="ridge" else 4.
        model.coef_[0] = .25 if family=="ridge" else .1
        model.n_features_in_ = x.shape[1]
        name = "forest_05" if family=="forest" else family+"_00"
        config = dict(alpha=i+1)
        value = 1. if family=="forest" else float(i+2)
        row = dict(id=name,family=family,config=json.dumps(config),status="complete",error="",
                   daylight_n=2,all_day_n=3,daylight_mae_kw=value,daylight_rmse_kw=value,
                   daylight_bias_kw=0.,all_day_mae_kw=value)
        candidates.append(dict(id=name,family=family,config=config,effective_settings=model.get_params()))
        rows.append(row)
        best[family] = dict(trial=name,config=config,tuned_daylight_mae_kw=value)
        joblib.dump(dict(model=model,processor=processor,scaler=scaler,features=features.columns.tolist(),trial=row),
                    root/"hpo"/f"best_{family}.joblib")
    monkeypatch.setattr(evaluation.correction,"estimator",lambda family,config:Ridge())
    hpo = dict(plan=dict(candidates=candidates,counts={"validation":{"eligible":3}}),
               best_per_family=best,selected_trial="forest_05")
    selected,bundles = evaluation.selected_bundles(hpo,pd.DataFrame(rows),dict(daylight_n=2))
    plan = dict(protocol="stage2-four-family-correction-test-v1",split=evaluation.legacy.SPLIT,
                selected=selected,source_hashes={},disclosure="Previously explored test period")
    monkeypatch.setattr(evaluation,"frozen_artifacts",lambda:(plan,bundles))
    original = pd.DataFrame(dict(target_timestamp=features.index+pd.Timedelta(minutes=15),
        actual_target_power_kw=[0.,10.,20.],physics_forecast_kw=[2.,8.,21.],
        persistence_kw=features.origin_power_kw,forest_05=[-1.,12.,19.]),index=features.index)
    features["physics_forecast_kw"] = original.physics_forecast_kw
    for seed in evaluation.legacy.neural.SEEDS:
        original[f"classical_{seed}"] = original.actual_target_power_kw+(seed-2109)
    day = np.array([False,True,True])
    columns = ["physics_forecast_kw","persistence_kw","forest_05",*[f"classical_{s}" for s in evaluation.legacy.neural.SEEDS]]
    scores = evaluation.scores_with_r2(original,columns,day).drop(columns=["daylight_r2","all_day_r2"])
    original.to_parquet(evaluation.legacy.OUT/"predictions.parquet")
    scores.to_csv(evaluation.legacy.OUT/"metrics.csv")
    (evaluation.legacy.OUT/"settings.json").write_text(json.dumps(dict(protocol="stage2-frozen-test-v1",
        split=evaluation.legacy.SPLIT,source_hashes={})))
    direct = original[evaluation.KEYS+["persistence_kw"]].copy()
    direct["daylight"] = day
    for i,family in enumerate(evaluation.correction.MODELS):
        direct["direct_"+family] = direct.actual_target_power_kw+(i+1)
    for seed in evaluation.legacy.neural.SEEDS:
        direct[f"direct_classical_{seed}"] = direct.actual_target_power_kw+(seed-2108)
    names = ["physics_forecast_kw","persistence_kw",*["direct_"+f for f in evaluation.correction.MODELS],
             *[f"direct_classical_{s}" for s in evaluation.legacy.neural.SEEDS]]
    direct.to_parquet(evaluation.direct.OUT/"test/predictions.parquet")
    evaluation.scores_with_r2(direct,names,day).to_csv(evaluation.direct.OUT/"test/metrics.csv")
    (evaluation.direct.OUT/"test/settings.json").write_text(json.dumps(dict(protocol="stage2-direct-frozen-test-v1",
        training_plan=dict(source_hashes={},split=evaluation.legacy.SPLIT),source_hashes={})))
    (evaluation.direct.OUT/"ml/selection.json").write_text(json.dumps(dict(best_per_family={f:best[f]["trial"] for f in best})))
    table = original.copy()
    table["correction_kw"] = table.actual_target_power_kw-table.physics_forecast_kw
    solar = lambda times:pd.Series(np.where(day,1.,-1.),index=times)
    monkeypatch.setattr(evaluation,"solar_elevation",solar)
    monkeypatch.setattr(shared,"solar_elevation",solar)
    monkeypatch.setattr(evaluation.legacy,"load_testing_examples",lambda:(features,table,day,{}))
    return plan,bundles,original,features,table,day,hpo,pd.DataFrame(rows)


def test_winners_require_fitted_original_preprocessing_and_selected_configuration(artifacts):
    _,_,_,_,_,_,hpo,ledger = artifacts
    changed = copy.deepcopy(hpo)
    changed["best_per_family"]["ridge"]["trial"] = "ridge_other"
    with pytest.raises(ValueError,match="validation-selected"):
        evaluation.selected_bundles(changed,ledger,dict(daylight_n=2))
    path = evaluation.correction.OUT/"hpo/best_ridge.joblib"
    bundle = joblib.load(path)
    bundle["scaler"].mean_[0] += 1
    joblib.dump(bundle,path)
    with pytest.raises(ValueError,match="preprocessing"):
        evaluation.selected_bundles(hpo,ledger,dict(daylight_n=2))
    path.unlink()
    with pytest.raises(FileNotFoundError,match="restore this exact winning bundle"):
        evaluation.selected_bundles(hpo,ledger,dict(daylight_n=2))


def test_preparation_has_no_test_access_and_requires_explicit_gate(artifacts,monkeypatch):
    def forbidden(*args): raise AssertionError("No test access or prediction in preparation")
    monkeypatch.setattr(evaluation.legacy,"load_testing_examples",forbidden)
    monkeypatch.setattr(evaluation,"evaluate",forbidden)
    monkeypatch.setattr(sys,"argv",["correction_test","--evaluate"])
    with pytest.raises(SystemExit): evaluation.main()
    monkeypatch.setattr(sys,"argv",["correction_test","--prepare"])
    evaluation.main()
    assert [p.name for p in evaluation.OUT.iterdir()]==["settings.json"]


def test_signed_reconstruction_original_preprocessing_and_rf_reuse(artifacts,monkeypatch):
    _,bundles,original,features,_,day,_,_ = artifacts
    def forbidden(*args): raise AssertionError("RF must reuse saved predictions")
    monkeypatch.setattr(bundles["forest"]["model"],"predict",forbidden)
    prediction = evaluation.predictions_from_bundles(bundles,features,original,day)
    np.testing.assert_array_equal(prediction.correction_forest,original.forest_05)
    raw = bundles["ridge"]["processor"].transform(features)
    scaled = bundles["ridge"]["scaler"].transform(raw)
    np.testing.assert_allclose(prediction.predicted_correction_ridge_kw,scaled[:,0]*.25-5.)
    np.testing.assert_allclose(prediction.predicted_correction_histogram_kw,raw[:,0]*.1+4.)
    assert prediction.correction_ridge.iloc[0]<0
    assert prediction.predicted_correction_forest_kw.lt(0).any() and prediction.predicted_correction_forest_kw.gt(0).any()
    for family in bundles:
        np.testing.assert_allclose(prediction["correction_"+family],original.physics_forecast_kw+
                                   prediction[f"predicted_correction_{family}_kw"])


def test_synthetic_evaluation_and_saved_only_comparison_preserve_artifacts(artifacts,monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    _,_,original,features,table,day,_,_ = artifacts
    paths=list(evaluation.legacy.OUT.iterdir())+list((evaluation.direct.OUT/"test").iterdir())
    before={p:p.read_bytes() for p in paths}
    def forbidden(*args,**kwargs): raise AssertionError("No fitting during evaluation/reporting")
    monkeypatch.setattr(SimpleImputer,"fit",forbidden)
    monkeypatch.setattr(StandardScaler,"fit",forbidden)
    monkeypatch.setattr(sys,"argv",["correction_test","--prepare"])
    evaluation.main()
    wrong=table.copy();wrong.correction_kw*=-1
    monkeypatch.setattr(evaluation.legacy,"load_testing_examples",lambda:(features,wrong,day,{}))
    monkeypatch.setattr(sys,"argv",["correction_test","--evaluate","--allow-test"])
    with pytest.raises(AssertionError): evaluation.main()
    assert not (evaluation.OUT/"predictions.parquet").exists()
    monkeypatch.setattr(evaluation.legacy,"load_testing_examples",lambda:(features,table,day,{}))
    evaluation.main()
    predictions=pd.read_parquet(evaluation.OUT/"predictions.parquet")
    scores=pd.read_csv(evaluation.OUT/"metrics.csv",index_col=0)
    errors=predictions.correction_ridge.to_numpy()-original.actual_target_power_kw.to_numpy()
    assert scores.loc["correction_ridge","daylight_mae_kw"]==pytest.approx(np.abs(errors[day]).mean())
    assert scores.loc["correction_ridge","all_day_bias_kw"]==pytest.approx(errors.mean())
    with pytest.raises(ValueError,match="overwrite"): evaluation.main()
    monkeypatch.setattr(evaluation.legacy,"load_testing_examples",forbidden)
    monkeypatch.setattr(evaluation,"predictions_from_bundles",forbidden)
    monkeypatch.setattr(sys,"argv",["correction_test","--compare"])
    evaluation.main()
    comparison=pd.read_csv(evaluation.OUT/"comparison.csv")
    assert len(comparison)==14
    assert comparison.family.eq("classical_neural").sum()==4
    for family in evaluation.correction.MODELS:
        pair=comparison.loc[comparison.family.eq(family)].set_index("scheme")
        assert pair.loc["correction","daylight_mae_minus_direct_kw"]==pytest.approx(
            pair.loc["correction","daylight_mae_kw"]-pair.loc["direct","daylight_mae_kw"])
    means=comparison.loc[comparison.family.eq("classical_neural")&comparison.statistic.eq("mean")].set_index("scheme")
    assert means.loc["correction","daylight_mae_kw"]==pytest.approx(1.2)
    with pytest.raises(ValueError,match="overwrite"): evaluation.main()
    for path,content in before.items(): assert path.read_bytes()==content


def test_comparison_rejects_changed_keys_labels_and_missing_saved_artifacts(artifacts,monkeypatch):
    monkeypatch.setattr(sys,"argv",["correction_test","--prepare"])
    evaluation.main()
    monkeypatch.setattr(sys,"argv",["correction_test","--evaluate","--allow-test"])
    evaluation.main()
    path=evaluation.direct.OUT/"test/predictions.parquet"
    original=pd.read_parquet(path)
    changed=original.copy();changed.actual_target_power_kw+=1
    changed.to_parquet(path)
    with pytest.raises(AssertionError): evaluation.compare()
    changed=original.copy();changed.index=pd.DatetimeIndex([original.index[0]]*len(original))
    changed.to_parquet(path)
    with pytest.raises(ValueError,match="origin/target"): evaluation.compare()
    path.unlink()
    with pytest.raises(FileNotFoundError,match="Missing saved evaluation"): evaluation.compare()
    assert not (evaluation.OUT/"comparison.csv").exists()
