import copy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import compare_stage2_direct as direct
import evaluate_stage2_quantum as shared_evaluation


class ConstantEstimator:
    calls = []

    def __init__(self, settings):
        self.settings = settings.copy()

    def set_params(self, **settings):
        self.settings.update(settings)
        return self

    def get_params(self):
        return self.settings.copy()

    def fit(self, x, y):
        self.calls.append((x.copy(), y.copy()))
        self.value = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


@pytest.fixture
def small(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    previous = torch.get_default_dtype()
    monkeypatch.setattr(direct, "OUT", tmp_path/"direct")
    direct.OUT.mkdir()
    rng = np.random.default_rng(123)
    features, tables = {}, {}
    for split, size, start in (("training", 17, "2020-03-01 12:00"), ("validation", 7, "2022-03-01 12:00")):
        index = pd.date_range(start, periods=size, freq="5min", name="origin")
        features[split] = pd.DataFrame(rng.normal(size=(size,len(direct.FEATURES))), index=index, columns=direct.FEATURES)
        features[split].iloc[0,-1] = np.nan
        tables[split] = pd.DataFrame(dict(target_timestamp=index+pd.Timedelta(minutes=15),
            actual_target_power_kw=np.arange(size, dtype=float)*10, physics_forecast_kw=1e6), index=index)
    candidates = direct.correction.search_candidates()
    plan = dict(protocol="synthetic-direct-v1", features=direct.FEATURES, source_hashes={}, source_commit="synthetic",
        counts={name:{"eligible":len(frame)} for name,frame in features.items()}, seeds=direct.SEEDS,
        settings={k:direct.neural.SETTINGS[k] for k in direct.TRAINING_KEYS},
        target_scale_kw=float(tables["training"].actual_target_power_kw.std(ddof=0)),
        daylight_n=4, candidates=[next(c for c in candidates if c["family"]==n) for n in direct.correction.MODELS],
        versions=direct.neural.dependencies() | {n:direct.metadata.version(n) for n in ("xgboost","threadpoolctl")},
        disclosure="Test period previously explored")
    plan["settings"]["epochs"] = 3
    day = np.array([False,True,True,False,True,True,False])
    yield features,tables,day,plan
    torch.set_default_dtype(previous)


def test_physics_excluded_before_imputation_training_only_power_scaling_and_architecture(small):
    import torch
    features,tables,_,plan = small
    contaminated = features["training"].assign(physics_forecast_kw=1e9, physics_indicator=1, actual_future_power=1e9)
    pd.testing.assert_frame_equal(direct.direct_features(contaminated), features["training"])
    assert len(direct.FEATURES)==6 and all("physics" not in n for n in direct.FEATURES)
    preprocessing = direct.preprocessing_for(features,plan,fit=True)
    assert preprocessing["dimension"]==2*len(direct.FEATURES)
    assert not any("physics" in n for n in preprocessing["processor"].get_feature_names_out())
    assert preprocessing["target_scale_kw"]==tables["training"].actual_target_power_kw.std(ddof=0)
    assert preprocessing["target_scale_kw"]!=tables["training"].physics_forecast_kw.std(ddof=0)
    before = joblib.hash(preprocessing)
    changed = features["validation"].copy()*1e9
    result = direct.transformed(preprocessing,changed,"neural")
    assert result[0,-1]==1 and np.isfinite(result).all()
    assert joblib.hash(preprocessing)==before
    raw = preprocessing["processor"].transform(features["training"])
    np.testing.assert_allclose(preprocessing["scaler"].mean_,raw.mean(axis=0))
    with pytest.raises(ValueError,match="exclude physics"):
        direct.transformed(preprocessing,contaminated,"neural")
    np.testing.assert_array_equal(direct.direct_power(np.array([-1.,0.,1.]),3.),[-3.,0.,3.])
    torch.set_default_dtype(torch.float64)
    model = direct.make_neural(preprocessing["dimension"],2107)
    torch.set_default_dtype(torch.float32)
    reference = direct.neural.make_model("classical",preprocessing["dimension"],2107)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)
    assert direct.neural.gradient_check(model,preprocessing["dimension"])==4*preprocessing["dimension"]+29


def test_exact_correction_candidate_budget_without_reusing_fitted_models():
    saved = json.loads((direct.correction.OUT/"hpo/settings.json").read_text())["plan"]["candidates"]
    candidates = direct.correction.search_candidates()
    assert direct.correction.canonical(candidates)==direct.correction.canonical(saved)
    assert {n:sum(c["family"]==n for c in candidates) for n in direct.correction.MODELS}==dict(ridge=7,forest=12,histogram=12,xgboost=12)


def test_preparation_never_fits_or_reads_test_data(small,monkeypatch):
    features,tables,day,plan = small
    def forbidden(*args,**kwargs): raise AssertionError("No fitting/inference/test access in preparation")
    monkeypatch.setattr(direct,"development_data",lambda:(features,tables,day,plan))
    for name in ("preprocess","train_ml","train_neural","preprocessing_for"):
        monkeypatch.setattr(direct,name,forbidden)
    monkeypatch.setattr(direct.evaluation,"load_testing_examples",forbidden)
    monkeypatch.setattr(sys,"argv",["direct","--prepare"])
    direct.main()
    assert [p.name for p in direct.OUT.iterdir()]==["settings.json"]
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys,"argv",["direct","--train-neural","--seed","2112"])
        direct.main()


def test_ml_full_rows_direct_labels_and_recovery_before_trial_ledger(small,monkeypatch):
    features,tables,day,plan = small
    ConstantEstimator.calls = []
    by_family = {c["family"]:c for c in plan["candidates"]}
    monkeypatch.setattr(direct,"estimator",lambda family,config:ConstantEstimator(by_family[family]["effective_settings"]))
    save = direct.correction.save_trials
    def interrupted(*args): raise KeyboardInterrupt("After saving the first winning model")
    monkeypatch.setattr(direct.correction,"save_trials",interrupted)
    with pytest.raises(KeyboardInterrupt): direct.train_ml(features,tables,day,plan,False)
    monkeypatch.setattr(direct.correction,"save_trials",save)
    direct.train_ml(features,tables,day,plan,True)
    bundles = direct.completed_ml(plan)
    assert set(bundles)==set(direct.correction.MODELS)
    assert len(ConstantEstimator.calls)==len(plan["candidates"])
    for x,y in ConstantEstimator.calls:
        assert x.shape==(len(features["training"]),2*len(direct.FEATURES))
        np.testing.assert_array_equal(y,tables["training"].actual_target_power_kw)
    saved = pd.read_parquet(direct.OUT/"ml/validation_predictions.parquet")
    for family in bundles:
        np.testing.assert_allclose(saved[family],tables["training"].actual_target_power_kw.mean())
    direct.train_ml(features,tables,day,plan,True)
    assert len(ConstantEstimator.calls)==len(plan["candidates"])
    preprocessing = joblib.load(direct.OUT/"preprocessing.joblib")
    preprocessing["target_scale_kw"] *= 2
    joblib.dump(preprocessing,direct.OUT/"preprocessing.joblib")
    with pytest.raises(ValueError,match="bundle"):
        direct.completed_ml(plan)


def test_neural_resume_restores_identical_updates_best_model_and_predictions(small,monkeypatch,tmp_path):
    import torch
    features,tables,day,plan = small
    save = direct.neural.checkpoint
    uninterrupted = direct.OUT
    direct.train_neural(2107,False,features,tables,day,plan)
    resumed = tmp_path/"resumed"
    resumed.mkdir()
    monkeypatch.setattr(direct,"OUT",resumed)
    def interrupt(path,state):
        save(path,state)
        if path.name=="last.pt" and state["epoch"]==1: raise KeyboardInterrupt("Atomic epoch checkpoint")
    monkeypatch.setattr(direct.neural,"checkpoint",interrupt)
    with pytest.raises(KeyboardInterrupt): direct.train_neural(2107,False,features,tables,day,plan)
    monkeypatch.setattr(direct.neural,"checkpoint",save)
    direct.train_neural(2107,True,features,tables,day,plan)
    preprocessing = direct.preprocessing_for(features,plan)
    spec = direct.run_specification(plan,preprocessing,2107)
    for name in ("best.pt","last.pt"):
        first = direct.read_checkpoint(uninterrupted/"neural/seed_2107"/name,spec,4*preprocessing["dimension"]+29)
        second = direct.read_checkpoint(resumed/"neural/seed_2107"/name,spec,4*preprocessing["dimension"]+29)
        for key in first["model"]: torch.testing.assert_close(first["model"][key],second["model"][key],rtol=0,atol=0)
        torch.testing.assert_close(first["shuffle_rng"],second["shuffle_rng"],rtol=0,atol=0)
        assert first["epoch"]==second["epoch"]
        assert [{k:v for k,v in r.items() if k!="seconds"} for r in first["history"]]==[
            {k:v for k,v in r.items() if k!="seconds"} for r in second["history"]]
    first = pd.read_parquet(uninterrupted/"neural/seed_2107/validation_predictions.parquet")
    second = pd.read_parquet(resumed/"neural/seed_2107/validation_predictions.parquet")
    pd.testing.assert_frame_equal(first,second)
    model = direct.completed_neural(2107,plan,preprocessing)
    expected = direct.neural.inference(model,direct.transformed(preprocessing,features["validation"],"neural"))*plan["target_scale_kw"]
    np.testing.assert_allclose(second.direct_power_kw,expected)
    before = (resumed/"neural/seed_2107/metrics.json").read_bytes()
    direct.train_neural(2107,True,features,tables,day,plan)
    assert (resumed/"neural/seed_2107/metrics.json").read_bytes()==before
    changed = copy.deepcopy(plan)
    changed["settings"]["batch_size"] = 256
    with pytest.raises(ValueError,match="Incompatible"):
        direct.train_neural(2107,True,features,tables,day,changed)


def test_failed_ml_trials_are_reported_without_replacement_or_automatic_retry(small,monkeypatch):
    features,tables,day,plan = small
    calls = []
    by_family = {c["family"]:c for c in plan["candidates"]}
    def construct(family,config):
        model = ConstantEstimator(by_family[family]["effective_settings"])
        if family=="ridge":
            def fail(*args):
                calls.append(family)
                raise RuntimeError("Synthetic estimator failure")
            model.fit = fail
        return model
    monkeypatch.setattr(direct,"estimator",construct)
    for resume in (False,True):
        with pytest.raises(ValueError,match="Failed direct trials retained"):
            direct.train_ml(features,tables,day,plan,resume)
    rows = pd.read_csv(direct.OUT/"ml/trials.csv")
    assert len(rows)==len(plan["candidates"]) and rows.status.eq("failed").sum()==1
    assert calls==["ridge"]
    with pytest.raises(ValueError,match="finish successfully"):
        direct.completed_ml(plan)


def test_test_readiness_requires_all_five_seeds_without_creating_run_settings(small,monkeypatch):
    features,_,_,plan = small
    preprocessing = direct.preprocessing_for(features,plan,fit=True)
    monkeypatch.setattr(direct,"completed_ml",lambda plan:{})
    checked = []
    checker = direct.completed_neural
    def completed(seed,*args):
        checked.append(seed)
        if seed==2111: raise FileNotFoundError("seed_2111/best.pt")
    monkeypatch.setattr(direct,"completed_neural",completed)
    with pytest.raises(FileNotFoundError,match="2111"): direct.test_plan(plan)
    assert checked==[2107,2108,2109,2110,2111]
    assert not (direct.OUT/"neural/seed_2111").exists()
    # The real completion checker must also remain read-only for a missing run.
    with pytest.raises(FileNotFoundError): checker(2111,plan,preprocessing)
    assert not (direct.OUT/"neural").exists()


def test_gated_test_alignment_direct_predictions_and_preservation(small,monkeypatch,tmp_path):
    features,tables,day,plan = small
    preprocessing = direct.preprocessing_for(features,plan,fit=True)
    baseline = tmp_path/"existing_test"
    baseline.mkdir()
    monkeypatch.setattr(direct.evaluation,"OUT",baseline)
    table = tables["validation"].copy()
    full = features["validation"].copy()
    full["physics_forecast_kw"] = table.physics_forecast_kw
    full = full[direct.correction.FEATURES]
    saved = table.copy()
    saved["persistence_kw"] = full.origin_power_kw
    saved.to_parquet(baseline/"predictions.parquet")
    pd.DataFrame(dict(all_day_n=[len(table)],daylight_n=[day.sum()])).to_csv(baseline/"metrics.csv",index=False)
    before = {p:p.read_bytes() for p in baseline.iterdir()}
    frozen = dict(training_plan=plan,source_hashes={},disclosure=plan["disclosure"])
    monkeypatch.setattr(direct,"test_plan",lambda plan:frozen)
    def forbidden(*args): raise AssertionError("Test access is gated")
    monkeypatch.setattr(direct.evaluation,"load_testing_examples",forbidden)
    (direct.OUT/"settings.json").write_text(json.dumps(plan))
    monkeypatch.setattr(sys,"argv",["direct","--evaluate"])
    with pytest.raises(SystemExit): direct.main()
    monkeypatch.setattr(sys,"argv",["direct","--prepare-test"])
    direct.main()
    monkeypatch.setattr(direct.evaluation,"load_testing_examples",lambda:(full,table,day,{}))
    monkeypatch.setattr(shared_evaluation,"solar_elevation",lambda times:pd.Series(np.where(day,1.,-1.),index=times))
    bundles = {}
    for family in direct.correction.MODELS:
        model = ConstantEstimator({})
        model.value = 7.
        bundles[family] = dict(model=model)
    monkeypatch.setattr(direct,"completed_ml",lambda plan:bundles)
    def neural_model(seed,*args):
        import torch
        model = direct.make_neural(preprocessing["dimension"],seed)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(seed-2106)
        return model
    monkeypatch.setattr(direct,"completed_neural",neural_model)
    changed = table.copy()
    changed.iloc[0,changed.columns.get_loc("actual_target_power_kw")] += 1
    monkeypatch.setattr(direct.evaluation,"load_testing_examples",lambda:(full,changed,day,{}))
    monkeypatch.setattr(sys,"argv",["direct","--evaluate","--allow-test"])
    with pytest.raises(AssertionError): direct.main()
    assert not (direct.OUT/"test/predictions.parquet").exists()
    monkeypatch.setattr(direct.evaluation,"load_testing_examples",lambda:(full,table,day,{}))
    direct.main()
    predictions = pd.read_parquet(direct.OUT/"test/predictions.parquet")
    for family in bundles: np.testing.assert_allclose(predictions["direct_"+family],7.)
    for seed in direct.SEEDS:
        np.testing.assert_allclose(predictions[f"direct_classical_{seed}"],(seed-2106)*plan["target_scale_kw"])
    scores = pd.read_csv(direct.OUT/"test/metrics.csv",index_col=0)
    names = [f"direct_classical_{seed}" for seed in direct.SEEDS]
    assert scores.loc[names,"all_day_n"].eq(len(table)).all()
    assert scores.loc[names,"daylight_n"].eq(day.sum()).all()
    assert scores.loc["direct_classical_sample_std","daylight_mae_kw"]==pytest.approx(scores.loc[names,"daylight_mae_kw"].std(ddof=1))
    with pytest.raises(ValueError,match="overwrite"): direct.main()
    for path,content in before.items(): assert path.read_bytes()==content
