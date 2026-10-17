import copy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import evaluate_stage2_quantum as evaluation


@pytest.fixture
def runs(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import StandardScaler
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluation, "OUT", tmp_path/"quantum_test")
    monkeypatch.setattr(evaluation.native, "OUT", tmp_path/"quantum_runs")
    monkeypatch.setattr(evaluation.native.neural, "OUT", tmp_path/"classical")
    monkeypatch.setattr(evaluation.baseline, "OUT", tmp_path/"ml_test")
    monkeypatch.setattr(evaluation.baseline, "CORRECTION", tmp_path/"correction")
    monkeypatch.setattr(evaluation.baseline, "WEATHER", tmp_path/"weather")
    for path in (evaluation.native.OUT, evaluation.native.neural.OUT, evaluation.baseline.OUT,
                 evaluation.baseline.CORRECTION, evaluation.baseline.WEATHER, Path("scripts")):
        path.mkdir()
    sources = {}
    for name in ("train_stage2_neural.py", "train_stage2_quantum_cpu.py", "train_stage2_quantum_gpu.py"):
        path = Path("scripts")/name
        path.write_text("synthetic source identity")
        sources[path.as_posix()] = evaluation.digest(path)
    training = pd.DataFrame(np.arange(28.).reshape(4, 7), columns=evaluation.FEATURES)
    training.iloc[0, -1] = np.nan
    processor = FeatureUnion([("median", SimpleImputer(keep_empty_features=True)),
                              ("missing", MissingIndicator(features="all"))])
    transformed = processor.fit_transform(training)
    scaler = StandardScaler().fit(transformed)
    ridge = evaluation.baseline.CORRECTION/"ridge.joblib"
    joblib.dump(dict(processor=processor, scaler=scaler), ridge)
    preprocessing = dict(processor=processor, scaler=scaler, features=evaluation.FEATURES,
                         continuous_columns=7, target_scale_kw=2.)
    joblib.dump(preprocessing, evaluation.native.OUT/"preprocessing.joblib")
    reference = dict(features=evaluation.FEATURES, dimension=14, split=evaluation.baseline.SPLIT,
        counts={}, daylight_n=2, target_scale_kw=2., target_scale_ddof=0, target_centered=False,
        preprocessing="synthetic fitted training data", frame_fingerprints={}, versions={},
        preprocessing_sha256=evaluation.digest(ridge), source_commit="synthetic",
        source_hashes={"scripts/train_stage2_neural.py": sources["scripts/train_stage2_neural.py"]})
    reference_path = evaluation.native.neural.OUT/"settings.json"
    reference_path.write_text(json.dumps(reference))
    spec = dict(reference, architecture="stage2-native-batched-cpu-v1", settings=evaluation.native.SETTINGS,
                seeds=evaluation.native.SEEDS, source_hashes=sources, classical_reference_commit="synthetic",
                classical_reference_sha256=evaluation.digest(reference_path))
    (evaluation.native.OUT/"settings.json").write_text(json.dumps(spec))
    for seed in evaluation.native.SEEDS:
        directory = evaluation.native.OUT/"quantum"/f"seed_{seed}"
        directory.mkdir(parents=True)
        run_spec = dict(spec, model="quantum", seed=seed)
        (directory/"settings.json").write_text(json.dumps(run_spec))
        model = evaluation.native.make_model(14, seed, "default.qubit", batched=True)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(seed-2106)
        history = [dict(epoch=n, daylight_mae_kw=2. if n==1 else 3.) for n in range(1,12)]
        state = dict(specification=run_spec, model=model.state_dict(), best_mae=2.)
        torch.save(dict(state, epoch=1, stale=0, history=history[:1]), directory/"best.pt")
        torch.save(dict(state, epoch=11, stale=10, history=history), directory/"last.pt")
        (directory/"metrics.json").write_text(json.dumps(dict(stopping_epoch=11, best_epoch=1, daylight_mae_kw=2.)))
    origins = pd.date_range("2023-03-01 12:00", periods=3, freq="5min", name="origin")
    saved = pd.DataFrame(dict(target_timestamp=origins+pd.Timedelta(minutes=15),
        actual_target_power_kw=[0.,10.,20.], physics_forecast_kw=[1.,9.,22.], persistence_kw=[3.,4.,5.]), index=origins)
    saved.to_parquet(evaluation.baseline.OUT/"predictions.parquet")
    pd.DataFrame(dict(all_day_n=[3], daylight_n=[2])).to_csv(evaluation.baseline.OUT/"metrics.csv", index=False)
    (evaluation.baseline.OUT/"settings.json").write_text(json.dumps(dict(neural_reference=reference,
        protocol="stage2-frozen-test-v1", split=evaluation.baseline.SPLIT,
        source_hashes={str(ridge): evaluation.digest(ridge)}, disclosure="Previously explored test period")))
    for name in ("forecasts_testing.parquet", "physics_testing.parquet", "testing_settings.json", "physics_testing_settings.json"):
        (evaluation.baseline.WEATHER/name).write_text("synthetic input identity")
    features = training.iloc[:3].copy()
    features.index = origins
    features["origin_power_kw"] = saved.persistence_kw
    features["physics_forecast_kw"] = saved.physics_forecast_kw
    day = np.array([False,True,True])
    monkeypatch.setattr(evaluation.native.neural, "dependencies", lambda: {})
    monkeypatch.setattr(evaluation, "solar_elevation", lambda times: pd.Series([-1.,1.,2.], index=times))
    monkeypatch.setattr(evaluation.baseline, "load_testing_examples", lambda: (features, saved, day, {}))
    yield spec, saved, features, day
    torch.set_default_dtype(previous)


def test_completed_checkpoints_and_preprocessing_compatibility(runs):
    spec, _, _, _ = runs
    plan = evaluation.frozen_artifacts()
    assert plan["quantum_reference"] == spec
    settings = evaluation.native.OUT/"settings.json"
    for key, value in (("architecture", "other"), ("source_hashes", {}), ("target_scale_kw", 3.)):
        settings.write_text(json.dumps(dict(spec, **{key: value})))
        with pytest.raises(ValueError):
            evaluation.frozen_artifacts()
    settings.write_text(json.dumps(spec))
    changed = copy.deepcopy(spec)
    changed["settings"]["batch_size"] = 128
    with pytest.raises(ValueError, match="settings"):
        evaluation.completed_model(2107, changed)
    path = evaluation.native.OUT/"preprocessing.joblib"
    preprocessing = joblib.load(path)
    preprocessing["scaler"].mean_[0] += 1
    joblib.dump(preprocessing, path)
    with pytest.raises(ValueError, match="preprocessing"):
        evaluation.frozen_artifacts()


def test_checkpoint_rejects_wrong_state_and_unfinished_run(runs):
    import torch
    spec, _, _, _ = runs
    directory = evaluation.native.OUT/"quantum/seed_2107"
    run_spec = dict(spec, model="quantum", seed=2107)
    state = evaluation.native.read_checkpoint(directory/"best.pt", run_spec)
    state["model"]["head.weight"] = state["model"]["head.weight"].float()
    torch.save(state, directory/"best.pt")
    with pytest.raises(ValueError, match="parameters"):
        evaluation.completed_model(2107, spec)
    state["model"]["head.weight"] = state["model"]["head.weight"].double()
    torch.save(state, directory/"best.pt")
    last = evaluation.native.read_checkpoint(directory/"last.pt", run_spec)
    last.update(epoch=2, stale=1, history=last["history"][:2])
    torch.save(last, directory/"last.pt")
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.completed_model(2107, spec)
    (directory/"best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="best.pt"):
        evaluation.frozen_artifacts()


def test_alignment_rejects_changed_keys_labels_features_and_daylight(runs):
    _, saved, features, day = runs
    scores = pd.DataFrame(dict(all_day_n=[3], daylight_n=[2]))
    evaluation.aligned_inputs(saved, features, saved.copy(), day, scores)
    changed = saved.copy()
    changed.iloc[0, changed.columns.get_loc("actual_target_power_kw")] = 1.
    with pytest.raises(AssertionError):
        evaluation.aligned_inputs(saved, features, changed, day, scores)
    with pytest.raises(ValueError, match="alignment"):
        evaluation.aligned_inputs(saved, features.iloc[::-1], saved, day, scores)
    repeated = saved.copy()
    repeated.index = pd.DatetimeIndex([saved.index[0]]*3)
    with pytest.raises(ValueError, match="keys"):
        evaluation.aligned_inputs(repeated, features, saved, day, scores)
    with pytest.raises(ValueError, match="daylight"):
        evaluation.aligned_inputs(saved, features, saved, ~day, scores)


def test_metrics_include_r2_bias_and_zeros():
    data = pd.DataFrame(dict(actual_target_power_kw=[0.,10.,20.], quantum=[1.,9.,22.]))
    result = evaluation.scores_with_r2(data, ["quantum"], np.array([False,True,True])).loc["quantum"]
    assert result.daylight_mae_kw == 1.5
    assert result.daylight_rmse_kw == pytest.approx(np.sqrt(2.5))
    assert result.daylight_bias_kw == .5
    assert result.all_day_bias_kw == pytest.approx(2/3)
    assert result.daylight_r2 == pytest.approx(.9)
    assert result.all_day_r2 == pytest.approx(.97)


def test_prepare_and_gate_never_score_then_synthetic_evaluation_preserves_results(runs, monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    spec, saved, _, _ = runs
    originals = {p: p.read_bytes() for p in evaluation.baseline.OUT.iterdir()}
    def forbidden(*args): raise AssertionError("No test access/scoring or fitting in preparation")
    monkeypatch.setattr(SimpleImputer, "fit", forbidden)
    monkeypatch.setattr(StandardScaler, "fit", forbidden)
    loader = evaluation.baseline.load_testing_examples
    monkeypatch.setattr(evaluation.baseline, "load_testing_examples", forbidden)
    monkeypatch.setattr(sys, "argv", ["quantum_test", "--evaluate"])
    with pytest.raises(SystemExit): evaluation.main()
    monkeypatch.setattr(sys, "argv", ["quantum_test", "--prepare"])
    evaluation.main()
    assert not (evaluation.OUT/"predictions.parquet").exists()
    monkeypatch.setattr(evaluation.baseline, "load_testing_examples", loader)
    monkeypatch.setattr(sys, "argv", ["quantum_test", "--evaluate", "--allow-test"])
    evaluation.main()
    result = pd.read_parquet(evaluation.OUT/"predictions.parquet")
    pd.testing.assert_frame_equal(result[evaluation.KEYS], saved[evaluation.KEYS], check_freq=False)
    for seed in evaluation.native.SEEDS:
        model = evaluation.completed_model(seed, spec)
        assert sum(p.numel() for p in model.parameters()) == 89
        np.testing.assert_allclose(result[f"quantum_{seed}_correction_kw"], 2.*(seed-2106))
        np.testing.assert_allclose(result[f"quantum_{seed}"],
                                   result.physics_forecast_kw+result[f"quantum_{seed}_correction_kw"])
    scores = pd.read_csv(evaluation.OUT/"metrics.csv", index_col=0)
    np.testing.assert_allclose(scores.loc["std", "daylight_mae_kw"], scores.iloc[:3].daylight_mae_kw.std(ddof=1))
    with pytest.raises(ValueError, match="overwrite"): evaluation.main()
    for path, content in originals.items(): assert path.read_bytes() == content
