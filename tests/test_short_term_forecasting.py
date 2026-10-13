import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_short_term_forecasting import window_mask, origin_features, paired_intervals, pairwise_comparisons


def test_windows_do_not_cross_split_or_bridge_missing_readings():
    idx = pd.date_range("2020-12-31 23:00", "2021-01-01 01:00", freq="5min")
    valid = pd.Series(True, index=idx)
    mask = pd.Series(window_mask(idx, valid, 15), index=idx)
    assert not mask.loc["2020-12-31 23:50"]
    assert not mask.loc["2021-01-01 00:15"]
    assert mask.loc["2021-01-01 00:30"]
    valid.loc["2021-01-01 00:20"] = False
    assert not pd.Series(window_mask(idx, valid, 15), index=idx).loc["2021-01-01 00:30"]


def test_future_measurements_cannot_change_origin_features():
    idx = pd.date_range("2022-06-01 10:00", periods=30, freq="5min")
    power = pd.Series(np.arange(30.), index=idx)
    env = pd.DataFrame({"ambient_temperature_c": 20., "wind_speed_m_s": 2.}, index=idx)
    old = origin_features(power, power*2, power*0, env, power*0+45)
    origin = idx[12]
    later = idx > origin
    changed = power.copy(); changed.loc[later] = 10000.
    changed_env = env.copy(); changed_env.loc[later] = 10000.
    new = origin_features(changed, changed*2, power*0, changed_env, power*0+45)
    pd.testing.assert_series_equal(old.loc[origin], new.loc[origin])
    assert old.loc[origin,"ac_kw_lag30"] == power.loc[origin-pd.Timedelta(minutes=30)]


def test_paired_blocks_keep_known_advantage():
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    frame = pd.DataFrame({"actual_kw": 20., "persistence": 10., "model": 15.}, index=idx)
    ci = paired_intervals(frame, "model")
    assert ci["mae_improvement_lower"] == 50
    assert ci["mae_improvement_upper"] == 50


def test_zero_baseline_error_has_no_relative_improvement():
    idx = pd.date_range("2022-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"actual_kw": 0., "persistence": 0., "model": 5.}, index=idx)
    ci = paired_intervals(frame, "model")
    assert np.isnan(ci["mae_improvement_lower"])
    assert ci["mae_saving_upper"] == -5
    assert ci["undefined_relative_bootstrap_fraction"] == 1


def test_pairwise_comparison_uses_named_reference():
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    frame = pd.DataFrame({"split": "evaluation", "horizon_minutes": 5, "actual_kw": 20.,
                          "persistence": 10., "ridge": 15., "boosting": 18.}, index=idx)
    result = pairwise_comparisons(frame)
    comparison = result.loc[result.reference.eq("ridge")].iloc[0]
    assert comparison.mae_improvement_pct == 60
    assert comparison.mae_improvement_lower == 60
    assert frame.persistence.eq(10).all()
