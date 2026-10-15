import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from calculate_stage2_physics import complete_fleet_power, physics_labels


def forecasts():
    origin = pd.date_range("2019-06-01 12:00", periods=4, freq="5min", name="origin")
    return pd.DataFrame({"target_timestamp":origin+pd.Timedelta(minutes=15),
        "training_cutoff":pd.Timestamp("2019-01-01"), "forecast_poa":[0.,1400.,500.,np.nan],
        "forecast_ambient":25., "forecast_wind":1., "actual_poa":999.,
        "actual_ambient":999., "actual_wind":999.}, index=origin)


def test_complete_fleet_preserves_zero_and_rejects_missing_negative_qc():
    index = pd.date_range("2019-01-01", periods=5, freq="5min")
    ac = pd.DataFrame(0., index=index, columns=range(24))
    ac.iloc[1,0] = np.nan
    ac.iloc[2,0] = -1
    ac.iloc[4] = 2.
    result = complete_fleet_power(ac, index[[3]])
    assert result.iloc[0] == 0
    assert result.iloc[1:4].isna().all()
    assert result.iloc[4] == 48
    with pytest.raises(ValueError, match="24"):
        complete_fleet_power(ac.iloc[:,:23], [])


def test_exact_targets_units_signed_tare_corrections_and_missing_rows():
    f = forecasts()
    actual = pd.Series([0.,700.,1e6,123.], index=pd.to_datetime([
        "2019-06-01 12:15", "2019-06-01 12:20", "2019-06-01 12:24", "2019-06-01 12:30"]))
    r = physics_labels(f, actual)
    assert r.index.equals(f.index)
    assert r.physics_forecast_kw.iloc[0] == pytest.approx(-8.28*24/1000)
    assert r.physics_forecast_kw.iloc[1] == pytest.approx(720.)
    assert r.correction_kw.iloc[0] > 0
    assert r.correction_kw.iloc[1] == pytest.approx(-20.)
    assert r.validity.tolist() == ["labelled", "labelled", "actual_missing", "physics_failed"]
    assert r.correction_kw.iloc[2:].isna().all()
    valid = r.validity.eq("labelled")
    np.testing.assert_allclose(r.loc[valid,"actual_target_power_kw"],
        r.loc[valid,"physics_forecast_kw"]+r.loc[valid,"correction_kw"])
    bad = f.copy(); bad.iloc[0,bad.columns.get_loc("target_timestamp")] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError):
        physics_labels(bad, actual)


def test_measured_future_weather_and_actual_power_cannot_enter_physics():
    f = forecasts()
    before = f.copy(deep=True)
    first = physics_labels(f, pd.Series(0., index=pd.DatetimeIndex(f.target_timestamp)))
    changed = f.copy()
    changed[["actual_poa","actual_ambient","actual_wind"]] = -1e9
    second = physics_labels(changed, pd.Series(1e9, index=pd.DatetimeIndex(f.target_timestamp)))
    pd.testing.assert_series_equal(first.physics_forecast_kw, second.physics_forecast_kw)
    pd.testing.assert_frame_equal(f, before)
