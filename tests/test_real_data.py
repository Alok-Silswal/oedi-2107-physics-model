"""Prepared-data integration preserves wall clock, missingness and measurements."""

import numpy as np
import pandas as pd
import pytest

from oedi2107 import SystemConfig, run_dc_foundation
from oedi2107.real_data import discover_channels, prepared_dc_baseline, measured_dc_for_inverter, FLAGS
from oedi2107.validation import validate_dc


@pytest.fixture
def prepared():
    result = pd.DataFrame({
        "measured_on": ["2019-11-03 01:00:00", "2019-11-03 01:05:00", "2019-11-03 01:10:00"],
        "poa_w_m2": [600., np.nan, 1400.],
        "ambient_temperature_c": [25., 25., 25.], "wind_speed_m_s": [1., 1., 1.],
    })
    for flag in FLAGS:
        result[flag] = True
    result["model_inputs_available"] = [True, False, True]
    return result


def test_baseline_preserves_flags_timestamps_and_missing_rows(prepared):
    before = prepared.copy(deep=True)
    result = prepared_dc_baseline(prepared)
    pd.testing.assert_frame_equal(prepared, before)
    assert result.index.tz is None
    assert result.measured_on.tolist() == prepared.measured_on.tolist()
    assert len(result) == len(prepared)
    assert result.dc_model_computed.tolist() == [True, False, True]
    assert result.expected_dc_power_ideal.iloc[1:2].isna().all()
    assert result.poa_w_m2.iloc[2] == 1400.
    assert result.dst_fall_ambiguous.all() and result.electrical_qc_flag.all()
    assert "expected_ac_power" not in result and "dc_loss_factor" not in result
    assert SystemConfig().dc_losses.k_dc is None
    valid = prepared.loc[[0, 2], ["poa_w_m2", "ambient_temperature_c", "wind_speed_m_s"]].rename(
        columns={"poa_w_m2": "poa", "ambient_temperature_c": "temp_air", "wind_speed_m_s": "wind_speed"})
    valid.index = result.index[[0, 2]]
    nominal = run_dc_foundation(valid, timestamp_basis="pacific_wall_clock")
    np.testing.assert_allclose(result.expected_dc_power_ideal.iloc[[0, 2]], nominal.expected_dc_power)


def test_measured_dc_units_and_missing_inv05(prepared):
    baseline = prepared_dc_baseline(prepared)
    channel = {"inverter_id": "inv_05", "dc_voltage": "v", "dc_current": "i", "ac_power": "unused"}
    readings = pd.DataFrame({"measured_on": prepared.measured_on, "v": [np.nan] * 3, "i": [10., 20., 30.]})
    result = measured_dc_for_inverter(baseline, readings, channel)
    assert result.measured_dc_voltage.isna().all() and result.measured_dc_power.isna().all()
    readings["v"] = [600., np.inf, 700.]
    result = measured_dc_for_inverter(baseline, readings, {**channel, "inverter_id": "inv_01"})
    assert result.measured_dc_power.iloc[0] == 6000.
    assert np.isnan(result.measured_dc_power.iloc[1])
    assert result.measured_dc_power.iloc[2] == 21000.
    assert np.isinf(result.measured_dc_voltage.iloc[1])


def test_invalid_weather_is_identified_not_cleaned(prepared):
    prepared.loc[0, "wind_speed_m_s"] = -1.
    result = prepared_dc_baseline(prepared)
    assert result.wind_speed_m_s.iloc[0] == -1.
    assert result.dc_model_status.iloc[0] == "unsupported negative inputs"
    assert np.isnan(result.expected_dc_power_ideal.iloc[0])


def test_wall_clock_validation_requires_explicit_opt_in(prepared):
    result = prepared_dc_baseline(prepared)
    expected = result.filter(regex="expected_dc_").rename(columns=lambda c: c.removesuffix("_ideal"))
    expected["inverter_id"] = "inv_01"
    measured = expected.rename(columns=lambda c: c.replace("expected_", "measured_"))
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_dc(expected, measured)
    validation = validate_dc(expected, measured, timestamp_basis="pacific_wall_clock")
    assert (validation.metrics.valid_paired_samples == 2).all()
    assert (validation.metrics.mae == 0).all()
    assert validation.table.timestamp.dt.tz is None


def test_actual_channel_discovery_rejects_duplicates():
    columns = [f"inv_{i:02d}_{q}_inv_{i * 10 + j}" for i in range(1, 25)
               for j, q in enumerate(("dc_voltage", "dc_current", "ac_power"))]
    assert len(discover_channels(columns)) == 24
    with pytest.raises(ValueError, match="Multiple"):
        discover_channels(columns + ["inv_01_dc_voltage_inv_99999"])
