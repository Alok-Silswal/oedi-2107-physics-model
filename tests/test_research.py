import numpy as np
import pandas as pd

from oedi2107.research import research_baseline, meter_intervals, complete_power_scales
from oedi2107.real_data import FLAGS
from oedi2107.config import SystemConfig


def test_frozen_path_preserves_missing_rows_and_bypasses_unset_losses():
    prepared = pd.DataFrame({
        "measured_on": ["2022-06-01 00:00:00", "2022-06-01 12:00:00", "2022-06-01 12:05:00"],
        "poa_w_m2": [0., 1400., np.nan], "ambient_temperature_c": 25., "wind_speed_m_s": 1.,
    })
    for flag in FLAGS:
        prepared[flag] = False
    prepared["model_inputs_available"] = [True, True, False]
    before = prepared.copy(deep=True)
    result = research_baseline(prepared)
    assert result.expected_ac_power.iloc[0] == -8.28
    assert result.expected_ac_power.iloc[1] == 30000.
    assert result.expected_ac_power_cec.iloc[1] == 27600.
    assert result.model_available.tolist() == [True, True, False]
    assert np.isnan(result.expected_dc_power.iloc[2])
    np.testing.assert_allclose(result.expected_dc_power, result.expected_dc_power_ideal, equal_nan=True)
    assert "dc_loss_factor" not in result
    assert SystemConfig().dc_losses.k_dc is None
    pd.testing.assert_frame_equal(prepared, before)


def test_start_interval_alignment_requires_three_samples_and_preserves_tare():
    index = pd.date_range("2022-01-01", periods=6, freq="5min", name="timestamp")
    plant = pd.DataFrame({"expected_plant_ac_power": [-198.72, 3000., 6000., 9000., np.nan, 9000.]}, index=index)
    meter = pd.DataFrame({"meter_ac_power_kw": [3., 9.]}, index=index[[0, 3]])
    before = meter.copy(deep=True)
    table = meter_intervals(plant, meter, index[[3]])
    assert table.expected_plant_ac_power_kw.iloc[0] == (-198.72 + 3000 + 6000) / 3000
    assert np.isnan(table.expected_plant_ac_power_kw.iloc[1])
    assert table.model_sample_count.tolist() == [3, 2]
    assert table.meter_qc_clear.tolist() == [True, False]
    pd.testing.assert_frame_equal(meter, before)


def test_hourly_power_and_energy_require_complete_intervals():
    index = pd.date_range("2022-01-01", periods=8, freq="15min", name="timestamp")
    table = pd.DataFrame({"expected_plant_ac_power_kw": 100., "meter_ac_power_kw": 90., "meter_qc_clear": True}, index=index)
    table.loc[index[4], "expected_plant_ac_power_kw"] = np.nan
    result = complete_power_scales(table, "h")
    assert result.expected_plant_ac_power_kw.iloc[0] == 100.
    assert result.expected_energy_kwh.iloc[0] == 100.
    assert result.measured_energy_kwh.iloc[0] == 90.
    assert result.valid_interval_count.tolist() == [4, 3]
    assert np.isnan(result.expected_energy_kwh.iloc[1])
    assert result.paired_expected_average_kw.iloc[1] == 100.
    assert result.paired_measured_average_kw.iloc[1] == 90.
    assert result.observed_expected_energy_kwh.iloc[1] == 75.
    assert result.observed_measured_energy_kwh.iloc[1] == 67.5
    assert result.paired_coverage_fraction.iloc[1] == .75
