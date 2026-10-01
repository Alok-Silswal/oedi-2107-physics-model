"""Physical sanity checks using pvlib's actual bundled CEC parameters."""

import logging
import numpy as np
import pandas as pd
import pytest
from oedi2107 import SystemConfig, FaimanConfig, run_dc_foundation
from oedi2107.module import load_cec_module, calculate_module_dc
from oedi2107.inputs import validate_measurements
from oedi2107.thermal import estimate_cell_temperature


@pytest.fixture(scope="module")
def parameters():
    return load_cec_module(SystemConfig().cec_module_entry)


def module_at(parameters, irradiance, temperature):
    index = pd.date_range("2026-01-01", periods=len(irradiance), freq="h", tz="UTC")
    return calculate_module_dc(pd.Series(irradiance, index=index), pd.Series(temperature, index=index), parameters)


def test_dark_module_has_zero_power(parameters):
    result = module_at(parameters, [0, 0], [25, -10])
    assert (result[["Vmp", "Imp", "Pmp", "Voc", "Isc"]] == 0).all().all()


def test_more_irradiance_increases_current_and_power(parameters):
    result = module_at(parameters, [200, 600, 1000], [25, 25, 25])
    assert (np.diff(result.Imp) > 0).all()
    assert (np.diff(result.Pmp) > 0).all()


def test_hotter_cells_reduce_high_irradiance_power(parameters):
    result = module_at(parameters, [1000, 1000, 1000], [25, 45, 65])
    assert (np.diff(result.Pmp) < 0).all()


def test_stc_agrees_with_cec_reference(parameters):
    result = module_at(parameters, [1000], [25]).iloc[0]
    assert result.Pmp == pytest.approx(float(parameters.V_mp_ref) * float(parameters.I_mp_ref), rel=0.005)
    assert result.Voc == pytest.approx(float(parameters.V_oc_ref), rel=0.005)


def test_pipeline_array_scaling_and_identity():
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame({"poa": [0, 0.01, 500, 1000], "temp_air": 25., "wind_speed": 1.}, index=index)
    result = run_dc_foundation(data)
    np.testing.assert_allclose(result.expected_dc_voltage, 20 * result.Vmp)
    np.testing.assert_allclose(result.expected_dc_current, 6 * result.Imp)
    np.testing.assert_allclose(result.expected_dc_power, 120 * result.Pmp)
    np.testing.assert_allclose(result.expected_dc_power, result.expected_dc_voltage * result.expected_dc_current, rtol=1e-10, atol=1e-9)
    assert result.expected_dc_power.iloc[0] == 0
    pd.testing.assert_series_equal(result.poa, data.poa, check_dtype=False)
    assert result.index.equals(index)


def test_faiman_equation_and_dark_temperature():
    config = FaimanConfig()
    assert estimate_cell_temperature(1000., 25., 1., config) == pytest.approx(25 + 1000 / (25 + 6.84))
    assert estimate_cell_temperature(0., 12., 0., config) == 12


def test_exact_entry_logged(caplog):
    with caplog.at_level(logging.INFO):
        row = load_cec_module(SystemConfig().cec_module_entry)
    assert row.name in caplog.text
    assert "I_L_ref" in caplog.text


def test_missing_entry_fails_without_substitution():
    with pytest.raises(ValueError, match="review candidates"):
        load_cec_module("not-a-module")


@pytest.mark.parametrize("column,value", [("poa", -1), ("wind_speed", -1), ("temp_air", np.nan)])
def test_invalid_measurements_rejected(column, value):
    data = pd.DataFrame({"poa": [1000.], "temp_air": [25.], "wind_speed": [1.]}, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    data.loc[data.index[0], column] = value
    with pytest.raises(ValueError):
        validate_measurements(data)


def test_unknown_timestamp_timezone_rejected():
    data = pd.DataFrame({"poa": [1000.], "temp_air": [25.], "wind_speed": [1.]}, index=pd.date_range("2026-01-01", periods=1))
    with pytest.raises(ValueError, match="timezone"):
        validate_measurements(data)


def test_inferred_module_count():
    config = SystemConfig()
    assert config.modules_per_inverter == 120
    assert config.total_modules == 2880
