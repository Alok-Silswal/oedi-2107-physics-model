"""CEC/Sandia checks with official parameters; factors below are test inputs only."""

import logging

import numpy as np
import pandas as pd
import pytest
import pvlib

from oedi2107 import (DCLossConfig, SystemConfig, load_cec_inverter,
                      calculate_inverter_ac, run_dc_model, run_inverter_model)
from oedi2107.inverter import MODEL_FIELDS


@pytest.fixture(scope="module")
def parameters():
    return load_cec_inverter(SystemConfig().cec_inverter_entry)


def dc_inputs(powers, voltage=715.):
    return pd.DataFrame({"expected_dc_voltage": voltage, "expected_dc_power": powers},
                        index=pd.date_range("2026-01-01", periods=len(powers), freq="h", tz="UTC"))


def test_zero_and_near_zero_input_retains_tare(parameters):
    dc = dc_inputs([0., 1e-6, float(parameters.Pso) / 2])
    dc.loc[dc.index[0], "expected_dc_voltage"] = 0
    result = calculate_inverter_ac(dc, parameters)
    np.testing.assert_allclose(result.expected_ac_power, -float(parameters.Pnt))
    assert np.isfinite(result.expected_ac_power).all()


def test_power_increases_below_clipping(parameters):
    result = calculate_inverter_ac(dc_inputs([1000., 5000., 10000., 20000.]), parameters)
    assert (np.diff(result.expected_ac_power) > 0).all()
    assert (result.expected_ac_power < float(parameters.Paco)).all()


@pytest.mark.parametrize("voltage", [520., 715., 800.])
def test_finite_output_and_model_clipping(parameters, voltage):
    dc = dc_inputs([0., 1000., 20000., float(parameters.Pdco), 40000., 60000.], voltage)
    result = calculate_inverter_ac(dc, parameters)
    assert np.isfinite(result.expected_ac_power).all()
    assert (result.expected_ac_power <= float(parameters.Paco)).all()
    assert result.expected_ac_power.iloc[-1] == float(parameters.Paco)
    np.testing.assert_allclose(result.expected_ac_power,
                               pvlib.inverter.sandia(dc.expected_dc_voltage, dc.expected_dc_power, parameters))


def test_reference_power_matches_official_paco(parameters):
    result = calculate_inverter_ac(dc_inputs([float(parameters.Pdco)], float(parameters.Vdco)), parameters)
    assert result.expected_ac_power.iloc[0] == pytest.approx(float(parameters.Paco))


def test_official_row_and_inputs_unchanged(parameters):
    official = pvlib.pvsystem.retrieve_sam("cecinverter")[SystemConfig().cec_inverter_entry]
    before = parameters.copy(deep=True)
    dc = dc_inputs([0., 10000., 40000.])
    dc.attrs["dc_loss_factor"] = 0.9  # Synthetic metadata, no parameter fitting.
    dc_before = dc.copy(deep=True)
    result = calculate_inverter_ac(dc, parameters)
    pd.testing.assert_series_equal(parameters, before)
    pd.testing.assert_series_equal(parameters, official)
    pd.testing.assert_frame_equal(dc, dc_before)
    pd.testing.assert_frame_equal(result[dc.columns], dc_before)
    assert result.attrs["dc_loss_factor"] == dc.attrs["dc_loss_factor"]
    assert result.attrs["cec_inverter_entry"] == official.name
    for field in MODEL_FIELDS:
        assert result.attrs["inverter_parameters"][field] == official[field]


def test_exact_entry_and_candidates_logged(caplog):
    with caplog.at_level(logging.INFO):
        row = load_cec_inverter(SystemConfig().cec_inverter_entry)
    assert row.name in caplog.text
    assert "ABB__TRIO_27_6_TL_OUTD_S1B_US_480_A__480V_" in caplog.text
    assert "27600" in caplog.text
    assert "sam-library-cec-inverters-2019-03-05.csv" in caplog.text


def test_no_fuzzy_fallback():
    with pytest.raises(ValueError, match="review candidates"):
        load_cec_inverter("ABB TRIO-27.6-TL-OUTD-S1B")


@pytest.mark.parametrize("voltage,power", [(-1., 100.), (715., -1.), (np.nan, 100.),
                                         (715., np.inf), (715., np.nan), (0., 1.)])
def test_invalid_dc_rejected(parameters, voltage, power):
    with pytest.raises(ValueError):
        calculate_inverter_ac(dc_inputs([power], voltage), parameters)


@pytest.mark.parametrize("factor", [None, 1.])
def test_pipeline_keeps_prior_blocks_and_unset_factor(factor):
    weather = pd.DataFrame({"poa": [0., 500., 1000.], "temp_air": 25., "wind_speed": 1.},
                           index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"))
    config = SystemConfig(dc_losses=DCLossConfig(k_dc=factor))
    dc = run_dc_model(weather, config)
    result = run_inverter_model(weather, config)
    pd.testing.assert_frame_equal(result[dc.columns], dc)
    assert result.attrs["dc_loss_factor"] == factor
    if factor is None:
        assert result.expected_ac_power.isna().all()
        assert "K_DC unset" in result.attrs["ac_status"]
    else:
        assert np.isfinite(result.expected_ac_power).all()
        assert result.expected_ac_power.iloc[0] == -8.28
