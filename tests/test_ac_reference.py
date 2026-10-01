"""Diagnostic no-loss conversion and loading/plateau calculations only."""

import numpy as np
import pandas as pd
import pytest

from oedi2107.ac_reference import calculate_no_loss_ac, loading_summary, measured_upper_region
from oedi2107.config import SystemConfig
from oedi2107.inverter import load_cec_inverter, calculate_inverter_ac


@pytest.fixture(scope="module")
def parameters():
    return load_cec_inverter(SystemConfig().cec_inverter_entry)


def test_diagnostic_calls_existing_sandia_preserves_normal_output_and_missing(parameters):
    index=pd.date_range("2022-06-01",periods=4,freq="5min",name="timestamp")
    ideal=pd.DataFrame({"expected_dc_voltage_ideal":[0.,715.,715.,np.nan],
        "expected_dc_power_ideal":[0.,10000.,40000.,np.nan],"expected_ac_power":[0.,8500.,26000.,np.nan]},index=index)
    before=ideal.copy(deep=True)
    params_before=parameters.copy(deep=True)
    result=calculate_no_loss_ac(ideal,parameters)
    direct=calculate_inverter_ac(ideal.iloc[:3,:2].rename(columns={"expected_dc_voltage_ideal":"expected_dc_voltage","expected_dc_power_ideal":"expected_dc_power"}),parameters)
    np.testing.assert_allclose(result.expected_ac_power_no_loss.iloc[:3],direct.expected_ac_power)
    assert result.expected_ac_power_no_loss.iloc[0]==-float(parameters.Pnt)
    assert result.expected_ac_power_no_loss.iloc[2]==float(parameters.Paco)
    assert np.isnan(result.expected_ac_power_no_loss.iloc[3])
    assert result.no_loss_ac_computed.tolist()==[True,True,True,False]
    assert "expected_dc_power" not in result and "dc_loss_factor" not in result
    pd.testing.assert_series_equal(result.expected_ac_power,before.expected_ac_power)
    pd.testing.assert_frame_equal(result[ideal.columns],before)
    pd.testing.assert_frame_equal(ideal,before)
    pd.testing.assert_series_equal(parameters,params_before)
    assert result.attrs["diagnostic_k_dc_applied"] is False
    assert SystemConfig().dc_losses.k_dc is None


def test_no_calibrated_ac_interface_created(parameters):
    result=calculate_no_loss_ac(pd.DataFrame({"expected_dc_voltage_ideal":[715.],"expected_dc_power_ideal":[10000.]}),parameters)
    assert "expected_ac_power" not in result


@pytest.mark.parametrize("v,p",[(-1.,1000.),(715.,-1.),(0.,1.),(np.inf,1000.)])
def test_invalid_finite_or_infinite_ideal_dc_rejected(parameters,v,p):
    with pytest.raises(ValueError):
        calculate_no_loss_ac(pd.DataFrame({"expected_dc_voltage_ideal":[v],"expected_dc_power_ideal":[p]}),parameters)


def test_loading_bin_clipping_boundary_and_residual():
    expected=pd.Series([500.,1000.,1000.,-8.28])
    measured=pd.Series([450.,1100.,np.nan,0.])
    result=loading_summary(expected,measured,paco=1000.,basis="expected")
    cap=result.loc[result.loading_bin.eq("[1.0, 1.05)")].iloc[0]
    assert cap.total_samples==2 and cap.valid_pairs==1 and cap.excluded_pairs==1
    assert cap.median_residual_w==100.
    assert cap.median_ratio==pytest.approx(1.1)
    tare=result.loc[result.loading_bin.eq("[-inf, 0.0)")].iloc[0]
    assert np.isnan(tare.median_ratio) and tare.valid_ratio_count==0


def test_measured_plateau_does_not_become_a_model_rating():
    index=pd.date_range("2022-06-01 12:00",periods=20,freq="5min")
    values=pd.Series([29950.]*15+[30025.]*3+[105000.,np.nan],index=index)
    before=values.copy(deep=True)
    stats=measured_upper_region(values,paco=27600.)
    assert stats["dominant_high_band_lower_w"]==29900.
    assert stats["highest_repeated_band_lower_w"]==29900.
    assert stats["longest_dominant_band_run_samples"]==15
    assert stats["maximum_w"]==105000.
    assert stats["above_paco_count"]==19 and stats["excluded_nonfinite"]==1
    pd.testing.assert_series_equal(values,before)


def test_loading_unavailable_basis_is_counted():
    result=loading_summary([np.nan,500.],[100.,450.],paco=1000.,basis="expected")
    assert result.total_samples.sum()==2 and result.valid_pairs.sum()==1
    missing=result.loc[result.loading_bin.eq("unavailable_loading")].iloc[0]
    assert missing.excluded_pairs==1
