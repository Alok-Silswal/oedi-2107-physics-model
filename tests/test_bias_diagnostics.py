"""Geometry never changes timestamps; matching and thermal scenarios never fit."""

import numpy as np
import pandas as pd
import pvlib
import pytest

from oedi2107.bias_diagnostics import diagnostic_geometry, matched_ampm, matched_summary, thermal_sensitivity, FAIMAN_SCENARIOS
from oedi2107.config import SystemConfig
from oedi2107.module import load_cec_module
from oedi2107.pipeline import run_dc_foundation


LOCATION = dict(latitude=38.996306,longitude=-122.134111,altitude=10.,tilt=25.,azimuth=180.)


def test_diagnostic_localization_is_copy_and_dst_is_unresolved():
    index = pd.DatetimeIndex(["2022-03-13 02:30", "2022-06-01 09:00", "2022-06-01 16:00", "2022-11-06 01:30"],name="timestamp")
    original=index.copy()
    result=diagnostic_geometry(index,**LOCATION,timezone="America/Los_Angeles")
    assert result.index.equals(original) and index.tz is None
    assert result.diagnostic_clock_valid.tolist() == [False,True,True,False]
    assert result.aoi_deg.iloc[[0,3]].isna().all()
    assert result.solar_phase.iloc[1:3].tolist() == ["AM","PM"]
    direct=pvlib.solarposition.get_solarposition(index[1:3].tz_localize("America/Los_Angeles"),
        LOCATION["latitude"],LOCATION["longitude"],altitude=10.)
    np.testing.assert_allclose(result.solar_azimuth_deg.iloc[1:3],direct.azimuth)


def test_geometry_timezone_hypothesis_does_not_shift_measurements():
    index=pd.date_range("2022-07-01 09:00",periods=3,freq="5min")
    primary=diagnostic_geometry(index,**LOCATION,timezone="America/Los_Angeles")
    alternate=diagnostic_geometry(index,**LOCATION,timezone="Etc/GMT+8")
    assert primary.index.equals(alternate.index)
    assert not np.allclose(primary.aoi_deg,alternate.aoi_deg)


def test_geometry_rejects_unconfirmed_or_already_aware_coordinates():
    index=pd.date_range("2022-06-01",periods=2,tz="UTC")
    with pytest.raises(ValueError,match="naive"):
        diagnostic_geometry(index,**LOCATION,timezone="America/Los_Angeles")
    with pytest.raises(ValueError,match="finite"):
        diagnostic_geometry(index.tz_localize(None),**{**LOCATION,"latitude":np.nan},timezone="America/Los_Angeles")


def matched_example():
    index=pd.date_range("2021-06-01 09:00",periods=6,freq="5min")
    return pd.DataFrame({"poa_w_m2":[510,530,550,520,540,560],"temp_cell":35.,"wind_speed_m_s":1.2,
        "aoi_deg":50.,"season":"JJA","solar_phase":["AM"]*3+["PM"]*3,
        "voltage_ratio":[1.]*3+[.98]*3,"current_ratio":[1.1]*3+[.9]*3,
        "power_ratio":[1.1]*3+[.882]*3},index=index)


def test_matching_preserves_cells_and_balances_counts_without_fitting():
    frame=matched_example()
    before=frame.copy(deep=True)
    result=matched_ampm(frame,minimum_per_phase=3,include_year=True,include_aoi=True,include_wind=True)
    assert len(result)==1 and result.balanced_weight.iloc[0]==3
    summary=matched_summary(result)
    assert summary["voltage_PM_minus_AM"]==pytest.approx(-.02)
    assert summary["current_PM_minus_AM"]==pytest.approx(-.2)
    assert summary["power_PM_minus_AM"]==pytest.approx(-.218)
    pd.testing.assert_frame_equal(frame,before)
    # Sparse and one-sided cells are reported, not silently dropped from the table.
    frame.loc[frame.index[0],"poa_w_m2"]=650
    sparse=matched_ampm(frame,minimum_per_phase=3)
    assert len(sparse)==2 and not sparse.adequate_both_phases.any()
    assert matched_summary(sparse)["balanced_samples_per_phase"]==0


def test_matching_includes_year_to_prevent_calendar_mixing():
    frame=matched_example()
    frame.index=pd.DatetimeIndex([*frame.index[:3], *[date.replace(year=2022) for date in frame.index[3:]]])
    assert matched_summary(matched_ampm(frame,minimum_per_phase=2))["adequate_bins"]==1
    assert matched_summary(matched_ampm(frame,minimum_per_phase=2,include_year=True))["adequate_bins"]==0


def test_fixed_thermal_sensitivity_preserves_reference_and_parameters():
    index=pd.date_range("2022-06-01 10:00",periods=3,freq="5min")
    weather=pd.DataFrame({"poa":[400.,500.,600.],"temp_air":25.,"wind_speed":1.},index=index)
    reference=run_dc_foundation(weather,timestamp_basis="pacific_wall_clock")
    frame=weather.rename(columns={"poa":"poa_w_m2","temp_air":"ambient_temperature_c","wind_speed":"wind_speed_m_s"})
    frame["temp_cell"]=reference.temp_cell
    for quantity in ("voltage","current","power"):
        frame[f"expected_dc_{quantity}_ideal"]=reference[f"expected_dc_{quantity}"]
        frame[f"measured_dc_{quantity}_median"]=reference[f"expected_dc_{quantity}"]*.98
    before=frame.copy(deep=True)
    parameters=load_cec_module(SystemConfig().cec_module_entry)
    before_parameters=parameters.copy(deep=True)
    result,equivalent=thermal_sensitivity(frame,parameters)
    assert len(result)==len(frame)*len(FAIMAN_SCENARIOS)
    assert result.scenario.nunique()==7
    assert result.loc[result.scenario.eq("reference"),"voltage_ratio"].eq(.98).all()
    warm=result.loc[result.scenario.eq("combined_warmer")]
    cool=result.loc[result.scenario.eq("combined_cooler")]
    assert (warm.diagnostic_temp_cell.to_numpy()>cool.diagnostic_temp_cell.to_numpy()).all()
    assert (warm.diagnostic_expected_dc_voltage.to_numpy()<cool.diagnostic_expected_dc_voltage.to_numpy()).all()
    assert equivalent.dvmp_dt_v_per_c.lt(0).all()
    assert equivalent.linearized_voltage_equivalent_temperature_offset_c.gt(0).all()
    pd.testing.assert_frame_equal(frame,before)
    pd.testing.assert_series_equal(parameters,before_parameters)
    assert SystemConfig().dc_losses.k_dc is None
    assert "dc_loss_factor" not in result


@pytest.mark.parametrize("options",[{"poa_width":0},{"temperature_width":np.nan},{"minimum_per_phase":0}])
def test_bad_matching_thresholds_are_rejected(options):
    with pytest.raises(ValueError):
        matched_ampm(matched_example(),**options)
