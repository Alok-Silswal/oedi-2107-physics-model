import numpy as np
import pandas as pd
import pytest

from oedi2107.config import SystemConfig
from oedi2107.inverter import load_cec_inverter
from oedi2107.measured_conversion import measured_conversion, conversion_metrics, dc_loading_summary


def test_measured_power_missingness_qc_efficiency_and_fixed_cap():
    parameters=load_cec_inverter(SystemConfig().cec_inverter_entry)
    before_parameters=parameters.copy(deep=True)
    index=pd.date_range("2022-06-01 12:00",periods=5,freq="5min")
    raw=pd.DataFrame({"measured_dc_voltage":[700.,700.,np.nan,700.,700.],"measured_dc_current":[20.,44.,20.,20.,20.],
        "measured_ac_power":[13800.,30000.,13000.,15000.,13800.],"electrical_qc_clear":[True,True,True,True,False]},index=index)
    before=raw.copy(deep=True)
    result=measured_conversion(raw,parameters)
    assert result.measured_dc_power.iloc[0]==14000.
    assert np.isnan(result.measured_dc_power.iloc[2])
    assert result.efficiency.iloc[0]==pytest.approx(13800/14000)
    assert result.sandia_ac_from_measured_dc.iloc[1]==27600.
    assert result.residual_ac_power.iloc[1]==2400.
    assert result.ac_dc_power_ratio.iloc[3]>1 and np.isnan(result.efficiency.iloc[3])
    assert not result.conversion_analysis_mask.iloc[4]
    metrics=conversion_metrics(result,result.conversion_analysis_mask)
    assert metrics["valid_pairs"]==3
    bins=dc_loading_summary(result,pdco=float(parameters.Pdco))
    assert bins.records.sum()==3 and bins.ratio_gt_1_count.sum()==1
    pd.testing.assert_frame_equal(raw,before)
    pd.testing.assert_series_equal(parameters,before_parameters)
    assert SystemConfig().dc_losses.k_dc is None
    assert not any("expected_dc" in name for name in result)


def test_daytime_proxy_and_power_floor_are_explicit():
    index=pd.DatetimeIndex(["2022-06-01 01:00","2022-06-01 12:00","2022-06-01 12:05"])
    raw=pd.DataFrame({"measured_dc_voltage":700.,"measured_dc_current":[10.,.1,10.],
        "measured_ac_power":[6900.,50.,6900.],"electrical_qc_clear":True},index=index)
    result=measured_conversion(raw,load_cec_inverter(SystemConfig().cec_inverter_entry))
    assert result.conversion_analysis_mask.tolist()==[False,False,True]
    assert result.efficiency.iloc[:2].isna().all()
