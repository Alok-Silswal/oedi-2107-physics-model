"""Only the new timing lookup/summary logic; no unrelated sensitivity tests."""

import numpy as np
import pandas as pd
import pytest

from oedi2107.config import SystemConfig
from oedi2107.module import load_cec_module, calculate_module_dc
from oedi2107.poa_timing import OFFSETS_MINUTES, offset_expected_current, summarize_current_ratio


def test_exact_lookup_sign_no_interpolation_and_nonmutation():
    index=pd.date_range("2022-06-01 10:00",periods=9,freq="5min")
    poa=pd.Series(np.arange(9)*50.+300,index=index)
    qc=pd.Series(False,index=index)
    candidate=pd.DataFrame({"temp_cell":35.},index=index[[3,4,5]])
    copies=(poa.copy(),qc.copy(),candidate.copy(deep=True))
    parameters=load_cec_module(SystemConfig().cec_module_entry)
    before=parameters.copy(deep=True)
    result=offset_expected_current(candidate,poa,qc,parameters)
    assert tuple(result.columns)==OFFSETS_MINUTES
    assert (result[15]>result[0]).all() and (result[-15]<result[0]).all()
    direct=calculate_module_dc(poa.loc[candidate.index],candidate.temp_cell,parameters).Imp*6
    np.testing.assert_allclose(result[0],direct)
    pd.testing.assert_series_equal(poa,copies[0])
    pd.testing.assert_series_equal(qc,copies[1])
    pd.testing.assert_frame_equal(candidate,copies[2])
    pd.testing.assert_series_equal(parameters,before)
    missing=poa.drop(index[2])
    absent=offset_expected_current(candidate,missing,qc.loc[missing.index],parameters)
    assert np.isnan(absent.loc[index[3],-5])
    assert np.isfinite(absent.loc[index[3],0])
    assert SystemConfig().dc_losses.k_dc is None


def test_qc_and_nonpositive_sources_are_unavailable_not_filled():
    index=pd.date_range("2022-06-01 10:00",periods=3,freq="5min")
    poa=pd.Series([500.,0.,600.],index=index)
    qc=pd.Series([True,False,False],index=index)
    candidate=pd.DataFrame({"temp_cell":35.},index=index)
    result=offset_expected_current(candidate,poa,qc,load_cec_module(SystemConfig().cec_module_entry))
    assert result[0].isna().tolist()==[True,True,False]
    assert result[15].isna().all()


def test_ratio_summary_sign_dispersion_and_counts():
    index=pd.date_range("2022-06-01",periods=5,freq="5min")
    ratio=pd.Series([1.1,1.2,.8,.9,np.nan],index=index)
    phase=pd.Series(["AM","AM","PM","PM","AM"],index=index)
    result=summarize_current_ratio(ratio,phase)
    assert result["am_median"]==pytest.approx(1.15)
    assert result["pm_median"]==pytest.approx(.85)
    assert result["am_minus_pm"]==pytest.approx(.3)
    assert result["overall_std"]==pytest.approx(np.std([1.1,1.2,.8,.9],ddof=1))
    assert result["overall_iqr"]==pytest.approx(.25)
    assert result["valid_pairs"]==4 and result["excluded"]==1


def test_misaligned_qc_rejected():
    index=pd.date_range("2022-06-01",periods=2,freq="5min")
    with pytest.raises(ValueError,match="aligned"):
        offset_expected_current(pd.DataFrame({"temp_cell":35.},index=index),pd.Series(500.,index=index),
            pd.Series(False,index=index+pd.Timedelta(minutes=5)),load_cec_module(SystemConfig().cec_module_entry))
