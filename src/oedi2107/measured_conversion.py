"""Measured electrical conversion diagnostic, independent of modelled PV DC."""

import numpy as np
import pandas as pd

from .inverter import calculate_inverter_ac
from .validation import mae, rmse, mbe


def measured_conversion(readings, parameters):
    """Measured V/A/AC W + QC -> V×I and fixed Sandia -> measured conversion table.

    Daytime proxy is Pacific local 06:00–20:00, combined with positive generation.
    Main operation requires DC>=1% Pdco and AC>=1% Paco to avoid very-low-power
    ratio instability. These are screening floors, not calibrated ratings.
    Ratios above one remain visible, but are not labelled physical efficiencies.
    """
    required={"measured_dc_voltage","measured_dc_current","measured_ac_power","electrical_qc_clear"}
    if not required.issubset(readings) or not isinstance(readings.index,pd.DatetimeIndex) or readings.index.tz is not None:
        raise ValueError("Need measured V/A/W/QC and original naive local timestamps")
    if not readings.index.is_unique or not readings.index.is_monotonic_increasing:
        raise ValueError("Electrical timestamps must be unique and increasing")
    if not pd.api.types.is_bool_dtype(readings.electrical_qc_clear) or readings.electrical_qc_clear.isna().any():
        raise ValueError("QC state must be explicit Boolean")
    result=readings.copy(deep=True)
    finite_dc=np.isfinite(result.measured_dc_voltage)&np.isfinite(result.measured_dc_current)
    result["measured_dc_power"]=(result.measured_dc_voltage*result.measured_dc_current).where(finite_dc)
    positive=finite_dc & np.isfinite(result.measured_dc_power) & np.isfinite(result.measured_ac_power) & result.measured_dc_voltage.gt(0) & result.measured_dc_current.gt(0) & result.measured_ac_power.gt(0)
    result["daytime_clock_proxy"]=(result.index.hour>=6)&(result.index.hour<20)
    result["positive_day_qc"] = positive & result.daytime_clock_proxy & result.electrical_qc_clear
    result["conversion_analysis_mask"] = result.positive_day_qc & result.measured_dc_power.ge(.01*float(parameters.Pdco)) & result.measured_ac_power.ge(.01*float(parameters.Paco))
    result["sandia_ac_from_measured_dc"] = np.nan
    valid=result.conversion_analysis_mask
    inputs=result.loc[valid,["measured_dc_voltage","measured_dc_power"]].rename(columns={"measured_dc_voltage":"expected_dc_voltage","measured_dc_power":"expected_dc_power"})
    result.loc[valid,"sandia_ac_from_measured_dc"]=calculate_inverter_ac(inputs,parameters).expected_ac_power
    result["ac_dc_power_ratio"]=(result.measured_ac_power/result.measured_dc_power).where(valid)
    result["efficiency_physically_plausible"]=valid & result.ac_dc_power_ratio.gt(0) & result.ac_dc_power_ratio.le(1)
    result["efficiency"]=result.ac_dc_power_ratio.where(result.efficiency_physically_plausible)
    result["residual_ac_power"]=(result.measured_ac_power-result.sandia_ac_from_measured_dc).where(valid)
    result.attrs.update(k_dc=None,use="measured electrical diagnostic only",efficiency_policy="retain ratio>1, physical efficiency undefined there")
    return result


def conversion_metrics(table, mask):
    """Explicit selected records -> finite measured/Sandia pairs -> AC error W/counts."""
    chosen=table.loc[mask]
    paired=np.isfinite(chosen.measured_ac_power)&np.isfinite(chosen.sandia_ac_from_measured_dc)
    e=chosen.loc[paired,"sandia_ac_from_measured_dc"]
    m=chosen.loc[paired,"measured_ac_power"]
    return {"selected_samples":len(chosen),"valid_pairs":int(paired.sum()),"excluded_pairs":int((~paired).sum()),
            "mae_w":mae(e,m),"rmse_w":rmse(e,m),"mbe_w":mbe(e,m)}


def dc_loading_summary(table, *, pdco):
    """Measured DC / official Pdco -> explicit loading bins -> conversion summaries."""
    selected=table.loc[table.conversion_analysis_mask].copy()
    selected["dc_loading_bin"]=pd.cut(selected.measured_dc_power/pdco,[0,.01,.05,.1,.25,.5,.75,.9,1,1.05,1.1,1.2,1.5,2,np.inf],right=False)
    rows=[]
    for label,group in selected.groupby("dc_loading_bin",observed=True):
        rows.append({"dc_loading_bin":str(label),"records":len(group),"median_measured_dc_w":group.measured_dc_power.median(),
            "median_measured_ac_w":group.measured_ac_power.median(),"median_sandia_ac_w":group.sandia_ac_from_measured_dc.median(),
            "efficiency_valid_count":int(group.efficiency.notna().sum()),"ratio_gt_1_count":int(group.ac_dc_power_ratio.gt(1).sum()),
            "efficiency_p10":group.efficiency.quantile(.1),"median_efficiency":group.efficiency.median(),"efficiency_p90":group.efficiency.quantile(.9),
            "median_raw_ac_dc_ratio":group.ac_dc_power_ratio.median(),"median_residual_w":group.residual_ac_power.median(),
            **conversion_metrics(group,pd.Series(True,index=group.index))})
    return pd.DataFrame(rows)
