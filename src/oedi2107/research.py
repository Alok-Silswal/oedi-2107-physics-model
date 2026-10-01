"""Frozen research path and strict start-of-interval meter alignment."""

import numpy as np
import pandas as pd

from .config import SystemConfig
from .inverter import calculate_inverter_ac_30kw_ceiling, load_cec_inverter
from .real_data import prepared_dc_baseline


def research_baseline(prepared):
    """Prepared weather -> unchanged ideal DC -> 30 kW extension, without K_DC."""
    config = SystemConfig()
    baseline = prepared_dc_baseline(prepared, config)
    for quantity in ("voltage", "current", "power"):
        baseline[f"expected_dc_{quantity}"] = baseline[f"expected_dc_{quantity}_ideal"]
    parameters = load_cec_inverter(config.cec_inverter_entry)
    valid = baseline.dc_model_computed
    ac = calculate_inverter_ac_30kw_ceiling(
        baseline.loc[valid, ["expected_dc_voltage", "expected_dc_power"]], parameters
    )
    baseline["expected_ac_power_cec"] = ac.expected_ac_power_cec.reindex(baseline.index)
    baseline["expected_ac_power"] = ac.expected_ac_power_30kw.reindex(baseline.index)
    baseline["model_available"] = np.isfinite(baseline.expected_ac_power)
    baseline.attrs.update(ac.attrs, research_dc_policy="ideal pre-loss; K_DC bypassed, not set to 1",
                          research_inverter_policy="30 kW external ceiling; official coefficients")
    return baseline


def meter_intervals(plant, meter, flagged_timestamps):
    """5-minute plant W + meter kW at t -> exact t/t+5/t+10 mean -> kW pairs."""
    if not plant.index.is_unique or not meter.index.is_unique:
        raise ValueError("Duplicate local timestamps require explicit resolution")
    result = meter.copy()
    values = np.column_stack([
        plant.expected_plant_ac_power.reindex(meter.index + pd.Timedelta(minutes=offset)).to_numpy()
        for offset in (0, 5, 10)
    ])
    result["model_sample_count"] = np.isfinite(values).sum(axis=1)
    result["expected_plant_ac_power_kw"] = np.where(
        result.model_sample_count.eq(3), values.sum(axis=1) / 3000., np.nan
    )
    result["meter_qc_clear"] = ~result.index.isin(flagged_timestamps)
    result["model_available"] = result.model_sample_count.eq(3)
    result["residual_kw"] = result.meter_ac_power_kw - result.expected_plant_ac_power_kw
    return result


def complete_power_scales(intervals, frequency):
    """15-minute average kW -> complete nominal wall-clock bins -> mean kW/kWh.

    Require 4 samples/hour, 96/day, or days-in-month*96/month. No partial-bin
    energy extrapolation. DST ambiguity remains; durations are nominal local.
    QC-clear totals require every constituent interval to be clear.
    """
    data = intervals.copy()
    paired = np.isfinite(data.expected_plant_ac_power_kw) & np.isfinite(data.meter_ac_power_kw)
    data["valid_pair"] = paired
    data["paired_expected_kw"] = data.expected_plant_ac_power_kw.where(paired)
    data["paired_measured_kw"] = data.meter_ac_power_kw.where(paired)
    grouped = data.resample(frequency, label="left", closed="left")
    result = grouped[["expected_plant_ac_power_kw", "meter_ac_power_kw"]].mean()
    result["valid_interval_count"] = grouped.valid_pair.sum()
    required = {"h": 4, "D": 96}.get(frequency)
    if frequency == "MS":
        required = result.index.days_in_month * 96
    if required is None:
        raise ValueError("Supported scales: h, D, MS")
    result["required_interval_count"] = required
    result["paired_coverage_fraction"] = result.valid_interval_count / result.required_interval_count
    # Partial-bin statistics use exactly the same observed intervals on both sides.
    result["paired_expected_average_kw"] = grouped.paired_expected_kw.mean()
    result["paired_measured_average_kw"] = grouped.paired_measured_kw.mean()
    result["observed_expected_energy_kwh"] = grouped.paired_expected_kw.sum(min_count=1) * .25
    result["observed_measured_energy_kwh"] = grouped.paired_measured_kw.sum(min_count=1) * .25
    complete = result.valid_interval_count.eq(result.required_interval_count)
    result["model_available"] = complete
    result["meter_qc_clear"] = grouped.meter_qc_clear.all()
    for column in ("expected_plant_ac_power_kw", "meter_ac_power_kw"):
        result[column] = result[column].where(complete)
        energy_name = "expected_energy_kwh" if column.startswith("expected") else "measured_energy_kwh"
        result[energy_name] = result[column] * result.required_interval_count * .25
    result["residual_kw"] = result.meter_ac_power_kw - result.expected_plant_ac_power_kw
    return result
