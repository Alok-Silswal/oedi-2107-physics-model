"""Thin prepared-data mapping; no cleaning or inferred electrical measurements."""

import re

import numpy as np
import pandas as pd

from .config import SystemConfig
from .pipeline import run_dc_foundation

INPUT_MAPPING = {"poa_w_m2": "poa", "ambient_temperature_c": "temp_air", "wind_speed_m_s": "wind_speed"}
FLAGS = ("poa_available", "environment_available", "model_inputs_available",
         "all_inverter_ac_available", "electrical_qc_flag", "irradiance_qc_flag", "dst_fall_ambiguous")


def discover_channels(columns) -> pd.DataFrame:
    """Actual schema names → exact channel parsing → 24-ID DC/AC source mapping."""
    records = {}
    for column in columns:
        match = re.fullmatch(r"inv_(\d{2})_(dc_voltage|dc_current|ac_power)_inv_(\d+)", column)
        if match:
            identifier, quantity, metric = match.groups()
            record = records.setdefault(identifier, {"inverter_id": f"inv_{identifier}"})
            if quantity in record:
                raise ValueError(f"Multiple {quantity} channels for inverter {identifier}")
            record[quantity] = column
    if set(records) != {f"{i:02d}" for i in range(1, 25)}:
        raise ValueError("Prepared schema must represent inverter IDs 01–24")
    mapping = pd.DataFrame([records[key] for key in sorted(records)])
    if set(mapping.columns) != {"inverter_id", "dc_voltage", "dc_current", "ac_power"} or mapping.isna().any().any():
        raise ValueError("Prepared schema needs one DC V/I and provenance AC channel per inverter")
    return mapping


def confirm_dc_units(mapping: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Prepared channel IDs + raw metadata → unit checks → V/A/W provenance.

    Parquet numeric types carry no units. Prepared channels match original
    metadata with no V/I scaling in the accepted cleaning workflow.
    """
    result = mapping.copy()
    for quantity, unit in (("dc_voltage", "V"), ("dc_current", "A")):
        for column in result[quantity]:
            record = metadata["Metrics"].get(column)
            if record is None or record.get("units") != unit:
                raise ValueError(f"Cannot confirm {unit} units for {column}")
        result[f"{quantity}_unit"] = unit
    result["derived_dc_power_unit"] = "W (V × A)"
    return result


def prepared_dc_baseline(prepared: pd.DataFrame, config: SystemConfig | None = None) -> pd.DataFrame:
    """Prepared weather/flags → available-row Blocks 1–3 → all-row ideal DC table.

    Parses timestamp text without timezone assignment. Invalid weather remains
    unchanged and identifiable with NaN predictions; QC/DST flags do not select
    or modify rows. No K_DC, inverter conversion, filling, clipping or shifting.
    """
    required = {"measured_on", *INPUT_MAPPING, *FLAGS}
    if not required.issubset(prepared.columns):
        raise ValueError(f"Missing prepared columns: {sorted(required - set(prepared.columns))}")
    index = pd.DatetimeIndex(pd.to_datetime(prepared.measured_on, format="%Y-%m-%d %H:%M:%S", errors="raise"), name="timestamp")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Prepared timestamps must remain unique/increasing; DST resolution is deferred")
    result = prepared.loc[:, ["measured_on", *INPUT_MAPPING, *FLAGS]].copy().set_axis(index)
    if "wind_direction_deg" in prepared:
        result["wind_direction_deg"] = prepared.wind_direction_deg.to_numpy()
    for flag in FLAGS:
        if result[flag].isna().any() or not pd.api.types.is_bool_dtype(result[flag]):
            raise ValueError(f"Prepared flag must be explicitly boolean: {flag}")
    weather = result[list(INPUT_MAPPING)].rename(columns=INPUT_MAPPING)
    finite = np.isfinite(weather).all(axis=1)
    supported = finite & weather.poa.ge(0) & weather.wind_speed.ge(0)
    eligible = result.model_inputs_available & supported
    result["dc_model_computed"] = eligible
    result["dc_model_status"] = np.select(
        [~result.model_inputs_available, ~finite, ~supported],
        ["inputs unavailable", "nonfinite inputs", "unsupported negative inputs"], default="computed ideal",
    )
    output_names = ["temp_cell", "Vmp", "Imp", "Pmp", "Voc", "Isc",
                    "expected_dc_voltage_ideal", "expected_dc_current_ideal", "expected_dc_power_ideal"]
    result[output_names] = np.nan
    if eligible.any():
        dc = run_dc_foundation(weather.loc[eligible], config, timestamp_basis="pacific_wall_clock")
        dc = dc.rename(columns={f"expected_dc_{q}": f"expected_dc_{q}_ideal" for q in ("voltage", "current", "power")})
        result.loc[eligible, output_names] = dc[output_names]
        result.attrs.update(dc.attrs)
    result.attrs.update(dc_stage="IDEAL / PRE-LOSS; Blocks 1–3 and inferred array scaling only",
                        timestamp_basis="pacific_wall_clock", dst_policy="unresolved; no localization or UTC conversion",
                        k_dc=None, cleaning="none")
    return result


def measured_dc_for_inverter(baseline: pd.DataFrame, readings: pd.DataFrame, channel: dict) -> pd.DataFrame:
    """One ideal baseline + mapped DC readings → named all-row comparison table.

    Reads V/A unchanged, derives W only from finite V and I. Inverter 05 voltage
    stays missing. AC mapping is provenance only and never enters calculation.
    """
    if len(readings) != len(baseline) or not np.array_equal(readings.measured_on.to_numpy(), baseline.measured_on.to_numpy()):
        raise ValueError("Measured DC rows/timestamp strings must match the prepared baseline exactly")
    result = baseline.copy()
    result["inverter_id"] = channel["inverter_id"]
    result["measured_dc_voltage"] = readings[channel["dc_voltage"]].to_numpy()
    result["measured_dc_current"] = readings[channel["dc_current"]].to_numpy()
    finite = np.isfinite(result.measured_dc_voltage) & np.isfinite(result.measured_dc_current)
    result["measured_dc_power"] = np.nan
    result.loc[finite, "measured_dc_power"] = result.loc[finite, "measured_dc_voltage"] * result.loc[finite, "measured_dc_current"]
    result.attrs.update(channel_provenance=channel, measured_dc_power_unit="W = V × A")
    return result
