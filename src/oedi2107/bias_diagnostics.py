"""Diagnostic geometry, matched comparisons and fixed thermal perturbations only."""

import numpy as np
import pandas as pd
import pvlib

from .array import scale_module_mpp
from .config import FaimanConfig, SystemConfig
from .module import calculate_module_dc
from .thermal import estimate_cell_temperature

# Deliberate exploratory bounds, not literature confidence intervals or fitted values.
FAIMAN_SCENARIOS = (
    ("reference", 25.0, 6.84),
    ("u0_lower_20pct", 20.0, 6.84),
    ("u0_upper_20pct", 30.0, 6.84),
    ("u1_lower_25pct", 25.0, 5.13),
    ("u1_upper_25pct", 25.0, 8.55),
    ("combined_warmer", 20.0, 5.13),
    ("combined_cooler", 30.0, 8.55),
)


def diagnostic_geometry(index, *, latitude, longitude, altitude, tilt, azimuth, timezone):
    """Original wall clock + metadata -> pvlib solar position/AOI -> diagnostic columns.

    Only a copy is localized. Ambiguous/nonexistent local times remain unresolved
    (NaT); no inference, shifts, UTC reassignment or production-model changes.
    AM/PM means east/west of solar noon, not clock noon. AOI uses the direct-beam
    sun vector; it does not describe the diffuse field or modify measured POA.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None or not index.is_unique:
        raise ValueError("Geometry requires unique original naive wall-clock timestamps")
    if not np.isfinite([latitude, longitude, altitude, tilt, azimuth]).all():
        raise ValueError("Confirmed finite site/mount metadata are required")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or not 0 <= tilt <= 90 or not 0 <= azimuth < 360:
        raise ValueError("Invalid site coordinates or mount orientation")
    localized = index.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    valid = ~localized.isna()
    output = pd.DataFrame(index=index)
    output["diagnostic_clock_valid"] = valid
    output[["solar_zenith_deg", "solar_azimuth_deg", "solar_elevation_deg", "aoi_deg"]] = np.nan
    if valid.any():
        solar = pvlib.solarposition.get_solarposition(localized[valid], latitude, longitude, altitude=altitude)
        output.loc[valid, "solar_zenith_deg"] = solar.apparent_zenith.to_numpy()
        output.loc[valid, "solar_azimuth_deg"] = solar.azimuth.to_numpy()
        output.loc[valid, "solar_elevation_deg"] = solar.apparent_elevation.to_numpy()
        output.loc[valid, "aoi_deg"] = pvlib.irradiance.aoi(tilt, azimuth, solar.apparent_zenith, solar.azimuth).to_numpy()
    day = output.solar_elevation_deg.gt(0)
    output["solar_phase"] = pd.Series(pd.NA, index=index, dtype="string")
    output.loc[day & output.solar_azimuth_deg.lt(180), "solar_phase"] = "AM"
    output.loc[day & output.solar_azimuth_deg.ge(180), "solar_phase"] = "PM"
    output["clock_phase"] = np.where(index.hour < 12, "AM", "PM")
    output.attrs.update(timezone_hypothesis=timezone, use="diagnostic only", atmospheric_temperature_c=12.0)
    return output


def matched_ampm(frame, *, poa_width=100.0, temperature_width=10.0, include_year=False,
                 include_aoi=False, include_wind=False, minimum_per_phase=10):
    """Frozen candidate conditions -> shared AM/PM bins -> ratios/differences/counts.

    Bin starts are multiples of the specified widths. Season is always matched;
    year, 10-degree AOI and 1 m/s wind matching are optional confounding checks.
    Every occupied bin is returned, including insufficient/unmatched bins.
    The count cutoff is only for balanced descriptive summaries, not selection.
    """
    if not np.isfinite([poa_width, temperature_width]).all() or min(poa_width, temperature_width) <= 0:
        raise ValueError("Matching bin widths must be finite and positive")
    if not isinstance(minimum_per_phase, int) or minimum_per_phase < 1:
        raise ValueError("minimum_per_phase must be a positive integer")
    if not frame.index.is_unique or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Matching requires unique original timestamps")
    work = frame.copy()
    work["poa_bin_start"] = np.floor(work.poa_w_m2 / poa_width) * poa_width
    work["temp_bin_start"] = np.floor(work.temp_cell / temperature_width) * temperature_width
    keys = ["poa_bin_start", "temp_bin_start", "season"]
    if include_year:
        work["year"] = work.index.year
        keys.append("year")
    if include_aoi:
        work["aoi_bin_start"] = np.floor(work.aoi_deg / 10) * 10
        keys.append("aoi_bin_start")
    if include_wind:
        work["wind_bin_start"] = np.floor(work.wind_speed_m_s)
        keys.append("wind_bin_start")
    rows = []
    for values, group in work.groupby(keys, observed=True, dropna=False):
        row = dict(zip(keys, values))
        for phase in ("AM", "PM"):
            subset = group.loc[group.solar_phase.eq(phase)]
            row[f"{phase}_samples"] = len(subset)
            for name in ("voltage_ratio", "current_ratio", "power_ratio", "poa_w_m2", "temp_cell", "wind_speed_m_s", "aoi_deg"):
                row[f"{phase}_{name}_median"] = subset[name].median()
        row["balanced_weight"] = min(row["AM_samples"], row["PM_samples"])
        row["adequate_both_phases"] = row["balanced_weight"] >= minimum_per_phase
        for name in ("voltage", "current", "power"):
            row[f"{name}_PM_minus_AM"] = row[f"PM_{name}_ratio_median"] - row[f"AM_{name}_ratio_median"]
        rows.append(row)
    return pd.DataFrame(rows)


def matched_summary(table):
    """Shared populated bins -> min(AM count, PM count) weights -> descriptive contrasts."""
    selected = table.loc[table.adequate_both_phases] if len(table) else table
    result = {"occupied_bins":len(table), "adequate_bins":len(selected),
              "balanced_samples_per_phase":int(selected.balanced_weight.sum()) if len(selected) else 0}
    for quantity in ("voltage", "current", "power"):
        for phase in ("AM", "PM"):
            result[f"{quantity}_{phase}"] = float(np.average(selected[f"{phase}_{quantity}_ratio_median"], weights=selected.balanced_weight)) if len(selected) else np.nan
        result[f"{quantity}_PM_minus_AM"] = result[f"{quantity}_PM"] - result[f"{quantity}_AM"]
    return result


def thermal_sensitivity(frame, parameters, config=None):
    """Frozen candidates + unchanged CEC/topology -> fixed Faiman scenarios -> diagnostic DC.

    Original temperature and ideal outputs are preserved. Coefficients are not
    optimized; no scenario is adopted or used to change candidate eligibility.
    A central ±1°C derivative gives a linearized voltage-equivalent temperature
    offset, not measured cell temperature or a fitted temperature parameter.
    """
    config = config or SystemConfig()
    if config.dc_losses.k_dc is not None:
        raise ValueError("K_DC must remain unset for this diagnostic stage")
    frames = []
    for label, u0, u1 in FAIMAN_SCENARIOS:
        thermal = FaimanConfig(u0=u0, u1=u1, provenance="diagnostic fixed perturbation; never adopted")
        temperature = estimate_cell_temperature(frame.poa_w_m2, frame.ambient_temperature_c, frame.wind_speed_m_s, thermal)
        dc = scale_module_mpp(calculate_module_dc(frame.poa_w_m2, temperature, parameters), config)
        result = frame.copy()
        result["scenario"] = label
        result["diagnostic_u0"], result["diagnostic_u1"] = u0, u1
        result["diagnostic_temp_cell"] = temperature
        for quantity in ("voltage", "current", "power"):
            expected = dc[f"expected_dc_{quantity}"]
            result[f"diagnostic_expected_dc_{quantity}"] = expected
            result[f"{quantity}_ratio"] = result[f"measured_dc_{quantity}_median"] / expected
            if label == "reference":
                np.testing.assert_allclose(expected, frame[f"expected_dc_{quantity}_ideal"], rtol=1e-10, atol=1e-8)
        frames.append(result)
    plus = scale_module_mpp(calculate_module_dc(frame.poa_w_m2, frame.temp_cell + 1, parameters), config)
    minus = scale_module_mpp(calculate_module_dc(frame.poa_w_m2, frame.temp_cell - 1, parameters), config)
    derivative = (plus.expected_dc_voltage - minus.expected_dc_voltage) / 2
    equivalent = (frame.measured_dc_voltage_median - frame.expected_dc_voltage_ideal) / derivative
    reference = pd.DataFrame({"dvmp_dt_v_per_c":derivative, "linearized_voltage_equivalent_temperature_offset_c":equivalent}, index=frame.index)
    return pd.concat(frames, names=None), reference
