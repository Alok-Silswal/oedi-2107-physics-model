"""Ideal/pre-loss DC -> unchanged Sandia -> diagnostic no-loss AC reference."""

import numpy as np
import pandas as pd

from .inverter import calculate_inverter_ac, inverter_provenance


def calculate_no_loss_ac(ideal: pd.DataFrame, parameters: pd.Series) -> pd.DataFrame:
    """Ideal array V [V], P [W] -> existing Sandia implementation -> labelled AC [W].

    Bypasses Block 4 without assigning K_DC. Missing ideal predictions remain
    identifiable. Original columns, timestamps and any normal AC output are
    preserved. Clipping and signed tare come solely from the Sandia function.
    """
    required = ["expected_dc_voltage_ideal", "expected_dc_power_ideal"]
    if not set(required).issubset(ideal):
        raise ValueError("Ideal/pre-loss DC voltage and power are required")
    if "expected_ac_power_no_loss" in ideal or "no_loss_ac_computed" in ideal:
        raise ValueError("No-loss diagnostic output already exists")
    if np.isinf(ideal[required].to_numpy(dtype=float)).any():
        raise ValueError("Infinite ideal DC predictions are invalid")
    finite = np.isfinite(ideal[required]).all(axis=1)
    temporary = ideal.loc[finite, required].rename(columns={
        "expected_dc_voltage_ideal":"expected_dc_voltage",
        "expected_dc_power_ideal":"expected_dc_power",
    })
    converted = calculate_inverter_ac(temporary, parameters)
    result = ideal.copy(deep=True)
    result["expected_ac_power_no_loss"] = np.nan
    result.loc[finite, "expected_ac_power_no_loss"] = converted.expected_ac_power
    result["no_loss_ac_computed"] = finite
    result.attrs.update(inverter_provenance(parameters),
                        diagnostic_stage="IDEAL DC -> Sandia -> NO-LOSS AC REFERENCE; not calibrated",
                        diagnostic_k_dc_applied=False)
    return result


def loading_summary(expected, measured, *, paco, basis):
    """Aligned AC [W] -> explicit Paco loading bins -> paired medians and counts.

    Residual is measured minus no-loss expected. Ratios are measured/expected
    only for positive expected AC; startup/tare ratios remain unavailable.
    Caller selects the daylight/QC population; this function does no cleaning.
    """
    if isinstance(expected, pd.Series) and isinstance(measured, pd.Series) and not expected.index.equals(measured.index):
        raise ValueError("AC pairs must have exactly aligned indexes")
    e, m = np.asarray(expected, dtype=float), np.asarray(measured, dtype=float)
    if e.ndim != 1 or e.shape != m.shape or basis not in ("expected", "measured"):
        raise ValueError("Require aligned one-dimensional AC and an explicit loading basis")
    if not np.isfinite(paco) or paco <= 0:
        raise ValueError("Official Paco must be finite and positive")
    finite = np.isfinite(e) & np.isfinite(m)
    ratio = np.divide(m, e, out=np.full_like(e, np.nan), where=finite & (e > 0))
    frame = pd.DataFrame({"expected":e, "measured":m, "residual":m-e, "ratio":ratio, "paired":finite})
    edges = [-np.inf,0,.05,.1,.25,.5,.75,.9,.95,1,1.05,1.1,1.15,1.2,np.inf]
    frame["bin"] = pd.cut(frame[basis] / paco, edges, right=False)
    rows = []
    for label, group in frame.groupby("bin", observed=True, dropna=False):
        paired = group.loc[group.paired]
        rows.append({"loading_basis":basis,"loading_bin":"unavailable_loading" if pd.isna(label) else str(label),"total_samples":len(group),
                     "valid_pairs":len(paired),"excluded_pairs":len(group)-len(paired),
                     "median_measured_ac_w":paired.measured.median(),"median_expected_ac_no_loss_w":paired.expected.median(),
                     "median_residual_w":paired.residual.median(),"median_ratio":paired.ratio.median(),
                     "valid_ratio_count":int(paired.ratio.notna().sum())})
    return pd.DataFrame(rows)


def measured_upper_region(measured: pd.Series, *, paco, bin_width_w=100., minimum_repeats=12):
    """Finite measured telemetry -> percentiles and repeated upper bins -> plateau evidence.

    Caller explicitly selects QC policy. The 100 W histogram and minimum 12
    observations identify repeated regions, not a physical clipping limit.
    Longest run counts exact five-minute adjacent records in the dominant band.
    No maximum is removed, capped, fitted or adopted as Paco.
    """
    if not np.isfinite([paco,bin_width_w]).all() or min(paco,bin_width_w) <= 0 or minimum_repeats < 1:
        raise ValueError("Ratings/bin width/repeat count must be positive")
    if not isinstance(measured.index, pd.DatetimeIndex) or not measured.index.is_unique or not measured.index.is_monotonic_increasing:
        raise ValueError("Upper-region diagnostics require unique increasing timestamps")
    values = measured[np.isfinite(measured)]
    positive = values[values > 0]
    high = values[values >= paco]
    counts = (np.floor(high / bin_width_w) * bin_width_w).value_counts().sort_index()
    repeated = counts[counts >= minimum_repeats]
    mode = float(counts.idxmax()) if len(counts) else np.nan
    in_band = measured.ge(mode) & measured.lt(mode + bin_width_w)
    adjacent = measured.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    run_id = (~in_band | ~adjacent).cumsum()
    runs = in_band[in_band].groupby(run_id[in_band]).size()
    return {"total_samples":len(measured),"finite_samples":len(values),"excluded_nonfinite":len(measured)-len(values),
            "positive_samples":len(positive),"above_paco_count":int(values.gt(paco).sum()),
            "fraction_above_paco_all_finite":float(values.gt(paco).mean()) if len(values) else np.nan,
            "fraction_above_paco_positive":float(positive.gt(paco).mean()) if len(positive) else np.nan,
            "p95_w":values.quantile(.95),"p99_w":values.quantile(.99),"p999_w":values.quantile(.999),
            "maximum_w":values.max(),"dominant_high_band_lower_w":mode,"dominant_high_band_upper_w":mode+bin_width_w,
            "dominant_high_band_count":int(counts.loc[mode]) if len(counts) else 0,
            "highest_repeated_band_lower_w":float(repeated.index.max()) if len(repeated) else np.nan,
            "minimum_repeats":minimum_repeats,"histogram_width_w":bin_width_w,
            "longest_dominant_band_run_samples":int(runs.max()) if len(runs) else 0}
