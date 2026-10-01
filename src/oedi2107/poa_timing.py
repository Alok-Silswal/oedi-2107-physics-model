"""Fixed-set POA timing diagnostics; never changes source measurements or clocks."""

import numpy as np
import pandas as pd

from .module import calculate_module_dc

OFFSETS_MINUTES = (-15, -10, -5, 0, 5, 10, 15)


def offset_expected_current(candidates, source_poa, source_qc, parameters, *, parallel_strings=6):
    """Electrical t + exact POA(t+offset) + frozen cell temperature -> diagnostic Idc.

    Positive offset uses a later POA label; no source index/value is modified.
    Missing, nonpositive or QC-flagged POA has no diagnostic prediction. Faiman
    is not rerun: temperature stays at electrical t to isolate POA timing.
    All seven alternatives are returned, including unavailable predictions.
    """
    for index in (candidates.index, source_poa.index):
        if not isinstance(index, pd.DatetimeIndex) or index.tz is not None or not index.is_unique:
            raise ValueError("Require unique original naive wall-clock timestamps")
    if not source_qc.index.equals(source_poa.index) or not pd.api.types.is_bool_dtype(source_qc) or source_qc.isna().any():
        raise ValueError("POA QC must be explicit Boolean and exactly aligned")
    if not np.isfinite(candidates.temp_cell).all():
        raise ValueError("Frozen candidate temperature must be finite")
    if not isinstance(parallel_strings, int) or parallel_strings <= 0:
        raise ValueError("parallel_strings must be a positive integer")
    result = pd.DataFrame(index=candidates.index, columns=OFFSETS_MINUTES, dtype=float)
    for offset in OFFSETS_MINUTES:
        lookup = candidates.index + pd.Timedelta(minutes=offset)
        poa = pd.Series(source_poa.reindex(lookup).to_numpy(), index=candidates.index)
        qc = source_qc.reindex(lookup)
        valid = np.isfinite(poa) & poa.gt(0) & pd.Series(qc.eq(False).to_numpy(), index=candidates.index)
        if valid.any():
            dc = calculate_module_dc(poa.loc[valid], candidates.loc[valid, "temp_cell"], parameters)
            result.loc[valid, offset] = dc.Imp * parallel_strings
    result.attrs.update(offset_definition="At electrical t use POA(t+offset); positive is later POA label",
                        temperature="original cell temperature at electrical t, frozen", calibration="none")
    return result


def summarize_current_ratio(ratio, phase):
    """Finite current ratios + original solar phase -> medians/gap/dispersion/counts."""
    if not ratio.index.equals(phase.index):
        raise ValueError("Ratios and AM/PM phase must align exactly")
    valid = np.isfinite(ratio) & phase.isin(["AM", "PM"])
    values = ratio.loc[valid]
    am = values.loc[phase.loc[valid].eq("AM")]
    pm = values.loc[phase.loc[valid].eq("PM")]
    return {"total_timestamps":len(ratio), "valid_pairs":int(valid.sum()), "excluded":int((~valid).sum()),
            "am_samples":len(am), "pm_samples":len(pm), "am_median":am.median(), "pm_median":pm.median(),
            "am_minus_pm":am.median()-pm.median(), "overall_std":values.std(ddof=1),
            "overall_iqr":values.quantile(.75)-values.quantile(.25)}
