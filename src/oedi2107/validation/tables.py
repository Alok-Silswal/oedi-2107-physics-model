"""Block 7 foundation: exact-key comparisons, explicit masks and counted metrics."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import Normalization, mae, rmse, mbe, normalized_mae, normalized_rmse, r_squared


@dataclass(frozen=True)
class ValidationResult:
    """All union-key comparison rows and a counted, per-signal/per-device summary."""

    table: pd.DataFrame
    metrics: pd.DataFrame


def _keyed(data: pd.DataFrame, column: str, by_inverter: bool) -> pd.DataFrame:
    """Signal table → unique exact timestamp/device keys → one-signal frame."""
    if column not in data:
        raise ValueError(f"Required comparison column missing: {column}")
    times = data["timestamp"] if "timestamp" in data else data.index
    if not isinstance(times.dtype, pd.DatetimeTZDtype):
        raise ValueError("Comparison timestamps must already be timezone-aware datetimes")
    if pd.isna(times).any():
        raise ValueError("Comparison timestamps cannot be missing")
    index = pd.DatetimeIndex(times, name="timestamp")
    if by_inverter:
        if "inverter_id" not in data:
            raise ValueError("Inverter-level comparison requires inverter_id")
        ids = data.inverter_id
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids):
            raise ValueError("Comparison inverter IDs must be nonempty strings")
        index = pd.MultiIndex.from_arrays([index, ids], names=["timestamp", "inverter_id"])
    if not index.is_unique:
        raise ValueError("Duplicate comparison timestamp/inverter keys are not allowed")
    return pd.DataFrame({"value": data[column].to_numpy(dtype=float, na_value=np.nan)}, index=index)


def _annotations(series: pd.Series, index: pd.Index, boolean: bool) -> pd.Series:
    """Caller labels/mask → exact-key coverage and type checks → aligned values.

    Timestamp-only annotations explicitly apply to every inverter at that time;
    MultiIndex annotations apply to exact timestamp/inverter pairs. No missing
    label or mask is filled. Broader annotation indexes may cover other periods.
    """
    if not isinstance(series, pd.Series) or not series.index.is_unique:
        raise ValueError("Labels/masks must be Series with unique explicit keys")
    if isinstance(series.index, pd.MultiIndex) and series.index.names != ["timestamp", "inverter_id"]:
        raise ValueError("Device annotations need a (timestamp, inverter_id) MultiIndex")
    times = series.index.get_level_values("timestamp") if isinstance(series.index, pd.MultiIndex) else series.index
    target_times = index.get_level_values("timestamp")
    if not isinstance(times, pd.DatetimeIndex) or times.tz is None or times.hasnans:
        raise ValueError("Annotation timestamps must be valid timezone-aware datetimes")
    if str(times.tz) != str(target_times.tz):
        raise ValueError("Annotation timezones must match comparison timestamps explicitly")
    if isinstance(series.index, pd.MultiIndex):
        if series.index.names != ["timestamp", "inverter_id"] or not isinstance(index, pd.MultiIndex):
            raise ValueError("Device annotations need a (timestamp, inverter_id) MultiIndex")
        result = series.reindex(index)
    else:
        result = pd.Series(series.reindex(target_times).to_numpy(), index=index)
    if result.isna().any():
        raise ValueError("Labels/masks must cover every comparison union key; missing annotations are not inferred")
    if boolean:
        if any(not isinstance(value, (bool, np.bool_)) for value in result):
            raise ValueError("Masks must contain explicit boolean values")
        return result.astype(bool)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError("Period labels must be nonempty strings")
    return result


def _validate(
    expected: pd.DataFrame, measured: pd.DataFrame,
    signals: Mapping[str, tuple[str, str, str]], by_inverter: bool,
    normalizations: Mapping[str, Normalization] | None = None,
    period_labels: pd.Series | None = None,
    period_masks: Mapping[str, pd.Series] | None = None,
    periods: Sequence[str] | None = None,
    healthy_mask: pd.Series | None = None,
) -> ValidationResult:
    """Exact outer-key pairs → explicit finite/period/healthy selection → reports."""
    if period_labels is not None and period_masks is not None:
        raise ValueError("Supply period_labels or period_masks, not both")
    normalizations = normalizations or {}
    if set(normalizations) - set(signals):
        raise ValueError("Normalization keys must match the requested signal names")
    if any(not isinstance(value, Normalization) for value in normalizations.values()):
        raise ValueError("Use explicit Normalization(value, basis) objects")
    if periods is not None and (isinstance(periods, str) or not periods or any(not isinstance(p, str) for p in periods)):
        raise ValueError("periods must be a nonempty sequence of explicit labels")
    frames, summaries = [], []
    for quantity, (expected_column, measured_column, unit) in signals.items():
        left = _keyed(expected, expected_column, by_inverter).rename(columns={"value": "expected"})
        right = _keyed(measured, measured_column, by_inverter).rename(columns={"value": "measured"})
        left_times, right_times = left.index.get_level_values("timestamp"), right.index.get_level_values("timestamp")
        if str(left_times.tz) != str(right_times.tz):
            raise ValueError("Expected/measured timezones must match explicitly; no automatic conversion")
        left["expected_record_present"] = True
        right["measured_record_present"] = True
        pairs = left.join(right, how="outer").sort_index()
        # False here describes absence of a record, never a substituted reading.
        for column in ("expected_record_present", "measured_record_present"):
            pairs[column] = pairs[column].eq(True)
        pairs["finite_pair"] = np.isfinite(pairs.expected) & np.isfinite(pairs.measured)
        pairs["period_label"] = "unlabelled"
        if period_labels is not None:
            pairs["period_label"] = _annotations(period_labels, pairs.index, False)
        if period_masks is not None:
            occupied = pd.Series(False, index=pairs.index)
            for label, mask in period_masks.items():
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("Period-mask labels must be nonempty strings")
                aligned = _annotations(mask, pairs.index, True)
                if (occupied & aligned).any():
                    raise ValueError("Period masks overlap; choose unambiguous labels externally")
                pairs.loc[aligned, "period_label"] = label
                occupied |= aligned
        if periods is not None and period_labels is None and period_masks is None:
            raise ValueError("Period selection requires externally supplied labels or masks")
        pairs["period_selected"] = True if periods is None else pairs.period_label.isin(periods)
        pairs["healthy"] = True if healthy_mask is None else _annotations(healthy_mask, pairs.index, True)
        pairs["valid_pair"] = pairs.finite_pair & pairs.period_selected & pairs.healthy
        pairs["residual"] = np.nan
        finite = pairs.loc[pairs.finite_pair]
        pairs.loc[pairs.finite_pair, "residual"] = finite.measured - finite.expected
        pairs["quantity"], pairs["unit"] = quantity, unit
        pairs = pairs.reset_index()
        normalization = normalizations.get(quantity)
        groups = pairs.groupby("inverter_id", sort=True) if by_inverter else [(None, pairs)]
        for inverter_id, group in groups:
            selected = group.loc[group.valid_pair]
            n = len(selected)
            metrics = {"quantity": quantity, "unit": unit,
                       "total_samples": len(group), "valid_paired_samples": n,
                       "excluded_samples": len(group) - n,
                       "finite_paired_samples": int(group.finite_pair.sum()),
                       "missing_or_nonfinite_samples": int((~group.finite_pair).sum()),
                       "excluded_by_masks_samples": int((group.finite_pair & ~group.valid_pair).sum()),
                       "missing_expected_records": int((~group.expected_record_present).sum()),
                       "missing_measured_records": int((~group.measured_record_present).sum()),
                       "mae": mae(selected.expected, selected.measured),
                       "rmse": rmse(selected.expected, selected.measured),
                       "mbe": mbe(selected.expected, selected.measured),
                       "r_squared": r_squared(selected.expected, selected.measured),
                       "normalized_mae": np.nan if normalization is None else normalized_mae(selected.expected, selected.measured, normalization),
                       "normalized_rmse": np.nan if normalization is None else normalized_rmse(selected.expected, selected.measured, normalization),
                       "normalization_basis": None if normalization is None else normalization.basis,
                       "normalization_value": np.nan if normalization is None else normalization.value,
                       "metric_status": "computed" if n else "no valid pairs"}
            if by_inverter:
                metrics["inverter_id"] = inverter_id
            summaries.append(metrics)
        frames.append(pairs)
    table = pd.concat(frames, ignore_index=True)
    table.attrs.update(alignment="outer union of exact keys; no resampling/filling/interpolation",
                       residual_sign="measured - expected", healthy_mask_supplied=healthy_mask is not None,
                       selected_periods=None if periods is None else tuple(periods),
                       expected_provenance=dict(expected.attrs), measured_provenance=dict(measured.attrs))
    return ValidationResult(table=table, metrics=pd.DataFrame(summaries))


def validate_dc(expected: pd.DataFrame, measured: pd.DataFrame, **options) -> ValidationResult:
    """Named post-loss DC predictions + measurements → exact pairs → V/I/P reports.

    Both tables need timestamp (column or aware index) and inverter_id. Measured
    columns: measured_dc_voltage [V], measured_dc_current [A], measured_dc_power
    [W]. No power is inferred from measured voltage/current.
    """
    return _validate(expected, measured, {
        "dc_voltage": ("expected_dc_voltage", "measured_dc_voltage", "V"),
        "dc_current": ("expected_dc_current", "measured_dc_current", "A"),
        "dc_power": ("expected_dc_power", "measured_dc_power", "W"),
    }, True, **options)


def validate_inverter_ac(expected: pd.DataFrame, measured: pd.DataFrame, **options) -> ValidationResult:
    """Named AC predictions + measured_ac_power [W] → exact pairs → per-ID reports."""
    return _validate(expected, measured, {"ac_power": ("expected_ac_power", "measured_ac_power", "W")}, True, **options)


def validate_plant_ac(expected: pd.DataFrame, measured: pd.DataFrame, **options) -> ValidationResult:
    """Plant AC + measured_meter_power [W] → exact-time pairs → meter-level report.

    Power inputs must already have reviewed units/semantics. No interpretation
    or conversion of energy, averaging intervals, sign or timezones is made.
    """
    return _validate(expected, measured, {"plant_ac_power": ("expected_plant_ac_power", "measured_meter_power", "W")}, False, **options)
