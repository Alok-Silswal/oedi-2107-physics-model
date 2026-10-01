"""Block 6: named inverter records and strict same-timestamp AC aggregation."""

from dataclasses import dataclass
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import SystemConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlantPrediction:
    """Both model output levels; powers are W and timestamps are unchanged."""

    inverter_outputs: pd.DataFrame
    plant_outputs: pd.DataFrame


def _inverter_ids(config: SystemConfig, inverter_ids: Sequence[str] | None) -> tuple[str, ...]:
    """Configured count and optional IDs → roster checks → unique named roster."""
    if isinstance(inverter_ids, str):
        raise ValueError("Supply a sequence of inverter IDs, not a single string")
    ids = tuple(inverter_ids) if inverter_ids is not None else tuple(
        f"inverter_{number:02d}" for number in range(1, config.inverter_count + 1)
    )
    if len(ids) != config.inverter_count:
        raise ValueError(f"Complete plant prediction requires exactly {config.inverter_count} inverter IDs")
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids):
        raise ValueError("Inverter IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Inverter IDs must be unique")
    return ids


def _timestamps(index: pd.Index) -> pd.DatetimeIndex:
    """Prediction timestamp axis → validity checks → unchanged datetime axis."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Prediction timestamps must be a timezone-aware DatetimeIndex")
    if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Prediction timestamps must be nonempty, unique, increasing and valid")
    return index


def represent_inverter_outputs(
    nominal: pd.DataFrame,
    config: SystemConfig | None = None,
    inverter_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One nominal inverter table → shared prediction per roster ID → long table.

    Returns columns timestamp, inverter_id, expected_ac_power and all original
    DC/module/weather quantities. Shared inputs/configuration deliberately yield
    identical predictions initially. IDs are placeholders unless supplied; no
    measured-device identity, separate calibration or availability is inferred.
    No temporal interpolation, resampling or missing-power fill is performed.
    """
    config = config or SystemConfig()
    ids = _inverter_ids(config, inverter_ids)
    timestamps = _timestamps(nominal.index)
    if "expected_ac_power" not in nominal:
        raise ValueError("Nominal inverter prediction needs expected_ac_power")
    if {"timestamp", "inverter_id"} & set(nominal.columns):
        raise ValueError("Nominal table cannot already contain timestamp/inverter_id columns")
    frames = []
    for identifier in ids:
        frame = nominal.copy(deep=True)
        frame.insert(0, "inverter_id", identifier)
        frames.append(frame.rename_axis("timestamp").reset_index())
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "inverter_id"], ignore_index=True
    )
    result.attrs.update(nominal.attrs, expected_inverter_ids=ids,
                        expected_timestamps=tuple(timestamps),
                        inverter_representation="shared nominal model; separate named records")
    return result


def aggregate_inverter_outputs(
    inverter_outputs: pd.DataFrame,
    config: SystemConfig | None = None,
    inverter_ids: Sequence[str] | None = None,
    expected_timestamps: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Named same-time inverter AC [W] → strict roster sum → plant AC [W].

    Duplicate, missing or unexpected inverter records raise with timestamp/IDs.
    Exactly the configured roster (24 by default) is required at every timestamp.
    NaN AC is reported with uncomputed_inverter_ids and makes the entire plant
    sum NaN; it is never zero-filled or summed as a partial plant. Infinity is
    rejected. Negative Sandia tare is summed unchanged. Expected timestamps may
    be supplied or inherited from representation metadata to detect fully absent
    timestamps; without either, only observed timestamps can be checked.
    """
    config = config or SystemConfig()
    if inverter_ids is None:
        inverter_ids = inverter_outputs.attrs.get("expected_inverter_ids")
    ids = _inverter_ids(config, inverter_ids)
    required = {"timestamp", "inverter_id", "expected_ac_power"}
    if not required.issubset(inverter_outputs.columns):
        raise ValueError(f"Missing inverter output columns: {sorted(required - set(inverter_outputs.columns))}")
    data = inverter_outputs.loc[:, ["timestamp", "inverter_id", "expected_ac_power"]].copy()
    if data.empty:
        raise ValueError("No inverter predictions to aggregate")
    timestamp_axis = pd.DatetimeIndex(data.timestamp)
    if timestamp_axis.tz is None or timestamp_axis.hasnans:
        raise ValueError("Inverter timestamps must be valid and timezone-aware")
    if data.inverter_id.isna().any():
        raise ValueError("Inverter IDs cannot be missing")
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in data.inverter_id):
        raise ValueError("Inverter IDs must be nonempty strings")
    duplicates = data.duplicated(["timestamp", "inverter_id"], keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate inverter IDs within timestamp: {data.loc[duplicates, ['timestamp', 'inverter_id']].to_dict('records')}")
    observed = pd.DatetimeIndex(data.timestamp.unique()).sort_values()
    if expected_timestamps is None and "expected_timestamps" in inverter_outputs.attrs:
        expected_timestamps = pd.DatetimeIndex(inverter_outputs.attrs["expected_timestamps"])
    if expected_timestamps is not None:
        expected = _timestamps(expected_timestamps)
        missing_times = expected.difference(observed)
        extra_times = observed.difference(expected)
        if len(missing_times) or len(extra_times):
            raise ValueError(f"Incomplete timestamp coverage: missing={list(missing_times)}, unexpected={list(extra_times)}")
    power = data.expected_ac_power.to_numpy(dtype=float)
    if np.isinf(power).any():
        raise ValueError("Expected inverter AC must be finite or explicitly uncomputed (NaN)")
    data["expected_ac_power"] = power
    records = []
    for timestamp, group in data.groupby("timestamp", sort=True):
        actual = set(group.inverter_id)
        missing, unexpected = set(ids) - actual, actual - set(ids)
        if missing or unexpected:
            raise ValueError(f"Incomplete inverter set at {timestamp}: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        uncomputed = tuple(group.loc[group.expected_ac_power.isna(), "inverter_id"])
        if uncomputed:
            LOGGER.warning("Plant AC not computed at %s; NaN inverter predictions: %s", timestamp, uncomputed)
        records.append({
            "timestamp": timestamp,
            "expected_plant_ac_power": group.expected_ac_power.sum(min_count=config.inverter_count),
            "inverter_record_count": len(group),
            "finite_ac_prediction_count": group.expected_ac_power.notna().sum(),
            "uncomputed_inverter_ids": uncomputed,
            "plant_prediction_status": "uncomputed" if uncomputed else "complete",
        })
    result = pd.DataFrame(records).set_index("timestamp")
    result.attrs.update(inverter_outputs.attrs, expected_inverter_ids=ids,
                        aggregation="sum signed inverter AC at identical timestamps; complete roster required")
    return result
