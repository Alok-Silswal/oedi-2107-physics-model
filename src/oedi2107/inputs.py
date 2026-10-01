"""Measured POA and weather validation; no irradiance transposition."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("poa", "temp_air", "wind_speed")


def check_timestamp_basis(index: pd.DatetimeIndex, timestamp_basis: str = "aware") -> None:
    """Explicit timestamp policy → checks → preserve the input without localization."""
    if timestamp_basis not in ("aware", "pacific_wall_clock"):
        raise ValueError("timestamp_basis must be aware or pacific_wall_clock")
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Use a DatetimeIndex")
    if timestamp_basis == "aware" and index.tz is None:
        raise ValueError("Use a timezone-aware DatetimeIndex; site timezone is unknown")
    if timestamp_basis == "pacific_wall_clock" and index.tz is not None:
        raise ValueError("Pacific wall-clock mode requires unlocalized timestamps")


def validate_measurements(data: pd.DataFrame, *, timestamp_basis: str = "aware") -> pd.DataFrame:
    """Measured POA [W/m²], air [°C], wind [m/s] → checks → numeric copy.

    Inputs need a unique, increasing DatetimeIndex. Default mode requires a
    timezone; explicit pacific_wall_clock mode preserves unlocalized source time.
    Missing/nonfinite readings, negative POA and negative wind are rejected;
    no clipping, interpolation or resampling is performed. Extra channels stay
    available to the caller but are not used in this modelling stage.
    """
    check_timestamp_basis(data.index, timestamp_basis)
    if data.empty or data.index.hasnans or not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Measurements need nonempty, unique, increasing valid timestamps")
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")
    result = data.loc[:, list(REQUIRED_COLUMNS)].astype(float).copy()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Measurements must be finite; resolve missing data explicitly")
    if (result[["poa", "wind_speed"]] < 0).any().any():
        raise ValueError("POA and wind speed must be nonnegative; review sensor offsets")
    return result
