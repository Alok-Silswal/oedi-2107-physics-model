"""Measured POA and weather validation; no irradiance transposition."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("poa", "temp_air", "wind_speed")


def validate_measurements(data: pd.DataFrame) -> pd.DataFrame:
    """Measured POA [W/m²], air [°C], wind [m/s] → checks → numeric copy.

    Inputs must have a unique, increasing, timezone-aware DatetimeIndex.
    Missing/nonfinite readings, negative POA and negative wind are rejected;
    no clipping, interpolation or resampling is performed. Extra channels stay
    available to the caller but are not used in this modelling stage.
    """
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None:
        raise ValueError("Use a timezone-aware DatetimeIndex; site timezone is unknown")
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
