"""Block 2: steady-state Faiman thermal calculation."""

import numpy as np
import pandas as pd
from pvlib.temperature import faiman

from .config import FaimanConfig


def estimate_cell_temperature(poa, temp_air, wind_speed, config: FaimanConfig):
    """POA [W/m²] + air [°C] + wind [m/s] → Faiman → temperature [°C].

    Computes T_air + POA/(u0 + u1*wind). Uses module temperature as the
    cell-temperature proxy without an invented cell/module offset. Scalars
    and identically shaped arrays are supported; Series indexes must match.
    """
    values = [np.asarray(x, dtype=float) for x in (poa, temp_air, wind_speed)]
    if len({v.shape for v in values}) != 1:
        raise ValueError("Thermal inputs must have identical shapes")
    indexes = [x.index for x in (poa, temp_air, wind_speed) if isinstance(x, pd.Series)]
    if indexes and any(not indexes[0].equals(i) for i in indexes[1:]):
        raise ValueError("Thermal Series indexes must match")
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError("Thermal inputs must be finite")
    if (values[0] < 0).any() or (values[2] < 0).any():
        raise ValueError("POA and wind speed must be nonnegative")
    return faiman(poa, temp_air, wind_speed, u0=config.u0, u1=config.u1)
