"""Block 4: one static, plant-wide aggregate DC power derate."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd

IDEAL_COLUMNS = (
    "expected_dc_voltage_ideal",
    "expected_dc_current_ideal",
    "expected_dc_power_ideal",
)


@dataclass(frozen=True)
class DCLossConfig:
    """Dimensionless K_DC shared by all 24 arrays; None means uncalibrated.

    No numerical factor is selected here. A future value must represent only
    normal static baseline DC losses, not temperature, inverter conversion or
    clipping, availability, faults, abnormal underperformance or degradation.
    """

    k_dc: float | None = None

    def __post_init__(self):
        if self.k_dc is not None and (
            isinstance(self.k_dc, bool)
            or not isinstance(self.k_dc, Real)
            or not math.isfinite(self.k_dc)
            or not 0 < self.k_dc <= 1
        ):
            raise ValueError("K_DC must be one finite scalar with 0 < K_DC <= 1, or None (uncalibrated)")


def apply_dc_losses(ideal_dc: pd.DataFrame, config: DCLossConfig) -> pd.DataFrame:
    """Ideal array V [V], I [A], P [W] + K_DC → aggregate derate → DC table.

    P = K_DC * P_ideal; V = V_ideal; I = P / V for V > 0.
    At V = 0, consistent ideal inputs require P = 0 and output I = 0.
    Current is an accounting equivalent, not a simulated lossy I-V curve.
    Preserves every input column, including all three *_ideal quantities,
    timestamps and metadata, without modifying the input. Requires a supplied
    numerical factor; no default loss, calibration or per-inverter fit is used.
    """
    if config.k_dc is None:
        raise ValueError("K_DC is unset/uncalibrated; supply a reviewed factor before applying DC losses")
    missing = set(IDEAL_COLUMNS) - set(ideal_dc.columns)
    if missing:
        raise ValueError(f"Missing ideal DC columns: {sorted(missing)}")
    values = ideal_dc.loc[:, list(IDEAL_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Ideal DC voltage, current and power must be finite and nonnegative")
    voltage, current, power = values.T
    if not np.allclose(power, voltage * current, rtol=1e-10, atol=1e-9):
        raise ValueError("Ideal DC power must equal ideal voltage times current")
    if ((voltage == 0) & (power != 0)).any():
        raise ValueError("Nonzero ideal power at zero voltage is inconsistent")
    result = ideal_dc.copy(deep=True)
    result["dc_loss_factor"] = float(config.k_dc)
    result["expected_dc_voltage"] = result["expected_dc_voltage_ideal"]
    result["expected_dc_power"] = result["expected_dc_power_ideal"] * config.k_dc
    result["expected_dc_current"] = np.divide(
        result["expected_dc_power"].to_numpy(), voltage,
        out=np.zeros_like(voltage), where=voltage > 0,
    )
    result.attrs.update(dc_loss_factor=float(config.k_dc), dc_loss_status="supplied",
                        dc_loss_scope="plant-wide static factor; same for every inverter array")
    return result
