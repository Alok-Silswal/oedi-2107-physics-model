"""Ideal inferred series/parallel scaling only."""

import pandas as pd
from .config import SystemConfig


def scale_module_mpp(module_dc: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    """Module Vmp/Imp/Pmp → ideal series/parallel scaling → one array DC.

    Output expected_dc_voltage [V], expected_dc_current [A], expected_dc_power
    [W] describes one inverter's DC array at MPP, not its measured operating
    point. Assumes identical modules and uniform irradiance/temperature.
    """
    return pd.DataFrame({
        "expected_dc_voltage": module_dc["Vmp"] * config.modules_per_string,
        "expected_dc_current": module_dc["Imp"] * config.strings_per_inverter,
        "expected_dc_power": module_dc["Pmp"] * config.modules_per_inverter,
    }, index=module_dc.index)
