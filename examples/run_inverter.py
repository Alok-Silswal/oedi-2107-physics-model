"""Standalone Block 5 demo with synthetic post-loss DC, not a selected K_DC."""

import logging

import pandas as pd

from oedi2107 import SystemConfig, load_cec_inverter, calculate_inverter_ac


def main():
    """Synthetic post-loss DC → official CEC/Sandia → printed single-inverter AC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parameters = load_cec_inverter(SystemConfig().cec_inverter_entry)
    dc = pd.DataFrame({
        "expected_dc_voltage": [0., 715., 715., 715., 715.],
        "expected_dc_power": [0., 1., 10000., 20000., 40000.],
    }, index=pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"))
    print("SYNTHETIC post-loss DC, one inverter; no K_DC selected or calibrated")
    print(calculate_inverter_ac(dc, parameters).to_string())


if __name__ == "__main__":
    main()
