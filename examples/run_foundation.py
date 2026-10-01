"""Run the foundation on explicitly synthetic weather, not OEDI measurements."""

import logging
import pandas as pd
from oedi2107 import run_dc_model


def main():
    """Synthetic weather → DC foundation → printed module/array metrics."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    weather = pd.DataFrame(
        {"poa": [0., 200., 600., 1000.], "temp_air": [20., 20., 25., 25.], "wind_speed": [1., 1., 2., 2.]},
        index=pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
    )
    result = run_dc_model(weather)
    print("SYNTHETIC example; UTC is not an assumed site timezone")
    print("K_DC is unset: post-loss current/power are not computed (NaN)")
    print(result[["temp_cell", "Vmp", "Imp", "Pmp", "Voc", "Isc",
                  "expected_dc_voltage_ideal", "expected_dc_current_ideal", "expected_dc_power_ideal",
                  "dc_loss_factor", "expected_dc_voltage", "expected_dc_current", "expected_dc_power"]].to_string())


if __name__ == "__main__":
    main()
