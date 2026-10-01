"""Block 3: explicit CEC adjustment followed by a single-diode solution."""

import logging
import numpy as np
import pandas as pd
import pvlib

LOGGER = logging.getLogger(__name__)
CEC_FIELDS = ("alpha_sc", "a_ref", "I_L_ref", "I_o_ref", "R_sh_ref", "R_s", "Adjust")


def load_cec_module(entry: str) -> pd.Series:
    """Exact database key → pvlib bundled CEC lookup → module parameters.

    Logs the exact entry and complete row. No fuzzy match or substitute is
    accepted. A missing entry raises with candidate keys for human review.
    """
    database = pvlib.pvsystem.retrieve_sam("CECMod")
    if entry not in database:
        candidates = [key for key in database if "310TI" in key]
        raise ValueError(f"CEC entry {entry!r} unavailable; review candidates: {candidates}")
    parameters = database[entry].copy()
    if not np.isfinite(parameters.loc[list(CEC_FIELDS)].astype(float)).all():
        raise ValueError(f"Incomplete CEC diode parameters for {entry}")
    LOGGER.info("pvlib %s; exact CEC module entry: %s\n%s", pvlib.__version__, entry,
                parameters.to_string(float_format=lambda value: f"{value:.12g}"))
    return parameters


def calculate_module_dc(poa: pd.Series, temp_cell: pd.Series, parameters: pd.Series) -> pd.DataFrame:
    """POA [W/m²] + cell [°C] + CEC row → adjusted diode → module DC.

    Uses measured POA directly as effective irradiance (no spectral or IAM
    correction). Retains adjusted diode parameters and Vmp/Voc [V], Imp/Isc
    [A], Pmp [W]. At exactly zero POA electrical outputs are explicitly zero.
    """
    if not poa.index.equals(temp_cell.index):
        raise ValueError("POA and cell-temperature indexes must match")
    if not np.isfinite(poa).all() or not np.isfinite(temp_cell).all() or (poa < 0).any():
        raise ValueError("Use finite temperatures and finite nonnegative POA")
    diode = pvlib.pvsystem.calcparams_cec(
        poa, temp_cell, **{key: float(parameters[key]) for key in CEC_FIELDS}
    )
    result = pd.DataFrame(dict(zip(
        ("photocurrent", "saturation_current", "resistance_series", "resistance_shunt", "nNsVth"), diode
    )), index=poa.index)
    columns = {"v_mp": "Vmp", "i_mp": "Imp", "p_mp": "Pmp", "v_oc": "Voc", "i_sc": "Isc"}
    result[list(columns.values())] = 0.0
    daylight = poa > 0
    if daylight.any():
        # Avoid the degenerate zero-irradiance MPP search, not its warnings.
        solution = pvlib.pvsystem.singlediode(*(value.loc[daylight] for value in diode), method="lambertw")
        for source, target in columns.items():
            result.loc[daylight, target] = solution[source]
    if not np.isfinite(result[["Vmp", "Imp", "Pmp", "Voc", "Isc"]]).all().all():
        raise ValueError("Single-diode solver returned nonfinite electrical outputs")
    return result
