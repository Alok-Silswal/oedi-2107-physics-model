"""Block 5: explicit Sandia conversion with unchanged bundled CEC parameters."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

LOGGER = logging.getLogger(__name__)
MODEL_FIELDS = ("Paco", "Pdco", "Vdco", "Pso", "C0", "C1", "C2", "C3", "Pnt")


def load_cec_inverter(entry: str) -> pd.Series:
    """Exact configured key → bundled CEC lookup → unchanged inverter row.

    Logs all ABB TRIO-27.6 candidates for review and the selected full row.
    Candidate discovery never selects a key. No fuzzy fallback or parameter
    fitting is used. The installed S1B's US/480 V and _A suffix need review.
    """
    database = pvlib.pvsystem.retrieve_sam("cecinverter")
    candidates = [key for key in database if key.startswith("ABB__TRIO_27_6_")]
    LOGGER.warning("ABB TRIO-27.6 CEC candidates (review installed voltage/suffix): %s", candidates)
    if entry not in database:
        raise ValueError(f"Exact CEC inverter entry {entry!r} unavailable; review candidates: {candidates}")
    parameters = database[entry].copy(deep=True)
    _validate_parameters(parameters)
    source = Path(pvlib.__file__).parent / "data" / "sam-library-cec-inverters-2019-03-05.csv"
    parameters.attrs.update(
        pvlib_version=pvlib.__version__, cec_inverter_entry=entry,
        inverter_database="pvlib bundled CEC inverter database",
        inverter_database_file=str(source), inverter_candidates=candidates,
        inverter_variant_review="Confirm installed US/480 V variant and _A suffix; no fuzzy selection",
    )
    LOGGER.info("pvlib %s; exact CEC inverter entry: %s; source: %s\n%s",
                pvlib.__version__, entry, source,
                parameters.to_string(float_format=lambda value: f"{value:.12g}"))
    return parameters


def _validate_parameters(parameters: pd.Series) -> None:
    """Required CEC fields → finite/rating checks → valid model parameters."""
    missing = set(MODEL_FIELDS) - set(parameters.index)
    if missing:
        raise ValueError(f"Missing Sandia inverter parameters: {sorted(missing)}")
    if not np.isfinite(parameters.loc[list(MODEL_FIELDS)].astype(float)).all():
        raise ValueError("Sandia inverter parameters must be finite")
    if any(float(parameters[key]) <= 0 for key in ("Paco", "Pdco", "Vdco")):
        raise ValueError("CEC Paco, Pdco and Vdco must be positive")
    if float(parameters.Pso) < 0 or float(parameters.Pnt) < 0:
        raise ValueError("CEC Pso and Pnt must be nonnegative")


def inverter_provenance(parameters: pd.Series) -> dict:
    """Loaded CEC row → metadata copy → database identity and full parameter row."""
    return {
        **parameters.attrs,
        "cec_inverter_entry": parameters.name,
        "inverter_model": "pvlib.inverter.sandia",
        "inverter_parameters": parameters.to_dict(),
        "inverter_parameter_policy": "official CEC row unchanged; no calibration",
        "nighttime_ac_policy": "retain signed Sandia output, including negative tare",
    }


def calculate_inverter_ac(dc: pd.DataFrame, parameters: pd.Series) -> pd.DataFrame:
    """Post-loss DC V [V] + P [W] + CEC row → Sandia → one-inverter AC [W].

    Preserves all DC columns/timestamps and adds expected_ac_power. Clipping
    and startup/tare come only from pvlib.inverter.sandia; no second cap or
    zero clamp is applied. Requires finite nonnegative DC, including (0 V, 0 W)
    at night. Uncomputed DC from unset K_DC is rejected. Sandia does not enforce
    MPPT voltage windows or maximum DC current; these remain review limitations.
    """
    _validate_parameters(parameters)
    required = ("expected_dc_voltage", "expected_dc_power")
    missing = set(required) - set(dc.columns)
    if missing:
        raise ValueError(f"Missing post-loss DC columns: {sorted(missing)}")
    values = dc.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Post-loss DC must be finite and nonnegative; K_DC may still be unset/uncalibrated")
    if ((values[:, 0] == 0) & (values[:, 1] > 0)).any():
        raise ValueError("Positive DC power requires positive DC voltage")
    ac = pvlib.inverter.sandia(dc.expected_dc_voltage, dc.expected_dc_power, parameters)
    if not np.isfinite(ac).all():
        raise ValueError("Sandia returned nonfinite AC power for supplied operating inputs")
    result = dc.copy(deep=True)
    result["expected_ac_power"] = ac
    result.attrs.update(inverter_provenance(parameters), ac_status="computed")
    return result


def calculate_inverter_ac_30kw_ceiling(
    dc: pd.DataFrame, parameters: pd.Series
) -> pd.DataFrame:
    """DC V/W + official CEC row -> original Sandia equation -> AC W at 30 kW.

    Project extension: Paco remains the reference rating in the efficiency
    equation; only the external ceiling is 30,000 W. No coefficients are fit.
    Preserves the ordinary pvlib baseline separately. Like pvlib, startup is
    determined by p_dc < official Pso and returns -Pnt, without a zero clamp.
    The extension does not enforce MPPT/current or thermal derating limits.
    """
    result = calculate_inverter_ac(dc, parameters).rename(
        columns={"expected_ac_power": "expected_ac_power_cec"}
    )
    voltage = dc.expected_dc_voltage
    power = dc.expected_dc_power
    delta_v = voltage - float(parameters.Vdco)
    a = float(parameters.Pdco) * (1 + float(parameters.C1) * delta_v)
    b = float(parameters.Pso) * (1 + float(parameters.C2) * delta_v)
    c = float(parameters.C0) * (1 + float(parameters.C3) * delta_v)
    pre_clip = (
        (float(parameters.Paco) / (a - b) - c * (a - b)) * (power - b)
        + c * (power - b) ** 2
    )
    extended = np.minimum(pre_clip, 30_000.0)
    extended = extended.where(power >= float(parameters.Pso), -float(parameters.Pnt))
    if not np.isfinite(extended).all():
        raise ValueError("Sandia extension returned nonfinite AC power")
    result["expected_ac_power_30kw"] = extended
    result.attrs.update(
        inverter_extension="official Sandia pre-clipping equation; external 30000 W ceiling",
        reference_paco_w=float(parameters.Paco), maximum_ac_output_w=30_000.0,
        extension_parameter_policy="official coefficients unchanged; no refit",
    )
    return result
