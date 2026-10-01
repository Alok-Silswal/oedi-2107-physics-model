"""Compose through named inverter/plant AC; retain previous block entry points."""

import logging
import numpy as np
import pandas as pd
import pvlib
from .config import SystemConfig
from .inputs import validate_measurements
from .thermal import estimate_cell_temperature
from .module import load_cec_module, calculate_module_dc
from .array import scale_module_mpp
from .dc_losses import apply_dc_losses
from .inverter import load_cec_inverter, calculate_inverter_ac, inverter_provenance
from .plant import PlantPrediction, represent_inverter_outputs, aggregate_inverter_outputs
from collections.abc import Sequence


def run_dc_foundation(measurements: pd.DataFrame, config: SystemConfig | None = None, *, timestamp_basis: str = "aware") -> pd.DataFrame:
    """Measured weather → Faiman → CEC module MPP → ideal one-array DC table.

    Includes weather, temperature, diode parameters, module metrics and array
    metrics, preserving timestamps. No measured DC/AC channels enter the model.
    Explicit pacific_wall_clock mode keeps prepared local timestamps unlocalized.
    """
    config = config or SystemConfig()
    weather = validate_measurements(measurements, timestamp_basis=timestamp_basis)
    parameters = load_cec_module(config.cec_module_entry)
    logging.getLogger(__name__).warning(
        "%s; Faiman u0=%g, u1=%g (%s); module temperature used as cell proxy; POA used directly as effective irradiance",
        config.topology_provenance, config.faiman.u0, config.faiman.u1, config.faiman.provenance,
    )
    temperature = estimate_cell_temperature(weather.poa, weather.temp_air, weather.wind_speed, config.faiman)
    module_dc = calculate_module_dc(weather.poa, temperature, parameters)
    result = pd.concat([weather, temperature.rename("temp_cell"), module_dc, scale_module_mpp(module_dc, config)], axis=1)
    result.attrs.update(pvlib_version=pvlib.__version__, cec_module_entry=parameters.name,
                        topology=config.topology_provenance, faiman_provenance=config.faiman.provenance,
                        timestamp_basis=timestamp_basis)
    return result


def run_dc_model(measurements: pd.DataFrame, config: SystemConfig | None = None) -> pd.DataFrame:
    """Measured weather → ideal DC foundation → optional global derate → table.

    Retains explicit *_ideal quantities. When K_DC is unset, loss factor and
    post-loss current/power are NaN (not computed); voltage remains the ideal
    baseline. With a supplied factor, applies Block 4 and retains both stages.
    No AC/inverter modelling or plant aggregation is performed.
    """
    config = config or SystemConfig()
    ideal = run_dc_foundation(measurements, config).rename(columns={
        f"expected_dc_{quantity}": f"expected_dc_{quantity}_ideal"
        for quantity in ("voltage", "current", "power")
    })
    if config.dc_losses.k_dc is not None:
        return apply_dc_losses(ideal, config.dc_losses)
    ideal["dc_loss_factor"] = np.nan
    ideal["expected_dc_voltage"] = ideal["expected_dc_voltage_ideal"]
    ideal["expected_dc_current"] = np.nan
    ideal["expected_dc_power"] = np.nan
    ideal.attrs.update(dc_loss_factor=None, dc_loss_status="uncalibrated",
                       dc_loss_scope="plant-wide static factor; same for every inverter array")
    logging.getLogger(__name__).warning("K_DC is unset/uncalibrated; post-loss DC current and power are not computed")
    return ideal


def run_inverter_model(measurements: pd.DataFrame, config: SystemConfig | None = None) -> pd.DataFrame:
    """Weather → Blocks 1–4 DC → unchanged CEC/Sandia → one-inverter AC table.

    Retains ideal and post-loss DC. If global K_DC is unset, AC is also NaN,
    even at night, rather than bypassing the uncalibrated DC stage. Database
    identity and official parameters are retained in metadata in either case.
    No fitting, inverter aggregation or meter comparison is performed.
    """
    config = config or SystemConfig()
    parameters = load_cec_inverter(config.cec_inverter_entry)
    dc = run_dc_model(measurements, config)
    if config.dc_losses.k_dc is not None:
        return calculate_inverter_ac(dc, parameters)
    dc["expected_ac_power"] = np.nan
    dc.attrs.update(inverter_provenance(parameters), ac_status="not computed: K_DC unset/uncalibrated")
    logging.getLogger(__name__).warning("AC is not computed because K_DC is unset/uncalibrated")
    return dc


def run_plant_model(
    measurements: pd.DataFrame,
    config: SystemConfig | None = None,
    inverter_ids: Sequence[str] | None = None,
) -> PlantPrediction:
    """Shared weather/config → 24 named predictions → same-time signed plant sum.

    Returns both inverter_outputs (long format) and plant_outputs (timestamp
    index). Uses one common Blocks 1–5 configuration without per-inverter fits.
    Unset K_DC preserves uncomputed AC at both levels. No measured electrical
    joins, meter comparison, resampling or timestamp-semantics assumption.
    """
    config = config or SystemConfig()
    nominal = run_inverter_model(measurements, config)
    inverters = represent_inverter_outputs(nominal, config, inverter_ids)
    plant = aggregate_inverter_outputs(inverters, config, inverter_ids, nominal.index)
    return PlantPrediction(inverter_outputs=inverters, plant_outputs=plant)
