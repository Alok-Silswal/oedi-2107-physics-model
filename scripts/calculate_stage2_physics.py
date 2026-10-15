"""Saved weather forecasts -> frozen physics and exact complete-fleet correction labels."""
from dataclasses import asdict
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_stage2_readiness import indexed, digest
from oedi2107.config import SystemConfig
from oedi2107.real_data import FLAGS, INPUT_MAPPING, discover_channels
from oedi2107.research import research_baseline

OUT = Path("outputs/2107_stage2_weather")
DATA = Path("data/2107/processed")


def complete_fleet_power(ac, flagged):
    """Prepared AC is kW; require 24 finite, nonnegative, QC-clear readings."""
    if ac.shape[1] != 24 or not ac.columns.is_unique or not ac.index.is_unique:
        raise ValueError("Expected 24 distinct inverter channels and unique timestamps")
    valid = np.isfinite(ac).all(axis=1) & ac.ge(0).all(axis=1) & ~ac.index.isin(flagged)
    return ac.sum(axis=1, min_count=24).where(valid)


def physics_labels(forecasts, actual):
    targets = pd.DatetimeIndex(forecasts.target_timestamp, name="timestamp")
    if not forecasts.index.is_unique or not targets.is_unique or not targets.is_monotonic_increasing:
        raise ValueError("Forecast origin/target keys must be unique and ordered")
    if not ((targets-forecasts.index) == pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected exact 15-minute targets")
    prepared = pd.DataFrame({"measured_on": targets.strftime("%Y-%m-%d %H:%M:%S")})
    for column, variable in zip(INPUT_MAPPING, ("poa", "ambient", "wind")):
        prepared[column] = forecasts[f"forecast_{variable}"].to_numpy()
    weather = prepared[list(INPUT_MAPPING)]
    supported = np.isfinite(weather).all(axis=1) & weather.poa_w_m2.ge(0) & weather.wind_speed_m_s.ge(0)
    # Flags describe forecast inputs, never future measured weather or availability.
    for flag in FLAGS:
        prepared[flag] = supported.to_numpy() if flag in ("poa_available", "environment_available", "model_inputs_available") else False
    baseline = research_baseline(prepared)
    result = forecasts[["target_timestamp", "training_cutoff", "forecast_poa", "forecast_ambient", "forecast_wind"]].copy()
    result["physics_forecast_kw"] = baseline.expected_ac_power.to_numpy()*SystemConfig().inverter_count/1000.
    result["actual_target_power_kw"] = actual.reindex(targets).to_numpy()
    physics_valid = np.isfinite(result.physics_forecast_kw)
    actual_valid = np.isfinite(result.actual_target_power_kw)
    result["correction_kw"] = (result.actual_target_power_kw-result.physics_forecast_kw).where(physics_valid & actual_valid)
    result["validity"] = np.select([~physics_valid & ~actual_valid, ~physics_valid, ~actual_valid],
        ["physics_failed_and_actual_missing", "physics_failed", "actual_missing"], default="labelled")
    result.attrs = {}
    return result


def load_actual():
    paths = [DATA/"2107_electrical_clean.parquet", DATA/"2107_electrical_qc_events.parquet"]
    mapping = discover_channels(pq.ParquetFile(paths[0]).schema_arrow.names)
    ac = indexed(paths[0], ["measured_on", *mapping.ac_power])
    flagged = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(paths[1]).measured_on))
    return complete_fleet_power(ac, flagged), paths


def main():
    from forecast_stage2_weather import SPLIT, split_masks, load_data, prepare
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--allow-test", action="store_true", help="Confirm all pipeline choices are frozen")
    parser.add_argument("--step", choices=["training_1", "training_2", "training_3", "validation", "testing"])
    args = parser.parse_args()
    if args.step == "testing" and not args.prepare and not args.allow_test:
        parser.error("Testing requires --allow-test after all pipeline choices are frozen")
    if args.prepare or args.step is None:
        features, targets, elevation, _ = load_data()
        prepare(features, targets, elevation)
        print("Physics plan: consume compatible saved step forecasts; frozen calculation; exact QC-clear fleet labels. No physics calculation or writes in preparation.")
        return
    step = args.step
    source, weather_settings = OUT/f"forecasts_{step}.parquet", OUT/f"{step}_settings.json"
    saved = json.loads(weather_settings.read_text())
    if saved["specification"]["protocol"] != "elapsed-70-15-15-v1" or saved["specification"]["split"] != SPLIT:
        raise ValueError("Incompatible weather protocol")
    forecasts = pd.read_parquet(source)
    label = "training" if step.startswith("training") else step
    if not split_masks(forecasts.index)[label].all():
        raise ValueError("Forecast targets cross the shared split")
    actual, electrical_paths = load_actual()
    paths = [source, weather_settings, *electrical_paths]
    specification = dict(protocol="elapsed-70-15-15-v1", split=SPLIT, step=step,
        source_hashes={str(p):digest(p) for p in paths}, code_hash=digest(Path(__file__)), configuration=asdict(SystemConfig()),
        versions={p:metadata.version(p) for p in ("pvlib", "numpy", "pandas", "scipy", "pyarrow")},
        physics="research_baseline at target; predicted weather only; ideal DC; Sandia 30 kW extension; signed tare; inverter W * 24 / 1000 -> plant kW",
        actual="Exact QC-clear target; 24 finite nonnegative inverter kW readings; no filling",
        correction="actual_target_power_kw - physics_forecast_kw")
    destination, provenance = OUT/f"physics_{step}.parquet", OUT/f"physics_{step}_settings.json"
    if provenance.exists():
        if json.loads(provenance.read_text()) != specification:
            raise ValueError("Incompatible physics settings; refusing reuse")
        if destination.exists():
            print("Compatible completed physics reused:", destination)
            return
    elif destination.exists():
        raise ValueError("Physics table lacks provenance; refusing overwrite")
    result = physics_labels(forecasts, actual)
    result["split"] = label
    labelled = result.validity.eq("labelled")
    np.testing.assert_allclose(result.loc[labelled, "actual_target_power_kw"],
        result.loc[labelled, "physics_forecast_kw"]+result.loc[labelled, "correction_kw"], atol=1e-9)
    assert specification["source_hashes"] == {str(p):digest(p) for p in paths}
    result.to_parquet(destination)
    provenance.write_text(json.dumps(specification, indent=2), encoding="utf-8")
    print(result.validity.value_counts().to_string())


if __name__ == "__main__":
    main()
