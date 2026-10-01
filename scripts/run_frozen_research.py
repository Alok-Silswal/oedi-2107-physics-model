"""Complete frozen research run, validation only; no tuning or data cleaning."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oedi2107.config import SystemConfig
from oedi2107.real_data import INPUT_MAPPING, FLAGS, discover_channels, measured_dc_for_inverter
from oedi2107.research import research_baseline, meter_intervals, complete_power_scales
from oedi2107.validation import mae, rmse, mbe, r_squared


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def metrics(expected, measured, mask):
    """Explicit population -> finite aligned pairs -> counts/errors and pool sums."""
    e, m = expected.loc[mask], measured.loc[mask]
    valid = np.isfinite(e) & np.isfinite(m)
    e, m = e.loc[valid], m.loc[valid]
    residual = m - e
    return {"total_samples": int(mask.sum()), "valid_pairs": len(e),
            "excluded_samples": int(mask.sum()) - len(e), "mae": mae(e, m),
            "rmse": rmse(e, m), "mbe": mbe(e, m), "r_squared": r_squared(e, m)}, np.array([
                mask.sum(), len(e), residual.abs().sum(), (residual ** 2).sum(),
                residual.sum(), m.sum(), (m ** 2).sum(),
            ], dtype=float)


def main():
    source = Path("data/2107/processed/2107_model_5min.parquet")
    meter_path = Path("data/2107/processed/2107_meter_clean.parquet")
    meter_qc = Path("data/2107/processed/2107_meter_qc_events.parquet")
    candidates = Path("outputs/2107_dc_candidates/inverter_candidate_masks.parquet")
    output = Path("outputs/2107_research_baseline")
    output.mkdir(parents=True, exist_ok=True)
    paths = [source, meter_path, meter_qc, candidates]
    before = {str(path): digest(path) for path in paths}
    mapping = discover_channels(pq.ParquetFile(source).schema_arrow.names)
    metadata = json.loads(Path("data/2107/raw/2107_system_metadata.json").read_text(encoding="utf-8-sig"))
    for channel in mapping.to_dict("records"):
        for quantity, unit in (("dc_voltage", "v"), ("dc_current", "a"), ("ac_power", "kw")):
            metric = channel[quantity].rsplit("_", 1)[-1]
            matches = [v for v in metadata["Metrics"].values() if str(v.get("metric_id")) == metric]
            if len(matches) != 1 or matches[0]["units"].lower() != unit or matches[0]["calc_scale"] != 1 or matches[0]["calc_offset"] != 0:
                raise ValueError(f"Unconfirmed units: {channel[quantity]}")
    prepared = pd.read_parquet(source, columns=["measured_on", *INPUT_MAPPING, *FLAGS, "plant_ac_power_kw"])
    baseline = research_baseline(prepared)
    baseline.to_parquet(output / "nominal_model.parquet")
    daylight = (baseline.poa_w_m2 > 0) & ~baseline.electrical_qc_flag & ~baseline.irradiance_qc_flag & ~baseline.dst_fall_ambiguous
    rows, pools, expected_arrays = [], {}, []
    writer = None
    try:
        for channel in mapping.to_dict("records"):
            identifier = channel["inverter_id"]
            readings = pd.read_parquet(source, columns=["measured_on", channel["dc_voltage"], channel["dc_current"], channel["ac_power"]])
            table = measured_dc_for_inverter(baseline, readings, channel)
            table["measured_ac_power"] = readings[channel["ac_power"]].to_numpy() * 1000.
            candidate = pd.read_parquet(candidates, filters=[("inverter_id", "==", identifier)]).set_index("timestamp")
            if not candidate.index.equals(table.index):
                raise ValueError("Existing candidate timestamps differ from model input")
            table["candidate_healthy_mask"] = candidate.candidate_healthy_mask
            table["candidate_dc_power_eligible"] = candidate.dc_power_eligible
            table["qc_clear_daytime"] = daylight
            if identifier == "inv_05" and not table.measured_dc_voltage.isna().all():
                raise ValueError("Inverter 05 measured Vdc must remain unavailable")
            for quantity in ("dc_voltage", "dc_current", "dc_power", "ac_power"):
                table[f"residual_{quantity}"] = table[f"measured_{quantity}"] - table[f"expected_{quantity}"]
                for population, mask in (("all_pairs", pd.Series(True, index=table.index)),
                                         ("qc_clear_daytime", daylight),
                                         ("candidate_healthy", table.candidate_healthy_mask)):
                    values, sums = metrics(table[f"expected_{quantity}"], table[f"measured_{quantity}"], mask)
                    rows.append({"inverter_id": identifier, "population": population, "quantity": quantity, **values})
                    pools.setdefault((population, quantity), np.zeros(7))[:] += sums
            exported = table.reset_index()
            exported.attrs = {}
            arrow = pa.Table.from_pandas(exported, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output / "inverter_outputs.parquet", arrow.schema, compression="snappy")
            writer.write_table(arrow, row_group_size=100000)
            expected_arrays.append(table.expected_ac_power.rename(identifier))
            print(f"Saved {identifier}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    pd.DataFrame(rows).to_csv(output / "per_inverter_metrics.csv", index=False)
    fleet = []
    for (population, quantity), (total, count, absolute, squared, signed, measured_sum, measured_squared) in pools.items():
        sst = measured_squared - measured_sum ** 2 / count
        fleet.append({"population": population, "quantity": quantity, "total_samples": int(total),
                      "valid_pairs": int(count), "excluded_samples": int(total - count),
                      "mae": absolute / count, "rmse": np.sqrt(squared / count),
                      "mbe": signed / count, "r_squared": 1 - squared / sst if sst > 0 else np.nan})
    pd.DataFrame(fleet).to_csv(output / "fleet_metrics.csv", index=False)
    wide = pd.concat(expected_arrays, axis=1)
    if wide.shape[1] != 24 or not wide.columns.is_unique:
        raise ValueError("Plant requires exactly 24 distinct inverter predictions")
    plant = pd.DataFrame({"expected_plant_ac_power": wide.sum(axis=1, min_count=24),
                          "measured_plant_ac_power": prepared.plant_ac_power_kw.to_numpy() * 1000.,
                          "finite_inverter_predictions": np.isfinite(wide).sum(axis=1)})
    plant["model_available"] = plant.finite_inverter_predictions.eq(24)
    plant["residual_ac_power"] = plant.measured_plant_ac_power - plant.expected_plant_ac_power
    plant.to_parquet(output / "plant_5min.parquet")
    plant_metrics = []
    timestamp_candidates = pd.read_parquet("outputs/2107_dc_candidates/timestamp_candidate_masks.parquet")
    for population, mask in (("all_pairs", pd.Series(True, index=plant.index)), ("qc_clear_daytime", daylight),
                             ("candidate_healthy", timestamp_candidates.candidate_healthy_mask.reindex(plant.index))):
        values, _ = metrics(plant.expected_plant_ac_power, plant.measured_plant_ac_power, mask)
        plant_metrics.append({"population": population, "quantity": "plant_inverter_sum_W", **values})
    pd.DataFrame(plant_metrics).to_csv(output / "plant_inverter_sum_metrics.csv", index=False)
    meter = pd.read_parquet(meter_path)
    meter.index = pd.DatetimeIndex(pd.to_datetime(meter.pop("measured_on")), name="timestamp")
    meter = meter.loc[(meter.index >= plant.index.min()) & (meter.index <= plant.index.max())]
    flagged = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(meter_qc).measured_on))
    intervals = meter_intervals(plant, meter, flagged)
    intervals.to_parquet(output / "meter_15min.parquet")
    scale_metrics = []
    for scale, table in [("15min", intervals)] + [(freq, complete_power_scales(intervals, freq)) for freq in ("h", "D", "MS")]:
        if scale != "15min":
            table.to_parquet(output / f"meter_{scale}.parquet")
        for population, mask in (("all_pairs", pd.Series(True, index=table.index)), ("meter_qc_clear", table.meter_qc_clear)):
            values, _ = metrics(table.expected_plant_ac_power_kw, table.meter_ac_power_kw, mask)
            scale_metrics.append({"scale": scale, "population": population, "quantity": "average_power_kw", **values})
            if scale != "15min":
                values, _ = metrics(table.expected_energy_kwh, table.measured_energy_kwh, mask)
                scale_metrics.append({"scale": scale, "population": population, "quantity": "energy_kwh", **values})
    pd.DataFrame(scale_metrics).to_csv(output / "meter_scale_metrics.csv", index=False)
    write_paired_scale_metrics(intervals, output)
    after = {str(path): digest(path) for path in paths}
    if before != after:
        raise ValueError("Input source changed")
    provenance = {"configuration": asdict(SystemConfig()), "frozen_research_policy": baseline.attrs,
                  "source_hashes": before, "sources_unchanged": True,
                  "rows": len(baseline), "model_available_rows": int(baseline.model_available.sum()),
                  "inverter_rows": len(baseline) * 24, "qc_daytime_timestamps": int(daylight.sum()),
                  "candidate_timestamps": int(timestamp_candidates.candidate_healthy_mask.sum()),
                  "normalization": "none", "fleet_metrics": "pooled inverter observations, not mean inverter scores",
                  "meter": "start t, average model t/t+5/t+10; W/1000 to kW; no meter*0.25",
                  "scales": "strict complete nominal wall-clock bins; average kW and integrated kWh; unresolved DST",
                  "calibration": "none", "residual_sign": "measured minus expected"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(fleet).to_string(index=False))
    print(pd.DataFrame(scale_metrics).to_string(index=False))


def write_paired_scale_metrics(intervals, output):
    """Saved aligned pairs -> observed-support summaries, not full-period totals."""
    rows = []
    for frequency in ("h", "D", "MS"):
        for population, data in (("all_pairs", intervals),
                                 ("meter_qc_clear", intervals.loc[intervals.meter_qc_clear])):
            table = complete_power_scales(data, frequency)
            table.to_parquet(output / f"meter_{frequency}_{population}_observed_support.parquet")
            mask = pd.Series(True, index=table.index)
            values, _ = metrics(table.paired_expected_average_kw, table.paired_measured_average_kw, mask)
            rows.append({"scale": frequency, "population": population,
                         "quantity": "paired_observed_average_power_kw", **values})
            values, _ = metrics(table.observed_expected_energy_kwh, table.observed_measured_energy_kwh, mask)
            rows.append({"scale": frequency, "population": population,
                         "quantity": "paired_observed_energy_kwh_not_full_period", **values})
    pd.DataFrame(rows).to_csv(output / "meter_observed_support_metrics.csv", index=False)


if __name__ == "__main__":
    main()
