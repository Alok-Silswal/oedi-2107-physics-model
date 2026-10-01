"""First real-data ideal DC run; source files are read-only and never cleaned."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pvlib

from oedi2107 import SystemConfig
from oedi2107.real_data import (INPUT_MAPPING, FLAGS, discover_channels, confirm_dc_units,
                               prepared_dc_baseline, measured_dc_for_inverter)
from oedi2107.validation import validate_dc


def sha256(path):
    """Source bytes → SHA256 → unchanged-input evidence."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_ranges(table, channel, limits):
    """Finite DC pairs → full-record ranges and illuminated ratios → evidence only."""
    rows = []
    for quantity, module_quantity, topology in (("voltage", "Vmp", 20), ("current", "Imp", 6), ("power", "Pmp", 120)):
        expected = table[f"expected_dc_{quantity}_ideal"]
        measured = table[f"measured_dc_{quantity}"]
        paired = np.isfinite(expected) & np.isfinite(measured)
        expected_finite, measured_finite = expected[np.isfinite(expected)], measured[np.isfinite(measured)]
        # This is an explicitly labelled irradiance slice, not a healthy-period mask.
        illuminated = paired & table.poa_w_m2.ge(600) & table[module_quantity].gt(0)
        ratio = measured[illuminated] / table.loc[illuminated, module_quantity]
        rows.append({
            "inverter_id": channel["inverter_id"], "quantity": f"dc_{quantity}",
            "valid_pairs": int(paired.sum()),
            "expected_min": expected_finite.min(), "expected_max": expected_finite.max(),
            "measured_min": measured_finite.min(), "measured_max": measured_finite.max(),
            "measured_negative_count": int((measured_finite < 0).sum()),
            "illuminated_pairs_poa_ge_600": int(illuminated.sum()),
            "measured_per_module_p10": ratio.quantile(.1),
            "measured_per_module_median": ratio.median(),
            "measured_per_module_p90": ratio.quantile(.9),
            "inferred_reference": topology,
            "above_cec_reference_count": int((measured_finite > limits[{"voltage": "Vdcmax", "current": "Idcmax", "power": "Pdco"}[quantity]]).sum()),
            "cec_reference_parameter": {"voltage": "Vdcmax", "current": "Idcmax", "power": "Pdco"}[quantity],
            "limit_is_not_a_cleaning_rule": True,
        })
    return rows


def plot_review(table, output):
    """Named review windows → DC traces/scatter → standalone PNGs; no data edits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = ("2018-06-15", "2021-01-15", "2023-06-15")
    identifier = table.inverter_id.iloc[0]
    for date in windows:
        subset = table.loc[date:date + " 23:59:59"]
        fig, axes = plt.subplots(3, 2, figsize=(12, 9))
        for row, (quantity, unit) in enumerate((("voltage", "V"), ("current", "A"), ("power", "W"))):
            expected = subset[f"expected_dc_{quantity}_ideal"]
            measured = subset[f"measured_dc_{quantity}"]
            axes[row, 0].plot(subset.index, expected, label="Ideal model", linewidth=1)
            axes[row, 0].plot(subset.index, measured, label="Measured", linewidth=1)
            axes[row, 0].set_ylabel(f"DC {quantity} [{unit}]")
            finite = np.isfinite(expected) & np.isfinite(measured)
            axes[row, 1].scatter(expected[finite], measured[finite], s=6, alpha=.4)
            if finite.any():
                low = min(expected[finite].min(), measured[finite].min())
                high = max(expected[finite].max(), measured[finite].max())
                axes[row, 1].plot([low, high], [low, high], "k--", linewidth=.6)
            axes[row, 1].set_xlabel(f"Expected ideal [{unit}]")
            axes[row, 1].set_ylabel(f"Measured [{unit}]")
            axes[row, 0].legend(fontsize=8)
        fig.suptitle(f"{identifier}: {date}, Pacific wall clock; all QC retained, uncalibrated")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / f"{identifier}_{date}_dc.png", dpi=130)
        plt.close(fig)


def main():
    """Prepared Parquet → Blocks 1–3 baseline → per-inverter diagnostics/artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/2107/processed/2107_model_5min.parquet"))
    parser.add_argument("--metadata", type=Path, default=Path("data/2107/raw/2107_system_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/2107_dc_ideal"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.input, args.metadata]
    before = {str(path): sha256(path) for path in paths}
    source = pq.ParquetFile(args.input)
    mapping = discover_channels(source.schema_arrow.names)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    mapping = confirm_dc_units(mapping, metadata)
    mapping.to_csv(args.output / "channel_mapping.csv", index=False)
    prepared = pd.read_parquet(args.input, columns=["measured_on", *INPUT_MAPPING, *FLAGS, "wind_direction_deg"])
    config = SystemConfig()
    baseline = prepared_dc_baseline(prepared, config)
    assert baseline.index.tz is None and len(baseline) == source.metadata.num_rows
    baseline.to_parquet(args.output / "nominal_dc_ideal.parquet")
    limits = pvlib.pvsystem.retrieve_sam("cecinverter")[config.cec_inverter_entry]
    summaries, ranges = [], []
    writer = None
    try:
        for channel in mapping.to_dict("records"):
            readings = pd.read_parquet(args.input, columns=["measured_on", channel["dc_voltage"], channel["dc_current"]])
            table = measured_dc_for_inverter(baseline, readings, channel)
            if channel["inverter_id"] == "inv_05":
                assert table.measured_dc_voltage.isna().all() and table.measured_dc_power.isna().all()
            expected = table[["inverter_id", *[f"expected_dc_{q}_ideal" for q in ("voltage", "current", "power")]]].rename(
                columns={f"expected_dc_{q}_ideal": f"expected_dc_{q}" for q in ("voltage", "current", "power")})
            measured = table[["inverter_id", "measured_dc_voltage", "measured_dc_current", "measured_dc_power"]]
            validation = validate_dc(expected, measured, timestamp_basis="pacific_wall_clock")
            summary = validation.metrics.drop(columns=["normalized_mae", "normalized_rmse", "normalization_basis", "normalization_value"])
            summaries.append(summary)
            del validation
            ranges.extend(diagnostic_ranges(table, channel, limits))
            exported = table.reset_index()
            exported.attrs = {}  # Full shared/model/channel provenance is in the sidecars.
            arrow = pa.Table.from_pandas(exported, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output / "inverter_dc_ideal.parquet", arrow.schema, compression="snappy")
            writer.write_table(arrow, row_group_size=100000)
            if channel["inverter_id"] in ("inv_01", "inv_05", "inv_04", "inv_24"):
                plot_review(table, args.output)
            logging.info("Completed %s: %s", channel["inverter_id"], summary[["quantity", "valid_paired_samples", "mae"]].to_dict("records"))
            del table, exported, arrow, expected, measured, readings
    finally:
        if writer is not None:
            writer.close()
    pd.concat(summaries, ignore_index=True).to_csv(args.output / "dc_metrics.csv", index=False)
    pd.DataFrame(ranges).to_csv(args.output / "dc_ranges_and_topology.csv", index=False)
    after = {str(path): sha256(path) for path in paths}
    if before != after:
        raise ValueError("Source files changed during execution")
    expected_records = len(baseline) * 24
    if pq.ParquetFile(args.output / "inverter_dc_ideal.parquet").metadata.num_rows != expected_records:
        raise ValueError("Output inverter record count mismatch")
    report = {
        "source_sha256": before, "source_hashes_unchanged": True,
        "source_schema": source.schema_arrow.names,
        "rows": len(baseline), "inverter_records": expected_records,
        "first_timestamp": str(baseline.index[0]), "last_timestamp": str(baseline.index[-1]),
        "model_inputs_available": int(baseline.model_inputs_available.sum()),
        "dc_model_computed": int(baseline.dc_model_computed.sum()),
        "model_status_counts": baseline.dc_model_status.value_counts().to_dict(),
        "flag_counts": {flag: int(baseline[flag].sum()) for flag in FLAGS},
        "nominal_ranges": baseline[["temp_cell", "expected_dc_voltage_ideal", "expected_dc_current_ideal", "expected_dc_power_ideal"]].agg(["min", "max"]).to_dict(),
        "physics_configuration": {"cec_module_entry": config.cec_module_entry, "u0": config.faiman.u0, "u1": config.faiman.u1,
                                  "modules_per_string": 20, "strings_per_inverter": 6, "topology": "inferred", "k_dc": None},
        "pvlib_version": pvlib.__version__, "timestamp_basis": "Pacific local wall clock, unresolved fall DST; no UTC conversion",
        "dc_units": "measured V and A from metadata, unchanged prepared readings; V × A = W",
        "limits_provenance_only": {"cec_inverter_entry": config.cec_inverter_entry, "Idcmax": float(limits.Idcmax), "Vdcmax": float(limits.Vdcmax), "Pdco": float(limits.Pdco)},
        "diagnostic_slice": "POA >= 600 W/m² and finite comparisons; not a healthy mask or calibration split",
        "normalization": "none", "calibration": "none", "qc_exclusions": "none",
        "plots": {"inverters": ["inv_01", "inv_04", "inv_05", "inv_24"], "dates": ["2018-06-15", "2021-01-15", "2023-06-15"], "selection": "illustrative seasonal/year slices, not inferred healthy periods"},
    }
    (args.output / "run_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "inverter_records", "dc_model_computed", "source_hashes_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
