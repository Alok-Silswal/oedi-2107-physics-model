"""Reporting reads saved values, converts units and fails clearly; no model run."""

import csv
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location(
    "stage1_reporting", Path(__file__).resolve().parents[1] / "scripts/show_stage1_results.py"
)
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def test_missing_sources_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required frozen outputs missing"):
        reporting.load_results(tmp_path)


def test_saved_metrics_units_and_footer_checks(tmp_path):
    directory = tmp_path / reporting.RESULTS
    directory.mkdir(parents=True)
    (directory / "provenance.json").write_text(json.dumps({
        "configuration": {"inverter_count": 2}, "rows": 3, "inverter_rows": 6,
    }))
    for name, count in (("inverter_outputs.parquet", 6), ("plant_5min.parquet", 3), ("meter_15min.parquet", 2)):
        pq.write_table(pa.table({"row": range(count)}), directory / name)
    fleet = []
    for quantity in ("dc_voltage", "dc_current", "dc_power", "ac_power"):
        fleet.append({"population": "all_pairs" if quantity == "ac_power" else "candidate_healthy",
                      "quantity": quantity, "valid_pairs": 2, "mae": 2000., "rmse": 3000.,
                      "mbe": -1000., "r_squared": .75})
    meter = [{"population": "all_pairs", "scale": "15min", "quantity": "average_power_kw",
              "total_samples": 2, "valid_pairs": 1, "mae": 20., "rmse": 30., "mbe": -10., "r_squared": .5}]
    for name, rows in (("fleet_metrics.csv", fleet), ("meter_scale_metrics.csv", meter)):
        with (directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _, metrics = reporting.load_results(tmp_path)
    assert metrics[0]["mae"] == 2000. and metrics[0]["unit"] == "V"
    assert metrics[1]["mae"] == 2000. and metrics[1]["unit"] == "A"
    assert metrics[2]["mae"] == 2. and metrics[2]["mbe"] == -1.
    assert metrics[3]["rmse"] == 3. and metrics[3]["r2"] == .75
    assert metrics[4]["mae"] == 20. and metrics[4]["pair_count"] == 1
    with pytest.raises(ValueError, match="found 2"):
        reporting.select([fleet[0], fleet[0]], quantity="dc_voltage")
    pq.write_table(pa.table({"row": [1]}), directory / "plant_5min.parquet")
    with pytest.raises(ValueError, match="coverage/provenance mismatch"):
        reporting.load_results(tmp_path)
