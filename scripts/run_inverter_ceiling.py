"""Task 11 measured-only pairs -> two fixed ceilings -> concise error comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oedi2107.config import SystemConfig
from oedi2107.inverter import (
    calculate_inverter_ac_30kw_ceiling, inverter_provenance, load_cec_inverter,
)
from oedi2107.measured_conversion import conversion_metrics


def main():
    source = Path("outputs/2107_measured_conversion/inverter_measured_conversion.parquet")
    output = Path("outputs/2107_inverter_ceiling")
    output.mkdir(parents=True, exist_ok=True)
    parameters = load_cec_inverter(SystemConfig().cec_inverter_entry)
    original = parameters.copy(deep=True)
    rows = []
    pooled = {}
    columns = ["timestamp", "inverter_id", "measured_dc_voltage", "measured_dc_power",
               "measured_ac_power", "conversion_analysis_mask"]
    for number in range(1, 25):
        if number == 5:
            continue
        identifier = f"inv_{number:02d}"
        measured = pd.read_parquet(source, columns=columns,
                                   filters=[("inverter_id", "==", identifier)])
        measured = measured.loc[measured.conversion_analysis_mask].set_index("timestamp")
        dc = measured[["measured_dc_voltage", "measured_dc_power"]].rename(columns={
            "measured_dc_voltage": "expected_dc_voltage", "measured_dc_power": "expected_dc_power",
        })
        predicted = calculate_inverter_ac_30kw_ceiling(dc, parameters)
        below = (measured.measured_dc_power < .9 * float(parameters.Pdco)) & (
            measured.measured_ac_power < .9 * float(parameters.Paco))
        populations = {
            "all_selected": pd.Series(True, index=measured.index),
            "below_clipping_guards": below,
            "above_cec_ac_rating": measured.measured_ac_power > float(parameters.Paco),
            "around_30kw_ac": measured.measured_ac_power.between(29900., 30100., inclusive="left"),
        }
        for population, mask in populations.items():
            for model in ("cec", "30kw"):
                comparison = pd.DataFrame({
                    "measured_ac_power": measured.measured_ac_power,
                    "sandia_ac_from_measured_dc": predicted[f"expected_ac_power_{model}"],
                })
                error = conversion_metrics(comparison, mask)
                rows.append({"inverter_id": identifier, "population": population,
                             "model": model, **error})
                residual = (comparison.measured_ac_power - comparison.sandia_ac_from_measured_dc).loc[mask]
                key = (population, model)
                totals = pooled.setdefault(key, np.zeros(4))
                totals += [len(residual), residual.abs().sum(), (residual ** 2).sum(), residual.sum()]
        predicted = predicted.drop(columns=["expected_dc_voltage", "expected_dc_power"])
        measured.join(predicted).reset_index().to_parquet(output / f"{identifier}.parquet", index=False)
    pd.DataFrame(rows).to_csv(output / "per_inverter_errors.csv", index=False)
    summary = []
    for (population, model), (count, absolute, squared, signed) in pooled.items():
        summary.append({"population": population, "model": model, "valid_pairs": int(count),
                        "mae_w": absolute / count, "rmse_w": np.sqrt(squared / count),
                        "mbe_w": signed / count})
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "pooled_errors.csv", index=False)
    pd.testing.assert_series_equal(parameters, original)
    provenance = {**inverter_provenance(parameters), "measured_input_table": str(source),
                  "selection": "unchanged Task 11 masks; inv_05 excluded",
                  "reference_paco_w": float(parameters.Paco), "extension_ceiling_w": 30000.,
                  "calibration": "none", "k_dc": None, "production_default_changed": False,
                  "residual_sign": "measured minus predicted"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
