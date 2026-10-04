"""Display saved Stage 1 results and write reports; never execute the model."""

import csv
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("outputs/2107_research_baseline")
SOURCES = ("provenance.json", "fleet_metrics.csv", "meter_scale_metrics.csv",
           "inverter_outputs.parquet", "plant_5min.parquet", "meter_15min.parquet")


def select(rows, **criteria):
    """Exact saved subset/quantity keys -> one unambiguous metric row."""
    matches = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved metric for {criteria}; found {len(matches)}")
    return matches[0]


def load_results(root=ROOT):
    """Saved CSV/JSON and Parquet footers -> verified metadata and five metrics."""
    directory = root / RESULTS
    missing = [str(directory / name) for name in SOURCES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Required frozen outputs missing:\n" + "\n".join(missing))
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    with (directory / "fleet_metrics.csv").open(newline="", encoding="utf-8") as stream:
        fleet = list(csv.DictReader(stream))
    with (directory / "meter_scale_metrics.csv").open(newline="", encoding="utf-8") as stream:
        meter = list(csv.DictReader(stream))
    for name, count in (("inverter_outputs.parquet", provenance["inverter_rows"]),
                        ("plant_5min.parquet", provenance["rows"])):
        if pq.ParquetFile(directory / name).metadata.num_rows != count:
            raise ValueError(f"Frozen coverage/provenance mismatch: {name}")
    config = provenance["configuration"]
    if provenance["inverter_rows"] != provenance["rows"] * config["inverter_count"]:
        raise ValueError("Frozen inverter-record coverage is inconsistent")
    metrics = []
    selections = [
        ("DC", "Vdc", "candidate", "V", 1., select(fleet, population="candidate_healthy", quantity="dc_voltage")),
        ("DC", "Idc", "candidate", "A", 1., select(fleet, population="candidate_healthy", quantity="dc_current")),
        ("DC", "Pdc", "candidate", "kW", 1000., select(fleet, population="candidate_healthy", quantity="dc_power")),
        ("inverter_ac", "Pac", "all_valid", "kW", 1000., select(fleet, population="all_pairs", quantity="ac_power")),
        ("plant_meter", "Pac", "15min", "kW", 1., select(meter, scale="15min", population="all_pairs", quantity="average_power_kw")),
    ]
    meter_row = selections[-1][-1]
    if pq.ParquetFile(directory / "meter_15min.parquet").metadata.num_rows != int(meter_row["total_samples"]):
        raise ValueError("Frozen meter coverage/metrics mismatch")
    for level, quantity, subset, unit, divisor, row in selections:
        metrics.append({"level": level, "quantity": quantity, "subset": subset,
                        "pair_count": int(row["valid_pairs"]),
                        **{key: float(row[key]) / divisor for key in ("mae", "rmse", "mbe")},
                        "r2": float(row["r_squared"]), "unit": unit})
    return provenance, metrics


def render_reports(provenance, metrics):
    """Verified saved results -> deterministic compact terminal and Markdown text."""
    config = provenance["configuration"]
    policy = provenance["frozen_research_policy"]
    availability = 100 * provenance["model_available_rows"] / provenance["rows"]
    modules = config["modules_per_string"] * config["strings_per_inverter"]
    topology = f'{config["modules_per_string"]} series x {config["strings_per_inverter"]} parallel'
    coverage = [
        ("Plant nominal DC capacity", f'~{config["approximate_dc_capacity_kw"]:g} kW'),
        ("Inverters", f'{config["inverter_count"]:,}'),
        ("Total timestamps", f'{provenance["rows"]:,}'),
        ("Model-input / computed timestamps", f'{provenance["model_available_rows"]:,}'),
        ("Model-input availability", f"{availability:.2f} %"),
        ("Total inverter records", f'{provenance["inverter_rows"]:,}'),
    ]
    frozen = [
        ("POA alignment", "0 min"), ("Temperature model", "Faiman"),
        ("Faiman u0", f'{config["faiman"]["u0"]:g}'),
        ("Faiman u1", f'{config["faiman"]["u1"]:g}'),
        ("PV module", config["module_name"]), ("Array topology (inferred)", topology),
        ("Modules / inverter", str(modules)),
        ("Total modules", str(modules * config["inverter_count"])),
        ("Global K_DC", "UNSET; bypassed" if config["dc_losses"]["k_dc"] is None else str(config["dc_losses"]["k_dc"])),
        ("Inverter model", "CEC / Sandia ceiling extension"),
        ("Sandia coefficients", "Official CEC; unchanged"),
        ("CEC reference rating", f'{policy["reference_paco_w"] / 1000:.1f} kW'),
        ("Maximum AC ceiling", f'{policy["maximum_ac_output_w"] / 1000:.1f} kW'),
    ]
    width = 79
    rule = "=" * width
    lines = [rule, "OEDI SYSTEM 2107".center(width),
             "STAGE 1 | FROZEN PHYSICS-BASED RESEARCH BASELINE".center(width), rule, ""]

    def section(title):
        lines.extend([title, "-" * width])

    def fields(items):
        lines.extend(f"  {key:<40}{value:>35}" for key, value in items)
        lines.append("")

    def terminal_table(selected, digits):
        widths = (8, 12, 11, 11, 11, 7)
        border = "+" + "+".join("-" * (size + 2) for size in widths) + "+"

        def row(cells):
            return "| " + " | ".join(
                cell.ljust(size) if column == 0 else cell.rjust(size)
                for column, (cell, size) in enumerate(zip(cells, widths))
            ) + " |"

        # ASCII notation survives legacy Windows console code pages.
        lines.extend([border, row(("Quantity", "Pairs", "MAE", "RMSE", "MBE", "R^2")), border])
        for metric in selected:
            cells = [metric["quantity"], f'{metric["pair_count"]:,}'] + [
                f'{metric[key]:.{digits}f} {metric["unit"]}' for key in ("mae", "rmse", "mbe")
            ] + [f'{metric["r2"]:.3f}']
            lines.append(row(cells))
        lines.extend([border, ""])

    section("SYSTEM / COVERAGE")
    fields(coverage)
    section("FROZEN MODEL")
    fields(frozen)
    section("1. DC VALIDATION - CANDIDATE SCREENED OPERATION")
    terminal_table(metrics[:3], 2)
    lines.extend(["  Unchanged candidate subset; pooled inverter observations.",
                  "  Inverter 05: measured Vdc unavailable; no Vdc/Pdc validation.", ""])
    for title, metric, digits in (
        ("2. INVERTER AC VALIDATION - ALL VALID PAIRS", metrics[3], 3),
        ("3. PLANT-METER VALIDATION - 15-MINUTE", metrics[4], 2),
    ):
        section(title)
        terminal_table([metric], digits)
    lines.extend(["  Meter pairs are valid 15-minute intervals.",
                  "  Alignment: mean(model at t, t+5, t+10) vs meter at interval-start t.",
                  "  All three samples required; average power [kW], no energy conversion.", ""])
    section("STAGE 1 CONCLUSION")
    lines.extend(["  STATUS     COMPLETE / FROZEN / UNCALIBRATED", "",
                  "  Plant-power dynamics captured; systematic overprediction remains.",
                  "  Negative MBE: measured power is below expected power on average.",
                  "  Residual = measured - expected",
                  "  Residuals include real losses AND model/input uncertainty.",
                  "  Baseline is not an unbiased forecast or an absolute fault threshold.", "",
                  "  NEXT       Stage 2: availability-aware residual analysis and",
                  "             inverter underperformance detection (not started).", rule])
    text = "\n".join(lines) + "\n"

    def markdown_table(selected):
        rows = ["| Quantity | Pairs | MAE | RMSE | MBE | R² |",
                "|---|---:|---:|---:|---:|---:|"]
        for metric in selected:
            rows.append(f'| **{metric["quantity"]}** | **{metric["pair_count"]:,}** | ' + " | ".join(
                f'{metric[key]:.3f} {metric["unit"]}' for key in ("mae", "rmse", "mbe")
            ) + f' | {metric["r2"]:.3f} |')
        return "\n".join(rows)

    markdown = f'''# OEDI System 2107 — Stage 1 validation summary

## System / coverage

**Stage 1 complete, frozen and uncalibrated.**

| Coverage | Result |
|---|---:|
| Nominal DC capacity / inverters | ~{config["approximate_dc_capacity_kw"]:g} kW / {config["inverter_count"]} |
| Total timestamps | {provenance["rows"]:,} |
| Model-input / computed timestamps | {provenance["model_available_rows"]:,} |
| Model-input availability | **{availability:.2f}%** |
| Inverter records | {provenance["inverter_rows"]:,} |

Missing-input rows remain identifiable.

## Frozen modelling decisions

Measured POA → Faiman → unchanged Hyundai CEC single-diode model → inferred
{topology} array → ideal pre-loss DC → documented 30 kW Sandia ceiling extension.
**K_DC is unset and bypassed**; its infrastructure is retained. No coefficients are fitted.
Official {policy["reference_paco_w"] / 1000:g} kW CEC reference output is retained separately.

## DC validation

Unchanged candidate-screened subset; pooled inverter observations.

{markdown_table(metrics[:3])}

Inverter 05 has no measured Vdc, so measured Vdc/Pdc comparisons are unavailable.

## Inverter AC validation

All finite valid pairs, pooled across the fleet; not mean per-inverter scores.

{markdown_table(metrics[3:4])}

## Plant-meter validation

15-minute all-valid interval comparisons; model plant AC is the mean at
t, t+5, t+10, requiring all three complete 24-inverter predictions.
Compare with interval-start meter average power in kW; do not multiply meter power by 0.25.

{markdown_table(metrics[4:5])}

## Interpretation

**Residual/MBE = measured − expected.** Negative MBE indicates systematic
overprediction. Plant dynamics are captured, but the baseline is not an unbiased
production forecast. Residuals contain real losses/underperformance **plus model
and input uncertainty**; they cannot be attributed solely to faults or losses.
R² uses 1−SSE/SST(measured); no normalized metrics are reported. These different
populations must not be treated as directly comparable model scores.
Stage 1 is frozen; Stage 2 availability-aware residual analysis and inverter
underperformance detection remain future work.

## Assumptions / inferences / limitations

### Assumptions / modelling choices

Measured POA directly as effective irradiance; zero shift;
Faiman u0={config["faiman"]["u0"]:g}, u1={config["faiman"]["u1"]:g}; {config["module_name"]} CEC representation;
official Sandia efficiency coefficients unchanged; no fitted global K_DC.
Thermal diagnostics did not identify unique calibrated coefficients; structured
healthy-candidate residuals did not support one identifiable static loss factor.

### Inferences

**{config["modules_per_string"]} modules/string × {config["strings_per_inverter"]} strings/inverter** =
{modules} modules/inverter and {modules * config["inverter_count"]} modules total, supported by DC voltage/current
behaviour and capacity. The 30 kW maximum-output ceiling is supported by measured
fleet behaviour and [manufacturer documentation](inverter_ceiling.md).

### Limitations

Unresolved POA representativeness/averaging; estimated cell
temperature; missing inverter 05 Vdc; Pacific local wall-clock/fall-DST ambiguity;
{availability:.2f}% input coverage; common AM/PM, seasonal and yearly residual structure.
Candidate screening is provisional, not independent holdout validation.
Inverter suffix/settings and CEC Idcmax interpretation remain provenance limitations;
the ceiling extension extrapolates official efficiency above its reference rating.
Negative model nighttime tare is retained despite measured nighttime zeros.
The baseline is neither an unbiased forecast nor an absolute fault-threshold model.

Sources: `outputs/2107_research_baseline/provenance.json`, `fleet_metrics.csv`,
`meter_scale_metrics.csv`; Parquet footers checked for coverage consistency.
See [full frozen-baseline report](research_baseline.md) for other populations/scales.
Regenerate with `python scripts/show_stage1_results.py`; no physics model is run.
'''
    return text, markdown


def main(root=ROOT):
    provenance, metrics = load_results(root)
    text, markdown = render_reports(provenance, metrics)
    directory = root / RESULTS
    (directory / "stage1_validation_summary.txt").write_text(text, encoding="utf-8")
    with (directory / "stage1_validation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (root / "docs" / "stage1_validation_summary.md").write_text(markdown, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as error:
        sys.exit(f"Stage 1 reporting failed: {error}")
