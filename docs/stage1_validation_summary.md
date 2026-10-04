# OEDI System 2107 — Stage 1 validation summary

## System / coverage

**Stage 1 complete, frozen and uncalibrated.**

| Coverage | Result |
|---|---:|
| Nominal DC capacity / inverters | ~893 kW / 24 |
| Total timestamps | 622,294 |
| Model-input / computed timestamps | 526,068 |
| Model-input availability | **84.54%** |
| Inverter records | 14,935,056 |

Missing-input rows remain identifiable.

## Frozen modelling decisions

Measured POA → Faiman → unchanged Hyundai CEC single-diode model → inferred
20 series x 6 parallel array → ideal pre-loss DC → documented 30 kW Sandia ceiling extension.
**K_DC is unset and bypassed**; its infrastructure is retained. No coefficients are fitted.
Official 27.6 kW CEC reference output is retained separately.

## DC validation

Unchanged candidate-screened subset; pooled inverter observations.

| Quantity | Pairs | MAE | RMSE | MBE | R² |
|---|---:|---:|---:|---:|---:|
| **Vdc** | **129,444** | 14.853 V | 17.693 V | -13.057 V | 0.627 |
| **Idc** | **135,072** | 2.633 A | 3.169 A | -1.732 A | 0.757 |
| **Pdc** | **129,444** | 1.911 kW | 2.302 kW | -1.508 kW | 0.704 |

Inverter 05 has no measured Vdc, so measured Vdc/Pdc comparisons are unavailable.

## Inverter AC validation

All finite valid pairs, pooled across the fleet; not mean per-inverter scores.

| Quantity | Pairs | MAE | RMSE | MBE | R² |
|---|---:|---:|---:|---:|---:|
| **Pac** | **12,625,632** | 1.633 kW | 4.721 kW | -1.333 kW | 0.796 |

## Plant-meter validation

15-minute all-valid interval comparisons; model plant AC is the mean at
t, t+5, t+10, requiring all three complete 24-inverter predictions.
Compare with interval-start meter average power in kW; do not multiply meter power by 0.25.

| Quantity | Pairs | MAE | RMSE | MBE | R² |
|---|---:|---:|---:|---:|---:|
| **Pac** | **154,611** | 44.940 kW | 77.101 kW | -39.983 kW | 0.896 |

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
Faiman u0=25, u1=6.84; Hyundai HiS-M310TI CEC representation;
official Sandia efficiency coefficients unchanged; no fitted global K_DC.
Thermal diagnostics did not identify unique calibrated coefficients; structured
healthy-candidate residuals did not support one identifiable static loss factor.

### Inferences

**20 modules/string × 6 strings/inverter** =
120 modules/inverter and 2880 modules total, supported by DC voltage/current
behaviour and capacity. The 30 kW maximum-output ceiling is supported by measured
fleet behaviour and [manufacturer documentation](inverter_ceiling.md).

### Limitations

Unresolved POA representativeness/averaging; estimated cell
temperature; missing inverter 05 Vdc; Pacific local wall-clock/fall-DST ambiguity;
84.54% input coverage; common AM/PM, seasonal and yearly residual structure.
Candidate screening is provisional, not independent holdout validation.
Inverter suffix/settings and CEC Idcmax interpretation remain provenance limitations;
the ceiling extension extrapolates official efficiency above its reference rating.
Negative model nighttime tare is retained despite measured nighttime zeros.
The baseline is neither an unbiased forecast nor an absolute fault-threshold model.

Sources: `outputs/2107_research_baseline/provenance.json`, `fleet_metrics.csv`,
`meter_scale_metrics.csv`; Parquet footers checked for coverage consistency.
See [full frozen-baseline report](research_baseline.md) for other populations/scales.
Regenerate with `python scripts/show_stage1_results.py`; no physics model is run.
