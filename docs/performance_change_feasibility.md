# Bounded performance-change feasibility experiment

**Verdict: gradual decline is inconclusive.** Regression and one modest ML weather model agree: substantial delivered-performance changes and recoveries are visible, but a persistent negative slope is not established after recorded zeros and previously screened operational changes are represented. This is performance-change evidence, not physical-ageing evidence.

## Fixed reference and comparable support

[Code](../scripts/experiment_performance_change.py) reads the completed readiness audit, cleaned AC and saved nominal weather/quality flags. It repeats no equipment, temperature or timing checks. One Ridge regression (eight weather interaction terms) and one histogram gradient-boosted model (100 iterations, depth <=4, <=15 leaves) learn the **2019 fleet-median AC reference once**, from odd-numbered dates with median AC >2 kW. Predictors: contemporaneous POA, ambient air temperature, wind, clock hour and fixed annual phase; no year, trend, target telemetry or estimated cell-temperature truth. References are saved and never updated for later years.

On 144 separate even-numbered 2019 dates, regression/ML MAE is **0.830/0.558 kW**, RMSE **1.492/1.176 kW**. This validates reference interpolation on held-out dates, not a verified healthy-power counterfactual or later-year accuracy. No hyperparameter collection was trained.

Annual comparisons use **January–October in every year**, because the weather overlap ends October 2023. Eligible readings have complete 24-unit AC, available weather, POA >=200 W/m2 and existing clear quality/DST flags. Fixed weather cells: calendar month, three-hour bin, POA/150 W/m2 and ambient/10 C. Each retained cell has >=10 observations in **every 2019–2023 year**; 2019 cell counts are fixed weights within each month, and annual results give ten months equal weight. The 2018 sensitivity additionally requires support in 2018 and does not change the main cohort. Wind and remaining within-cell weather differences are adjusted by the fixed model.

Indices average measured/predicted power within these cells, normalized to the unit's 2019 reference. They describe standardized performance, **not whole-year delivered energy**. Main delivered support is 223 cells and **122,272 simultaneous timestamps** (2,934,528 unit readings). Missing-weather periods are not filled or estimated as zeros.

- **Delivered:** retain recorded zeros, poor operation and high-power/clipped observations; fixed reference predictions are bounded to 0.5–30 kW.
- **Generating:** actual AC >2 kW and <27 kW, and **both** fixed predictions <27 kW. Both methods use identical observations and cell support. This conditional measure can hide losses by excluding zero/very-low/clipped operation. Common-support filtering excludes inverter 08/21 February and inverter 17 February–March from every main year; the conditional index cannot resolve those monthly event onsets. Delivered observations remain available.

## Results

Annual changes below are ML index points versus 2019, with descriptive 95% paired-calendar-month intervals; the regression results are close and supplied in [annual estimates](../outputs/2107_performance_change/annual_indices.csv).

| Year | Delivered change | Generating change | Standardized recorded-zero fraction |
|---|---:|---:|---:|
| 2020 | +4.92 [0.43, 9.41] | -2.19 [-5.71, 1.34] | 2.9% |
| 2021 | -5.96 [-22.68, 10.77] | +0.31 [-1.19, 1.81] | 15.2% |
| 2022 | +3.33 [-3.52, 10.18] | +0.49 [-1.27, 2.25] | 6.6% |
| 2023 | +6.61 [-0.04, 13.27] | +0.67 [-1.70, 3.03] | 3.6% |

2019 standardized zero fraction is 8.3%. The 2021 delivered dip and subsequent recovery are largely absent in the generating outcome. Excluding poor operation therefore materially changes the apparent history.

Regression on monthly unit indices includes unit and calendar-month intercepts and a shared continuous slope. The regime sensitivity adds unit-specific level changes at the readiness audit's screened boundaries, including documented inverter 04 recovery in October 2023. Drops and recoveries remain in the data; none is called a confirmed fault. A further delivered sensitivity conditions on monthly zero fraction; it is a decomposition, **not the delivered outcome itself**. These controls are observational and may also absorb genuine persistent changes.

| Main 2019–2023 annual slope, index points/year (95% interval) | Regression | ML |
|---|---:|---:|
| Delivered, untreated | +1.12 [-0.77, 3.00] | +1.16 [-0.74, 3.07] |
| Delivered, screened steps represented | +0.34 [-1.64, 2.31] | +0.38 [-1.59, 2.35] |
| Delivered, steps and zero fraction represented | -0.04 [-0.50, 0.43] | +0.00 [-0.46, 0.46] |
| Generating, screened steps represented | -0.15 [-0.70, 0.40] | -0.11 [-0.63, 0.42] |

**Sensitivity:** generating step-adjusted slopes are regression/ML **-0.05/+0.01** with 2018, **-0.34/-0.24** for 2019–2022, and **+0.63/+0.58** for 2020–2023. Relaxing the guard to 29 kW gives **-0.18/-0.14** for the main period. Every corresponding interval spans zero. Without regime controls, generating slopes instead suggest +0.37/+0.39 points/year. The evidence does not support a stable negative trend across period/treatment choices.

Plant-wide changes are preserved: [strict plant delivered indices](../outputs/2107_performance_change/strict_plant_delivered.csv) use the simultaneous **24-unit sum**, not peer normalization. Untreated plant slopes are regression **+0.87 [-0.92, 2.67]**, ML **+0.92 [-0.89, 2.73]** points/year. Separate unit weather and fleet-relative slopes are in [slope diagnostics](../outputs/2107_performance_change/slopes.csv). Peer indices intentionally remove shared plant change and cannot rule out shared deterioration. Inverter 17's 2023 drop/recovery, inverter 04's drop/recovery and inverter 05's late drop remain visible.

## Plots, uncertainty and reproduction

- [Monthly delivered and generating performance](../outputs/2107_performance_change/plant_monthly.png).
- [Annual changes and uncertainty](../outputs/2107_performance_change/annual_changes.png).
- [Unit weather versus peer indices](../outputs/2107_performance_change/operational_changes.png).

Slope intervals use **site-month clustered HAC scores, lag three months**; units on one date are not independent plants. Annual intervals use ten paired monthly differences, not millions of independent readings. Intervals are conditional on fitted references, support and screened boundaries: they omit weather-reference fitting uncertainty, sensor calibration/drift, missing-weather losses and uncertainty in operational labels. Adjacent months and shared interruptions further limit precision. A small real decline remains possible; so do residual weather-model bias and operational changes. No sensor/maintenance records resolve those causes. No simulated decline was used to infer real ageing; the synthetic test only checks recovery of a known slope with a reversible step.

```powershell
.\.venv\Scripts\python -B scripts/experiment_performance_change.py
.\.venv\Scripts\python -B -m pytest tests/test_performance_change.py -q -p no:cacheprovider
```

Requires scikit-learn **1.7.2**, installed in the repository virtual environment; other versions and input hashes are in [provenance](../outputs/2107_performance_change/provenance.json). Outputs include frozen learned references, monthly/annual indices, support, regime boundaries and sensitivities. Three focused tests pass. Hash checks preserve all 21 frozen baseline files and the used data/audit inputs; no source data or baseline parameters/code were changed.

**Next action:** reconcile the large common zero periods and unit drops/recoveries with operational and POA-calibration records before pursuing a physical-ageing estimate. The dataset already supports descriptive performance-change research; this bounded experiment does not establish gradual degradation. Forecasting has not been started.
