# Final 15-minute PV forecasting results

Power errors are **kW**; bias = prediction − actual; R² is unitless. Blank/— means unavailable. Tables are rounded for presentation; `outputs/2107_final_reporting/results.csv` retains stored precision. Quantum test values retain exactly their six-decimal source precision in that file.

Quantum test scores are **friend-verified, user-accepted final reported results**, copied from `reported_quantum_results.md`. Quantum test predictions and evaluator provenance were not supplied locally. No quantum R² was supplied. Reported quantum test means/sample SD are retained as supplied, not recalculated from rounded seeds.

## Method and scope

- One site, 24 inverters; strict complete-fleet targets, recorded zeros retained, 15-minute forecasts. Inverters are units within one plant.
- Shared elapsed-time 70/15/15 split by target timestamp: validation starts 2022-01-21 16:45; testing starts 2022-12-11 20:20; record ends 2023-10-31 23:50 exclusive. Exported naive Pacific timestamps and existing QC/availability rules are retained.
- Weather → frozen Stage 1 physics → signed correction; direct models exclude physics. Settings/checkpoints selected on validation, with no test refit or new selection. Search ranges were specified after baseline inspection.
- Parts of the chronological test period appeared in earlier exploratory work: held out from these fits, not entirely unseen. Logger/current-measurement availability assumptions remain unverified physical latency.
- NN summaries are mean and **sample SD across five seeds**, not ensemble forecasts or confidence intervals. Quantum uses three seeds, four qubits/two layers in CPU simulation and batch 256; classical NN uses batch 128. Different input dimensions/settings preclude claims of identical capacity or quantum advantage.
- Returned quantum training metadata (Python 3.13.1/Torch 2.14.1 and reference fingerprints) differs from the local reference. Historical metadata and compatibility checks remain unchanged. Quantum test reporting follows the accepted friend source.

Validation: **70,555 rows / 43,949 daylight**. Test: **77,561 rows / 48,190 daylight**. Daylight uses target solar elevation >0°. Missing validation descriptive metrics are calculated only from saved validation forecasts; no model inference or held-out evaluation is performed by the reporting script. Classical correction test per-seed R² is absent from its saved metric table; available aggregate R² is retained from the saved comparison.

## Validation — model selection

### ML families and references

| Model / statistic | Best/stop | Day MAE | Day RMSE | Day bias | Day R² | All MAE | All RMSE | All bias | All R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics + Ridge (ridge_00) | — | 34.630 | 66.193 | -2.951 | 0.9225 | 22.383 | 52.260 | -1.041 | 0.9567 |
| Physics + RF (forest_05) | — | 32.870 | 64.964 | -1.296 | 0.9253 | 20.478 | 51.272 | -0.807 | 0.9583 |
| Physics + Histogram (histogram_05) | — | 33.888 | 65.323 | 0.108 | 0.9245 | 21.173 | 51.557 | 0.010 | 0.9579 |
| Physics + XGBoost (xgboost_11) | — | 33.957 | 65.735 | -1.002 | 0.9236 | 21.205 | 51.881 | -0.674 | 0.9573 |
| Physics | — | 74.013 | 108.776 | 53.981 | 0.7907 | 46.179 | 85.850 | 33.549 | 0.8832 |
| Persistence | — | 44.073 | 76.915 | 0.106 | 0.8953 | 27.476 | 60.706 | 0.088 | 0.9416 |
| Direct Ridge (ridge_00) | — | 34.563 | 66.202 | -2.960 | 0.9225 | 22.402 | 52.271 | -0.986 | 0.9567 |
| Direct RF (forest_05) | — | 31.975 | 64.556 | -1.700 | 0.9263 | 19.919 | 50.951 | -1.058 | 0.9588 |
| Direct Histogram (histogram_05) | — | 32.529 | 64.575 | -1.060 | 0.9262 | 20.322 | 50.966 | -0.608 | 0.9588 |
| Direct XGBoost (xgboost_11) | — | 32.535 | 65.022 | -2.255 | 0.9252 | 20.317 | 51.318 | -1.365 | 0.9582 |

### Neural / quantum seeds and summaries

| Model / statistic | Best/stop | Day MAE | Day RMSE | Day bias | Day R² | All MAE | All RMSE | All bias | All R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct NN 2107 | 14/24 | 33.129 | 65.804 | -0.457 | 0.9234 | 22.248 | 52.011 | 1.280 | 0.9571 |
| Direct NN 2108 | 59/69 | 32.099 | 65.201 | 0.500 | 0.9248 | 20.653 | 51.475 | 0.881 | 0.9580 |
| Direct NN 2109 | 13/23 | 33.007 | 66.178 | -0.514 | 0.9225 | 21.761 | 52.273 | 0.844 | 0.9567 |
| Direct NN 2110 | 9/19 | 33.588 | 66.487 | -2.328 | 0.9218 | 21.864 | 52.498 | -0.530 | 0.9563 |
| Direct NN 2111 | 6/16 | 34.015 | 67.047 | -3.859 | 0.9205 | 21.843 | 52.933 | -2.532 | 0.9556 |
| Direct NN mean | — | 33.168 | 66.143 | -1.331 | 0.9226 | 21.674 | 52.238 | -0.011 | 0.9567 |
| Direct NN sample SD | — | 0.718 | 0.696 | 1.744 | 0.0016 | 0.601 | 0.544 | 1.567 | 0.0009 |
| Physics + NN 2107 | 15/25 | 33.635 | 67.243 | -2.041 | 0.9200 | 21.387 | 53.078 | -1.056 | 0.9553 |
| Physics + NN 2108 | 36/46 | 32.419 | 65.810 | 0.525 | 0.9234 | 20.827 | 51.954 | 0.836 | 0.9572 |
| Physics + NN 2109 | 31/41 | 32.337 | 66.291 | -0.137 | 0.9223 | 20.512 | 52.326 | 0.188 | 0.9566 |
| Physics + NN 2110 | 47/57 | 32.407 | 65.422 | 1.687 | 0.9243 | 21.246 | 51.666 | 2.044 | 0.9577 |
| Physics + NN 2111 | 6/16 | 33.743 | 67.017 | -1.044 | 0.9205 | 21.576 | 52.906 | -1.185 | 0.9556 |
| Physics + NN mean | — | 32.908 | 66.357 | -0.202 | 0.9221 | 21.110 | 52.386 | 0.166 | 0.9565 |
| Physics + NN sample SD | — | 0.715 | 0.774 | 1.431 | 0.0018 | 0.433 | 0.604 | 1.350 | 0.0010 |
| Physics + Quantum 2107 | 27/37 | 32.409 | 65.706 | -1.331 | — | 20.538 | 51.865 | -1.033 | — |
| Physics + Quantum 2108 | 23/33 | 32.542 | 65.966 | -0.553 | — | 20.858 | 52.076 | -0.851 | — |
| Physics + Quantum 2109 | 33/43 | 33.922 | 66.687 | -0.233 | — | 22.045 | 52.662 | 0.519 | — |
| Physics + Quantum mean | — | 32.958 | 66.120 | -0.706 | — | 21.147 | 52.201 | -0.455 | — |
| Physics + Quantum sample SD | — | 0.838 | 0.508 | 0.564 | — | 0.794 | 0.413 | 0.848 | — |

## Held-out test — final reporting

### ML families and references

| Model / statistic | Best/stop | Day MAE | Day RMSE | Day bias | Day R² | All MAE | All RMSE | All bias | All R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics + Ridge (ridge_00) | — | 33.303 | 68.123 | -2.140 | 0.9232 | 21.531 | 53.716 | -0.496 | 0.9547 |
| Physics + RF (forest_05) | — | 32.243 | 67.799 | -0.726 | 0.9239 | 20.036 | 53.442 | -0.450 | 0.9551 |
| Physics + Histogram (histogram_05) | — | 33.010 | 67.969 | -1.495 | 0.9235 | 20.566 | 53.576 | -0.978 | 0.9549 |
| Physics + XGBoost (xgboost_11) | — | 32.808 | 68.158 | -1.784 | 0.9231 | 20.419 | 53.725 | -1.141 | 0.9547 |
| Direct Ridge (ridge_00) | — | 33.227 | 68.119 | -2.225 | 0.9232 | 21.555 | 53.716 | -0.479 | 0.9547 |
| Direct RF (forest_05) | — | 30.922 | 66.930 | -0.277 | 0.9259 | 19.215 | 52.757 | -0.170 | 0.9563 |
| Direct Histogram (histogram_05) | — | 31.967 | 67.482 | -1.455 | 0.9246 | 19.929 | 53.193 | -0.843 | 0.9555 |
| Direct XGBoost (xgboost_11) | — | 31.741 | 67.867 | -1.229 | 0.9238 | 19.771 | 53.495 | -0.719 | 0.9550 |
| Physics | — | 62.913 | 98.702 | 39.976 | 0.8387 | 39.165 | 77.801 | 24.762 | 0.9049 |
| Persistence | — | 43.041 | 79.382 | 0.019 | 0.8957 | 26.766 | 62.573 | 0.034 | 0.9385 |

### Neural / quantum seeds and summaries

| Model / statistic | Best/stop | Day MAE | Day RMSE | Day bias | Day R² | All MAE | All RMSE | All bias | All R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Physics + NN mean | — | 31.071 | 68.007 | 1.413 | 0.9234 | 19.887 | 53.618 | 1.183 | 0.9548 |
| Physics + NN sample SD | — | 0.820 | 0.688 | 0.637 | 0.0016 | 0.434 | 0.536 | 0.860 | 0.0009 |
| Direct NN mean | — | 31.402 | 67.810 | -0.318 | 0.9239 | 20.530 | 53.484 | 0.649 | 0.9551 |
| Direct NN sample SD | — | 0.548 | 0.459 | 1.063 | 0.0010 | 0.531 | 0.359 | 1.121 | 0.0006 |
| Physics + NN 2107 | — | 31.680 | 68.513 | 0.803 | — | 20.082 | 54.009 | 0.807 | — |
| Physics + NN 2108 | — | 30.458 | 67.495 | 1.503 | — | 19.560 | 53.215 | 1.374 | — |
| Physics + NN 2109 | — | 30.623 | 67.950 | 1.476 | — | 19.404 | 53.567 | 1.222 | — |
| Physics + NN 2110 | — | 30.390 | 67.209 | 2.394 | — | 19.889 | 53.004 | 2.434 | — |
| Physics + NN 2111 | — | 32.201 | 68.866 | 0.888 | — | 20.500 | 54.293 | 0.080 | — |
| Direct NN 2107 | — | 31.487 | 67.665 | 0.320 | 0.9242 | 21.214 | 53.414 | 1.814 | 0.9552 |
| Direct NN 2108 | — | 30.637 | 67.141 | 0.896 | 0.9254 | 19.736 | 52.939 | 1.216 | 0.9560 |
| Direct NN 2109 | — | 31.080 | 67.818 | 0.055 | 0.9239 | 20.478 | 53.495 | 1.171 | 0.9550 |
| Direct NN 2110 | — | 31.867 | 68.051 | -1.361 | 0.9233 | 20.699 | 53.661 | 0.038 | 0.9548 |
| Direct NN 2111 | — | 31.939 | 68.373 | -1.503 | 0.9226 | 20.520 | 53.911 | -0.996 | 0.9543 |
| Physics + Quantum 2107 | — | 30.572 | 67.495 | -0.307 | — | 19.361 | 53.208 | -0.458 | — |
| Physics + Quantum 2108 | — | 30.616 | 67.624 | 0.265 | — | 19.611 | 53.315 | -0.379 | — |
| Physics + Quantum 2109 | — | 32.155 | 68.506 | 0.839 | — | 20.888 | 54.027 | 1.327 | — |
| Physics + Quantum mean | — | 31.114 | 67.875 | 0.265 | — | 19.953 | 53.517 | 0.163 | — |
| Physics + Quantum sample SD | — | 0.901 | 0.551 | 0.573 | — | 0.819 | 0.446 | 1.008 | — |

## Sources and figures

- ML: correction HPO settings/trials and saved best-validation forecasts; direct `results_summary.csv`; correction `test/comparison.csv` and direct test metrics.
- Classical NN: per-seed saved validation metrics/forecasts; `outputs/2107_stage2_test/metrics.csv`; direct test metrics and saved comparison summaries.
- Quantum: byte-verified returned CPU training bundle and `outputs/2107_final_reporting/reported_quantum_results.md`. Downloads retains the original ZIP backup. Five duplicate downloaded Parquet files and `metrics(3).csv` were not copied.
- Full source hashes, archive import verification and reporting provenance: `outputs/2107_final_reporting/provenance.json`. The external compiled reference was cross-checked; malformed blank model labels were replaced using authoritative local files.
- Final held-out figures: `test_ml_mae`, `test_ml_rmse`, `test_neural_seed_summary` (300-dpi PNG and vector SVG) under the same output folder. Earlier exploratory plots are preserved separately.

This Markdown is suitable for deliberate Git inclusion. Outputs remain ignored; `.gitignore` is unchanged.
