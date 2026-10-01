# Frozen System 2107 research baseline

Run `.venv/Scripts/python scripts/run_frozen_research.py` from the project root.
Outputs are in `outputs/2107_research_baseline/`. This is the final research path;
the older optional DC-loss/calibrated pipeline is retained for reproducibility.

## Frozen decisions

- Measured POA is effective irradiance, with zero shift, no interpolation or
  optical correction. Sensor representativeness and averaging semantics remain
  unresolved; diagnostics do not justify a production timestamp change.
- Faiman u0=25, u1=6.84; module temperature is the cell-temperature proxy.
  Thermal diagnostics showed voltage sensitivity, not uniquely identifiable
  calibrated coefficients. No coefficient fitting is performed.
- Exact module: `Hyundai_Heavy_Industries_Green_Energy_Co__HiS_M310TI`;
  unchanged CEC single-diode parameters and explicit pvlib calculation.
- Inferred, strongly DC-evidence-supported 20 modules/string, 6 strings/inverter,
  24 nominal arrays; fixed tilt 25 degrees, azimuth 180 degrees.
- K_DC remains unset and its infrastructure remains available. The research
  path bypasses that layer, using ideal pre-loss DC directly. Structured
  time/year/season residuals prevented identification of one static loss factor.
  No assignment of K_DC=1 and no representation of this as a loss-calibrated model.
- Exact inverter: `ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_`. Official
  coefficients/Paco=27.6 kW remain unchanged. Expected research AC uses the
  [30 kW extension](inverter_ceiling.md); official clipped AC is retained.
- Negative Sandia night tare is retained. No new parameters or per-inverter fits.

## Outputs and validation

- `nominal_model.parquet`: all-row common weather/physics baseline.
- `inverter_outputs.parquet`: all timestamps for all 24 IDs, weather, availability,
  original QC/DST flags, cell temperature, ideal/DC aliases, measured DC/AC,
  research/official AC, unchanged candidate mask and signed residuals.
  DC is in V/A/W; AC is W. Inverter 05 measured Vdc/Pdc stay missing.
- `plant_5min.parquet`: explicit sum of all 24 named predicted inverter outputs;
  requires 24 finite outputs, otherwise NaN. Measured inverter-sum is the prepared
  complete-fleet `plant_ac_power_kw` converted to W, never partial telemetry.
- `meter_15min.parquet`: meter and model interval-average kW, sample count,
  model availability, original suspicious-event mask, signed residual kW.
  Meter time t is interval start; require exact model t, t+5, t+10 and average W
  then divide by 1000. Do not convert the source meter values to energy.
- `per_inverter_metrics.csv` and `fleet_metrics.csv`: counts, MAE/RMSE/MBE/R²
  for DC V/I/P and AC, with all-pair, QC-clear daytime and previous candidate
  populations. QC-clear daytime means POA>0 and electrical/irradiance/DST flags
  clear; it is not a newly selected healthy population. Candidate masks are read
  unchanged, not optimized. All-pair metrics retain QC extremes intentionally.
- Fleet metrics pool aligned individual inverter observations, including R²;
  they are not an arithmetic average of inverter metrics. Plant-sum metrics are
  separately saved in `plant_inverter_sum_metrics.csv`.
- `meter_scale_metrics.csv` and `meter_h/D/MS.parquet`: complete nominal bins,
  average kW and integrated kWh. Require 4 intervals/hour, 96/day, or calendar
  days times 96/month. Incomplete bins are NaN, not extrapolated totals.
- `meter_observed_support_metrics.csv` and corresponding observed-support
  Parquets: additional same-pair hourly/daily/monthly averages and summed kWh
  over available intervals only, with paired coverage. These are not whole-bin
  averages or full-month energy estimates. QC-clear variants remove only the
  existing meter flags before pairing; excluded periods are not interpolated.
- All metrics use finite pairs only and report excluded counts; normalization
  is absent. MBE/residual = measured minus expected. R² is 1-SSE/SST(measured).
- `provenance.json`: frozen configuration, model/source provenance, hashes,
  coverage, selection and unit/time conventions. Source data are never modified.

## Publication limitations

This is an uncalibrated physics research baseline, not a validated fault detector
or loss/degradation estimator. The upstream common-mode AM/PM current bias and
year/season dependence remain. POA optical/averaging equivalence is unverified;
thermal parameters/cell proxy are provisional. Topology is inferred. Inverter
US/480 V suffix, settings and CEC Idcmax interpretation still warrant hardware
provenance review. The ceiling extension extrapolates official efficiency above
the reference rating and does not enforce dynamic thermal/grid/MPPT limits.

Timestamps remain naive Pacific local wall clock; fall-DST ambiguity is unresolved.
Higher-scale durations are nominal wall-clock durations, not UTC elapsed time.
Missing intervals, retained suspicious extremes and support-dependent scores
must be disclosed. Meter flags are screening labels, not corrected measurements.
Nighttime measured zeros and negative predicted tare use different conventions.
The earlier candidate subset is provisional and selected using a 27.6 kW guard;
it is preserved for comparison, not independent holdout evidence. No train/test
split or calibration is claimed. No residual thresholds, faults or ML are added.

## Completed run

622,294 timestamps and 14,935,056 named inverter rows were preserved.
526,068 timestamps (84.54%) have computed model output; 96,226 remain uncomputed.
The original candidate set contains 5,628 timestamps. Input hashes are unchanged.
Three research tests and the one inverter-ceiling test passed.

Pooled fleet diagnostics (V, A, W as labelled; R² is dimensionless):

| Population | Quantity | Pairs | MAE | RMSE | MBE | R² |
|---|---|---:|---:|---:|---:|---:|
| All | Vdc V | 12,099,563 | 48.27 | 145.89 | -23.50 | .809 |
| Day/QC clear | Vdc V | 6,994,185 | 67.62 | 172.61 | -56.17 | -.017 |
| Candidate | Vdc V | 129,444 | 14.85 | 17.69 | -13.06 | .627 |
| All | Idc A | 12,625,631 | 2.674 | 7.562 | -2.156 | .780 |
| Day/QC clear | Idc A | 7,298,280 | 4.611 | 9.925 | -3.722 | .649 |
| Candidate | Idc A | 135,072 | 2.633 | 3.169 | -1.732 | .757 |
| All | Pdc W | 12,099,562 | 1,842 | 5,022 | -1,535 | .778 |
| Day/QC clear | Pdc W | 6,994,185 | 3,178 | 6,590 | -2,649 | .640 |
| Candidate | Pdc W | 129,444 | 1,911 | 2,302 | -1,508 | .704 |
| All | Inverter AC W | 12,625,632 | 1,633 | 4,721 | -1,333 | .796 |
| Day/QC clear | Inverter AC W | 7,298,280 | 2,812 | 6,195 | -2,306 | .672 |
| Candidate | Inverter AC W | 135,072 | 1,854 | 2,233 | -1,441 | .714 |

Candidate Vdc/Pdc cover only 23 inverters; Idc/AC cover 24. Day/QC-clear
performance includes outages and startup, and is not a healthy-operation filter.
Nighttime zeros increase all-pair scores; narrower daytime/candidate populations
have different measured variances, so R² scores are not directly interchangeable.

Complete-bin meter comparison, average power in kW:

| Scale | Valid bins | MAE kW | RMSE kW | MBE kW | R² |
|---|---:|---:|---:|---:|---:|
| 15 min | 154,611 | 44.94 | 77.10 | -39.98 | .896 |
| Hour | 33,737 | 49.73 | 78.75 | -45.26 | .891 |
| Day | 626 | 35.03 | 42.92 | -35.03 | .434 |
| Month | 11 | 40.92 | 43.11 | -40.92 | -1.102 |

Excluding existing meter flags gives 154,587 valid 15-minute intervals:
MAE 44.93 kW, RMSE 77.02 kW, MBE -39.99 kW, R² .896. Complete-bin flags
reduce hourly/day/month counts to 33,717/611/5. Corresponding scores and energy
metrics are in the scale CSV; the small monthly sample is not representative.

Observed-support averages additionally cover 45,504 hours, 2,085 days and all
71 months. Their MAE/RMSE/MBE/R² are respectively 37.81/69.84/-34.35/.907,
40.32/52.06/-40.23/.620 and 40.92/44.00/-40.92/.452 (kW except R²).
They summarize available paired intervals, not complete calendar production.

Plant inverter-sum all-pair MAE/RMSE/MBE are 36.55/86.58/-32.00 kW,
R² .870 over 526,068 timestamps. This instantaneous comparison and the meter
interval comparison use different supports and are not interchangeable.

The run exposes substantial systematic overprediction, not a catastrophic
numerical failure. Monthly accuracy is weak, and no unbiased production forecast
or absolute fault threshold is justified. The model is defensible to freeze as
a transparent, reproducible, uncalibrated research baseline with these disclosed
limitations. No validation result changed any parameter or cleaning decision.
