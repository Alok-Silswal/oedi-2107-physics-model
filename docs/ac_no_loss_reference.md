# Diagnostic no-loss inverter AC reference

Run:

```powershell
.\.venv\Scripts\python scripts/run_no_loss_ac.py
```

`calculate_no_loss_ac` in `src/oedi2107/ac_reference.py` maps ideal/pre-loss V/P
temporarily into the existing `calculate_inverter_ac` implementation. It adds
`expected_ac_power_no_loss` and a computation flag, preserving all original
columns/rows and any normal `expected_ac_power` column. **Block 4 is bypassed;
K_DC remains unset, not assigned 1.** This is an uncalibrated reference, not a
validated plant model or an upper bound under uncertain model inputs.

The unchanged entry is `ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_`.
Official Paco is 27,600 W, Pdco 28,199.173828 W, Vdco 715 V, Pnt 8.28 W.
All coefficients and full database provenance are retained. The existing
`pvlib.inverter.sandia` supplies clipping and signed tare; no second cap is used.
No thermal/DC rerun, POA/timing study, parameter fitting, plant aggregation or
meter comparison occurs.

## Coverage and selections

The comparison retains **14,935,056 records**, each with original timestamp and
inverter ID. Each inverter has 622,294 finite measured AC readings and 526,068
finite no-loss comparisons; 96,226 unavailable predictions remain NaN. Total
paired AC records: **12,625,632**.

Prepared AC metadata confirms kW; diagnostic outputs convert to W. The corrected
inverter-15 header is linked by its exact metric ID, not fuzzy matching. Original
Pacific wall-clock timestamps and measurements are untouched.

Loading summaries use POA>0, computed reference and existing electrical,
irradiance and fall-DST flags clear: 304,095 timestamps/inverter, or 7,298,280
paired records. A separately labelled measured-AC>0 slice has 6,522,877 records.
This removes zeros for a shape review, but does not establish healthy operation.
No suspicious unflagged observations are clipped or removed.

Both expected- and measured-power loading bases use the official Paco denominator.
Bins are explicit in the CSVs; expected/Paco=1 is the [1,1.05) model-clipped bin.
Residual = measured − no-loss expected. Tables give median paired residual and
median individual measured/expected ratio, not differences/ratios of separate
medians. Ratios require expected AC>0. Missing-basis/pair counts are reported.

## Loading-dependent bias

Pooled **positive-measured, illuminated, QC-clear** records; powers in kW:

| Expected/Paco bin | Measured median | No-loss median | Median residual | Median ratio | Records |
|---|---:|---:|---:|---:|---:|
| 0.10–0.25 | 4.132 | 4.634 | −0.366 | 0.920 | 795,670 |
| 0.25–0.50 | 9.187 | 10.133 | −0.916 | 0.909 | 1,008,945 |
| 0.50–0.75 | 15.951 | 17.331 | −1.633 | 0.906 | 896,374 |
| 0.75–0.90 | 21.457 | 22.883 | −1.701 | 0.926 | 637,011 |
| 0.90–0.95 | 23.891 | 25.553 | −1.729 | 0.932 | 251,349 |
| 0.95–1.00 | 25.287 | 26.936 | −1.681 | 0.938 | 296,096 |
| Model clipped at 1.00 | 28.528 | 27.600 | +0.928 | 1.034 | 1,447,938 |

All illuminated records, including zeros, have clipped-bin measured median
28.225 kW and residual +0.625 kW. The reference clips at 66,332 illuminated
timestamps/inverter, but many corresponding measurements are below the ceiling:
ideal DC is not calibrated and operation is not uniformly available.

When **measured** AC is 1.05–1.10 Paco, its pooled median is 29.872 kW versus
27.600 expected; median paired residual is +2.272 kW (622,274 records). The
reference's high-load sign reversal follows its lower clipping ceiling. These
statistics must not be interpreted as inverter-efficiency or K_DC estimates.

Below clipping the relationship is broadly increasing and physically plausible,
but a 6–10% central overprediction over much of the loading range cannot be
assigned to inverter efficiency: the upstream ideal DC has established bias.
The present comparison does not independently validate Sandia coefficients.

## Measured upper-power evidence

Per-inverter percentiles, maxima, repeated bands, counts and fractions are in
`measured_upper_regions.csv`. Two populations are explicit: all telemetry and
electrical-QC-clear telemetry (622,288 samples/unit, excluding six documented
row-level electrical events). No weather availability is required for these
measurement-only statistics; high-power readings are retained unchanged.

- QC-clear p99: **29.905–29.962 kW** across 24 units; p99.9:
  **29.985–29.998 kW**. QC-clear maxima: **30.082–30.106 kW**.
- All 24 have at least 12 measurements in the **30.0–30.1 kW** band.
  The densest ≥Paco band is **29.9–30.0 kW for 23 units**; inv_21's is
  **29.8–29.9 kW**, while its upper percentiles still approach 30 kW.
- Longest exact five-minute runs within each dominant band contain **43–57
  records**, supporting plateau-like persistence rather than isolated peaks.
- **4.95–8.05% of all QC-clear telemetry** is above 27.6 kW; among strictly
  positive measured AC, **11.54–16.97%** is above it. Denominators differ and
  include differing inverter availability. These are not fractions of healthy
  operating time or proven clipping time.
- Raw maxima for inv_04/inv_07 are about 103.710/105.341 kW; those documented
  suspect events remain in the all-telemetry results. They are excluded only
  from the labelled QC-clear summaries and do not define the apparent ceiling.

The 100 W bands and minimum 12 repeats are descriptive histogram choices, not
a fitted rating. A plateau is evidence of limiting/reporting behaviour, not
proof of a particular physical mechanism. **27.6 kW is not the observed AC
telemetry ceiling.** Approximately 30 kW is strongly supported across the fleet,
but no new Paco is inferred or adopted.

## Zero-POA tare and remaining review

Zero-POA, QC-clear records are separate (221,819 per unit). Sandia retains
**−8.28 W** throughout; measured median is zero for every unit. Zero POA is a
nighttime proxy, not astronomical proof of night. Some positive measured readings
remain. Negative expected AC is never clamped; ratios at nonpositive expected
AC are unavailable. Do not use these records to assess daytime fit or silently
change the later meter nighttime sign convention.

Before plant-level interpretation, confirm installed voltage/CEC variant,
nominal versus permitted maximum output, inverter settings, and AC telemetry
units/active-versus-apparent power definitions. Also resolve the upstream
representativeness/healthy-operation questions, availability handling, meter
alignment semantics and nighttime accounting. The 24 units share a similar upper
ceiling but have different operating coverage; equality of health/efficiency
is not established by the shared plateau.

Recommended next action: audit nameplate/settings and telemetry definitions for
the 27.6-versus-30 kW discrepancy, then compare measured DC V×I directly with
measured AC on the 23 DC-voltage-equipped units under unclipped operation. That
would test conversion independently of POA bias. Keep official parameters and
K_DC unchanged pending evidence; do not calibrate from this no-loss comparison.

Artifacts under `outputs/2107_ac_no_loss/` include labelled nominal and
inverter-level Parquet, channel mapping, coverage/loading/upper-region/tare CSVs,
four PNGs and unchanged-input/database provenance. Prepared data, ideal DC and
metadata SHA256 hashes match before/after. Tests cover this diagnostic path,
loading/plateau summaries and the directly used existing inverter implementation.
