# First uncalibrated real-data DC results

Run: 2107 prepared 5-minute data, 2017-12-01 00:00 to 2023-10-31 23:45,
pvlib 0.16.1. Blocks 1–3 plus inferred array scaling only. No K_DC, AC prediction,
calibration, normalized metrics, cleaning changes or healthy-period selection.

## Coverage and integrity

- Source: **622,294 rows, 134 columns**; **526,068** ideal predictions computed.
- **96,226** unavailable-input rows remain with NaN predictions and status/flags.
- Inverter table: **14,935,056 records**, all 24 logical channel IDs retained.
- Prepared data and metadata SHA256 hashes are unchanged; see provenance JSON.
- All 60 fall-DST ambiguity flags remain; 57 have computable DC inputs.
- All 11 POA readings of 1400 W/m² remain. QC marks do not exclude rows.
- Original metadata confirms all DC channels use V/A with scale 1 and offset 0.
  Finite V×I gives W directly. Inverter 05 Vdc/Pdc remain entirely missing.
- **86 tests passed**, including unchanged Blocks 1–7 tests and integration tests.

Each inverter normally has 526,068 finite V/I/P comparison pairs. Exceptions:

| Inverter | Vdc pairs | Idc pairs | Pdc pairs |
| --- | ---: | ---: | ---: |
| 05 | 0 | 526068 | 0 |
| 06 | 526068 | 526067 | 526067 |
| 07 | 526067 | 526068 | 526067 |

Complete per-inverter counts and metrics are in
[dc_metrics.csv](../outputs/2107_dc_ideal/dc_metrics.csv); finite ranges and
topology evidence are in
[dc_ranges_and_topology.csv](../outputs/2107_dc_ideal/dc_ranges_and_topology.csv).

## Ideal model ranges

| Quantity | Minimum | Maximum |
| --- | ---: | ---: |
| Temperature | -4.78 °C | 72.12 °C |
| Array Vmp | 0 V | 793.95 V |
| Array Imp | 0 A | 71.88 A |
| Array Pmp | 0 W | 48665.41 W |

Expected DC is finite wherever computed and remains NaN elsewhere. Power agrees
with voltage×current within 1.5e-11 W. No rating cap is applied. High irradiance
can produce ideal power above nominal STC power; 1400 W/m² validity is unresolved.

## Provisional metrics

Ranges below span inverter summaries, excluding undefined inverter-05 V/P scores.
They include all finite pairs, including nighttime and retained questionable or
low-output operating periods. Residual/MBE sign is measured minus expected.

| Signal | MAE range | RMSE range | MBE range | R² range |
| --- | --- | --- | --- | --- |
| Voltage [V] | 35.11–84.83 | 114.89–215.31 | -66.04–-8.20 | 0.587–0.879 |
| Current [A] | 1.91–4.05 | 5.46–10.44 | -3.47–-1.32 | 0.552–0.890 |
| Power [W] | 1313.56–2804.51 | 3595.99–6824.38 | -2387.37–-924.74 | 0.566–0.892 |

These differences are not estimates of normal static DC loss. Ideal MPP and
measured inverter operating point need not coincide; operating restrictions,
availability, model assumptions and measurement anomalies remain unresolved.

## Topology and CEC-current evidence

Diagnostic slice: POA ≥600 W/m², finite model/measurement pairs; no healthy filter.

- **20 modules/string:** inverter median measured-Vdc/module-Vmp ratios are
  **19.39–19.86** (23 voltage-equipped inverters). This broadly supports 20 as a
  provisional hypothesis, but cannot verify wiring independently of thermal and
  operating-point assumptions.
- **6 parallel strings:** median measured-Idc/module-Imp ratios are **5.39–5.65**;
  per-inverter p90 ratios are **6.02–6.21**. Six strings remains compatible with
  the observations. Lower medians do not establish another topology or K_DC.
  Low-current periods remain included; some p10 ratios are zero.
- Measured current exceeds CEC `Idcmax=39.439404 A` in **1,503,155** retained
  inverter records; maximum measured current is **55.843 A**. The CEC value also
  numerically equals `Pdco/Vdco = 28199.173828/715` to rounding. This identity is
  evidence about the selected database row, not proof of a manufacturer's hard
  current limit. Confirm the real inverter/datasheet limit and variant before
  imposing any operating restriction. No current cap is applied.
- **82,087** retained voltage records exceed CEC `Vdcmax=800 V`. Ideal 20-module
  Voc reaches **928.48 V**, so this limit/variant also needs review. Voltage above
  MPP or the selected CEC field alone is not classified as a fault.

No topology, CEC parameters, Faiman coefficients or POA treatment was changed.

## Bias interpretation and next calibration prerequisites

All finite-score inverter summaries have negative MBE in voltage, current and
power: the ideal model overpredicts on average. This is a full-record diagnostic,
not evidence that one static DC-loss factor explains the differences. Nighttime,
low-output periods and inverter operating-point restrictions remain included.
Clipping treatment is unchanged: ideal DC is not capped, and this run performs
no inverter AC calculation or clipping compensation. Plant meter data are unused.

Faiman's predicted temperature range is finite and does not expose a numerical
failure. The slightly low measured/module voltage ratios warrant thermal/MPP
review, but there are no measured cell temperatures here to isolate u0/u1 error.
The observed current/power deficits likewise cannot distinguish POA calibration,
optical effects, normal DC loss and operating-point restrictions. POA=1400 W/m²,
nighttime POA anomalies, wind-floor and hourly environment behaviour remain
suspicions, not demonstrated reasons to change the accepted preparation.

No new cleaning incompatibility was found. The two preserved voltage extremes
now have quantitative model-incompatibility evidence, but were already flagged;
their source validity requires review before any additional replacement. No
measurement or QC rule was changed.

Before estimating K_DC, supply reviewed healthy periods, confirm telemetry/units
and equipment mapping, inspect temperature/POA representativeness, and separate
MPP-compatible operation from startup, clipping and unavailable operation. Review
the CEC variant and voltage/current limits. A static DC factor must not absorb
temperature effects, inverter efficiency/clipping, outages, faults or degradation;
long-term dataset changes must not be assumed to be static loss. Keep unresolved
DST intervals visible until an explicit policy is agreed. No calibration dates,
healthy algorithm, exclusion threshold or factor value is chosen by this run.

## Retained extreme-voltage evidence

| Timestamp (Pacific wall clock) | Inverter | Measured Vdc | Ideal array Voc |
| --- | --- | ---: | ---: |
| 2019-11-04 08:45 | 02 | 1750.755 V | 857.41 V |
| 2022-06-27 13:15 | 04 | 1487.769 V | 811.74 V |

Both were already QC-flagged and remain unchanged. They are incompatible with
the current model's 20-module voltage bounds at those inputs, and require source
review. They are not used to revise topology or removed from diagnostics.
No negative DC readings were found in the prepared V/I channels. Other above-CEC
or suspicious ranges remain unproven; no new cleaning rule is introduced.

## Review artifacts and pending decisions

- [Inverter-level table](../outputs/2107_dc_ideal/inverter_dc_ideal.parquet)
- [Nominal ideal baseline](../outputs/2107_dc_ideal/nominal_dc_ideal.parquet)
- [Source-channel mapping](../outputs/2107_dc_ideal/channel_mapping.csv)
- [Run provenance](../outputs/2107_dc_ideal/run_provenance.json)
- [Example V/I/P time series and scatters](../outputs/2107_dc_ideal/inv_01_2018-06-15_dc.png)

Twelve PNGs cover IDs 01/04/05/24 and three illustrative summer/winter dates;
no calibration/validation split is implied. Blank inverter-05 measured voltage
and power traces are intentional.

Next review should address real equipment mapping, healthy operating periods,
MPP versus measured operating-point differences, the two voltage extremes and
CEC hardware-limit/variant interpretation. Faiman coefficients, direct POA use,
20×6 topology, unset K_DC, inverter variant/Paco, clipping discrepancy, wall-clock
DST ambiguity and other prepared-data anomalies remain unresolved and unchanged.
