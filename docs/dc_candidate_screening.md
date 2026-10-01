# Candidate DC calibration periods

Run from the project root:

```powershell
.\.venv\Scripts\python scripts/identify_dc_candidates.py
```

This stage reads the existing ideal DC comparison and prepared AC channels. It
does not rerun physics, change cleaning, fit parameters, or select a data split.
`K_DC` remains unset. Output is a **candidate mask, not verified healthy/MPP operation**.

## Common screening rules

All conditions apply at the same original Pacific wall-clock timestamp. All 24
currents and measured AC channels participate. Voltage/power checks use the 23
equipped inverters; inverter 05 voltage/power stays missing. No per-inverter
historical baseline or loss factor is learned.

| Rule, in order | Explicit value / calculation | Rationale |
|---|---|---|
| Model inputs | Existing input-available/computed flags and finite ideal V/I/P | Required physics inputs/predictions must exist. |
| Known QC / DST | Exclude electrical QC, irradiance QC and fall-DST flags | Avoid documented suspicious observations and ambiguous clock records; no source changes. |
| Irradiance | POA ≥ 300 W/m² | Provisional moderate-light screen to reduce low-light/startup sensitivity. Not a universal threshold. |
| DC/AC availability | Finite V/I/P for all equipped channels; positive V/I/P; finite AC for all 24 | No incomplete plant or zero-current array is accepted. |
| Operation | Every measured AC ≥ 0.10 × Paco = 2,760 W | Provisional startup/shutdown/outage guard; AC is screening telemetry only. |
| Rating guards | Every measured AC < 0.90 × Paco = 24,840 W **and** ideal DC < 0.90 × Pdco = 25,379.256445 W | Conservatively avoid clipping/curtailment near rating, including high ideal demand with depressed measured AC. These are not estimates of actual clipping. |
| MPP-voltage compatibility | Every equipped measured/ideal voltage ratio in [0.90, 1.10] | Broad provisional tolerance around predicted MPP; does not demonstrate MPPT. |
| Plant consensus | Every current and AC within ±10% of its simultaneous 24-inverter median | Reject a persistently weak inverter using a common plant reference, rather than absorbing it into a personal baseline. |
| Adjacent POA stability | Both exact ±5-minute neighbours exist; max absolute POA change / central POA ≤ 0.20 | Provisional steady-input proxy for quasi-steady thermal/MPP calculations. No gap filling or timestamp shifting. |

CEC parameters are unchanged: exact entry
`ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_`, Paco 27,600 W,
Pdco 28,199.173828 W. AC metadata confirms kW, converted to W for screening.
Idcmax is not used as a current rejection limit. No AC predictions or meter data
are involved. No ratio target selects a desired loss percentage.

One-at-a-time sensitivity uses POA 200/300/400 W/m², AC floor 5/10/15% of
Paco, rating guards 80/90/95%, voltage tolerance 5/10/15%, consensus tolerance
5/10/15/20%, and adjacent-POA tolerance 10/20/30% or disabled. These are
transparent screening alternatives, not optimized thresholds. Strict all-inverter
consensus deliberately trades coverage for protection against individual weakness.

## Outputs and interpretation

`outputs/2107_dc_candidates/` contains:

- `timestamp_candidate_masks.parquet`: every original timestamp, each independent
  rule and common final `candidate_healthy_mask`.
- `inverter_candidate_masks.parquet`: timestamp, inverter ID, common candidate
  mask and `dc_power_eligible`. Inverter 05 always has false power eligibility.
- `timestamp_diagnostics.parquet`: unchanged weather/ideal quantities plus
  simultaneous plant medians, measured/ideal ratios and screening diagnostics.
- Filter counts, threshold sensitivity, per-inverter coverage/disagreement,
  preliminary/candidate distributions, POA/temperature/hour/season/year bins,
  joint condition tables, monthly row counts and contiguous candidate runs.
- Standalone PNGs, review-date provenance and source SHA256 provenance.
- `report.md`: compact run counts, thresholds and ratio summaries.

Power ratios use the median of the **23 measured powers**, divided by common
ideal power; they are not the product of independently computed median V/I.
Current ratios use all 24 currents. All ratios are descriptive measurements,
never a fitted or recommended K_DC. Summary bins are not power resampling or
measurement alignment. Sparse bins and joint weather/calendar confounding need
review. Plot axis limits only affect display, never tables or masks.

The preliminary comparison population passes through the rating-guard stage;
the final population additionally passes voltage, consensus and ramp rules.
Consensus cannot exclude shared faults, shared curtailment, common POA bias or
plant-wide degradation. Voltage screening also conditions assessment of thermal
bias. Candidates must not automatically become calibration data.

## Completed run

| Stage | Remaining timestamps | Removed at this stage |
|---|---:|---:|
| Original | 622,294 | 0 |
| Valid model inputs | 526,068 | 96,226 |
| Known QC / DST clear | 525,914 | 154 |
| POA ≥ 300 | 178,752 | 347,162 |
| DC/AC available and DC positive | 114,364 | 64,388 |
| All operating above AC floor | 79,756 | 34,608 |
| Below rating guards | 39,440 | 40,316 |
| MPP-voltage compatible | 35,094 | 4,346 |
| All-inverter consensus | 6,661 | 28,433 |
| Stable adjacent POA | 5,628 | 1,033 |

These represent 129,444 candidate power/voltage pairs across 23 inverters and
135,072 current pairs across 24. No minimum run duration is imposed; exact
five-minute contiguous runs are reported separately. Counts describe records,
not independent statistical samples or proven healthy duration.

| Plant-median ratio | Preliminary p10 / median / p90 | Candidate p10 / median / p90 |
|---|---|---|
| Voltage | 0.9595 / 0.9820 / 1.0054 | 0.9627 / 0.9817 / 1.0026 |
| Current | 0.7753 / 0.9290 / 1.0890 | 0.8357 / 0.9142 / 1.0603 |
| Power | 0.7699 / 0.9097 / 1.0688 | 0.8202 / 0.8978 / 1.0401 |

Distributions narrow, but substantial systematic variation remains. Candidate
POA p10/median/p90 is 368.8/564.8/696.2 W/m²; ambient temperature
12.94/25.17/34.18 °C; estimated cell temperature 24.21/37.84/47.81 °C.
Ideal DC power is 10.40–25.38 kW (median 20.28 kW); measured plant-median
DC power is 6.00–24.95 kW (median 18.87 kW). Ranges are of timestamp medians,
not the full range of individual devices.

**A single global static loss factor is not supported for calibration yet.**
Candidate median power ratio is 1.024 at 09:00, 0.865 at 16:00 and 0.840 at
17:00. Within POA 500–600 W/m² it is 1.023 at 09:00 versus 0.859 at 16:00.
There are 1,385 candidate timestamps with plant-median power ratio above one;
a derate constrained to ≤1 cannot explain those observations.

The association persists within year and broad weather bins: for 2022,
POA 500–600 W/m² and estimated cell temperature 30–40 °C, the median power
ratio is 1.041 at 09:00 (115 timestamps) versus 0.866 at 16:00 (41 timestamps).
Bin conditioning reduces confounding but does not establish its cause.

Power ratio medians vary with POA (0.842 at 300–400, 0.929 at 500–600,
0.904 at 600–700 W/m²), estimated temperature (0.970 at 20–30 versus 0.882
at 40–50 °C), and season (0.934 DJF, 0.887 JJA, 0.902 MAM, 0.877 SON).
These are confounded descriptive associations, not diagnoses or fitted effects.

Coverage is uneven: 2018/2019/2020/2021/2022/2023 retain
208/121/1,099/1,835/2,161/204 timestamps; none remain in December 2017.
The 2018 records are all in September and have median power ratio 0.703.
Other year medians are 0.900/0.880/0.900/0.906/0.947. This is not a degradation
estimate. Shared underperformance can pass consensus, and calendar selection is
strong. Do not treat the unusually low 2018 slice as normal static DC losses.

Sensitivity materially affects eligibility: consensus 5/10/15/20% retains
4,431/5,628/6,409/20,439 timestamps; rating guard 80/90/95% retains
3,792/5,628/6,734. POA 200/300/400 retains 6,742/5,628/4,783. AC-floor
changes have no effect on the final mask. No preferred parameter is chosen from
these outcomes.

Measured AC/DC consistency is broadly reasonable in the candidates: the median
of each timestamp's maximum AC/DC ratio across the 23 equipped inverters is
0.993, p90 is 0.998, and the maximum is 1.050. These readings do not indicate
a large AC/DC unit mismatch; small ratio excursions still warrant telemetry
review. No empirical inverter efficiency is fitted or used to select candidates.
Preliminary current disagreements greater than 10% of the simultaneous median
occur most often for inv_21 (35.6%) and inv_19 (27.9%), followed by inv_02
(24.0%). These are descriptive disagreement rates, not fault diagnoses. The
strict common mask never normalizes such deviations away per inverter.

There are 1,282 contiguous candidate runs (median 2 samples, maximum 28).
The short/uneven coverage reinforces the need to review representativeness.

Before calibration, independently review common-mode operation and persistent
inverter differences, POA sensor orientation/representativeness and timing,
thermal proxy/wind assumptions, and the actual clipping/curtailment regime.
Use joint-bin diagnostics and maintenance/availability records to separate these
possibilities. A time-of-day association alone does not justify shifting POA,
redoing cleaning, changing Faiman, or fitting a loss factor. All existing
assumptions, actual-ID uncertainty and nighttime conventions remain unchanged.
