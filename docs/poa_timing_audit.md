# Focused POA timing / representativeness audit

Run `scripts/audit_poa_timing.py` with the project Python. Results are in
`outputs/2107_poa_timing_audit/`. Only the seven requested offsets were evaluated.

## Metadata findings

- POA metric `poa_irradiance_o_149574` (149574) is named `poa_irradiance`,
  W/m², aggregation `avg`, scale 1, offset 0, source type `OTHER`, source ID null.
  No sensor tilt/azimuth, manufacturer/model, serial number or individual
  placement is given. Its association with weather station `ws_fss_1` (5294)
  cannot be confirmed from the null source link; that station's manufacturer
  is Unknown and model/serial fields are empty.
- The **array** mount is 25°/180°. This does not establish POA sensor orientation;
  metadata neither supports nor excludes a different sensor plane.
- Inverter current also has aggregation `avg`, source type `INVERTER`, null
  source ID. Inverters/weather station have `time_interval: L`, but the provided
  metadata does not define that code or establish start/end/center averaging
  windows, clock synchronization or averaging duration per channel.
- System metadata explicitly identifies US/Pacific (`PST8PDT`). Audits support
  Pacific wall-clock exports, with unresolved fall ambiguity. Separate POA and
  electrical CSVs, different gaps/coverage and different source categories
  suggest distinct reporting paths; no logger IDs prove separate clocks.
- Notebook 06 documents the exact POA join, no filling/interpolation/shift/UTC
  conversion, and preserved QC/DST flags. Prepared Parquet has no embedded
  provenance metadata. Notebook 05 previously favored zero POA-to-AC offset
  overall and in every year (equal-support active correlation 0.9469); that
  broader alignment result was inspected, not rerun.

## Fixed-offset comparison

At electrical timestamp **t**, positive offset means evaluating measured
**POA(t + offset)**. Original POA and timestamps are never edited. Temperature
remains the existing estimate at t, isolating the timing/current effect; no
Faiman rerun, fitting or IAM occurs. The unchanged CEC module model and six
parallel strings produce diagnostic expected current.

All seven lookups have finite positive, existing-QC-clear POA on all 5,628 frozen
candidates: equal support is 5,628, exclusions 0 (AM 2,469; PM 3,159).
Phase is the original solar AM/PM classification from Task 8.
Ratios use the simultaneous 24-inverter median measured current. Dispersion is
sample standard deviation (ddof=1) and IQR across timestamp ratios; nothing is
clipped or removed to improve a lag score.

| Offset, minutes | AM median | PM median | AM−PM | Std | IQR |
|---:|---:|---:|---:|---:|---:|
| −15 | 1.1397 | 0.8032 | 0.3365 | 0.2795 | 0.3414 |
| −10 | 1.0990 | 0.8275 | 0.2716 | 0.2016 | 0.2752 |
| −5 | 1.0634 | 0.8527 | 0.2107 | 0.1322 | 0.2112 |
| 0 | 1.0273 | 0.8842 | 0.1431 | 0.0989 | 0.1421 |
| +5 | 1.0013 | 0.9152 | 0.0862 | 0.0917 | 0.0933 |
| +10 | 0.9716 | 0.9510 | 0.0206 | 0.1986 | 0.0494 |
| +15 | 0.9454 | 0.9875 | −0.0420 | 0.2417 | 0.0697 |

**+10 minutes is best by smallest absolute median phase gap**, reducing it by
85.6% and IQR by 65.2%. It meaningfully removes most of the *central* asymmetry,
but does not eliminate it consistently or identify a clock correction. Std
doubles, with 15 ratios above 2 and maximum 6.11 (zero-offset maximum 1.58).
These are retained observations, not newly flagged corrupt data. Nearby transient
POA changes can become mismatched under the lookups. **+5 is best by std**, with
only a 7.3% reduction and a remaining gap of 0.0862. +15 reverses the phase gap.
No single offset improves all diagnostic criteria.

## Common-mode consistency and next action

At zero offset, AM>PM in all six represented years (gaps 0.094–0.151), all four
seasons (0.125–0.173), and all 24 inverters (0.126–0.171). The median same-time
fraction of inverters with ratio>1 is 100% in AM and 0% in PM. This supports a
common input/model/operation issue, not an inverter-specific explanation for
the dominant phase pattern; it does not diagnose individual health.

At +10, year gaps range −0.0047 to +0.0586 and season gaps +0.0048 to +0.0401;
23 inverter gaps remain positive, with one essentially zero. Coverage is uneven,
especially the 2018 September-only slice. The uniform sign at zero is robust,
but correction size is not fully stable across calendar slices.

POA representativeness remains a major concern, alongside clock/averaging-support
semantics. Smooth geometric sensor mismatch can resemble a timing offset, so
this audit cannot rank orientation versus true clock delay conclusively. The
prior broader alignment result favoring zero and new tail deterioration argue
against automatically shifting production POA by +10 minutes.

Recommended next action: obtain sensor mounting/placement and logger clock/
averaging-window documentation; inspect independently verified rapid irradiance
transitions to distinguish real delay from geometric response. Then test only
the supported timing or sensor-response explanation on held-out periods before
any K_DC calibration. Retain existing zero-offset alignment in the meantime.

Source hashes are unchanged for prepared data, Task-8 context, metadata and the
three inspected notebooks. Only four new-code tests were run; all passed.
