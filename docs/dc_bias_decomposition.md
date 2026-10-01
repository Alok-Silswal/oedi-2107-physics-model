# Diagnostic decomposition of DC bias

**The dominant remaining asymmetry is current/irradiance-side, with a smaller
wind/thermal-associated voltage bias. Cause is not uniquely identified. K_DC
remains unset; no correction or parameter is adopted.**

Run:

```powershell
.\.venv\Scripts\python scripts/decompose_dc_bias.py
```

## Method and safeguards

The existing 5,628 candidate timestamps remain frozen. The 39,440 records passing
through the prior rating-guard stage provide a broader comparison, before MPP,
consensus and ramp selection. Ratios use existing simultaneous plant medians:
23 equipped inverters for voltage/power and all 24 for current. Inverter 05 is
not reconstructed. Original ideal DC, measurements, flags and timestamps remain
unchanged; output goes to `outputs/2107_dc_bias/`.

Raw metadata confirms latitude 38.996306°, longitude −122.134111°, elevation
10 m and mount tilt/azimuth 25°/180°. Its timezone comment specifies US/Pacific.
Only a diagnostic timestamp copy is localized to `America/Los_Angeles` for
[pvlib solar position](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html)
and [AOI](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.aoi.html).
Ambiguous/nonexistent times become NaT without inference; all candidates have
valid daylight geometry. Saved timestamps stay naive Pacific wall clock.
Solar AM/PM is east/west of solar noon, determined by azimuth <180°/≥180°.
Default atmospheric temperature 12 °C is used only in solar position.

A fixed-PST (`Etc/GMT+8`) calculation checks sensitivity to clock interpretation;
it is not a timestamp correction. AOI is the beam sun-vector angle, not an
effective diffuse-field angle. No transposition or IAM is applied.

AM/PM matching uses POA bins of 100 W/m², original estimated-temperature bins
of 10 °C, and DJF/MAM/JJA/SON seasons. Additional comparisons match year,
10° AOI bins or 1 m/s wind bins. Narrow-bin sensitivity uses 50 W/m² and 5 °C.
Bins start at integer multiples of their width. All occupied cells are saved;
shared summaries require ≥10 records per phase, with a ≥5 sensitivity. These
cutoffs reduce sparse-cell instability, not candidate eligibility. Cell medians
are weighted by min(AM count, PM count), giving equal phase representation.
These are binned comparisons, not exact matches or independent paired samples.

Spearman correlations and quantiles are descriptive; no regression, causal
effect, significance test, optimization or period split is fitted. All finite
and excluded counts are reported. Serial dependence and selection remain.

## Matched AM/PM evidence

The main year-matched comparison has 54 shared cells, with balanced weight
1,383 records per phase. Actual observations are not duplicated or resampled.

| Matching | Voltage AM / PM | Current AM / PM | Power AM / PM | Balanced records per phase |
|---|---|---|---|---:|
| POA, temperature, season | 0.9854 / 0.9781 | 1.0236 / 0.8815 | 1.0074 / 0.8602 | 1,751 |
| Plus year | 0.9847 / 0.9778 | 1.0300 / 0.8772 | 1.0130 / 0.8575 | 1,383 |
| Plus year and AOI | 0.9855 / 0.9777 | 1.0256 / 0.8812 | 1.0105 / 0.8608 | 583 |
| Plus year and wind | 0.9916 / 0.9882 | 0.9915 / 0.8582 | 0.9836 / 0.8473 | 175 |
| Narrow POA/temperature, plus year | 0.9855 / 0.9765 | 1.0197 / 0.8794 | 1.0039 / 0.8581 | 565 |

The year-matched power gap is 0.1554 ratio units; the current gap is 0.1528,
while the voltage gap is 0.0069. This strongly points toward a current-side
systematic effect. Independently aggregated medians do not obey an exact V×I
identity; their median closure ratio is 0.9997, making that distinction small
here without claiming an exact factor decomposition.

The broader preliminary population also shows a year-matched current gap of
0.1804 and voltage gap of 0.0055. The asymmetry is not created solely by final
candidate selection. Fixed-PST interpretation leaves a current gap of 0.1549:
clock interpretation affects AOI substantially in summer, but does not remove
the observed phase contrast. Neither source clock correctness nor POA-channel
clock agreement has been proved by this check.

## Current, AOI and effective irradiance

Current ratio correlates strongly with local hour (Spearman −0.826). Within AM,
its AOI correlation is +0.632; within PM it is −0.600. The opposite branches
matter more than the combined AOI correlation (−0.314).

| AOI band | AM current median | PM current median | AM / PM records |
|---|---:|---:|---:|
| 40–50° | 1.018 | 0.903 | 1,043 / 916 |
| 50–60° | 1.052 | 0.885 | 913 / 1,200 |
| 60–70° | 1.121 | 0.851 | 52 / 726 |

The contrast persists after matching AOI along with POA, temperature, season
and year. A scalar AOI-only IAM is therefore unlikely to be sufficient. An
attenuating module IAM alone would also move already >1 morning measured/model
ratios farther above one. Diffuse fractions are unavailable here, so these
observations do not isolate optical loss, shading or sensor response.

**Direct POA-as-effective-irradiance remains a provisional production assumption,
but is not adequate evidence for a stable calibration baseline.** Review POA
sensor orientation/coplanarity, relative sensor/module angular response, spatial
shading, cross-channel time semantics and irradiance representativeness. These
are hypotheses supported by the current pattern, not confirmed causes. A
diagnostic optical/sensor-response comparison deserves testing next; blindly
adding symmetric module IAM does not.

## Voltage and fixed Faiman sensitivity

Voltage ratio has weak overall estimated-temperature correlation (+0.114) but
stronger wind correlation (−0.517; −0.415 in the broader population). Its median
is 0.992 at wind 1–2 m/s, 0.973 at 3–4 and 0.970 at 4–6. Temperature response
is nonmonotonic: medians are approximately 0.983 at 30–40 °C, 0.979 at 40–50
and 1.004 at 50–60. Wind/thermal dependence merits targeted investigation, but
does not establish that Faiman coefficients alone cause the voltage bias.

[pvlib's Faiman defaults](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html)
come from empirical open-rack measurements, rather than this site. Fixed
perturbations use u0 20/25/30 (±20%) and u1 5.13/6.84/8.55 (±25%), one at a
time and combined. These are illustrative plausible sensitivity bounds around
the defaults, not measured site coefficients or literature confidence intervals.
CEC parameters and topology stay unchanged. The subset and matching bins use
the original temperature for every scenario; no scenario is optimized or adopted.

| Scenario (u0, u1) | Median temperature change | Overall voltage median | Matched voltage PM−AM |
|---|---:|---:|---:|
| Reference (25, 6.84) | 0 °C | 0.9817 | −0.0069 |
| Lower u0 (20, 6.84) | +1.61 °C | 0.9886 | −0.0109 |
| Upper u0 (30, 6.84) | −1.27 °C | 0.9759 | −0.0042 |
| Lower u1 (25, 5.13) | +1.40 °C | 0.9878 | −0.0062 |
| Upper u1 (25, 8.55) | −1.14 °C | 0.9767 | −0.0072 |
| Combined warmer (20, 5.13) | +3.53 °C | 0.9970 | −0.0103 |
| Combined cooler (30, 8.55) | −2.24 °C | 0.9719 | −0.0047 |

Reasonable illustrative thermal variation can explain much of the *overall*
voltage offset, but none of these scenarios removes its AM/PM gap. Improving
the overall offset can worsen the phase gap and wind association. Wind matching
reduces the observed voltage gap, but changes coverage sharply (175 balanced
records), so it does not establish a causal contribution.

The ±1 °C CEC derivative gives median array dVmp/dT ≈−3.269 V/°C. The voltage
deficit is equivalent to a median +3.93 °C temperature offset under a local
linearization. This is not measured temperature: irradiance, actual MPPT voltage,
wiring/sensor bias and module parameters can also affect voltage.

Across the seven thermal scenarios, overall current median remains
0.91407–0.91442 and the matched current gap stays approximately 0.1528. Thermal
changes of this size cannot explain the large current/power phase asymmetry.

## Calendar and operating constraints

Month, season, year, phase and operating-load summaries are retained. Seasonal
voltage/current medians are DJF 0.984/0.955, MAM 0.981/0.916,
JJA 0.969/0.919 and SON 0.989/0.890. Coverage is uneven and associations are
confounded. The 208 September-2018 candidates have voltage/current medians
1.005/0.698, consistent with a large common current-side discrepancy that
thermal voltage correction does not explain. This is not a degradation estimate.

All candidates remain below the prior conservative 90% CEC AC/DC guards. Low
ratios occur well below those guards, not only near a power ceiling: for maximum
measured AC 20–40% of Paco, the median power ratio is 0.808. This is a descriptive
load slice, not controlled evidence, because measured load depends on the bias.
Normal clipping near the unresolved ~30 kW level is unlikely to explain the whole
pattern, but shared curtailment, shared shading, unavailable strings, MPPT
constraints or telemetry bias remain possible. Consensus cannot prove health.
No new AC prediction, clipping cap or inverter fit is introduced.

## Review before calibration

- Targeted temperature/wind checks are justified; fixed Faiman changes alone do
  not explain the dominant asymmetry. Obtain independent module-temperature and
  wind-height/exposure evidence where possible.
- Prioritize irradiance representation and common-mode operation. An optical
  diagnostic should distinguish sensor response, module IAM, diffuse light and
  asymmetric shading rather than assuming an AOI-only correction.
- No cleaning error is established. A focused cross-channel timestamp/POA-sensor
  audit is newly motivated; no shifting, filling or cleaning redesign was done.
- A constant K_DC might become defensible only after an independently supported
  systematic correction yields stable ratios across phase, irradiance, temperature
  and year in independently verified healthy/MPP operation. This run does not
  demonstrate that outcome or justify calibration.

Artifacts include AM/PM cell tables (including sparse/unmatched cells), binned
ratio/count tables, correlations, counterfactual thermal outputs, phase/calendar
summaries, four standalone PNGs, a concise generated report and source SHA256
provenance. All six consumed source files have unchanged hashes. Production
physics functions, configuration, inverter parameters and cleaned data are untouched.
