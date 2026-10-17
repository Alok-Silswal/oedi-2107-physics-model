# OEDI System 2107: frozen Stage 1 physics and research diagnostics

This package implements measured POA Ã¢â€ â€™ Faiman temperature Ã¢â€ â€™ explicit CEC
single-diode module calculation. The requested ideal 20-series Ãƒâ€” 6-parallel
scaling gives MPP for one inverter's DC array. Block 4 adds an optional global
static DC power derate with its numerical factor currently unset. Block 5 adds
one inverter's DC-to-AC conversion using official CEC/Sandia parameters. Block 6
retains 24 named inverter records and sums AC at each original timestamp.
Block 7 provides reusable exact-key comparison metrics and arithmetic residuals.
Real-data validation is complete; Stage 1 is frozen and uncalibrated. No DC-loss
factor or Sandia inverter coefficients have been fitted.

The canonical [frozen Stage 1 route](docs/research_baseline.md) is
`scripts/run_frozen_research.py` using ideal pre-loss DC, bypassing unset K_DC,
and the documented 30 kW inverter ceiling extension. The optional Block 4Ã¢â‚¬â€œ6
interfaces below remain available; they are not the frozen research route.
Read the [saved Stage 1 summary](docs/stage1_validation_summary.md) without
regenerating it.

The [short-term forecasting benchmark](docs/short_term_forecasting_benchmark.md)
is complete at 5, 15 and 30 minutes. The earlier
[readiness report](docs/stage2_readiness.md) and
[inconclusive decline experiment](docs/performance_change_feasibility.md)
preserve historical recommendations and findings; their next-action statements
describe the status at the time, not the current plan.

**Agreed Stage 2 direction: weather Ã¢â€ â€™ frozen physics calculation Ã¢â€ â€™ learned
correction.** `scripts/forecast_stage2_weather.py` implements the first component:
15-minute weather forecasts. `scripts/calculate_stage2_physics.py` applies frozen
physics and constructs correction labels. The classical correction comparison is implemented in `scripts/compare_stage2_correction.py`. The earlier historical-feature AC
benchmark does not implement that physics-plus-ML route.

## Setup and run (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install -e ".[test,real-data]" -c requirements-lock.txt
.\.venv\Scripts\python examples/run_foundation.py
.\.venv\Scripts\python examples/run_inverter.py
.\.venv\Scripts\python -m pytest
```

These setup commands create/install into `.venv`. The examples print synthetic
results without writing scientific outputs; pytest may write caches and temporary
test files. `test` supplies pytest; `real-data` supplies script-diagnostic imports:
matplotlib, scikit-learn, scipy, joblib, threadpoolctl and XGBoost. Joblib/threadpoolctl/scipy
are direct imports even though other packages also depend on them. PyArrow is
already a base dependency. Plotting/ML support packages such as contourpy and
cloudpickle are transitive and belong in the lock, not the optional declaration.

Run scripts from the repository root only when regeneration is intended:

| Command (`.venv/Scripts/python` followed by) | Files written or regenerated |
|---|---|
| `scripts/run_frozen_research.py` | Frozen model/validation outputs in `outputs/2107_research_baseline/`; reruns overwrite results. |
| `scripts/show_stage1_results.py` | Reads saved results, then overwrites `stage1_validation_summary.txt` and `stage1_validation_metrics.csv` in that output directory, and `docs/stage1_validation_summary.md`. No physics calculation, but **not read-only**. |
| `scripts/benchmark_short_term_forecasting.py` | Fits the fixed benchmark and regenerates models, predictions, scores, plots and provenance in `outputs/2107_short_term_forecasting/`. |
| `scripts/forecast_stage2_weather.py --prepare` | Loads data, checks shared splits and prints eligibility/planned work; no fitting or output writes. |
| `scripts/forecast_stage2_weather.py --step training_1` (then `training_2`, `training_3`, `validation`) | Runs only the named weather step; compatible completed fits/results are reused in `outputs/2107_stage2_weather/`. |
| `scripts/calculate_stage2_physics.py --prepare` | Shared preparation only; no physics calculation or output writes. |
| `scripts/calculate_stage2_physics.py --step training_1` (same step names) | Applies frozen physics to saved step weather forecasts and creates correction labels. |
| `scripts/compare_stage2_correction.py` | Four fixed classical correction fits on saved training blocks; validation-only predictions/metrics and reusable fitted bundles in `outputs/2107_stage2_correction/`. Compatible completed fits are reused. |

Earlier diagnostic runners and the data-preparation notebooks also write derived
files as documented in their reports. Their historical evidence is retained;
running them is not required to review saved findings.

The example uses synthetic weather and enables INFO logging, which prints the
exact CEC key, full parameter row and pvlib version. Library callers should enable
INFO logging for `oedi2107.module` to see the same provenance. No database network
access is needed at runtime; `retrieve_sam("CECMod")` reads pvlib's bundled table.
The exact key is configured explicitly; an unavailable key raises an error with
candidates rather than silently choosing another module. No diode parameters
are manually hard-coded.

Verified exact entry in pvlib 0.16.1:
`Hyundai_Heavy_Industries_Green_Energy_Co__HiS_M310TI`.
The distinct `HiS_S310TI` entry is not used.
The CEC row reports STC = 309.6 W (36 V Ãƒâ€” 8.6 A) and `N_s=60`;
review these database fields against the nameplate/datasheet. They are retained
unchanged. CEC STC Ãƒâ€” 2880 is 891.648 kW, distinct from nominal 892.8 kW.

`requirements-lock.txt` is the existing flat installed-version snapshot for the
base, test and script-diagnostic dependencies. The setup above installs its exact
versions and constrains the editable install to them. Diagnostic additions retain
every frozen baseline pin; unrelated notebook/kernel tooling is not included.

## Layout and interface

- `src/oedi2107/config.py`: system metadata, inferred topology, thermal assumptions.
- `inputs.py`: measured-input validation; no irradiance transposition.
- `thermal.py`: `estimate_cell_temperature(poa, temp_air, wind_speed, config)`.
- `module.py`: database loading and explicit `calcparams_cec` Ã¢â€ â€™ `singlediode`.
- `array.py`: ideal one-inverter array scaling.
- `dc_losses.py`: `DCLossConfig` and `apply_dc_losses` (global DC derate).
- `inverter.py`: exact CEC retrieval and explicit `pvlib.inverter.sandia` conversion.
- `plant.py`: named inverter records, strict complete-roster aggregation and
  `PlantPrediction` containing both output tables.
- `validation/`: strict paired metrics and separate DC, inverter AC and plant
  interfaces, with explicit normalization, external labels/masks and sample counts.
- `pipeline.py`: `run_dc_foundation` (ideal only), `run_dc_model` (through Block 4),
  `run_inverter_model` (through Block 5, one inverter only), and `run_plant_model`
  (through Block 6, inverter records plus plant sum).
- `examples/run_foundation.py`: executable synthetic example.
- `examples/run_inverter.py`: standalone synthetic post-loss DC conversion.
- `tests/test_physics.py`: physical sanity and input/provenance checks.
- `tests/test_dc_losses.py`: global-factor validation, preservation and consistency.
- `tests/test_inverter.py`: official CEC preservation, clipping, tare and integration.
- `tests/test_plant.py`: identity, completeness, signed sums and pipeline integration.
- `tests/test_validation.py`: known metrics, finite-pair accounting, identity,
  externally supplied labels/masks and no model fitting.

Supply a pandas DataFrame with a unique, increasing timezone-aware DatetimeIndex
and columns `poa` [W/mÃ‚Â²], `temp_air` [Ã‚Â°C], `wind_speed` [m/s]. Inputs must be aligned
and finite. Negative POA/wind and missing readings raise errors; decide sensor
offset correction and data quality treatment upstream. POA is retained unchanged.
Extra measurement channels (DC voltage/current, inverter AC, plant meter) are
not used. The model does not infer timestamps or the site's timezone.

Output retains `temp_cell` [Ã‚Â°C], adjusted diode parameters (photocurrent and
saturation current [A], series/shunt resistance [ohm], `nNsVth` [V]), module
`Vmp`, `Voc` [V], `Imp`, `Isc` [A], `Pmp` [W], and `expected_dc_voltage` [V],
`expected_dc_current` [A], `expected_dc_power` [W]. In the ideal foundation,
zero POA gives exactly zero DC electrical outputs; its infinite shunt resistance is the CEC model's mathematical
limit, not a missing value. Results preserve timestamps and carry pvlib/key
provenance in DataFrame attributes (attributes are not preserved by CSV).

## Block 4: global static DC loss

`SystemConfig().dc_losses.k_dc` is **None (unset/uncalibrated)**. No numerical
factor has been selected or estimated, and no PVWatts 14% assumption is used.
One dimensionless scalar K_DC is common to all 24 nominally identical inverter
arrays; it is static over time. The model has no separate per-inverter factors.

For a supplied finite scalar with `0 < K_DC <= 1`, the exact calculation is:

```text
expected_dc_power   = expected_dc_power_ideal * K_DC
expected_dc_voltage = expected_dc_voltage_ideal
expected_dc_current = expected_dc_power / expected_dc_voltage  (voltage > 0)
expected_dc_current = 0                                      (voltage = power = 0)
dc_loss_factor      = K_DC
```

This is an aggregate power adjustment: voltage remains the ideal MPP baseline,
and current is an equivalent consistent with the adjusted power. This does not
predict a physical voltage drop, a new MPP or a lossy I-V curve. The loss function
rejects inconsistent ideal inputs, including nonzero power at zero voltage.

`run_dc_model` retains `expected_dc_voltage_ideal`, `expected_dc_current_ideal`,
`expected_dc_power_ideal`, `dc_loss_factor`, `expected_dc_voltage`,
`expected_dc_current` and `expected_dc_power`, together with module and weather
outputs. Without a factor, `dc_loss_factor` and post-loss current/power are NaN
even at night, indicating **not computed**; voltage still retains its baseline.
Metadata marks this state `dc_loss_status="uncalibrated"`. The lower-level
`apply_dc_losses(ideal_dc, DCLossConfig())` raises rather than substituting a value.

```python
from oedi2107 import DCLossConfig, SystemConfig, run_dc_model

# measured_weather uses the input schema above. No factor is assumed:
result = run_dc_model(measured_weather)

# Only after a future reviewed estimate exists:
config = SystemConfig(dc_losses=DCLossConfig(k_dc=reviewed_k_dc))
result = run_dc_model(measured_weather, config)
```

`run_dc_foundation` retains its original ideal-only interface and column names
for Blocks 1Ã¢â‚¬â€œ3. A factor in the configuration does not derate that function's
outputs. Both entry points stop at one array's DC output, without aggregation.

K_DC will later be estimated from **healthy periods of cleaned real operating
data**. Its intended scope is unmodelled normal static baseline DC effects such
as wiring, mismatch and connectors. Calibration must not absorb temperature
effects (already modelled), inverter efficiency or clipping, outages/availability,
inverter faults, abnormal underperformance or long-term degradation. This task
implements no calibration, optimization or healthy-period selection.

The voltage/current simplification and scope of the global estimate remain
provisional. A numerical factor remains pending cleaned healthy data;
the example and tests do not establish a plant K_DC.

## Block 5: explicit CEC/Sandia inverter conversion

`load_cec_inverter` calls `pvlib.pvsystem.retrieve_sam("cecinverter")` and selects
the configured key **exactly**, with no fuzzy fallback:

`ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_`

This is a **provisional US/480 V variant match**, not confirmation of the full
installed nameplate. A second S1B candidate is
`ABB__TRIO_27_6_TL_OUTD_S1B_US_480_A__480V_`. The database also has S, S1 and S1A
versions, each with and without `_A`. All eight rows are identical in the tested
pvlib 0.16.1 database. Retrieval logs all eight keys at WARNING and the selected
key/full row at INFO. Confirm the installed suffix and nominal AC voltage before
Block 6. An unavailable configured key raises; no candidate is automatically chosen.

Provenance: pvlib 0.16.1 bundled SAM snapshot
`sam-library-cec-inverters-2019-03-05.csv`. This is the installed official CEC
table, not a live download. `CEC_Date` is missing in these rows; no certification
date is inferred. The exact key, pvlib version, source path, candidates and full
unchanged row are retained in result `.attrs`; CSV does not retain this metadata.

| Parameter | Official value | Unit |
| --- | ---: | --- |
| Paco | 27600 | W AC |
| Pdco | 28199.173828 | W DC |
| Vdco | 715 | V DC |
| Pso | 92.134544 | W DC |
| C0 | -2.513804e-7 | 1/W |
| C1 | -3.1e-5 | 1/V |
| C2 | -0.001336 | 1/V |
| C3 | -0.001753 | 1/V |
| Pnt | 8.28 | W AC |

`calculate_inverter_ac(post_loss_dc, parameters)` explicitly calls:

```python
pvlib.inverter.sandia(
    post_loss_dc.expected_dc_voltage,
    post_loss_dc.expected_dc_power,
    parameters,
)
```

It preserves all input DC columns and timestamps and adds `expected_ac_power`
[W] for **one inverter**. Finite nonnegative DC inputs are required; `(0 V, 0 W)`
is allowed and positive power at zero voltage is rejected. No ModelChain is used.
Paco and all coefficients remain unchanged; no fitted clipping limit is provided.

Clipping comes solely from Sandia's upper bound of **27600 W**. There is no second
clipping cap. In pvlib 0.16.1, DC below `Pso` gives **-Pnt = -8.28 W**, including
zero input. Signed Sandia AC is retained without clamping negative values to zero.
For later comparisons, first confirm inverter/meter sign conventions and whether
nighttime self-consumption is recorded or suppressed. Any later comparison policy
should preserve raw model AC; no meter conversion or comparison is implemented here.

The [pvlib Sandia documentation](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.sandia.html)
states that the function does **not** enforce MPPT voltage windows or maximum DC
current. The CEC row lists `Mppt_low=520 V`, `Mppt_high=800 V`, `Vdcmax=800 V`,
and `Idcmax=39.439404 A`. Review these against the real device and inferred array:
the six-string ideal STC current is 51.6 A, above the listed `Idcmax`. No current
limit, MPPT cutoff or new operating-point model has been invented in V1.

```python
from oedi2107 import DCLossConfig, SystemConfig, run_inverter_model

# Default K_DC remains unset: ideal DC is available; post-loss DC and AC are NaN.
result = run_inverter_model(measured_weather)

# Once a future reviewed global estimate exists, compute post-loss DC and AC:
config = SystemConfig(dc_losses=DCLossConfig(k_dc=reviewed_k_dc))
result = run_inverter_model(measured_weather, config)
```

Unset K_DC leaves AC **not computed**, even at night. The lower-level conversion
rejects NaN DC instead of assuming a derate. `examples/run_inverter.py` uses
synthetic post-loss DC directly to demonstrate conversion, without choosing K_DC.

The US/480 V and `_A` identity, CEC DC operating limits,
future K_DC, and measured nighttime sign conventions remain provisional. Investigate any measured
clipping discrepancy later using real data; **do not change Paco in V1**.

## Block 6: named inverter outputs and plant AC aggregation

`run_plant_model(measured_weather, config=None, inverter_ids=None)` returns a
`PlantPrediction` with **both**:

- `inverter_outputs`: a long-format DataFrame with `timestamp`, `inverter_id`,
  `expected_ac_power` [W], and all preceding DC, module and weather columns.
  Every timestamp has 24 distinct inverter records, not just a multiplied value.
- `plant_outputs`: a timestamp-indexed DataFrame with
  `expected_plant_ac_power` [W], `inverter_record_count`,
  `finite_ac_prediction_count`, `uncomputed_inverter_ids` and
  `plant_prediction_status` (`complete` or `uncomputed`).

All 24 initially receive one shared nominal model prediction using the same
environmental inputs and `SystemConfig`. No per-inverter calibration or
availability parameter is added. Default IDs `inverter_01` through `inverter_24`
are **stable placeholders**, not measured-device identifiers. Pass an explicit
roster of 24 unique nonempty strings once the real device mapping is reviewed.
Useful model and database provenance is retained in both tables' `.attrs`.

```python
from oedi2107 import run_plant_model

# No K_DC is selected: inverter and plant AC remain uncomputed (NaN).
prediction = run_plant_model(measured_weather)
inverters = prediction.inverter_outputs
plant = prediction.plant_outputs
```

The lower-level functions are `represent_inverter_outputs(nominal, config,
inverter_ids)` and `aggregate_inverter_outputs(inverter_outputs, config,
inverter_ids, expected_timestamps)`. The aggregator uses the **actual named
records**, including any differing predictions in a supplied table:

```text
expected_plant_ac_power(t) = sum(expected_ac_power(t, inverter_id)
                                 for inverter_id in the complete 24-ID roster)
```

Duplicate timestamp/ID pairs, missing IDs, unexpected IDs and empty inputs raise
explicit errors. The expected roster is inherited from representation metadata
unless supplied explicitly. Without metadata or a supplied roster, the default
24 placeholder IDs are required. Missing records are never treated as zero.
NaN AC in an otherwise complete roster is reported with a warning and affected
IDs, and the entire plant sum is NaN: pandas `sum(min_count=24)` prevents an
accidental partial sum. Infinite AC values are rejected. An unset K_DC therefore
remains uncomputed at both levels, including at night.

The representation carries its expected timestamp axis in metadata; aggregation
also accepts an explicit `expected_timestamps` DatetimeIndex. This detects entire
missing timestamps as well as missing inverter records. For external tables
without either timestamp reference, only observed timestamps can be checked.
All timestamps remain timezone-aware and unchanged. There is **no 5-minute to
15-minute resampling**, measured inverter/meter join, meter comparison or energy
conversion. Nominal 5-minute inverter and 15-minute meter resolutions are context
only; power-versus-energy and timestamp semantics remain unresolved.

Signed nighttime outputs are preserved: 24 computed Sandia tare values of
`-8.28 W` sum to **-198.72 W**. There is no zero clamp. This is the inverter AC
sum, without transformer/collection losses or plant auxiliary-load modelling.

Before Block 7, confirm real inverter IDs, timestamp/timezone/interval semantics,
meter instantaneous-versus-average-power-versus-energy meaning, and nighttime
sign conventions. The following remain unresolved and unchanged: provisional
Faiman coefficients, measured POA as effective irradiance, unset numerical K_DC,
inferred 20Ãƒâ€”6 topology, ABB US/480 V variant, CEC `Idcmax` versus inferred array
current, and possible measured clipping discrepancy versus unchanged CEC Paco.
No calibration, residuals, validation metrics, fault or degradation analysis is
implemented by Block 6.

## Block 7 validation-framework foundation

See [validation interface and metric definitions](docs/validation.md) for schemas,
exact equations, missing-data counts and examples. Import from
`oedi2107.validation`: `validate_dc`, `validate_inverter_ac`, `validate_plant_ac`,
and the six reusable metric functions. Each interface returns both the full
outer-union comparison table and per-device/per-signal metrics.

Residual and MBE use **measured minus expected**. Normalized MAE/RMSE require
explicit `Normalization(value, basis)` in the signal's units and return fractions;
no convention or denominator is assumed. Finite exact-key pairs alone enter
metrics, with total/valid/excluded counts. Period labels/masks and healthy masks
are supplied externally; no dates or healthy-operation rules are selected.
No model parameter is fitted, and no 5-minute/15-minute resampling is implemented.
All earlier unresolved assumptions remain unchanged.

## Assumptions and human review

For the first uncalibrated real-data DC integration, see
[run instructions and output definitions](docs/real_dc_run.md). It stops at
ideal/pre-loss DC and preserves the prepared Pacific wall-clock dataset as-is.
The full run and diagnostics are saved under `outputs/2107_dc_ideal`.
See [first-run coverage, metrics and topology evidence](docs/real_dc_findings.md)
for the completed uncalibrated results.
The next diagnostic stage identifies common-plant candidate calibration periods:
see [screening rules, sensitivity and findings](docs/dc_candidate_screening.md).
These masks are provisional; K_DC remains unset and no calibration is performed.
The [DC bias decomposition](docs/dc_bias_decomposition.md) adds diagnostic solar
geometry, matched AM/PM comparisons and fixed Faiman perturbations. It preserves
the production model and does not establish a calibrated loss factor.
The [focused POA timing audit](docs/poa_timing_audit.md) compares only the seven
specified timing offsets and documents unresolved sensor/logger provenance.
The [no-loss AC reference](docs/ac_no_loss_reference.md) passes ideal DC directly
to the unchanged Sandia inverter model, without assigning K_DC or replacing
the normal AC output interface.
The [measured DC-to-AC diagnostic](docs/measured_conversion.md) removes upstream
PV-model bias by evaluating Sandia at measured DC voltage and V-times-I power.
It preserves the production inverter parameters and leaves K_DC unset.
The [controlled 30 kW ceiling extension](docs/inverter_ceiling.md) preserves
27.6 kW as the official Sandia reference rating and adds a separate 30 kW
maximum-output variant with unchanged efficiency coefficients.
The final [frozen research baseline](docs/research_baseline.md) uses pre-loss DC
directly and the 30 kW extension. Its runner saves complete inverter/plant outputs
and validates meter start-of-interval averages without fitting any parameters.

- **Topology is inferred:** 20 modules/string Ãƒâ€” 6 strings/inverter Ãƒâ€” 24 inverters
  = 120 modules/inverter and 2880 total. At nominal 310 W/module this is 892.8 kW DC,
  consistent with the approximate 893 kW description. Confirm against wiring records.
- Plant is fixed tilt, 25Ã‚Â° tilt and 180Ã‚Â° azimuth; these are metadata because POA is
  measured directly. The inverter database identity now drives Block 5 only.
- Faiman `u0=25`, `u1=6.84` are **provisional pvlib defaults**, not calibrated plant
  coefficients. Confirm mounting, wind sensor height/exposure and suitable values.
  Faiman module temperature is used as cell temperature without an extra offset.
- Measured POA is also used directly as CEC effective irradiance; spectral, incidence
  angle, shading and soiling corrections are absent at this stage. Check POA sensor
  calibration, alignment and representativeness.
- Modules are assumed identical with uniform irradiance and temperature. Expected
  voltage/current describe ideal array MPP, which can differ from measured operation.
- Site latitude, longitude, timezone and wind sensor height remain `None`; no site
  coordinates or timezone have been invented. Confirm timestamps and weather units.
- Verify the precise Hyundai database identity against the installed module nameplate
  before advancing. CEC silicon bandgap/reference conventions are pvlib's defaults.

References: [pvlib Faiman](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html)
and [explicit single-diode workflow](https://pvlib-python.readthedocs.io/en/stable/user_guide/modeling_topics/singlediode.html).

## Stage 2 chronological protocol

One shared target-time split uses the QC-valid date-range overlap, preserving
internal gaps: POA 2017-11-01 07:10 to 2023-11-01 23:55; ambient/wind
2017-12-01 00:00 to 2023-10-31 23:45; complete-fleet AC 2017-11-01 00:00
to 2023-11-01 23:55. Timestamps remain naive Pacific wall-clock; no UTC
reinterpretation. DST-ambiguous/nonexistent geometry remains excluded.

The common span is [2017-12-01 00:00, 2023-10-31 23:50), with the final
recorded target at 23:45. The exclusive end is one five-minute grid step past
that record. Boundaries are 70%/85% of elapsed span, rounded **up** to five
minutes: validation starts 2022-01-21 16:45; testing starts 2022-12-11 20:20.
Targets at a boundary enter the later split; historical inputs may precede it.
`SPLIT` in `scripts/forecast_stage2_weather.py` is the shared definition.
Previously explored 2022-2023 records overlap the eventual test period; it is
held out from new fitting/selection, not wholly unseen.

| Split | Eligible origins | POA labels | Ambient labels | Wind labels | Fleet-power labels |
|---|---:|---:|---:|---:|---:|
| Training | 377763 | 353530 | 125979 | 125982 | 377759 |
| Validation | 70559 | 61233 | 23467 | 23481 | 70557 |
| Testing | 77561 | 67187 | 25886 | 25890 | 77561 |

Eligibility retains the two-hour history, valid current POA and latest ambient/
wind inputs, with older missing lags imputed using fitting data only. Weather
lookup remains latest label 15-30 minutes before each feature timestamp, an
unverified availability assumption. Labels use exact timestamps without filling;
POA selection scores daylight only, environment scores all valid labels.

Proposed correction-training scheme: initial 40% of the training duration for
weather fitting/selection, then three expanding forecast blocks covering its
remaining 20% each. Block edges are 2019-07-29 01:55, 2020-05-26 14:55,
2021-03-25 03:50 and 2022-01-21 16:45. Weather configurations are selected once
within the initial period: first 80% fits candidates, last 20% selects. They
remain frozen for all three expanding refits and outer validation/test.
Each refit uses only labels preceding its forecast block. Outer validation
selects correction settings; it does not retune weather configurations.
All pipeline choices must freeze before testing. The initial period supplies
no correction-training forecasts.

Search: 11 configurations per variable (Ridge, bounded forest, histogram
boosting, XGBoost), seed 2107, four CPU threads. The unlimited-depth forest
candidate is removed; depth 12/leaf 5 and depth 20/leaf 5 remain. Other
candidates are unchanged. Planned fits: **33 initial selection + 9 training
refits + 3 validation refits = 45**; test inference uses saved models with zero
fits. Depth-20 200-tree forests can still be slow.
The initial step saves selection progress; each step saves winners, fitted
models/preprocessing and forecasts; reuse requires matching source hashes, split, code and settings.
Physics consumes those step forecasts using unchanged frozen calculations.

### Superseded exploratory Stage 2 results

The old January-June/July-September and January-September/October-December 2018
selection, followed by annual expanding 2019-2021 forecasts, is superseded.
These winners are exploratory, **not selections under the shared protocol**.
Seed 2107, four threads; software and retained configurations remain documented by the
existing lock, frozen Stage 1 configuration and unchanged model definitions.

| Variable | Exploratory winner | Pooled MAE / RMSE | Persistence MAE / RMSE |
|---|---|---:|---:|
| poa (W/m2) | forest {'max_depth': None, 'min_samples_leaf': 10} | 34.860089 / 76.871805 | 53.469264 / 92.344000 |
| ambient (C) | forest {'max_depth': 20, 'min_samples_leaf': 5} | 0.474238 / 0.837520 | 0.661258 / 1.248898 |
| wind (m/s) | ridge {'alpha': 0.1} | 0.221004 / 0.383598 | 0.193229 / 0.396264 |

Prior annual forecast/valid-power-label counts: 2019 104797/104795; 2020
84527/84527; 2021 87150/87150. All physics calculations were valid. Daylight
physics MAE / mean signed correction (actual minus physics), kW: 2019 68.070 / -50.726; 2020 62.540 / -43.964; 2021 94.793 / -76.810.
Wind ML did not improve persistence MAE. Large stale row tables and their
annual-run settings/scores were removed after retaining this summary.

The classical correction comparison uses seven agreed forecast/origin features,
training-only preprocessing, and four fixed engineering configurations. It selects
by daylight validation MAE, with physics and raw persistence references. Paired
calendar-week uncertainty describes validation comparisons after selection, not
confirmatory test evidence. Current measured power/POA are assumed available
at the origin; no testing inference or weather/physics regeneration is performed.

The correction HPO extends the same script: `--prepare` freezes/checks 43
candidates and reports reuse without fitting; `--hpo` executes or resumes that
plan sequentially. Results are separate in `outputs/2107_stage2_correction/hpo/`;
the original four-model comparison is preserved. Ridge uses seven fixed alphas;
each tree family includes its baseline plus 11 unique seeded random candidates.
Only family-winning bundles/predictions are retained; all trial scores, failures
and runtimes remain saved. Failed trials are not automatically retried or replaced.
Exact MAE ties use frozen candidate order. No validation refit or testing occurs.
Ranges and budget were specified after inspecting baseline results; validation
scores are model-selection results, not unbiased final performance estimates.
[Bergstra and Bengio (2012)](https://www.jmlr.org/papers/v13/bergstra12a.html)
support random search as a strategy, not these project-specific ranges/budgets.

Matched neural corrections use `scripts/train_stage2_neural.py`: the same seven
inputs and eligible examples, saved training-only imputation, binary missing
indicators, continuous-input standardization, and uncentered target-SD scaling.
Both use a four-unit tanh projection and linear readout; the middle is either
four-unit tanh or four-qubit, two-layer StronglyEntanglingLayers (`[1,2]`, CNOT,
analytic `lightning.qubit`/Torch adjoint). These are fixed engineering choices for
a controlled CPU simulation, not established optima or hardware execution.
Adam 0.001, normalized Huber delta 1, batch 128, at most 100 epochs, patience 10
with 0.001 kW daylight validation improvement; paired seeds 2107–2111, four threads.
Forest `forest_05` (32.87024469806436 kW daylight validation MAE) stays an external reference.

Optional dependencies are not part of the frozen baseline lock. Install CPU Torch,
then the extra with the lock as constraints; stop if resolution conflicts:
```powershell
.\.venv\Scripts\python -m pip install -c requirements-lock.txt "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -c requirements-lock.txt -e ".[neural-quantum]"
.\.venv\Scripts\python -B scripts/train_stage2_neural.py --prepare
.\.venv\Scripts\python -B scripts/train_stage2_neural.py --pilot
```
Preparation writes nothing. The pilot updates disposable models on the same 1,024
training examples and saves timing only in `outputs/2107_stage2_neural/pilot.json`.
Review its full-budget runtime estimates before `--train --model classical` (or
`quantum`) `--seed 2107 --accept-pilot`; add `--resume` after interruption.
Repeat seeds explicitly; `--summary` reports per-seed metrics and mean/sample SD,
without selecting a seed or averaging forecasts. Keep the entire ignored output
directory durably: preprocessing/settings, per-seed best/last complete checkpoints,
history, metrics and keyed validation predictions. Validation is used for stopping;
parts of the eventual test period were previously explored; testing remains gated.

Held-out ML/DL evaluation is prepared in `scripts/evaluate_stage2.py`. Freeze
`forest_05` and all five completed classical seeds before generating test inputs:
```powershell
.\.venv\Scripts\python -B scripts/evaluate_stage2.py --prepare
.\.venv\Scripts\python -B scripts/forecast_stage2_weather.py --step testing --allow-test
.\.venv\Scripts\python -B scripts/calculate_stage2_physics.py --step testing --allow-test
.\.venv\Scripts\python -B scripts/evaluate_stage2.py --evaluate --allow-test
```
These are manual commands: the weather step reuses all three saved validation
models (zero fits), then frozen physics and the selected corrections run once.
The evaluator requires unchanged artifact hashes, complete classical checkpoints,
exact target matching and common eligibility for physics, persistence, RF and
every neural seed. It saves keyed predictions and daylight/all-day MAE, RMSE and
prediction-minus-actual bias in `outputs/2107_stage2_test/`. It does not choose a
seed or ensemble. Parts of this test period appeared in earlier exploratory work.
The saved ML/DL test evaluation is complete: 77,561 eligible rows, including
48,190 daylight rows. Preserve these results; do not rerun the commands above.

CPU quantum testing is separate in `scripts/evaluate_stage2_quantum.py`. Return
the entire `2107_stage2_quantum_cpu/` folder, including root settings/preprocessing
and each seed's settings, best/last checkpoints and metrics. For a returned ZIP
containing that top-level folder, from the repository directory:
```powershell
if (Test-Path .\outputs\2107_stage2_quantum_cpu) { throw 'Existing runs: do not overwrite' }
Expand-Archive -LiteralPath .\quantum_cpu_results.zip -DestinationPath .\outputs
.\.venv\Scripts\python -B scripts/evaluate_stage2_quantum.py --prepare
# After reviewing readiness, explicitly authorize test scoring:
.\.venv\Scripts\python -B scripts/evaluate_stage2_quantum.py --evaluate --allow-test
```
Preparation checks completed validation-selected seeds 2107–2109, source/settings
identity and saved fitted preprocessing without loading or scoring test rows.
Testing uses CPU float64 native `default.qubit`/backprop, batch 256, on exactly the
saved ML/DL keys, labels and physics forecasts. It writes separate predictions,
provenance and daylight/all-day MAE, RMSE, prediction-minus-actual bias and R² to
`outputs/2107_stage2_quantum_test/`, including per-seed scores and their mean/sample
standard deviation. No fitting, seed selection, ensemble or overwrite is allowed.
Actual quantum testing remains pending returned checkpoints. The test period's
earlier exploratory exposure still applies.

### CPU quantum handover (start only after ML/DL test review)

`scripts/train_stage2_quantum_cpu.py` keeps the 89-parameter four-qubit/two-layer
circuit, float64/complex128, initialization, signed correction, training-fitted
preprocessing, uncentered target-SD scaling, Huber 1, Adam 0.001, 100 epochs,
patience 10 and 0.001 kW validation improvement. It uses native-batched
`default.qubit`/Torch backprop, **batch 256**, seeds **2107–2109**, four CPU threads.
Batch 256 is an approved setting change for this separate run, not an accuracy
equivalence claim. Retain the original CPU reference and completed classical runs.

GitHub alone omits the data and fitted preprocessing. Send the prepared ignored
`outputs/2107_stage2_quantum_cpu_friend.zip`: it contains the existing verified
`project.zip` unchanged, current CPU/helper code, focused tests and these steps.
The inner package includes only training/validation measurements, saved physics,
portable HPO manifest, hashed Ridge preprocessing and authoritative classical
reference settings. It omits test data, classical checkpoints, credentials and
environments. Kaggle and CUDA are not needed.

Friend setup (Windows PowerShell, Python 3.11; use a fresh folder):
```powershell
Expand-Archive .\2107_stage2_quantum_cpu_friend.zip -DestinationPath .\pv2107-friend
Set-Location .\pv2107-friend
if ((Get-FileHash .\project.zip -Algorithm SHA256).Hash -ne '91e3e2e73ebaf3b46149a4e1ed98f0888f85b755555ba96700c14e4e9a92c437') { throw 'Wrong input package' }
Expand-Archive .\project.zip -DestinationPath .\project
Copy-Item .\code\scripts\*.py .\project\scripts\
Copy-Item .\code\tests .\project\ -Recurse
Set-Location .\project
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
.\.venv\Scripts\python -m pip install -r requirements-lock.txt -c neural-constraints.txt -e '.[neural-quantum]'
.\.venv\Scripts\python -B -m pytest tests/test_stage2_quantum_cpu.py tests/test_stage2_neural.py -q
.\.venv\Scripts\python -B scripts/train_stage2_quantum_cpu.py --prepare
```
Use exact recorded versions; if a wheel/pin is unavailable, stop and report the
conflict rather than upgrading. Installation availability on the friend's machine
has not been verified. Preparation checks counts (221057/70555; daylight 43949),
data/preprocessing/source fingerprints and tiny output/loss/gradient equivalence;
it does not fit on the real data. No packages are installed automatically.

After our ML/DL test review and explicit start approval, run each seed once:
```powershell
foreach ($seed in 2107,2108,2109) {
    & .\.venv\Scripts\python -B scripts/train_stage2_quantum_cpu.py --train --seed $seed --accept-runtime
    if ($LASTEXITCODE -ne 0) { throw "Stopped at seed $seed" }
}
# Resume only the interrupted seed; change 2107 to its actual seed.
.\.venv\Scripts\python -B scripts/train_stage2_quantum_cpu.py --train --seed 2107 --resume --accept-runtime
.\.venv\Scripts\python -B scripts/train_stage2_quantum_cpu.py --summary
Compress-Archive .\outputs\2107_stage2_quantum_cpu -DestinationPath .\quantum_cpu_results.zip
```
Return the full results archive separately: best/last checkpoints (including
optimizer/RNG), preprocessing, keyed validation predictions, history and settings.
Also send per-seed `metrics.json`, `history.csv` and settings as a small summary
bundle suitable for deliberate Git inclusion. Report every seed and mean/sample
SD; no best-seed selection or ensemble. Interrupted work resumes from the last
atomic epoch checkpoint; updates after that checkpoint are repeated. No full-data
training was run during preparation. The saved CPU batch-256 benchmark estimates
about 16.7 hours for three seeds × 100 epochs including validation (observed timing
spread 10.0–18.2 hours); hardware/setup/I/O vary and early stopping is not assumed.

Kaggle GPU batches use `scripts/train_stage2_quantum_gpu.py`: the same 89-parameter
quantum model and unchanged trainer, with native `lightning.gpu`, complex128
statevectors, analytic expectations and adjoint gradients. Small Torch layers stay
on CPU in float64; shared initialization remains identical to completed classical
runs. The Linux-only `quantum-gpu` extra adds the native simulator without changing
the frozen lock. No CPU fallback is permitted. Four-qubit GPU speedup is unproven.

```powershell
.\.venv\Scripts\python -B scripts/train_stage2_quantum_gpu.py --prepare
.\.venv\Scripts\python -B scripts/kaggle_stage2_pilot.py --package
```

The package in `outputs/2107_kaggle_quantum_gpu/` contains separate private GPU
Script jobs in `pilot/` and `training/`, plus minimal inputs in `dataset/`.
Existing CPU preparation/results remain untouched. Packaging verifies original
classical source hashes, preprocessing, training/validation fingerprints and counts;
measurement copies exclude the test period. No credentials or environments are bundled.
Each generated launcher pins the archive's exact relative file paths and SHA-256
hashes, verifies the mounted inputs before setup, and records their content identity
in batch status. Dataset metadata names a slug, not a version: stale or mixed
attachments fail before installation. Original classical manifests remain unchanged.
Remote setup pins completed classical Torch/PennyLane versions and the frozen lock;
dependency conflicts fail rather than upgrade baseline packages.

Proposed dataset: `anokgaming/pv2107-quantum-inputs`; pilot:
`anokgaming/pv2107-quantum-correction`; separate training notebook:
`anokgaming/pv2107-quantum-training`. Check each resource's existence, privacy and
purpose before manually creating/pushing; do not overwrite unrelated resources.
Dataset creation defaults private (omit `--public`); both notebooks explicitly
set `is_private: true`, GPU enabled. No upload or launch happens during preparation.

The GPU pilot checks native GPU execution, float64 outputs and nonzero projection/
circuit gradients against the CPU reference (output atol/rtol 1e-10/1e-8; gradients
1e-9/1e-6), plus finite differences. It times the same 1,024 training examples,
warm-up, eight update batches and separate inference. Its five-seed, 100-epoch
extrapolation assumes no early-stopping savings and excludes setup/checkpoint I/O.
The saved version 5 pilot passed correctness, but its unsynchronized extrapolation
was 780.75 GPU hours versus 329.77 CPU hours. Optional `--package --diagnostic
--package-dir outputs/2107_kaggle_quantum_gpu_diagnostic` prepares a separate
`gpu_batch_diagnostic.json`: unchanged per-row calls versus one broadcast QNode
call per batch, plus the CPU reference. Lightning 0.45.0
[expands broadcasts into separate circuits](https://github.com/PennyLaneAI/pennylane-lightning/blob/v0.45.0/pennylane_lightning/lightning_gpu/lightning_gpu.py);
this is not native vectorized GPU simulation. Equivalence precedes timing; five
rotating, warmed repetitions reset parameters/Adam and use native
`DevPool.syncDevice()` before/after timed regions. Median, range and speed ratios
are reported; wide timing ranges remain inconclusive. Production training stays
per-row. The diagnostic cannot authorize training, and does not overwrite pilot
evidence. GPU source changes require a new dataset version and compatible normal
pilot before any later training acceptance; old pilot provenance remains intact.
For native batching, `--broadcast-benchmark --device cpu` on the same GPU script
compares the unchanged row/adjoint reference at batch 128 with `default.qubit` /
Torch backprop at batches 64, 128 and 256. The original four-qubit/two-layer circuit,
89 parameters, float64/complex128 precision, Adam and normalized Huber loss remain.
Five warmed repetitions follow prediction/loss/all-gradient equivalence checks;
the result is separate: `outputs/2107_stage2_quantum_broadcast/benchmark_cpu.json`.
Use `--device cuda` only with a CUDA-enabled build of the same Torch release and a
GPU: the model and native batched simulator then run on CUDA and synchronize there.
No fallback or installation occurs. Estimates cover three seeds × 100 epochs
(five also reported), using training-input inference to estimate validation cost.
The 30-hour GPU planning budget reserves five hours for setup/final evaluation;
this is an allowance, not a runtime guarantee. Alternate batch sizes are diagnostic
only; neither benchmark promotes a training path or authorizes full training.

For the short Kaggle notebook, reuse the already verified input package (no rebuild):
```powershell
.\.venv\Scripts\python -B scripts/kaggle_stage2_pilot.py --benchmark-notebook --package-dir outputs/2107_kaggle_quantum_gpu_reference_fix
```
Import `outputs/2107_stage2_quantum_broadcast/kaggle_benchmark.ipynb` into a private
Kaggle notebook, attach the existing private `anokgaming/pv2107-quantum-inputs`
version matching its hashes, select a single-GPU accelerator if available, enable
Internet, and Save Version / Run All. Only device 0 is used; hiding a second GPU
does not release that allocation.
It checks the exact archive and all 44 input hashes before setup, then times only
native batches 128/256 on 1,024 training rows, five warmed repetitions each.
The row reference remains for correctness only. CUDA state verification and
synchronized timing are mandatory. Download `native_benchmark/` results/status/logs;
stop any interactive session afterward. Target under one GPU-hour including setup,
not a guarantee; no retry or transition to training. Wrong inputs stop before setup.
The isolated notebook pins CUDA Torch 2.11.0+cu128 because the official CUDA 12.8
index lacks the saved CPU release 2.14.1. This explicit benchmark-only exception
leaves classical/training checks intact; saved CPU comparisons include hardware
and Torch-version differences. Three-seed estimates assume 100 epochs and include
validation; batch 256 remains unapproved for training. Other pins remain unchanged.

After downloading and reviewing `2107_stage2_quantum_gpu/gpu_pilot.json`, accept it
locally, then submit training separately:
```powershell
.\.venv\Scripts\python -B scripts/kaggle_stage2_pilot.py --accept-gpu-pilot outputs/2107_kaggle_quantum_gpu/pilot_output/2107_stage2_quantum_gpu/gpu_pilot.json
.\.venv-kaggle\Scripts\kaggle.exe kernels push -p outputs/2107_kaggle_quantum_gpu/training
```
Acceptance records the pilot hash and resolved native dependency versions in the
training launcher. Training requires that exact attached successful pilot and
matching code/data/preprocessing/settings/software/GPU architecture; it runs seeds
2107?2111 with the existing full-data budget, stopping policy and shuffled orders.
Neither job transitions to another job, retries, resubmits, keeps alive or sets a
custom wall-clock limit. Normal completion or failure saves status/logs and exits.

Download and retain `quantum_gpu_batch.log`, `quantum_gpu_batch_status.json` and the
entire `2107_stage2_quantum_gpu/` directory: pilot, settings/preprocessing, per-seed
best/last complete checkpoints, histories, metrics and keyed validation predictions.
For manual recovery, `--resume-output <downloaded/2107_stage2_quantum_gpu>` prepares
`dataset/resume.zip`; publish a new version of the same verified private dataset
and manually resubmit training. Compatible completed seeds are checked and skipped;
others resume from atomic epoch checkpoints including optimizer and RNG state.
Work within an interrupted epoch is lost. Kaggle-imposed termination may prevent
status writing or publication of working outputs: checkpoint recovery is possible
only when those files are actually retrievable. No custom duration limit guarantees
completion. Broadcast GPU correctness and synchronized speed remain unverified
until the manual diagnostic; testing remains untouched.
