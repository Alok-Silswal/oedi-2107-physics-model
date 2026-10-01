# OEDI System 2107 physics model (through Block 6; Block 7 validation tools)

This package implements measured POA → Faiman temperature → explicit CEC
single-diode module calculation. The requested ideal 20-series × 6-parallel
scaling gives MPP for one inverter's DC array. Block 4 adds an optional global
static DC power derate with its numerical factor currently unset. Block 5 adds
one inverter's DC-to-AC conversion using official CEC/Sandia parameters. Block 6
retains 24 named inverter records and sums AC at each original timestamp.
Block 7 provides reusable exact-key comparison metrics and arithmetic residuals
for future supplied data. No real-data validation, DC-loss fit, inverter fitting,
time resampling, fault detection or degradation analysis has been performed.

## Setup and run (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python examples/run_foundation.py
.\.venv\Scripts\python examples/run_inverter.py
.\.venv\Scripts\python -m pytest
```

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
The CEC row reports STC = 309.6 W (36 V × 8.6 A) and `N_s=60`;
review these database fields against the nameplate/datasheet. They are retained
unchanged. CEC STC × 2880 is 891.648 kW, distinct from nominal 892.8 kW.

`requirements-lock.txt` records the tested dependency versions. To reproduce
these versions, run `python -m pip install -r requirements-lock.txt` before
installing the editable package.

## Layout and interface

- `src/oedi2107/config.py`: system metadata, inferred topology, thermal assumptions.
- `inputs.py`: measured-input validation; no irradiance transposition.
- `thermal.py`: `estimate_cell_temperature(poa, temp_air, wind_speed, config)`.
- `module.py`: database loading and explicit `calcparams_cec` → `singlediode`.
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
and columns `poa` [W/m²], `temp_air` [°C], `wind_speed` [m/s]. Inputs must be aligned
and finite. Negative POA/wind and missing readings raise errors; decide sensor
offset correction and data quality treatment upstream. POA is retained unchanged.
Extra measurement channels (DC voltage/current, inverter AC, plant meter) are
not used. The model does not infer timestamps or the site's timezone.

Output retains `temp_cell` [°C], adjusted diode parameters (photocurrent and
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
for Blocks 1–3. A factor in the configuration does not derate that function's
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
inferred 20×6 topology, ABB US/480 V variant, CEC `Idcmax` versus inferred array
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

- **Topology is inferred:** 20 modules/string × 6 strings/inverter × 24 inverters
  = 120 modules/inverter and 2880 total. At nominal 310 W/module this is 892.8 kW DC,
  consistent with the approximate 893 kW description. Confirm against wiring records.
- Plant is fixed tilt, 25° tilt and 180° azimuth; these are metadata because POA is
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
