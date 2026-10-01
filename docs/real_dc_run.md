# First real-data ideal DC run

Run from the repository root:

```powershell
.\.venv\Scripts\python -m pip install -e ".[test,real-data]"
.\.venv\Scripts\python scripts/run_real_dc.py
```

Tested integration dependencies: pyarrow 25.0.1 and matplotlib 3.11.2.
The CLI accepts `--input`, `--metadata` and `--output`; defaults use the prepared
5-minute Parquet, original system metadata and `outputs/2107_dc_ideal`.

## Mapping and timestamp policy

The script inspects the actual Parquet schema, then discovers exactly one
DC-voltage, DC-current and provenance AC-power channel for each ID `inv_01`–`inv_24`.
Measured AC is not read or used in modelling. Weather maps `poa_w_m2` to `poa`,
`ambient_temperature_c` to `temp_air`, and `wind_speed_m_s` to `wind_speed`.

The Parquet schema has numeric types but no unit metadata. Original metric
metadata confirms prepared DC channels are V and A; cleaning preserved these
readings without scaling. Finite voltage × finite current therefore gives W.
Inverter 05 retains its entirely missing voltage and derived power.

Timestamp strings are preserved in `measured_on` and parsed exactly into naive
`timestamp` values. Explicit `pacific_wall_clock` mode accommodates this prepared
dataset in the existing model and validation interfaces. Default timezone-aware
behaviour remains unchanged. No localization, UTC conversion, time shift,
5-minute/15-minute resampling or fall-DST disambiguation is performed.

## Calculation and exclusions

Only available, finite required weather inputs supported by the existing model
(nonnegative POA/wind) enter measured POA → Faiman → CEC module → inferred 20×6
array scaling. All source rows remain, with `dc_model_computed` and
`dc_model_status`; other predictions remain NaN. Negative/nonfinite inputs,
if encountered, are reported without altering measurements. Existing availability,
electrical/irradiance QC and fall-DST flags are retained, not used as exclusion
rules. No new cleaning or synthetic input values are introduced.

Faiman coefficients, direct POA-as-effective-irradiance and CEC module parameters
are unchanged. K_DC stays unset: no unity derate is supplied. There is no inverter
AC calculation or plant/meter comparison. CEC inverter DC limits are retrieved
only as review provenance and do not clip data or predictions.

## Outputs

- `nominal_dc_ideal.parquet`: all 622,294 timestamps, weather/flags, temperature,
  module outputs and ideal one-array DC predictions.
- `inverter_dc_ideal.parquet`: all rows for every inverter, including invalid-input
  rows; timestamp, source string, ID, weather, flags, module/ideal array outputs,
  measured V/I and finite-only derived P. Stored by inverter in bounded row groups.
- `channel_mapping.csv`: exact source channels, IDs and DC-unit provenance.
- `dc_metrics.csv`: provisional MAE/RMSE/MBE/R² and total/valid/excluded counts per
  signal/device, using existing Block-7 utilities. Local copies map the ideal
  columns to that utility's names; saved predictions retain `_ideal` labels.
  No normalized metrics or denominators are used.
- `dc_ranges_and_topology.csv`: expected/measured finite min/max and valid-pair
  counts. Measured ranges include all retained readings; metrics use exact finite
  pairs. Counts of negative readings and values above labelled CEC references are
  evidence only, not fault decisions or cleaning criteria. `Pdco` is a reference
  DC power at `Vdco`, not a hard DC-power limit; its exceedance is not a violation.
- `run_provenance.json`: schema, configuration, input hashes, counts and policies.
- Twelve PNGs: V/I/P time series and expected-versus-measured scatters for IDs
  01, 04, 05 and 24 on 2018-06-15, 2021-01-15 and 2023-06-15. These illustrative
  review windows span seasons/years; they are not calibration or healthy splits.

The topology evidence slice is explicitly **POA ≥600 W/m²** with finite expected
and measured quantities. It reports measured V/module Vmp and measured I/module
Imp p10/median/p90 against the inferred 20 and 6 references. Power/module Pmp is
also reported. This slice is not a healthy selection and does not estimate
replacement parameters. Zero/outage and unusual readings remain included.

Interpret all metrics provisionally: ideal MPP is not necessarily the measured
operating point; no static losses, availability exclusions or calibration have
been applied. Full-record voltage metrics also compare ideal nighttime zero with
measured voltage, so daytime topology evidence should be read separately.

The script verifies input hashes are unchanged and writes exactly 24 × source
row-count records. It processes each inverter separately to avoid constructing
the full 14.9-million-record table in RAM. All prior unresolved assumptions and
anomalies remain available for review; topology is never changed automatically.
