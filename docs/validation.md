# Block 7 validation foundation

These tools compare supplied prediction and measurement tables. They do not run
or fit the physics model, choose dates, define healthy operation, or read real
2107 electrical data. Inputs must already have reviewed physical units, device
identity and timestamp semantics. Synthetic tests verify the implementation;
they do not establish real-data model accuracy.

## Metrics and normalization

For N selected finite pairs, let `e` be expected, `m` measured and `r = m - e`:

| Metric | Definition |
| --- | --- |
| MAE | `sum(abs(r)) / N` |
| RMSE | `sqrt(sum(r²) / N)` |
| MBE | `sum(r) / N` |
| Normalized MAE | `MAE / D` |
| Normalized RMSE | `RMSE / D` |
| R² | `1 - sum(r²) / sum((m - mean(m))²)` |

MAE/RMSE/MBE have the signal's units (V, A or W). Positive MBE means measured
values exceed expected values. R² uses **measured** variance and can be negative.
It is NaN for fewer than two selected pairs or constant measurements, including
a perfect constant prediction. All metrics are NaN when no pairs are selected.
No finite placeholder is substituted for an undefined metric.

`Normalization(value=D, basis="caller description")` requires a finite positive
scalar D in the same units as the signal and a nonempty basis description. The
caller might explicitly use a reviewed capacity or a mean over a named period;
the tools choose neither and never calculate the denominator automatically.
Normalized results are **dimensionless fractions**, not percentages. Multiplying
by 100 for presentation is a separate explicit choice. Without a normalization,
normalized metrics remain NaN and the summary records no basis/value.

Reusable functions in `oedi2107.validation` are `mae`, `rmse`, `mbe`,
`normalized_mae`, `normalized_rmse`, and `r_squared`. Low-level metrics reject
nonfinite values and unequal shapes. Series indexes must match exactly; arrays
are assumed already aligned by position. They do not filter or interpolate.

## Interfaces and input schemas

Every input supplies `timestamp` as a timezone-aware datetime column or a
timezone-aware DatetimeIndex. Both sides must use the same explicit timezone.
No string parsing, timezone conversion, rounding, resampling or nearest-time
matching is done. Duplicate keys raise errors.

The real-data integration may explicitly set `timestamp_basis="pacific_wall_clock"`
for the prepared unlocalized Pacific timestamps. Both sides then remain naive;
no timezone or DST offset is assigned. Default `aware` behaviour is unchanged.

| Interface | Expected columns | Measured columns | Join keys |
| --- | --- | --- | --- |
| `validate_dc` | `expected_dc_voltage`, `expected_dc_current`, `expected_dc_power` | `measured_dc_voltage`, `measured_dc_current`, `measured_dc_power` | timestamp + inverter_id |
| `validate_inverter_ac` | `expected_ac_power` | `measured_ac_power` | timestamp + inverter_id |
| `validate_plant_ac` | `expected_plant_ac_power` | `measured_meter_power` | timestamp |

Voltage is V, current A and power W. Inverter interfaces require a nonempty
string `inverter_id` column on both sides; devices are never cross-paired or
silently pooled. DC power must be supplied explicitly, not inferred from V×I.
The meter interface assumes **power already expressed in W**; it does not accept
responsibility for interpreting raw energy or interval-average semantics.

All three accept these optional keyword arguments:

- `normalizations`: mapping signal name to `Normalization`. Signal names are
  `dc_voltage`, `dc_current`, `dc_power`, `ac_power`, or `plant_ac_power` according
  to the interface. Omit any signal to leave its normalized metrics uncomputed.
- `period_labels`: externally supplied string-label Series. Labels such as
  `calibration`, `validation` and `holdout` are preserved; no dates are chosen.
- `period_masks`: alternative mapping of label to boolean Series. Overlapping
  masks are rejected. Unassigned rows carry `unlabelled`. Supply labels or masks,
  not both.
- `periods`: optional explicit label sequence selecting which rows contribute to
  metrics. Requires supplied labels/masks. If omitted, no period filter is applied.
- `healthy_mask`: externally supplied boolean Series. False excludes a finite
  pair from metrics; no algorithm defines healthy operation. If omitted, no
  health filter is applied (the table's mask-pass flag is True, not a diagnosis).

Annotations use a DatetimeIndex for timestamp-wide values, explicitly applied
to all inverters at that time, or a `(timestamp, inverter_id)` MultiIndex for
individual devices. They must cover **every outer-union comparison key**; missing
annotations raise rather than assuming a period or healthy state. Broader
annotation Series may include times/devices outside the current comparison.

```python
from oedi2107.validation import Normalization, validate_inverter_ac

# All values below must come from a later reviewed dataset/analysis.
result = validate_inverter_ac(
    expected_inverters,
    measured_inverters,
    normalizations={"ac_power": Normalization(reviewed_denominator_w,
                                              "reviewed AC power reference [W]")},
    period_labels=externally_chosen_labels,
    periods=["validation"],
    healthy_mask=externally_supplied_healthy_mask,
)
comparison_rows = result.table
metric_summary = result.metrics
```

## Outputs and missing-data accounting

`ValidationResult.table` retains the **outer union** of keys at each signal;
nothing is removed. Each row includes timestamp, inverter_id (device interfaces),
quantity, unit, expected, measured, residual, expected/measured record-presence
flags, finite_pair, period_label, period_selected, healthy and valid_pair.
DC is long format: three signal rows per timestamp/device key. Residuals are
computed only for finite exact pairs and are NaN elsewhere. Mask-excluded finite
pairs retain their residual for inspection; they do not enter reported metrics.
Signed nighttime predictions and measurements are left unchanged.

`ValidationResult.metrics` contains one row per quantity and inverter_id (or one
row per plant signal), with the six metrics, denominator/basis and:

- `total_samples`: number of unique outer-union keys for this device/signal.
- `finite_paired_samples`: exact keys with both expected and measured finite.
- `valid_paired_samples`: finite pairs also passing period and healthy filters.
- `excluded_samples`: total minus valid pairs.
- `missing_or_nonfinite_samples`: total minus finite pairs.
- `excluded_by_masks_samples`: finite pairs excluded by either mask (counted once).
- `missing_expected_records`, `missing_measured_records`: unmatched-key counts;
  these are additional diagnostics, not counts to add to excluded samples.
- `metric_status`: `computed` or `no valid pairs`.

Thus `excluded_samples = missing_or_nonfinite_samples + excluded_by_masks_samples`.
NaN and infinite readings are retained and excluded with counts. There is no
zero-fill, interpolation, automatic time split or partial silent drop. Reports
remain separate by inverter; an ID present on only one side has zero valid pairs.
To report different periods separately, call the same interface with each
explicit `periods` selection; labels and all rows remain visible each time.
The comparison table also retains supplied model/measurement provenance in
`.attrs` (not preserved by CSV).

## Pending cleaned-data decisions

No calibration of Faiman coefficients, K_DC, Paco or inverter-specific parameters
is implemented. Faiman u0/u1, POA as effective irradiance, unset numerical K_DC,
inferred 20×6 topology, ABB US/480 V variant, CEC Idcmax discrepancy, measured
clipping discrepancy, actual inverter IDs, timestamp/timezone semantics, meter
power-versus-energy meaning and nighttime sign conventions remain unresolved.

Before real comparisons, review units and measured channel names, map real
inverter IDs, confirm timestamps and 5-minute/15-minute semantics, choose any
alignment/resampling policy externally, supply healthy masks and date splits,
and select normalization denominators. No cleaned-data accuracy results exist
yet. Residuals here are arithmetic only: no thresholds, fault rules, CUSUM,
degradation, anomaly detection or ML. No dashboard or plotting dependency is added.
