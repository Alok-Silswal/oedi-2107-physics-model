# Measured DC to AC diagnostic

Run `.venv/Scripts/python scripts/run_measured_conversion.py` from the project
root. Results, per-inverter tables, loading bins, plot and source hashes are in
`outputs/2107_measured_conversion/`.

## Inputs and selection

- Only `2107_electrical_clean.parquet`, the existing electrical QC ledger and
  electrical-unit metadata are used. No expected PV DC or plant meter inputs.
- Full cleaned electrical span: 2017-11-01 to 2023-11-07, 632,952 timestamps.
  Inverter 05 is excluded because measured DC voltage is unavailable.
- Metadata confirm volts, amperes and AC kilowatts with unit scaling/zero offset:
  measured DC = V times I in W; measured AC = source kW times 1000.
- Keep timestamps and all source values. Screen analysis only: finite positive
  V/I/AC, local clock 06:00 <= hour < 20:00 and no existing electrical QC event.
  The clock window is a broad daytime proxy, not a solar-position calculation.
  All six QC-event timestamps are excluded across the fleet conservatively.
- DC >= 281.99173828 W (1% official Pdco) and AC >= 276 W (1% official Paco)
  avoid very-low-power ratio instability. These fixed screening choices do not
  establish healthy periods or calibrated ratings.
- Of 14,557,896 inverter records, 6,497,075 pass positive/daytime/QC selection;
  222,384 then fail the power floors, leaving 6,274,691 comparison pairs.
- Raw AC/DC ratios are retained, including values above one. Physical efficiency
  is reported only for 0 < ratio <= 1. Ratios above one occur in 1.159% of selected
  records; they indicate measurement/interval or operating inconsistency, not
  efficiencies to fit. Lower efficiency tails can also include abnormal operation.

## Fixed comparison

`calculate_inverter_ac` uses measured Vdc and measured Pdc, with the unchanged
`ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_` parameters. Sandia alone supplies its
27,600 W cap. Residual = measured AC minus Sandia AC. No K_DC is assigned.
DC-loading bins divide measured Pdc by official Pdco = 28,199.173828 W.
CSV summaries preserve inverter identities, counts and raw-ratio inconsistencies.

## Findings

- Per-inverter physical-efficiency medians: 96.82–98.40%. The median of inverter
  bin medians rises from 77.7% at 1–5% DC loading to 96.5% at 10–25%, and about
  97.8–98.2% at 25–100%. The full retained physical range is about 1.01–100%;
  its low tail is not evidence of normal converter efficiency.
- Below conservative clipping guards (DC < 90% Pdco and AC < 90% Paco),
  4,681,970 pairs have per-inverter MAE 95–227 W, RMSE 298–353 W and MBE
  -188 to +92 W. Median inverter MAE/RMSE/MBE: 108/335/+12 W.
- Across all selected loading levels, per-inverter MAE is 285–423 W, RMSE
  706–859 W and MBE +99 to +367 W. Median inverter values: 360/799/+275 W.
- At 27.5–27.7 kW measured AC, inverter median DC powers are 27.96–28.39 kW.
  At 29.9–30.1 kW measured AC, median DC powers are 30.35–30.77 kW;
  physical-efficiency medians remain 97.31–98.70%. There are 239,034 records
  in the latter band, including 4,312 raw ratios above one.
- All 23 equipped units repeatedly reach 30.0–30.1 kW. Dominant upper bands
  are 29.9–30.0 kW for 22 units and 29.8–29.9 kW for inverter 21. Consecutive
  dominant-band runs span 43–57 five-minute records. QC-clear p99 values are
  29.904–29.962 kW and maxima 30.082–30.106 kW.
- Median inverter-bin residuals rise from about +7.5 W at 25–50% loading to
  +129 W at 90–100%, then +751 W at 100–105% and +2,275 W at 105–110%.
  Sparse above-110% observations have lower efficiency and cannot alone
  establish the complete saturation curve.

The 27.6 kW ceiling mismatch survives removal of upstream PV-model bias.
The fleet supports an approximately 30 kW telemetry ceiling with plausible
conversion efficiency; this is evidence, not an independently verified nameplate.
Official Sandia efficiency behaviour is broadly reasonable below clipping.
Retain production parameters pending installed-variant/settings and AC-channel
active-power/scaling verification. If those confirm 30 kW active power, review an
explicitly documented inverter variant or ceiling change separately; do not fit
efficiency coefficients or let K_DC absorb the ceiling discrepancy.

Source hashes and official parameters were checked unchanged. Only the two new
tests in `tests/test_measured_conversion.py` were run; both passed.
