# Controlled 30 kW ceiling extension

`calculate_inverter_ac_30kw_ceiling()` in `src/oedi2107/inverter.py` preserves
both `expected_ac_power_cec` and `expected_ac_power_30kw`, in watts.
The ordinary `calculate_inverter_ac()` and production configuration are unchanged.
No K_DC or efficiency coefficients are fitted.

## Equation and provenance

The exact CEC row remains `ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_`.
Its Paco = 27,600 W remains the reference power in the equation, not the new cap.
With official parameters and measured or supplied DC voltage V and power P:

```text
A = Pdco * (1 + C1 * (V - Vdco))
B = Pso  * (1 + C2 * (V - Vdco))
C = C0   * (1 + C3 * (V - Vdco))
AC_preclip = (Paco / (A - B) - C * (A - B)) * (P - B) + C * (P - B)^2
AC_30kw = -Pnt                         if P < Pso
          min(AC_preclip, 30,000 W)    otherwise
```

Startup uses official Pso, as the installed pvlib implementation does, rather
than voltage-adjusted B. No zero clamp or additional minimum is introduced.
The official baseline still calls `pvlib.inverter.sandia()` independently.
The extension implements its pre-clipping equation explicitly, without using
private pvlib helpers or replacing Paco inside the efficiency equation.

This is a project-specific maximum-output extension, supported by independent
measured DC-to-AC evidence and the [ABB US product manual](https://library.e.abb.com/public/9ddffa465aa146bfa43bd0854abbcc81/TRIO-20.0-27.6-TL-US_Product_Manual.pdf),
whose technical table distinguishes 27,600 W nominal from 30,000 W maximum at
480 V. It is not a new CEC database row or a coefficient refit. Installed suffix,
settings and manufacturer operating-condition restrictions remain provenance
review items. A fixed cap does not implement thermal, grid or MPPT constraints;
the unchanged polynomial is extrapolated above its CEC reference rating.

## Measured-DC comparison

Run `.venv/Scripts/python scripts/run_inverter_ceiling.py` from the project root.
It reuses only measured electrical quantities and the explicit Task 11 selection
mask, excludes inverter 05, and preserves inverter identity/timestamps. No
upstream model DC, meter input, resampling or cleaning changes are used.
Results are in `outputs/2107_inverter_ceiling/`, including per-inverter paired
predictions, errors and full official parameter provenance.

Errors below are pooled across identical valid pairs; residual = measured minus
predicted. Below-clipping guards remain DC < 90% official Pdco and AC < 90% Paco.
The high-power band is 29,900 <= measured AC < 30,100 W.

| Population | Pairs | Model | MAE W | RMSE W | MBE W |
|---|---:|---|---:|---:|---:|
| All selected | 6,274,691 | CEC | 356.93 | 797.89 | +262.13 |
| All selected | 6,274,691 | 30 kW | 138.76 | 402.08 | +33.44 |
| Below clipping guards | 4,681,970 | Either | 115.72 | 332.89 | +5.08 |
| Measured AC > 27.6 kW | 953,788 | CEC | 1,630.22 | 1,821.35 | +1,630.22 |
| Measured AC > 27.6 kW | 953,788 | 30 kW | 190.68 | 383.81 | +130.00 |
| Around 30 kW AC | 239,034 | CEC | 2,357.54 | 2,367.50 | +2,357.54 |
| Around 30 kW AC | 239,034 | 30 kW | 166.66 | 340.13 | +142.37 |

Recommendation **B**: adopt the documented 30 kW extension while retaining
official Sandia coefficients and the official baseline for provenance.
The extension substantially reduces high-power residuals and leaves guarded
below-clipping behaviour unchanged. Residual measurement/operating variation
remains; this is not a calibrated whole-plant model. Production routing has not
been silently switched by this diagnostic task.

Only `tests/test_inverter_ceiling.py` was run: one test passed, covering several
voltages, below-clipping equivalence, the 30 kW cap, startup/night tare, unchanged
inputs and official parameter object, and unset K_DC.
