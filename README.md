# OEDI System 2107: 15-minute PV power forecasting

This semester project forecasts the combined AC power of **24 inverters at one site**, 15 minutes ahead. Training and local held-out testing are complete for:

- **Direct forecasting:** Ridge, Random Forest (RF), histogram gradient boosting, XGBoost and a classical neural network.
- **Physics plus correction:** frozen Stage 1 power plus the same four ML families and classical neural correction.
- **Optional quantum correction:** a **simulated variational quantum circuit (VQC)** with four qubits, two layers and three seeds, using native-batched CPU execution at batch 256. This is circuit simulation, not quantum hardware or demonstrated quantum advantage.

Final report/PPT compilation remains. No further model experiments are planned.

## Method

At forecast origin `t`, saved causal forecasts predict POA irradiance, ambient temperature and wind at `t + 15 minutes`. The two power routes are:

```text
Direct:      origin information + forecast weather -> future fleet AC power
Correction:  forecast weather -> frozen physics -> physics power + learned correction
             correction label = actual future power - physics power
```

The six direct inputs are forecast target POA, ambient temperature and wind; measured fleet power and POA at the origin; and fleet-power change from `t-15` to `t`. Correction adds `physics_forecast_kw` as its seventh input. Direct models exclude **all physics-derived inputs and their missing indicators** and never add physics to their predictions.

### Chronology and availability

The shared split is defined once in [forecast_stage2_weather.py](scripts/forecast_stage2_weather.py), by **target timestamp and elapsed time**, not row count:

| Portion | Target interval, start inclusive / end exclusive |
| --- | --- |
| Training, 70% | 2017-12-01 00:00 to 2022-01-21 16:45 |
| Validation, 15% | 2022-01-21 16:45 to 2022-12-11 20:20 |
| Testing, 15% | 2022-12-11 20:20 to 2023-10-31 23:50 |

The 70%/85% boundaries round **up** to the five-minute grid; the last included target is 2023-10-31 23:45. Internal gaps do not change these percentages. Targets enter the later portion at a boundary; historical inputs may extend backwards into earlier portions. Every target is exactly origin +15 minutes.

Exports remain **naive Pacific wall-clock timestamps**, without reinterpretation as UTC. Ambiguous/nonexistent local times cannot supply valid solar geometry. Daylight scoring uses **target solar elevation >0 degrees**, not future measured sunlight.

Weather inputs cover 25 five-minute positions over the preceding two hours, including the origin. At each position, temperature/wind use the latest exported record in `[feature time-30 minutes, feature time-15 minutes]`. There is no future-value interpolation. This conservative lookup is an **availability assumption, not verified logger latency**; origin AC/POA are assumed available at the origin. Exact recorded future weather supplies weather labels only; missing targets are never filled.

Weather configurations were selected once within the initial 40% of the training duration, using its chronological 80/20 fit/selection subdivision. Selection uses daylight POA MAE and all-valid-label ambient/wind MAE. Three expanding, earlier-only forecast blocks cover the remaining training period, beginning 2019-07-29 01:55, 2020-05-26 14:55 and 2021-03-25 03:50. These out-of-sample forecasts supply correction training. Configurations stay frozen for outer validation/testing; testing reuses the validation weather models without fitting. Selected weather models are depth-20 RF for POA/ambient and Ridge alpha 0.1 for wind; full settings are [saved here](outputs/2107_stage2_weather/training_1_settings.json).

**Parts of the test period appeared in earlier exploratory work.** It was held out from these model fits and validation selections, but was not an entirely unseen dataset.

### Fitting and comparison rules

- Identical eligible rows are used across direct/correction comparisons: **221,057 training**, **70,555 validation / 43,949 daylight**, and **77,561 test / 48,190 daylight**. The initial weather-fitting period is not correction-training data.
- Fleet actual power requires all 24 QC-valid, finite, nonnegative inverter readings. **Genuine zeros remain; missing readings never become zeros or partial fleet sums.** Current fleet power/POA are required. Missing older power changes remain eligible with training-fitted median imputation and missing indicators.
- Imputation and scaling fit training rows only. Direct preprocessing is separate from correction preprocessing; transformed dimensions include indicators (12 direct, 14 correction). Neural targets use their own training-only standard deviation **without mean subtraction**, then convert back to kW.
- Each direct/correction ML search used **43 full-training-set candidates**: seven Ridge settings and 12 per tree family. The four-family correction search is complete, not a small timing pilot. Every family's validation-selected winner was retained; no refit on validation. Ranges/budget were specified after inspecting initial correction results. [Random search literature](https://www.jmlr.org/papers/v13/bergstra12a.html) supports the strategy, not these exact ranges.
- Power-model configuration and best-checkpoint selection use daylight validation MAE only. All four ML families are reported on testing. Classical neural runs use seeds **2107-2111**; quantum runs use **2107-2109**. Report every seed and mean/**sample SD**, with no best-test-seed selection or ensemble.
- Classical networks use `d -> 4 tanh -> 4 tanh -> 1`. Quantum replaces the middle block with RY encoding and two StronglyEntanglingLayers, ranges `[1,2]`, CNOT entanglers and four Pauli-Z expectations. Both use normalized Huber loss (delta 1), Adam 0.001, float64, up to 100 epochs, patience 10 and minimum validation-MAE improvement 0.001 kW.
- Classical batch size is **128**; final quantum uses **256**, native `default.qubit`/Torch backpropagation, analytic expectations and complex128 simulation. This is a training-setting difference, not an accuracy-equivalence claim. Parameter counts are 77 direct NN, 85 correction NN and 89 quantum correction; capacity is not identical. CPU threads are capped at four.
- Final power is not additionally clipped or postprocessed. Forecasting bias means **prediction minus actual**; the correction label has the opposite sign. Historical Stage 1 residual/MBE uses measured minus expected.

## Frozen physics and material limitations

The [frozen Stage 1 route](docs/research_baseline.md) uses measured POA as effective irradiance, Faiman temperature (`u0=25, u1=6.84`), the CEC single-diode module model and Sandia inverter conversion. Stage 2 passes **predicted** weather through those same calculations at the **target timestamp**, without joining future measured weather. Inverter watts become plant kW as `24 * inverter_W / 1000`.

- The inferred arrangement is **20 modules in series x 6 parallel strings per inverter**, 24 nominal arrays. It is supported by electrical evidence, not independently confirmed wiring. Balanced 3+3 tracker allocation is compatible; cold-voltage design and permitted oversizing remain unverified.
- The unchanged CEC module is `Hyundai_Heavy_Industries_Green_Energy_Co__HiS_M310TI`. Metadata technology conflicts with the manufacturer's multicrystalline description; CEC `N_s=60` conflicts with the December-2015 datasheet's 72 cells. Electrical/temperature-coefficient and fitted-current inconsistencies remain. Preserve the full fitted row: changing cell count alone does not correct the fitted diode parameters.
- The provisional inverter match is `ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_`; installed voltage/suffix/input mode/settings are unconfirmed. The [30 kW extension](docs/inverter_ceiling.md) retains official 27.6 kW coefficients inside the efficiency equation and changes only the upper cap. Manufacturer maximum output is conditional on unity power factor and input/output/current/temperature limits, with thermal and altitude derating. The extension does not enforce tracker, MPPT, grid or thermal limits and extrapolates above the CEC reference rating.
- No module/backsheet/cell-temperature measurement exists: `temp_cell` is a Faiman estimate. No empirical loss factor was fitted; `K_DC` stays unset and the frozen route bypasses that optional layer. Signed Sandia nighttime tare is retained.
- Array tilt/azimuth are documented as 25/180 degrees; POA sensor orientation/calibration is unknown. POA is not corrected for optical effects, shading, soiling or spectral response. An ideal MPP need not equal the measured operating point.
- Export timestamp spacing does not establish sensor update cadence or averaging-window alignment. Ambient/wind have repeated values despite 15-minute timestamps. Zero shift remains: rapid-transition evidence did not support a five-minute-grid clock correction; older `POA(t+10)` tests did not prove a physical ten-minute generation delay.
- Inverter 05 has no measured DC-voltage channel; its usable AC/current remain included. Directly recorded DC power is unavailable; finite V x I is a derived diagnostic and current's input-versus-whole-inverter meaning is not independently established. The historical 5,628 screening candidates are not confirmed faults or the full training record. Missing intervals, common zeros and abrupt/reversible operating changes have no confirmed maintenance/curtailment cause. Matched AM/PM current bias persisted after thermal/AOI sensitivity checks; no fitted loss or timing correction was adopted.
- Earlier weather-adjusted decline analysis was **inconclusive**, not evidence of physical ageing. Sensor/operational effects remain confounded. This is a single-site predictive comparison, not demonstrated fault diagnosis, RUL, external-site generalization or quantum advantage.

Primary evidence remains in the [original metadata](data/2107/raw/2107_system_metadata.json), [equipment checks](outputs/2107_stage2_readiness/equipment_checks.json) and [manufacturer source manifest](outputs/2107_stage2_readiness/manufacturer_sources.json). See also [saved Stage 1 validation](docs/stage1_validation_summary.md), [timing evidence](outputs/2107_poa_timing_audit/), [thermal/bias sensitivity](outputs/2107_dc_bias/) and the [decline finding](docs/performance_change_feasibility.md). Historical recommendations describe their original dates, not unfinished experiments. Metrics use finite exact pairs without implicit normalization; R² is undefined for constant actual targets. Historical populations differ from final forecasting support.

## Final local test results

All rows below use the same **48,190 daylight / 77,561 all-day** targets. Errors are kW; R² is unitless. Neural error entries are mean ± **sample SD across seeds**, not confidence intervals or ensemble errors; neural R² entries are seed means. Power persistence predicts the latest origin fleet power unchanged.

| Model | Day MAE | Day RMSE | Day R² | All-day MAE |
| --- | ---: | ---: | ---: | ---: |
| Physics only | 62.91 | 98.70 | 0.8387 | 39.16 |
| Power persistence | 43.04 | 79.38 | 0.8957 | 26.77 |
| Direct Ridge | 33.23 | 68.12 | 0.9232 | 21.55 |
| Physics + Ridge | 33.30 | 68.12 | 0.9232 | 21.53 |
| **Direct RF** | **30.92** | **66.93** | **0.9259** | **19.21** |
| Physics + RF | 32.24 | 67.80 | 0.9239 | 20.04 |
| Direct histogram boosting | 31.97 | 67.48 | 0.9246 | 19.93 |
| Physics + histogram boosting | 33.01 | 67.97 | 0.9235 | 20.57 |
| Direct XGBoost | 31.74 | 67.87 | 0.9238 | 19.77 |
| Physics + XGBoost | 32.81 | 68.16 | 0.9231 | 20.42 |
| Direct NN, 5 seeds | 31.40 ± 0.55 | 67.81 ± 0.46 | 0.9239 | 20.53 ± 0.53 |
| Physics + NN, 5 seeds | 31.07 ± 0.82 | 68.01 ± 0.69 | 0.9234 | 19.89 ± 0.43 |
| Physics + simulated VQC, 3 seeds | 31.11 ± 0.90 | 67.87 ± 0.55 | 0.9237 | 19.95 ± 0.82 |

Direct RF performs best overall among the selected ML models and neural seed means. Correction NN has a modest MAE benefit over direct NN (about 0.33 kW in daylight), but not an RMSE benefit. Quantum correction shows **no clear advantage** over classical correction. Small differences and seed SD alone do not establish statistical significance.

Sources: [direct/correction comparison](outputs/2107_stage2_correction/test/comparison.csv), [direct per-seed/full metrics](outputs/2107_stage2_direct/test/metrics.csv), [correction neural metrics](outputs/2107_stage2_test/metrics.csv), and **[completed local quantum metrics, including R²](outputs/2107_stage2_quantum_test/metrics.csv)**. These contain bias and additional all-day metrics; keyed predictions and provenance accompany them.

Returned quantum inference used the returned fitted preprocessing/target scale. The [local compatibility record](outputs/2107_stage2_quantum_test/settings.json) documents numerical tolerances, proven line-ending equivalence and validation replay checks. Historical training-frame/preprocessing/reference fingerprints remain unmatched: inference compatibility does **not** prove identical historical training frames.

The earlier [friend-reported quantum summary](outputs/2107_final_reporting/reported_quantum_results.md) remains accepted historical source material, separate from completed local testing. The existing [full reporting tables](docs/final_results.md), [machine-readable snapshot](outputs/2107_final_reporting/results.csv) and [MAE](outputs/2107_final_reporting/test_ml_mae.svg), [RMSE](outputs/2107_final_reporting/test_ml_rmse.svg), [seed-summary](outputs/2107_final_reporting/test_neural_seed_summary.svg) plots predate local quantum testing. **The reporting snapshot/generator requires refresh before final report/PPT use**: it still reads the friend summary and says local quantum predictions/R² were unavailable. Nothing was regenerated for this README.

## Repository and required artifacts

| Location | Role |
| --- | --- |
| [src/oedi2107/](src/oedi2107/) | Frozen physics, real-data/QC interfaces and metrics. |
| [scripts/](scripts/), [tests/](tests/) | Reproduction/evaluation commands and correctness/leakage checks. |
| [notebooks/](notebooks/) | Original data cleaning, QC and alignment; not required to view saved results. |
| [data/2107/raw/](data/2107/raw/), [data/2107/processed/](data/2107/processed/) | Source measurements/metadata, clean channels, QC events and aligned Stage 1 input. |
| [outputs/2107_research_baseline/](outputs/2107_research_baseline/) | Frozen Stage 1 outputs, validation and provenance. |
| [outputs/2107_stage2_weather/](outputs/2107_stage2_weather/) | Shared-protocol weather selection/models, forecasts, physics/correction labels and per-step settings. |
| [outputs/2107_stage2_correction/](outputs/2107_stage2_correction/) | Original fixed fits; HPO settings/trials/four winning bundles; four-family test results under `test/`. |
| [outputs/2107_stage2_direct/](outputs/2107_stage2_direct/) | Direct preprocessing/scale, ML search/winners, five neural runs and `test/` outputs. |
| [outputs/2107_stage2_neural/](outputs/2107_stage2_neural/), [outputs/2107_stage2_quantum_cpu/](outputs/2107_stage2_quantum_cpu/) | Separate training-fitted preprocessing/scaling, settings and per-seed best/last checkpoints, histories and validation predictions. |
| [outputs/2107_stage2_test/](outputs/2107_stage2_test/), [outputs/2107_stage2_quantum_test/](outputs/2107_stage2_quantum_test/) | Original RF/classical-neural and completed local quantum test predictions, metrics and frozen provenance. |
| [outputs/2107_final_reporting/](outputs/2107_final_reporting/) | Reporting snapshot and presentation exports; refresh limitation above applies. |

**A Git clone is insufficient:** data and nearly all model/output artifacts are ignored. Restore them from verified backups at their recorded paths. Preserve preprocessing, settings, best **and last** checkpoints, histories, predictions and manifests; evaluators use completion/selection evidence as well as weights. Never edit fingerprints to force compatibility. CPU model/checkpoint helpers now live in [stage2_shared.py](scripts/stage2_shared.py). Exact historical source and an explicit approved-runtime hash map are preserved in [historical_sources.zip](docs/historical_sources.zip); retired diagnostics are recorded separately and are never silently treated as verified files. The canonical portable historical input bundle remains [the friend handover ZIP](outputs/2107_stage2_quantum_cpu_friend.zip).

Current external dependencies also remain:
- `C:/Users/silsw/Documents/Codex/2026-10-16/sy/outputs/pv_final_result_tables.md`: reporting generator input and provenance reference.
- `C:/Users/silsw/Documents/Codex/2026-10-16/sy/outputs/quantum_return_audit.md`: recorded local quantum compatibility evidence.

The workflow is not yet fully self-contained. The [inventory](docs/cleanup_inventory.md) records the pre-cleanup review; the historical-source archive records approved source migrations and retired diagnostic paths. Final reporting refresh remains separate.

## Practical use: PowerShell

Run from the repository root. Existing saved results should be read, not regenerated.

```powershell
Set-Location 'C:\Users\silsw\OneDrive\Desktop\oedi-2107-physics-model'
$py = '.\.venv\Scripts\python.exe'
$ErrorActionPreference = 'Stop'
```

### Environment and saved-result inspection

The inspected environment is Python **3.11.9**, pvlib **0.16.1**, Torch **2.14.1+cpu**, PennyLane **0.45.1** and PennyLane-Lightning **0.45.0**. Base/diagnostic versions are pinned in [requirements-lock.txt](requirements-lock.txt); extras are declared in [pyproject.toml](pyproject.toml). Neural packages are not pinned by that base lock; saved run settings record their exact versions.

For a **new checkout without an existing environment**, recreation commands are below. They install packages; do not run them merely to inspect results. Wheel/index availability on another machine is not verified; stop on conflicts rather than upgrading frozen packages.

```powershell
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
& $py -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Locked installation failed' }
& $py -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
if ($LASTEXITCODE -ne 0) { throw 'CPU Torch installation failed' }
& $py -m pip install -c requirements-lock.txt -e '.[test,real-data,neural-quantum]' 'PennyLane==0.45.1' 'PennyLane-Lightning==0.45.0'
if ($LASTEXITCODE -ne 0) { throw 'Constrained installation failed' }
```

Read existing results without fitting, inference or report writes:

```powershell
Import-Csv .\outputs\2107_stage2_correction\test\comparison.csv | Format-Table
Import-Csv .\outputs\2107_stage2_quantum_test\metrics.csv | Format-Table
& $py -B scripts/train_stage2_neural.py --summary
if ($LASTEXITCODE -ne 0) { throw 'Classical summary failed' }
& $py -B scripts/train_stage2_quantum_cpu.py --summary
if ($LASTEXITCODE -ne 0) { throw 'Quantum summary failed' }
```

### Development/reproduction commands: write artifacts or fit models

**All final runs already exist.** These commands document reproduction, not pending work or permission to overwrite completed results. Run only when intentionally restoring/reproducing an appropriate compatible workspace. Preparation modes do not fit full-data models, but some freeze/check settings; quantum preparation also runs tiny equivalence checks. All commands below use `& $py -B` before the script/arguments shown.

| Workflow | Commands in order |
| --- | --- |
| Weather and physics | `scripts/forecast_stage2_weather.py --prepare`; `scripts/calculate_stage2_physics.py --prepare`; then the expanding-step commands below. |
| Correction ML | `scripts/compare_stage2_correction.py` runs/reuses the four fixed baselines; `scripts/compare_stage2_correction.py --prepare` freezes/checks HPO; `scripts/compare_stage2_correction.py --hpo` executes/resumes the full 43-candidate search. |
| Classical correction NN | `scripts/train_stage2_neural.py --prepare`; `scripts/train_stage2_neural.py --train --model classical --seed 2107 --accept-pilot`, for each seed 2107-2111. Requires the existing compatible `outputs/2107_stage2_neural/pilot.json`; this timing pilot is separate from ML HPO. |
| Direct ML/NN | `scripts/compare_stage2_direct.py --prepare`; `scripts/compare_stage2_direct.py --train-ml`; `scripts/compare_stage2_direct.py --train-neural --seed 2107`, for each seed 2107-2111. |
| Final CPU quantum | `scripts/train_stage2_quantum_cpu.py --prepare`; `scripts/train_stage2_quantum_cpu.py --train --seed 2107 --accept-runtime`, for each seed 2107-2109. Returned runs must remain intact; training reproduces the local specification, not a claim of identical friend training provenance. |

The expanding weather/physics steps, only for intentional reproduction:

```powershell
foreach ($step in 'training_1','training_2','training_3','validation') {
    & $py -B scripts/forecast_stage2_weather.py --step $step
    if ($LASTEXITCODE -ne 0) { throw "Weather step $step failed" }
    & $py -B scripts/calculate_stage2_physics.py --step $step
    if ($LASTEXITCODE -ne 0) { throw "Physics step $step failed" }
}
```

Neural continuation adds `--resume` to the same model/seed command. Direct ML continuation is `scripts/compare_stage2_direct.py --train-ml --resume`; correction HPO resumes via `--hpo`. Compatibility checks must pass; do not restart completed seeds. Restore missing winning bundles rather than rerunning the entire search. After each native command check `if ($LASTEXITCODE -ne 0) { throw 'Step failed' }` before continuing.

Stage 1 reproduction is `& $py -B scripts/run_frozen_research.py`, which **overwrites frozen outputs** and is not part of routine review. It first requires the deliberately retired inverter/timestamp candidate masks; historical DC/candidate reruns must restore or separately regenerate their retired derived inputs. Final Stage 2 inference uses the retained forecasts and does not need those diagnostics. `& $py -B scripts/show_stage1_results.py` also writes: it regenerates two Stage 1 summary files and `docs/stage1_validation_summary.md`. Read the saved summary instead. Notebook execution and historical diagnostic runners also write derived artifacts.

### Held-out evaluation: already completed, explicit authorization required

These are the actual entry points for reproduction. **Do not rerun existing evaluations**: they refuse accidental overwrites. Preparation checks/freezes models/settings before scoring; returned-quantum preparation additionally replays a small validation sample, not test rows.

| Workflow | Preparation | Held-out inference/scoring |
| --- | --- | --- |
| Original RF + five correction NN seeds | `scripts/evaluate_stage2.py --prepare` | `scripts/evaluate_stage2.py --evaluate --allow-test` |
| All four correction ML families | `scripts/evaluate_stage2_correction.py --prepare` | `scripts/evaluate_stage2_correction.py --evaluate --allow-test` |
| All four direct ML families + five NN seeds | `scripts/compare_stage2_direct.py --prepare-test` | `scripts/compare_stage2_direct.py --evaluate --allow-test` |
| Returned CPU quantum, three seeds | `scripts/evaluate_stage2_quantum_returned.py --prepare` | `scripts/evaluate_stage2_quantum_returned.py --evaluate --allow-test` |

Prefix each table command with `& $py -B`. In a reproduction lacking test weather/physics inputs, after freezing original RF/NN readiness run `scripts/forecast_stage2_weather.py --step testing --allow-test`, then `scripts/calculate_stage2_physics.py --step testing --allow-test`. These create test inputs without retraining. Original ML/DL evaluation supplies the saved alignment reference needed by the subsequent direct/correction/quantum evaluations.

Saved-result comparison is `& $py -B scripts/evaluate_stage2_correction.py --compare`: **no inference**, but it writes comparison CSV/settings and refuses existing outputs. Reporting regeneration is `& $py -B scripts/report_stage2_results.py`: **no inference**, but it overwrites reporting tables, figures, provenance and `docs/final_results.md`. Defer that command until its local-quantum source handling is refreshed; it currently requires the external compiled table and earlier friend summary.

Correctness checks can be run with `& $py -B -m pytest -q -p no:cacheprovider`; tests may use temporary files. No training, real evaluation, installation, reporting or remote command was executed for this README update.

<details>
<summary>Historical context and retained exploratory results</summary>

Earlier decline work was inconclusive; the separate [5/15/30-minute benchmark](outputs/2107_short_term_forecasting/) used a calendar-year split and different features/daylight rules. Its boosted model beat persistence in daylight but did not reliably improve shutdown/restart forecasts; saved metrics and paired date-block uncertainty remain. Neither is the final shared-split experiment. GPU investigation passed circuit correctness but per-row simulator overhead was impractical; native batching enabled the final CPU route. Kaggle/GPU launch code has been retired.

The superseded setup selected on January-June / July-September 2018 and January-September / October-December 2018, then forecast annual 2019-2021 blocks. Its pooled weather MAE/RMSE results (model vs persistence) were: POA RF depth=None/leaf=10, **34.860089/76.871805 vs 53.469264/92.344000 W/m²**; ambient RF depth=20/leaf=5, **0.474238/0.837520 vs 0.661258/1.248898 °C**; wind Ridge alpha=0.1, **0.221004/0.383598 vs 0.193229/0.396264 m/s**. Seed 2107, four threads. These are exploratory winners, **not current protocol selections**; wind ML did not beat persistence MAE.

| Superseded annual forecast year | Forecasts / valid power labels | Day physics MAE kW | Mean correction kW |
| --- | ---: | ---: | ---: |
| 2019 | 104,797 / 104,795 | 68.070 | -50.726 |
| 2020 | 84,527 / 84,527 | 62.540 | -43.964 |
| 2021 | 87,150 / 87,150 | 94.793 | -76.810 |

All those physics calculations were valid. This compact record survives because the old annual tables/settings were previously removed; it is not final validation/test evidence. Detailed historical methods and scientific evidence remain unchanged under `docs/` and their saved output directories.

</details>
