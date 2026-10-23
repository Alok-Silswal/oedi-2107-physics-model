# Cleanup inventory

Inventory date: 2026-10-23. Recommendations only; nothing was removed, moved, regenerated or trained. This document is the only addition made by this audit.

Sizes below are logical file sizes, using MiB/GiB (1024-based), not measured disk allocation. OneDrive attributes mean these figures are not promises of immediately recovered local disk space. Status: **T** tracked, **I** ignored, **U** untracked. Categories:

- **A Keep:** final workflow, reproducibility or scientific evidence.
- **B Safe removal candidate:** disposable or verified duplicate, subject to separate approval and stated conditions.
- **C Archive outside the repository:** useful history or expensive artifacts; verify a durable, recoverable copy before removing the local copy.
- **D Consolidate first:** unique content or dependencies must survive before removal is considered.
- **E Uncertain:** a specific evidence gap prevents a removal recommendation.

## Git state and size overview

Existing work was preserved: `README.md` was already modified. These untracked files are current scientific/reporting work, **A**, not clutter:

- `docs/final_results.md`
- `scripts/compare_stage2_direct.py`
- `scripts/evaluate_stage2_correction.py`
- `scripts/evaluate_stage2_quantum_returned.py`
- `scripts/report_stage2_results.py`
- `tests/test_stage2_correction_evaluation.py`
- `tests/test_stage2_direct.py`
- `tests/test_stage2_quantum_returned.py`

| Exact path/group | Status; files | Approximate size | Recommendation |
| --- | --- | ---: | --- |
| `outputs/` | I 407, T 2 | 6.38 GiB | Classify individual artifacts below; never remove the whole directory. |
| `data/` | I; 13 | 752.75 MiB | A, raw data, prepared inputs and QC. |
| `.venv/` | I; 47,158 | 1.40 GiB | A, current scientific environment; not a scientific-output duplicate. |
| `.venv-kaggle/` | I; 3,424 | 45.96 MiB | Conditional B; separate tooling environment, see below. |
| `.git/` | Git internal; 1,061 | 5.16 MiB | A; no history rewrite or object cleanup proposed. |
| `notebooks/` | T; 6 | 2.14 MiB | A, executable preparation/QC evidence. |
| `scripts/` | T 21, U 4, I 6 | 550.58 KiB | Keep source; ignored bytecode is B. |
| `tests/` | T 25, U 3, I 14 | 513.55 KiB | Keep source; ignored bytecode is B. |
| `src/` | T 20, I 25 | 233.92 KiB | Keep source; distinguish bytecode from installation metadata. |
| `docs/` before this inventory | T 14, U 1 | 119.50 KiB | Consolidation map below. |
| `examples/` | T; 2 | 1.85 KiB | A, useful documented API demonstrations. |
| `.pytest_cache/` | I; 5 | 11.08 KiB | B, reproducible cache. |
| `README.md` | T, modified | 53.11 KiB | A; eventual canonical consolidated guide. |
| `.gitignore`, `pyproject.toml`, `requirements-lock.txt`, `run_classical.ps1` | T; 4 | 1.97 KiB | A, installation, ignore rules and reproducible launch commands. |

Total inspected logical size is about **8.57 GiB**. No tracked checkpoint, archive or other tracked file above 1 MB was found. Only `outputs/2107_research_baseline/stage1_validation_metrics.csv` and `stage1_validation_summary.txt` are tracked within `outputs/`. GitHub alone therefore does not preserve the datasets, fitted models or predictions.

## Authoritative scientific record: keep

Every row in this section is **A**. Preserve the files and their relative paths. Ignored status does not imply disposability. These are authority/traceability findings from code, settings and saved artifacts, not a fresh model-performance audit.

| Repository-relative paths | Status; size | Purpose and dependency/provenance evidence |
| --- | --- | --- |
| `data/2107/raw/2107_electrical_data.csv`, `2107_environment_data.csv`, `2107_irradiance_data.csv`, `2107_meter_15m_data.csv`, `2107_system_metadata.json` | I; 445.03 MiB total | Original measurements, equipment/site metadata and units; notebooks and readiness evidence derive from these. Raw and processed files are not redundant copies. |
| `data/2107/processed/2107_electrical_clean.parquet`, `2107_environment_clean.parquet`, `2107_irradiance_clean.parquet`, `2107_meter_clean.parquet`, `2107_model_5min.parquet`; electrical/irradiance/meter `2107_*_qc_events.parquet` | I; 307.72 MiB total | Prepared inputs and separate QC evidence. Frozen Stage 1 uses the aligned table; causal Stage 2 loaders use separate cleaned channels. Retain both routes. |
| `src/oedi2107/` Python files, `scripts/run_frozen_research.py`, `scripts/show_stage1_results.py` | T; source is small | Frozen equipment, Faiman/CEC/Sandia calculations, documented 30 kW extension, strict fleet aggregation and validation. `show_stage1_results.py` writes the tracked summaries and `docs/stage1_validation_summary.md`; it is not a read-only viewer. |
| `outputs/2107_research_baseline/` | I + 2 T; 1.63 GiB | Frozen nominal/per-inverter/plant/meter outputs, signed AC and validation support variants, metrics and provenance. In particular keep `inverter_outputs.parquet` (1.57 GiB); size is not grounds to discard frozen evidence. |
| `scripts/forecast_stage2_weather.py`, `scripts/calculate_stage2_physics.py`; `outputs/2107_stage2_weather/` except the three archival candidates below | T code, I outputs; full output group 1.63 GiB | Single shared split, weather search/configurations, causal lookup, OOF forecasts and frozen physics labels. Retain `training_1_selection.csv`, all five step settings, all `forecasts_{training_1,training_2,training_3,validation,testing}.parquet`, all matching `physics_*.parquet` and `physics_*_settings.json`. These preserve issuance rows, actual labels, missingness and weather/physics provenance. |
| `outputs/2107_stage2_weather/validation_models.joblib` | I; 534.68 MiB | Frozen weather bundle used for test forecasts; recorded in final evaluation source hashes. Not interchangeable with the historical training-block bundles. |
| `scripts/compare_stage2_correction.py`; `outputs/2107_stage2_correction/` | T code, I outputs; 114.09 MiB | Fixed four-model comparison, full HPO candidate plan/43-trial ledger, per-family winners, validation predictions and four-family test/comparison results. Keep root `ridge.joblib`, `forest.joblib`, `histogram.joblib`, `xgboost.joblib`, `settings.json`, `metrics.csv`, `validation_predictions.parquet`: HPO records baseline hashes, and neural preparation uses root Ridge preprocessing. Tuned winners do not make these originals removable. |
| `outputs/2107_stage2_correction/hpo/{settings.json,trials.csv,best_ridge.joblib,best_forest.joblib,best_histogram.joblib,best_xgboost.joblib,best_validation_predictions.parquet}` | I; within preceding total | Full selection evidence, including `forest_05`, and all four validation-selected models; retain settings and predictions rather than only reported scores. |
| `scripts/train_stage2_neural.py`; `outputs/2107_stage2_neural/` | T code, I outputs; 14.08 MiB | Training-only preprocessing, uncentered residual scale, settings, original pilot evidence and five classical seeds. Root `pilot.json` is still explicitly required/matched by the training command even with `--accept-pilot`. |
| `scripts/compare_stage2_direct.py`; `outputs/2107_stage2_direct/` | U code, I outputs; 103.16 MiB | Independent direct preprocessing/actual-power target scale, 43-trial ML selection, four winning models, five neural runs, keyed validation/test outputs and small `results_summary.csv`. Preserve physics-feature exclusion evidence and every per-family winner. |
| `scripts/train_stage2_quantum_cpu.py`, `scripts/train_stage2_quantum_gpu.py`; `outputs/2107_stage2_quantum_cpu/` | T code, I outputs; 8.44 MiB | Returned training-fitted preprocessing/settings plus six files per seed 2107–2109. Four-qubit/two-layer native-batched CPU circuit, float64, batch 256. GPU-named source is a live CPU dependency, not disposable packaging. Preserve original returned specifications. |
| `scripts/evaluate_stage2.py`; `outputs/2107_stage2_test/{settings.json,predictions.parquet,metrics.csv}` | T code, I outputs; 6.48 MiB | Original RF + five classical-neural held-out reference: exact keys, actual labels, physics, daylight and prediction columns. Shared test alignment authority, not superseded by later comparisons. |
| `scripts/evaluate_stage2_correction.py`; `outputs/2107_stage2_correction/test/{settings.json,predictions.parquet,metrics.csv,comparison.csv,comparison_settings.json}` | U code, I outputs; 5.99 MiB within correction total | Four-family correction test results and direct/correction comparison provenance. Original ML/DL reference remains required. |
| `outputs/2107_stage2_direct/test/{settings.json,predictions.parquet,metrics.csv}` | I; within direct total | All four direct ML families and five direct neural seeds on the shared saved test rows. |
| `scripts/evaluate_stage2_quantum.py`, `scripts/evaluate_stage2_quantum_returned.py`; `outputs/2107_stage2_quantum_test/{settings.json,predictions.parquet,metrics.csv}` | T + U code, I outputs; 6.40 MiB | **Completed local returned-model quantum test record**, protocol `stage2-returned-cpu-quantum-test-v1`. The original strict evaluator supplies shared alignment/metric/checkpoint helpers; the returned evaluator records the justified inference compatibility path. Keep both. |
| `outputs/2107_final_reporting/{reported_quantum_results.md,provenance.json,results.csv,test_ml_mae.png,test_ml_mae.svg,test_ml_rmse.png,test_ml_rmse.svg,test_neural_seed_summary.png,test_neural_seed_summary.svg}`, `scripts/report_stage2_results.py`, `docs/final_results.md` | I outputs, U code/document; output total 0.78 MiB | Existing reporting snapshot and its generator/source receipt. Quantum numbers here came from the friend-verified, user-accepted reported summary; these are distinct from subsequently completed local quantum testing. Preserve both sources until final compilation explicitly reconciles them. |

### Preserve checkpoint selection and resume evidence

Across direct neural, classical correction and returned quantum runs, keep root preprocessing/settings and each seed's `settings.json`, `history.csv`, `metrics.json`, `validation_predictions.parquet`, `best.pt`, `last.pt`.

- Weights, architecture, feature order, fitted preprocessing and target scale support inference. Histories, validation predictions and best-epoch metadata explain selection.
- `last.pt` also carries optimizer/RNG/epoch state for deterministic resume. Although this state is not mathematically needed for inference, current completion/evaluation checks inspect **both** checkpoints and their terminal/best-selection evidence. Frozen evaluation plans hash them. Do not remove `last.pt` as an alleged redundant checkpoint.
- No model/checkpoint deserialization or new inference was needed for this inventory. Existing saved compatibility evidence is retained, not independently regenerated here.

### Preserve unresolved returned-model provenance

`outputs/2107_stage2_quantum_test/settings.json` explicitly distinguishes inference compatibility from proof of identical historical training frames/reference files. It records:

- Returned/local target scales `95.7946774290642` and `95.79467742904654`; numerical preprocessing comparisons and explicit tolerances.
- Exact-byte or CRLF/LF source-equivalence findings; original byte hashes remain different where applicable.
- Unmatched frame fingerprints, serialized preprocessing hash, classical reference hash and reference commit, plus software/Python differences. Exact friend source data/reference files were not supplied; equality is not established.
- Sampled validation replay agreement and the returned preprocessing/scale used for inference.

Keep this evidence, returned settings/checkpoints, local classical reference and the external return audit. Do not rewrite history to mark unmatched fingerprints as matching. Local testing is completed; unresolved training provenance is a separate limitation, not a reason to label all quantum test results pending.

## Dependency blockers: do not remove or refactor in this cleanup

Imports, test imports, commands, notebook cells, project configuration and nested saved hash maps were inspected. Final evaluation/selection/reporting manifests collectively reference 136 repository-relative paths; including historical manifests increases this to 196. These are recorded references, not a claim that every edge is checked by every command.

| Paths | Status/size | Blocker and recommendation |
| --- | --- | --- |
| `scripts/train_stage2_quantum_gpu.py` | T; 28.64 KiB | **A.** CPU training imports `make_model`, `read_checkpoint`, `compare`, `broadcast_evidence`; direct and original ML/DL evaluation also import restricted checkpoint reading. CPU construction still traverses the shared neural/quantum helpers. Filename does not establish obsolescence. |
| `scripts/evaluate_stage2_quantum.py` | T; 11.08 KiB | **A.** Direct training/evaluation, correction evaluation, returned evaluator and reporting reuse alignment/metric helpers. `tests/test_stage2_quantum_returned.py` also imports a fixture from `tests/test_stage2_quantum_evaluation.py`. Neither original evaluator nor its tests can simply be deleted. |
| `scripts/audit_stage2_readiness.py` | T; 20.32 KiB | **A.** Live `digest`, `indexed`, `solar_elevation` helpers are used by retained forecasting/physics/correction/reporting paths. Also preserves equipment/coverage reproducibility. No helper relocation proposed. |
| `scripts/benchmark_short_term_forecasting.py` | T; 15.27 KiB | **A.** Weather forecasting reuses causal `available_environment`; correction comparison reuses paired date-block uncertainty utilities. Its earlier benchmark is history, but its code remains active. |
| `scripts/kaggle_stage2_pilot.py`, `tests/test_stage2_quantum_gpu.py` | T; 29.60/29.98 KiB | **A for now.** Tests exercise packaging plus numerical broadcasting/gradient behavior, and import this launcher. Generated remote packages can be archived without deleting shared source/test code. No source removal is justified just because remote execution ended. |
| `scripts/experiment_performance_change.py`, `tests/test_performance_change.py` | T; 15.79/1.44 KiB | **A.** Only reproducible decline/regime experiment, with tests importing `slope_fit`/`aggregate`. Preserve the inconclusive finding, not a claim of physical ageing. |
| `scripts/run_real_dc.py`, `run_no_loss_ac.py`, `identify_dc_candidates.py`, `decompose_dc_bias.py`, `audit_poa_timing.py`, `run_inverter_ceiling.py`, `run_measured_conversion.py`; associated `src/oedi2107/` modules and tests | T; small relative to artifacts | **A.** Independent diagnostic entry points, README/report references, public exports and tests explain frozen decisions. No detected justification to remove source or its tests. Large derived tables are separate archival candidates below. |
| `pyproject.toml`, `requirements-lock.txt`, `.gitignore`, `run_classical.ps1` | T; 1.97 KiB total | **A.** Package discovery is `src`, pytest uses `tests`; optional dependencies separate real-data/neural/GPU support. The PowerShell wrapper is an independent five-seed classical reproduction command. No dependency removals proposed. |

Historical source/artifact hashes make even seemingly harmless edits or moves capable of breaking preparation, evaluation or resume. Keep fingerprinted source unchanged. Archived artifact proposals below require exact-path restoration before the historical workflow is replayed.

Two external files are also recorded dependencies, not repository cleanup candidates:

- `C:/Users/silsw/Documents/Codex/2026-10-16/sy/outputs/pv_final_result_tables.md`: hard-coded input to `scripts/report_stage2_results.py`, and recorded in reporting `provenance.json`.
- `C:/Users/silsw/Documents/Codex/2026-10-16/sy/outputs/quantum_return_audit.md`: recorded in completed local quantum test `settings.json`; the evaluator includes it when available during plan preparation. Keep its exact content accessible for frozen-plan verification.

The location/content dependence is a blocker to casually moving these files, not a request to refactor paths in this audit.

## Small source, notebooks and examples

| Paths | Status; size | Purpose, references and category |
| --- | --- | --- |
| All 25 `scripts/*.py`, 20 `src/oedi2107/**/*.py`, 28 `tests/test_*.py` | T/U; about 388/88/175 KiB | **A.** Entry points and regression checks cover physics, units/signs, QC, causal lookup, target boundaries, complete-fleet coverage, training-only preprocessing, checkpoint compatibility and saved-result alignment. No whole test/source group is demonstrably disposable. Retained diagnostic code also has independent command/document/test use. |
| `examples/run_foundation.py`, `examples/run_inverter.py` | T; 1.85 KiB combined | **A.** README invokes/links both. Synthetic API demonstrations: optional DC foundation (including unset `K_DC`) and CEC inverter conversion/tare/clipping. No output files; neither is a final scientific pipeline. No test imports or packaging requirement detected; `src` package discovery and generated `SOURCES.txt` do not include them. Small, useful documentation examples; no worthwhile saving. |
| `notebooks/01_electrical_data_audit.ipynb`, `02_irradiance_data_audit.ipynb`, `03_environment_data_audit.ipynb`, `04_meter_data_audit.ipynb` | T; 0.62/0.48/0.25/0.58 MiB | **A.** Executable cleaning/QC basis, preserved genuine zeros and missingness, environmental unit conversion, weather repetition/cadence and meter meaning. Saved outputs are evidence, not disposable merely because bulky relative to source. |
| `notebooks/05_dataset_alignment.ipynb`, `06_build_aligned_dataset.ipynb` | T; 0.19/0.025 MiB | **A.** Time/alignment findings and construction of `2107_model_5min.parquet`. The historical aligned table's same-interval weather treatment must not be confused with Stage 2's conservative backward availability lookup. |

Notebook output payloads total roughly 1.91 MiB. No blanket output clearing is recommended: little space is gained, while QC tables/figures and interval/units evidence would disappear. Notebooks were inspected as JSON/text, never executed. No separate abandoned `examples/` packaging/test dependency was found.

## B: immediate mechanical candidates, with conditions

These are recommendations for later approval, not deletions performed by this audit.

| Exact paths | Status; size | Evidence, proposed action and condition |
| --- | --- | --- |
| `.pytest_cache/`; `scripts/__pycache__/`; `src/oedi2107/__pycache__/`; `src/oedi2107/validation/__pycache__/`; `tests/__pycache__/` | I; **0.64 MiB combined** | Reproducible pytest state and compiled Python bytecode. No source/manifest dependency detected. Remove only these cache directories, with no concurrent Python/test writer; retain source/tests. Do not include environment package directories. |
| `outputs/2107_stage2_quantum_broadcast/dataset_upload/project.zip` | I; **45.82 MiB** | Byte-identical to retained `outputs/2107_kaggle_quantum_gpu_reference_fix/dataset/project.zip` and the `project.zip` member inside the retained friend archive. SHA-256 `91e3e2e73ebaf3b46149a4e1ed98f0888f85b755555ba96700c14e4e9a92c437`. No final consumer/hash reference to this duplicate path detected. Remove only this ZIP after verifying the retained copies remain readable; restoring the dataset-upload layout would require copying it back. Keep its metadata/notebook/CPU timing record separately. |
| `.venv-kaggle/` | I; **45.96 MiB**, optional | Configuration-only tooling environment, installed Kaggle CLI **2.2.4**, metadata requires Python `>=3.11`. No final scientific inference dependency. Candidate only after confirming no further Kaggle work is wanted and recording/reconciling its setup instructions. Unlike caches, this requires reinstatement before CLI use. Do not touch external credentials or the main `.venv/`. |

The first two candidates total **46.46 MiB**. Including the separately conditional Kaggle environment would make **92.42 MiB**. No other file was classified disposable solely by age, extension, ignored state or lack of imports.

`src/oedi_2107_physics_model.egg-info/` is ignored generated installation metadata (five files, about 2.62 KiB). **E:** editable-install/package metadata use prevents treating it as a free cache deletion. Negligible gain; retain unless an installation maintenance task establishes safe regeneration. No loose temporary/backup files warranting a separate deletion proposal were found in the inventoried project tree.

## C: archive candidates, not disposable outputs

Archive only to a durable external destination, verify content/readability and a restore location, then separately approve local removal. Preserve small metrics, provenance, assumptions and links in place. None of these proposals authorizes deletion of an entire output folder.

| Exact paths | Status; size | Purpose/references; conditions before removing local copy |
| --- | --- | --- |
| `outputs/2107_stage2_weather/training_1_models.joblib`, `training_2_models.joblib`, `training_3_models.joblib` | I; 271.62 + 370.99 + 453.09 MiB = **1.07 GiB** | Expensive historical expanding weather fits. Forecast rows/settings already preserve correction-training inputs; no recorded hash dependency on these three bundles was detected. Still needed to replay their weather inference without retraining. Archive verified bundles; retain all training-block forecasts, physics tables, settings and selection ledger locally. **Exclude `validation_models.joblib`.** |
| `outputs/2107_dc_ideal/inverter_dc_ideal.parquet` | I; **981.77 MiB** | Historical full-inverter ideal-DC diagnostic, referenced by historical provenance. Archive only with exact-path restoration available; retain nominal diagnostic, metrics and provenance. Not the frozen research-baseline `inverter_outputs.parquet`. |
| `outputs/2107_ac_no_loss/inverter_ac_no_loss.parquet` | I; **616.29 MiB** | Earlier nominal-CEC no-loss AC diagnostic used by independent scripts/reports. No final recorded hash edge detected, but this is unique output, not a duplicate. Archive; keep nominal AC, comparisons and provenance locally. |
| `outputs/2107_measured_conversion/inverter_measured_conversion.parquet` | I; **485.82 MiB** | Unique measured-DC-to-AC diagnostic underpinning the inverter-ceiling reasoning. Archive with supporting code/settings and restore ability; keep compact metrics/summary/provenance locally. |
| `outputs/2107_inverter_ceiling/inv_01.parquet`, `inv_02.parquet`, `inv_03.parquet`, `inv_04.parquet`, `inv_06.parquet`, `inv_07.parquet`, `inv_08.parquet`, `inv_09.parquet`, `inv_10.parquet`, `inv_11.parquet`, `inv_12.parquet`, `inv_13.parquet`, `inv_14.parquet`, `inv_15.parquet`, `inv_16.parquet`, `inv_17.parquet`, `inv_18.parquet`, `inv_19.parquet`, `inv_20.parquet`, `inv_21.parquet`, `inv_22.parquet`, `inv_23.parquet`, `inv_24.parquet` | I; **224.10 MiB** combined | 23 measured-input diagnostic tables; no `inv_05` table because its voltage channel is absent. Archive together; retain `pooled_errors.csv`, `per_inverter_errors.csv`, `provenance.json` and method documentation. Do not reconstruct inverter 05 or infer missing equals zero. |
| `outputs/2107_dc_candidates/inverter_candidate_masks.parquet`, `timestamp_diagnostics.parquet`, `timestamp_candidate_masks.parquet` | I; **153.13 MiB**, optional later | Unique screening masks/diagnostics with historical hash references. Less attractive than the preceding candidates because they encode exact screening support. Archive only with the rules, candidate summaries and provenance, keeping these locally or restoring exact files for historical verification. Screening labels are not confirmed faults. |

The four bulk diagnostic rows, excluding optional candidate masks, total **2.25 GiB**. With the three weather bundles, these offer **3.32 GiB** of potential local archival savings, not scientifically disposable space. No final keyed validation/test predictions are proposed for removal or archival out of the active final workflow.

### Historical Kaggle packages, logs and timing evidence

All following output paths are **I**. The six `2107_kaggle*` groups occupy 275.32 MiB altogether. Preserve the exact canonical `reference_fix/dataset/project.zip` locally for now; the remainder is approximately **229.49 MiB**, **C**, only after a verified external historical archive. Archive packages together with their paired metadata/launchers/logs; do not flatten variants by filename or size.

| Exact group | Approximate size | Purpose and dependency/condition |
| --- | ---: | --- |
| `outputs/2107_kaggle_pilot/` | 45.83 MiB | Original CPU batch package: `dataset/{dataset-metadata.json,project.zip}`, `kernel/{kaggle_stage2_pilot.py,kernel-metadata.json}`. Historical self-contained source/input snapshot; archive intact. |
| `outputs/2107_kaggle_quantum_gpu/` | 45.96 MiB | GPU launch preparation plus `pilot`, `pilot_env_fix`, `training` metadata/scripts and `pilot_output`, `pilot_env_fix_output`, `pilot_portable_output`, `pilot_verified_output` downloaded status/logs. Preserve failure phases/environment/fingerprint history with its package; not evidence of completed GPU training. |
| `outputs/2107_kaggle_quantum_gpu_portable/`, `outputs/2107_kaggle_quantum_gpu_verified/` | 45.86/45.87 MiB | Root/path verification variants, each with `dataset`, `pilot`, `training` files. Archive as distinct versions; equal ZIP size does not imply equal bytes. |
| `outputs/2107_kaggle_quantum_gpu_diagnostic/` | 45.88 MiB | Optional vectorization/synchronization diagnostic package, not final CPU training. Archive its source/metadata/package together; no inference dependency detected. |
| `outputs/2107_kaggle_quantum_gpu_reference_fix/` | 45.92 MiB total | **Keep `dataset/project.zip` (45.82 MiB) locally.** Archive other files only with their context. Keep the successful GPU correctness/timing evidence recoverable: `pilot_output/2107_stage2_quantum_gpu/gpu_pilot.json`, companion batch log/status, launcher and metadata. |
| `outputs/2107_stage2_quantum_broadcast/kaggle_benchmark.ipynb`, `kernel-metadata.json`, `dataset_upload/dataset-metadata.json` | 64.81 KiB combined | C, bounded remote benchmark preparation. Keep `benchmark_cpu.json` (19.12 KiB), **A**, as actual native-batch CPU engineering timing evidence; duplicate ZIP is the separate B candidate above. |
| `outputs/2107_stage2_quantum_cpu_friend.zip` | **45.84 MiB** | **A, not a deletion candidate.** Six-member handover with `project.zip`, source/test snapshots and `HANDOVER.md`. Inner project contains 44 runtime/source/input files, including classical reference settings, correction Ridge preprocessing/HPO settings, cleaned inputs, training/validation physics and weather settings, dependency constraints and source. It may be the only complete portable handover bundle. Keep until a verified equivalent durable replacement is agreed. It does not prove the returned run used byte-identical historical inputs. |

Archive comparison actually performed: SHA-256 of all eight outer ZIPs and their member inventories, plus the friend ZIP's inner project SHA. The broadcast ZIP and reference-fix ZIP match the inner friend project exactly. The other standalone ZIP hashes differ, including several with the same byte length. No claim of semantic equivalence across those different variants is justified without comparing their individual payloads. They are **C**, not B.

The repository-root `quantum_cpu_results.zip` is already absent; it is not a new cleanup candidate. The preserved Downloads return ZIP and any off-repository source/audit backups must remain untouched. This audit did not revalidate the external backup's durability or inspect credentials.

## Earlier scientific evidence: retain compact records

These are **A** unless a specific large table is listed as C above. Older does not mean obsolete.

| Exact group | Status; size | Unique scientific value and references |
| --- | --- | --- |
| `outputs/2107_stage2_readiness/` | I; 15.72 MiB | Equipment scenarios, original coverage by year/inverter, persistent jumps, ramp/timing tests, weather support, `manufacturer_sources.json`, provenance. Keep `sources/abb_trio_us_manual.pdf`, `abb_trio_us_revg.pdf`, `hyundai_ti_dec2015.pdf` **and** corresponding `.txt` extraction evidence. PDF and searchable extraction have different purposes, not blind duplicates. |
| `outputs/2107_poa_timing_audit/` | I; 0.45 MiB | Offset scores, metadata, seasonal coverage and expected-current support; protects the inconclusive timing finding and zero-shift baseline. |
| `outputs/2107_dc_bias/` | I; 18.41 MiB | Matched-support AM/PM decomposition and thermal/AOI sensitivity evidence. Do not rerun thermal experiments or remove the evidence that estimated cell temperature is not ground truth. |
| `outputs/2107_dc_candidates/` apart from optional archived Parquets | I; full group 154.71 MiB | Screening criteria/coverage and saved event evidence; candidate timestamps are not the full usable record and are not confirmed alarms. |
| `outputs/2107_performance_change/` | I; 2.80 MiB | Annual/monthly support, fixed weather-reference model, regime boundaries, indices/slopes and plots. Inconclusive long-term decline versus reversible operational changes; simulated validation does not prove real ageing. No final forecasting-test claims should be attached to these plots. |
| `outputs/2107_short_term_forecasting/` | I; 13.45 MiB | Earlier 5/15/30-minute benchmark: specification, model, keyed predictions, support, metrics, date-block comparisons, provenance and two plots. Historical calendar split differs from final shared split. Keep as prior exploration and test-exposure evidence; never relabel its plots as final results. |
| Remaining small files in `outputs/2107_dc_ideal/`, `outputs/2107_ac_no_loss/`, `outputs/2107_measured_conversion/`, `outputs/2107_inverter_ceiling/` | I; group totals 1015.82/651.14/486.27/224.12 MiB before proposed archival | Nominal model tables, comparisons, parameter assumptions and provenance are useful explanations of frozen Stage 1. Archive only the specifically listed bulk files, not the folders. |

No redundant final prediction export was identified for deletion: direct, correction, original ML/DL and local quantum tables contain different model columns and serve exact alignment/provenance checks. PNG/SVG pairs are raster/vector delivery formats, not gratuitous duplicates. The previous downloaded duplicate Parquet/CSV files are not additional copies in this repository inventory.

## D: documentation consolidation map

Prefer a current comprehensive README, but do not remove unique science to shorten navigation. Sizes are KiB. All documents below are T except `docs/final_results.md` (U). Before any D document is removed: incorporate the listed content into the appropriate README methods/history section, retain its evidence links, update all README/notebook/command references, and check recorded hashes. No such consolidation was performed here.

| Exact path | Size | Category; unique content and required destination/condition |
| --- | ---: | --- |
| `README.md` | 53.11 | **A / consolidation destination.** Existing modified master guide includes setup, commands and scientific assumptions. Historical completion/status statements and GPU/five-seed preparation instructions need a later targeted reconciliation with the completed three-seed CPU/local-test record. It is not yet a wholly current final-results guide. Preserve unrelated edits. |
| `docs/research_baseline.md` | 8.33 | **A compact methods record.** Frozen ideal/pre-loss route, equipment/array assumptions, signed AC/30 kW extension, bypass of unset empirical `K_DC`, no calibrated losses. Retain as frozen source of truth rather than removing during navigation cleanup. |
| `docs/inverter_ceiling.md` | 3.70 | **A compact methods record.** Exact controlled extension, measured-DC support excluding inverter-05 voltage, manufacturer's conditional maximum versus nominal rating; extension is not merely replacing `Paco`. README should link it. |
| `docs/stage1_validation_summary.md` | 4.18 | **A.** Generated frozen numerical validation and support definitions; its generator also writes two tracked output summaries. Keep exact results, do not rerun for cosmetic cleanup. |
| `docs/performance_change_feasibility.md` | 7.79 | **A.** Inconclusive decline/regime finding and limitations; additionally pinned by `outputs/2107_short_term_forecasting/provenance.json` protected hashes. A README summary cannot replace the original hashed file. |
| `docs/real_dc_run.md` | 4.74 | **D.** Channel mappings/units, naive Pacific/DST limitations, missingness, aligned inputs, run/output contracts. Incorporate into README data/preparation/commands; preserve distinction between historical alignment and causal forecasting inputs. |
| `docs/real_dc_findings.md` | 7.76 | **D.** Uncalibrated DC bias, inferred topology, missing inverter-05 voltage and extreme-value limitations. Merge findings/evidence links into README frozen assumptions/data quality; do not convert screening into fault diagnoses. |
| `docs/validation.md` | 8.09 | **D.** Units/signs, finite paired support, metric/normalization conventions and workflow contracts. Merge into README validation section without weakening tests or dropping edge-case interpretation. |
| `docs/ac_no_loss_reference.md` | 7.28 | **D.** Nominal CEC 27.6 kW/tare/clipping versus observed near-30 kW behavior, nighttime/loading bias. Merge the distinction and evidence into README equipment/method limitations before removal. |
| `docs/measured_conversion.md` | 4.33 | **D.** Measured `V*I` diagnostic, current-channel uncertainty, conversion bias and conditional ceiling evidence. Merge into README assumptions with exact saved comparisons; do not imply directly recorded DC power. |
| `docs/dc_candidate_screening.md` | 9.20 | **D.** Common screening rules, 5,628 candidate timestamps, exclusion/support rules and candidate-label limitations. Merge into README historical diagnostics; retain saved masks/provenance locally or in verified archive. |
| `docs/dc_bias_decomposition.md` | 10.47 | **D.** Matched current/power asymmetry, thermal/AOI sensitivity and what remains unexplained. Merge concise findings into README limitations; retain experiments and avoid treating estimated temperature as truth. |
| `docs/poa_timing_audit.md` | 5.21 | **D.** `POA(t+10)` at electrical time `t`, fixed temperature in older tests, uncertain interval semantics; not proof of a generation delay. Merge into README timing/availability, retaining zero-shift baseline and evidence paths. |
| `docs/stage2_readiness.md` | 19.81 | **D, substantial unique evidence.** Equipment primary-source conflicts, array compatibility versus confirmed wiring, cadence, missingness/coverage, no measured panel temperature, abrupt/reversible changes and research limits. Merge all essential facts below into README, keeping manufacturer documents and saved diagnostics. Not disposable merely because readiness is complete. |
| `docs/short_term_forecasting_benchmark.md` | 8.31 | **D.** Prior chronological benchmark, origin-only availability, paired date-block uncertainty, shutdown/rapid-change limitations and test exposure. Merge a brief historical result/method summary into README; retain code because its utilities are live and keep saved predictions/provenance. |
| `docs/final_results.md` | 10.29 | **A current reporting draft**, reconcile during final compilation rather than delete. It and the generator describe the earlier friend-reported quantum source and predate the completed local test record. Retain accepted reported results separately; update future final tables from explicit authoritative sources, not by overwriting historical provenance. |

Mandatory content to survive any consolidation:

- Hyundai metadata/December-2015 manufacturer/CEC technology and 72-versus-60-cell conflict; fitted CEC parameters must not be mixed or blindly corrected. Inferred 20-series × 6-parallel is compatible evidence, not independently confirmed wiring. Module and irradiance-sensor orientations are separate facts.
- ABB variant/region uncertainty, nominal versus conditional 30 kW maximum, per-tracker versus whole-inverter limits and operating restrictions; model extension does not represent every thermal/MPPT/grid limit.
- No measured panel/backsheet/cell temperature; Faiman `temp_cell` is calculated. Missing voltage for inverter 05 does not erase its usable channels. Recorded zeros remain; missing fleet readings never become zeros. No confirmed maintenance/alarms/curtailment records should be invented from screening labels.
- Five-minute exported timestamps do not prove five-minute sensor updates or instantaneous readings. Preserve unknown interval semantics and conservative weather availability: latest record at least 15 minutes before each feature timestamp, tolerance 15 minutes, without later-value interpolation. This is an availability assumption, not verified logger latency.
- Shared elapsed-time target split `[2017-12-01 00:00, 2023-10-31 23:50)`, validation from `2022-01-21 16:45`, test from `2022-12-11 20:20`; boundaries ceil to five minutes, timestamps remain naive Pacific wall-clock. Internal gaps do not alter percentages; historical inputs may precede a boundary, labels may not cross it.
- Two-hour weather history/15-minute horizon, earlier-only weather selection and expanding forecasts; training-only preprocessing/scaling, direct versus residual feature/target distinction and residual sign `actual - physics`.
- Validation-only search/checkpoint selection; five classical neural seeds versus three CPU quantum seeds, quantum batch 256 as a changed training setting, no best-test-seed selection/ensemble, sample SD rather than confidence intervals. Parts of the test period were explored previously.
- Returned-model numerical/source-line-ending compatibility and still-unmatched historical provenance; accepted friend-reported results and completed local evaluation remain distinguishable records.

## E: uncertainties and final-report blockers

- **Reporting freshness:** `scripts/report_stage2_results.py` still reads `reported_quantum_results.md`; its generated prose says local quantum predictions/provenance were not supplied. Local `outputs/2107_stage2_quantum_test/` now exists. The next final-report task must choose/label the local versus reported source explicitly and regenerate presentation artifacts only when authorized. Do not silently discard either source or treat old plots as new local results.
- **External completeness:** current reporting and quantum evaluation plans depend on the two external files listed above. Historical return fingerprints still lack exact friend input/reference artifacts. A cleanup cannot resolve these facts or justify stronger claims.
- **Package variants:** only one exact duplicate ZIP was demonstrated. Distinct outer hashes do not prove different scientific payloads, and equal sizes do not prove equivalence. Verify payload equivalence before any further deduplication; otherwise archive intact.
- **Durable backups:** no offsite backup/recovery rehearsal was performed. C recommendations require one, especially expensive fitted weather bundles and diagnostic Parquets. Preserve source settings/hashes and restoration paths; a summary table alone is not a backup.
- **Environments:** `.venv/` is currently useful for reporting/evaluation. Recreating all installed versions from public availability/offline wheels was not established. Keep it; do not prune packages or dependencies from this audit. Kaggle's optional environment is a separate user decision, not scientific cleanup.
- **Space estimates:** logical sizes exclude OneDrive allocation behavior; externally archiving a cloud-backed file may not recover its full logical size locally. No cache/environment savings should be counted twice with scientific artifacts.

## Proposed approvals, in order

1. **B — mechanical:** remove only the five listed project cache directories, about **0.64 MiB**, after checking no concurrent writer.
2. **B — exact duplicate:** remove only broadcast `dataset_upload/project.zip`, **45.82 MiB**, keeping the byte-verified reference-fix ZIP and friend handover bundle.
3. **C — expensive historical fits:** externally archive the three named weather training-block model bundles, **1.07 GiB**; retain all forecast/physics tables and settings locally. Verify restoration first.
4. **C — bulk historical diagnostics:** externally archive the three named full-inverter Parquets plus 23 ceiling Parquets, **2.25 GiB**. Keep compact evidence and frozen research outputs untouched; restore exact paths for historical reruns/checks.
5. **C — remote execution history:** externally archive historical Kaggle package/launcher/log groups **except** the retained canonical reference-fix ZIP, about **229.49 MiB**. Keep CPU timing and live source/helpers/tests. Preserve successful and failed pilot context together.
6. **D — navigation:** consolidate the ten D documents into README using the map above, only after all unique assumptions/findings/evidence links survive. Keep the four compact A methods/validation records, especially the hashed decline report. Small space saving; the benefit is a coherent current guide.
7. **Optional, separate decisions:** retire `.venv-kaggle/` (**45.96 MiB**) if no more remote work is wanted; archive the three candidate-mask/diagnostic Parquets (**153.13 MiB**) only with verified exact restoration. Retain uncertain installation metadata and the main scientific environment.

No source refactoring, helper migration, dependency changes, test deletion, model rerun or whole-folder scientific-output deletion is proposed. The highest-value actions are verified archival, not removal of the final scientific record.
