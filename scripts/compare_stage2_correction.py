"""Four fixed classical correction fits, using saved Stage 2 training/validation only."""
from pathlib import Path
import importlib.metadata as metadata
import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from audit_stage2_readiness import indexed, digest, solar_elevation
from benchmark_short_term_forecasting import paired_intervals
from calculate_stage2_physics import complete_fleet_power
from forecast_stage2_weather import SPLIT, SEED, THREADS, DATA, OUT as WEATHER, estimator, preprocess, split_masks, period_mask, training_blocks
from oedi2107.real_data import discover_channels
from oedi2107.validation import mae, rmse

OUT = Path("outputs/2107_stage2_correction")
MODELS = {"ridge":{"alpha":10.}, "forest":{"max_depth":12,"min_samples_leaf":5},
          "histogram":{"max_leaf_nodes":15}, "xgboost":{"max_depth":5}}
FEATURES = ["forecast_poa", "forecast_ambient", "forecast_wind", "physics_forecast_kw",
            "origin_power_kw", "origin_poa", "power_change_15_kw"]


def correction_features(table, plant, poa):
    if not table.index.is_unique or not table.target_timestamp.is_unique or not table.index.is_monotonic_increasing:
        raise ValueError("Origin/target keys must be unique and ordered")
    if not ((table.target_timestamp-table.index)==pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected exact origin +15 minute targets")
    result = table[FEATURES[:4]].copy()
    result["origin_power_kw"] = plant.reindex(table.index)
    result["origin_poa"] = poa.reindex(table.index)
    result["power_change_15_kw"] = result.origin_power_kw-plant.reindex(table.index-pd.Timedelta(minutes=15)).to_numpy()
    return result


def examples(table, plant, poa):
    features = correction_features(table, plant, poa)
    labels = np.isfinite(table[["physics_forecast_kw","actual_target_power_kw","correction_kw"]]).all(axis=1)
    np.testing.assert_allclose(table.loc[labels,"actual_target_power_kw"],
        table.loc[labels,"physics_forecast_kw"]+table.loc[labels,"correction_kw"],atol=1e-9)
    eligible = labels.copy()
    counts = {"saved_rows":len(table), "missing_label_or_physics":int((~labels).sum())}
    for name, valid in [("invalid_forecast_inputs",np.isfinite(features.iloc[:,:4]).all(axis=1)),
                        ("missing_current_power",np.isfinite(features.origin_power_kw)),
                        ("missing_current_poa",np.isfinite(features.origin_poa))]:
        counts[name] = int((eligible & ~valid).sum())
        eligible &= valid
    counts.update(eligible=int(eligible.sum()), older_power_missing_retained=int(features.loc[eligible,"power_change_15_kw"].isna().sum()))
    return features.loc[eligible], table.loc[eligible], counts


def load_examples():
    steps = [n for n,_,_ in training_blocks()]+["validation"]
    blocks = {n:(a,b) for n,a,b in training_blocks()}
    blocks["validation"] = (SPLIT["validation"],SPLIT["test"])
    tables, paths = [], []
    winners = json.loads((WEATHER/"training_1_settings.json").read_text())["winners"]
    for step in steps:
        source, settings, weather_settings = WEATHER/f"physics_{step}.parquet", WEATHER/f"physics_{step}_settings.json", WEATHER/f"{step}_settings.json"
        table = pd.read_parquet(source)
        for path in (settings,weather_settings):
            saved = json.loads(path.read_text())
            spec = saved.get("specification",saved)
            if spec["split"]!=SPLIT or spec["protocol"]!="elapsed-70-15-15-v1":
                raise ValueError("Incompatible saved Stage 2 protocol")
            if path==weather_settings and saved["winners"]!=winners:
                raise ValueError("Weather configurations were not frozen")
        label = "validation" if step=="validation" else "training"
        if not table['split'].eq(label).all() or not period_mask(table.index,*blocks[step]).all():
            raise ValueError("Saved targets cross block boundaries")
        tables.append(table)
        paths += [source,settings,weather_settings]
    table = pd.concat(tables).sort_index()
    electrical,eqc,irradiance,iqc = [DATA/f"2107_{name}.parquet" for name in ("electrical_clean","electrical_qc_events","irradiance_clean","irradiance_qc_events")]
    # Push down the cutoff: testing measurements never enter correction development.
    filters = [("measured_on","<",SPLIT["test"])]
    mapping = discover_channels(pq.ParquetFile(electrical).schema_arrow.names)
    ac = indexed(electrical,["measured_on",*mapping.ac_power],filters=filters)
    flagged = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(eqc).measured_on))
    plant = complete_fleet_power(ac,flagged)
    poa = indexed(irradiance,filters=filters).iloc[:,0]
    flags = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(iqc).measured_on))
    local = poa.index.tz_localize("America/Los_Angeles",ambiguous="NaT",nonexistent="NaT")
    poa = poa.where(np.isfinite(poa)&poa.ge(0)&~poa.index.isin(flags)&~local.isna())
    x, paired, counts = {}, {}, {}
    for name in ("training","validation"):
        mask = split_masks(table.index)[name]
        x[name],paired[name],counts[name] = examples(table.loc[mask],plant,poa)
    return x, paired, counts, paths+[electrical,eqc,irradiance,iqc]


def metrics(actual,prediction,daylight):
    error = prediction-actual
    return dict(daylight_n=int(daylight.sum()), all_day_n=len(actual),
        daylight_mae_kw=mae(prediction[daylight],actual[daylight]),
        daylight_rmse_kw=rmse(prediction[daylight],actual[daylight]),
        daylight_bias_kw=float(error[daylight].mean()),all_day_mae_kw=mae(prediction,actual))


def main():
    features, tables, counts, paths = load_examples()
    paths += [Path(__file__),Path("scripts/forecast_stage2_weather.py"),Path("scripts/audit_stage2_readiness.py"),Path("scripts/benchmark_short_term_forecasting.py"),Path("scripts/calculate_stage2_physics.py")]
    specification = dict(protocol="classical-correction-fixed-v1",split=SPLIT,features=FEATURES,
        seed=SEED,cpu_threads=THREADS,models=MODELS,counts=counts,
        effective_settings={name:estimator(name,config).get_params() for name,config in MODELS.items()},
        source_hashes={str(p):digest(p) for p in paths},
        versions={p:metadata.version(p) for p in ("numpy","pandas","scikit-learn","scipy","xgboost","joblib","threadpoolctl","pvlib")},
        preprocessing="Training-only median and all-column missing indicators; training-only standardization for Ridge",
        sign="correction = actual - physics; final = physics + predicted correction; bias = prediction - actual; no postprocessing",
        uncertainty="2000 paired calendar-week date-block bootstrap draws, seed 2107; validation comparisons after selection, not confirmatory",
        availability="Current power/POA assumed available at forecast origin; exported weather latency assumption inherited; no future measured inputs")
    OUT.mkdir(exist_ok=True)
    settings_path = OUT/"settings.json"
    if settings_path.exists():
        if json.dumps(json.loads(settings_path.read_text())["specification"],sort_keys=True) != json.dumps(specification,sort_keys=True,default=str):
            raise ValueError("Incompatible completed correction results; refusing overwrite")
    else:
        settings_path.write_text(json.dumps(dict(specification=specification),indent=2,default=str))
    train,valid = tables["training"],tables["validation"]
    x,xv,scaler,processor = preprocess(features["training"],features["validation"],True)
    prediction = valid[["target_timestamp","actual_target_power_kw","physics_forecast_kw"]].copy()
    prediction["persistence_kw"] = features["validation"].origin_power_kw
    fits = {}
    print(json.dumps(counts,indent=2),flush=True)
    for family,config in MODELS.items():
        path = OUT/f"{family}.joblib"
        if path.exists():
            bundle = joblib.load(path)
            print(f"{family}: reusing completed fit",flush=True)
        else:
            model = estimator(family,config)
            print(f"{family}: fitting {len(train)} rows",flush=True)
            begin = time.monotonic()
            model.fit(scaler.transform(x) if family=="ridge" else x,train.correction_kw)
            elapsed = time.monotonic()-begin
            bundle = dict(model=model,processor=processor,scaler=scaler,features=FEATURES,fit_seconds=elapsed)
            joblib.dump(bundle,path)
            print(f"{family}: fit completed in {elapsed:.2f}s",flush=True)
        transformed = bundle["processor"].transform(features["validation"])
        if family=="ridge": transformed = bundle["scaler"].transform(transformed)
        correction = bundle["model"].predict(transformed)
        prediction[family] = valid.physics_forecast_kw+correction
        fits[family] = bundle["fit_seconds"]
    daylight = solar_elevation(pd.DatetimeIndex(valid.target_timestamp)).to_numpy()>0
    actual = valid.actual_target_power_kw.to_numpy()
    scores = {name:metrics(actual,prediction[column].to_numpy(),daylight) for name,column in
        [("physics","physics_forecast_kw"),("persistence","persistence_kw"),*[(n,n) for n in MODELS]]}
    for name,row in scores.items():
        row["fit_seconds"] = fits.get(name,0.)
        for reference in ("physics","persistence"):
            base = scores[reference]["daylight_mae_kw"]
            row[f"mae_improvement_over_{reference}_pct"] = 100*(base-row["daylight_mae_kw"])/base if base else np.nan
    winner = min(MODELS,key=lambda name:scores[name]["daylight_mae_kw"])
    uncertainty = {}
    for reference,column in (("physics","physics_forecast_kw"),("persistence","persistence_kw")):
        frame = pd.DataFrame({"actual_kw":actual[daylight],"persistence":prediction[column].to_numpy()[daylight],winner:prediction[winner].to_numpy()[daylight]},index=pd.DatetimeIndex(valid.target_timestamp[daylight]))
        intervals = paired_intervals(frame,winner)
        uncertainty[reference] = dict(mae_difference_kw=scores[winner]["daylight_mae_kw"]-scores[reference]["daylight_mae_kw"],
            lower_kw=-float(intervals["mae_saving_upper"]),upper_kw=-float(intervals["mae_saving_lower"]),blocks=intervals["blocks"])
    assert specification["source_hashes"]=={str(p):digest(p) for p in paths}
    prediction.to_parquet(OUT/"validation_predictions.parquet")
    result = pd.DataFrame.from_dict(scores,orient="index").rename_axis("model")
    result.to_csv(OUT/"metrics.csv")
    settings_path.write_text(json.dumps(dict(specification=specification,selected_model=winner,uncertainty=uncertainty),indent=2,default=str))
    print(result.to_string(),flush=True)
    print(json.dumps(dict(selected_model=winner,uncertainty=uncertainty),indent=2),flush=True)

# HPO search
import argparse
import hashlib
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils.validation import check_is_fitted

HPO_SPACES = {
    "forest": {"n_estimators":[200], "max_depth":[6,12,18], "min_samples_leaf":[5,10,20], "max_features":[.5,1.]},
    "histogram": {"learning_rate":[.03,.05,.1], "max_iter":[100,200,300], "max_leaf_nodes":[7,15,31], "min_samples_leaf":[20,50], "early_stopping":[False]},
    "xgboost": {"learning_rate":[.03,.05,.1], "n_estimators":[100,200,300], "max_depth":[3,5,7], "min_child_weight":[1,5,10]},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def search_candidates():
    candidates = []
    for family in MODELS:
        baseline = estimator(family,MODELS[family]).get_params()
        if family == "ridge":
            configs = [{"alpha":a} for a in [.001,.01,.1,1,10,100,1000]]
            base = {"alpha":10}
        else:
            base = {k:baseline[k] for k in HPO_SPACES[family]}
            if family == "xgboost":
                base["min_child_weight"] = 1  # Effective default of the original baseline.
            remaining = [c for c in ParameterGrid(HPO_SPACES[family]) if c!=base]
            # Singleton grids give uniform, without-replacement sampling excluding baseline.
            draws = ParameterSampler([{k:[v] for k,v in c.items()} for c in remaining],n_iter=11,random_state=SEED)
            configs = [base,*list(draws)]
        for i,config in enumerate(configs):
            model = estimator(family,MODELS[family]).set_params(**config)
            candidates.append(dict(id=f"{family}_{i:02d}",family=family,config=config,
                baseline=config==base,effective_settings=model.get_params()))
    return candidates


def freeze_search(path,plan):
    def normalized(value):
        value = value.copy()
        for key in ("source_hashes","baseline_hashes"):
            if key in value:
                hashes = {str(Path(name).resolve()):sha for name,sha in value[key].items()}
                if len(hashes)!=len(value[key]):
                    raise ValueError("Duplicate source paths in HPO plan")
                value[key] = hashes
        return value
    if path.exists():
        if canonical(normalized(json.loads(path.read_text())["plan"]))!=canonical(normalized(plan)):
            raise ValueError("Incompatible HPO cache/candidate plan; refusing reuse or overwrite")
    else:
        path.write_text(json.dumps(dict(plan=plan),indent=2,default=str))


def baseline_reuse(features,tables,counts):
    """Require the original baseline body, sources, settings and exact scoring keys."""
    reused, reasons = {}, {}
    try:
        saved = json.loads((OUT/"settings.json").read_text())["specification"]
        for key,value in dict(split=SPLIT,features=FEATURES,models=MODELS,counts=counts,seed=SEED,cpu_threads=THREADS).items():
            if canonical(saved[key])!=canonical(value):
                raise ValueError(f"baseline {key} mismatch")
        # Appending CLI/search code must not invalidate an unchanged original experiment.
        body = Path(__file__).read_text(encoding="utf-8-sig").split("# HPO search",1)[0].rstrip("\n")
        original = body+'\n\n\nif __name__=="__main__":\n    with threadpool_limits(limits=THREADS):\n        main()\n'
        original_hashes = {hashlib.sha256(text.encode()).hexdigest() for text in (original,original.replace('\n','\r\n'))}
        for name,expected in saved["source_hashes"].items():
            if Path(name).resolve()==Path(__file__).resolve():
                if expected not in original_hashes: raise ValueError("original baseline implementation mismatch")
            elif digest(Path(name))!=expected:
                raise ValueError(f"baseline source/code mismatch: {name}")
        for package,version in saved["versions"].items():
            if metadata.version(package)!=version: raise ValueError(f"baseline version mismatch: {package}")
        predictions = pd.read_parquet(OUT/"validation_predictions.parquet")
        valid = tables["validation"]
        if not predictions.index.equals(valid.index): raise ValueError("baseline scoring origins mismatch")
        pd.testing.assert_frame_equal(predictions[["target_timestamp","actual_target_power_kw","physics_forecast_kw"]],valid[["target_timestamp","actual_target_power_kw","physics_forecast_kw"]])
        np.testing.assert_array_equal(predictions.persistence_kw,features["validation"].origin_power_kw)
    except Exception as error:
        return {},{family:str(error) for family in MODELS},None
    for family in MODELS:
        try:
            bundle = joblib.load(OUT/f"{family}.joblib")
            if bundle["features"]!=FEATURES or canonical(bundle["model"].get_params())!=canonical(saved["effective_settings"][family]):
                raise ValueError("baseline effective settings/features mismatch")
            for obj in (bundle["model"],bundle["processor"],bundle["scaler"]): check_is_fitted(obj)
            if not np.isfinite(predictions[family]).all(): raise ValueError("incomplete baseline predictions")
            reused[family] = bundle
            reasons[family] = "compatible original body, data, preprocessing/settings and scoring timestamps"
        except Exception as error:
            reasons[family] = str(error)
    return reused,reasons,predictions


def save_trials(path,rows):
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(path)


def run_search(prepare=False):
    features,tables,counts,paths = load_examples()
    candidates = search_candidates()
    reused,reasons,baseline_predictions = baseline_reuse(features,tables,counts)
    valid = tables["validation"]
    daylight = solar_elevation(pd.DatetimeIndex(valid.target_timestamp)).to_numpy()>0
    fingerprints = {}
    for split in ("training","validation"):
        frame = features[split].join(tables[split][["target_timestamp","actual_target_power_kw","correction_kw"]])
        fingerprints[split] = hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()
    paths += [Path(__file__),Path("scripts/forecast_stage2_weather.py"),Path("scripts/audit_stage2_readiness.py"),Path("scripts/calculate_stage2_physics.py")]
    baseline_paths = [OUT/"settings.json",OUT/"validation_predictions.parquet",OUT/"metrics.csv",*[OUT/f"{n}.joblib" for n in MODELS]]
    plan = dict(protocol="classical-correction-hpo-v1",split=SPLIT,features=FEATURES,counts=counts,
        candidates=candidates,seed=SEED,cpu_threads=THREADS,frame_fingerprints=fingerprints,
        daylight_fingerprint=hashlib.sha256(daylight.tobytes()).hexdigest(),
        source_hashes={str(p):digest(p) for p in paths},
        baseline_hashes={str(p):digest(p) for p in baseline_paths if p.exists()},baseline_compatibility=reasons,
        versions={p:metadata.version(p) for p in ("numpy","pandas","scikit-learn","scipy","xgboost","joblib","threadpoolctl","pvlib")},
        selection="Lowest final-power daylight validation MAE; exact ties use fixed candidate order; no refit on validation",
        disclosure="Ranges and budget specified after inspecting baseline results; validation model-selection scores, not unbiased final estimates",
        preprocessing="Training-only median/all-column indicators; Ridge training-only standardization; unchanged sign/inputs/QC",
        sampling="Ridge grid; trees baseline plus 11 uniformly sampled discrete combinations without replacement; seed 2107")
    directory = OUT/"hpo"
    directory.mkdir(exist_ok=True)
    freeze_search(directory/"settings.json",plan)
    trial_path = directory/"trials.csv"
    rows = pd.read_csv(trial_path).to_dict("records") if trial_path.exists() else []
    by_id = {c["id"]:c for c in candidates}
    if len({r["id"] for r in rows})!=len(rows) or any(r["id"] not in by_id for r in rows):
        raise ValueError("Invalid HPO trial cache")
    # A winner is saved before its metrics; recover that completed fit after interruption.
    bundles = {}
    for family in MODELS:
        path = directory/f"best_{family}.joblib"
        if path.exists():
            bundle = joblib.load(path)
            if bundle["trial"]["id"] not in by_id: raise ValueError("Unknown cached winning trial")
            bundles[family] = bundle
            if not any(r["id"]==bundle["trial"]["id"] for r in rows): rows.append(bundle["trial"])
    for row in rows:
        candidate = by_id[row["id"]]
        if row["family"]!=candidate["family"] or bool(row["baseline"])!=candidate["baseline"] or canonical(json.loads(row["config"]))!=canonical(candidate["config"]):
            raise ValueError("Cached trial configuration mismatch")
        if row["status"] not in ("complete","failed"):
            raise ValueError("Invalid cached trial status")
    for family in MODELS:
        successful = [r for r in rows if r["family"]==family and r["status"]=="complete"]
        if successful:
            expected = min(successful,key=lambda r:(r["daylight_mae_kw"],list(by_id).index(r["id"])))
            if family not in bundles or bundles[family]["trial"]["id"]!=expected["id"]:
                raise ValueError("Completed winning trial lacks its compatible bundle")
        if family in bundles:
            bundle = bundles[family]
            candidate = by_id[bundle["trial"]["id"]]
            if bundle["features"]!=FEATURES or canonical(bundle["model"].get_params())!=canonical(candidate["effective_settings"]):
                raise ValueError("Cached winning model settings mismatch")
    completed = {r["id"] for r in rows}
    reusable = [c for c in candidates if c["baseline"] and c["family"] in reused and c["id"] not in completed]
    print("Frozen candidates:",{n:sum(c["family"]==n for c in candidates) for n in MODELS},flush=True)
    print("Baseline compatibility:",reasons,flush=True)
    print(f"Candidates {len(candidates)}; completed {len(completed)}; compatible baseline reuse {len(reusable)}; new fits needed {len(candidates)-len(completed)-len(reusable)}",flush=True)
    print("HPO outputs:",directory,"; original comparison preserved at",OUT,flush=True)
    if prepare:
        print("Preparation complete: no estimator fitting or prediction generation. Candidate plan saved.")
        return
    if rows:
        save_trials(trial_path,rows)
    if reused:
        first = next(iter(reused.values()))
        processor,scaler = first["processor"],first["scaler"]
        x,xv = processor.transform(features["training"]),processor.transform(features["validation"])
    else:
        x,xv,scaler,processor = preprocess(features["training"],features["validation"],True)
    actual = valid.actual_target_power_kw.to_numpy()
    for candidate in candidates:
        if candidate["id"] in completed: continue
        family = candidate["family"]
        row = dict(id=candidate["id"],family=family,config=canonical(candidate["config"]),baseline=candidate["baseline"])
        begin = time.monotonic()
        print(f"{candidate['id']}: starting",flush=True)
        try:
            if candidate["baseline"] and family in reused:
                bundle = reused[family].copy()
                bundle["model"].set_params(**candidate["config"])
                prediction = baseline_predictions[family].to_numpy()
                row.update(source="baseline_reuse",fit_seconds=bundle["fit_seconds"])
            else:
                model = estimator(family,MODELS[family]).set_params(**candidate["config"])
                model.fit(scaler.transform(x) if family=="ridge" else x,tables["training"].correction_kw)
                elapsed = time.monotonic()-begin
                prediction = valid.physics_forecast_kw.to_numpy()+model.predict(scaler.transform(xv) if family=="ridge" else xv)
                bundle = dict(model=model,processor=processor,scaler=scaler,features=FEATURES)
                row.update(source="new_fit",fit_seconds=elapsed)
            if not np.isfinite(prediction).all(): raise ValueError("Nonfinite validation predictions")
            row.update(metrics(actual,prediction,daylight),status="complete",error="")
        except Exception as error:
            row.update(status="failed",error=f"{type(error).__name__}: {error}",fit_seconds=time.monotonic()-begin)
            print(f"{candidate['id']}: FAILED {row['error']}",flush=True)
        if row["status"]=="complete":
            previous = bundles.get(family)
            rank = list(by_id).index(row["id"])
            if previous is None or (row["daylight_mae_kw"],rank)<(previous["trial"]["daylight_mae_kw"],list(by_id).index(previous["trial"]["id"])):
                bundle["trial"] = row.copy()
                path = directory/f"best_{family}.joblib"
                temporary = path.with_suffix(".tmp")
                joblib.dump(bundle,temporary)
                temporary.replace(path)
                bundles[family] = bundle
        row["wall_seconds"] = time.monotonic()-begin
        rows.append(row)
        save_trials(trial_path,rows)
        print(f"{candidate['id']}: {row['status']}; {row['fit_seconds']:.2f}s",flush=True)
    prediction = valid[["target_timestamp","actual_target_power_kw","physics_forecast_kw"]].copy()
    prediction["persistence_kw"] = features["validation"].origin_power_kw
    best = {}
    for family,bundle in bundles.items():
        transformed = bundle["processor"].transform(features["validation"])
        if family=="ridge": transformed = bundle["scaler"].transform(transformed)
        prediction[family] = valid.physics_forecast_kw.to_numpy()+bundle["model"].predict(transformed)
        baseline = next((r for r in rows if r["family"]==family and r["baseline"] and r["status"]=="complete"),None)
        best[family] = dict(trial=bundle["trial"]["id"],config=by_id[bundle["trial"]["id"]]["config"],
            baseline_daylight_mae_kw=baseline["daylight_mae_kw"] if baseline else None,
            tuned_daylight_mae_kw=bundle["trial"]["daylight_mae_kw"])
    successful = [r for r in rows if r["status"]=="complete"]
    selected = min(successful,key=lambda r:(r["daylight_mae_kw"],list(by_id).index(r["id"]))) if successful else None
    assert plan["source_hashes"]=={str(p):digest(p) for p in paths}
    prediction.to_parquet(directory/"best_validation_predictions.parquet")
    (directory/"settings.json").write_text(json.dumps(dict(plan=plan,best_per_family=best,selected_trial=selected["id"] if selected else None,
        failed_trials=[r["id"] for r in rows if r["status"]=="failed"]),indent=2,default=str))
    print(json.dumps(dict(best_per_family=best,selected_trial=selected["id"] if selected else None),indent=2),flush=True)
if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo",action="store_true",help="Execute/resume the frozen correction search")
    parser.add_argument("--prepare",action="store_true",help="Freeze/check HPO plan without fitting")
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if args.hpo or args.prepare:
            run_search(prepare=args.prepare)
        else:
            main()
