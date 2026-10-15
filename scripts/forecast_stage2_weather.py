"""Fixed 2018 weather selection, followed by 2019ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ2021 historical forecasts."""
from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import platform
import time

import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_union
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from audit_stage2_readiness import indexed, digest, solar_elevation
from benchmark_short_term_forecasting import available_environment
from oedi2107.validation import mae, rmse

OUT = Path("outputs/2107_stage2_weather")
DATA = Path("data/2107/processed")
SEED, THREADS = 2107, 4
VARIABLES = {"poa": "W/m2", "ambient": "C", "wind": "m/s"}
# Half-open target intervals; ceil elapsed-time boundaries to the five-minute grid.
SPLIT = {
    "start": "2017-12-01 00:00:00", "validation": "2022-01-21 16:45:00",
    "test": "2022-12-11 20:20:00", "end": "2023-10-31 23:50:00",
}


def split_masks(origins):
    edges = list(SPLIT.values())
    return {name: period_mask(origins, start, end)
            for name, start, end in zip(("training", "validation", "testing"), edges, edges[1:])}


def training_blocks():
    start, end = pd.Timestamp(SPLIT["start"]), pd.Timestamp(SPLIT["validation"])
    edges = [start+(end-start)*fraction for fraction in (.4, .6, .8)]
    edges = [t.ceil("5min") for t in edges]+[end]
    return [(f"training_{i}", a, b) for i, (a,b) in enumerate(zip(edges, edges[1:]), 1)]


SEARCH = {
    "ridge": [{"alpha": a} for a in (.1, 10., 1000.)],
    "forest": [{"max_depth": d, "min_samples_leaf": leaf} for d,leaf in [(12,5), (20,5)]],
    "histogram": [{"max_leaf_nodes": n} for n in (7,15,31)],
    "xgboost": [{"max_depth": d} for d in (3,5,7)],
}


def weather_features(poa, environment, origins):
    """25 exact POA lags and 25 availability-limited lags for each environment sensor."""
    columns = {}
    for lag in range(0, 121, 5):
        times = origins-pd.Timedelta(minutes=lag)
        columns[f"poa_lag{lag}"] = poa.reindex(times).to_numpy()
        env = available_environment(environment, times)
        columns[f"ambient_lag{lag}"] = env.ambient_temperature_c.to_numpy()
        columns[f"wind_lag{lag}"] = env.wind_speed_m_s.to_numpy()
    return pd.DataFrame(columns, index=origins)


def recorded_targets(poa, environment, origins):
    """Exact recorded t+15 labels; never reuse the backward input lookup for labels."""
    times = origins+pd.Timedelta(minutes=15)
    return pd.DataFrame({"poa": poa.reindex(times).to_numpy(),
        "ambient": environment.ambient_temperature_c.reindex(times).to_numpy(),
        "wind": environment.wind_speed_m_s.reindex(times).to_numpy()}, index=origins)


def period_mask(index, start, end):
    """Allocate by target only; legitimate historical inputs can precede the block."""
    targets = index+pd.Timedelta(minutes=15)
    return (targets >= pd.Timestamp(start)) & (targets < pd.Timestamp(end))


def constrained(prediction, variable, elevation):
    prediction = np.asarray(prediction, dtype=float).copy()
    if variable in ("poa", "wind"):
        prediction = np.maximum(0, prediction)
    if variable == "poa":
        prediction[np.asarray(elevation) <= 0] = 0
    return prediction


def estimator(family, config):
    if family == "ridge":
        return Ridge(**config, random_state=SEED)
    if family == "forest":
        return RandomForestRegressor(**config, n_estimators=200, max_features=1.,
                                     random_state=SEED, n_jobs=THREADS)
    if family == "histogram":
        return HistGradientBoostingRegressor(**config, learning_rate=.05, max_iter=200,
            min_samples_leaf=20, early_stopping=False, random_state=SEED)
    return XGBRegressor(**config, objective="reg:squarederror", tree_method="hist",
        n_estimators=200, learning_rate=.05, subsample=1., colsample_bytree=1.,
        random_state=SEED, n_jobs=THREADS)


def preprocess(train, predict, return_processor=False):
    # Indicators for every column also cover gaps first encountered after fitting.
    processor = make_union(SimpleImputer(strategy="median", keep_empty_features=True),
                           MissingIndicator(features="all"))
    fitted = processor.fit_transform(train)
    transformed = processor.transform(predict)
    scaler = StandardScaler().fit(fitted)
    return (fitted, transformed, scaler, processor) if return_processor else (fitted, transformed, scaler)


def score(actual, prediction):
    return dict(n=len(actual), mae=mae(prediction, actual), rmse=rmse(prediction, actual))


def load_data():
    paths = [DATA/"2107_irradiance_clean.parquet", DATA/"2107_environment_clean.parquet",
             DATA/"2107_irradiance_qc_events.parquet"]
    poa = indexed(paths[0]).iloc[:,0]
    env = indexed(paths[1], ["measured_on", "ambient_temperature_c", "wind_speed_m_s"])
    env = env.copy()
    flags = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(paths[2]).measured_on))
    local = poa.index.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")
    poa = poa.where(np.isfinite(poa) & poa.ge(0) & ~poa.index.isin(flags) & ~local.isna())
    env.ambient_temperature_c = env.ambient_temperature_c.where(np.isfinite(env.ambient_temperature_c))
    env.wind_speed_m_s = env.wind_speed_m_s.where(np.isfinite(env.wind_speed_m_s) & env.wind_speed_m_s.ge(0))
    origins = poa.index[period_mask(poa.index, SPLIT["start"], SPLIT["end"])]
    elevation = solar_elevation(origins+pd.Timedelta(minutes=15)).set_axis(origins)
    features = weather_features(poa, env, origins)
    eligible = features[[f"{v}_lag0" for v in VARIABLES]].notna().all(axis=1) & elevation.notna()
    features = features.loc[eligible]
    targets = recorded_targets(poa, env, features.index)
    return features, targets, elevation.loc[eligible], paths


def select_models(features, targets, elevation, folds, destination=None):
    rows = []
    if destination is not None and destination.exists():
        cached = pd.read_csv(destination)
        rows = cached.loc[cached.fold.astype(str).ne("pooled")].to_dict("records")

    def save(row):
        rows.append(row)
        if destination is not None:
            pd.DataFrame(rows).to_csv(destination, index=False)

    for variable in VARIABLES:
        for fold, (start, cutoff, end) in enumerate(folds, 1):
            train = period_mask(features.index, start, cutoff) & targets[variable].notna()
            valid = period_mask(features.index, cutoff, end) & targets[variable].notna()
            if variable == "poa":
                valid &= elevation.gt(0)
            x, xv, scaler = preprocess(features.loc[train], features.loc[valid])
            xr, xvr = scaler.transform(x), scaler.transform(xv)
            y, actual = targets.loc[train,variable], targets.loc[valid,variable]
            e = elevation.loc[valid]
            base = constrained(features.loc[valid,f"{variable}_lag0"], variable, e)
            if not any(r["variable"]==variable and r["family"]=="persistence" and str(r["fold"])==str(fold) for r in rows):
                save(dict(variable=variable, unit=VARIABLES[variable], fold=fold,
                    family="persistence", config="{}", train_n=int(train.sum()), **score(actual,base)))
            for family, configs in SEARCH.items():
                for config in configs:
                    if any(r["variable"]==variable and r["family"]==family and r["config"]==json.dumps(config,sort_keys=True) and str(r["fold"])==str(fold) for r in rows):
                        continue
                    begin = time.monotonic()
                    model = estimator(family, config)
                    model.fit(xr if family=="ridge" else x, y)
                    prediction = constrained(model.predict(xvr if family=="ridge" else xv), variable, e)
                    save(dict(variable=variable, unit=VARIABLES[variable], fold=fold,
                        family=family, config=json.dumps(config,sort_keys=True), train_n=int(train.sum()),
                        **score(actual,prediction)))
                    print(f"{variable} fold {fold}: {family} {config}; n={len(actual)}; {time.monotonic()-begin:.1f}s", flush=True)
    result = pd.DataFrame(rows)
    pooled = []
    for (variable,family,config), group in result.groupby(["variable","family","config"], sort=False):
        n = int(group.n.sum())
        pooled.append(dict(variable=variable, unit=VARIABLES[variable], fold="pooled", family=family,
            config=config, train_n=np.nan, n=n, mae=np.average(group.mae,weights=group.n),
            rmse=np.sqrt(np.average(group.rmse**2,weights=group.n))))
    result = pd.concat([result,pd.DataFrame(pooled)],ignore_index=True)
    winners = {}
    for variable in VARIABLES:
        candidates = result.loc[result.variable.eq(variable) & result.fold.eq("pooled") & ~result.family.eq("persistence")]
        best = candidates.loc[candidates.mae.idxmin()]
        winners[variable] = dict(family=best.family, config=json.loads(best.config))
    return result, winners


def forecast_block(features, targets, elevation, winners, cutoff, end, models=None):
    """Fit preceding labels only, or reuse complete fitted models for held-out inference."""
    issue = period_mask(features.index, cutoff, end)
    part = targets.loc[issue].rename(columns=lambda v:f"actual_{v}").copy()
    part["target_timestamp"] = part.index+pd.Timedelta(minutes=15)
    part["training_cutoff"] = pd.Timestamp(cutoff)
    fitted = {} if models is None else models.copy()
    for variable, choice in winners.items():
        ridge = choice["family"] == "ridge"
        if variable not in fitted:
            train = period_mask(features.index, SPLIT["start"], cutoff) & targets[variable].notna()
            x, xp, scaler, processor = preprocess(features.loc[train], features.loc[issue], True)
            model = estimator(choice["family"], choice["config"])
            model.fit(scaler.transform(x) if ridge else x, targets.loc[train, variable])
            fitted[variable] = (processor, scaler, model)
        else:
            processor, scaler, model = fitted[variable]
            xp = processor.transform(features.loc[issue])
        part[f"forecast_{variable}"] = constrained(model.predict(scaler.transform(xp) if ridge else xp), variable, elevation.loc[issue])
    return part.rename_axis("origin"), fitted


def prepare(features, targets, elevation):
    from calculate_stage2_physics import load_actual
    actual, _ = load_actual()
    power = actual.reindex(features.index+pd.Timedelta(minutes=15)).to_numpy()
    rows = []
    for name, mask in split_masks(features.index).items():
        row = dict(split=name, origins=int(mask.sum()), power_labels=int(np.isfinite(power[mask]).sum()))
        row.update({f"{v}_labels":int(targets.loc[mask,v].notna().sum()) for v in VARIABLES})
        row["poa_daylight_labels"] = int((targets.poa.notna() & elevation.gt(0) & mask).sum())
        rows.append(row)
    start, validation, test, end = [pd.Timestamp(t) for t in SPLIT.values()]
    assert validation == (start+(end-start)*.7).ceil("5min")
    assert test == (start+(end-start)*.85).ceil("5min")
    assert (sum(split_masks(features.index).values()) == 1).all()
    print(json.dumps(SPLIT, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print("Naive Pacific wall-clock; last included target 2023-10-31 23:45; upward five-minute rounding; internal gaps unchanged.")
    print("Training blocks:", [(n,str(a),str(b)) for n,a,b in training_blocks()])
    initial_end = training_blocks()[0][1]
    tune = (start+(initial_end-start)*.8).ceil("5min")
    print("One initial selection: fit targets before", tune, "; select targets before", initial_end, "; configurations then frozen.")
    print("Planned: initial-period selection 33 fits; 3 training blocks + validation x 3 refits; total 45 weather fits. Test inference: 0 fits.")
    print("Expensive retained configurations: 200-tree forests at depth 12 or 20; unlimited depth removed. No fits or outputs in preparation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--allow-test", action="store_true", help="Confirm all pipeline choices are frozen before test inference")
    parser.add_argument("--step", choices=["training_1", "training_2", "training_3", "validation", "testing"])
    args = parser.parse_args()
    if args.step == "testing" and not args.prepare and not args.allow_test:
        parser.error("Testing requires --allow-test after all pipeline choices are frozen")
    features, targets, elevation, paths = load_data()
    if args.prepare or args.step is None:
        prepare(features, targets, elevation)
        return
    OUT.mkdir(exist_ok=True)
    step = args.step
    blocks = {n:(a,b) for n,a,b in training_blocks()}
    blocks.update(validation=(pd.Timestamp(SPLIT["validation"]), pd.Timestamp(SPLIT["test"])),
                  testing=(pd.Timestamp(SPLIT["test"]), pd.Timestamp(SPLIT["end"])))
    cutoff, end = blocks[step]
    settings_path = OUT/f"{step}_settings.json"
    versions = {p:metadata.version(p) for p in ["numpy","pandas","scikit-learn","scipy","pvlib","pyarrow","xgboost","joblib","threadpoolctl"]}
    specification = dict(protocol="elapsed-70-15-15-v1", split=SPLIT, step=step,
        cutoff=str(cutoff), end=str(end), seed=SEED, cpu_threads=THREADS, search=SEARCH,
        horizon_minutes=15, lag_minutes=list(range(0,121,5)), versions=versions,
        python_version=platform.python_version(), units=VARIABLES,
        source_hashes={str(p):digest(p) for p in paths}, code_hash=digest(Path(__file__)),
        tuning="Select once in initial training period: first 80% fit / last 20% select; freeze configurations for all expanding refits and outer validation/test",
        availability="Environment latest label in [feature_time-30, feature_time-15]; unverified latency",
        preprocessing="Training-only median/all-column missing indicators; Ridge training-only scaling",
        scoring="POA target elevation>0; environment all valid exact labels; persistence reference only",
        rules="Nonnegative POA/wind; POA zero at target elevation<=0; ambient unclamped")
    initial_path = OUT/"training_1_settings.json"
    if step != "training_1":
        if not initial_path.exists():
            raise ValueError("Run training_1 selection first; subsequent steps require its frozen configurations")
        initial = json.loads(initial_path.read_text())
        for key in ("protocol", "split", "source_hashes", "versions", "seed", "search", "code_hash", "tuning"):
            if initial["specification"][key] != json.loads(json.dumps(specification[key])):
                raise ValueError("Incompatible initial weather selection")
        if "winners" not in initial:
            raise ValueError("Initial weather selection is incomplete")
        specification["frozen_winners"] = initial["winners"]
    if settings_path.exists():
        saved = json.loads(settings_path.read_text())
        if saved["specification"] != json.loads(json.dumps(specification)):
            raise ValueError("Incompatible saved step settings; refusing reuse")
    else:
        settings_path.write_text(json.dumps(dict(specification=specification), indent=2))
        saved = {}
    destination = OUT/f"forecasts_{step}.parquet"
    if destination.exists():
        print("Compatible completed forecasts reused:", destination)
        return
    if step == "testing":
        selected = json.loads((OUT/"validation_settings.json").read_text())
        expected = selected["specification"]
        for key in ("protocol", "split", "source_hashes", "versions", "seed", "search", "code_hash", "frozen_winners"):
            if expected[key] != json.loads(json.dumps(specification[key])):
                raise ValueError("Incompatible validation models")
        winners = selected["winners"]
        models = joblib.load(OUT/"validation_models.joblib")
        part, _ = forecast_block(features, targets, elevation, winners, SPLIT["validation"], end, models)
        part = part.loc[period_mask(part.index, cutoff, end)]
    else:
        if "winners" in saved:
            winners = saved["winners"]
        elif step == "training_1":
            tune = (pd.Timestamp(SPLIT["start"])+(cutoff-pd.Timestamp(SPLIT["start"]))*.8).ceil("5min")
            folds = [(SPLIT["start"], tune, cutoff)]
            scores, winners = select_models(features, targets, elevation, folds, OUT/"training_1_selection.csv")
            scores.to_csv(OUT/"training_1_selection.csv", index=False)
        else:
            winners = specification["frozen_winners"]
        saved = dict(specification=specification, winners=winners,
            effective_estimator_settings={v:estimator(c["family"],c["config"]).get_params() for v,c in winners.items()})
        settings_path.write_text(json.dumps(saved, indent=2, default=str))
        model_path = OUT/f"{step}_models.joblib"
        models = joblib.load(model_path) if model_path.exists() else {}
        # Save each expensive completed refit before moving to the next variable.
        parts = []
        for variable in VARIABLES:
            part, fitted = forecast_block(features, targets, elevation, {variable:winners[variable]}, cutoff, end, models)
            models.update(fitted)
            joblib.dump(models, model_path)
            parts.append(part[[f"forecast_{variable}"]])
        part = part.drop(columns=[c for c in part if c.startswith("forecast_")]).join(pd.concat(parts, axis=1))
    assert specification["source_hashes"] == {str(p):digest(p) for p in paths}
    part["split"] = "testing" if step == "testing" else "validation" if step == "validation" else "training"
    part.to_parquet(destination)
    print("Saved", destination, len(part))


if __name__=="__main__":
    with threadpool_limits(limits=THREADS):
        main()
