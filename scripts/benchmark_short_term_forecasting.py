"""Frozen, origin-only complete-fleet AC forecasting benchmark."""
from pathlib import Path
import importlib.metadata as metadata
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from audit_stage2_readiness import digest, indexed, solar_elevation

OUT = Path("outputs/2107_short_term_forecasting")
DATA = Path("data/2107/processed")
CONFIG = {
    "horizons_minutes": [5, 15, 30], "lookback_minutes": 30,
    "split": {"training": [2018, 2019, 2020], "validation": [2021], "evaluation": [2022, 2023]},
    "daytime": "origin and target deterministic solar elevation >10 degrees; no future POA filter",
    "features": "AC and POA at t,t-5,t-15,t-30; AC/POA changes over 5/15/30 min; origin zero-unit fraction; historical ambient/wind; cyclic hour/year phase; origin and target solar elevation",
    "weather_availability": "latest label <=t-15 min, age <=30 min; assumed available 15 min after its label",
    "history": "all seven five-minute AC/POA readings in [t-30,t] valid, no interpolation",
    "models": {"ridge_alpha": 100., "boosting": {"max_iter": 150, "max_leaf_nodes": 15,
        "max_depth": 4, "min_samples_leaf": 100, "l2_regularization": 10,
        "learning_rate": .06, "random_state": 2107}},
    "learning_target": "future strict fleet kW minus origin fleet kW; predictions bounded below by zero",
    "rapid_change": "abs(target AC-origin AC)>=100 kW; evaluation slice only",
    "uncertainty": "2000 paired bootstrap draws of whole calendar-week date blocks; seed 2107",
    "selection": "configuration fixed before fit; validation reported without tuning or retraining",
}


def split_labels(index):
    years = pd.DatetimeIndex(index).year
    return np.select([np.isin(years, [2018, 2019, 2020]), years == 2021,
                      np.isin(years, [2022, 2023])], [1, 2, 3], default=0)


def window_mask(index, valid, horizon):
    """Exact history and target in one split; no row-shift across missing labels."""
    labels = split_labels(index)
    result = np.asarray(valid, dtype=bool).copy() & (labels != 0)
    validity = pd.Series(np.asarray(valid, dtype=bool), index=index)
    for lag in range(0, 31, 5):
        times = index-pd.Timedelta(minutes=lag)
        result &= validity.reindex(times, fill_value=False).to_numpy()
        result &= split_labels(times) == labels
    result &= split_labels(index+pd.Timedelta(minutes=horizon)) == labels
    return result


def available_environment(environment, index):
    """Latest label 15–30 minutes before each requested feature timestamp."""
    return environment.reindex(index-pd.Timedelta(minutes=15), method="ffill",
                               tolerance=pd.Timedelta(minutes=15))


def origin_features(plant, poa, zero_fraction, environment, elevation):
    index = plant.index
    f = pd.DataFrame(index=index)
    for label, s in [("ac_kw", plant), ("poa", poa)]:
        for lag in (0, 5, 15, 30):
            f[f"{label}_lag{lag}"] = s.reindex(index-pd.Timedelta(minutes=lag)).to_numpy()
        for lag in (5, 15, 30):
            f[f"{label}_change{lag}"] = f[f"{label}_lag0"]-f[f"{label}_lag{lag}"]
    # Backward lookup only; the 15-minute allowance avoids a forward weather join.
    weather = available_environment(environment, index)
    f["ambient_c"] = weather.ambient_temperature_c.to_numpy()
    f["wind_m_s"] = weather.wind_speed_m_s.to_numpy()
    f["zero_unit_fraction"] = zero_fraction
    hour = 2*np.pi*(index.hour+index.minute/60)/24
    year_phase = 2*np.pi*(index.dayofyear-1)/365.25
    f["hour_sin"], f["hour_cos"] = np.sin(hour), np.cos(hour)
    f["season_sin"], f["season_cos"] = np.sin(year_phase), np.cos(year_phase)
    f["origin_elevation"] = elevation
    return f


def metrics(frame, model):
    error = frame[model]-frame.actual_kw
    persistence = frame.persistence-frame.actual_kw
    mae, rmse = float(abs(error).mean()), float(np.sqrt((error**2).mean()))
    pmae, prmse = float(abs(persistence).mean()), float(np.sqrt((persistence**2).mean()))
    return dict(samples=len(frame), dates=frame.index.normalize().nunique(),
        mae_kw=mae, rmse_kw=rmse, bias_kw=float(error.mean()),
        mae_improvement_pct=100*(pmae-mae)/pmae if pmae else np.nan,
        rmse_improvement_pct=100*(prmse-rmse)/prmse if prmse else np.nan)


def paired_intervals(frame, model):
    """Resample identical date blocks for model and persistence."""
    if not len(frame):
        return {}
    errors = frame[model]-frame.actual_kw
    persistence = frame.persistence-frame.actual_kw
    d = pd.DataFrame({"n": 1, "mae": abs(errors), "pmae": abs(persistence),
                      "mse": errors**2, "pmse": persistence**2}, index=frame.index)
    blocks = d.groupby(d.index.to_period("W-SUN")).sum().to_numpy()
    rng = np.random.default_rng(2107)
    totals = blocks[rng.integers(0, len(blocks), size=(2000, len(blocks)))].sum(axis=1)
    n, a, p, s, ps = totals.T
    difference = (p-a)/n
    relative, rmse_relative = np.full(len(p), np.nan), np.full(len(p), np.nan)
    nonzero = p > 0
    relative[nonzero] = 100*(p[nonzero]-a[nonzero])/p[nonzero]
    rmse_relative[nonzero] = 100*(np.sqrt(ps[nonzero]/n[nonzero])-
        np.sqrt(s[nonzero]/n[nonzero]))/np.sqrt(ps[nonzero]/n[nonzero])
    def bounds(values):
        finite = values[np.isfinite(values)]
        return np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    relative_low, relative_high = bounds(relative)
    rmse_low, rmse_high = bounds(rmse_relative)
    return dict(blocks=len(blocks), mae_saving_lower=float(np.quantile(difference, .025)),
        mae_saving_upper=float(np.quantile(difference, .975)),
        mae_improvement_lower=float(relative_low), mae_improvement_upper=float(relative_high),
        rmse_improvement_lower=float(rmse_low), rmse_improvement_upper=float(rmse_high),
        undefined_relative_bootstrap_fraction=float((~nonzero).mean()))


def score_slices(predictions):
    rows = []
    for (split, horizon), group in predictions.groupby(["split", "horizon_minutes"]):
        slices = {"all_daytime": group, "rapid_change": group.loc[group.rapid_change],
                  "target_recorded_zero": group.loc[group.actual_kw.eq(0)],
                  "target_generating": group.loc[group.actual_kw.gt(0)],
                  "origin_recorded_zero": group.loc[group.origin_kw.eq(0)],
                  "origin_generating": group.loc[group.origin_kw.gt(0)]}
        slices.update({"season_"+k: g for k,g in group.groupby("season")})
        slices.update({"year_"+str(k): g for k,g in group.groupby(group.index.year)})
        for slice_name, frame in slices.items():
            if not len(frame):
                continue
            for model in ("persistence", "ridge", "boosting"):
                row = dict(split=split, horizon_minutes=horizon, slice=slice_name, model=model,
                           **metrics(frame, model))
                if split == "evaluation" and model != "persistence":
                    row.update(paired_intervals(frame, model))
                rows.append(row)
    return pd.DataFrame(rows)


def pairwise_comparisons(predictions):
    rows = []
    evaluation = predictions.loc[predictions.split.eq("evaluation")]
    for horizon, frame in evaluation.groupby("horizon_minutes"):
        for model, reference in [("ridge", "persistence"), ("boosting", "persistence"),
                                 ("boosting", "ridge")]:
            paired = frame.copy()
            paired["persistence"] = frame[reference]
            rows.append(dict(horizon_minutes=horizon, model=model, reference=reference,
                             **metrics(paired, model), **paired_intervals(paired, model)))
    return pd.DataFrame(rows)


def plots(scores, predictions):
    primary = scores.loc[scores.split.eq("evaluation") & scores.slice.eq("all_daytime")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model, group in primary.groupby("model"):
        axes[0].plot(group.horizon_minutes, group.mae_kw, marker="o", label=model)
        axes[1].plot(group.horizon_minutes, group.rmse_kw, marker="o", label=model)
    for ax, label in zip(axes, ["MAE (kW)", "RMSE (kW)"]):
        ax.set_ylabel(label); ax.set_xlabel("Minutes ahead"); ax.set_xticks([5,15,30]); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("2022–2023 daytime complete-fleet forecasts")
    fig.tight_layout(); fig.savefig(OUT/"daytime_errors.png", dpi=160); plt.close(fig)
    selected = scores.loc[scores.split.eq("evaluation") & scores.model.eq("boosting") &
        scores.slice.isin(["all_daytime", "rapid_change", "season_DJF", "season_MAM", "season_JJA",
                          "season_SON", "target_recorded_zero", "target_generating"])]
    names = selected.slice.drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for h, g in selected.groupby("horizon_minutes"):
        positions = np.array([names.index(s) for s in g.slice])+(h-15)/100
        ax.errorbar(positions, g.mae_improvement_pct,
            yerr=np.vstack([g.mae_improvement_pct-g.mae_improvement_lower,
                            g.mae_improvement_upper-g.mae_improvement_pct]),
            fmt="o", capsize=3, label=f"{h} min")
    ax.axhline(0, color="grey", lw=.8); ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_ylabel("MAE improvement over persistence (%)"); ax.legend(); ax.grid(alpha=.2)
    ax.set_title("Boosting comparisons: paired calendar-week 95% intervals")
    fig.tight_layout(); fig.savefig(OUT/"slice_improvements.png", dpi=160); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # This is written before any model sees training or validation targets.
    (OUT/"benchmark_specification.json").write_text(json.dumps(CONFIG, indent=2))
    source = [DATA/f"2107_{s}_clean.parquet" for s in ("electrical", "irradiance", "environment")]
    source += [DATA/f"2107_{s}_qc_events.parquet" for s in ("electrical", "irradiance")]
    protected = source + list(Path("outputs/2107_research_baseline").glob("*"))
    protected += list(Path("outputs/2107_performance_change").glob("*"))
    protected += [Path("docs/performance_change_feasibility.md")]
    before = {str(p): digest(p) for p in protected if p.is_file()}
    names = pq.ParquetFile(source[0]).schema_arrow.names
    ac = indexed(source[0], ["measured_on", *[c for c in names if "_ac_power_" in c]])
    assert ac.shape[1] == 24
    plant = ac.sum(axis=1, min_count=24)
    poa = indexed(source[1]).iloc[:,0].reindex(ac.index)
    env = indexed(source[2])[["ambient_temperature_c", "wind_speed_m_s"]].sort_index()
    eflags = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(source[3]).measured_on))
    pflags = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(source[4]).measured_on))
    electric_valid = ac.notna().all(axis=1) & ac.ge(0).all(axis=1) & ~ac.index.isin(eflags)
    history_valid = electric_valid & poa.notna() & poa.ge(0) & ~ac.index.isin(pflags)
    elevation = solar_elevation(ac.index)
    features = origin_features(plant, poa, ac.eq(0).mean(axis=1), env, elevation)
    print("Origin features prepared", flush=True)
    fitted, predictions, support = {}, [], []
    for horizon in CONFIG["horizons_minutes"]:
        target_times = ac.index+pd.Timedelta(minutes=horizon)
        target = plant.reindex(target_times).to_numpy()
        target_elevation = elevation.reindex(target_times).to_numpy()
        x = features.assign(target_elevation=target_elevation,
                            elevation_change=target_elevation-elevation.to_numpy())
        valid = window_mask(ac.index, history_valid, horizon)
        valid &= electric_valid.reindex(target_times, fill_value=False).to_numpy()
        valid &= np.isfinite(target) & x.notna().all(axis=1).to_numpy()
        valid &= elevation.gt(10).to_numpy() & (target_elevation > 10)
        x = x.loc[valid]
        y = target[valid]
        current = plant.loc[valid].to_numpy()
        labels = split_labels(x.index)
        train = labels == 1
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=CONFIG["models"]["ridge_alpha"]))
        boosting = HistGradientBoostingRegressor(**CONFIG["models"]["boosting"])
        ridge.fit(x.loc[train], y[train]-current[train])
        boosting.fit(x.loc[train], y[train]-current[train])
        fitted[horizon] = dict(ridge=ridge, boosting=boosting, features=x.columns.tolist())
        for label, split in [(1,"training"), (2,"validation"), (3,"evaluation")]:
            use = labels == label
            support.append(dict(horizon_minutes=horizon, split=split, samples=int(use.sum()),
                dates=int(x.index[use].normalize().nunique()), start=str(x.index[use].min()),
                end=str(x.index[use].max())))
            if split == "training":
                continue
            part = pd.DataFrame({"actual_kw": y[use], "origin_kw": current[use],
                "persistence": current[use], "ridge": np.maximum(0, current[use]+ridge.predict(x.loc[use])),
                "boosting": np.maximum(0,current[use]+boosting.predict(x.loc[use])),
                "target_timestamp": target_times[valid][use]}, index=x.index[use])
            part["split"], part["horizon_minutes"] = split, horizon
            part["rapid_change"] = abs(part.actual_kw-part.origin_kw) >= 100
            part["season"] = np.array(["DJF","MAM","JJA","SON"])[(part.index.month % 12)//3]
            predictions.append(part)
        print(f"{horizon}-minute models fitted once", flush=True)
    predictions = pd.concat(predictions).sort_index()
    predictions.index.name = "forecast_origin"
    predictions.to_parquet(OUT/"predictions.parquet")
    joblib.dump(fitted, OUT/"forecast_models.joblib")
    scores = score_slices(predictions)
    scores.to_csv(OUT/"metrics.csv", index=False)
    pairwise_comparisons(predictions).to_csv(OUT/"pairwise_comparisons.csv", index=False)
    pd.DataFrame(support).to_csv(OUT/"support.csv", index=False)
    plots(scores, predictions)
    after = {p: digest(Path(p)) for p in before}
    assert before == after, "Protected source/baseline/decline findings changed"
    import sklearn
    provenance = dict(protected_hashes=before, protected_unchanged=True,
        specification_sha256=digest(OUT/"benchmark_specification.json"),
        code_sha256=digest(Path(__file__)), sklearn_version=sklearn.__version__,
        pandas_version=pd.__version__, numpy_version=np.__version__,
        package_versions={name: metadata.version(name) for name in
            ["numpy", "pandas", "scipy", "scikit-learn", "pvlib", "pyarrow", "matplotlib", "joblib", "threadpoolctl"]},
        timestamp_basis="Original Pacific wall-clock labels; no electrical or POA shifts",
        evaluation_status="2022–2023 already explored; held out only from forecasting fitting",
        latency_assumption="AC/POA labeled t and older are available at origin t; weather available 15 min after label. Unverified logger/averaging latency prevents deployment claim.")
    (OUT/"provenance.json").write_text(json.dumps(provenance, indent=2))
    print(scores.loc[scores.slice.eq("all_daytime")].to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
