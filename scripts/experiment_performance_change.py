"""Bounded fixed-reference performance experiment; no baseline writes."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from scipy.stats import t
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent))
from audit_stage2_readiness import indexed, digest

OUT = Path("outputs/2107_performance_change")
BASE = Path("outputs/2107_research_baseline")
DATA = Path("data/2107/processed")


def statistical_features(x):
    p, a, w, hour, season = x.T
    return np.column_stack([p, p*p, p*a, p*w, p*hour, p*hour*hour,
                            p*np.sin(season), p*np.cos(season)])


def slope_fit(frame, jumps, regimes=False, zeros=False):
    """Common slope, unit/season intercepts; site-time HAC, not 24 plant replicates."""
    f = frame.dropna(subset=["index"]).copy()
    dates = pd.to_datetime(f.month)
    time = (dates.dt.year - 2019) + (dates.dt.month - 1) / 12
    columns = [np.ones(len(f)), time.to_numpy()]
    columns.extend(pd.get_dummies(f.inverter, drop_first=True).to_numpy().T)
    columns.extend(pd.get_dummies(dates.dt.month, drop_first=True).to_numpy().T)
    if regimes:
        for row in jumps.itertuples():
            indicator = ((f.inverter == row.inverter_id) & (f.month >= row.boundary)).astype(float)
            if indicator.any() and ((f.inverter == row.inverter_id) & (f.month < row.boundary)).any():
                columns.append(indicator.to_numpy())
    if zeros:
        columns.append(f.zero_fraction.to_numpy())
    x = np.column_stack(columns).astype(float)
    y = f["index"].to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    scores = pd.DataFrame(x * residual[:, None]).groupby(f.month.to_numpy()).sum().to_numpy()
    meat = scores.T @ scores
    for lag in range(1, 4):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag/4) * (cross + cross.T)
    covariance = bread @ meat @ bread
    se = np.sqrt(max(0, covariance[1, 1]))
    critical = t.ppf(.975, max(1, len(scores)-x.shape[1]//24-2))
    return dict(slope_pp_year=beta[1], lower=beta[1]-critical*se,
                upper=beta[1]+critical*se, months=len(scores), observations=len(f),
                rank=int(np.linalg.matrix_rank(x)), parameters=x.shape[1])


def aggregate(frame, ac, prediction, threshold, guard_prediction=None):
    """Fixed 2019 weather-cell weights; equal calendar-month weight across years."""
    rows, support = [], []
    years = list(range(2018, 2024))
    for outcome in ("delivered", "generating"):
        for i, column in enumerate(ac.columns):
            d = frame.copy()
            d["ratio"] = ac[column].to_numpy() / prediction
            d["zero"] = ac[column].eq(0).to_numpy().astype(float)
            valid = ac[column].notna().to_numpy() & ac[column].ge(0).to_numpy()
            if outcome == "generating":
                valid &= (ac[column].to_numpy() > 2) & (ac[column].to_numpy() < threshold)
                valid &= (prediction if guard_prediction is None else guard_prediction) < threshold
            d = d.loc[valid]
            cells = d.groupby(["year", "cell"]).agg(ratio=("ratio", "mean"),
                    zero=("zero", "mean"), n=("ratio", "size"))
            counts = cells.n.unstack("year").reindex(columns=years)
            main = counts.loc[:, 2019:2023].ge(10).all(axis=1)
            # Including 2018 cannot change the main experiment's support.
            for period, use_years, selected in [("main", years[1:], main),
                    ("with2018", years, main & counts[2018].ge(10))]:
                keys = counts.index[selected]
                weights = cells.loc[2019].loc[keys, "n"]
                for year in use_years:
                    g = cells.loc[year].loc[keys].copy()
                    g["weight"] = weights
                    g["calendar_month"] = g.index.str.split("|").str[0].astype(int)
                    for month, part in g.groupby("calendar_month"):
                        rows.append(dict(inverter=column[:6], year=year, month=f"{year}-{month:02d}",
                            period=period, outcome=outcome, ratio=np.average(part.ratio, weights=part.weight),
                            zero_fraction=np.average(part.zero, weights=part.weight),
                            samples=int(part.n.sum()), cells=len(part)))
                support.append(dict(inverter=column[:6], outcome=outcome, period=period,
                                    cells=len(keys), retained_samples=int(cells.loc[use_years].loc[
                                        cells.loc[use_years].index.get_level_values("cell").isin(keys), "n"].sum())))
    result = pd.DataFrame(rows)
    reference = result.loc[result.year.eq(2019)].groupby(["inverter", "period", "outcome"]).ratio.mean()
    result["index"] = [100*r.ratio/reference.loc[(r.inverter, r.period, r.outcome)] for r in result.itertuples()]
    return result, pd.DataFrame(support)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [BASE/"nominal_model.parquet", DATA/"2107_electrical_clean.parquet",
               Path("outputs/2107_stage2_readiness/persistent_peer_jumps.csv")]
    protected = sources + list(BASE.glob("*"))
    before = {str(p): digest(p) for p in protected if p.is_file()}
    nominal = indexed(sources[0])
    names = pq.ParquetFile(sources[1]).schema_arrow.names
    cols = [c for c in names if "_ac_power_" in c]
    ac = indexed(sources[1], ["measured_on", *cols]).reindex(nominal.index)
    valid = (nominal.model_available & ~nominal.electrical_qc_flag & ~nominal.irradiance_qc_flag
             & ~nominal.dst_fall_ambiguous & nominal.poa_w_m2.ge(200)
             & nominal.index.year.isin(range(2018, 2024)) & (nominal.index.month <= 10)
             & ac.notna().all(axis=1))
    n, ac = nominal.loc[valid], ac.loc[valid]
    idx = n.index
    x = np.column_stack([n.poa_w_m2/1000, n.ambient_temperature_c/25,
                          n.wind_speed_m_s/5, (idx.hour+idx.minute/60-12)/6,
                          2*np.pi*idx.dayofyear/365.25])
    target = ac.median(axis=1).to_numpy()
    dates = idx.normalize()
    train = (idx.year == 2019) & (idx.day % 2 == 1) & (target > 2)
    validation = (idx.year == 2019) & (idx.day % 2 == 0) & (target > 2)
    stat = make_pipeline(StandardScaler(), Ridge(alpha=10))
    ml = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, max_depth=4,
            min_samples_leaf=100, l2_regularization=10, learning_rate=.08, random_state=2107)
    stat.fit(statistical_features(x[train]), target[train])
    ml.fit(x[train], target[train])
    joblib.dump({"regression": stat, "ML": ml}, OUT/"fixed_2019_weather_models.joblib")
    predictions = {"regression": np.clip(stat.predict(statistical_features(x)), .5, 30),
                   "ML": np.clip(ml.predict(x), .5, 30)}
    # Coarse support bins are fixed before looking at trends; month retains seasonal support.
    frame = pd.DataFrame({"year": idx.year}, index=idx)
    frame["cell"] = [f"{m}|{h//3}|{int(p//150)}|{int(a//10)}" for m,h,p,a in
                      zip(idx.month, idx.hour, n.poa_w_m2, n.ambient_temperature_c)]
    jumps = pd.read_csv(sources[2])
    # October recovery is documented in the audit but its two-month detector cannot flag it.
    jumps = pd.concat([jumps, pd.DataFrame([dict(inverter_id="inv_04", boundary="2023-10")])], ignore_index=True)
    jumps.to_csv(OUT/"regime_boundaries.csv", index=False)
    monthly, supports, scores = [], [], []
    for method, pred in predictions.items():
        for threshold in (27, 29):
            result, support = aggregate(frame, ac, pred, threshold,
                                        np.maximum(*predictions.values()))
            result["method"], result["clip_guard"] = method, threshold
            support["method"], support["clip_guard"] = method, threshold
            monthly.append(result); supports.append(support)
        error = pred[validation]-target[validation]
        scores.append(dict(method=method, training_samples=int(train.sum()),
            validation_samples=int(validation.sum()), validation_dates=int(dates[validation].nunique()),
            mae_kw=float(np.mean(abs(error))), rmse_kw=float(np.sqrt(np.mean(error**2))),
            bias_kw=float(np.mean(error))))
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.to_csv(OUT/"monthly_indices.csv", index=False)
    pd.concat(supports).to_csv(OUT/"support.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT/"reference_validation.csv", index=False)
    slopes, annual = [], []
    for key, group in monthly.groupby(["method", "clip_guard", "period", "outcome"]):
        meta = dict(zip(["method", "clip_guard", "period", "outcome"], key))
        for treatment, regimes, zeros in [("raw", False, False), ("steps", True, False),
                                        ("steps_and_zeros", True, True)]:
            if zeros and key[-1] == "generating":
                continue
            slopes.append({**meta, "treatment": treatment, **slope_fit(group, jumps, regimes, zeros)})
            if key[2] == "main":
                for label, subset in [("2019_2022", group.loc[group.year.le(2022)]),
                                      ("2020_2023", group.loc[group.year.ge(2020)])]:
                    slopes.append({**meta, "period": label, "treatment": treatment,
                                   **slope_fit(subset, jumps, regimes, zeros)})
        # Peer normalization intentionally removes shared plant change; retained separately.
        peer = group.copy()
        peer["index"] = 100*peer["index"]/peer.groupby("month")["index"].transform("median")
        for inv, unit in peer.groupby("inverter"):
            slopes.append({**meta, "treatment": "peer_steps", "inverter": inv,
                           **slope_fit(unit, jumps, True, False)})
        for inv, unit in group.groupby("inverter"):
            slopes.append({**meta, "treatment": "weather_steps", "inverter": inv,
                           **slope_fit(unit, jumps, True, False)})
        site = group.groupby(["year", "month"])[["index", "zero_fraction"]].mean().reset_index()
        ref = site.loc[site.year.eq(2019)].set_index(site.loc[site.year.eq(2019), "month"].str[-2:])["index"]
        for year, part in site.groupby("year"):
            differences = part["index"].to_numpy()-ref.reindex(part.month.str[-2:]).to_numpy()
            margin = t.ppf(.975, len(part)-1)*np.std(differences, ddof=1)/np.sqrt(len(part))
            annual.append({**meta, "year": int(year), "index": 100+np.mean(differences),
                "change_from2019_pp": np.mean(differences), "lower": np.mean(differences)-margin,
                "upper": np.mean(differences)+margin, "zero_fraction": part.zero_fraction.mean(),
                "months": len(part)})
    pd.DataFrame(slopes).to_csv(OUT/"slopes.csv", index=False)
    pd.DataFrame(annual).to_csv(OUT/"annual_indices.csv", index=False)
    # Delivered cell weights are identical for every unit, so this is the strict
    # 24-unit sum divided by 24 times predicted reference power, not a peer metric.
    plant = monthly.loc[monthly.outcome.eq("delivered") & monthly.clip_guard.eq(27)].groupby(
        ["method", "period", "year", "month"])[["ratio", "zero_fraction"]].mean().reset_index()
    for (method, period), g in plant.groupby(["method", "period"]):
        ref = g.loc[g.year.eq(2019), "ratio"].mean()
        plant.loc[g.index, "index"] = 100*g.ratio/ref
    plant.to_csv(OUT/"strict_plant_delivered.csv", index=False)
    pd.DataFrame([dict(method=m, period=p, **slope_fit(
        g.assign(inverter="plant"), jumps, False, False)) for (m,p),g in
        plant.groupby(["method", "period"])]).to_csv(OUT/"strict_plant_slopes.csv", index=False)
    plot_results(monthly)
    plot_annual(pd.DataFrame(annual))
    after = {p: digest(Path(p)) for p in before}
    assert before == after, "Protected input changed"
    provenance = dict(input_hashes=before, protected_unchanged=True, random_seed=2107,
        reference="2019 odd-numbered dates; fleet median >2 kW; Jan-Oct",
        features="POA, ambient, wind, clock hour, fixed annual phase; no year/trend/AC predictors",
        support="month x 3-hour x 150 W/m2 POA x 10 C ambient; >=10 samples each year",
        annual_interval="paired calendar-month differences vs 2019, t interval; descriptive, ten months",
        slope_interval="monthly site-cluster HAC lag3; conditional on models/support/screened boundaries",
        package_versions={"numpy": np.__version__, "pandas": pd.__version__})
    import sklearn
    provenance["package_versions"]["scikit-learn"] = sklearn.__version__
    (OUT/"provenance.json").write_text(json.dumps(provenance, indent=2))
    print(pd.DataFrame(scores).to_string(index=False))
    print(pd.DataFrame(slopes).query("treatment not in ['peer_steps','weather_steps'] and clip_guard == 27").to_string(index=False))


def plot_results(monthly):
    main = monthly.loc[monthly.period.eq("main") & monthly.clip_guard.eq(27)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for outcome, ax in zip(["delivered", "generating"], axes):
        for method, group in main.loc[main.outcome.eq(outcome)].groupby("method"):
            line = group.groupby("month")["index"].mean()
            line.index = pd.to_datetime(line.index)
            line = line.reindex(pd.date_range(line.index.min(), line.index.max(), freq="MS"))
            ax.plot(line.index, line, label=method)
        ax.axhline(100, color="grey", lw=.7)
        ax.set_ylabel(f"{outcome.title()} index")
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Fixed 2019 weather reference; zeros retained in delivered performance")
    fig.tight_layout(); fig.savefig(OUT/"plant_monthly.png", dpi=160); plt.close(fig)
    selected = main.loc[main.method.eq("ML") & main.outcome.eq("generating")].copy()
    selected["peer"] = 100*selected["index"]/selected.groupby("month")["index"].transform("median")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for inv in ["inv_04", "inv_05", "inv_17"]:
        part = selected.loc[selected.inverter.eq(inv)]
        part = part.set_index(pd.to_datetime(part.month)).reindex(pd.date_range("2019-01-01", "2023-10-01", freq="MS"))
        axes[0].plot(part.index, part["index"], label=inv)
        axes[1].plot(part.index, part.peer, label=inv)
    for ax in axes:
        ax.axhline(100, color="grey", lw=.7); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel("Weather-adjusted index")
    axes[1].set_ylabel("Fleet-relative index")
    fig.suptitle("Previously screened operational changes: cause unconfirmed")
    fig.tight_layout(); fig.savefig(OUT/"operational_changes.png", dpi=160); plt.close(fig)


def plot_annual(annual):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    main = annual.loc[annual.period.eq("main") & annual.clip_guard.eq(27)]
    for outcome, ax in zip(["delivered", "generating"], axes):
        for method, g in main.loc[main.outcome.eq(outcome)].groupby("method"):
            shift = -.06 if method == "ML" else .06
            ax.errorbar(g.year+shift, g.change_from2019_pp,
                yerr=np.vstack([g.change_from2019_pp-g.lower, g.upper-g.change_from2019_pp]),
                marker="o", capsize=3, label=method)
        ax.axhline(0, color="grey", lw=.7); ax.set_title(outcome.title())
        ax.set_xticks(range(2019, 2024)); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel("Index-point change from 2019 (95% interval)")
    fig.suptitle("Jan–Oct, identical seasonal support; paired-month uncertainty")
    fig.tight_layout(); fig.savefig(OUT/"annual_changes.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
