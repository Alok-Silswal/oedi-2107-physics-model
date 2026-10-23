"""Consolidate saved results and accepted reported quantum scores; no model inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_stage2_readiness import solar_elevation
from evaluate_stage2_quantum import scores_with_r2

OUT = Path("outputs/2107_final_reporting")
DOC = Path("docs/final_results.md")
CORRECTION = Path("outputs/2107_stage2_correction")
DIRECT = Path("outputs/2107_stage2_direct")
QUANTUM = Path("outputs/2107_stage2_quantum_cpu")
NEURAL = Path("outputs/2107_stage2_neural")
METRICS = [f"{period}_{metric}" for period in ("daylight", "all_day")
           for metric in ("mae_kw", "rmse_kw", "bias_kw", "r2")]
SOURCES = set()


def source(path):
    path = Path(path)
    SOURCES.add(path)
    return path


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect():
    rows = []
    reference = pd.read_parquet(source(CORRECTION/"hpo/best_validation_predictions.parquet"))
    day = solar_elevation(pd.DatetimeIndex(reference.target_timestamp)).to_numpy() > 0
    assert (len(reference), int(day.sum())) == (70555, 43949)

    def append(phase, scheme, family, statistic, values, origin, **extra):
        counts = (43949, 70555) if phase == "validation" else (48190, 77561)
        rows.append(dict(phase=phase, scheme=scheme, family=family, statistic=statistic,
                         daylight_n=counts[0], all_day_n=counts[1], source=str(origin),
                         **{m: values.get(m, np.nan) for m in METRICS}, **extra))

    def validation(frame, column, recorded):
        assert frame.index.is_unique and frame.target_timestamp.is_unique
        pd.testing.assert_frame_equal(frame[["target_timestamp", "actual_target_power_kw"]],
            reference[["target_timestamp", "actual_target_power_kw"]], check_dtype=False, check_exact=True, check_freq=False)
        result = scores_with_r2(frame, [column], day).loc[column].to_dict()
        for key, value in recorded.items():
            if key in result:
                np.testing.assert_allclose(result[key], value, rtol=1e-12, atol=1e-9)
                result[key] = value
        return result

    hpo = read_json(CORRECTION/"hpo/settings.json")
    trials = pd.read_csv(source(CORRECTION/"hpo/trials.csv"), float_precision="round_trip").set_index("id")
    for family, selection in hpo["best_per_family"].items():
        recorded = trials.loc[selection["trial"]].to_dict()
        values = validation(reference, family, recorded)
        append("validation", "correction", family, "selected_model", values,
               "hpo/trials.csv + best_validation_predictions.parquet", trial=selection["trial"])
    for column, family in (("physics_forecast_kw", "physics"), ("persistence_kw", "persistence")):
        append("validation", "reference", family, "reference", validation(reference, column, {}),
               str(CORRECTION/"hpo/best_validation_predictions.parquet"))

    direct = pd.read_csv(source(DIRECT/"results_summary.csv"), float_precision="round_trip")
    for _, row in direct.iterrows():
        name = row["model"]
        if name.startswith("direct_classical_"):
            suffix = name.removeprefix("direct_classical_")
            statistic = suffix if suffix in ("mean", "sample_std") else "seed"
            append("validation", "direct", "classical_neural", statistic, row,
                   str(DIRECT/"results_summary.csv"), seed=row.get("seed"),
                   best_epoch=row.get("best_epoch"), stopping_epoch=row.get("stopping_epoch"))
        else:
            append("validation", "direct", name.removeprefix("direct_"), "selected_model", row,
                   str(DIRECT/"results_summary.csv"), trial=row["trial"])

    for root, family, seeds in ((NEURAL/"classical", "classical_neural", range(2107,2112)),
                                (QUANTUM/"quantum", "quantum_neural", range(2107,2110))):
        group = []
        for seed in seeds:
            directory = root/f"seed_{seed}"
            recorded = read_json(directory/"metrics.json")
            frame = pd.read_parquet(source(directory/"validation_predictions.parquet"))
            values = validation(frame, "final_power_kw", recorded)
            if family == "quantum_neural":
                # No quantum R2 is supplied in the reporting source.
                values["daylight_r2"] = values["all_day_r2"] = np.nan
            group.append({m: values[m] for m in METRICS})
            append("validation", "correction", family, "seed", values, str(directory), seed=seed,
                   best_epoch=recorded["best_epoch"], stopping_epoch=recorded["stopping_epoch"])
        summary = pd.DataFrame(group).agg(["mean", "std"])
        for statistic in summary.index:
            append("validation", "correction", family, "mean" if statistic=="mean" else "sample_std",
                   summary.loc[statistic], str(root)+"/seed_*/validation_predictions.parquet")

    comparison = pd.read_csv(source(CORRECTION/"test/comparison.csv"), float_precision="round_trip")
    assert comparison.daylight_n.eq(48190).all() and comparison.all_day_n.eq(77561).all()
    for _, row in comparison.iterrows():
        family = {"physics_forecast_kw":"physics", "persistence_kw":"persistence"}.get(row.family, row.family)
        append("test", row.scheme, family, row.statistic, row, str(CORRECTION/"test/comparison.csv"), trial=row.get("trial"))
    for path, scheme, prefix in ((Path("outputs/2107_stage2_test/metrics.csv"), "correction", "classical_"),
                                  (DIRECT/"test/metrics.csv", "direct", "direct_classical_")):
        table = pd.read_csv(source(path), index_col=0, float_precision="round_trip")
        for seed in range(2107,2112):
            row = table.loc[prefix+str(seed)]
            assert (row.daylight_n,row.all_day_n)==(48190,77561)
            append("test", scheme, "classical_neural", "seed", row, str(path), seed=seed)

    text = source(OUT/"reported_quantum_results.md").read_text(encoding="utf-8-sig")
    block = text.split("## Held-Out Test Results",1)[1].split("## Files",1)[0]
    assert "77,561" in block and "48,190" in block
    reported = [line.split("|")[1:-1] for line in block.splitlines() if line.startswith("|")][2:]
    test_metrics = ["daylight_mae_kw", "daylight_rmse_kw", "daylight_bias_kw",
                    "all_day_mae_kw", "all_day_rmse_kw", "all_day_bias_kw"]
    assert [cells[0].strip() for cells in reported]==["2107","2108","2109","Mean","Sample SD"]
    for cells in reported:
        label, *values = [c.strip() for c in cells]
        assert all(len(v.split(".")[-1])==6 for v in values)
        statistic = {"Mean":"mean", "Sample SD":"sample_std"}.get(label,"seed")
        append("test", "correction", "quantum_neural", statistic, dict(zip(test_metrics,values)),
               str(OUT/"reported_quantum_results.md"), seed=int(label) if statistic=="seed" else None,
               status="Friend-verified; user-accepted final reported results; source precision 6 decimals")
    result = pd.DataFrame(rows)
    assert not result.duplicated(["phase","scheme","family","statistic","seed"]).any()
    assert len(result)==58
    for phase in ("validation","test"):
        for scheme in ("direct","correction"):
            actual = result.loc[(result.phase==phase)&(result.scheme==scheme)&result.statistic.eq("seed")
                                &result.family.eq("classical_neural"),"seed"]
            assert set(actual)==set(range(2107,2112))
    return result


def pretty_table(frame):
    columns = ["Model / statistic", "Best/stop", "Day MAE", "Day RMSE", "Day bias", "Day R²",
               "All MAE", "All RMSE", "All bias", "All R²"]
    labels = {"forest":"RF", "histogram":"Histogram", "xgboost":"XGBoost", "ridge":"Ridge",
              "classical_neural":"NN", "quantum_neural":"Quantum", "physics":"Physics", "persistence":"Persistence"}
    lines = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for _, row in frame.iterrows():
        label = labels[row.family]
        if row.scheme!="reference": label = ("Direct " if row.scheme=="direct" else "Physics + ")+label
        if row.statistic=="seed": label += f" {int(row.seed)}"
        elif row.statistic in ("mean","sample_std"): label += " mean" if row.statistic=="mean" else " sample SD"
        elif pd.notna(row.get("trial")): label += " ("+row.trial+")"
        epochs = f"{int(row.best_epoch)}/{int(row.stopping_epoch)}" if pd.notna(row.get("best_epoch")) else "—"
        values = ["—" if pd.isna(row[m]) else f"{float(row[m]):.{4 if m.endswith('r2') else 3}f}" for m in METRICS]
        lines.append("| "+" | ".join([label,epochs,*values])+" |")
    return "\n".join(lines)


def write_summary(frame):
    text = """# Final 15-minute PV forecasting results

Power errors are **kW**; bias = prediction − actual; R² is unitless. Blank/— means unavailable. Tables are rounded for presentation; `outputs/2107_final_reporting/results.csv` retains stored precision. Quantum test values retain exactly their six-decimal source precision in that file.

Quantum test scores are **friend-verified, user-accepted final reported results**, copied from `reported_quantum_results.md`. Quantum test predictions and evaluator provenance were not supplied locally. No quantum R² was supplied. Reported quantum test means/sample SD are retained as supplied, not recalculated from rounded seeds.

## Method and scope

- One site, 24 inverters; strict complete-fleet targets, recorded zeros retained, 15-minute forecasts. Inverters are units within one plant.
- Shared elapsed-time 70/15/15 split by target timestamp: validation starts 2022-01-21 16:45; testing starts 2022-12-11 20:20; record ends 2023-10-31 23:50 exclusive. Exported naive Pacific timestamps and existing QC/availability rules are retained.
- Weather → frozen Stage 1 physics → signed correction; direct models exclude physics. Settings/checkpoints selected on validation, with no test refit or new selection. Search ranges were specified after baseline inspection.
- Parts of the chronological test period appeared in earlier exploratory work: held out from these fits, not entirely unseen. Logger/current-measurement availability assumptions remain unverified physical latency.
- NN summaries are mean and **sample SD across five seeds**, not ensemble forecasts or confidence intervals. Quantum uses three seeds, four qubits/two layers in CPU simulation and batch 256; classical NN uses batch 128. Different input dimensions/settings preclude claims of identical capacity or quantum advantage.
- Returned quantum training metadata (Python 3.13.1/Torch 2.14.1 and reference fingerprints) differs from the local reference. Historical metadata and compatibility checks remain unchanged. Quantum test reporting follows the accepted friend source.

Validation: **70,555 rows / 43,949 daylight**. Test: **77,561 rows / 48,190 daylight**. Daylight uses target solar elevation >0°. Missing validation descriptive metrics are calculated only from saved validation forecasts; no model inference or held-out evaluation is performed by the reporting script. Classical correction test per-seed R² is absent from its saved metric table; available aggregate R² is retained from the saved comparison.
"""
    for phase, title in (("validation","Validation — model selection"),("test","Held-out test — final reporting")):
        part=frame.loc[frame.phase.eq(phase)]
        text += "\n## "+title+"\n\n### ML families and references\n\n"
        text += pretty_table(part.loc[~part.family.isin(["classical_neural","quantum_neural"])])+"\n"
        text += "\n### Neural / quantum seeds and summaries\n\n"
        text += pretty_table(part.loc[part.family.isin(["classical_neural","quantum_neural"])])+"\n"
    text += """
## Sources and figures

- ML: correction HPO settings/trials and saved best-validation forecasts; direct `results_summary.csv`; correction `test/comparison.csv` and direct test metrics.
- Classical NN: per-seed saved validation metrics/forecasts; `outputs/2107_stage2_test/metrics.csv`; direct test metrics and saved comparison summaries.
- Quantum: byte-verified returned CPU training bundle and `outputs/2107_final_reporting/reported_quantum_results.md`. Downloads retains the original ZIP backup. Five duplicate downloaded Parquet files and `metrics(3).csv` were not copied.
- Full source hashes, archive import verification and reporting provenance: `outputs/2107_final_reporting/provenance.json`. The external compiled reference was cross-checked; malformed blank model labels were replaced using authoritative local files.
- Final held-out figures: `test_ml_mae`, `test_ml_rmse`, `test_neural_seed_summary` (300-dpi PNG and vector SVG) under the same output folder. Earlier exploratory plots are preserved separately.

This Markdown is suitable for deliberate Git inclusion. Outputs remain ignored; `.gitignore` is unchanged.
"""
    DOC.write_text(text,encoding="utf-8")


def figures(frame):
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False,
                         "svg.fonttype":"none", "savefig.facecolor":"white"})
    test=frame.loc[frame.phase.eq("test")]
    colors=["#2864AD","#D56C35","#7450A3"]
    families=["ridge","forest","histogram","xgboost"]
    def save(fig,name):
        fig.savefig(OUT/(name+".png"),dpi=300)
        fig.savefig(OUT/(name+".svg"))
        plt.close(fig)
    for metric in ("mae","rmse"):
        fig,axes=plt.subplots(1,2,figsize=(12,5.2))
        fig.subplots_adjust(left=.18,right=.98,bottom=.17,top=.77,wspace=.19)
        fig.suptitle("Held-out test | Direct vs physics + ML correction",fontsize=16,y=.97)
        fig.text(.5,.89,"15 minutes ahead | 77,561 eligible rows; 48,190 daylight",ha="center",fontsize=11)
        for ax,period,label in zip(axes,("daylight","all_day"),("Daylight","All-day")):
            for k,scheme in enumerate(("direct","correction")):
                values=[float(test.loc[test.scheme.eq(scheme)&test.family.eq(f),period+"_"+metric+"_kw"].iloc[0]) for f in families]
                bars=ax.barh(np.arange(4)+(k-.5)*.34,values,height=.30,color=colors[k],label="Direct" if k==0 else "Physics + correction")
                ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=10)
            ax.set(yticks=np.arange(4),yticklabels=["Ridge","Random Forest","Histogram boosting","XGBoost"],
                   xlabel=metric.upper()+" (kW)",title=label)
            ax.invert_yaxis()
            ax.set_xlim(0,ax.get_xlim()[1]*1.13)
            ax.grid(axis="x",alpha=.16);ax.set_axisbelow(True)
        axes[1].tick_params(axis="y",labelleft=False)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
        save(fig,"test_ml_"+metric)
    groups=[("direct","classical_neural","Direct NN\n5 seeds"),("correction","classical_neural","Physics + NN\n5 seeds"),
            ("correction","quantum_neural","Physics + quantum\n3 seeds; reported")]
    fig,axes=plt.subplots(2,2,figsize=(12,8.2))
    fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.84,hspace=.46,wspace=.23)
    fig.suptitle("Held-out test | Neural and quantum seed summaries",fontsize=16,y=.98)
    fig.text(.5,.925,"15 minutes ahead | 77,561 eligible rows; 48,190 daylight",ha="center")
    for ax,(period,metric) in zip(axes.flat,[("daylight","mae"),("all_day","mae"),("daylight","rmse"),("all_day","rmse")]):
        means=[];sds=[]
        for scheme,family,_ in groups:
            group=test.loc[test.scheme.eq(scheme)&test.family.eq(family)]
            means.append(float(group.loc[group.statistic.eq("mean"),period+"_"+metric+"_kw"].iloc[0]))
            sds.append(float(group.loc[group.statistic.eq("sample_std"),period+"_"+metric+"_kw"].iloc[0]))
        ax.bar(range(3),means,yerr=sds,color=colors,width=.52,capsize=5,error_kw=dict(elinewidth=1.3))
        for i,(mean,sd) in enumerate(zip(means,sds)):
            ax.text(i,mean+sd+max(means)*.025,f"{mean:.2f} ± {sd:.2f}",ha="center",fontsize=10)
        ax.set(xticks=range(3),xticklabels=[g[2] for g in groups],ylabel=metric.upper()+" (kW)",
               title=("Daylight" if period=="daylight" else "All-day"),ylim=(0,max(np.array(means)+sds)*1.19))
        ax.grid(axis="y",alpha=.16);ax.set_axisbelow(True)
    fig.text(.5,.045,"Error bars: sample SD across seeds, not confidence intervals; no ensemble.\nQuantum: friend-verified, user-accepted reported scores (six-decimal source); CPU simulation.",ha="center",fontsize=10)
    save(fig,"test_neural_seed_summary")


def main():
    frame=collect()
    compiled=Path("C:/Users/silsw/Documents/Codex/2026-10-16/sy/outputs/pv_final_result_tables.md")
    block=source(compiled).read_text(encoding="utf-8-sig").split("## Held-out comparison:",1)[1].split("## Held-out direct",1)[0]
    cells=[[c.strip() for c in line.split("|")[1:-1]] for line in block.splitlines() if line.startswith("|")]
    comparison=pd.read_csv(CORRECTION/"test/comparison.csv",float_precision="round_trip")
    for values in cells[2:]:
        recorded=dict(zip(cells[0],values))
        selected=comparison.loc[comparison.scheme.eq(recorded["scheme"])&comparison.family.eq(recorded["family"])
                                &comparison.statistic.eq(recorded["statistic"])]
        assert len(selected)==1
        for metric in METRICS:
            assert f"{selected.iloc[0][metric]:.6f}"==recorded[metric],"Compiled reference differs: "+metric
    for c in ("seed","best_epoch","stopping_epoch","daylight_n","all_day_n"):
        frame[c]=frame[c].astype("Int64")
    frame.to_csv(OUT/"results.csv",index=False)
    write_summary(frame)
    figures(frame)
    provenance=json.loads((OUT/"provenance.json").read_text())
    returned=read_json(QUANTUM/"settings.json")
    local=read_json(NEURAL/"settings.json")
    fields=["frame_fingerprints","preprocessing_sha256","versions","python","target_scale_kw"]
    provenance.update(reporting=dict(rows=len(frame),rows_per_phase=frame.groupby("phase").size().to_dict(),
        units="Power metrics kW; bias=prediction-actual; R2 unitless",sample_sd_ddof=1,
        validation_additions="Missing descriptive metrics from saved validation predictions only; original stored metrics retained",
        quantum_r2="Not reported; left blank",quantum_test_precision="Six decimals as supplied, including aggregates",
        quantum_metadata_differences={k:dict(returned=returned.get(k),local=local.get(k)) for k in fields if returned.get(k)!=local.get(k)},
        source_sha256={str(p):digest(p) for p in sorted(SOURCES|{Path(__file__)})},
        compiled_reference_checks=len(cells[2:])*len(METRICS),
        matplotlib_version=matplotlib.__version__,plots="Held-out saved aggregate metrics only; earlier figures not reused"))
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print("Reporting rows:",frame.groupby("phase").size().to_dict())
    print("Tables:",OUT/"results.csv",DOC)
    print("Figures:",OUT,"(3 PNG + 3 SVG)")


if __name__=="__main__":
    main()
