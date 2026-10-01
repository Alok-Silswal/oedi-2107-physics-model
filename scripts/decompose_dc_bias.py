"""Diagnostic-only decomposition of existing ideal DC bias; no parameter fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from oedi2107.bias_diagnostics import diagnostic_geometry, matched_ampm, matched_summary, thermal_sensitivity, FAIMAN_SCENARIOS
from oedi2107.config import SystemConfig
from oedi2107.module import load_cec_module


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio_bins(frame, *, paco, pdco):
    """Explicit weather/calendar/geometry bins -> finite ratio quantiles and counts."""
    groups = {
        "temp_cell_c":pd.cut(frame.temp_cell, [-20,10,20,30,40,50,60,70,90], right=False),
        "ambient_c":pd.cut(frame.ambient_temperature_c, [-20,0,10,20,30,40,50], right=False),
        "wind_m_s":pd.cut(frame.wind_speed_m_s, [0,.5,1,2,3,4,6,10,30], right=False),
        "poa_w_m2":pd.cut(frame.poa_w_m2, [0,300,400,500,600,700,800,1000,1500], right=False),
        "aoi_deg":pd.cut(frame.aoi_deg, [0,20,30,40,50,60,70,80,90,180], right=False),
        "hour_local":frame.index.hour, "month":frame.index.month, "season":frame.season,
        "year":frame.index.year, "solar_phase":frame.solar_phase,
        "ac_fraction_paco":pd.cut(frame.maximum_ac_power_w / paco, [0,.2,.4,.6,.8,.9,1,2,5], right=False),
        "ideal_dc_fraction_pdco":pd.cut(frame.expected_dc_power_ideal / pdco, [0,.2,.4,.6,.8,.9,1,2], right=False),
    }
    rows = []
    for name, labels in groups.items():
        for value, subset in frame.groupby(labels, observed=True, dropna=False):
            row = {"dimension":name, "bin":str(value), "total_timestamps":len(subset)}
            for quantity in ("voltage", "current", "power"):
                ratios = subset[f"{quantity}_ratio"]
                finite = ratios[np.isfinite(ratios)]
                row.update({f"{quantity}_finite_count":len(finite), f"{quantity}_excluded_count":len(subset)-len(finite),
                            f"{quantity}_p10":finite.quantile(.1), f"{quantity}_median":finite.median(), f"{quantity}_p90":finite.quantile(.9)})
            rows.append(row)
    return pd.DataFrame(rows)


def correlations(frame):
    """Unfitted descriptive correlations; no significance/causal interpretation."""
    numeric = frame.assign(hour_local=frame.index.hour + frame.index.minute / 60)
    rows = []
    for phase in ("all", "AM", "PM"):
        subset = numeric if phase == "all" else numeric.loc[numeric.solar_phase.eq(phase)]
        for quantity in ("voltage", "current", "power"):
            for variable in ("temp_cell", "ambient_temperature_c", "wind_speed_m_s", "poa_w_m2", "hour_local", "aoi_deg"):
                paired = subset[[f"{quantity}_ratio",variable]].replace([np.inf,-np.inf],np.nan).dropna()
                rows.append({"phase":phase,"quantity":quantity,"variable":variable,"total":len(subset),"valid":len(paired),"excluded":len(subset)-len(paired),
                             "spearman":paired.corr(method="spearman").iloc[0,1] if len(paired)>1 else np.nan})
    return pd.DataFrame(rows)


def review_plots(frame, sensitivity, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14,10))
    variables = [("aoi_deg","AOI [deg]"),("temp_cell","Original estimated cell temperature [°C]"),("poa_w_m2","Measured POA [W/m²]")]
    for row, quantity in enumerate(("voltage","current","power")):
        for col, (variable,label) in enumerate(variables):
            ax = axes[row,col]
            for phase, color in (("AM","tab:blue"),("PM","tab:orange")):
                subset = frame.loc[frame.solar_phase.eq(phase)]
                ax.scatter(subset[variable],subset[f"{quantity}_ratio"],s=3,alpha=.2,label=phase,color=color)
            ax.axhline(1,color="black",linewidth=.6)
            ax.set_xlabel(label)
            ax.set_ylabel(f"Measured / ideal {quantity}")
    axes[0,0].legend()
    fig.suptitle("Frozen candidate subset; all ratios retained; diagnostic solar geometry only")
    fig.tight_layout()
    fig.savefig(output / "ratio_aoi_temperature_poa.png",dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax, variable, label in zip(axes,("ambient_temperature_c","wind_speed_m_s","hour_local"),("Ambient [°C]","Wind [m/s]","Pacific wall-clock hour")):
        for phase,color in (("AM","tab:blue"),("PM","tab:orange")):
            subset = frame.loc[frame.solar_phase.eq(phase)]
            x = subset.index.hour + subset.index.minute/60 if variable == "hour_local" else subset[variable]
            ax.scatter(x,subset.voltage_ratio,s=3,alpha=.2,color=color,label=phase)
        ax.set_xlabel(label)
        ax.set_ylabel("Voltage ratio")
        ax.axhline(1,color="black",linewidth=.6)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "voltage_wind_ambient_hour.png",dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax,quantity in zip(axes,("voltage","current","power")):
        x=np.arange(len(summaries))
        ax.plot(x,summaries[f"{quantity}_AM"],"o-",label="Matched AM")
        ax.plot(x,summaries[f"{quantity}_PM"],"o-",label="Matched PM")
        ax.set_xticks(x,summaries.scenario,rotation=65,ha="right",fontsize=7)
        ax.set_ylabel(f"{quantity} ratio; weighted shared-bin medians")
        ax.axhline(1,color="black",linewidth=.6)
    axes[0].legend()
    fig.suptitle("Fixed Faiman perturbations; original POA/temperature/season/year bins and candidate mask frozen")
    fig.tight_layout()
    fig.savefig(output / "faiman_sensitivity_matched.png",dpi=150)
    plt.close(fig)
    yearly = frame.assign(year=frame.index.year).groupby(["year","solar_phase"],observed=True)[["voltage_ratio","current_ratio","power_ratio"]].median().unstack("solar_phase")
    yearly.plot(subplots=True,figsize=(12,10),marker="o",layout=(3,2))
    plt.suptitle("Calendar coverage differs; year/phase medians are descriptive, not degradation estimates")
    plt.tight_layout()
    plt.savefig(output / "year_phase_ratios.png",dpi=150)
    plt.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",type=Path,default=Path("outputs/2107_dc_candidates"))
    parser.add_argument("--ideal",type=Path,default=Path("outputs/2107_dc_ideal/nominal_dc_ideal.parquet"))
    parser.add_argument("--metadata",type=Path,default=Path("data/2107/raw/2107_system_metadata.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/2107_dc_bias"))
    args=parser.parse_args()
    sources=[args.ideal,args.metadata,*[args.candidates/name for name in ("timestamp_diagnostics.parquet","timestamp_candidate_masks.parquet","inverter_candidate_coverage.csv","provenance.json")]]
    hashes={str(path):sha256(path) for path in sources}
    config=SystemConfig()
    if config.dc_losses.k_dc is not None:
        raise ValueError("K_DC must remain unset")
    metadata=json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    site=metadata["Site"]
    location=dict(latitude=float(site["latitude"]),longitude=float(site["longitude"]),altitude=float(site["elevation(m)"]),tilt=config.tilt_deg,azimuth=config.azimuth_deg)
    mount=metadata["Mount"]["Mount 0"]
    if float(mount["tilt"]) != config.tilt_deg or float(mount["azimuth"]) != config.azimuth_deg:
        raise ValueError("Review model/metadata mount disagreement before diagnostic AOI")
    if metadata["System"]["timezone_code"] != "PST8PDT" or "US/Pacific" not in metadata["System"]["comments"]:
        raise ValueError("Review metadata timezone before diagnostic localization")
    prior=json.loads((args.candidates/"provenance.json").read_text())
    if prior["k_dc"] is not None or prior["calibration"] != "none":
        raise ValueError("Requires uncalibrated candidate outputs")
    features=pd.read_parquet(args.candidates/"timestamp_diagnostics.parquet")
    masks=pd.read_parquet(args.candidates/"timestamp_candidate_masks.parquet")
    if not features.index.equals(masks.index):
        raise ValueError("Diagnostic features and candidate masks must align exactly")
    baseline=pd.read_parquet(args.ideal,columns=["temp_cell",*[f"expected_dc_{q}_ideal" for q in ("voltage","current","power")]])
    pd.testing.assert_frame_equal(features[baseline.columns],baseline,check_flags=False,check_names=True)
    preliminary=masks[["valid_model_inputs","known_qc_and_dst_clear","meaningful_irradiance",
                       "available_dc_and_ac","all_inverters_operating","below_rating_guards"]].all(axis=1)
    frame=features.loc[preliminary].copy()
    frame["candidate_healthy_mask"]=masks.loc[preliminary,"candidate_healthy_mask"]
    frame=frame.join(diagnostic_geometry(frame.index,**location,timezone="America/Los_Angeles"))
    frame["season"]=pd.Series(frame.index.month,index=frame.index).map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})
    fixed=diagnostic_geometry(frame.index,**location,timezone="Etc/GMT+8")
    frame["aoi_fixed_pst_deg"]=fixed.aoi_deg
    frame["phase_fixed_pst"]=fixed.solar_phase
    args.output.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(args.output/"diagnostic_context.parquet")
    candidate=frame.loc[frame.candidate_healthy_mask].copy()
    print(f"Frozen candidates: {len(candidate)}; preliminary context: {len(frame)}",flush=True)
    for label,subset in (("candidate",candidate),("preliminary",frame)):
        ratio_bins(subset,paco=prior["paco_w"],pdco=prior["pdco_w"]).to_csv(args.output/f"{label}_ratio_bins.csv",index=False)
        correlations(subset).to_csv(args.output/f"{label}_correlations.csv",index=False)
        # Separate phase branches prevent marginal AOI associations from hiding AM/PM differences.
        subset.assign(aoi_bin=pd.cut(subset.aoi_deg,[0,20,30,40,50,60,70,80,90,180],right=False)).groupby(
            ["aoi_bin","solar_phase"],observed=True).agg(timestamps=("current_ratio","size"),
            voltage_ratio_median=("voltage_ratio","median"),current_ratio_median=("current_ratio","median"),power_ratio_median=("power_ratio","median")).to_csv(args.output/f"{label}_aoi_phase.csv")
    matches=[]
    preliminary_match=matched_ampm(frame,include_year=True)
    preliminary_match.to_csv(args.output/"matched_preliminary_plus_year.csv",index=False)
    matches.append({"comparison":"preliminary_plus_year",**matched_summary(preliminary_match)})
    for label,options in (("poa_temp_season",{}),("plus_year",{"include_year":True}),
                          ("plus_year_aoi",{"include_year":True,"include_aoi":True}),
                          ("plus_year_wind",{"include_year":True,"include_wind":True}),
                          ("narrow_50w_5c",{"include_year":True,"poa_width":50.,"temperature_width":5.}),
                          ("minimum_5_per_phase",{"include_year":True,"minimum_per_phase":5})):
        table=matched_ampm(candidate,**options)
        table.to_csv(args.output/f"matched_{label}.csv",index=False)
        matches.append({"comparison":label,**matched_summary(table)})
    alternate=candidate.copy()
    alternate["aoi_deg"]=alternate.aoi_fixed_pst_deg
    alternate["solar_phase"]=alternate.phase_fixed_pst
    fixed_match=matched_ampm(alternate,include_year=True)
    fixed_match.to_csv(args.output/"matched_fixed_pst_hypothesis.csv",index=False)
    matches.append({"comparison":"fixed_pst_hypothesis",**matched_summary(fixed_match)})
    matched_results=pd.DataFrame(matches)
    matched_results.to_csv(args.output/"matched_summary.csv",index=False)
    print(matched_results.to_string(index=False),flush=True)
    parameters=load_cec_module(config.cec_module_entry)
    original_parameters=parameters.copy(deep=True)
    sensitivity,equivalent=thermal_sensitivity(candidate,parameters,config)
    pd.testing.assert_series_equal(parameters,original_parameters)
    sensitivity.to_parquet(args.output/"faiman_sensitivity.parquet")
    equivalent.to_parquet(args.output/"voltage_equivalent_temperature.parquet")
    sensitivity_rows=[]
    for scenario,subset in sensitivity.groupby("scenario",sort=False):
        table=matched_ampm(subset,include_year=True)
        table.to_csv(args.output/f"faiman_matched_{scenario}.csv",index=False)
        row={"scenario":scenario,"u0":float(subset.diagnostic_u0.iloc[0]),"u1":float(subset.diagnostic_u1.iloc[0]),
             "temp_delta_median_c":float((subset.diagnostic_temp_cell-subset.temp_cell).median()),
             "temp_delta_p10_c":float((subset.diagnostic_temp_cell-subset.temp_cell).quantile(.1)),
             "temp_delta_p90_c":float((subset.diagnostic_temp_cell-subset.temp_cell).quantile(.9)),**matched_summary(table)}
        for quantity in ("voltage","current","power"):
            row[f"{quantity}_overall_median"]=float(subset[f"{quantity}_ratio"].median())
        row["voltage_wind_spearman"]=float(subset[["voltage_ratio","wind_speed_m_s"]].corr(method="spearman").iloc[0,1])
        row["voltage_original_temperature_spearman"]=float(subset[["voltage_ratio","temp_cell"]].corr(method="spearman").iloc[0,1])
        sensitivity_rows.append(row)
    sensitivity_summary=pd.DataFrame(sensitivity_rows)
    sensitivity_summary.to_csv(args.output/"faiman_sensitivity_summary.csv",index=False)
    print(sensitivity_summary.to_string(index=False),flush=True)
    equivalent.join(candidate[["season","solar_phase"]]).groupby(["season","solar_phase"],observed=True).agg(
        timestamps=("dvmp_dt_v_per_c","size"),temperature_offset_median_c=("linearized_voltage_equivalent_temperature_offset_c","median")).to_csv(args.output/"voltage_equivalent_temperature_phase.csv")
    candidate.assign(closure=candidate.power_ratio/(candidate.voltage_ratio*candidate.current_ratio)).closure.describe(percentiles=[.1,.5,.9]).to_csv(args.output/"median_ratio_closure.csv")
    review_plots(candidate,sensitivity,sensitivity_summary,args.output)
    after={str(path):sha256(path) for path in sources}
    if hashes != after:
        raise RuntimeError("Source files changed during diagnostic run")
    provenance={"sources_sha256_before":hashes,"sources_sha256_after":after,"sources_unchanged":hashes==after,
                "site_metadata":location,"timezone_metadata":metadata["System"]["timezone_code"],
                "diagnostic_timezone_hypothesis":"America/Los_Angeles","alternate_clock_hypothesis":"Etc/GMT+8 (fixed PST)",
                "clock_policy":"localize diagnostic copy only; ambiguous/nonexistent -> NaT; original timestamps unchanged",
                "candidate_timestamps":len(candidate),"preliminary_timestamps":len(frame),
                "unresolved_geometry_clock_candidates":int((~candidate.diagnostic_clock_valid).sum()),
                "candidates_without_daylight_phase":int(candidate.solar_phase.isna().sum()),
                "aoi_pst_minus_pacific_median_deg":float((candidate.aoi_fixed_pst_deg-candidate.aoi_deg).median()),
                "aoi_pst_minus_pacific_p10_deg":float((candidate.aoi_fixed_pst_deg-candidate.aoi_deg).quantile(.1)),
                "aoi_pst_minus_pacific_p90_deg":float((candidate.aoi_fixed_pst_deg-candidate.aoi_deg).quantile(.9)),
                "faiman_scenarios":FAIMAN_SCENARIOS,"pvlib_version":pvlib.__version__,"cec_module_entry":config.cec_module_entry,
                "production_configuration_unchanged":True,"k_dc":None,"parameter_optimization":"none","candidate_mask_policy":"frozen; no refiltering",
                "matching":"100 W/m2 POA and 10 C original temp bins + season; year/AOI/wind/narrow-bin/count sensitivity",
                "summary_weights":"min(AM count, PM count); only cells with >=10 each unless explicit sensitivity >=5",
                "atmosphere":"solarposition default temperature 12 C; altitude from metadata; geometry only"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    lines=["# Diagnostic DC bias decomposition", "", "No calibration; K_DC remains unset. Original measurements, timestamps and physics parameters are unchanged.",
           "", f"Frozen candidates: {len(candidate)}; preliminary comparison: {len(frame)}. Sources verified unchanged.",
           "", "## Matched comparisons", "", "```",matched_results.to_string(index=False),"```", "", "## Fixed Faiman sensitivity", "", "```",sensitivity_summary.to_string(index=False),"```", "",
           "Tables report descriptive associations, not causal attribution or optimized parameters. See docs/dc_bias_decomposition.md for interpretation and limitations."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
