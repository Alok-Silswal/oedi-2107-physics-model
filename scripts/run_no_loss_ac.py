"""IDEAL DC -> existing Sandia -> diagnostic inverter AC; no loss-factor assignment."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oedi2107.ac_reference import calculate_no_loss_ac, loading_summary, measured_upper_region
from oedi2107.config import SystemConfig
from oedi2107.inverter import load_cec_inverter, inverter_provenance
from oedi2107.real_data import discover_channels


def digest(path):
    hasher=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()


def plots(expected, measured, daylight, upper, paco, output):
    """QC-clear positive measured daylight AC -> four compact review figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs,ys=[],[]
    for column in measured:
        positions=np.flatnonzero((daylight & measured[column].gt(0)).to_numpy())
        positions=positions[::max(1,len(positions)//2500)]
        xs.append(expected.iloc[positions].to_numpy()/1000)
        ys.append(measured[column].iloc[positions].to_numpy()/1000)
    x,y=np.concatenate(xs),np.concatenate(ys)
    for name,residual in (("expected_vs_measured",False),("residual_vs_expected",True)):
        fig,ax=plt.subplots(figsize=(8,5))
        ax.hexbin(x,y-x if residual else y,gridsize=70,mincnt=1,bins="log")
        if residual:
            ax.axhline(0,color="black",linewidth=.8)
        else:
            ax.plot([0,32],[0,32],"k--",linewidth=.8,label="Equal AC power")
            ax.axhline(paco/1000,color="tab:red",linewidth=.8,label="Official Paco")
        ax.axvline(paco/1000,color="tab:red",linewidth=.8)
        ax.set_xlabel("Expected no-loss AC [kW]")
        ax.set_ylabel("Measured − expected [kW]" if residual else "Measured inverter AC [kW]")
        ax.set_title("All 24 inverters; QC-clear POA>0, measured AC>0; uncalibrated")
        if not residual:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/f"{name}.png",dpi=150)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    high=(x>=24)&(y>=24)
    axes[0].scatter(x[high],y[high],s=2,alpha=.15)
    axes[0].axvline(paco/1000,color="tab:red",label="Official Paco 27.6 kW")
    axes[0].axhline(paco/1000,color="tab:red")
    axes[0].axhline(30,color="black",linestyle="--",label="30 kW reference; not a new rating")
    axes[0].set(xlim=(24,28),ylim=(24,31),xlabel="Expected no-loss AC [kW]",ylabel="Measured AC [kW]")
    axes[0].legend(fontsize=8)
    band=upper.loc[(upper.inverter_id=="inv_01") & (upper.population=="electrical_qc_clear")].iloc[0]
    near=measured.inv_01.ge(band.dominant_high_band_lower_w) & measured.inv_01.lt(band.dominant_high_band_upper_w) & daylight
    date=near.groupby(near.index.date).sum().idxmax()
    index=measured.index[measured.index.normalize()==pd.Timestamp(date)]
    hour=index.hour+index.minute/60
    for identifier in measured:
        axes[1].plot(hour,measured.loc[index,identifier]/1000,color="gray",alpha=.15,linewidth=.7)
    axes[1].plot(hour,expected.loc[index]/1000,label="Expected no-loss",color="tab:blue")
    axes[1].plot(hour,measured.loc[index,"inv_01"]/1000,label="Measured inv_01",color="tab:orange")
    axes[1].axhline(paco/1000,color="tab:red",linewidth=.8)
    axes[1].set(xlabel="Pacific wall-clock hour",ylabel="AC [kW]",title=f"{date}: most inv_01 upper-band paired records")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"clipping_zoom.png",dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    edges=np.arange(26000,31200,100)
    for identifier in measured:
        values=measured.loc[daylight,identifier].to_numpy()
        counts,_=np.histogram(values,bins=edges)
        ax.step((edges[:-1]+50)/1000,counts,where="mid",alpha=.5,linewidth=.8)
    ax.axvline(paco/1000,color="tab:red",label="Official CEC Paco")
    ax.axvline(30,color="black",linestyle="--",label="30 kW reference only")
    ax.set(xlabel="Measured inverter AC [kW]",ylabel="Records per 100 W band",title="Upper AC distribution: 24 separate inverter histograms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"upper_ac_distribution.png",dpi=150)
    plt.close(fig)
    (output/"plot_provenance.json").write_text(json.dumps({"sampling":"up to about 2500 evenly spaced paired records per inverter; no data edits",
        "clipping_review_date":str(date),"histogram_width_w":100,"zoom_is_display_only":True}),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,default=Path("data/2107/processed/2107_model_5min.parquet"))
    parser.add_argument("--ideal",type=Path,default=Path("outputs/2107_dc_ideal/nominal_dc_ideal.parquet"))
    parser.add_argument("--metadata",type=Path,default=Path("data/2107/raw/2107_system_metadata.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/2107_ac_no_loss"))
    args=parser.parse_args()
    sources=[args.prepared,args.ideal,args.metadata]
    before={str(path):digest(path) for path in sources}
    config=SystemConfig()
    if config.dc_losses.k_dc is not None:
        raise ValueError("This diagnostic run requires production K_DC to remain unset")
    parameters=load_cec_inverter(config.cec_inverter_entry)
    original_parameters=parameters.copy(deep=True)
    baseline=pd.read_parquet(args.ideal)
    reference=calculate_no_loss_ac(baseline,parameters)
    pd.testing.assert_frame_equal(reference[baseline.columns],baseline)
    pd.testing.assert_series_equal(parameters,original_parameters)
    mapping=discover_channels(pq.ParquetFile(args.prepared).schema_arrow.names)
    metadata=json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    for channel in mapping.ac_power:
        metric=channel.rsplit("_",1)[-1]
        records=[record for record in metadata["Metrics"].values() if str(record.get("metric_id"))==metric]
        if len(records)!=1 or str(records[0].get("units")).lower()!="kw" or records[0].get("calc_scale")!=1 or records[0].get("calc_offset")!=0:
            raise ValueError(f"Cannot confirm unchanged prepared AC kW for {channel}")
    source=pd.read_parquet(args.prepared,columns=["measured_on",*mapping.ac_power])
    if not np.array_equal(source.measured_on.to_numpy(),baseline.measured_on.to_numpy()):
        raise ValueError("Prepared AC and ideal DC timestamps/order must match exactly")
    measured=pd.DataFrame(source[mapping.ac_power].to_numpy(dtype=float)*1000,index=baseline.index,columns=mapping.inverter_id)
    paco=float(parameters.Paco)
    args.output.mkdir(parents=True,exist_ok=True)
    reference.to_parquet(args.output/"nominal_ac_no_loss.parquet")
    mapping.to_csv(args.output/"channel_mapping.csv",index=False)
    qc_clear=~baseline.electrical_qc_flag & ~baseline.irradiance_qc_flag & ~baseline.dst_fall_ambiguous
    daylight=reference.no_loss_ac_computed & baseline.poa_w_m2.gt(0) & qc_clear
    night=reference.no_loss_ac_computed & baseline.poa_w_m2.eq(0) & qc_clear
    coverage,loadings,upper,nights=[],[],[],[]
    columns=["poa_w_m2","temp_cell","expected_dc_voltage_ideal","expected_dc_current_ideal","expected_dc_power_ideal",
             "expected_ac_power_no_loss","no_loss_ac_computed","model_inputs_available","electrical_qc_flag","irradiance_qc_flag","dst_fall_ambiguous"]
    writer=None
    try:
        for identifier in measured:
            table=reference[columns].copy()
            table["inverter_id"]=identifier
            table["measured_ac_power"]=measured[identifier]
            paired=np.isfinite(table.expected_ac_power_no_loss) & np.isfinite(table.measured_ac_power)
            table["residual_no_loss_ac_power"]=np.where(paired,table.measured_ac_power-table.expected_ac_power_no_loss,np.nan)
            table["ratio_measured_to_no_loss_ac"]=np.divide(table.measured_ac_power,table.expected_ac_power_no_loss,
                out=np.full(len(table),np.nan),where=paired & table.expected_ac_power_no_loss.gt(0))
            table.attrs={}
            arrow=pa.Table.from_pandas(table.reset_index(),preserve_index=False)
            if writer is None:
                writer=pq.ParquetWriter(args.output/"inverter_ac_no_loss.parquet",arrow.schema,compression="snappy")
            writer.write_table(arrow,row_group_size=100000)
            coverage.append({"inverter_id":identifier,"total_rows":len(table),"finite_measured_ac":int(np.isfinite(table.measured_ac_power).sum()),
                "valid_pairs":int(paired.sum()),"excluded_pairs":int((~paired).sum()),"illuminated_qc_clear_pairs":int((paired & daylight).sum()),
                "operating_illuminated_pairs":int((paired & daylight & table.measured_ac_power.gt(0)).sum()),
                "zero_poa_qc_clear_pairs":int((paired & night).sum()),
                "expected_at_paco_illuminated_count":int((daylight & table.expected_ac_power_no_loss.eq(paco)).sum())})
            for population,selection in (("all_illuminated_qc_clear",daylight),("positive_measured_illuminated_qc_clear",daylight & table.measured_ac_power.gt(0))):
                for basis in ("expected","measured"):
                    bins=loading_summary(table.loc[selection,"expected_ac_power_no_loss"],table.loc[selection,"measured_ac_power"],paco=paco,basis=basis)
                    loadings.append(bins.assign(inverter_id=identifier,population=population))
            for population,series in (("all_telemetry",measured[identifier]),
                                      ("electrical_qc_clear",measured[identifier].loc[~baseline.electrical_qc_flag])):
                upper.append({"inverter_id":identifier,"population":population,**measured_upper_region(series,paco=paco)})
            selected=table.loc[night & paired]
            nights.append({"inverter_id":identifier,"zero_poa_pairs":len(selected),
                "expected_median_w":selected.expected_ac_power_no_loss.median(),"measured_median_w":selected.measured_ac_power.median(),
                "expected_negative_count":int(selected.expected_ac_power_no_loss.lt(0).sum()),
                "measured_negative_count":int(selected.measured_ac_power.lt(0).sum()),"measured_zero_count":int(selected.measured_ac_power.eq(0).sum())})
            print(f"Completed {identifier}",flush=True)
    finally:
        if writer is not None:
            writer.close()
    pd.DataFrame(coverage).to_csv(args.output/"comparison_coverage.csv",index=False)
    pd.concat(loadings,ignore_index=True).to_csv(args.output/"inverter_loading_bins.csv",index=False)
    upper=pd.DataFrame(upper)
    upper.to_csv(args.output/"measured_upper_regions.csv",index=False)
    pd.DataFrame(nights).to_csv(args.output/"zero_poa_tare_summary.csv",index=False)
    # Pool observations only for descriptive bins; preserve the separate inverter records above.
    pooled_e=np.repeat(reference.loc[daylight,"expected_ac_power_no_loss"].to_numpy(),24)
    pooled_m=measured.loc[daylight].to_numpy().reshape(-1)
    pooled=[]
    for population,selection in (("all_illuminated_qc_clear",np.ones(len(pooled_m),dtype=bool)),
                                 ("positive_measured_illuminated_qc_clear",pooled_m>0)):
        for basis in ("expected","measured"):
            pooled.append(loading_summary(pooled_e[selection],pooled_m[selection],paco=paco,basis=basis).assign(population=population))
    pd.concat(pooled,ignore_index=True).to_csv(args.output/"pooled_loading_bins.csv",index=False)
    plots(reference.expected_ac_power_no_loss,measured,daylight,upper,paco,args.output)
    after={str(path):digest(path) for path in sources}
    if before!=after:
        raise RuntimeError("Inputs changed during diagnostic AC run")
    records=pq.ParquetFile(args.output/"inverter_ac_no_loss.parquet").metadata.num_rows
    if records!=len(baseline)*24:
        raise RuntimeError("Inverter representation is incomplete")
    provenance={**inverter_provenance(parameters),"source_sha256_before":before,"source_sha256_after":after,
        "sources_unchanged":True,"configuration":asdict(config),"k_dc":None,"dc_loss_layer_applied":False,
        "stage":"IDEAL DC -> Sandia -> diagnostic NO-LOSS AC REFERENCE; not calibrated",
        "records":records,"reference_computed_timestamps":int(reference.no_loss_ac_computed.sum()),
        "measured_ac_units":"confirmed prepared kW converted to W only in diagnostic results",
        "alignment":"exact original Pacific local wall-clock timestamps; no shift/resampling/UTC conversion",
        "daylight_analysis":"POA>0 and computed reference, existing electrical/irradiance/DST QC flags clear; not healthy-period selection",
        "operating_analysis":"same selection and measured AC>0; outages/abnormal operation can still remain",
        "night_analysis":"POA=0, computed reference and same clear flags; negative Sandia tare preserved separately",
        "upper_region_policy":"all finite telemetry and electrical-QC-clear separately; 100 W bins, >=12 observations, exact 5-minute runs",
        "residual":"measured - no-loss expected AC [W]","ratio":"measured/no-loss expected only when expected>0",
        "clipping":"solely unchanged pvlib Sandia; no additional cap or inferred Paco",
        "plant_aggregation":"none","meter_data":"not used","calibration":"none"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(json.dumps({k:provenance[k] for k in ("records","reference_computed_timestamps","k_dc","sources_unchanged")}),flush=True)


if __name__=="__main__":
    main()
