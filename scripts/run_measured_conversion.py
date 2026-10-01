"""Cleaned electrical V×I -> measured AC / unchanged Sandia; no PV-model inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oedi2107.ac_reference import measured_upper_region
from oedi2107.config import SystemConfig
from oedi2107.inverter import load_cec_inverter, inverter_provenance
from oedi2107.measured_conversion import measured_conversion, conversion_metrics, dc_loading_summary
from oedi2107.real_data import discover_channels


def digest(path):
    hasher=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    source=Path("data/2107/processed/2107_electrical_clean.parquet")
    qc_path=Path("data/2107/processed/2107_electrical_qc_events.parquet")
    meta_path=Path("data/2107/raw/2107_system_metadata.json")
    output=Path("outputs/2107_measured_conversion")
    output.mkdir(parents=True,exist_ok=True)
    paths=[source,qc_path,meta_path]
    before={str(p):digest(p) for p in paths}
    metadata=json.loads(meta_path.read_text(encoding="utf-8-sig"))
    mapping=discover_channels(pq.ParquetFile(source).schema_arrow.names)
    parameters=load_cec_inverter(SystemConfig().cec_inverter_entry)
    original=parameters.copy(deep=True)
    flags=set(pd.read_parquet(qc_path).measured_on)
    metrics,bins,bands,upper,efficiency,coverage,plot_samples=[],[],[],[],[],[],[]
    writer=None
    try:
        for channel in mapping.to_dict("records"):
            identifier=channel["inverter_id"]
            if identifier=="inv_05":
                continue
            for quantity,unit in (("dc_voltage","v"),("dc_current","a"),("ac_power","kw")):
                metric=channel[quantity].rsplit("_",1)[-1]
                records=[r for r in metadata["Metrics"].values() if str(r.get("metric_id"))==metric]
                if len(records)!=1 or records[0]["units"].lower()!=unit or records[0]["calc_scale"]!=1 or records[0]["calc_offset"]!=0:
                    raise ValueError(f"Unconfirmed electrical units for {channel[quantity]}")
            raw=pd.read_parquet(source,columns=["measured_on",channel["dc_voltage"],channel["dc_current"],channel["ac_power"]])
            index=pd.DatetimeIndex(pd.to_datetime(raw.measured_on),name="timestamp")
            readings=pd.DataFrame({"measured_dc_voltage":raw[channel["dc_voltage"]].to_numpy(),
                "measured_dc_current":raw[channel["dc_current"]].to_numpy(),"measured_ac_power":raw[channel["ac_power"]].to_numpy()*1000,
                "electrical_qc_clear":~raw.measured_on.isin(flags).to_numpy(),"inverter_id":identifier},index=index)
            table=measured_conversion(readings,parameters)
            table.attrs={}
            arrow=pa.Table.from_pandas(table.reset_index(),preserve_index=False)
            if writer is None:
                writer=pq.ParquetWriter(output/"inverter_measured_conversion.parquet",arrow.schema,compression="snappy")
            writer.write_table(arrow,row_group_size=100000)
            mask=table.conversion_analysis_mask
            unclipped=mask & table.measured_dc_power.lt(.9*float(parameters.Pdco)) & table.measured_ac_power.lt(.9*float(parameters.Paco))
            for label,selection in (("main",mask),("below_rating_guards",unclipped)):
                metrics.append({"inverter_id":identifier,"population":label,**conversion_metrics(table,selection)})
            bins.append(dc_loading_summary(table,pdco=float(parameters.Pdco)).assign(inverter_id=identifier))
            for label,low,high in (("around_27_6kw",27500.,27700.),("around_30kw",29900.,30100.)):
                subset=table.loc[mask & table.measured_ac_power.ge(low) & table.measured_ac_power.lt(high)]
                bands.append({"inverter_id":identifier,"ac_band":label,"records":len(subset),"median_dc_w":subset.measured_dc_power.median(),
                    "median_ac_w":subset.measured_ac_power.median(),"median_sandia_w":subset.sandia_ac_from_measured_dc.median(),
                    "median_raw_ratio":subset.ac_dc_power_ratio.median(),"median_efficiency":subset.efficiency.median(),
                    "ratio_gt_1_count":int(subset.ac_dc_power_ratio.gt(1).sum()),"median_residual_w":subset.residual_ac_power.median()})
            upper.append({"inverter_id":identifier,**measured_upper_region(table.loc[table.electrical_qc_clear,"measured_ac_power"],paco=float(parameters.Paco))})
            selected=table.loc[mask]
            efficiency.append({"inverter_id":identifier,"records":len(selected),"ratio_gt_1_count":int(selected.ac_dc_power_ratio.gt(1).sum()),
                "raw_ratio_median":selected.ac_dc_power_ratio.median(),"physical_efficiency_min":selected.efficiency.min(),
                "physical_efficiency_p10":selected.efficiency.quantile(.1),"physical_efficiency_median":selected.efficiency.median(),
                "physical_efficiency_p90":selected.efficiency.quantile(.9),"physical_efficiency_max":selected.efficiency.max()})
            coverage.append({"inverter_id":identifier,"total_rows":len(table),"finite_dc_power":int(np.isfinite(table.measured_dc_power).sum()),
                "positive_day_qc":int(table.positive_day_qc.sum()),"above_power_floors":int(mask.sum()),"excluded_low_power":int((table.positive_day_qc & ~mask).sum())})
            positions=np.flatnonzero(mask.to_numpy())
            positions=positions[::max(1,len(positions)//2500)]
            plot_samples.append(table.iloc[positions][["measured_dc_power","measured_ac_power","sandia_ac_from_measured_dc","ac_dc_power_ratio","residual_ac_power","inverter_id"]])
            print(identifier,metrics[-1],flush=True)
    finally:
        if writer is not None:
            writer.close()
    for name,records in (("ac_error",metrics),("high_ac_bands",bands),("upper_regions",upper),("efficiency",efficiency),("coverage",coverage)):
        pd.DataFrame(records).to_csv(output/f"{name}.csv",index=False)
    pd.concat(bins,ignore_index=True).to_csv(output/"dc_loading_bins.csv",index=False)
    points=pd.concat(plot_samples,ignore_index=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    axes[0,0].scatter(points.measured_dc_power/1000,points.measured_ac_power/1000,s=2,alpha=.15,label="Measured")
    axes[0,0].scatter(points.measured_dc_power/1000,points.sandia_ac_from_measured_dc/1000,s=2,alpha=.15,label="Sandia at same measured DC")
    axes[0,0].set(xlim=(0,40),ylim=(0,33),xlabel="Measured DC [kW]",ylabel="AC [kW]")
    axes[0,0].legend()
    axes[0,1].scatter(points.measured_dc_power/1000,points.ac_dc_power_ratio,s=2,alpha=.15)
    axes[0,1].set(xlim=(0,40),ylim=(.7,1.05),xlabel="Measured DC [kW]",ylabel="Raw AC/DC ratio; >1 is not physical efficiency")
    axes[0,1].axhline(1,color="black",linewidth=.7)
    axes[1,0].scatter(points.measured_dc_power/1000,points.residual_ac_power/1000,s=2,alpha=.15)
    axes[1,0].set(xlim=(0,40),ylim=(-5,5),xlabel="Measured DC [kW]",ylabel="Measured − Sandia AC [kW]")
    axes[1,0].axhline(0,color="black",linewidth=.7)
    axes[1,1].scatter(points.measured_dc_power/1000,points.measured_ac_power/1000,s=2,alpha=.15)
    axes[1,1].scatter(points.measured_dc_power/1000,points.sandia_ac_from_measured_dc/1000,s=2,alpha=.15)
    axes[1,1].set(xlim=(27,36),ylim=(26,31),xlabel="Measured DC [kW]",ylabel="AC ceiling zoom [kW]")
    axes[1,1].axhline(float(parameters.Paco)/1000,color="red",linewidth=.7)
    fig.suptitle("23 inverters; cleaned measured electrical only; no calibrated parameters. Axis limits are display only.")
    fig.tight_layout()
    fig.savefig(output/"measured_conversion.png",dpi=150)
    plt.close(fig)
    after={str(p):digest(p) for p in paths}
    if before!=after:
        raise RuntimeError("Measured source files changed")
    pd.testing.assert_series_equal(parameters,original)
    provenance={**inverter_provenance(parameters),"sources_before":before,"sources_after":after,"sources_unchanged":True,
        "excluded_inverter":"inv_05: no measured DC voltage","inputs":"cleaned electrical V, A, AC kW only; V*A=W; kW*1000=W",
        "daytime":"local wall-clock 06<=hour<20 plus positive V/I/AC; broad proxy, not astronomical proof",
        "floors":{"dc_w":.01*float(parameters.Pdco),"ac_w":.01*float(parameters.Paco),"rationale":"1% official ratings avoid very-low-power ratio instability"},
        "qc":"exclude all six timestamps in existing electrical QC ledger from analyses; keep original values in output",
        "efficiency":"raw AC/DC ratio retained for main mask; physical efficiency only 0<ratio<=1; inconsistencies reported, not clipped",
        "below_rating_guards":"measured DC<90% Pdco and measured AC<90% Paco; avoids clipping; not a healthy mask",
        "k_dc":None,"production_model_changed":False,"upstream_physics_used":False,"calibration":"none"}
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
