"""Focused seven-offset audit using frozen candidates and unchanged prepared POA."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oedi2107.config import SystemConfig
from oedi2107.module import load_cec_module
from oedi2107.poa_timing import OFFSETS_MINUTES, offset_expected_current, summarize_current_ratio
from oedi2107.real_data import discover_channels


def digest(path):
    hasher=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,default=Path("data/2107/processed/2107_model_5min.parquet"))
    parser.add_argument("--context",type=Path,default=Path("outputs/2107_dc_bias/diagnostic_context.parquet"))
    parser.add_argument("--metadata",type=Path,default=Path("data/2107/raw/2107_system_metadata.json"))
    parser.add_argument("--output",type=Path,default=Path("outputs/2107_poa_timing_audit"))
    args=parser.parse_args()
    notebooks=[Path("notebooks")/f"{name}.ipynb" for name in ("02_irradiance_data_audit","05_dataset_alignment","06_build_aligned_dataset")]
    sources=[args.prepared,args.context,args.metadata,*notebooks]
    before={str(path):digest(path) for path in sources}
    metadata=json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    mapping=discover_channels(pq.ParquetFile(args.prepared).schema_arrow.names)
    for name in mapping.dc_current:
        if metadata["Metrics"][name]["units"] != "A":
            raise ValueError(f"Unconfirmed measured-current units: {name}")
    prepared=pd.read_parquet(args.prepared,columns=["measured_on","poa_w_m2","irradiance_qc_flag","dst_fall_ambiguous",*mapping.dc_current])
    prepared.index=pd.DatetimeIndex(pd.to_datetime(prepared.measured_on),name="timestamp")
    context=pd.read_parquet(args.context)
    candidate=context.loc[context.candidate_healthy_mask].copy()
    if not prepared.index.is_unique or not candidate.index.is_unique:
        raise ValueError("Source/candidate timestamps must be unique")
    np.testing.assert_allclose(prepared.poa_w_m2.reindex(candidate.index),candidate.poa_w_m2,rtol=0,atol=0)
    current=prepared.loc[candidate.index,mapping.dc_current].copy()
    current.columns=mapping.inverter_id.tolist()
    np.testing.assert_allclose(np.median(current.to_numpy(),axis=1),candidate.measured_dc_current_median,rtol=0,atol=0)
    config=SystemConfig()
    if config.dc_losses.k_dc is not None:
        raise ValueError("K_DC must remain unset")
    parameters=load_cec_module(config.cec_module_entry)
    original_parameters=parameters.copy(deep=True)
    source_qc=prepared.irradiance_qc_flag | prepared.dst_fall_ambiguous
    expected=offset_expected_current(candidate,prepared.poa_w_m2,source_qc,parameters,parallel_strings=config.strings_per_inverter)
    np.testing.assert_allclose(expected[0],candidate.expected_dc_current_ideal,rtol=1e-10,atol=1e-8)
    pd.testing.assert_series_equal(parameters,original_parameters)
    # Equal support prevents different POA gaps/QC exclusions from selecting a winning lag.
    common=np.isfinite(expected).all(axis=1) & expected.gt(0).all(axis=1) & candidate.solar_phase.isin(["AM","PM"])
    rows=[]
    availability=[]
    ratios=pd.DataFrame(index=candidate.index)
    for offset in OFFSETS_MINUTES:
        ratio=candidate.measured_dc_current_median / expected[offset]
        ratios[offset]=ratio
        rows.append({"offset_minutes":offset,**summarize_current_ratio(ratio.loc[common],candidate.loc[common,"solar_phase"])})
        availability.append({"offset_minutes":offset,**summarize_current_ratio(ratio,candidate.solar_phase)})
    scores=pd.DataFrame(rows)
    # Best within the authorized finite set by smallest absolute AM-PM gap only.
    best=int(scores.loc[scores.am_minus_pm.abs().idxmin(),"offset_minutes"])
    args.output.mkdir(parents=True,exist_ok=True)
    scores.to_csv(args.output/"offset_comparison.csv",index=False)
    pd.DataFrame(availability).to_csv(args.output/"offset_individual_coverage.csv",index=False)
    tails=[]
    for offset in OFFSETS_MINUTES:
        ratio=ratios.loc[common,offset]
        tails.append({"offset_minutes":offset,"minimum":ratio.min(),"p01":ratio.quantile(.01),
                      "p99":ratio.quantile(.99),"maximum":ratio.max(),"ratio_gt_2_count":int(ratio.gt(2).sum())})
    pd.DataFrame(tails).to_csv(args.output/"ratio_tail_summary.csv",index=False)
    expected.rename(columns=lambda x:f"diagnostic_expected_current_offset_{x}").join(common.rename("common_support")).to_parquet(args.output/"offset_expected_current.parquet")
    grouped=[]
    per_inverter=[]
    simultaneous=[]
    for offset in OFFSETS_MINUTES:
        ratio=ratios[offset].loc[common]
        selected=candidate.loc[common]
        for dimension,labels in (("year",pd.Series(selected.index.year,index=selected.index)),("season",selected.season)):
            for label,index in labels.groupby(labels).groups.items():
                grouped.append({"offset_minutes":offset,"dimension":dimension,"bin":str(label),
                    **summarize_current_ratio(ratio.loc[index],selected.loc[index,"solar_phase"])})
        all_ratios=current.loc[common].div(expected.loc[common,offset],axis=0)
        for identifier in all_ratios:
            per_inverter.append({"offset_minutes":offset,"inverter_id":identifier,
                **summarize_current_ratio(all_ratios[identifier],selected.solar_phase)})
        # Same-time sign agreement is descriptive; no inverter fault threshold is introduced.
        fractions=all_ratios.gt(1).mean(axis=1)
        for phase in ("AM","PM"):
            selected_phase=selected.solar_phase.eq(phase)
            simultaneous.append({"offset_minutes":offset,"solar_phase":phase,"timestamps":int(selected_phase.sum()),
                "median_fraction_inverters_ratio_gt_1":float(fractions.loc[selected_phase].median())})
    pd.DataFrame(grouped).to_csv(args.output/"year_season_offsets.csv",index=False)
    pd.DataFrame(per_inverter).to_csv(args.output/"inverter_offsets.csv",index=False)
    pd.DataFrame(simultaneous).to_csv(args.output/"simultaneous_inverter_direction.csv",index=False)
    poa_records={name:record for name,record in metadata["Metrics"].items() if "poa" in name.lower()}
    audit={"poa_metrics":poa_records,"other_instruments":metadata["Other Instruments"],"pv_mount":metadata["Mount"],
        "site":metadata["Site"],"system_timezone":{"code":metadata["System"]["timezone_code"],"comments":metadata["System"]["comments"]},
        "inverter_example":next(iter(metadata["Inverters"].values())),"current_metric_example":metadata["Metrics"][mapping.dc_current.iloc[0]],
        "findings":{"poa_sensor_tilt_azimuth":"not provided; array mount is not sensor orientation",
                    "sensor_identity_location":"metric 149574 identified; instrument source_id null; manufacturer/model/individual location unknown",
                    "timestamp_semantics":"averaged metrics, but averaging support/start/end/clock synchronization undocumented; time_interval L not defined here",
                    "logging_systems":"OTHER versus INVERTER sources, separate exports and different gaps; no logger identity or clock-offset evidence",
                    "prepared_provenance":"notebook 06 exact POA timestamp join, no interpolation/shift/UTC conversion; fall ambiguity preserved",
                    "prior_alignment":"notebook 05 favors zero POA-to-AC offset overall and each year; not rerun"}}
    (args.output/"metadata_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    after={str(path):digest(path) for path in sources}
    if before != after:
        raise RuntimeError("Audit inputs changed")
    provenance={"source_sha256_before":before,"source_sha256_after":after,"sources_unchanged":before==after,
        "offsets_minutes":OFFSETS_MINUTES,"offset_sign":"At electrical t evaluate measured POA at t+offset; positive is later POA label",
        "candidate_count":len(candidate),"equal_support_count":int(common.sum()),"excluded_from_equal_support":int((~common).sum()),
        "dispersion":"sample standard deviation ddof=1 and IQR over timestamp plant-median current ratios",
        "temperature":"original estimated cell temperature fixed at electrical t; no thermal sensitivity or Faiman rerun",
        "source_qc":"POA source lookup excludes existing irradiance QC/DST flags, nonfinite and nonpositive POA; no new cleaning",
        "best_offset_minutes_by_gap":best,"best_policy":"smallest absolute AM-PM median gap in the seven supplied offsets; not a clock calibration",
        "k_dc":None,"model_parameters":"unchanged","timestamp_and_poa_policy":"exact lookups only; original indices/values unmodified"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    lines=["# Focused POA timing audit","","Positive offset uses POA(t+offset) for electrical t; original data unchanged.",
        f"Equal support: {int(common.sum())} of {len(candidate)} frozen candidates. Temperature remains fixed.","","```",
        scores.to_string(index=False),"```","",f"Smallest absolute AM-PM gap within the fixed set: {best:+d} minutes.",
        "This descriptive ranking does not identify a logger clock correction. See metadata_audit.json and year/season/inverter comparisons."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(scores.to_string(index=False),flush=True)
    print(json.dumps({k:provenance[k] for k in ("equal_support_count","excluded_from_equal_support","best_offset_minutes_by_gap","sources_unchanged")}),flush=True)


if __name__ == "__main__":
    main()
