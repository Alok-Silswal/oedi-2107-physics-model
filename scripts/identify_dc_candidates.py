"""Screen existing ideal DC results; descriptive diagnostics only, never fitting."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oedi2107.config import SystemConfig
from oedi2107.dc_candidates import CandidateThresholds, candidate_features, select_candidates
from oedi2107.inverter import load_cec_inverter


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_measurements(comparison, baseline, prepared, mapping):
    """Existing comparison and cleaned AC -> exact timestamp/ID mapping -> V/A/W matrices."""
    ids = mapping.inverter_id.tolist()
    dc = {q: pd.DataFrame(np.nan, index=baseline.index, columns=ids) for q in ("voltage", "current", "power")}
    seen = np.zeros((len(baseline), len(ids)), dtype=bool)
    columns = ["timestamp", "inverter_id", *[f"measured_dc_{q}" for q in dc]]
    for batch in pq.ParquetFile(comparison).iter_batches(batch_size=100000, columns=columns):
        table = batch.to_pandas()
        for identifier, group in table.groupby("inverter_id", sort=False):
            if identifier not in ids:
                raise ValueError(f"Unexpected inverter ID: {identifier}")
            column = ids.index(identifier)
            positions = baseline.index.get_indexer(group.timestamp)
            if (positions < 0).any() or len(np.unique(positions)) != len(positions) or seen[positions, column].any():
                raise ValueError("DC comparison timestamps are missing, duplicated or incompatible")
            seen[positions, column] = True
            for quantity in dc:
                dc[quantity].iloc[positions, column] = group[f"measured_dc_{quantity}"].to_numpy()
    if not seen.all():
        raise ValueError("Existing DC comparison must preserve every timestamp and inverter ID")
    source = pd.read_parquet(prepared, columns=["measured_on", *mapping.ac_power])
    timestamps = pd.DatetimeIndex(pd.to_datetime(source.measured_on), name="timestamp")
    if not timestamps.equals(baseline.index):
        raise ValueError("Prepared AC and existing ideal DC timestamps must match exactly")
    ac = pd.DataFrame(source[mapping.ac_power].to_numpy(dtype=float) * 1000, index=baseline.index, columns=ids)
    return dc, ac


def summarize(frame, columns):
    """Descriptive finite distributions, not estimated loss factors."""
    return frame[columns].describe(percentiles=[.1, .25, .5, .75, .9]).T.rename_axis("quantity").reset_index()


def grouped_ratios(frame, groups):
    rows = []
    for name, labels in groups.items():
        for label, subset in frame.groupby(labels, observed=True):
            record = {"dimension": name, "bin": str(label), "timestamps": len(subset)}
            for quantity in ("voltage", "current", "power"):
                series = subset[f"{quantity}_ratio"]
                record.update({f"{quantity}_p10": series.quantile(.1), f"{quantity}_median": series.median(), f"{quantity}_p90": series.quantile(.9)})
            rows.append(record)
    return pd.DataFrame(rows)


def plots(features, preliminary, candidate, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    xvalues = {"POA [W/m²]": features.poa_w_m2, "Estimated cell temperature [°C]": features.temp_cell,
               "Pacific wall-clock hour": pd.Series(features.index.hour + features.index.minute / 60, index=features.index),
               "Year": pd.Series(features.index.year, index=features.index)}
    for row, quantity in enumerate(("voltage", "current", "power")):
        for col, (label, x) in enumerate(xvalues.items()):
            ax = axes[row, col]
            for mask, color, name in ((preliminary, "gray", "Before MPP/consensus/ramp"), (candidate, "tab:blue", "Candidates")):
                positions = np.flatnonzero(mask.to_numpy())
                positions = positions[::max(1, len(positions) // 6000)]
                ax.scatter(x.iloc[positions], features[f"{quantity}_ratio"].iloc[positions], s=2, alpha=.25, color=color, label=name)
            ax.set_xlabel(label)
            ax.set_ylabel(f"Measured/ideal {quantity} (plant median)")
            ax.set_ylim(.5, 1.3)
            ax.axhline(1, color="black", linewidth=.6)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Candidate screens only; no fitted K_DC. Plot y-limits are display limits; CSVs retain all ratios.")
    fig.tight_layout()
    fig.savefig(output / "ratio_dependencies.png", dpi=150)
    plt.close(fig)
    counts = pd.DataFrame({"preliminary": preliminary.astype(int), "candidate": candidate.astype(int)}).resample("MS").sum()
    counts.to_csv(output / "monthly_candidate_counts.csv", index_label="month")
    # Calendar summaries count rows only; electrical power is never resampled/aligned.
    counts.plot(figsize=(12, 4), ylabel="Number of 5-minute timestamps")
    plt.tight_layout()
    plt.savefig(output / "candidate_coverage.png", dpi=150)
    plt.close()
    # Pick the most-covered candidate date within named years, solely for review plots.
    selected = features.loc[candidate]
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    review_dates = []
    for col, year in enumerate((2018, 2021, 2022)):
        covered = selected.loc[selected.index.year == year]
        if covered.empty:
            continue
        date = covered.groupby(covered.index.date).size().idxmax()
        review_dates.append(str(date))
        day = features.loc[str(date)]
        hour = day.index.hour + day.index.minute / 60
        for row, quantity in enumerate(("voltage", "current", "power")):
            ax = axes[row, col]
            ax.plot(hour, day[f"expected_dc_{quantity}_ideal"], label="Ideal", linewidth=1)
            ax.plot(hour, day[f"measured_dc_{quantity}_median"], label="Measured plant median", linewidth=1)
            kept = candidate.loc[day.index]
            ax.scatter(hour[kept], day.loc[kept,f"measured_dc_{quantity}_median"], s=8, color="green", label="Candidate")
            ax.set_title(str(date))
            ax.set_ylabel(f"DC {quantity} [{ {'voltage':'V','current':'A','power':'W'}[quantity] }]")
            ax.set_xlabel("Pacific wall-clock hour")
    axes[0,0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "representative_candidate_days.png", dpi=140)
    plt.close(fig)
    (output / "review_plot_dates.json").write_text(json.dumps(review_dates), encoding="utf-8")


def write_report(features, masks, counts, thresholds, output):
    """Saved diagnostic quantities -> concise text report -> review artifact, no fitting."""
    candidate = masks.candidate_healthy_mask
    report = ["# DC candidate diagnostics", "", "Candidates only; no verified-health claim, calibration or K_DC value.",
              "Source hashes unchanged. See ../../docs/dc_candidate_screening.md for rationale and limitations.", "",
              "## Filtering", "", "```", counts.to_string(index=False), "```", "", "## Explicit thresholds", "",
              "```json", json.dumps(asdict(thresholds), indent=2), "```", "", "## Candidate ratios", ""]
    for quantity in ("voltage", "current", "power"):
        ratio = features.loc[candidate, f"{quantity}_ratio"]
        report.append(f"- {quantity}: p10/median/p90 = {ratio.quantile(.1):.4f}/{ratio.median():.4f}/{ratio.quantile(.9):.4f}")
    report.extend(["", f"Power-ratio > 1 timestamps: {int(features.loc[candidate,'power_ratio'].gt(1).sum())}.", "",
                   "A constant global loss factor is not demonstrated. Review weather/time dependencies and calendar coverage.",
                   "Consensus cannot remove common-mode abnormal operation or common irradiance/thermal bias.",
                   "Threshold sensitivity and joint-condition CSVs retain full descriptive evidence; no parameter is estimated."])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dc-results", type=Path, default=Path("outputs/2107_dc_ideal"))
    parser.add_argument("--prepared", type=Path, default=Path("data/2107/processed/2107_model_5min.parquet"))
    parser.add_argument("--metadata", type=Path, default=Path("data/2107/raw/2107_system_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/2107_dc_candidates"))
    args = parser.parse_args()
    sources = [args.prepared, args.metadata, *[args.dc_results / name for name in ("nominal_dc_ideal.parquet", "inverter_dc_ideal.parquet", "channel_mapping.csv")]]
    before = {str(path): digest(path) for path in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(args.dc_results / "nominal_dc_ideal.parquet")
    mapping = pd.read_csv(args.dc_results / "channel_mapping.csv")
    expected_ids = [f"inv_{i:02d}" for i in range(1, 25)]
    if mapping.inverter_id.tolist() != expected_ids:
        raise ValueError("Source mapping must contain ordered unique inverter IDs 01–24")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    for channel in mapping.ac_power:
        metric = channel.rsplit("_", 1)[-1]
        records = [record for record in metadata["Metrics"].values() if str(record.get("metric_id")) == metric]
        if len(records) != 1 or str(records[0].get("units")).lower() != "kw":
            raise ValueError(f"Cannot verify cleaned measured AC kW units: {channel}")
    print("Loading existing DC comparison and cleaned AC telemetry", flush=True)
    dc, ac = read_measurements(args.dc_results / "inverter_dc_ideal.parquet", baseline, args.prepared, mapping)
    features = candidate_features(baseline, dc["voltage"], dc["current"], dc["power"], ac)
    config = SystemConfig()
    if config.dc_losses.k_dc is not None:
        raise ValueError("This diagnostic stage requires K_DC to remain unset")
    inverter = load_cec_inverter(config.cec_inverter_entry)
    thresholds = CandidateThresholds()
    masks, counts = select_candidates(features, paco=float(inverter.Paco), pdco=float(inverter.Pdco), thresholds=thresholds)
    candidate = masks.candidate_healthy_mask
    preliminary = masks.iloc[:, :6].all(axis=1)
    masks.to_parquet(args.output / "timestamp_candidate_masks.parquet")
    features.to_parquet(args.output / "timestamp_diagnostics.parquet")
    counts.to_csv(args.output / "filter_counts.csv", index=False)
    print(counts.to_string(index=False), flush=True)
    columns = ["poa_w_m2", "ambient_temperature_c", "temp_cell", *[f"expected_dc_{q}_ideal" for q in dc],
               *[f"measured_dc_{q}_median" for q in dc], "voltage_ratio", "current_ratio", "power_ratio", "maximum_ac_to_dc_power_ratio"]
    for label, mask in (("preliminary", preliminary), ("candidate", candidate)):
        summarize(features.loc[mask], columns).to_csv(args.output / f"{label}_distributions.csv", index=False)
        subset = features.loc[mask]
        groups = {"poa_w_m2": pd.cut(subset.poa_w_m2, [0, 200, 300, 400, 500, 600, 700, 800, 1000, 1500], right=False),
                  "temp_cell_c": pd.cut(subset.temp_cell, [-20, 0, 10, 20, 30, 40, 50, 60, 80], right=False),
                  "hour_local": subset.index.hour, "year": subset.index.year,
                  "season": pd.Series(subset.index.month, index=subset.index).map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})}
        grouped_ratios(subset, groups).to_csv(args.output / f"{label}_ratio_dependencies.csv", index=False)
        joint = subset.assign(poa_bin=pd.cut(subset.poa_w_m2, [300,400,500,600,700,800,1500], right=False),
                              temperature_bin=pd.cut(subset.temp_cell, [-20,10,20,30,40,50,60,80], right=False))
        joint.groupby(["poa_bin", "temperature_bin"], observed=True).agg(timestamps=("power_ratio", "size"),
            power_ratio_median=("power_ratio", "median"), voltage_ratio_median=("voltage_ratio", "median"),
            current_ratio_median=("current_ratio", "median")).to_csv(args.output / f"{label}_poa_temperature.csv")
        subset.assign(poa_bin=joint.poa_bin, year=subset.index.year).groupby(["year", "poa_bin"], observed=True).agg(
            timestamps=("power_ratio", "size"), power_ratio_median=("power_ratio", "median")).to_csv(args.output / f"{label}_year_poa.csv")
        joint.assign(hour=joint.index.hour).groupby(["poa_bin", "hour"], observed=True).agg(
            timestamps=("power_ratio", "size"), power_ratio_median=("power_ratio", "median"),
            temp_cell_median=("temp_cell", "median")).to_csv(args.output / f"{label}_hour_poa.csv")
        joint.assign(hour=joint.index.hour, year=joint.index.year).groupby(["year", "poa_bin", "temperature_bin", "hour"], observed=True).agg(
            timestamps=("power_ratio", "size"), power_ratio_median=("power_ratio", "median")).to_csv(args.output / f"{label}_year_poa_temperature_hour.csv")
    sensitivity = []
    for name, values in {"minimum_poa": [200,300,400], "minimum_ac_fraction": [.05,.10,.15], "rating_margin": [.80,.90,.95],
                         "voltage_tolerance": [.05,.10,.15], "consensus_tolerance": [.05,.10,.15,.20], "maximum_poa_ramp": [.10,.20,.30,None]}.items():
        for value in values:
            alternative, _ = select_candidates(features, paco=float(inverter.Paco), pdco=float(inverter.Pdco), thresholds=replace(thresholds, **{name:value}))
            selected = alternative.candidate_healthy_mask
            sensitivity.append({"threshold":name, "value":str(value), "timestamps":int(selected.sum()),
                                "voltage_ratio_median":features.loc[selected,"voltage_ratio"].median(),
                                "current_ratio_median":features.loc[selected,"current_ratio"].median(),
                                "power_ratio_median":features.loc[selected,"power_ratio"].median()})
    pd.DataFrame(sensitivity).to_csv(args.output / "threshold_sensitivity.csv", index=False)
    per_inverter = []
    consensus_medians = {"current":features.measured_dc_current_median, "ac":features.measured_ac_power_median_w}
    for identifier in expected_ids:
        row = {"inverter_id":identifier, "candidate_timestamps":int(candidate.sum())}
        for quantity, table in dc.items():
            measured = table[identifier]
            expected = features[f"expected_dc_{quantity}_ideal"]
            valid = candidate & np.isfinite(measured) & expected.gt(0)
            ratio = measured[valid] / expected[valid]
            row.update({f"{quantity}_valid_pairs":int(valid.sum()), f"{quantity}_ratio_p10":ratio.quantile(.1),
                        f"{quantity}_ratio_median":ratio.median(), f"{quantity}_ratio_p90":ratio.quantile(.9)})
        # Diagnostics against a shared instantaneous median, not individualized health labels.
        for name, table in (("current",dc["current"]),("ac",ac)):
            median = consensus_medians[name]
            deviation = (table[identifier] - median).abs() / median
            row[f"{name}_disagreement_fraction_preliminary"] = float(deviation[preliminary].gt(thresholds.consensus_tolerance).mean())
        per_inverter.append(row)
    pd.DataFrame(per_inverter).to_csv(args.output / "inverter_candidate_coverage.csv", index=False)
    # Explicit identity-ready masks for later external healthy_mask use; voltage is never reconstructed.
    import pyarrow as pa
    with pq.ParquetWriter(args.output / "inverter_candidate_masks.parquet", pa.Table.from_pandas(pd.DataFrame({
            "timestamp":features.index, "inverter_id":expected_ids[0], "candidate_healthy_mask":candidate.to_numpy(),
            "dc_power_eligible":candidate.to_numpy()}), preserve_index=False).schema) as writer:
        for identifier in expected_ids:
            writer.write_table(pa.Table.from_pandas(pd.DataFrame({"timestamp":features.index, "inverter_id":identifier,
                "candidate_healthy_mask":candidate.to_numpy(), "dc_power_eligible":candidate.to_numpy() & np.isfinite(dc["power"][identifier].to_numpy())}), preserve_index=False))
    selected_index = features.index[candidate]
    runs = pd.Series(selected_index, index=selected_index).to_frame("timestamp")
    if len(runs):
        run_id = runs.timestamp.diff().ne(pd.Timedelta(minutes=5)).cumsum()
        runs.groupby(run_id).agg(start=("timestamp","min"), last_timestamp=("timestamp","max"), samples=("timestamp","size")).to_csv(args.output / "candidate_runs.csv", index=False)
    plots(features, preliminary, candidate, args.output)
    after = {str(path): digest(path) for path in sources}
    if before != after:
        raise RuntimeError("Input files changed during diagnostic run")
    provenance = {"thresholds":asdict(thresholds), "cec_inverter_entry":config.cec_inverter_entry,
                  "paco_w":float(inverter.Paco), "pdco_w":float(inverter.Pdco), "ac_units":"cleaned kW -> W for screening only",
                  "timestamp_basis":"Pacific local wall clock; no timezone conversion; flagged DST excluded only from candidates",
                  "candidate_timestamps":int(candidate.sum()), "candidate_power_pairs":int(candidate.sum())*23,
                  "candidate_current_pairs":int(candidate.sum())*24, "k_dc":None, "calibration":"none",
                  "source_sha256_before":before,"source_sha256_after":after,"source_files_unchanged":before==after,
                  "model_configuration_unchanged":asdict(config), "status":"candidates, not confirmed healthy/MPP; no split selected"}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_report(features, masks, counts, thresholds, args.output)
    print(json.dumps({k:provenance[k] for k in ("candidate_timestamps","candidate_power_pairs","candidate_current_pairs","k_dc")}), flush=True)


if __name__ == "__main__":
    main()
