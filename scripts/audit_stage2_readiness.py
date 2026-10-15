"""Read existing records; write Stage 2 coverage and ramp-timing diagnostics only."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pvlib
import pyarrow.parquet as pq

OUT = Path("outputs/2107_stage2_readiness")
DATA = Path("data/2107/processed")
OFFSETS = (-15, -10, -5, 0, 5, 10, 15)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def indexed(path, columns=None, filters=None):
    frame = pd.read_parquet(path, columns=columns, filters=filters)
    if "measured_on" in frame:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("measured_on")), name="timestamp")
    return frame


def solar_elevation(index):
    local = index.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")
    result = pd.Series(np.nan, index=index)
    valid = ~local.isna()
    result.loc[valid] = pvlib.solarposition.get_solarposition(
        local[valid], 38.996306, -122.134111, altitude=10
    ).apparent_elevation.to_numpy()
    return result


def runs(mask, label, min_samples=12):
    """Boolean observations -> exact five-minute runs, without spanning gaps."""
    breaks = mask.ne(mask.shift()) | mask.index.to_series().diff().ne(pd.Timedelta(minutes=5))
    result = []
    for _, s in mask.groupby(breaks.cumsum()):
        if s.iloc[0] and len(s) >= min_samples:
            result.append(dict(kind=label, start=s.index[0], end=s.index[-1], samples=len(s),
                               nominal_hours=len(s) / 12))
    return result


def coverage(electrical, poa, nominal, clear):
    elevation = solar_elevation(electrical.index)
    daytime = elevation.gt(10)
    weather = nominal.model_available.reindex(electrical.index, fill_value=False)
    poa_ok = poa.notna() & clear
    lit = daytime & poa.ge(200) & clear
    ac_columns = [c for c in electrical if "_ac_power_" in c]
    rows, gaps, monthly = [], [], []
    peer = electrical[ac_columns].median(axis=1)
    gaps.extend(runs(daytime & poa.isna(), "missing_POA_during_solar_day"))
    gaps.extend(runs(poa.isna(), "missing_POA_all_clock_labels", min_samples=288))
    gaps.extend(runs(daytime & ~weather, "weather_or_baseline_unavailable_during_solar_day"))
    fleet_zero = electrical[ac_columns].eq(0).all(axis=1)
    gaps.extend(runs(daytime & fleet_zero, "all_24_AC_recorded_zero_during_solar_day"))
    expected = nominal.expected_ac_power.reindex(electrical.index) / 1000
    for column in ac_columns:
        inverter = column[:6]
        ac = electrical[column]
        valid = lit & ac.notna()
        matched = valid & weather
        unclip = matched & expected.between(3, 27, inclusive="left") & ac.between(0, 27, inclusive="left")
        peer_ok = valid & peer.between(3, 27, inclusive="left") & ac.between(0, 27, inclusive="left")
        for year in sorted(electrical.index.year.unique()):
            y = electrical.index.year == year
            denominator = int((daytime & y).sum())
            rows.append(dict(inverter_id=inverter, year=int(year), solar_day_labels=denominator,
                             poa_observed_qc_clear=int((daytime & poa_ok & y).sum()),
                             daytime_poa_ge200=int((lit & y).sum()),
                             ac_poa_pairs=int((valid & y).sum()), weather_ac_pairs=int((matched & y).sum()),
                             ac_zeros=int((valid & ac.eq(0) & y).sum()),
                             ac_ge29kw=int((valid & ac.ge(29) & y).sum()),
                             guarded_pairs=int((unclip & y).sum()),
                             weather_coverage_of_solar_day=(matched & y).sum() / denominator))
        gaps.extend({**r, "inverter_id": inverter} for r in runs(daytime & ac.isna(), "missing_AC"))
        ratios = pd.DataFrame({"weather_ratio": (ac / expected).where(unclip),
                               "peer_ratio": (ac / peer).where(peer_ok)})
        for month, group in ratios.groupby(ratios.index.to_period("M")):
            monthly.append(dict(inverter_id=inverter, month=str(month),
                                weather_pairs=int(group.weather_ratio.count()),
                                weather_days=int(group.weather_ratio.dropna().index.normalize().nunique()),
                                weather_ratio_median=group.weather_ratio.median(),
                                peer_pairs=int(group.peer_ratio.count()),
                                peer_days=int(group.peer_ratio.dropna().index.normalize().nunique()),
                                peer_ratio_median=group.peer_ratio.median()))
    pd.DataFrame(rows).to_csv(OUT / "coverage_year_inverter.csv", index=False)
    pd.DataFrame(gaps).sort_values("samples", ascending=False).to_csv(OUT / "daytime_runs.csv", index=False)
    months = pd.DataFrame(monthly)
    months.to_csv(OUT / "monthly_performance.csv", index=False)
    jumps = []
    for inverter, group in months.groupby("inverter_id"):
        group = group.set_index(pd.PeriodIndex(group.month, freq="M"))
        values = group.peer_ratio_median.where((group.peer_pairs >= 200) & (group.peer_days >= 10))
        for j in range(2, len(group) - 1):
            before, after = values.iloc[j-2:j], values.iloc[j:j+2]
            if before.notna().all() and after.notna().all():
                shift = after.median() - before.median()
                if abs(shift) >= .10 and max(before.max()-before.min(), after.max()-after.min()) <= .05:
                    jumps.append(dict(inverter_id=inverter, boundary=str(group.index[j]),
                                      peer_before=before.median(), peer_after=after.median(), shift=shift))
    pd.DataFrame(jumps, columns=["inverter_id", "boundary", "peer_before", "peer_after", "shift"]).to_csv(
        OUT / "persistent_peer_jumps.csv", index=False)
    weather_matched_years(electrical, poa, nominal, lit)
    return {"solar_day_labels": int(daytime.sum()), "poa_ge200_labels": int(lit.sum()),
            "weather_and_poa_ge200_labels": int((lit & weather).sum()), "persistent_peer_jump_candidates": len(jumps)}


def weather_matched_years(electrical, poa, nominal, lit):
    """Compare year with 2019 on shared calendar/weather bins; not ageing rates."""
    expected = nominal.expected_ac_power.reindex(electrical.index) / 1000
    temp = nominal.ambient_temperature_c.reindex(electrical.index)
    base = pd.DataFrame({"year": electrical.index.year, "month": electrical.index.month,
                         "hour_bin": electrical.index.hour // 2, "poa_bin": np.floor(poa / 100),
                         "ambient_bin": np.floor(temp / 5)})
    keys = ["month", "hour_bin", "poa_bin", "ambient_bin"]
    rows = []
    for c in [c for c in electrical if "_ac_power_" in c]:
        ac = electrical[c]
        valid = lit & expected.between(3, 27, inclusive="left") & ac.between(0, 27, inclusive="left") & temp.notna()
        d = base.loc[valid].copy()
        d["ratio"] = (ac / expected).loc[valid]
        bins = d.groupby(["year", *keys]).ratio.agg(["median", "count"])
        ref = bins.loc[2019]
        for year in sorted(set(d.year) - {2019}):
            pairs = bins.loc[year].join(ref, lsuffix="_year", rsuffix="_2019", how="inner")
            pairs = pairs.loc[(pairs.count_year >= 20) & (pairs.count_2019 >= 20)]
            weights = pairs[["count_year", "count_2019"]].min(axis=1)
            rows.append(dict(inverter_id=c[:6], year=int(year), reference_year=2019,
                             shared_bins=len(pairs), balanced_weight=int(weights.sum()),
                             ratio_year=np.average(pairs.median_year, weights=weights) if len(pairs) else np.nan,
                             ratio_2019=np.average(pairs.median_2019, weights=weights) if len(pairs) else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "year_weather_matched.csv", index=False)


def ramp_pairs(poa, response, valid, min_ramp):
    """Same exact observations for all offsets; score changes, not daily phase."""
    index = response.index
    previous = index - pd.Timedelta(minutes=5)
    delta = response - response.reindex(previous).to_numpy()
    common = valid & valid.reindex(previous, fill_value=False).to_numpy() & delta.abs().ge(min_ramp)
    ramps = {}
    for offset in OFFSETS:
        end = index + pd.Timedelta(minutes=offset)
        start = end - pd.Timedelta(minutes=5)
        p1, p0 = poa.reindex(end).to_numpy(), poa.reindex(start).to_numpy()
        common &= np.isfinite(p1) & np.isfinite(p0) & (p1 >= 100) & (p0 >= 100)
        ramps[offset] = p1 - p0
    selected = pd.DataFrame({"electrical_change": delta.loc[common]})
    for offset in OFFSETS:
        selected[f"poa_change_{offset}"] = ramps[offset][common.to_numpy()]
    return selected


def timing(electrical, poa, nominal, clear):
    ac = electrical[[c for c in electrical if "_ac_power_" in c]]
    current = electrical[[c for c in electrical if "_dc_current_" in c]]
    response = ac.median(axis=1)
    expected = nominal.expected_ac_power.reindex(electrical.index) / 1000
    valid = (clear & ac.notna().all(axis=1) & current.notna().all(axis=1)
             & current.min(axis=1).gt(0)
             & ac.min(axis=1).gt(2) & ac.max(axis=1).lt(27)
             & expected.between(3, 27, inclusive="left"))
    clean_poa = poa.where(clear)
    # Alternate complete dates within every year/month; held-out dates never select offsets.
    rows, events, fitted = [], [], {}
    for signal, series, threshold in [("AC_kW", response, 2), ("DC_current_A", current.median(axis=1), 3)]:
        pairs = ramp_pairs(clean_poa, series, valid, threshold)
        date_keys = pairs.index.normalize().unique().sort_values()
        date_frame = pd.DataFrame({"date": date_keys})
        date_frame["month"] = date_frame.date.dt.to_period("M")
        date_frame["split"] = np.where(date_frame.groupby("month").cumcount() % 2 == 0, "selection", "validation")
        split_map = date_frame.set_index("date").split
        pairs["split"] = split_map.reindex(pairs.index.normalize()).to_numpy()
        pairs["year"] = pairs.index.year
        pairs["season"] = np.array(["DJF", "MAM", "JJA", "SON"])[(pairs.index.month % 12) // 3]
        events.append(pairs.assign(signal=signal))
        gain = {}
        for offset in OFFSETS:
            selection = pairs.loc[pairs.split.eq("selection")]
            x, y = selection[f"poa_change_{offset}"], selection.electrical_change
            gain[offset] = max(0., float((x * y).sum() / (x * x).sum()))
        for split in ["selection", "validation"]:
            part = pairs.loc[pairs.split.eq(split)]
            groups = [("all", "all", part)]
            groups += [("year", str(k), g) for k, g in part.groupby("year")]
            groups += [("season", k, g) for k, g in part.groupby("season")]
            groups += [("direction", k, part.loc[part.electrical_change.gt(0).eq(k == "rise")]) for k in ["rise", "fall"]]
            for grouping, value, group in groups:
                for offset in OFFSETS:
                    x, y = group[f"poa_change_{offset}"], group.electrical_change
                    error = y - gain[offset] * x
                    rows.append(dict(signal=signal, split=split, grouping=grouping, value=value,
                                     offset_minutes=offset, samples=len(group), days=group.index.normalize().nunique(),
                                     gain_from_selection=gain[offset], correlation=x.corr(y),
                                     rmse=np.sqrt((error ** 2).mean()), mae=error.abs().mean(),
                                     direction_agreement=(np.sign(x) == np.sign(y)).mean()))
        scores = [r for r in rows if r["signal"] == signal and r["split"] == "selection" and r["grouping"] == "all"]
        chosen = min(scores, key=lambda r: r["rmse"])["offset_minutes"]
        fitted[signal] = {"selected_offset_minutes": chosen, "pairs": len(pairs),
                          "selection_days": int((date_frame.split == "selection").sum()),
                          "validation_days": int((date_frame.split == "validation").sum())}
    pd.DataFrame(rows).to_csv(OUT / "ramp_offset_scores.csv", index=False)
    pd.concat(events).to_parquet(OUT / "ramp_observations.parquet")
    return fitted


def forecast_support(electrical, poa, nominal, clear):
    """Count exact historical-feature/target pairs; never use future weather."""
    ac = electrical[[c for c in electrical if "_ac_power_" in c]]
    plant = ac.sum(axis=1, min_count=24)
    origins = clear & poa.ge(200)
    for minutes in (0, 5, 10, 15):
        lag = electrical.index - pd.Timedelta(minutes=minutes)
        origins &= plant.reindex(lag).notna().to_numpy()
        origins &= poa.reindex(lag).notna().to_numpy()
        origins &= clear.reindex(lag, fill_value=False).to_numpy()
    weather = nominal.model_available.reindex(electrical.index, fill_value=False)
    rows = []
    for horizon in (5, 15, 30, 60):
        target = electrical.index + pd.Timedelta(minutes=horizon)
        usable = origins & plant.reindex(target).notna().to_numpy() & clear.reindex(target, fill_value=False).to_numpy()
        for year in sorted(electrical.index.year.unique()):
            y = electrical.index.year == year
            rows.append(dict(year=int(year), horizon_minutes=horizon,
                             power_poa_history_pairs=int((usable & y).sum()),
                             plus_origin_weather_pairs=int((usable & weather & y).sum())))
    pd.DataFrame(rows).to_csv(OUT / "forecast_support.csv", index=False)


def equipment(nominal):
    from oedi2107.config import SystemConfig
    from oedi2107.module import calculate_module_dc, load_cec_module
    config = SystemConfig()
    row = load_cec_module(config.cec_module_entry)
    grid = pd.MultiIndex.from_product([[200., 1000.], [-40., -10., 0., 25., 65.]], names=["poa", "cell_temperature"])
    scenario = pd.DataFrame(index=grid)
    poa = pd.Series(grid.get_level_values(0), index=grid)
    temp = pd.Series(grid.get_level_values(1), index=grid)
    dc = calculate_module_dc(poa, temp, row)
    alternative = row.copy(); alternative["N_s"] = 72
    pd.testing.assert_frame_equal(dc, calculate_module_dc(poa, temp, alternative), check_exact=True)
    scenario["cec_string_voc"] = dc.Voc * 20
    scenario["cec_string_vmp"] = dc.Vmp * 20
    scenario["cec_three_string_imp"] = dc.Imp * 3
    scenario["cec_three_string_isc"] = dc.Isc * 3
    scenario["cec_three_string_power_w"] = dc.Pmp * 60
    scenario["datasheet_string_voc_linear"] = 20 * 45.3 * (1 - .0032 * (temp - 25))
    scenario.to_csv(OUT / "equipment_operating_scenarios.csv")
    # Extrema remain estimates, not measured cell temperature or design minima.
    actual = nominal.loc[nominal.model_available & nominal.poa_w_m2.gt(0)]
    moderate = actual.loc[actual.poa_w_m2.ge(200)]
    metadata = json.loads(Path("data/2107/raw/2107_system_metadata.json").read_text(encoding="utf-8-sig"))
    headers = {p.name: pd.read_csv(p, nrows=0).columns.tolist() for p in Path("data/2107/raw").glob("*.csv")}
    temp_metrics = {k: v for k, v in metadata["Metrics"].items() if "temp" in str(v).lower()}
    cec_file = Path(pvlib.__file__).parent / "data/sam-library-cec-modules-2019-03-05.csv"
    stc = dc.loc[(1000., 25.)].to_dict()
    summary = {
        "pvlib_version": pvlib.__version__, "module_key": config.cec_module_entry,
        "module_row": row.to_dict(), "changing_only_N_s_has_zero_prediction_effect": True,
        "cec_database_file": str(cec_file), "cec_database_sha256": digest(cec_file),
        "temperature_metrics": temp_metrics,
        "raw_temperature_columns": {k: [c for c in v if "temp" in c.lower()] for k, v in headers.items()},
        "raw_header_inventory": headers, "mount_metadata": metadata["Mount"],
        "stc_single_diode_solution": stc,
        "cec_relative_temperature_coefficients_percent_per_c": {
            "alpha_sc": float(row.alpha_sc / row.I_sc_ref * 100),
            "alpha_sc_after_Adjust": float(row.alpha_sc * (1 - row.Adjust / 100) / row.I_sc_ref * 100),
            "beta_oc": float(row.beta_oc / row.V_oc_ref * 100), "gamma_r": float(row.gamma_r)},
        "nominal_dc_per_inverter_w": 120 * 310,
        "dc_to_nominal_ac_ratio": 120 * 310 / 27600,
        "dc_to_maximum_ac_ratio": 120 * 310 / 30000,
        "datasheet_1000v_voc_threshold_cell_c": 25 + (1000 / (20 * 45.3) - 1) / -.0032,
        "estimated_cell_temperature_min_day": actual.temp_cell.min(),
        "estimated_cell_temperature_max_day": actual.temp_cell.max(),
        "ideal_string_voc_max": float((actual.Voc * 20).max()),
        "ideal_string_vmp_range": [actual.expected_dc_voltage_ideal.min(), actual.expected_dc_voltage_ideal.max()],
        "ideal_current_above_61_8a": int(actual.expected_dc_current_ideal.gt(61.8).sum()),
        "ideal_power_above_32000w": int(actual.expected_dc_power_ideal.gt(32000).sum()),
        "poa_ge200_ideal_vmp_below520v": int(moderate.expected_dc_voltage_ideal.lt(520).sum()),
    }
    (OUT / "equipment_checks.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    urls = {
        "hyundai_ti_dec2015.pdf": "https://s3.amazonaws.com/ecodirect_docs/HYUNDAI/Hyundai-TI-Series-solar-panel-datasheet_EN-340-350.pdf",
        "abb_trio_us_manual.pdf": "https://library.e.abb.com/public/9ddffa465aa146bfa43bd0854abbcc81/TRIO-20.0-27.6-TL-US_Product_Manual.pdf",
        "abb_trio_us_revg.pdf": "https://library.e.abb.com/public/7c240e9554df44a4bc77bcc2ecc59f96/TRIO-20.0-27.6-EN%20Rev%20G.pdf",
    }
    sources = [{"path": str(OUT / "sources" / name), "url": url,
                "sha256": digest(OUT / "sources" / name)} for name, url in urls.items()
               if (OUT / "sources" / name).exists()]
    (OUT / "manufacturer_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = [DATA / f"2107_{s}_clean.parquet" for s in ["electrical", "irradiance"]]
    source += [DATA / f"2107_{s}_qc_events.parquet" for s in ["electrical", "irradiance"]]
    source += [Path("outputs/2107_research_baseline/nominal_model.parquet"),
               Path("data/2107/raw/2107_system_metadata.json")]
    before = {str(p): digest(p) for p in source}
    baseline_sources = list(Path("outputs/2107_research_baseline").glob("*"))
    baseline_before = {str(p): digest(p) for p in baseline_sources if p.is_file()}
    names = pq.ParquetFile(source[0]).schema_arrow.names
    columns = ["measured_on"] + [c for c in names if "_ac_power_" in c or "_dc_current_" in c]
    electrical = indexed(source[0], columns)
    poa = indexed(source[1]).iloc[:, 0].reindex(electrical.index)
    nominal = indexed(source[4])
    flagged = pd.DatetimeIndex(pd.to_datetime(pd.concat([pd.read_parquet(p).measured_on for p in source[2:4]])))
    clear = pd.Series(~electrical.index.isin(flagged), index=electrical.index)
    clear &= ~nominal.dst_fall_ambiguous.reindex(electrical.index, fill_value=False)
    summary = coverage(electrical, poa, nominal, clear)
    print("Coverage completed", summary, flush=True)
    summary["ramp_timing"] = timing(electrical, poa, nominal, clear)
    forecast_support(electrical, poa, nominal, clear)
    equipment(nominal)
    after = {str(p): digest(p) for p in source}
    baseline_after = {str(p): digest(p) for p in baseline_sources if p.is_file()}
    if before != after or baseline_before != baseline_after:
        raise RuntimeError("An input changed during diagnostics")
    summary.update(source_hashes=before, frozen_baseline_hashes=baseline_before, sources_unchanged=True,
                   timestamp_basis="Pacific wall clock; no source shifts or filling",
                   timing_offset="electrical change at t versus POA change at t+offset; identical support",
                   coverage="full electrical record; solar elevation >10 deg, POA>=200 for useful-light pairs",
                   longitudinal="descriptive matched ratios; zero outputs retained; not a degradation estimate",
                   split="alternate complete dates per month/year; selection-fitted gain held fixed on validation")
    (OUT / "provenance.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
