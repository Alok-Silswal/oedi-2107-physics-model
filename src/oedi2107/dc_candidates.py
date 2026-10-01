"""Common-plant screening masks, not verified health or a loss-factor fit."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateThresholds:
    """Explicit provisional screening choices; no model parameters are changed."""

    minimum_poa: float = 300.0
    minimum_ac_fraction: float = 0.10
    rating_margin: float = 0.90
    voltage_tolerance: float = 0.10
    consensus_tolerance: float = 0.10
    maximum_poa_ramp: float | None = 0.20

    def __post_init__(self):
        if not np.isfinite(self.minimum_poa) or self.minimum_poa <= 0:
            raise ValueError("minimum_poa must be finite and positive")
        if not 0 < self.minimum_ac_fraction < self.rating_margin < 1:
            raise ValueError("AC floor and rating margin must satisfy 0 < floor < margin < 1")
        for value in (self.voltage_tolerance, self.consensus_tolerance):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("Relative tolerances must be finite and between zero and one")
        if self.maximum_poa_ramp is not None and (
            not np.isfinite(self.maximum_poa_ramp) or self.maximum_poa_ramp <= 0
        ):
            raise ValueError("POA ramp must be positive or explicitly disabled with None")


def _maximum_deviation(values):
    """Same-time inverter measurements -> median-relative spread -> consensus diagnostic."""
    median = np.median(values, axis=1)
    deviation = np.divide(np.abs(values - median[:, None]), median[:, None],
                          out=np.full_like(values, np.nan), where=median[:, None] > 0)
    return np.max(deviation, axis=1)


def candidate_features(baseline, voltage, current, power, ac_power_w):
    """Aligned ideal/DC/AC tables -> physical diagnostics -> all-timestamp features.

    V/A/W measurements are unchanged. All 24 currents and AC channels participate;
    inverter 05 is explicitly omitted only from V/P tests because voltage is absent.
    Medians are simultaneous plant consensus, never per-inverter fitted baselines.
    """
    ids = [f"inv_{number:02d}" for number in range(1, 25)]
    if not isinstance(baseline.index, pd.DatetimeIndex) or not baseline.index.is_unique or not baseline.index.is_monotonic_increasing:
        raise ValueError("Baseline needs unique increasing timestamps")
    for table in (voltage, current, power, ac_power_w):
        if list(table.columns) != ids or not table.index.equals(baseline.index):
            raise ValueError("All measurements must have exactly 24 ordered IDs and identical timestamps")
    if voltage.inv_05.notna().any() or power.inv_05.notna().any():
        raise ValueError("Inverter 05 voltage/power must remain missing")
    features = baseline.copy()
    v = voltage.drop(columns="inv_05").to_numpy(dtype=float)
    i = current.to_numpy(dtype=float)
    p = power.drop(columns="inv_05").to_numpy(dtype=float)
    ac = ac_power_w.to_numpy(dtype=float)
    features["finite_dc"] = np.isfinite(v).all(axis=1) & np.isfinite(i).all(axis=1) & np.isfinite(p).all(axis=1)
    features["positive_dc"] = (v > 0).all(axis=1) & (i > 0).all(axis=1) & (p > 0).all(axis=1)
    features["finite_ac"] = np.isfinite(ac).all(axis=1)
    features["minimum_ac_power_w"] = np.min(ac, axis=1)
    features["maximum_ac_power_w"] = np.max(ac, axis=1)
    features["measured_ac_power_median_w"] = np.median(ac, axis=1)
    ac_with_dc = ac[:, [number for number in range(24) if number != 4]]
    ac_dc_ratio = np.divide(ac_with_dc, p, out=np.full_like(p, np.nan), where=p > 0)
    features["maximum_ac_to_dc_power_ratio"] = np.max(ac_dc_ratio, axis=1)
    for name, values in (("voltage", v), ("current", i), ("power", p)):
        features[f"measured_dc_{name}_median"] = np.median(values, axis=1)
        expected = features[f"expected_dc_{name}_ideal"].to_numpy(dtype=float)
        ratio = np.divide(values, expected[:, None], out=np.full_like(values, np.nan), where=expected[:, None] > 0)
        features[f"{name}_ratio"] = np.median(ratio, axis=1)
        if name == "voltage":
            features["maximum_voltage_mpp_deviation"] = np.max(np.abs(ratio - 1), axis=1)
    features["maximum_current_consensus_deviation"] = _maximum_deviation(i)
    features["maximum_ac_consensus_deviation"] = _maximum_deviation(ac)
    # Exact five-minute neighbours only; gaps and endpoints fail this optional screen.
    poa = features.poa_w_m2
    adjacent = features.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    previous = (poa - poa.shift()).abs() / poa
    following = (poa - poa.shift(-1)).abs() / poa
    features["maximum_adjacent_poa_ramp"] = pd.concat([
        previous.where(adjacent), following.where(adjacent.shift(-1, fill_value=False))
    ], axis=1).max(axis=1, skipna=False)
    return features


def select_candidates(features, *, paco, pdco, thresholds=None):
    """All-row features + explicit screens -> common Boolean mask and removal counts.

    Every stage must pass for the whole plant. No measurement changes, filling,
    fitted losses, inverter-specific offsets, or confirmed-health claims occur.
    Rating screens are conservative guards, not an inferred measured clipping cap.
    """
    t = thresholds or CandidateThresholds()
    if not np.isfinite([paco, pdco]).all() or min(paco, pdco) <= 0:
        raise ValueError("CEC ratings must be finite and positive")
    for flag in ("model_inputs_available", "dc_model_computed", "electrical_qc_flag", "irradiance_qc_flag", "dst_fall_ambiguous"):
        if not pd.api.types.is_bool_dtype(features[flag]) or features[flag].isna().any():
            raise ValueError(f"Explicit Boolean flag required: {flag}")
    ideal = features[[f"expected_dc_{q}_ideal" for q in ("voltage", "current", "power")]]
    rules = {
        "valid_model_inputs": features.model_inputs_available & features.dc_model_computed & np.isfinite(ideal).all(axis=1),
        "known_qc_and_dst_clear": ~features.electrical_qc_flag & ~features.irradiance_qc_flag & ~features.dst_fall_ambiguous,
        "meaningful_irradiance": features.poa_w_m2.ge(t.minimum_poa),
        "available_dc_and_ac": features.finite_dc & features.positive_dc & features.finite_ac,
        "all_inverters_operating": features.minimum_ac_power_w.ge(t.minimum_ac_fraction * paco),
        "below_rating_guards": features.maximum_ac_power_w.lt(t.rating_margin * paco) & features.expected_dc_power_ideal.lt(t.rating_margin * pdco),
        "voltage_mpp_compatible": features.maximum_voltage_mpp_deviation.le(t.voltage_tolerance),
        "all_inverters_consensus": features.maximum_current_consensus_deviation.le(t.consensus_tolerance) & features.maximum_ac_consensus_deviation.le(t.consensus_tolerance),
        "stable_adjacent_poa": pd.Series(True, index=features.index) if t.maximum_poa_ramp is None else features.maximum_adjacent_poa_ramp.le(t.maximum_poa_ramp),
    }
    masks = pd.DataFrame(rules, index=features.index).fillna(False).astype(bool)
    cumulative = pd.Series(True, index=features.index)
    counts = [{"stage": "all_timestamps", "remaining_timestamps": len(features), "removed_at_stage": 0}]
    for name in masks:
        before = int(cumulative.sum())
        cumulative &= masks[name]
        counts.append({"stage": name, "remaining_timestamps": int(cumulative.sum()),
                       "removed_at_stage": before - int(cumulative.sum()),
                       "failed_rule_independently": int((~masks[name]).sum())})
    masks["candidate_healthy_mask"] = cumulative
    masks.attrs.update(status="candidate only; MPP/health not verified", k_dc=None,
                       voltage_power_inverters=23, current_ac_inverters=24)
    return masks, pd.DataFrame(counts)
