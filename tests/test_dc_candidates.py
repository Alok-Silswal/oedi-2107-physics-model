"""Common screens preserve measurements and cannot fit individual loss baselines."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from oedi2107.config import SystemConfig
from oedi2107.dc_candidates import CandidateThresholds, candidate_features, select_candidates


@pytest.fixture
def example():
    index = pd.date_range("2020-06-01 10:00", periods=7, freq="5min", name="timestamp")
    baseline = pd.DataFrame({"poa_w_m2":500., "expected_dc_voltage_ideal":700.,
        "expected_dc_current_ideal":25., "expected_dc_power_ideal":17500.,
        "model_inputs_available":True, "dc_model_computed":True, "electrical_qc_flag":False,
        "irradiance_qc_flag":False, "dst_fall_ambiguous":False}, index=index)
    ids = [f"inv_{i:02d}" for i in range(1,25)]
    voltage = pd.DataFrame(680., index=index, columns=ids)
    voltage.inv_05 = np.nan
    current = pd.DataFrame(24., index=index, columns=ids)
    power = voltage * current
    ac = pd.DataFrame(15000., index=index, columns=ids)
    return baseline, voltage, current, power, ac


def select(example, thresholds=None):
    features = candidate_features(*example)
    masks, counts = select_candidates(features, paco=27600., pdco=28199.173828, thresholds=thresholds)
    return features, masks, counts


def test_common_mask_counts_and_nonmutation(example):
    original = [table.copy(deep=True) for table in example]
    features, masks, counts = select(example)
    assert masks.candidate_healthy_mask.tolist() == [False,True,True,True,True,True,False]
    assert counts.remaining_timestamps.iloc[-1] == 5
    assert counts.remaining_timestamps.diff().dropna().le(0).all()
    assert counts.removed_at_stage.sum() == 2
    assert features.voltage_ratio.iloc[2] == pytest.approx(680/700)
    assert features.current_ratio.iloc[2] == pytest.approx(24/25)
    for before, after in zip(original, example):
        pd.testing.assert_frame_equal(before,after)
    assert SystemConfig().dc_losses.k_dc is None
    assert not any("loss_factor" in column for column in features)


def test_persistent_weak_inverter_rejects_whole_plant(example):
    example[2].inv_12 *= .7
    example[3].inv_12 *= .7
    example[4].inv_12 *= .7
    _, masks, _ = select(example)
    assert not masks.all_inverters_consensus.any()
    assert not masks.candidate_healthy_mask.any()


def test_missing_inverter05_is_permitted_other_missing_is_not(example):
    example[2].loc[example[0].index[2], "inv_05"] = np.nan
    _, masks, _ = select(example)
    assert not masks.available_dc_and_ac.iloc[2]
    assert masks.candidate_healthy_mask.iloc[3]
    example[1].inv_05 = 680.
    with pytest.raises(ValueError, match="05"):
        select(example)


def test_qc_dst_startup_clipping_and_ideal_guard(example):
    baseline, _, _, _, ac = example
    baseline.loc[baseline.index[1],"electrical_qc_flag"] = True
    baseline.loc[baseline.index[2],"dst_fall_ambiguous"] = True
    ac.loc[baseline.index[3],"inv_03"] = 1000.
    ac.loc[baseline.index[4],"inv_01"] = .9*27600
    baseline.loc[baseline.index[5],"expected_dc_power_ideal"] = .9*28199.173828
    _, masks, _ = select(example)
    assert not masks.candidate_healthy_mask.any()
    assert not masks.known_qc_and_dst_clear.iloc[1:3].any()
    assert not masks.all_inverters_operating.iloc[3]
    assert not masks.below_rating_guards.iloc[4:6].any()


def test_no_ramp_filling_across_gap_or_endpoints(example):
    gap = example[0].index.delete(3)
    example = tuple(table.loc[gap] for table in example)
    _, masks, _ = select(example)
    assert masks.stable_adjacent_poa.tolist() == [False,True,False,False,True,False]
    _, disabled, _ = select(example, replace(CandidateThresholds(), maximum_poa_ramp=None))
    assert disabled.candidate_healthy_mask.all()


def test_exact_alignment_and_ids_required(example):
    example[2].index = example[2].index + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="timestamps"):
        select(example)


@pytest.mark.parametrize("kwargs",[{"minimum_poa":np.nan},{"consensus_tolerance":0},
    {"rating_margin":1},{"maximum_poa_ramp":-1},{"minimum_ac_fraction":.95}])
def test_invalid_screen_thresholds(kwargs):
    with pytest.raises(ValueError):
        CandidateThresholds(**kwargs)
