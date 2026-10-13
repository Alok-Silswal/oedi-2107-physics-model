"""Check lag direction/common support and exact-gap handling on synthetic data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("stage2", Path(__file__).parents[1] / "scripts/audit_stage2_readiness.py")
stage2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage2)


def test_ramps_recover_known_delay_without_filling_missing_labels():
    index = pd.date_range("2022-04-01 10:00", periods=200, freq="5min")
    poa = pd.Series(np.random.default_rng(2107).uniform(200, 900, len(index)), index=index)
    response = poa.shift(1) * .03
    valid = pd.Series(True, index=index)
    before = poa.copy()
    pairs = stage2.ramp_pairs(poa, response, valid, .1)
    correlations = {o: pairs.electrical_change.corr(pairs[f"poa_change_{o}"]) for o in stage2.OFFSETS}
    assert max(correlations, key=correlations.get) == -5
    assert correlations[-5] > .999999
    reduced = stage2.ramp_pairs(poa.drop(index[50]), response, valid, .1)
    assert len(reduced) < len(pairs)
    assert index[50] not in reduced.index
    assert index[51] not in reduced.index
    assert reduced.notna().all().all()
    pd.testing.assert_series_equal(poa, before)


def test_true_runs_do_not_bridge_missing_timestamps():
    index = pd.to_datetime(["2022-04-01 10:00", "2022-04-01 10:05", "2022-04-01 10:15", "2022-04-01 10:20"])
    result = stage2.runs(pd.Series(True, index=index), "missing", min_samples=2)
    assert len(result) == 2
    assert [r["samples"] for r in result] == [2, 2]
