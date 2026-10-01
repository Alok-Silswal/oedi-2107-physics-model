"""Known arithmetic and strict alignment/mask checks on synthetic data only."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from oedi2107 import SystemConfig
from oedi2107.validation import (Normalization, mae, rmse, mbe, normalized_mae,
                                 normalized_rmse, r_squared, validate_dc,
                                 validate_inverter_ac, validate_plant_ac)


def times(n=3):
    return pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")


def test_known_metrics_and_explicit_normalization():
    expected, measured = [1., 2., 4.], [2., 2., 3.]
    normalization = Normalization(10., "synthetic reference 10 W")
    assert mae(expected, measured) == pytest.approx(2 / 3)
    assert rmse(expected, measured) == pytest.approx(np.sqrt(2 / 3))
    assert mbe(expected, measured) == 0
    assert mbe([0., 0.], [1., 3.]) == 2
    assert normalized_mae(expected, measured, normalization) == pytest.approx(2 / 30)
    assert normalized_rmse(expected, measured, normalization) == pytest.approx(np.sqrt(2 / 3) / 10)
    assert r_squared(expected, measured) == pytest.approx(-2.)
    assert r_squared(measured, measured) == 1.


@pytest.mark.parametrize("function", [mae, rmse, mbe, r_squared])
def test_metric_inputs_must_be_finite_and_aligned(function):
    with pytest.raises(ValueError, match="finite"):
        function([1., np.nan], [1., 2.])
    with pytest.raises(ValueError, match="aligned"):
        function(pd.Series([1.], index=times(1)), pd.Series([1.], index=times(2)[1:]))
    with pytest.raises(ValueError):
        function([1.], [1., 2.])


def test_undefined_metrics_are_not_forced_finite():
    assert np.isnan(mae([], []))
    assert np.isnan(r_squared([1.], [1.]))
    assert np.isnan(r_squared([1., 1.], [1., 1.]))


@pytest.mark.parametrize("value,basis", [(0., "reference"), (-1., "reference"),
                                        (np.inf, "reference"), (10., ""), (True, "reference")])
def test_invalid_normalization_rejected(value, basis):
    with pytest.raises(ValueError):
        Normalization(value, basis)


def test_outer_union_counts_nan_inf_and_missing_records():
    expected = pd.DataFrame({"expected_plant_ac_power": [10., np.nan, np.inf]}, index=times())
    measured = pd.DataFrame({"measured_meter_power": [12., 15., 20.]}, index=times(4)[[0, 1, 3]])
    result = validate_plant_ac(expected, measured)
    row = result.metrics.iloc[0]
    assert row.total_samples == 4
    assert row.valid_paired_samples == 1
    assert row.excluded_samples == 3
    assert row.missing_expected_records == 1
    assert row.missing_measured_records == 1
    assert row.missing_or_nonfinite_samples == 3
    assert row.mae == 2
    assert row.mbe == 2
    assert result.table.residual.iloc[0] == 2
    assert result.table.residual.iloc[1:].isna().all()
    assert np.isnan(row.normalized_mae)
    assert row.normalization_basis is None
    assert result.table.timestamp.tolist() == list(times(4))


def test_no_five_to_fifteen_minute_resampling():
    expected = pd.DataFrame({"expected_plant_ac_power": [1., 2., 3., 4.]}, index=times(4))
    measured = pd.DataFrame({"measured_meter_power": [1., 4.]}, index=times(4)[[0, 3]])
    result = validate_plant_ac(expected, measured)
    assert result.metrics.iloc[0].total_samples == 4
    assert result.metrics.iloc[0].valid_paired_samples == 2
    assert result.table.loc[1:2, "measured"].isna().all()


def test_inverter_identity_cannot_cross_pair():
    expected = pd.DataFrame({"timestamp": [times(1)[0]] * 2, "inverter_id": ["A", "B"],
                             "expected_ac_power": [10., 100.]})
    measured = pd.DataFrame({"timestamp": [times(1)[0]] * 2, "inverter_id": ["B", "C"],
                             "measured_ac_power": [110., 500.]})
    result = validate_inverter_ac(expected, measured)
    metrics = result.metrics.set_index("inverter_id")
    assert set(metrics.index) == {"A", "B", "C"}
    assert metrics.loc["B", "mae"] == 10
    assert metrics.loc["A", "valid_paired_samples"] == 0
    assert metrics.loc["C", "valid_paired_samples"] == 0
    assert result.table.loc[result.table.inverter_id == "A", "measured"].isna().all()


def test_dc_reports_separate_signals_and_units():
    expected = pd.DataFrame({"inverter_id": ["A"] * 3, "expected_dc_voltage": [600.] * 3,
                             "expected_dc_current": [10.] * 3, "expected_dc_power": [6000.] * 3}, index=times())
    measured = pd.DataFrame({"inverter_id": ["A"] * 3, "measured_dc_voltage": [601.] * 3,
                             "measured_dc_current": [11.] * 3, "measured_dc_power": [6100.] * 3}, index=times())
    result = validate_dc(expected, measured, normalizations={"dc_power": Normalization(10000., "synthetic power reference [W]")})
    metrics = result.metrics.set_index("quantity")
    assert metrics.loc["dc_voltage", "mae"] == 1
    assert metrics.loc["dc_current", "mae"] == 1
    assert metrics.loc["dc_power", "mae"] == 100
    assert metrics.loc["dc_power", "normalized_mae"] == 0.01
    assert dict(metrics.unit) == {"dc_voltage": "V", "dc_current": "A", "dc_power": "W"}
    assert len(result.table) == 9


def test_period_labels_and_healthy_masks_respected():
    expected = pd.DataFrame({"expected_plant_ac_power": [10., 20., 30.]}, index=times())
    measured = pd.DataFrame({"measured_meter_power": [11., 25., 100.]}, index=times())
    labels = pd.Series(["calibration", "validation", "holdout"], index=times())
    healthy = pd.Series([True, True, False], index=times())
    result = validate_plant_ac(expected, measured, period_labels=labels, healthy_mask=healthy,
                               periods=["validation", "holdout"])
    assert result.table.period_label.tolist() == labels.tolist()
    assert result.table.healthy.tolist() == healthy.tolist()
    assert result.table.valid_pair.tolist() == [False, True, False]
    assert result.metrics.iloc[0].mae == 5
    assert result.metrics.iloc[0].excluded_samples == 2
    assert result.metrics.iloc[0].excluded_by_masks_samples == 2
    assert result.table.attrs["healthy_mask_supplied"]


def test_period_masks_and_device_healthy_annotation():
    expected = pd.DataFrame({"timestamp": list(times(2)) * 2, "inverter_id": ["A", "A", "B", "B"],
                             "expected_ac_power": [10.] * 4})
    measured = expected.rename(columns={"expected_ac_power": "measured_ac_power"})
    masks = {"calibration": pd.Series([True, False], index=times(2)),
             "holdout": pd.Series([False, True], index=times(2))}
    healthy = pd.Series([True, True, False, False],
                        index=pd.MultiIndex.from_frame(expected[["timestamp", "inverter_id"]]))
    result = validate_inverter_ac(expected, measured, period_masks=masks, healthy_mask=healthy)
    metrics = result.metrics.set_index("inverter_id")
    assert metrics.loc["A", "valid_paired_samples"] == 2
    assert metrics.loc["B", "valid_paired_samples"] == 0
    assert set(result.table.period_label) == {"calibration", "holdout"}


def test_missing_annotations_and_overlapping_masks_rejected():
    expected = pd.DataFrame({"expected_plant_ac_power": [1., 2., 3.]}, index=times())
    measured = pd.DataFrame({"measured_meter_power": [1., 2., 3.]}, index=times())
    with pytest.raises(ValueError, match="cover every"):
        validate_plant_ac(expected, measured, healthy_mask=pd.Series([True], index=times(1)))
    mask = pd.Series(True, index=times())
    with pytest.raises(ValueError, match="overlap"):
        validate_plant_ac(expected, measured, period_masks={"calibration": mask, "holdout": mask})
    with pytest.raises(ValueError, match="boolean"):
        validate_plant_ac(expected, measured, healthy_mask=pd.Series(1, index=times()))


def test_duplicates_and_timezone_conversion_rejected():
    expected = pd.DataFrame({"expected_plant_ac_power": [1., 2., 3.]}, index=times())
    measured = pd.DataFrame({"measured_meter_power": [1., 2., 3.]}, index=times())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_plant_ac(pd.concat([expected, expected]), measured)
    measured.index = measured.index.tz_convert("Asia/Calcutta")
    with pytest.raises(ValueError, match="timezones"):
        validate_plant_ac(expected, measured)


def test_no_fitting_or_input_mutation(monkeypatch):
    import oedi2107.pipeline as pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not run or fit the model")

    monkeypatch.setattr(pipeline, "run_plant_model", forbidden)
    config = SystemConfig()
    before = asdict(config)
    expected = pd.DataFrame({"expected_plant_ac_power": [-198.72, 100., 200.]}, index=times())
    measured = pd.DataFrame({"measured_meter_power": [-190., 110., 210.]}, index=times())
    expected_before, measured_before = expected.copy(deep=True), measured.copy(deep=True)
    result = validate_plant_ac(expected, measured)
    pd.testing.assert_frame_equal(expected, expected_before)
    pd.testing.assert_frame_equal(measured, measured_before)
    assert asdict(config) == before
    assert config.dc_losses.k_dc is None
    assert result.table.expected.iloc[0] == -198.72
