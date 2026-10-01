"""Named-record and complete-set aggregation checks; no measured-data joins."""

import numpy as np
import pandas as pd
import pytest

from oedi2107 import (DCLossConfig, SystemConfig, represent_inverter_outputs,
                      aggregate_inverter_outputs, run_inverter_model, run_plant_model)


@pytest.fixture
def nominal():
    return pd.DataFrame({
        "expected_ac_power": [-8.28, 10000., 27600.],
        "expected_dc_voltage": [0., 715., 715.],
        "expected_dc_current": [0., 14., 56.],
        "expected_dc_power": [0., 10010., 40040.],
    }, index=pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"))


def test_24_records_per_timestamp_and_dc_preserved(nominal):
    original = nominal.copy(deep=True)
    outputs = represent_inverter_outputs(nominal)
    assert len(outputs) == 24 * len(nominal)
    assert outputs.inverter_id.nunique() == 24
    assert (outputs.groupby("timestamp").inverter_id.nunique() == 24).all()
    assert not outputs.duplicated(["timestamp", "inverter_id"]).any()
    for _, inverter in outputs.groupby("inverter_id"):
        pd.testing.assert_frame_equal(inverter.drop(columns="inverter_id").set_index("timestamp"),
                                      nominal.rename_axis("timestamp"), check_freq=False)
    pd.testing.assert_frame_equal(nominal, original)


def test_signed_sum_uses_each_inverter_record(nominal):
    outputs = represent_inverter_outputs(nominal)
    # Deliberately distinct synthetic daytime values demonstrate record-wise summing.
    mask = outputs.timestamp == nominal.index[1]
    outputs.loc[mask, "expected_ac_power"] = np.arange(1, 25) * 100.
    result = aggregate_inverter_outputs(outputs)
    sums = outputs.groupby("timestamp").expected_ac_power.sum()
    np.testing.assert_allclose(result.expected_plant_ac_power, sums)
    assert result.expected_plant_ac_power.iloc[0] == pytest.approx(-24 * 8.28)
    assert result.expected_plant_ac_power.iloc[1] == pytest.approx(30000.)
    assert (result.inverter_record_count == 24).all()
    assert (result.plant_prediction_status == "complete").all()
    assert result.index.equals(nominal.index)


def test_missing_inverter_record_rejected(nominal):
    outputs = represent_inverter_outputs(nominal).drop(index=0)
    with pytest.raises(ValueError, match="missing=.*inverter_01"):
        aggregate_inverter_outputs(outputs)


def test_entire_missing_timestamp_rejected(nominal):
    outputs = represent_inverter_outputs(nominal)
    outputs = outputs.loc[outputs.timestamp != nominal.index[1]]
    with pytest.raises(ValueError, match="Incomplete timestamp coverage"):
        aggregate_inverter_outputs(outputs)


def test_duplicate_inverter_at_timestamp_rejected(nominal):
    outputs = represent_inverter_outputs(nominal)
    outputs.loc[1, "inverter_id"] = outputs.loc[0, "inverter_id"]
    with pytest.raises(ValueError, match="Duplicate inverter IDs within timestamp"):
        aggregate_inverter_outputs(outputs)


def test_unexpected_id_is_not_an_allowed_replacement(nominal):
    outputs = represent_inverter_outputs(nominal)
    outputs.loc[0, "inverter_id"] = "unknown_inverter"
    with pytest.raises(ValueError, match="unexpected=.*unknown_inverter"):
        aggregate_inverter_outputs(outputs)


@pytest.mark.parametrize("ids", [[f"i{i}" for i in range(23)], ["same"] * 24])
def test_invalid_roster_rejected(nominal, ids):
    with pytest.raises(ValueError):
        represent_inverter_outputs(nominal, inverter_ids=ids)


def test_supplied_device_ids_preserved(nominal):
    ids = tuple(f"device-{i}" for i in range(24))
    outputs = represent_inverter_outputs(nominal, inverter_ids=ids)
    assert set(outputs.inverter_id) == set(ids)
    assert (aggregate_inverter_outputs(outputs).inverter_record_count == 24).all()


def test_nan_prediction_reported_not_treated_as_zero(nominal, caplog):
    outputs = represent_inverter_outputs(nominal)
    outputs.loc[0, "expected_ac_power"] = np.nan
    result = aggregate_inverter_outputs(outputs)
    assert np.isnan(result.expected_plant_ac_power.iloc[0])
    assert result.finite_ac_prediction_count.iloc[0] == 23
    assert result.uncomputed_inverter_ids.iloc[0] == ("inverter_01",)
    assert result.plant_prediction_status.iloc[0] == "uncomputed"
    assert "inverter_01" in caplog.text


def test_infinite_prediction_rejected(nominal):
    outputs = represent_inverter_outputs(nominal)
    outputs.loc[0, "expected_ac_power"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        aggregate_inverter_outputs(outputs)


@pytest.mark.parametrize("factor", [None, 1.])
def test_pipeline_keeps_nominal_outputs_and_unset_factor(factor):
    weather = pd.DataFrame({"poa": [0., 600., 1000.], "temp_air": 25., "wind_speed": 1.},
                           index=pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"))
    config = SystemConfig(dc_losses=DCLossConfig(k_dc=factor))
    nominal = run_inverter_model(weather, config)
    result = run_plant_model(weather, config)
    assert len(result.inverter_outputs) == 72
    first = result.inverter_outputs.query("inverter_id == 'inverter_01'").drop(columns="inverter_id").set_index("timestamp")
    pd.testing.assert_frame_equal(first, nominal.rename_axis("timestamp"), check_freq=False)
    assert result.plant_outputs.index.equals(weather.index)
    assert result.plant_outputs.attrs["dc_loss_factor"] == factor
    if factor is None:
        assert result.inverter_outputs.expected_ac_power.isna().all()
        assert result.plant_outputs.expected_plant_ac_power.isna().all()
        assert (result.plant_outputs.finite_ac_prediction_count == 0).all()
    else:
        np.testing.assert_allclose(result.plant_outputs.expected_plant_ac_power, 24 * nominal.expected_ac_power)
