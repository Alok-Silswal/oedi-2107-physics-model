"""Aggregate derate checks; numerical factors here are test cases, not calibration."""

import numpy as np
import pandas as pd
import pytest

from oedi2107 import DCLossConfig, SystemConfig, apply_dc_losses, run_dc_foundation, run_dc_model


@pytest.fixture
def ideal_dc():
    return pd.DataFrame({
        "expected_dc_voltage_ideal": [0., 600., 700.],
        "expected_dc_current_ideal": [0., 20., 40.],
        "expected_dc_power_ideal": [0., 12000., 28000.],
    }, index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"))


@pytest.mark.parametrize("factor", [1., 0.9, 0.5])
def test_derate_preserves_ideal_and_consistent_dc(ideal_dc, factor):
    original = ideal_dc.copy(deep=True)
    result = apply_dc_losses(ideal_dc, DCLossConfig(k_dc=factor))
    pd.testing.assert_frame_equal(ideal_dc, original)
    pd.testing.assert_frame_equal(result[original.columns], original)
    pd.testing.assert_series_equal(result.expected_dc_voltage, original.expected_dc_voltage_ideal, check_names=False)
    np.testing.assert_allclose(result.expected_dc_power, factor * original.expected_dc_power_ideal)
    np.testing.assert_allclose(result.expected_dc_current, factor * original.expected_dc_current_ideal)
    np.testing.assert_allclose(result.expected_dc_power, result.expected_dc_voltage * result.expected_dc_current)
    assert (result.dc_loss_factor == factor).all()
    assert result.loc[result.index[0], "expected_dc_current"] == 0
    assert np.isfinite(result.to_numpy()).all()
    assert result.index.equals(original.index)


@pytest.mark.parametrize("factor", [0, -0.1, 1.01, np.nan, np.inf, -np.inf, True, "0.9", [0.9] * 24])
def test_invalid_or_non_global_factor_rejected(factor):
    with pytest.raises(ValueError, match="K_DC"):
        DCLossConfig(k_dc=factor)


def test_unset_factor_cannot_be_applied(ideal_dc):
    assert SystemConfig().dc_losses.k_dc is None
    with pytest.raises(ValueError, match="unset/uncalibrated"):
        apply_dc_losses(ideal_dc, DCLossConfig())


def test_inconsistent_ideal_dc_rejected(ideal_dc):
    ideal_dc.loc[ideal_dc.index[1], "expected_dc_power_ideal"] += 1
    with pytest.raises(ValueError, match="must equal"):
        apply_dc_losses(ideal_dc, DCLossConfig(k_dc=1))


def test_nonzero_power_at_zero_voltage_rejected(ideal_dc):
    ideal_dc.loc[ideal_dc.index[0], "expected_dc_power_ideal"] = 1e-12
    with pytest.raises(ValueError, match="zero voltage"):
        apply_dc_losses(ideal_dc, DCLossConfig(k_dc=1))


@pytest.mark.parametrize("factor", [None, 1., 0.9])
def test_model_integration_preserves_foundation_and_handles_unset(factor):
    weather = pd.DataFrame({"poa": [0., 600., 1000.], "temp_air": 25., "wind_speed": 1.},
                           index=pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"))
    config = SystemConfig(dc_losses=DCLossConfig(k_dc=factor))
    foundation = run_dc_foundation(weather, config)
    result = run_dc_model(weather, config)
    for quantity in ("voltage", "current", "power"):
        pd.testing.assert_series_equal(result[f"expected_dc_{quantity}_ideal"],
                                       foundation[f"expected_dc_{quantity}"], check_names=False)
    assert result.attrs["cec_module_entry"] == foundation.attrs["cec_module_entry"]
    if factor is None:
        assert result[["dc_loss_factor", "expected_dc_current", "expected_dc_power"]].isna().all().all()
        assert result.attrs["dc_loss_status"] == "uncalibrated"
    else:
        np.testing.assert_allclose(result.expected_dc_power, foundation.expected_dc_power * factor)
        np.testing.assert_allclose(result.expected_dc_power, result.expected_dc_voltage * result.expected_dc_current)
