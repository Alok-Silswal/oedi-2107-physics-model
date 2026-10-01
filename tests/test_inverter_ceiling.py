import numpy as np
import pandas as pd

from oedi2107.config import SystemConfig
from oedi2107.inverter import load_cec_inverter, calculate_inverter_ac_30kw_ceiling


def test_official_equation_reference_and_external_ceiling():
    parameters = load_cec_inverter(SystemConfig().cec_inverter_entry)
    original = parameters.copy(deep=True)
    # Include nighttime, startup boundaries and several operating voltages.
    power = np.array([0., parameters.Pso / 2, parameters.Pso, 1000., 14000.,
                      25000., 28199., 30000., 31000., 40000.])
    for voltage in (500., 715., 800.):
        dc = pd.DataFrame({"expected_dc_voltage": voltage, "expected_dc_power": power})
        before = dc.copy(deep=True)
        result = calculate_inverter_ac_30kw_ceiling(dc, parameters)
        below_clip = result.expected_ac_power_cec < parameters.Paco
        np.testing.assert_allclose(result.loc[below_clip, "expected_ac_power_30kw"],
                                   result.loc[below_clip, "expected_ac_power_cec"],
                                   rtol=1e-12, atol=1e-9)
        assert result.expected_ac_power_30kw.max() == 30000.
        assert (result.expected_ac_power_30kw <= 30000.).all()
        assert (result.expected_ac_power_30kw.iloc[:2] == -parameters.Pnt).all()
        assert (result.expected_ac_power_cec <= 27600.).all()
        assert (result.expected_ac_power_30kw > 27600.).any()
        pd.testing.assert_frame_equal(dc, before)
    pd.testing.assert_series_equal(parameters, original)
    assert parameters.attrs == original.attrs
    assert SystemConfig().dc_losses.k_dc is None
