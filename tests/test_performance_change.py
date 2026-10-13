import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from experiment_performance_change import slope_fit, aggregate


def test_shared_decline_and_reversible_step():
    rows = []
    for unit in ("inv_01", "inv_02"):
        for year in range(2019, 2024):
            for month in range(1, 11):
                date = f"{year}-{month:02d}"
                time = year-2019+(month-1)/12
                step = -15 if unit == "inv_01" and "2021-03" <= date < "2022-06" else 0
                rows.append(dict(inverter=unit, month=date, index=100-time+step))
    frame = pd.DataFrame(rows)
    jumps = pd.DataFrame({"inverter_id": ["inv_01", "inv_01"], "boundary": ["2021-03", "2022-06"]})
    result = slope_fit(frame, jumps, regimes=True)
    assert abs(result["slope_pp_year"]+1) < 1e-8
    assert result["rank"] == result["parameters"]


def test_recorded_zeros_retained_but_generating_is_conditional():
    frame = pd.DataFrame({"year": np.repeat(range(2018, 2024), 20), "cell": "1|4|3|2"})
    ac = pd.DataFrame({"inv_01_ac_power": np.tile([0.]*10+[10.]*10, 6)})
    result, _ = aggregate(frame, ac, np.full(len(frame), 10.), 27)
    delivered = result.loc[result.outcome.eq("delivered")]
    generating = result.loc[result.outcome.eq("generating")]
    assert delivered.ratio.eq(.5).all()
    assert delivered.zero_fraction.eq(.5).all()
    assert generating.ratio.eq(1).all()
