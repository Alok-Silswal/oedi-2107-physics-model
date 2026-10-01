"""Strict paired metrics; residual and MBE use measured minus expected."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Normalization:
    """Caller-selected positive denominator in the signal's physical units.

    basis describes the choice (e.g. reviewed rated power, measured mean over
    a named period). No denominator or convention is inferred. Normalized
    metrics are dimensionless fractions, not percentages.
    """

    value: float
    basis: str

    def __post_init__(self):
        if isinstance(self.value, bool) or not isinstance(self.value, Real) or not math.isfinite(self.value) or self.value <= 0:
            raise ValueError("Normalization value must be finite and positive")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ValueError("Normalization basis must explicitly describe the denominator")


def _pairs(expected, measured):
    """Already aligned vectors → finite-pair checks → numeric arrays."""
    if isinstance(expected, pd.Series) and isinstance(measured, pd.Series) and not expected.index.equals(measured.index):
        raise ValueError("Metric Series indexes must be exactly aligned")
    expected = np.asarray(expected, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if expected.ndim != 1 or measured.shape != expected.shape:
        raise ValueError("Metrics need equally sized one-dimensional aligned vectors")
    if not np.isfinite(expected).all() or not np.isfinite(measured).all():
        raise ValueError("Metric inputs must contain only finite paired samples")
    return expected, measured


def mae(expected, measured) -> float:
    """Aligned finite pairs → mean absolute measured-minus-expected → MAE."""
    expected, measured = _pairs(expected, measured)
    return float(np.mean(np.abs(measured - expected))) if expected.size else float("nan")


def rmse(expected, measured) -> float:
    """Aligned finite pairs → root mean squared residual → RMSE."""
    expected, measured = _pairs(expected, measured)
    return float(np.sqrt(np.mean((measured - expected) ** 2))) if expected.size else float("nan")


def mbe(expected, measured) -> float:
    """Aligned finite pairs → mean(measured - expected) → signed MBE."""
    expected, measured = _pairs(expected, measured)
    return float(np.mean(measured - expected)) if expected.size else float("nan")


def normalized_mae(expected, measured, normalization: Normalization) -> float:
    """Finite pairs + explicit denominator → MAE/denominator → fraction."""
    return mae(expected, measured) / normalization.value


def normalized_rmse(expected, measured, normalization: Normalization) -> float:
    """Finite pairs + explicit denominator → RMSE/denominator → fraction."""
    return rmse(expected, measured) / normalization.value


def r_squared(expected, measured) -> float:
    """Finite pairs → 1 - SSE/SST of measured values → R² (can be negative).

    Undefined for fewer than two pairs or constant measured values: returns
    NaN, including perfect constant predictions; no forced finite replacement.
    """
    expected, measured = _pairs(expected, measured)
    if expected.size < 2:
        return float("nan")
    sst = float(np.sum((measured - np.mean(measured)) ** 2))
    return 1 - float(np.sum((measured - expected) ** 2)) / sst if sst > 0 else float("nan")
