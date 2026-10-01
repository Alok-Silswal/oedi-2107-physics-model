"""System metadata and explicit, provisional modelling assumptions."""

from dataclasses import dataclass, field
import math
from .dc_losses import DCLossConfig


@dataclass(frozen=True)
class FaimanConfig:
    """Provisional pvlib defaults, not coefficients fitted to this plant.

    u0: W/(m² K); u1: W/(m² K)/(m/s).
    Module temperature is used as a cell-temperature proxy.
    """

    u0: float = 25.0
    u1: float = 6.84
    provenance: str = "pvlib Faiman defaults; uncalibrated for System 2107"

    def __post_init__(self):
        if not math.isfinite(self.u0) or self.u0 <= 0:
            raise ValueError("Faiman u0 must be finite and positive")
        if not math.isfinite(self.u1) or self.u1 < 0:
            raise ValueError("Faiman u1 must be finite and nonnegative")


@dataclass(frozen=True)
class SystemConfig:
    """Known metadata and inferred topology; unknown site fields stay unset."""

    system_id: str = "OEDI System 2107"
    module_name: str = "Hyundai HiS-M310TI"
    cec_module_entry: str = "Hyundai_Heavy_Industries_Green_Energy_Co__HiS_M310TI"
    inverter_name: str = "ABB TRIO-27.6-TL-OUTD-S1B"
    # Explicit S1B database key, provisional US/480 V variant; review _A alternate.
    cec_inverter_entry: str = "ABB__TRIO_27_6_TL_OUTD_S1B_US_480__480V_"
    mounting: str = "fixed-tilt"
    tilt_deg: float = 25.0
    azimuth_deg: float = 180.0
    approximate_dc_capacity_kw: float = 893.0
    modules_per_string: int = 20
    strings_per_inverter: int = 6
    inverter_count: int = 24
    topology_provenance: str = "INFERRED: 20 series modules × 6 parallel strings/inverter"
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None
    wind_measurement_height_m: float | None = None
    faiman: FaimanConfig = field(default_factory=FaimanConfig)
    dc_losses: DCLossConfig = field(default_factory=DCLossConfig)

    def __post_init__(self):
        for name in ("modules_per_string", "strings_per_inverter", "inverter_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def modules_per_inverter(self) -> int:
        return self.modules_per_string * self.strings_per_inverter

    @property
    def total_modules(self) -> int:
        return self.modules_per_inverter * self.inverter_count
