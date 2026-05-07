from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int


@dataclass
class MoistConfig:
    DT_MOIST: Float
    CONVECTION_FRACTION_MIN: Float
    CONVECTION_FRACTION_MAX: Float
    CONVECTION_FRACTION_EXP: Float
    USE_AEROSOL_NN: bool
