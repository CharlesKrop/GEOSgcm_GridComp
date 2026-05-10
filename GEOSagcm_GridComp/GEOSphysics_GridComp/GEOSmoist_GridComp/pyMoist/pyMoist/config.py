from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int, Bool


@dataclass
class MoistConfig:
    DT_MOIST: Float
    CONVECTION_FRACTION_MIN: Float
    CONVECTION_FRACTION_MAX: Float
    CONVECTION_FRACTION_EXP: Float
    USE_AEROSOL_NN: Bool
    NUMBER_OF_TRACERS: Int
    UPDATE_PRECIP_TYPE: Bool
    DIAGNOSE_PRECIP_TYPE: Bool
