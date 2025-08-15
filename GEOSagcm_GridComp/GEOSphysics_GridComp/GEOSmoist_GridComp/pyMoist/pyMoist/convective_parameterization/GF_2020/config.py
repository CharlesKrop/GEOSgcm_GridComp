from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int


@dataclass
class GF2020Config:
    STOCHASTIC_CONVECTION: bool
    STOCH_TOP: Float
    STOCH_BOT: Float
    GF_MIN_AREA: Float
    GF_ENV_SETTING: Int
    ENTRVERSION: Int
    CONVECTION_TRACER: Int
    C1: Float
    ADV_TRIGGER: Int
