from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int


@dataclass
class SHOCMFConfiguration:
    dtn: Float
    PRNUMBER: Float
    min_tke: Float
    BUOYOPT: Int
    LENOPT: Int
    LENFAC1: Float
    LENFAC2: Float
    LENFAC3: Float