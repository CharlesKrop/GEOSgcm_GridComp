from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int


@dataclass
class SHOCMFConfiguration:
    dtn: Float
    PRNUMBER: Float
    BUOYOPT: Int
    LENOPT: Int
    LENFAC1: Float
    LENFAC2: Float
    LENFAC3: Float
    CeFAC: Float
    CesFAC: Float
    Ck: Float
    shoc_lambda: Float
    ET: Int
    L0_EDMF: Float
    L0fac: Float
    NUP: Int