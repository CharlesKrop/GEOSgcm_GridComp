from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int


@dataclass
class SHOCMFConfiguration:
    dtn: Float
    PRNUMBER: Float
    min_tke: Float
    max_tke: Float
    BUOYOPT: Int
    LENOPT: Int
    LENFAC1: Float
    LENFAC2: Float
    LENFAC3: Float
    Ce: Float
    Ces: Float
    nitr: Int
    shoc_lambda: Float
    ck: Float
    # ET: Int
    # L0: Float
    # L0fac: Float
    # NUP: Int