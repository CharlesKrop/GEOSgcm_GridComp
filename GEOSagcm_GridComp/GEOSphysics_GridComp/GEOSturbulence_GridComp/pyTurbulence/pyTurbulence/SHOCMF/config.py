from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int


@dataclass
class SHOCMFConfiguration:
    USE_EIS: bool
    NCNST: Int
    rkfre: Float