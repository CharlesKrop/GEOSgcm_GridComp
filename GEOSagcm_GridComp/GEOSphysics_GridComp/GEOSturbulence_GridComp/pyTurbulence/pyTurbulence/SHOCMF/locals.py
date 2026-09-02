from dataclasses import dataclass

from ndsl import Local, NDSLRuntime, QuantityFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Int


@dataclass
class SHOCMFLocals: