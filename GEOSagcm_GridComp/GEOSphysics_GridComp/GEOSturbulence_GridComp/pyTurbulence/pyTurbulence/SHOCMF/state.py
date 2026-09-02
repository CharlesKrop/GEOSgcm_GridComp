import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM, Float


@dataclasses.dataclass
class SHOCMFState(State):