import dataclasses

from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float


@dataclasses.dataclass
class MoistLocals(LocalState):
    p_interface_mb: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    p_mb: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    p_pascals: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    p_kappa_interface: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    p_kappa: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    edge_height_above_surface: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    layer_height_above_surface: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    layer_thickness: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    saturation_specific_humidity: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dsaturation_specific_humidity: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    temporary_3d: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    temporary_2d: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    my_value_is_1_2d: Local = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
