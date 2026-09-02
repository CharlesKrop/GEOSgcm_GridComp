import dataclasses

from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int


@dataclasses.dataclass
class GFDL1MLocals(LocalState):
    reflectivity: Local = dataclasses.field(
        metadata={
            "name": "reflectivity",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "dBZ",
            "dtype": Float,
        }
    )
    p_interface_mb: Local = dataclasses.field(
        metadata={
            "name": "p_interface_mb",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    p_mb: Local = dataclasses.field(
        metadata={
            "name": "p_mb",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    edge_height_above_surface: Local = dataclasses.field(
        metadata={
            "name": "edge_height_above_surface",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    layer_height_above_surface: Local = dataclasses.field(
        metadata={
            "name": "layer_height_above_surface",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    layer_thickness: Local = dataclasses.field(
        metadata={
            "name": "layer_thickness",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    layer_thickness_negative: Local = dataclasses.field(
        metadata={
            "name": "layer_thickness_negative",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    dp: Local = dataclasses.field(
        metadata={
            "name": "dp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa",
            "dtype": Float,
        }
    )
    mass: Local = dataclasses.field(
        metadata={
            "name": "mass",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg m-2",
            "dtype": Float,
        }
    )
    mass_inverse: Local = dataclasses.field(
        metadata={
            "name": "mass_inverse",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg m-2",
            "dtype": Float,
        }
    )
    u_unmodified: Local = dataclasses.field(
        metadata={
            "name": "u_unmodified",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m s-1",
            "dtype": Float,
        }
    )
    v_unmodified: Local = dataclasses.field(
        metadata={
            "name": "v_unmodified",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m s-1",
            "dtype": Float,
        }
    )
    saturation_specific_humidity: Local = dataclasses.field(
        metadata={
            "name": "saturation_specific_humidity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dsaturation_specific_humidity: Local = dataclasses.field(
        metadata={
            "name": "dsaturation_specific_humidity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dcondensatedt: Local = dataclasses.field(
        metadata={
            "name": "dcondensate_dt",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    temporary_3d: Local = dataclasses.field(
        metadata={
            "name": "temporary_3d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "N/A",
            "dtype": Float,
        }
    )
    lcl_level: Local = dataclasses.field(
        metadata={
            "name": "lcl_level",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Int,
        }
    )
    lid_level: Int = -999
