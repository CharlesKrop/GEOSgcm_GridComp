import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int


@dataclasses.dataclass
class GFDL1MLocals(State):
    dcloud_fraction_dt: Quantity = dataclasses.field(
        metadata={
            "name": "dcloud_fraction_dt",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dp: Quantity = dataclasses.field(
        metadata={
            "name": "dp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa",
            "dtype": Float,
        }
    )
    dsaturation_specific_humidity: Quantity = dataclasses.field(
        metadata={
            "name": "dsaturation_specific_humidity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    edge_height_above_surface: Quantity = dataclasses.field(
        metadata={
            "name": "edge_height_above_surface",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    layer_height_above_surface: Quantity = dataclasses.field(
        metadata={
            "name": "layer_height_above_surface",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    layer_thickness: Quantity = dataclasses.field(
        metadata={
            "name": "layer_thickness",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    layer_thickness_negative: Quantity = dataclasses.field(
        metadata={
            "name": "layer_thickness_negative",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    lcl_level: Quantity = dataclasses.field(
        metadata={
            "name": "lcl_level",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Int,
        }
    )
    mass: Quantity = dataclasses.field(
        metadata={
            "name": "mass",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg m-2",
            "dtype": Float,
        }
    )
    mass_inverse: Quantity = dataclasses.field(
        metadata={
            "name": "mass_inverse",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg m-2",
            "dtype": Float,
        }
    )
    p_interface_mb: Quantity = dataclasses.field(
        metadata={
            "name": "p_interface_mb",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    p_mb: Quantity = dataclasses.field(
        metadata={
            "name": "p_mb",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    reflectivity: Quantity = dataclasses.field(
        metadata={
            "name": "reflectivity",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "dBZ",
            "dtype": Float,
        }
    )
    saturation_specific_humidity: Quantity = dataclasses.field(
        metadata={
            "name": "saturation_specific_humidity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    saturation_specific_humidity_liquid: Quantity = dataclasses.field(
        metadata={
            "name": "saturation_specific_humidity_liquid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1",
            "dtype": Float,
        }
    )
    saturation_specific_humidity_ice: Quantity = dataclasses.field(
        metadata={
            "name": "saturation_specific_humidity_ice",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1",
            "dtype": Float,
        }
    )
    temporary_3d: Quantity = dataclasses.field(
        metadata={
            "name": "temporary_3d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "N/A",
            "dtype": Float,
        }
    )
    temporary_3d_2: Quantity = dataclasses.field(
        metadata={
            "name": "temporary_3d_2",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "N/A",
            "dtype": Float,
        }
    )
    u_unmodified: Quantity = dataclasses.field(
        metadata={
            "name": "u_unmodified",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m s-1",
            "dtype": Float,
        }
    )
    v_unmodified: Quantity = dataclasses.field(
        metadata={
            "name": "v_unmodified",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m s-1",
            "dtype": Float,
        }
    )
    _dummy_field_no_read_no_write: Quantity = dataclasses.field(
        metadata={
            "name": "_dummy_field_no_read_no_write",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
