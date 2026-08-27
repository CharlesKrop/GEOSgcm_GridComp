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
    lcl_level: Local = dataclasses.field(
        metadata={
            "name": "lcl_level",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Int,
        }
    )


@dataclasses.dataclass
class GFDL1MLocals_old(LocalState):

    total_concentration: Local = dataclasses.field(
        metadata={
            "name": "total_concentration",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )

    @dataclasses.dataclass
    class DriverTendencies(LocalState):
        dvapordt: Local = dataclasses.field(
            metadata={
                "name": "dvapordt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dliquiddt: Local = dataclasses.field(
            metadata={
                "name": "dliquiddt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        draindt: Local = dataclasses.field(
            metadata={
                "name": "draindt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dicedt: Local = dataclasses.field(
            metadata={
                "name": "dicedt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dsnowdt: Local = dataclasses.field(
            metadata={
                "name": "dsnowdt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dgraupeldt: Local = dataclasses.field(
            metadata={
                "name": "dgraupeldt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dcloudfractiondt: Local = dataclasses.field(
            metadata={
                "name": "dcloudfractiondt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dtdt: Local = dataclasses.field(
            metadata={
                "name": "dtdt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dudt: Local = dataclasses.field(
            metadata={
                "name": "dudt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        dvdt: Local = dataclasses.field(
            metadata={
                "name": "dvdt",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    driver_tendencies: DriverTendencies
