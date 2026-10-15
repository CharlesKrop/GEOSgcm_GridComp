import dataclasses

from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float


@dataclasses.dataclass
class MPFullLocals(LocalState):
    @dataclasses.dataclass
    class Precip(LocalState):
        ice: Local = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        liquid: Local = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        graupel: Local = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        rain: Local = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        snow: Local = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )

    surface_precip_ice: Local = dataclasses.field(
        metadata={
            "name": "surface_precip_ice",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )

    @dataclasses.dataclass
    class SurfacePrecip(LocalState):
        liquid: Local = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )
        graupel: Local = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )
        rain: Local = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )
        snow: Local = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class TerminalVelocity(LocalState):
        ice: Local = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        liquid: Local = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        graupel: Local = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        rain: Local = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        snow: Local = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )

    precip: Precip
    surface_precip: SurfacePrecip
    terminal_velocity: TerminalVelocity
