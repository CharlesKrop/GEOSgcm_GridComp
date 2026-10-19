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
                "units": "?",
                "dtype": Float,
            }
        )
        liquid: Local = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        graupel: Local = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        rain: Local = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        snow: Local = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class SurfacePrecip(LocalState):
        ice: Local = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        liquid: Local = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        graupel: Local = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        rain: Local = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        snow: Local = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class TerminalVelocity(LocalState):
        ice: Local = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        liquid: Local = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        graupel: Local = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        rain: Local = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        snow: Local = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    precip: Precip
    surface_precip: SurfacePrecip
    terminal_velocity: TerminalVelocity
