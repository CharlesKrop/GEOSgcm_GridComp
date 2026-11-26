import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float


@dataclasses.dataclass
class MPFullLocals(State):
    @dataclasses.dataclass
    class Precip(State):
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        liquid: Quantity = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class SurfacePrecip(State):
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        liquid: Quantity = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class TerminalVelocity(State):
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        liquid: Quantity = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
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
