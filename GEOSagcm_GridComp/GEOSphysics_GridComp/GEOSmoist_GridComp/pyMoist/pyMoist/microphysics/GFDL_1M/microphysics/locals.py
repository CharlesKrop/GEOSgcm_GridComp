import dataclasses

from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int


@dataclasses.dataclass
class GFDLMPV3Locals(LocalState):
    mppcw: Local = dataclasses.field(
        metadata={
            "name": "mppcw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppew: Local = dataclasses.field(
        metadata={
            "name": "mppew",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppe1: Local = dataclasses.field(
        metadata={
            "name": "mppe1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpper: Local = dataclasses.field(
        metadata={
            "name": "mpper",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppdi: Local = dataclasses.field(
        metadata={
            "name": "mppdi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppd1: Local = dataclasses.field(
        metadata={
            "name": "mppd1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppds: Local = dataclasses.field(
        metadata={
            "name": "mppds",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppdg: Local = dataclasses.field(
        metadata={
            "name": "mppdg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppsi: Local = dataclasses.field(
        metadata={
            "name": "mppsi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpps1: Local = dataclasses.field(
        metadata={
            "name": "mpps1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppss: Local = dataclasses.field(
        metadata={
            "name": "mppss",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppsg: Local = dataclasses.field(
        metadata={
            "name": "mppsg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppfw: Local = dataclasses.field(
        metadata={
            "name": "mppfw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppfr: Local = dataclasses.field(
        metadata={
            "name": "mppfr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppar: Local = dataclasses.field(
        metadata={
            "name": "mppar",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppas: Local = dataclasses.field(
        metadata={
            "name": "mppas",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppag: Local = dataclasses.field(
        metadata={
            "name": "mppag",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpprs: Local = dataclasses.field(
        metadata={
            "name": "mpprs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mpprg: Local = dataclasses.field(
        metadata={
            "name": "mpprg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppxr: Local = dataclasses.field(
        metadata={
            "name": "mppxr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppxs: Local = dataclasses.field(
        metadata={
            "name": "mppxs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppxg: Local = dataclasses.field(
        metadata={
            "name": "mppxg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppmi: Local = dataclasses.field(
        metadata={
            "name": "mppmi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppms: Local = dataclasses.field(
        metadata={
            "name": "mppms",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppmg: Local = dataclasses.field(
        metadata={
            "name": "mppmg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppm1: Local = dataclasses.field(
        metadata={
            "name": "mppm1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppm2: Local = dataclasses.field(
        metadata={
            "name": "mppm2",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
    mppm3: Local = dataclasses.field(
        metadata={
            "name": "mppm3",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "millibars",
            "dtype": Float,
        }
    )
