import dataclasses

from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Float64


@dataclasses.dataclass
class GFDLMPV3Locals(LocalState):
    ccn: Local = dataclasses.field(
        metadata={
            "name": "ccn",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    cin: Local = dataclasses.field(
        metadata={
            "name": "cin",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    cloud_fraction: Local = dataclasses.field(
        metadata={
            "name": "cloud_fraction",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    convection_fraction: Local = dataclasses.field(
        metadata={
            "name": "convection_fraction",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    cpaut: Local = dataclasses.field(
        metadata={
            "name": "cpaut",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    density: Local = dataclasses.field(
        metadata={
            "name": "density",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    density_factor: Local = dataclasses.field(
        metadata={
            "name": "density_factor",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dp: Local = dataclasses.field(
        metadata={
            "name": "dp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dry_dp: Local = dataclasses.field(
        metadata={
            "name": "dry_dp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dz: Local = dataclasses.field(
        metadata={
            "name": "dz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    factor_eis: Local = dataclasses.field(
        metadata={
            "name": "factor_eis",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    factor_rc: Local = dataclasses.field(
        metadata={
            "name": "factor_rc",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    h_var: Local = dataclasses.field(
        metadata={
            "name": "h_var",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppag: Local = dataclasses.field(
        metadata={
            "name": "mppag",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppar: Local = dataclasses.field(
        metadata={
            "name": "mppar",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppas: Local = dataclasses.field(
        metadata={
            "name": "mppas",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppcw: Local = dataclasses.field(
        metadata={
            "name": "mppcw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppd1: Local = dataclasses.field(
        metadata={
            "name": "mppd1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppdg: Local = dataclasses.field(
        metadata={
            "name": "mppdg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppdi: Local = dataclasses.field(
        metadata={
            "name": "mppdi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppds: Local = dataclasses.field(
        metadata={
            "name": "mppds",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppe1: Local = dataclasses.field(
        metadata={
            "name": "mppe1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpper: Local = dataclasses.field(
        metadata={
            "name": "mpper",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppew: Local = dataclasses.field(
        metadata={
            "name": "mppew",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppfr: Local = dataclasses.field(
        metadata={
            "name": "mppfr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppfw: Local = dataclasses.field(
        metadata={
            "name": "mppfw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm1: Local = dataclasses.field(
        metadata={
            "name": "mppm1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm2: Local = dataclasses.field(
        metadata={
            "name": "mppm2",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm3: Local = dataclasses.field(
        metadata={
            "name": "mppm3",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppmg: Local = dataclasses.field(
        metadata={
            "name": "mppmg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppmi: Local = dataclasses.field(
        metadata={
            "name": "mppmi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppms: Local = dataclasses.field(
        metadata={
            "name": "mppms",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpprg: Local = dataclasses.field(
        metadata={
            "name": "mpprg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpprs: Local = dataclasses.field(
        metadata={
            "name": "mpprs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpps1: Local = dataclasses.field(
        metadata={
            "name": "mpps1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppsg: Local = dataclasses.field(
        metadata={
            "name": "mppsg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppsi: Local = dataclasses.field(
        metadata={
            "name": "mppsi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppss: Local = dataclasses.field(
        metadata={
            "name": "mppss",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxg: Local = dataclasses.field(
        metadata={
            "name": "mppxg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxr: Local = dataclasses.field(
        metadata={
            "name": "mppxr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxs: Local = dataclasses.field(
        metadata={
            "name": "mppxs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    one_minus_sigma: Local = dataclasses.field(
        metadata={
            "name": "one_minus_sigma",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    p_thickness: Local = dataclasses.field(
        metadata={
            "name": "p_thickness",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    surface_type: Local = dataclasses.field(
        metadata={
            "name": "surface_type",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    reflectivity: Local = dataclasses.field(
        metadata={
            "name": "reflectivity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "dBZ",
            "dtype": Float,
        }
    )
    t: Local = dataclasses.field(
        metadata={
            "name": "t",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    tracer_dilution_adjustment: Local = dataclasses.field(
        metadata={
            "name": "tracer_dilution_adjustment",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    u: Local = dataclasses.field(
        metadata={
            "name": "u",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    v: Local = dataclasses.field(
        metadata={
            "name": "v",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    w: Local = dataclasses.field(
        metadata={
            "name": "w",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )

    @dataclasses.dataclass
    class MixingRatio:
        vapor: Local = dataclasses.field(
            metadata={
                "name": "vapor",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
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
    class TotalEnergy:
        b_beg_d: Local = dataclasses.field(
            metadata={
                "name": "b_beg_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_beg_m: Local = dataclasses.field(
            metadata={
                "name": "b_beg_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_d: Local = dataclasses.field(
            metadata={
                "name": "b_end_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_m: Local = dataclasses.field(
            metadata={
                "name": "b_end_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_d: Local = dataclasses.field(
            metadata={
                "name": "beg_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_m: Local = dataclasses.field(
            metadata={
                "name": "beg_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        delta: Local = dataclasses.field(
            metadata={
                "name": "delta",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_d: Local = dataclasses.field(
            metadata={
                "name": "end_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_m: Local = dataclasses.field(
            metadata={
                "name": "end_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        loss: Local = dataclasses.field(
            metadata={
                "name": "loss",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        magnitude: Local = dataclasses.field(
            metadata={
                "name": "magnitude",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )

    @dataclasses.dataclass
    class TotalWater:
        b_beg_d: Local = dataclasses.field(
            metadata={
                "name": "b_beg_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_beg_m: Local = dataclasses.field(
            metadata={
                "name": "b_beg_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_d: Local = dataclasses.field(
            metadata={
                "name": "b_end_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_m: Local = dataclasses.field(
            metadata={
                "name": "b_end_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_d: Local = dataclasses.field(
            metadata={
                "name": "beg_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_m: Local = dataclasses.field(
            metadata={
                "name": "beg_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_d: Local = dataclasses.field(
            metadata={
                "name": "end_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_m: Local = dataclasses.field(
            metadata={
                "name": "end_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )

    mixing_ratio: MixingRatio
    total_energy: TotalEnergy
    total_water: TotalWater
