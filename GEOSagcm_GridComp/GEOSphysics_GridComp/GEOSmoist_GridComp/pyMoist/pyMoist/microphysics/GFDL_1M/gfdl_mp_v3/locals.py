import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Float64


@dataclasses.dataclass
class GFDLMPV3Locals(State):
    ccn: Quantity = dataclasses.field(
        metadata={
            "name": "ccn",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    cin: Quantity = dataclasses.field(
        metadata={
            "name": "cin",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "name": "cloud_fraction",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    condensate: Quantity = dataclasses.field(
        metadata={
            "name": "condensate",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    convection_fraction: Quantity = dataclasses.field(
        metadata={
            "name": "convection_fraction",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    cpaut: Quantity = dataclasses.field(
        metadata={
            "name": "cpaut",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    density: Quantity = dataclasses.field(
        metadata={
            "name": "density",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    density_factor: Quantity = dataclasses.field(
        metadata={
            "name": "density_factor",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dry_dp: Quantity = dataclasses.field(
        metadata={
            "name": "dry_dp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dz: Quantity = dataclasses.field(
        metadata={
            "name": "dz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    factor_eis: Quantity = dataclasses.field(
        metadata={
            "name": "factor_eis",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    factor_rc: Quantity = dataclasses.field(
        metadata={
            "name": "factor_rc",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    h_var: Quantity = dataclasses.field(
        metadata={
            "name": "h_var",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    kappa: Quantity = dataclasses.field(
        metadata={
            "name": "kappa",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    moist_dp_original: Quantity = dataclasses.field(
        metadata={
            "name": "moist_dp_original",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    moist_dp_end: Quantity = dataclasses.field(
        metadata={
            "name": "moist_dp_end",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppag: Quantity = dataclasses.field(
        metadata={
            "name": "mppag",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppar: Quantity = dataclasses.field(
        metadata={
            "name": "mppar",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppas: Quantity = dataclasses.field(
        metadata={
            "name": "mppas",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppcw: Quantity = dataclasses.field(
        metadata={
            "name": "mppcw",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppd1: Quantity = dataclasses.field(
        metadata={
            "name": "mppd1",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppdg: Quantity = dataclasses.field(
        metadata={
            "name": "mppdg",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppdi: Quantity = dataclasses.field(
        metadata={
            "name": "mppdi",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppds: Quantity = dataclasses.field(
        metadata={
            "name": "mppds",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppe1: Quantity = dataclasses.field(
        metadata={
            "name": "mppe1",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpper: Quantity = dataclasses.field(
        metadata={
            "name": "mpper",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppew: Quantity = dataclasses.field(
        metadata={
            "name": "mppew",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppfr: Quantity = dataclasses.field(
        metadata={
            "name": "mppfr",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppfw: Quantity = dataclasses.field(
        metadata={
            "name": "mppfw",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm1: Quantity = dataclasses.field(
        metadata={
            "name": "mppm1",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm2: Quantity = dataclasses.field(
        metadata={
            "name": "mppm2",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm3: Quantity = dataclasses.field(
        metadata={
            "name": "mppm3",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppmg: Quantity = dataclasses.field(
        metadata={
            "name": "mppmg",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppmi: Quantity = dataclasses.field(
        metadata={
            "name": "mppmi",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppms: Quantity = dataclasses.field(
        metadata={
            "name": "mppms",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpprg: Quantity = dataclasses.field(
        metadata={
            "name": "mpprg",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpprs: Quantity = dataclasses.field(
        metadata={
            "name": "mpprs",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpps1: Quantity = dataclasses.field(
        metadata={
            "name": "mpps1",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppsg: Quantity = dataclasses.field(
        metadata={
            "name": "mppsg",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppsi: Quantity = dataclasses.field(
        metadata={
            "name": "mppsi",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppss: Quantity = dataclasses.field(
        metadata={
            "name": "mppss",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxg: Quantity = dataclasses.field(
        metadata={
            "name": "mppxg",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxr: Quantity = dataclasses.field(
        metadata={
            "name": "mppxr",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxs: Quantity = dataclasses.field(
        metadata={
            "name": "mppxs",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    one_minus_sigma: Quantity = dataclasses.field(
        metadata={
            "name": "one_minus_sigma",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    p: Quantity = dataclasses.field(
        metadata={
            "name": "p_thickness",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    surface_type: Quantity = dataclasses.field(
        metadata={
            "name": "surface_type",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    reflectivity: Quantity = dataclasses.field(
        metadata={
            "name": "reflectivity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "dBZ",
            "dtype": Float,
        }
    )
    t: Quantity = dataclasses.field(
        metadata={
            "name": "t",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_liquid: Quantity = dataclasses.field(
        metadata={
            "name": "total_liquid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    total_solid: Quantity = dataclasses.field(
        metadata={
            "name": "total_solid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    tracer_dilution_adjustment: Quantity = dataclasses.field(
        metadata={
            "name": "tracer_dilution_adjustment",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    u: Quantity = dataclasses.field(
        metadata={
            "name": "u",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    v: Quantity = dataclasses.field(
        metadata={
            "name": "v",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    w: Quantity = dataclasses.field(
        metadata={
            "name": "w",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )

    @dataclasses.dataclass
    class MixingRatio:
        vapor: Quantity = dataclasses.field(
            metadata={
                "name": "vapor",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )
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
    class TotalEnergy:
        b_beg_d: Quantity = dataclasses.field(
            metadata={
                "name": "b_beg_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_beg_m: Quantity = dataclasses.field(
            metadata={
                "name": "b_beg_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_d: Quantity = dataclasses.field(
            metadata={
                "name": "b_end_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_m: Quantity = dataclasses.field(
            metadata={
                "name": "b_end_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_d: Quantity = dataclasses.field(
            metadata={
                "name": "beg_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_m: Quantity = dataclasses.field(
            metadata={
                "name": "beg_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        delta: Quantity = dataclasses.field(
            metadata={
                "name": "delta",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_d: Quantity = dataclasses.field(
            metadata={
                "name": "end_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_m: Quantity = dataclasses.field(
            metadata={
                "name": "end_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        loss: Quantity = dataclasses.field(
            metadata={
                "name": "loss",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        magnitude: Quantity = dataclasses.field(
            metadata={
                "name": "magnitude",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class TotalWater:
        b_beg_d: Quantity = dataclasses.field(
            metadata={
                "name": "b_beg_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_beg_m: Quantity = dataclasses.field(
            metadata={
                "name": "b_beg_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_d: Quantity = dataclasses.field(
            metadata={
                "name": "b_end_d",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        b_end_m: Quantity = dataclasses.field(
            metadata={
                "name": "b_end_m",
                "dims": [I_DIM, J_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_d: Quantity = dataclasses.field(
            metadata={
                "name": "beg_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        beg_m: Quantity = dataclasses.field(
            metadata={
                "name": "beg_m",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_d: Quantity = dataclasses.field(
            metadata={
                "name": "end_d",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "?",
                "dtype": Float64,
            }
        )
        end_m: Quantity = dataclasses.field(
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
