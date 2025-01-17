"""GEOS5 convection parameterization interface"""

import numpy as np
from gt4py.cartesian.gtscript import i32

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
from pyMoist.convective_parameterization.GF2020.setup_functions import outputs, temporaries, namelist_constants

#used only during construction
from pyMoist.convective_parameterization.GF2020.GF2020_flags import GF2020_flags
from pyMoist.convective_parameterization.GF2020.minor_stencils import setup_driver

class GEOS5:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        GF2020_flags: GF2020_flags,
    ):
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory
        self.GEOS5_flags = GF2020_flags

        self.outputs = outputs(self, quantity_factory)
        self.temporaries = temporaries(self, quantity_factory)
        self.namelist_constants = namelist_constants(self, GF2020_flags.MAXIENS)

        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self._setup_driver = self.stencil_factory.from_dims_halo(
            func=setup_driver,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            externals = {
                "USE_SCALE_DEP" = GF2020_flags.USE_SCALE_DEP,
                "N_TRACERS" = GF2020_flags.N_TRACERS,
            }
        )
    
    # def __call__:
    
    #     self._zero_output(

    #     )