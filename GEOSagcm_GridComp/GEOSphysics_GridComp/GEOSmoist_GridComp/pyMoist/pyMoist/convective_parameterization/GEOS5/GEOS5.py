"""GEOS5 convection parameterization interface"""

import numpy as np
from gt4py.cartesian.gtscript import i32

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
from pyMoist.convective_parameterization.GEOS5.call_minor_stencils import zero_output
from pyMoist.convective_parameterization.GEOS5.init_functions import outputs, temporaries

#used only during construction
from pyMoist.convective_parameterization.GEOS5.GEOS5_flags import GEOS5_flags

class GEOS5:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        GEOS5_flags: GEOS5_flags,
    ):
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory
        self.GEOS5_flags = GEOS5_flags

        self.outputs = outputs(self, quantity_factory)
        self.temporaries = temporaries(self, quantity_factory)

        orchestrate(obj=self, config=stencil_factory.config.dace_config)
        self._get_last = self.stencil_factory.from_dims_halo(
            func=zero_output,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._get_last = self.stencil_factory.from_dims_halo(
            func=zero_output,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            externals = {
                "kend" = kend,
                "USE_SCALE_DEP" = GEOS5_flags.USE_SCALE_DEP,
            }
        )
    
    # def __call__:
    
    #     self._zero_output(

    #     )