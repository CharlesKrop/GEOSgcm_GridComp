import numpy as np
import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    THIS_K,
)

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
from pyMoist.convective_parameterization.GEOS5.call_minor_stencils


def driver(
    
):
    with computation(FORWARD), interval(0,1):
        setup