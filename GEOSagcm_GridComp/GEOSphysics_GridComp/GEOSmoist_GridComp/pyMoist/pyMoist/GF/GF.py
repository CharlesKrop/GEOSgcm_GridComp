"""GF convection parameterization interface"""

import numpy as np
from gt4py.cartesian.gtscript import i32

import pyMoist.GFDL_1M.GFDL_1M_driver.GFDL_1M_driver_constants as driver_constants
from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int


