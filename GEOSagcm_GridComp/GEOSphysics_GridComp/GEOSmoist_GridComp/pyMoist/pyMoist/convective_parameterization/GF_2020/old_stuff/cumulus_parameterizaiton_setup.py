import numpy as np
import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    THIS_K,
    sqrt,
    round,
    max,
)

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
import pyMoist.constants as global_constants
import pyMoist.convective_parameterization.GF2020.constants as GF2020_constants
import pyMoist.convective_parameterization.shared_constants as convection_constants

# used only during construction
from pyMoist.convective_parameterization.GF2020.GF2020_flags import GF2020_flags


def which_plumes(GF2020_flags: GF2020_flags):
    plume_order = []
    if GF2020_flags.SH_MD_DP == True:
        if GF2020_flags.SHALLOW == True:
            plume_order.append("shallow")
        if GF2020_flags.CONGESTUS == True:
            plume_order.append("mid")
        if GF2020_flags.DEEP == True:
            plume_order.append("deep")
    else:
        if GF2020_flags.SHALLOW == True:
            plume_order.append("shallow")
        if GF2020_flags.DEEP == True:
            plume_order.append("deep")
        if GF2020_flags.CONGESTUS == True:
            plume_order.append("mid")
    return plume_order


def get_constants(current_plume, GF2020_flags: GF2020_flags):
    if current_plume == "shallow":
        HEI_DOWN_LAND = GF2020_flags.HEI_DOWN_LAND_SH
        HEI_DOWN_OCEAN = GF2020_flags.HEI_DOWN_OCEAN_SH
        HEI_UPDF_LAND = GF2020_flags.HEI_UPDF_LAND_SH
        HEI_UPDF_OCEAN = GF2020_flags.HEI_UPDF_OCEAN_SH
        MIN_EDT_LAND = GF2020_flags.MIN_EDT_LAND_SH
        MIN_EDT_OCEAN = GF2020_flags.MIN_EDT_OCEAN_SH
        MAX_EDT_LAND = GF2020_flags.MAX_EDT_LAND_SH
        MAX_EDT_OCEAN = GF2020_flags.MAX_EDT_OCEAN_SH
        FADJ_MASSFLX = GF2020_flags.FADJ_MASSFLX_SH
        USE_EXCESS = GF2020_flags.USE_EXCESS_SH
        AVE_LAYER = GF2020_flags.AVE_LAYER_SH

    if current_plume == "mid":
        HEI_DOWN_LAND = GF2020_flags.HEI_DOWN_LAND_MD
        HEI_DOWN_OCEAN = GF2020_flags.HEI_DOWN_OCEAN_MD
        HEI_UPDF_LAND = GF2020_flags.HEI_UPDF_LAND_MD
        HEI_UPDF_OCEAN = GF2020_flags.HEI_UPDF_OCEAN_MD
        MIN_EDT_LAND = GF2020_flags.MIN_EDT_LAND_MD
        MIN_EDT_OCEAN = GF2020_flags.MIN_EDT_OCEAN_MD
        MAX_EDT_LAND = GF2020_flags.MAX_EDT_LAND_MD
        MAX_EDT_OCEAN = GF2020_flags.MAX_EDT_OCEAN_MD
        FADJ_MASSFLX = GF2020_flags.FADJ_MASSFLX_MD
        USE_EXCESS = GF2020_flags.USE_EXCESS_MD
        AVE_LAYER = GF2020_flags.AVE_LAYER_MD

    if current_plume == "deep":
        HEI_DOWN_LAND = GF2020_flags.HEI_DOWN_LAND_DP
        HEI_DOWN_OCEAN = GF2020_flags.HEI_DOWN_OCEAN_DP
        HEI_UPDF_LAND = GF2020_flags.HEI_UPDF_LAND_DP
        HEI_UPDF_OCEAN = GF2020_flags.HEI_UPDF_OCEAN_DP
        MIN_EDT_LAND = GF2020_flags.MIN_EDT_LAND_DP
        MIN_EDT_OCEAN = GF2020_flags.MIN_EDT_OCEAN_DP
        MAX_EDT_LAND = GF2020_flags.MAX_EDT_LAND_DP
        MAX_EDT_OCEAN = GF2020_flags.MAX_EDT_OCEAN_DP
        FADJ_MASSFLX = GF2020_flags.FADJ_MASSFLX_DP
        USE_EXCESS = GF2020_flags.USE_EXCESS_DP
        AVE_LAYER = GF2020_flags.AVE_LAYER_DP

    return (
        HEI_DOWN_LAND,
        HEI_DOWN_OCEAN,
        HEI_UPDF_LAND,
        HEI_UPDF_OCEAN,
        MIN_EDT_LAND,
        MIN_EDT_OCEAN,
        MAX_EDT_LAND,
        MAX_EDT_OCEAN,
        FADJ_MASSFLX,
        USE_EXCESS,
        AVE_LAYER,
    )


class temporaries:
    def __init__(self, quantity_factory):
        self.t_excess = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.q_excess = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
