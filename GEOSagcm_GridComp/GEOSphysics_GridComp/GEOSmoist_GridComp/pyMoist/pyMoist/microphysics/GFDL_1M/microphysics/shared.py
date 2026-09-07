from ndsl.dsl.gt4py import function
from ndsl.dsl.typing import Bool, Float64, Float
from pyMoist.microphysics.GFDL_1M.microphysics.constants import ONE_R8, RGRAV


@function
def moist_heat_capacity_3(vapor: Float, total_liquid: Float, total_solid: Float, C1_VAP: Float, C1_LIQ: Float, C1_ICE: Float):
    """moist heat capacity, three input variables"""
    return Float64(ONE_R8 + vapor * C1_VAP + total_liquid * C1_LIQ + total_solid * C1_ICE)


@function
def moist_heat_capacity_4(dry: Float64, vapor: Float, total_liquid: Float, total_solid: Float, C1_VAP: Float, C1_LIQ: Float, C1_ICE: Float):
    """moist heat capacity, four input variables"""
    return Float64(dry + vapor * C1_VAP + total_liquid * C1_LIQ + total_solid * C1_ICE)


@function
def moist_heat_capacity_6(vapor: Float, ice: Float, liquid: Float, graupel: Float, rain: Float, snow: Float, C1_VAP: Float, C1_LIQ: Float, C1_ICE: Float):
    """moist heat capacity, six input variables"""
    total_liquid = liquid + rain
    total_solid = ice + snow + graupel
    return moist_heat_capacity_3(vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)


@function
def moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dp, C_AIR: Float, C1_VAP: Float, C1_LIQ: Float, C1_ICE: Float, moist_q: Bool = False):
    total_liquid = liquid + rain
    total_solid = ice + snow + graupel
    total_condensates = total_liquid + total_solid
    con_r8 = Float64(ONE_R8 - (vapor + total_condensates))
    if moist_q:
        cvm = moist_heat_capacity_4(con_r8, vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)
    else:
        cvm = moist_heat_capacity_3(vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)
    return Float64(RGRAV * cvm * C_AIR * t * dp)
