from ndsl.dsl.gt4py import function, isnan
from ndsl.dsl.typing import Bool, Float, Float64

from pyMoist.microphysics.GFDL_1M.microphysics.constants import ONE_R8, QCMIN, RGRAV, TICE


@function
def moist_heat_capacity_3(vapor: Float, total_liquid: Float, total_solid: Float, C1_VAP: Float64, C1_LIQ: Float64, C1_ICE: Float64):
    """moist heat capacity, three input variables"""
    return Float64(ONE_R8 + vapor * C1_VAP + total_liquid * C1_LIQ + total_solid * C1_ICE)


@function
def moist_heat_capacity_4(dry: Float64, vapor: Float, total_liquid: Float, total_solid: Float, C1_VAP: Float64, C1_LIQ: Float64, C1_ICE: Float64):
    """moist heat capacity, four input variables"""
    return Float64(dry + vapor * C1_VAP + total_liquid * C1_LIQ + total_solid * C1_ICE)


@function
def moist_heat_capacity_6(vapor: Float, ice: Float, liquid: Float, graupel: Float, rain: Float, snow: Float, C1_VAP: Float64, C1_LIQ: Float64, C1_ICE: Float64):
    """moist heat capacity, six input variables"""
    total_liquid = liquid + rain
    total_solid = ice + snow + graupel
    return moist_heat_capacity_3(vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)


@function
def moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dp, C_AIR: Float, C1_VAP: Float64, C1_LIQ: Float64, C1_ICE: Float64, moist_q: Bool = False):
    total_liquid = liquid + rain
    total_solid = ice + snow + graupel
    total_condensates = total_liquid + total_solid
    con_r8 = Float64(ONE_R8 - (vapor + total_condensates))
    if moist_q:
        cvm = moist_heat_capacity_4(con_r8, vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)
    else:
        cvm = moist_heat_capacity_3(vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)
    return Float64(RGRAV * cvm * C_AIR * t * dp)


@function
def calc_mhc_lhc(
    t: Float64,
    vapor: Float,
    ice: Float,
    liquid: Float,
    graupel: Float,
    rain: Float,
    snow: Float,
    # constants
    C1_VAP: Float64,
    C1_LIQ: Float64,
    C1_ICE: Float64,
    D1_ICE: Float64,
    D1_VAP: Float64,
    LI00: Float64,
    LI20: Float64,
    LV00: Float64,
    T_WFR: Float,
):
    # ensure 64 bit
    cvm: Float64 = 0.0
    total_energy: Float64 = 0.0

    total_liquid = liquid + rain
    total_solid = ice + snow + graupel
    cvm = moist_heat_capacity_3(vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE)
    total_energy = cvm * t + LV00 * vapor - LI00 * total_solid
    lcpk = (LV00 + D1_VAP * t) / cvm
    icpk = (LI00 + D1_ICE * t) / cvm
    tcpk = (LI20 + (D1_VAP + D1_ICE) * t) / cvm
    tcp3 = lcpk + icpk * min(1.0, max(TICE - t, 0.0) / (TICE - T_WFR))

    return total_liquid, total_solid, cvm, total_energy, lcpk, icpk, tcpk, tcp3


@function
def update_hydrometeors(
    cloud_fraction: Float,
    vapor: Float,
    ice: Float,
    liquid: Float,
    graupel: Float,
    rain: Float,
    snow: Float,
    dvapor: Float,
    dice: Float,
    dliquid: Float,
    dgraupel: Float,
    drain: Float,
    dsnow: Float,
    # constants
    DO_QA: Bool,
):

    # save previous total condensate
    if not DO_QA:
        initial_condensate = max(liquid + ice, QCMIN)

    vapor = vapor + dvapor
    ice = ice + dice
    liquid = liquid + dliquid
    graupel = graupel + dgraupel
    rain = rain + drain
    snow = snow + dsnow

    # total new condensate / old condensate
    if not DO_QA:
        cloud_fraction = max(0.0, min(1.0, cloud_fraction * (liquid + ice) / initial_condensate))

    return vapor, ice, liquid, graupel, rain, snow, cloud_fraction


@function
def update_hydrometeors_and_temperature(
    cloud_fraction: Float,
    vapor: Float,
    ice: Float,
    liquid: Float,
    graupel: Float,
    rain: Float,
    snow: Float,
    dvapor: Float,
    dice: Float,
    dliquid: Float,
    dgraupel: Float,
    drain: Float,
    dsnow: Float,
    t: Float64,
    total_energy: Float64,
    # constants
    DO_QA: Bool,
    D1_VAP: Float64,
    D1_ICE: Float64,
    LI00: Float64,
    LI20: Float64,
    LV00: Float64,
    T_WFR: Float,
):

    # save previous total condensate
    if not DO_QA:
        initial_condensate = max(liquid + ice, QCMIN)

    vapor = vapor + dvapor
    ice = ice + dice
    liquid = liquid + dliquid
    graupel = graupel + dgraupel
    rain = rain + drain
    snow = snow + dsnow

    # total new condensate / old condensate
    if not DO_QA:
        cloud_fraction = max(0.0, min(1.0, cloud_fraction * (liquid + ice) / initial_condensate))

    cvm = moist_heat_capacity_6(vapor, ice, liquid, graupel, rain, snow)
    t = (total_energy - LV00 * vapor + LI00 * (ice + snow + graupel)) / cvm

    lcpk = (LV00 + D1_VAP * t) / cvm
    icpk = (LI00 + D1_ICE * t) / cvm
    tcpk = (LI20 + (D1_VAP + D1_ICE) * t) / cvm
    tcp3 = lcpk + icpk * min(1.0, max((TICE - t), 0.0) / (TICE - T_WFR))

    return t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3
