from ndsl.dsl.gt4py import PARALLEL, computation, exp, function, interval, log, sqrt, FORWARD, K
from ndsl.dsl.typing import Bool, Float, Float64, FloatField, Int

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3TableL3, GFDLMPV3TableL5
from pyMoist.microphysics.GFDL_1M.microphysics.constants import ONE_R8, QCMIN, RGRAV, TICE
from pyMoist.shared.cloud_processes import ice_fraction


@function
def accretion_2d(
    condensate_x_density: Float,
    density_factor: Float,
    c: Float,
    blin: Float,
    mu: Float,
):
    """accretion function, Lin et al. (1983)"""
    return density_factor * c * exp((2 + mu + blin) / (mu + 3) * log(6 * condensate_x_density))


@function
def accretion_3d(
    v1: Float,
    v2: Float,
    condensate_1: Float,
    condensate_2: Float,
    density: Float,
    c: Float,
    acc1: Float,
    acc2: Float,
    acco: GFDLMPV3TableL3,
    VDIFFFLAG: Int,
):
    """accretion function, Lin et al. (1983)"""

    t1 = exp(1.0 / (acc1 + 3) * log(6 * condensate_1 * density))
    t2 = exp(1.0 / (acc2 + 3) * log(6 * condensate_2 * density))

    if VDIFFFLAG == 1:
        vdiff = abs(v1 - v2)
    if VDIFFFLAG == 2:
        vdiff = sqrt((1.20 * v1 - 0.95 * v2) ** 2.0 + 0.08 * v1 * v2)
    if VDIFFFLAG == 3:
        vdiff = sqrt((1.00 * v1 - 1.00 * v2) ** 2.0 + 0.04 * v1 * v2)

    accretion = c * vdiff / density

    tmp = 0
    i = 0
    while i <= 2:
        tmp = tmp + acco.A[i] * exp((6 + acc1 - i + 1) * log(t1)) * exp((acc2 + i) * log(t2))
        i += 1

    return accretion * tmp


@function
def calc_effective_diameter(
    condensate: Float,
    density: Float,
    mu: Float,
    eda: Float64,
    edb: Float64,
):
    return eda / edb * exp(1.0 / (mu + 3) * log(6 * density * condensate))


@function
def calc_mass_weighted_terminal_velocity(
    condensate: Float,
    density: Float,
    mu: Float,
    tva: Float64,
    tvb: Float64,
    blin: Float,
):
    return tva / tvb * exp(blin / (mu + 3) * log(6 * density * condensate))


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
    """Calculate moist heat capacities and latent heat coefficients at 0 C

    Args:
        t (Float64)
        vapor (Float)
        ice (Float)
        liquid (Float)
        graupel (Float)
        rain (Float)
        snow (Float)
        C1_VAP (Float64)
        C1_LIQ (Float64)
        C1_ICE (Float64)
        D1_ICE (Float64)
        D1_VAP (Float64)
        LI00 (Float64)
        LI20 (Float64)
        LV00 (Float64)
        T_WFR (Float)

    Returns:
        total_liquid (Float): total liquid water content (liquid + rain)
        total_solid (Float): total solid water content (ice + snow + graupel)
        cvm (Float64): moist heat capacity
        total_energy (Float64): total energy
        lcpk (Float64): latent heat coefficient for liquid
        icpk (Float64): latent heat coefficient for ice
        tcpk (Float64): combined ice + vapor latent heat coefficient
        tcp3 (Float64): combined ice + liquid latent heat coefficient
    """
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


def calc_mhc_lhc_wrapper(
    t: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    total_liquid: FloatField,
    total_solid: FloatField,
    cvm: FloatField,
    total_energy: FloatField,
    lcpk: FloatField,
    icpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
):
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, D1_ICE, D1_VAP, LI00, LI20, LV00, T_WFR

    with computation(PARALLEL), interval(...):
        total_liquid, total_solid, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = calc_mhc_lhc(
            t=t,
            vapor=vapor,
            ice=ice,
            liquid=liquid,
            graupel=graupel,
            rain=rain,
            snow=snow,
            C1_VAP=C1_VAP,
            C1_LIQ=C1_LIQ,
            C1_ICE=C1_ICE,
            D1_ICE=D1_ICE,
            D1_VAP=D1_VAP,
            LI00=LI00,
            LI20=LI20,
            LV00=LV00,
            T_WFR=T_WFR,
        )


@function
def calc_optical_extinction(
    condensate: Float,
    density: Float,
    mu: Float,
    oea: Float64,
    oeb: Float64,
):
    return oea / oeb * exp((mu + 2) / (mu + 3) * log(6 * density * condensate))


@function
def calc_particle_concentration(
    condensate: Float,
    density: Float,
    mu: Float,
    pca: Float64,
    pcb: Float64,
):
    return pca / pcb * exp(mu / (mu + 3) * log(6 * density * condensate))


@function
def calc_reflectivity_factor(
    condensate: Float,
    density: Float,
    mu: Float,
    rra: Float64,
    rrb: Float64,
):
    return rra / rrb * exp((mu + 6) / (mu + 3) * log(6 * density * condensate))


def linear_prof(
    precipitate: FloatField,
    dm: FloatField,
    h_var: FloatField,
):
    """vertical subgrid variability used for cloud ice and cloud water autoconversion
    edges: qe == qbar + / - dm

    Args:
        precipitate (FloatField)
        dm (FloatField)
        h_var (FloatField)
    """
    from __externals__ import IRAIN_F, Z_SLOPE, k_end

    with computation(FORWARD), interval(...):
        if IRAIN_F == 0 and Z_SLOPE and K >= 1:
            dprecipitate = 0.5 * (precipitate - precipitate[0, 0, -1])

    with computation(FORWARD), interval(0, 1):
        if IRAIN_F == 0 and Z_SLOPE:
            dm = 0.0

    with computation(FORWARD), interval(...):
        if IRAIN_F == 0 and Z_SLOPE:
            # use twice the strength of the positive definiteness limiter (Lin et al. 1994)
            dm = 0.5 * min(abs(dprecipitate + dprecipitate[0, 0, 1]), 0.5 * precipitate)
            if dprecipitate * dprecipitate[0, 0, 1] <= 0.0:
                if dprecipitate > 0.0:
                    dm = min(dm, dprecipitate, -dprecipitate[0, 0, 1])
                else:
                    dm = 0.0

    with computation(FORWARD), interval(-1, None):
        if IRAIN_F == 0 and Z_SLOPE:
            dm = 0.0

    with computation(FORWARD), interval(...):
        if IRAIN_F == 0 and Z_SLOPE:
            # impose a presumed background horizontal variability that is proportional to the value itself
            dm = max(dm, 0.0, h_var * precipitate)

    with computation(FORWARD), interval(...):
        if not (IRAIN_F == 0 and Z_SLOPE):
            dm = max(0.0, h_var * precipitate)


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
def new_ice_condensate(
    t: Float,
    liquid: Float,
    ice: Float,
    convection_fraction: Float,
    surface_type: Float,
):
    ifrac = ice_fraction(t, convection_fraction, surface_type)
    new_ice_condensate = min(max(0.0, ifrac * (liquid + ice) - ice), liquid)
    return new_ice_condensate


@function
def new_liquid_condensate(
    t: Float,
    liquid: Float,
    ice: Float,
    convection_fraction: Float,
    surface_type: Float,
):
    ifrac = ice_fraction(t, convection_fraction, surface_type)
    new_liq_condensate = min(max(0.0, (1.0 - ifrac) * (liquid + ice) - liquid), ice)
    return new_liq_condensate


@function
def p_sub(
    t_squared: Float,
    dcondensate: Float,
    condensate_x_density: Float,
    saturation_specific_humidity: Float,
    density: Float,
    density_factor: Float,
    blin: Float,
    mu: Float,
    cpk: Float,
    cvm: Float,
    c: GFDLMPV3TableL5,
):
    """sublimation or deposition function, Lin et al. (1983)

    Args:
        t_squared (Float)
        dcondensate (Float)
        condensate_x_density (Float)
        saturation_specific_humidity (Float)
        density (Float)
        density_factor (Float)
        blin (Float)
        mu (Float)
        cpk (Float)
        cvm (Float)
        c (GFDLMPV3TableL5)

    Returns:
        (Float): sublimation or deposition rate
    """
    return (
        c.A[0]
        * t_squared
        * dcondensate
        * exp((1 + mu) / (mu + 3) * log(6 * condensate_x_density))
        * vent_coeff(condensate_x_density, c.A[1], c.A[2], density_factor, blin, mu)
        / (c.A[3] * t_squared + c.A[4] * (cpk * cvm) ** 2 * saturation_specific_humidity * density)
    )


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


@function
def vent_coeff(
    density_factor: Float,
    condensate_x_density: Float,
    c1: Float,
    c2: Float,
    blin: Float,
    mu: Float,
):
    """ventilation coefficient, Lin et al. (1983)

    Args:
        density_factor (Float)
        condensate_x_density (Float)
        c1 (Float)
        c2 (Float)
        blin (Float)
        mu (Float)

    Returns:
        (Float): ventilation coefficient
    """
    return c1 + c2 * exp((3 + 2 * mu + blin) / (mu + 3) / 2 * log(6 * condensate_x_density)) * sqrt(density_factor) / exp(
        (1 + mu) / (mu + 3) * log(6 * condensate_x_density)
    )
