from ndsl.dsl.gt4py import PARALLEL, computation, exp, function, interval, log, sqrt, FORWARD, K
from ndsl.dsl.typing import Bool, Float, Float64, FloatField, Int, FloatField64, FloatFieldIJ

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3TableL3xL10, GFDLMPV3TableL5, GFDLMPV3TableL4
from pyMoist.microphysics.GFDL_1M.microphysics.constants import C_LIQ, ONE_R8, QCMIN, QFMIN, QPMIN, RGRAV, RHOW, RVGAS, TCOND, TICE, VDIFU
from pyMoist.shared.cloud_processes import ice_fraction
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3SaturationTable
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_table_functions import saturation_specific_humidity


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
    acco: GFDLMPV3TableL3xL10,
    acco_column: Int,
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
        tmp = tmp + acco.A[i, acco_column] * exp((6 + acc1 - i + 1) * log(t1)) * exp((acc2 + i) * log(t2))
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
    """Calculate moist heat capacities and latent heat coefficients at 0 C - function form

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
    """Calculate moist heat capacities and latent heat coefficients at 0 C - stencil form

    Args:
        t (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        total_liquid (FloatField)
        total_solid (FloatField)
        cvm (FloatField)
        total_energy (FloatField)
        lcpk (FloatField)
        icpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
    """
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


def p_bigg(
    t: FloatField64,
    dry_dp: FloatField,
    density: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    ccn: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    mppfw: FloatFieldIJ,
):
    """Bigg freezing mechanism, Bigg (1953)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        density (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        ccn (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        mppfw (FloatFieldIJ)
    """
    from __externals__ import CONV_FACTOR, D1_ICE, D1_VAP, DO_BIGG, DO_PSD_WATER_NUM, DO_QA, DT, LI00, LI20, LV00, MUW, PCAW, PCBW, T_WFR

    with computation(PARALLEL), interval(...):
        if DO_BIGG:
            tc = TICE - t

            if tc > 0 and liquid > QCMIN:
                if DO_PSD_WATER_NUM:
                    ccn = calc_particle_concentration(liquid, density, MUW, PCAW, PCBW)
                    ccn = ccn / density

                # Homogeneous freezing limit applied here
                if tc >= 40.0:
                    # Colder than -40C: ALL cloud liquid freezes instantaneously.
                    sink = liquid
                else:
                    # Warmer than -40C: Calculate probabilistic Bigg freezing normally
                    sink = 100.0 / (RHOW * ccn) * DT * (exp(0.66 * tc) - 1.0) * liquid**2

                sink = min(liquid, sink, tc / icpk)
                mppfw = mppfw + sink * dry_dp * CONV_FACTOR

                t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                    cloud_fraction=cloud_fraction,
                    vapor=vapor,
                    ice=ice,
                    liquid=liquid,
                    graupel=graupel,
                    rain=rain,
                    snow=snow,
                    dvapor=0.0,
                    dice=sink,
                    dliquid=-sink,
                    dgraupel=0.0,
                    drain=0.0,
                    dsnow=0.0,
                    DO_QA=DO_QA,
                    D1_VAP=D1_VAP,
                    D1_ICE=D1_ICE,
                    LI00=LI00,
                    LI20=LI20,
                    LV00=LV00,
                    T_WFR=T_WFR,
                )


def p_complete_freezing(
    t: FloatField64,
    dry_dp: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    mppfw: FloatFieldIJ,
):
    """enforce complete freezing below t_wfr, Lin et al. (1983)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        mppfw (FloatFieldIJ)
    """
    from __externals__ import (
        CONV_FACTOR,
        D1_ICE,
        D1_VAP,
        DO_QA,
        LI00,
        LI20,
        LV00,
        T_WFR,
    )

    with computation(PARALLEL), interval(...):
        tc = T_WFR - t

        if tc > 0.0 and liquid > QCMIN:
            sink = liquid * tc / DT_FR
            sink = min(liquid, sink, tc / icpk)
            mppfw = mppfw + sink * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=sink,
                dliquid=-sink,
                dgraupel=0.0,
                drain=0.0,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


def p_graupel_deposition_and_sublimation(
    t: FloatField64,
    dry_dp: FloatField,
    density: FloatField,
    density_factor: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    mppdg: FloatFieldIJ,
    mppsg: FloatFieldIJ,
    # tables
    CGSUB: GFDLMPV3TableL5,
    table_2: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """graupel deposition and sublimation, Lin et al. (1983)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        density (FloatField)
        density_factor (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        mppdg (FloatFieldIJ)
        mppsg (FloatFieldIJ)
        CGSUB (GFDLMPV3TableL5)
        table_2 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """
    from __externals__ import BLING, BLINH, CONV_FACTOR, D1_ICE, D1_VAP, DO_HAIL, DO_QA, DT, GS_FAC, LI00, LI20, LV00, MUG, MUH, T_SUB, T_WFR

    with computation(PARALLEL), interval(...):
        if graupel > QPMIN:
            t_in = t
            ice_saturation_humidity, dice_saturation_humidity = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            graupel_x_density = graupel * density
            t_squared = t * t
            dq = ice_saturation_humidity - vapor
            if DO_HAIL:
                pgsub = p_sublimation(
                    t_squared,
                    dq,
                    graupel_x_density,
                    ice_saturation_humidity,
                    density,
                    density_factor,
                    BLINH,
                    MUH,
                    tcpk,
                    cvm,
                    CGSUB,
                )
            else:
                pgsub = p_sublimation(
                    t_squared,
                    dq,
                    graupel_x_density,
                    ice_saturation_humidity,
                    density,
                    density_factor,
                    BLING,
                    MUG,
                    tcpk,
                    cvm,
                    CGSUB,
                )

            pgsub = DT * pgsub
            dq = dq / (1.0 + tcpk * dice_saturation_humidity)
            if pgsub > 0.0:
                sink = min(pgsub * min(1.0, max(t - T_SUB, 0.0) * GS_FAC), qg)
                mppsg = mppsg + sink * dry_dp * CONV_FACTOR
            else:
                sink = 0.0
                if t <= TICE:
                    sink = max(pgsub, dq, (t - TICE) / tcpk)
                mppdg = mppdg - sink * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=sink,
                dice=0.0,
                dliquid=0.0,
                dgraupel=-sink,
                drain=0.0,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


def p_ice_deposition_and_sublimation(
    t: FloatField64,
    dry_dp: FloatField,
    density: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    cin: FloatField,
    rsubl: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    one_minus_sigma: FloatFieldIJ,
    mppdi: FloatFieldIJ,
    mppsi: FloatFieldIJ,
    # tables
    table_2: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """cloud ice deposition and sublimation, Hong et al. (2004)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        density (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        cin (FloatField)
        rsubl (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        one_minus_sigma (FloatFieldIJ)
        mppdi (FloatFieldIJ)
        mppsi (FloatFieldIJ)
        table_2 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """
    from __externals__ import (
        CONV_FACTOR,
        D1_ICE,
        D1_VAP,
        DO_PSD_ICE_NUM,
        DO_QA,
        DT,
        IGFLAG,
        INFLAG,
        IS_FAC,
        LI00,
        LI20,
        LV00,
        MUI,
        PCAI,
        PCBI,
        PROG_CIN,
        QI_LIM,
        T_SUB,
        T_WFR,
    )

    with computation(PARALLEL), interval(...):
        if t < TICE:
            pidep = 0.0
            t_in = t
            ice_saturation_humidity, dice_saturation_humidity = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            dq = vapor - ice_saturation_humidity
            tmp = min(ice, dq / (1.0 + tcpk * dice_saturation_humidity))

            if ice > QCMIN:
                if DO_PSD_ICE_NUM:
                    cin = calc_particle_concentration(liquid, density, MUI, PCAI, PCBI)
                    cin = cin / density
                elif not PROG_CIN:
                    if INFLAG == 1:
                        cin = 5.38e7 * exp(0.75 * log(ice * density))
                    if INFLAG == 2:
                        cin = exp(-2.80 + 0.262 * (TICE - t)) * 1000.0
                    if INFLAG == 3:
                        cin = exp(-0.639 + 12.96 * (vapor / ice_saturation_humidity - 1.0)) * 1000.0
                    if INFLAG == 4:
                        cin = 5.0e-3 * exp(0.304 * (TICE - t)) * 1000.0
                    if INFLAG == 5:
                        cin = 1.0e-5 * exp(0.5 * (TICE - t)) * 1000.0
                pidep = (
                    DT
                    * dq
                    * 4.0
                    * 11.9
                    * exp(0.5 * log(ice * density * cin))
                    / (ice_saturation_humidity * density * (tcpk * cvm) ** 2 / (TCOND * RVGAS * t**2) + 1.0 / VDIFU)
                )

            if dq > 0.0:
                tc = TICE - t
                qi_gen = 4.92e-11 * exp(1.33 * log(1.0e3 * exp(0.1 * tc)))
                if IGFLAG == 1:
                    qi_crt = qi_gen / density
                if IGFLAG == 2:
                    qi_crt = qi_gen * min(QI_LIM, 0.1 * tc) / density
                if IGFLAG == 3:
                    qi_crt = 1.82e-6 * min(QI_LIM, 0.1 * tc) / density
                if IGFLAG == 4:
                    qi_crt = max(qi_gen, 1.82e-6) * min(QI_LIM, 0.1 * tc) / density
                sink = min(tmp, max(qi_crt - ice, pidep), tc / tcpk)
                mppdi = mppdi + sink * dry_dp * CONV_FACTOR
            else:
                pidep = pidep * min(1.0, max(t - T_SUB, 0.0) * IS_FAC)
                sink = max(pidep, tmp, -ice)
                sink = sink * one_minus_sigma  # resolution dependent subl 0:1 coarse:fine
                mppsi = mppsi - sink * dry_dp * CONV_FACTOR
                # 3D ice sublimation export
                rsubl = rsubl - sink * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=-sink,
                dice=sink,
                dliquid=0.0,
                dgraupel=0.0,
                drain=0.0,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


@function
def p_melt(
    t: Float,
    dcondensate: Float,
    condensate_x_density: Float,
    pxacw: Float,
    pxacr: Float,
    density: Float,
    density_factor: Float,
    blin: Float,
    mu: Float,
    lcpk: Float,
    icpk: Float,
    cvm: Float64,
    c: GFDLMPV3TableL4,
):
    return (c.A[0] / (icpk * cvm) * t / density - c.A[1] * lcpk / icpk * dcondensate) * exp((1 + mu) / (mu + 3) * log(6 * condensate_x_density)) * vent_coeff(
        density_factor, condensate_x_density, c.A[2], c.A[3], blin, mu
    ) + C_LIQ / (icpk * cvm) * t * (pxacw + pxacr)


def p_snow_deposition_and_sublimation(
    t: FloatField64,
    dry_dp: FloatField,
    density: FloatField,
    density_factor: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    mppds: FloatFieldIJ,
    mppss: FloatFieldIJ,
    # tables
    CSSUB: GFDLMPV3TableL5,
    table_2: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """snow deposition and sublimation, Lin et al. (1983)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        density (FloatField)
        density_factor (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        mppds (FloatFieldIJ)
        mppss (FloatFieldIJ)
        CSSUB (GFDLMPV3TableL5)
        table_2 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """
    from __externals__ import BLINS, CONV_FACTOR, D1_ICE, D1_VAP, DO_QA, DT, LI00, LI20, LV00, MUS, SS_FAC, T_SUB, T_WFR

    with computation(PARALLEL), interval(...):
        if snow > QPMIN:
            t_in = t
            ice_saturation_humidity, dice_saturation_humidity = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            snow_x_density = snow * density
            t_squared = t * t
            dq = ice_saturation_humidity - vapor
            pssub = p_sublimation(
                t_squared,
                dq,
                snow_x_density,
                ice_saturation_humidity,
                density,
                density_factor,
                BLINS,
                MUS,
                tcpk,
                cvm,
                CSSUB,
            )
            pssub = DT * pssub
            dq = dq / (1.0 + tcpk * dice_saturation_humidity)
            if pssub > 0.0:
                sink = min(pssub * min(1.0, max(t, T_SUB) * SS_FAC), snow)
                mppss = mppss + sink * dry_dp * CONV_FACTOR
            else:
                sink = 0.0
                if t <= TICE:
                    sink = max(pssub, dq, (t - TICE) / tcpk)
                mppds = mppds - sink * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=sink,
                dice=0.0,
                dliquid=0.0,
                dgraupel=0.0,
                drain=0.0,
                dsnow=-sink,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


@function
def p_sublimation(
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


def p_wbf(
    t: FloatField64,
    dry_dp: FloatField,
    density: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    cvm: FloatField64,
    icpk: FloatField,
    lcpk: FloatField,
    tcpk: FloatField,
    tcp3: FloatField,
    total_energy: FloatField64,
    one_minus_sigma: FloatFieldIJ,
    mppfw: FloatFieldIJ,
    # tables
    table_0: GFDLMPV3SaturationTable,
    table_2: GFDLMPV3SaturationTable,
    dtable_0: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """Wegener Bergeron Findeisen process, Storelvmo and Tan (2015)

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        density (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        cvm (FloatField64)
        icpk (FloatField)
        lcpk (FloatField)
        tcpk (FloatField)
        tcp3 (FloatField)
        total_energy (FloatField64)
        one_minus_sigma (FloatFieldIJ)
        mppfw (FloatFieldIJ)
        table_0 (GFDLMPV3SaturationTable)
        table_2 (GFDLMPV3SaturationTable)
        dtable_0 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """
    from __externals__ import CONV_FACTOR, D1_ICE, D1_VAP, DO_WBF, DO_QA, DT, LI00, LI20, LV00, PWBF_QI_CRT, TAU_WBF, T_WFR

    with computation(FORWARD), interval(0, 1):
        if DO_WBF:
            # internal parameters
            wbf_coarse_mult = 10.0  # how much slower WBF is at 50km vs 2km

    with computation(FORWARD), interval(0, 1):
        if DO_WBF:
            # -------------------------------------------------------------------
            # Scale tau_wbf:
            # If onemsig = 1.0 (2km),   tau_wbf_eff = tau_wbf
            # If onemsig = 0.0 (50km),  tau_wbf_eff = tau_wbf * wbf_coarse_mult
            # -------------------------------------------------------------------
            tau_wbf_eff = TAU_WBF * (wbf_coarse_mult * (1.0 - one_minus_sigma) + one_minus_sigma)

            # Calculate the time-step fraction using the effective timescale
            fac_wbf = 1.0 - exp(-DT / tau_wbf_eff)

    with computation(PARALLEL), interval(...):
        if DO_WBF:
            tc = TICE - t

            t_in = t
            liquid_saturation_humidity, _ = saturation_specific_humidity(t_in, density, table_0, dtable_0)
            ice_saturation_humidity, _ = saturation_specific_humidity(t_in, density, table_2, dtable_2)

            # heterogeneity and allow WBF to operate in large-scale updrafts
            # when the environment is supersaturated with respect to ice (qv > qsi)
            # and there is both liquid and ice present
            # Bypassed qi > qcmin constraint for colder temperatures to ensure initiation
            if tc > 0.0 and liquid > QCMIN and (ice > QCMIN or tc > 15.0) and vapor > ice_saturation_humidity:
                # 1. Homogeneous Freezing Limit (-40 C)
                if tc >= 40.0:
                    sink = liquid
                    tmp = 0.0  # All frozen liquid instantly becomes snow
                else:
                    # Normal WBF probabilistic freezing
                    sink = min(fac_wbf * liquid, tc / icpk)

                    # 2. Temperature-Dependent Snow Boost
                    # Scales from 1.0 (at 0 C) down to 0.0 (at -40 C)
                    # As tc gets larger (colder), the multiplier shrinks,
                    # reducing qim and forcing more mass to spill over into qs.
                    snow_boost_mult = max(0.0, 1.0 - (tc / 40.0))

                    qim = (PWBF_QI_CRT * snow_boost_mult) / density
                    tmp = min(sink, max(qim - ice, 0.0))

                mppfw = mppfw + sink * dry_dp * CONV_FACTOR

                t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                    cloud_fraction=cloud_fraction,
                    vapor=vapor,
                    ice=ice,
                    liquid=liquid,
                    graupel=graupel,
                    rain=rain,
                    snow=snow,
                    dvapor=0.0,
                    dice=tmp,
                    dliquid=-sink,
                    dgraupel=0.0,
                    drain=0.0,
                    dsnow=sink - tmp,
                    DO_QA=DO_QA,
                    D1_VAP=D1_VAP,
                    D1_ICE=D1_ICE,
                    LI00=LI00,
                    LI20=LI20,
                    LV00=LV00,
                    T_WFR=T_WFR,
                )


@function
def terminal_velocity_graupel_rain_snow(
    condensate: Float,
    density: Float,
    density_factor: Float,
    tva: Float64,
    tvb: Float64,
    blin: Float,
    mu: Float,
    v_min: Float,
    v_max: Float,
    v_fac: Float,
    const_v: Bool,
):
    """terminal velocity for rain, snow, and graupel, Lin et al. (1983) - function form
    """terminal velocity for rain, snow, and graupel, Lin et al. (1983) - function form

    Args:
        condensate (FloatField)
        density (FloatField)
        density_factor (FloatField)
        tva (Float64)
        tvb (Float64)
        blin (Float)
        mu (Float)
        v_min (Float)
        v_max (Float)
        v_fac (Float)
        const_v (Bool)
    """
    if const_v:
        terminal_velocity = 0.5 * (v_min + v_max)
    else:
        if condensate < QFMIN:
            terminal_velocity = calc_mass_weighted_terminal_velocity(condensate, density, mu, tva, tvb, blin)
            terminal_velocity = v_fac * terminal_velocity * density_factor
            terminal_velocity = min(v_max, max(v_min, terminal_velocity))

    return terminal_velocity


def terminal_velocity_graupel_rain_snow_wrapper(
    condensate: FloatField,
    density: FloatField,
    density_factor: FloatField,
    terminal_velocity: FloatField,
    tva: Float64,
    tvb: Float64,
    blin: Float,
    mu: Float,
    v_min: Float,
    v_max: Float,
    v_fac: Float,
    const_v: Bool,
):
    """terminal velocity for rain, snow, and graupel, Lin et al. (1983) - stencil form

    Args:
        condensate (FloatField)
        density (FloatField)
        density_factor (FloatField)
        terminal_velocity (FloatField)
        tva (Float64)
        tvb (Float64)
        blin (Float)
        mu (Float)
        v_min (Float)
        v_max (Float)
        v_fac (Float)
        const_v (Bool)
    """
    with computation(PARALLEL), interval(...):
        terminal_velocity = terminal_velocity_graupel_rain_snow(
            condensate,
            density,
            density_factor,
            terminal_velocity,
            tva,
            tvb,
            blin,
            mu,
            v_min,
            v_max,
            v_fac,
            const_v,
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
