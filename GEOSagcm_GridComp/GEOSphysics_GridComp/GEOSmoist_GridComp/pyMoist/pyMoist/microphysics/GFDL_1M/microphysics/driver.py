from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, interval, sqrt, exp, log, log10
from ndsl.dsl.typing import Bool, Float, Float64, FloatField, FloatField64, FloatFieldIJ, FloatFieldIJ64
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import copy_2d
from ndsl.stencils.basic_operations import copy

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.constants import (
    C_ICE,
    C_LIQ,
    CP_AIR,
    CP_VAP,
    GRAV,
    ONE_R8,
    QCMIN,
    QPMIN,
    RC,
    RDGAS,
    RGRAV,
    RHOG,
    RHOH,
    RHOI,
    RHOR,
    RHOS,
    TICE,
    ZVIR,
)
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3Tables, GFDLMPV3SaturationTable
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_table_functions import saturation_specific_humidity
from pyMoist.microphysics.GFDL_1M.microphysics.shared import (
    calc_mhc_lhc,
    calc_reflectivity_factor,
    moist_heat_capacity_3,
    moist_heat_capacity_4,
    moist_heat_capacity_6,
    moist_total_energy,
    terminal_velocity_graupel_rain_snow,
    update_hydrometeors,
    update_hydrometeors_and_temperature,
)
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.shared.atmos_recipes import compute_estimated_inversion_strength_factor, sigma
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full import MPFull


def calculate_base_total_energy(
    t: FloatField64,
    dp: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    total_energy: FloatField,
):
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR, CONSV_TE, HYDROSTATIC

    with computation(PARALLEL), interval(...):
        if CONSV_TE:
            if HYDROSTATIC:
                total_energy = -C_AIR * t * dp

            else:
                total_energy = -moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, True) * GRAV


def calculate_total_energy_change(
    t: FloatField64,
    dp: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    total_energy: FloatField,
):
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR, CONSV_TE, HYDROSTATIC

    with computation(PARALLEL), interval(...):
        if CONSV_TE:
            if HYDROSTATIC:
                total_energy = total_energy + C_AIR * t * dp
            else:
                total_energy = total_energy + moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, True) * GRAV


def cloud_fraction(
    t: FloatField64,
    p: FloatField,
    density: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    area: FloatFieldIJ,
    h_var: FloatField,
    # tables
    table_0: GFDLMPV3SaturationTable,
    table_2: GFDLMPV3SaturationTable,
    dtable_0: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """cloud fraction diagnostic

    Args:
        t (FloatField64)
        p (FloatField)
        density (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        area (FloatFieldIJ)
        h_var (FloatField)
        table_0 (GFDLMPV3SaturationTable)
        table_2 (GFDLMPV3SaturationTable)
        dtable_0 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """
    from __externals__ import (
        C1_ICE,
        C1_LIQ,
        C1_VAP,
        CFFLAG,
        CLD_MIN,
        D1_ICE,
        D1_VAP,
        DO_CLD_ADJ,
        F_DQ_M,
        F_DQ_P,
        ICLOUD_F,
        LI00,
        LI20,
        LV00,
        RAD_GRAUPEL,
        RAD_RAIN,
        RAD_SNOW,
        RH_THRES,
        T_WFR,
        XR_A,
        XR_B,
        XR_C,
    )

    with computation(FORWARD), interval(0, 1):
        grid_size = sqrt(area)

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        cvm: FloatField64 = 0.0
        total_energy: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        # calculate moist heat capacity and latent heat coefficients
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

        # combine water species

        ice_internal = total_solid
        total_solid = ice
        if RAD_SNOW:
            total_solid = ice + snow
            if RAD_GRAUPEL:
                total_solid = ice + snow + graupel

        liquid_internal = total_liquid
        total_liquid = liquid
        if RAD_RAIN:
            total_liquid = liquid + rain

        total_condensate = total_solid + total_liquid
        total_water = vapor + total_condensate

        # use the "liquid - frozen water temperature" (tin) to compute saturated specific humidity

        ice_internal = ice_internal - total_solid
        liquid_internal = liquid_internal - total_liquid
        t_in = (total_energy - LV00 * total_water + LI00 * ice) / moist_heat_capacity_3(total_water, liquid_internal, ice, C1_VAP, C1_LIQ, C1_ICE)

        # calculate saturated specific humidity

        if t_in <= T_WFR:
            sat_spec_humidity, dsaturation_specific_humidity = saturation_specific_humidity(t_in, density, table_2, dtable_2)
        elif t_in >= TICE:
            sat_spec_humidity, dsaturation_specific_humidity = saturation_specific_humidity(t_in, density, table_0, dtable_0)
        else:
            ice_saturation_humidity, dice_saturation_humidity = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            liquid_saturation_humidity, dliquid_saturation_humidity = saturation_specific_humidity(t_in, density, table_0, dtable_0)
            if total_condensate > QCMIN:
                ratio_ice = total_solid / total_condensate
            else:
                ratio_ice = (TICE - t_in) / (TICE - T_WFR)

            sat_spec_humidity = ratio_ice * ice_saturation_humidity + (1.0 - ratio_ice) * liquid_saturation_humidity

        # cloud schemes

        rh = total_water / sat_spec_humidity

        if CFFLAG == 1:
            if rh > RH_THRES and total_water > QCMIN:
                dq = h_var * total_water
                if DO_CLD_ADJ:
                    water_plus = total_water + dq * F_DQ_P * min(1.0, max(0.0, (p - 200.0e2) / (1000.0e2 - 200.0e2)))
                else:
                    water_plus = total_water + dq * F_DQ_P

                water_minus = total_water - dq * F_DQ_M

                if ICLOUD_F == 2:
                    if sat_spec_humidity < total_water:
                        cloud_fraction = 1.0
                    else:
                        cloud_fraction = 0.0
                elif ICLOUD_F == 3:
                    if sat_spec_humidity < total_water:
                        cloud_fraction = 1.0
                    else:
                        if sat_spec_humidity < water_plus:
                            cloud_fraction = (water_plus - sat_spec_humidity) / (dq * F_DQ_P)
                        else:
                            cloud_fraction = 0.0

                        if total_condensate > QCMIN:
                            cloud_fraction = max(CLD_MIN, cloud_fraction)

                        cloud_fraction = min(1.0, cloud_fraction)
                else:
                    if sat_spec_humidity < water_minus:
                        cloud_fraction = 1.0
                    else:
                        if sat_spec_humidity < water_plus:
                            if ICLOUD_F == 0:
                                cloud_fraction = (water_plus - sat_spec_humidity) / (dq * F_DQ_P + dq * F_DQ_M)
                            else:
                                cloud_fraction = (water_plus - sat_spec_humidity) / ((dq * F_DQ_P + dq * F_DQ_M) * (1.0 - total_condensate))
                        else:
                            cloud_fraction = 0.0

                        if total_condensate > QCMIN:
                            cloud_fraction = max(CLD_MIN, cloud_fraction)

                        cloud_fraction = min(1.0, cloud_fraction)
            else:
                cloud_fraction = 0.0

        if CFFLAG == 2:
            if rh >= 1.0:
                cloud_fraction = 1.0
            elif rh > RH_THRES and total_condensate > QCMIN:
                cloud_fraction = exp(XR_A * log(rh)) * (
                    1.0 - exp(-XR_B * max(0.0, total_condensate) / max(1.0e-5, exp(XR_C * log(max(1.0e-10, 1.0 - rh) * sat_spec_humidity))))
                )
                cloud_fraction = max(0.0, min(1.0, cloud_fraction))
            else:
                cloud_fraction = 0.0

        if CFFLAG == 3:
            if total_condensate > QCMIN:
                cloud_fraction = (
                    1.0
                    / 50.0
                    * (
                        5.77 * (100.0 - grid_size / 1000.0) * exp(1.07 * log(max(QCMIN * 1000.0, total_condensate * 1000.0)))
                        + 4.82 * (grid_size / 1000.0 - 50.0) * exp(0.94 * log(max(QCMIN * 1000.0, total_condensate * 1000.0)))
                    )
                )
                cloud_fraction = cloud_fraction * (0.92 / 0.96 * total_liquid / total_condensate + 1.0 / 0.96 * total_solid / total_condensate)
                cloud_fraction = max(0.0, min(1.0, cloud_fraction))
            else:
                cloud_fraction = 0.0

        if CFFLAG == 4:
            sigma = 0.28 + exp(0.49 * log(max(QCMIN * 1000.0, total_condensate * 1000.0)))
            gam = max(0.0, total_condensate * 1000.0) / sigma
            if gam < 0.18:
                qa10 = 0.0
            elif gam > 2.0:
                qa10 = 1.0
            else:
                qa10 = -0.1754 + 0.9811 * gam - 0.2223 * gam**2 + 0.0104 * gam**3
                qa10 = max(0.0, min(1.0, qa10))

            if gam < 0.12:
                qa100 = 0.0
            elif gam > 1.85:
                qa100 = 1.0
            else:
                qa100 = -0.0913 + 0.7213 * gam + 0.1060 * gam**2 - 0.0946 * gam**3
                qa100 = max(0.0, min(1.0, qa100))

            cloud_fraction = qa10 + (log10(grid_size / 1000.0) - 1) * (qa100 - qa10)
            cloud_fraction = max(0.0, min(1.0, cloud_fraction))


def compute_one_minus_sigma(one_minus_sigma: FloatFieldIJ, area: FloatFieldIJ):
    from __externals__ import DO_SCALE_DEP

    with computation(FORWARD), interval(0, 1):
        if DO_SCALE_DEP:
            one_minus_sigma = sigma(sqrt(area))
        else:
            one_minus_sigma = 1.0


def convert_temperature_start(
    t_state: FloatField,
    t_local: FloatField64,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
):
    from __externals__ import DO_INLINE_MP

    with computation(PARALLEL), interval(...):
        if DO_INLINE_MP:
            total_condensate = liquid + rain + ice + snow + graupel
            t_local = t_state / ((1.0 + ZVIR * vapor) * (1.0 - total_condensate))
        else:
            t_local = t_state


def convert_temperature_end(
    t_state: FloatField,
    t_local: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    total_liquid: FloatField,
    total_solid: FloatField,
):
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR, CP_HEATING, DO_INLINE_MP

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        c8: FloatField64 = 0.0
        cp8: FloatField64 = 0.0
        con_r8: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        if DO_INLINE_MP:
            total_condensate = liquid + rain + ice + snow + graupel
            if CP_HEATING:
                con_r8 = ONE_R8 - (vapor + total_condensate)
                c8 = moist_heat_capacity_4(con_r8, vapor, total_liquid, total_solid) * C_AIR
                cp8 = con_r8 * CP_AIR + vapor * CP_VAP + total_liquid * C_LIQ + total_solid * C_ICE
                dz = dz / t_state
                t_state = t_state + (t_local * ((1.0 + ZVIR * vapor) * (1.0 - total_condensate)) - t_state) * c8 / cp8
                dz = dz * t_state
            else:
                t_state = t_local * ((1.0 + ZVIR * vapor) * (1.0 - total_condensate))
        else:
            total_liquid = liquid + rain
            total_solid = ice + snow + graupel
            total_condensate = total_liquid + total_solid
            con_r8 = ONE_R8 - (vapor + total_condensate)
            c8 = moist_heat_capacity_4(con_r8, vapor, total_liquid, total_solid, C1_VAP, C1_LIQ, C1_ICE) * C_AIR
            t_state = t_state + (t_local - t_state) * c8 / CP_AIR


def compute_total_energy_change(
    total_energy: FloatField,
    t_local: FloatField,
    dp: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    snow: FloatField,
    rain: FloatField,
):

    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR, CONSV_TE, HYDROSTATIC

    with computation(PARALLEL), interval(...):
        if CONSV_TE:
            if HYDROSTATIC:
                total_energy = -C_AIR * t_local * dp
            else:
                total_energy = -moist_total_energy(t_local, vapor, liquid, rain, ice, snow, graupel, dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, True) * GRAV


def eis_factor_and_rates(
    estimated_inversion_strength: FloatFieldIJ,
    convection_fraction: FloatFieldIJ,
    factor_eis: FloatFieldIJ,
    factor_rc: FloatFieldIJ,
    cpaut: FloatFieldIJ,
) -> FloatField:
    from __externals__ import CPAUT0, RTHRESHS, RTHRESHU

    with computation(FORWARD), interval(0, 1):
        # Use estimated inversion strength to determine stable vs unstable areas
        factor_eis = compute_estimated_inversion_strength_factor(estimated_inversion_strength)

        # Adjust autoconversion rates and thresholds using decoupled regimes
        # 1. Rate scaling based on Boundary Layer Stability (EIS)
        # High inversion (fac_eis=1.0) -> reduced to 0.5 * cpaut0
        # Low inversion (fac_eis=0.0)  -> stays at 1.0 * cpaut0
        cpaut = CPAUT0 * (0.5 * factor_eis + 1.0 * (1.0 - factor_eis))
        # 2. Threshold scaling based on Deep Instability (CAPE / cnv_fraction)
        # convective (cnv_fraction=1) -> RTHRESHU
        # stratiform (cnv_fraction=0) -> RTHRESHS
        # NOTE: Consider raising RTHRESHU from 7.0e-6 to 8.0e-6 or 8.5e-6 to help suppress ITCZ over-precipitation
        factor_rc = RC * (RTHRESHU * convection_fraction + RTHRESHS * (1.0 - convection_fraction)) ** 3


def fix_negative_water_species(
    t: FloatField64,
    dry_dp: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    mppcw: FloatFieldIJ,
    mppfr: FloatFieldIJ,
):
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, CONV_FACTOR, D1_ICE, D1_VAP, DO_QA, LI00, LI20, LV00, T_WFR

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        cvm: FloatField64 = 0.0
        total_energy: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        # calculate moist heat capacity and latent heat coefficients
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

    with computation(PARALLEL), interval(...):  # dqv, dql, dqr, dqi, dqs, dqg
        ## fix negative solid-phase hydrometeors
        # if cloud ice < 0, borrow from snow
        if ice < 0.0:
            sink = min(-ice, max(0.0, snow))
            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=sink,
                dliquid=0.0,
                dgraupel=0.0,
                drain=0.0,
                dsnow=-sink,
                DO_QA=DO_QA,
            )

        # if snow < 0, borrow from graupel
        if snow < 0.0:
            sink = min(-snow, max(0.0, graupel))
            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=0.0,
                dliquid=0.0,
                dgraupel=-sink,
                drain=0.0,
                dsnow=sink,
                DO_QA=DO_QA,
            )

        # if grapuel < 0, borrow from rain
        if graupel < 0.0:
            sink = min(-graupel, max(0.0, rain))
            mppfr = mppfr + sink * dry_dp * CONV_FACTOR
            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=0.0,
                dliquid=0.0,
                dgraupel=sink,
                drain=-sink,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )

            ## fix negative liquid-phase hydrometeors
            # if rain < 0, borrow from cloud water
            if rain < 0:
                sink = min(-rain, max(0.0, liquid))
                vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                    cloud_fraction=cloud_fraction,
                    vapor=vapor,
                    ice=ice,
                    liquid=liquid,
                    graupel=graupel,
                    rain=rain,
                    snow=snow,
                    dvapor=0.0,
                    dice=0.0,
                    dliquid=-sink,
                    dgraupel=0.0,
                    drain=sink,
                    dsnow=0.0,
                    DO_QA=DO_QA,
                )

            # if cloud water < 0, borrow from water vapor
            if liquid < 0:
                sink = min(-liquid, max(0.0, vapor))
                mppcw = mppcw + sink * dry_dp * CONV_FACTOR
                t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                    cloud_fraction=cloud_fraction,
                    vapor=vapor,
                    ice=ice,
                    liquid=liquid,
                    graupel=graupel,
                    rain=rain,
                    snow=snow,
                    dvapor=-sink,
                    dice=0.0,
                    dliquid=sink,
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

    ## fix negative water vapor
    with computation(FORWARD), interval(0, -1):
        # if water vapor < 0, borrow water vapor from "below" (index 0 is TOA)
        if vapor < 0:
            vapor[0, 0, 1] = vapor[0, 0, 1] + vapor * dry_dp / dry_dp[0, 0, 1]
            vapor = 0.0

    with computation(FORWARD), interval(-1, None):
        # if water vapor < 0, borrow water vapor from "above" (index 0 is TOA)
        if vapor < 0.0 and vapor[0, 0, -1] > 0.0:
            dq = min(-vapor * dry_dp, vapor[0, 0, -1] * dry_dp[0, 0, -1])
            vapor[0, 0, -1] = vapor[0, 0, -1] - dq / dry_dp[0, 0, -1]
            vapor = vapor + dq / dry_dp


def generate_particle_nuclei(
    ccn: FloatField,
    cin: FloatField,
    concentration_liquid: FloatField,
    concentration_ice: FloatField,
    density: FloatField,
    surface_geopotential_height: FloatFieldIJ,
):
    from __externals__ import CCN_L, CCN_O, PROG_CCN, PROG_CIN

    with computation(FORWARD), interval(0, 1):
        ccn0: FloatFieldIJ = (
            CCN_L * min(1.0, abs(surface_geopotential_height) / (10.0 * GRAV)) + CCN_O * (1.0 - min(1.0, abs(surface_geopotential_height) / (10.0 * GRAV)))
        ) * 1.0e6
        cin0: FloatFieldIJ = 0.0

    with computation(PARALLEL), interval(...):
        if PROG_CCN:
            ccn = concentration_liquid / density
        else:
            ccn = ccn0 / density

        if PROG_CIN:
            cin = concentration_ice / density
        else:
            cin = cin0 / density


def horizontal_subgrid_variation(h_var: FloatField, critical_relative_humidity_for_pdf: FloatField):
    with computation(PARALLEL), interval(...):
        h_var = min(0.30, 1.0 - critical_relative_humidity_for_pdf)


def pressure_derived_fields_mixing_ratio_conversion_copy_state(
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    local_vapor: FloatField,
    local_ice: FloatField,
    local_liquid: FloatField,
    local_graupel: FloatField,
    local_rain: FloatField,
    local_snow: FloatField,
    local_cloud_fraction: FloatField,
    local_t: FloatField64,
    dp: FloatField,
    local_moist_dp_original: FloatField,
    local_dry_dp: FloatField,
    dz: FloatField,
    local_dz: FloatField,
    local_density: FloatField,
    local_density_factor: FloatField,
    local_p: FloatField,
    u: FloatField,
    local_u: FloatField,
    v: FloatField,
    local_v: FloatField,
    w: FloatField,
    local_w: FloatField,
):
    from __externals__ import DO_INLINE_MP, HYDROSTATIC

    # initialize 64 bit internal field
    with computation(PARALLEL), interval(...):
        con_r8: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        local_vapor = vapor
        local_ice = ice
        local_liquid = liquid
        local_graupel = graupel
        local_rain = rain
        local_snow = snow
        local_cloud_fraction = cloud_fraction

        # determine the dry air fraction based on DO_INLINE_MP setting
        if DO_INLINE_MP:
            total_condensate = local_liquid + local_rain + local_ice + local_snow + local_graupel
            con_r8 = ONE_R8 - (local_vapor + total_condensate)
        else:
            con_r8 = ONE_R8 - local_vapor

        # store original moist pressure thickness
        local_moist_dp_original = dp

        # convert total pressure thickness (dp) to dry air pressure thickness (dry_dp)
        local_dry_dp = local_moist_dp_original * con_r8

        # calculate factor to go from specific humidity to dry mixing ratio
        con_r8 = ONE_R8 / con_r8

        # convert all species to dry mixing ratios
        local_vapor = local_vapor * con_r8
        local_ice = local_ice * con_r8
        local_liquid = local_liquid * con_r8
        local_graupel = local_graupel * con_r8
        local_rain = local_rain * con_r8
        local_snow = local_snow * con_r8

        # dry air density and layer-mean pressure thickness
        local_dz = dz
        local_density = -local_moist_dp_original / (GRAV * local_dz)
        local_p = local_density * RDGAS * local_t

        # for sedi_momentum transport

        local_u = u
        local_v = v
        if not HYDROSTATIC:
            local_w = w


def radar_reflectivity(
    t: FloatField64,
    density: FloatField,
    density_factor: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    dbz: FloatField,
):
    """compute radar reflectivity

    Args:
        t (FloatField64)
        density (FloatField)
        density_factor (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        dbz (FloatField)
    """
    from __externals__ import (
        BLINH,
        BLING,
        BLINR,
        BLINS,
        CONST_VG,
        CONST_VR,
        CONST_VS,
        DO_HAIL,
        MUH,
        MUG,
        MUR,
        MUS,
        RADG_FLAG,
        RADR_FLAG,
        RADS_FLAG,
        RRAH,
        RRAG,
        RRAR,
        RRAS,
        RRBH,
        RRBG,
        RRBR,
        RRBS,
        TVAH,
        TVAG,
        TVAR,
        TVAS,
        TVBH,
        TVBG,
        TVBR,
        TVBS,
        VH_FAC,
        VH_MAX,
        VH_MIN,
        VG_FAC,
        VG_MAX,
        VG_MIN,
        VR_FAC,
        VR_MAX,
        VR_MIN,
        VS_FAC,
        VS_MAX,
        VS_MIN,
    )

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        z_e: FloatField64 = 0.0

    with computation(FORWARD), interval(0, 1):
        # initialize internal constants
        alpha: FloatFieldIJ = 0.176 / 0.930
        mp_const: FloatFieldIJ = 200 * exp(1.6 * log(3.6e6))

    with computation(PARALLEL), interval(...):
        # init output
        dbz = -20.0

    with computation(PARALLEL), interval(...):
        # fall speed
        if RADR_FLAG == 3:
            terminal_velocity_rain = terminal_velocity_graupel_rain_snow(
                rain,
                density,
                density_factor,
                TVAR,
                TVBR,
                BLINR,
                MUR,
                VR_MIN,
                VR_MAX,
                VR_FAC,
                CONST_VR,
            )
            terminal_velocity_rain = terminal_velocity_rain / RHOR

        if RADS_FLAG == 3:
            terminal_velocity_snow = terminal_velocity_graupel_rain_snow(
                snow,
                density,
                density_factor,
                TVAS,
                TVBS,
                BLINS,
                MUS,
                VS_MIN,
                VS_MAX,
                VS_FAC,
                CONST_VS,
            )
            terminal_velocity_snow = terminal_velocity_snow / RHOS

        if RADG_FLAG == 3:
            if DO_HAIL:
                terminal_velocity_graupel = terminal_velocity_graupel_rain_snow(
                    graupel,
                    density,
                    density_factor,
                    TVAH,
                    TVBH,
                    BLINH,
                    MUH,
                    VH_MIN,
                    VH_MAX,
                    VH_FAC,
                    CONST_VG,
                )
                terminal_velocity_graupel = terminal_velocity_graupel / RHOH
            else:
                terminal_velocity_graupel = terminal_velocity_graupel_rain_snow(
                    graupel,
                    density,
                    density_factor,
                    TVAG,
                    TVBG,
                    BLING,
                    MUG,
                    VG_MIN,
                    VG_MAX,
                    VG_FAC,
                    CONST_VG,
                )
                terminal_velocity_graupel = terminal_velocity_graupel / RHOG

    with computation(PARALLEL), interval(...):
        # radar reflectivity
        z_e = 0.0

        qden = density * rain
        if rain > QPMIN:
            fac_r = calc_reflectivity_factor(rain, density, MUR, RRAR, RRBR)
        else:
            fac_r = 0.0

        if RADR_FLAG == 1 or RADR_FLAG == 2:
            z_e = z_e + fac_r * 1.0e18

        if RADR_FLAG == 3:
            z_e = z_e + mp_const * exp(1.6 * log(qden * terminal_velocity_rain))

        qden = density * snow
        if snow > QPMIN:
            fac_s = calc_reflectivity_factor(snow, density, MUS, RRAS, RRBS)
        else:
            fac_s = 0.0

        if RADS_FLAG == 1:
            if t < TICE:
                z_e = z_e + fac_s * 1.0e18 * alpha * (RHOS / RHOI) ** 2
            else:
                z_e = z_e + fac_s * 1.0e18 * alpha * (RHOS / RHOI) ** 2 / alpha

        if RADS_FLAG == 2:
            if t < TICE:
                z_e = z_e + fac_s * 1.0e18 * alpha * (RHOS / RHOI) ** 2
            else:
                z_e = z_e + fac_s * 1.0e18

        if RADS_FLAG == 3:
            z_e = z_e + mp_const * exp(1.6 * log(qden * terminal_velocity_snow))

        qden = density * graupel
        if DO_HAIL:
            if graupel > QPMIN:
                fac_g = calc_reflectivity_factor(graupel, density, MUH, RRAH, RRBH)
            else:
                fac_g = 0.0

            if RADG_FLAG == 1:
                if t < TICE:
                    z_e = z_e + fac_g * 1.0e18 * alpha * (RHOH / RHOI) ** 2
                else:
                    z_e = z_e + fac_g * 1.0e18 * alpha * (RHOH / RHOI) ** 2 / alpha

            if RADG_FLAG == 2:
                if t < TICE:
                    z_e = z_e + fac_g * 1.0e18 * alpha * (RHOH / RHOI) ** 2
                else:
                    z_e = z_e + (fac_g * 1.0e18) ** 0.95
        else:
            if graupel > QPMIN:
                fac_g = calc_reflectivity_factor(graupel, density, MUG, RRAG, RRBG)
            else:
                fac_g = 0.0

            if RADG_FLAG == 1:
                if t < TICE:
                    z_e = z_e + fac_g * 1.0e18 * alpha * (RHOG / RHOI) ** 2
                else:
                    z_e = z_e + fac_g * 1.0e18 * alpha * (RHOG / RHOI) ** 2 / alpha

            if RADG_FLAG == 2:
                if t < TICE:
                    z_e = z_e + fac_g * 1.0e18 * alpha * (RHOG / RHOI) ** 2
                else:
                    z_e = z_e + (fac_g * 1.0e18) ** 0.95

        if RADG_FLAG == 3:
            z_e = z_e + mp_const * exp(1.6 * log(qden * terminal_velocity_graupel))

        dbz = 10.0 * log10(max(0.01, z_e))


def restore_and_update_humidities(
    t: FloatField64,
    dry_dp: FloatField,
    moist_dp_end: FloatField,
    moist_dp_original: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    cloud_fraction: FloatField,
    local_vapor: FloatField,
    local_ice: FloatField,
    local_liquid: FloatField,
    local_graupel: FloatField,
    local_rain: FloatField,
    local_snow: FloatField,
    local_cloud_fraction: FloatField,
    output_reflectivity: FloatField,
    local_reflectivity: FloatField,
    dcloud_fraction_dt: FloatField,
    local_condensate: FloatField,
    local_kappa: FloatField,
    local_total_liquid: FloatField,
    local_total_solid: FloatField,
):
    """Use dry_dp (dry mass) and moist_dp_original (initial total mass) to return safely to specific humidities consistent with the host model's current timestep.

    Args:
        t (FloatField64)
        dry_dp (FloatField)
        moist_dp_end (FloatField)
        moist_dp_original (FloatField)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        cloud_fraction (FloatField)
        local_vapor (FloatField)
        local_ice (FloatField)
        local_liquid (FloatField)
        local_graupel (FloatField)
        local_rain (FloatField)
        local_snow (FloatField)
        local_cloud_fraction (FloatField)
        output_reflectivity (FloatField)
        local_reflectivity (FloatField)
        dcloud_fraction_dt (FloatField)
        local_condensate (FloatField)
        local_kappa (FloatField)
        local_total_liquid (FloatField)
        local_total_solid (FloatField)
    """
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR, DO_INLINE_MP, DO_QA, DT_INVERSE, MOIST_KAPPA, USE_COND

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        con_r8: FloatField64 = 0.0
        c8: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        # 1. Calculate the NEW total-to-dry mass ratio
        if DO_INLINE_MP:
            # If inline MP is used, total mass includes vapor and all condensates.
            # If not, total mass is just dry air + vapor (condensates are "massless").
            internal_condensate = local_liquid + local_rain + local_ice + local_snow + local_graupel
            con_r8 = ONE_R8 + local_vapor + internal_condensate
        else:
            con_r8 = ONE_R8 + local_vapor

        # 2. Reconstruct the new moist pressure thickness
        # dry_dp currently holds the DRY pressure thickness.
        # set moist_dp_end represent the NEW total moist pressure thickness.
        moist_dp_end = dry_dp * con_r8

        # 3. Convert water species back to specific humidities
        # Calculate the reciprocal: (Dry Mass / New Total Mass)
        con_r8 = ONE_R8 / con_r8

        local_vapor = local_vapor * con_r8
        local_ice = local_ice * con_r8
        local_liquid = local_liquid * con_r8
        local_graupel = local_graupel * con_r8
        local_rain = local_rain * con_r8
        local_snow = local_snow * con_r8

        # 4. Calculate the Tracer Dilution Adjustment
        # Instead of the complex q1/q2 algebraic formula, we use the exact
        # physical definition of tracer dilution: Old Mass / New Mass.
        # moist_dp_original is the old total mass, and moist_dp_end is the new total mass.
        tracer_dilution_adjustment = moist_dp_original / moist_dp_end

        # update the relfectivity field which passed in from the overarching model
        output_reflectivity = local_reflectivity

        # return cloud fraction tendencies for GEOS
        if not DO_QA:
            dcloud_fraction_dt = DT_INVERSE * (cloud_fraction * sqrt(max(local_ice + local_liquid, QCMIN) / max(ice + liquid, QCMIN)) - cloud_fraction)
        else:
            cloud_fraction = local_cloud_fraction
            dcloud_fraction_dt = 0.0

        # update state mixing ratios
        vapor = local_vapor
        ice = local_ice
        liquid = local_liquid
        graupel = local_graupel
        rain = local_rain
        snow = local_snow

        # calculate some more variables needed outside
        local_total_liquid = local_liquid + local_rain
        local_total_solid = local_ice + local_snow + local_graupel
        total_condensate = local_total_liquid + local_total_solid
        con_r8 = ONE_R8 - (local_vapor + total_condensate)
        c8 = moist_heat_capacity_4(con_r8, local_vapor, local_total_liquid, local_total_solid, C1_VAP, C1_LIQ, C1_ICE) * C_AIR

        if USE_COND:
            local_condensate = total_condensate
        if MOIST_KAPPA:
            tmp = RDGAS * (1.0 + ZVIR * local_vapor)
            local_kappa = tmp / (tmp + c8)


def sedi_momentum_part_1(
    t: FloatField64,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    u: FloatField,
    local_u: FloatField,
    v: FloatField,
    local_v: FloatField,
):
    """update t based on u and v from sedimentation

    Args:
        t (FloatField64)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        u (FloatField)
        local_u (FloatField)
        v (FloatField)
        local_v (FloatField)
    """
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        c: FloatField64 = 0.0
        t_uv: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        c = moist_heat_capacity_6(vapor, ice, liquid, graupel, rain, snow, C1_VAP, C1_LIQ, C1_ICE) * C_AIR
        t_uv = 0.5 * (u**2 + v**2 - (local_u**2 + local_v**2)) / c
        t = t + t_uv


def sedi_momentum_part_2(
    t: FloatField64,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    w: FloatField,
    local_w: FloatField,
):
    """update t based on w from sedimentation

    Args:
        t (FloatField64)
        vapor (FloatField)
        ice (FloatField)
        liquid (FloatField)
        graupel (FloatField)
        rain (FloatField)
        snow (FloatField)
        w (FloatField)
        local_w (FloatField)
    """
    from __externals__ import C_AIR, C1_ICE, C1_LIQ, C1_VAP

    with computation(PARALLEL), interval(...):
        # initialize 64 bit internals
        c: FloatField64 = 0.0
        t_w: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        c = moist_heat_capacity_6(vapor, ice, liquid, graupel, rain, snow, C1_VAP, C1_LIQ, C1_ICE) * C_AIR
        t_w = 0.5 * (w**2 - local_w**2) / c
        t = t + t_w


def set_value_64_bit(field: FloatField64, value: Float64) -> None:
    """
    Sets every element of a field to a single value.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(PARALLEL), interval(...):
        field = value


def total_energy_and_water(
    t_local: FloatField64,
    total_energy: FloatField64,
    dtotal_energy: FloatFieldIJ64,
    total_water: FloatField64,
    total_energy_b: FloatFieldIJ64,
    total_water_b: FloatFieldIJ64,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    dp: FloatField,
    cloud_vapor: FloatField,
    cloud_ice: FloatField,
    cloud_liquid: FloatField,
    cloud_rain: FloatField,
    cloud_snow: FloatField,
    cloud_graupel: FloatField,
    precip_vapor: FloatFieldIJ,
    precip_ice: FloatFieldIJ,
    precip_liquid: FloatFieldIJ,
    precip_rain: FloatFieldIJ,
    precip_snow: FloatFieldIJ,
    precip_graupel: FloatFieldIJ,
    sen: FloatFieldIJ,
    stress: FloatFieldIJ,
    moist_q: Bool,
    save_te_loss: Bool,
    total_energy_loss: FloatFieldIJ64,
):
    from __externals__ import C_AIR, DT, HYDROSTATIC, LI00, LV00

    # initialize 64 bit internal fields
    with computation(PARALLEL), interval(...):
        cvm: FloatField64 = 0.0
        con_r8: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        total_liquid = cloud_liquid + precip_rain
        total_solid = cloud_ice + cloud_snow + cloud_graupel
        total_condensate = total_liquid + total_solid
        con_r8 = ONE_R8 - (cloud_vapor + total_condensate)
        if moist_q:
            cvm = moist_heat_capacity_4(con_r8, cloud_vapor, total_liquid, total_solid)
        else:
            cvm = moist_heat_capacity_3(cloud_vapor, total_liquid, total_solid)

        total_energy = (cvm * t_local + LV00 * cloud_vapor - LI00 * total_solid) * C_AIR
        if HYDROSTATIC:
            total_energy = total_energy + 0.5 * (u**2 + v**2)
        else:
            total_energy = total_energy + 0.5 * (u**2 + v**2 + w**2)
        total_energy = RGRAV * total_energy * dp
        total_water = RGRAV * (cloud_vapor + total_condensate) * dp

    with computation(FORWARD), interval(...):
        total_energy_b = dtotal_energy + (LV00 * C_AIR * precip_vapor - LI00 * C_AIR * (precip_ice + precip_snow + precip_graupel)) * DT / 86400 + sen * DT + stress * DT
        total_water_b = (precip_vapor + precip_liquid + precip_rain + precip_ice + precip_snow + precip_graupel) * DT / 86400

    if save_te_loss:
        # total energy change due to sedimentation and its heating
        total_energy_loss = dtotal_energy


class GFDLMPV3Driver(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        saturation_tables: GFDLMPV3Tables,
        gfdl_1m_config: GFDL1MConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        mp_config: GFDLMPV3CloudMPConfig,
    ):
        # initialize NDSLRuntime parent class
        super.__init__(stencil_factory)

        # compute driver specific constants
        # timesteps
        DRIVER_DT = gfdl_1m_config.DT_MOIST / mp_namelist.NTIMES
        DT_INVERSE = 1 / gfdl_1m_config.DT_MOIST
        # conversion factor to mm/day
        CONV_FACTOR = 86400.0 * RGRAV / gfdl_1m_config.DT_MOIST

        # initialize class specific locals
        self._gfdl_mp_v3_locals = GFDLMPV3Locals.make_locals(quantity_factory)

        # make config visible at runtime
        self._mp_config = mp_config
        self._mp_namelist = mp_namelist
        self._saturation_tables = saturation_tables

        # initialize subcomponents
        self._mp_full = MPFull(stencil_factory, quantity_factory, saturation_tables, mp_config, mp_namelist, CONV_FACTOR, DRIVER_DT)

        # construct stencils
        self._cloud_fraction = stencil_factory.from_dims_halo(
            func=cloud_fraction,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "CFFLAG": mp_namelist.CFFLAG,
                "CLD_MIN": mp_namelist.CLD_MIN,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_CLD_ADJ": mp_namelist.DO_CLD_ADJ,
                "F_DQ_M": mp_namelist.F_DQ_M,
                "F_DQ_P": mp_namelist.F_DQ_P,
                "ICLOUD_F": mp_namelist.ICLOUD_F,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "RAD_GRAUPEL": mp_namelist.RAD_GRAUPEL,
                "RAD_RAIN": mp_namelist.RAD_RAIN,
                "RAD_SNOW": mp_namelist.RAD_SNOW,
                "RH_THRES": mp_namelist.RH_THRES,
                "T_WFR": mp_config.T_WFR,
                "XR_A": mp_namelist.XR_A,
                "XR_B": mp_namelist.XR_B,
                "XR_C": mp_namelist.XR_C,
            },
        )
        self._compute_one_minus_sigma = stencil_factory.from_dims_halo(
            func=compute_one_minus_sigma,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DO_SCALE_DEP": mp_namelist.DO_SCALE_DEP},
        )
        self._calculate_base_total_energy = stencil_factory.from_dims_halo(
            func=calculate_base_total_energy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONSV_TE": mp_config.CONSV_TE,
                "HYDROSTATIC": gfdl_1m_config.LHYDROSTATIC,
                "C_AIR": mp_config.C_AIR,
                "C1_VAP": mp_config.C1_VAP,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_ICE": mp_config.C1_ICE,
            },
        )
        self._calculate_total_energy_change = stencil_factory.from_dims_halo(
            func=calculate_total_energy_change,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONSV_TE": mp_config.CONSV_TE,
                "HYDROSTATIC": gfdl_1m_config.LHYDROSTATIC,
                "C_AIR": mp_config.C_AIR,
                "C1_VAP": mp_config.C1_VAP,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_ICE": mp_config.C1_ICE,
            },
        )
        self._convert_temperature_start = stencil_factory.from_dims_halo(
            func=convert_temperature_start,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DO_INLINE_MP": mp_config.DO_INLINE_MP},
        )
        self._convert_temperature_end = stencil_factory.from_dims_halo(
            func=convert_temperature_end,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "C_AIR": mp_config.C_AIR,
                "CP_HEATING": mp_namelist.CP_HEATING,
                "DO_INLINE_MP": mp_config.DO_INLINE_MP,
            },
        )
        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._copy_2d = stencil_factory.from_dims_halo(
            func=copy_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._eis_factor_and_rates = stencil_factory.from_dims_halo(
            func=eis_factor_and_rates,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CPAUT0": mp_config.CPAUT0},
        )
        self._fix_negative_water_species = stencil_factory.from_dims_halo(
            func=fix_negative_water_species,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "DO_QA": mp_namelist.DO_QA,
                "C1_VAP": mp_config.C1_VAP,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_ICE": mp_config.C1_ICE,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._generate_particle_nuclei = stencil_factory.from_dims_halo(
            func=generate_particle_nuclei,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"PROG_CCN": mp_namelist.PROG_CCN, "PROG_CIN": mp_namelist.PROG_CIN, "CCN_L": mp_namelist.CCN_L, "CCN_O": mp_namelist.CCN_O},
        )
        self._horizontal_subgrid_variation = stencil_factory.from_dims_halo(
            func=horizontal_subgrid_variation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._pressure_derived_fields_mixing_ratio_conversion_copy_state = stencil_factory.from_dims_halo(
            func=pressure_derived_fields_mixing_ratio_conversion_copy_state,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DO_INLINE_MP": mp_config.DO_INLINE_MP,
                "HYDROSTATIC": gfdl_1m_config.LHYDROSTATIC,
            },
        )
        self._radar_reflectivity = stencil_factory.from_dims_halo(
            func=radar_reflectivity,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINH": mp_namelist.BLINH,
                "BLING": mp_namelist.BLING,
                "BLINR": mp_namelist.BLINR,
                "BLINS": mp_namelist.BLINS,
                "CONST_VG": mp_namelist.CONST_VG,
                "CONST_VR": mp_namelist.CONST_VR,
                "CONST_VS": mp_namelist.CONST_VS,
                "DO_HAIL": mp_namelist.DO_HAIL,
                "MUH": mp_namelist.MUH,
                "MUG": mp_namelist.MUG,
                "MUR": mp_namelist.MUR,
                "MUS": mp_namelist.MUS,
                "RADG_FLAG": mp_namelist.RADG_FLAG,
                "RADR_FLAG": mp_namelist.RADR_FLAG,
                "RADS_FLAG": mp_namelist.RADS_FLAG,
                "RRAH": mp_config.RRAH,
                "RRAG": mp_config.RRAG,
                "RRAR": mp_config.RRAR,
                "RRAS": mp_config.RRAS,
                "RRBH": mp_config.RRBH,
                "RRBG": mp_config.RRBG,
                "RRBR": mp_config.RRBR,
                "RRBS": mp_config.RRBS,
                "TVAH": mp_config.TVAH,
                "TVAG": mp_config.TVAG,
                "TVAR": mp_config.TVAR,
                "TVAS": mp_config.TVAS,
                "TVBH": mp_config.TVBH,
                "TVBG": mp_config.TVBG,
                "TVBR": mp_config.TVBR,
                "TVBS": mp_config.TVBS,
                "VH_FAC": mp_namelist.VH_FAC,
                "VH_MAX": mp_namelist.VH_MAX,
                "VH_MIN": mp_namelist.VH_MIN,
                "VG_FAC": mp_namelist.VG_FAC,
                "VG_MAX": mp_namelist.VG_MAX,
                "VG_MIN": mp_namelist.VG_MIN,
                "VR_FAC": mp_namelist.VR_FAC,
                "VR_MAX": mp_namelist.VR_MAX,
                "VR_MIN": mp_namelist.VR_MIN,
                "VS_FAC": mp_namelist.VS_FAC,
                "VS_MAX": mp_namelist.VS_MAX,
                "VS_MIN": mp_namelist.VS_MIN,
            },
        )
        self._restore_and_update_humidities = stencil_factory.from_dims_halo(
            func=restore_and_update_humidities,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "C_AIR": mp_config.C_AIR,
                "DO_INLINE_MP": mp_config.DO_INLINE_MP,
                "DO_QA": mp_namelist.DO_QA,
                "DT_INVERSE": DT_INVERSE,
                "MOIST_KAPPA": mp_config.MOIST_KAPPA,
                "USE_COND": mp_config.USE_COND,
            },
        )
        self._sedi_momentum_part_1 = stencil_factory.from_dims_halo(
            func=sedi_momentum_part_1,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "C_AIR": mp_config.C_AIR,
            },
        )
        self._sedi_momentum_part_2 = stencil_factory.from_dims_halo(
            func=sedi_momentum_part_2,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "C_AIR": mp_config.C_AIR,
            },
        )
        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._set_value_64_bit = stencil_factory.from_dims_halo(
            func=set_value_64_bit,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._total_energy_and_water = stencil_factory.from_dims_halo(
            func=total_energy_and_water,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DT": gfdl_1m_config.DT_MOIST,
                "HYDROSTATIC": gfdl_1m_config.LHYDROSTATIC,
                "LI00": mp_config.LI00,
                "LV00": mp_config.LV00,
                "C_AIR": mp_config.C_AIR,
            },
        )

        # dummy fields that are used as placeholders for optional inputs to stencils that are not called
        # they exist only as a thing to pass to stencils, since all inputs must always be supplied,
        # regardless of whether the associated option is enabled, and Float(0.0) cannot be supplied as an input
        self._dummy_field_no_read_no_write_2d_64_bit = quantity_factory.zeros([I_DIM, J_DIM], units="NOREADWRITE", dtype=Float64)
        self._all_zeros_no_write_3d = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], units="NOWRITE", dtype=Float)

    def __call__(self, state: GFDL1MState, gfdl_1m_locals: GFDL1MLocals):
        # -----------------------------------------------------------------------
        # reset mp locals to zero
        # -----------------------------------------------------------------------
        self._set_value(field=self._gfdl_mp_v3_locals.mppcw, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppew, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppe1, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mpper, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppdi, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppd1, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppds, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppdg, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppsi, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mpps1, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppss, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppsg, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppfw, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppfr, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppar, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppas, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppag, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mpprs, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mpprg, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppxr, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppxs, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppxg, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppmi, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppms, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppmg, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppm1, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppm2, value=Float(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.mppm3, value=Float(0.0))

        # -----------------------------------------------------------------------
        # initialization of total energy difference
        # -----------------------------------------------------------------------
        self._set_value_64_bit(field=self._gfdl_mp_v3_locals.total_energy.delta, value=Float64(0.0))
        self._set_value(field=self._gfdl_mp_v3_locals.tracer_dilution_adjustment, value=Float(1.0))

        # -----------------------------------------------------------------------
        # copy convection fraction and surface type, work with copy instead of the original
        # -----------------------------------------------------------------------
        self._copy_2d(input=state.convection_fraction, output=self._gfdl_mp_v3_locals.convection_fraction)
        self._copy_2d(input=state.surface_type, output=self._gfdl_mp_v3_locals.surface_type)

        # -----------------------------------------------------------------------
        # one minus sigma used to control resoluton sensitive parameters
        # -----------------------------------------------------------------------
        self._compute_one_minus_sigma(one_minus_sigma=self._gfdl_mp_v3_locals.one_minus_sigma, area=state.area)

        # -----------------------------------------------------------------------
        # Use estimated inversion strength to determine stable vs unstable areas
        # -----------------------------------------------------------------------
        self._eis_factor_and_rates(
            estimated_inversion_strength=state.estimated_inversion_strength,
            convection_fraction=self._gfdl_mp_v3_locals.convection_fraction,
            factor_eis=self._gfdl_mp_v3_locals.factor_eis,
            factor_rc=self._gfdl_mp_v3_locals.factor_rc,
            cpaut=self._gfdl_mp_v3_locals.cpaut,
        )

        # -----------------------------------------------------------------------
        # conversion of temperature
        # -----------------------------------------------------------------------
        self._convert_temperature_start(
            t_state=state.t,
            t_local=self._gfdl_mp_v3_locals.t,
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
        )

        # -----------------------------------------------------------------------
        # calculate base total energy
        # -----------------------------------------------------------------------
        self._calculate_base_total_energy(
            t=self._gfdl_mp_v3_locals.t,
            dp=gfdl_1m_locals.dp,
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
            total_energy=self._gfdl_mp_v3_locals.total_energy.magnitude,
        )

        # -----------------------------------------------------------------------
        # total_energy_checker
        # -----------------------------------------------------------------------
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._gfdl_mp_v3_locals.t_local,
                total_energy=self._gfdl_mp_v3_locals.total_energy_beg_m,
                total_water=self._gfdl_mp_v3_locals.total_water_beg_m,
                total_energy_b=self._gfdl_mp_v3_locals.total_energy_b_beg_m,
                total_water_b=self._gfdl_mp_v3_locals.total_water_b_beg_m,
                u=state.u,
                v=state.v,
                w=state.vertical_motion.velocity,
                dp=gfdl_1m_locals.dp,
                cloud_vapor=state.radiation_field.vapor,
                cloud_ice=state.radiation_field.ice,
                cloud_liquid=state.radiation_field.liquid,
                cloud_rain=state.radiation_field.rain,
                cloud_snow=state.radiation_field.snow,
                cloud_graupel=state.radiation_field.graupel,
                vapor=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                precip_ice=state.precipitation_at_surface.ice,
                precip_liquid=state.precipitation_at_surface.water,
                precip_rain=state.precipitation_at_surface.rain,
                precip_snow=state.precipitation_at_surface.snow,
                precip_graupel=state.precipitation_at_surface.graupel,
                dtotal_energy=self._gfdl_mp_v3_locals.total_energy.delta,
                sen=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                stress=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                moist_q=True,
                save_te_loss=False,
                total_energy_loss=self._dummy_field_no_read_no_write_2d_64_bit,
            )

        # -----------------------------------------------------------------------
        # initialize radar reflectivity
        # -----------------------------------------------------------------------
        self._set_value(field=self._gfdl_mp_v3_locals.reflectivity, value=Float(-30.0))

        # -----------------------------------------------------------------------
        # setup the local state - to be used throughout the rest of microphyscis
        # -----------------------------------------------------------------------
        self._pressure_derived_fields_mixing_ratio_conversion_copy_state(
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
            cloud_fraction=state.radiation_field.cloud_fraction,
            local_vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
            local_ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
            local_liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
            local_graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
            local_rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
            local_snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
            local_cloud_fraction=self._gfdl_mp_v3_locals.cloud_fraction,
            local_t=self._gfdl_mp_v3_locals.t,
            dp=gfdl_1m_locals.dp,
            local_moist_dp_original=self._gfdl_mp_v3_locals.moist_dp_original,
            local_dry_dp=self._gfdl_mp_v3_locals.dry_dp,
            dz=gfdl_1m_locals.layer_thickness_negative,
            local_dz=self._gfdl_mp_v3_locals.dz,
            local_density=self._gfdl_mp_v3_locals.density,
            local_density_factor=self._gfdl_mp_v3_locals.density_factor,
            local_p=self._gfdl_mp_v3_locals.p,
            u=state.u,
            local_u=self._gfdl_mp_v3_locals.u,
            v=state.v,
            local_v=self._gfdl_mp_v3_locals.v,
            w=state.w,
            local_w=self._gfdl_mp_v3_locals.w,
        )

        # -----------------------------------------------------------------------
        # total_energy_checker
        # -----------------------------------------------------------------------
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._gfdl_mp_v3_locals.t,
                total_energy=self._gfdl_mp_v3_locals.total_energy.beg_d,
                total_water=self._gfdl_mp_v3_locals.total_water.beg_d,
                total_energy_b=self._gfdl_mp_v3_locals.total_energy.b_beg_d,
                total_water_b=self._gfdl_mp_v3_locals.total_water.b_beg_d,
                u=self._gfdl_mp_v3_locals.u,
                v=self._gfdl_mp_v3_locals.v,
                w=self._gfdl_mp_v3_locals.w,
                dp=self._gfdl_mp_v3_locals.dry_dp,
                cloud_vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                cloud_ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                cloud_liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                cloud_rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                cloud_snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                vapor=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                precip_ice=state.precipitation_at_surface.ice,
                precip_liquid=state.precipitation_at_surface.water,
                precip_rain=state.precipitation_at_surface.rain,
                precip_snow=state.precipitation_at_surface.snow,
                precip_graupel=state.precipitation_at_surface.graupel,
                dtotal_energy=self._gfdl_mp_v3_locals.total_energy.delta,
                sen=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                stress=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                moist_q=False,
                save_te_loss=False,
                total_energy_loss=self._dummy_field_no_read_no_write_2d_64_bit,
            )

        # -----------------------------------------------------------------------
        # generate cloud condensation nuclei (CCN), cloud ice nuclei (CIN)
        # -----------------------------------------------------------------------
        self._generate_particle_nuclei(
            ccn=self._gfdl_mp_v3_locals.ccn,
            cin=self._gfdl_mp_v3_locals.cin,
            concentration_liquid=state.concentration.liquid,
            concentration_ice=state.concentration.ice,
            density=self._gfdl_mp_v3_locals.density,
            surface_geopotential_height=state.surface_geopotential_height,
        )

        # -----------------------------------------------------------------------
        # import horizontal subgrid variability with pressure dependence
        # total water subgrid deviation in horizontal direction
        # default area dependent form: use dx ~ 100 km as the base
        # -----------------------------------------------------------------------
        self._horizontal_subgrid_variation(h_var=self._gfdl_mp_v3_locals.h_var, critical_relative_humidity_for_pdf=state.critical_relative_humidity_for_pdf)

        # -----------------------------------------------------------------------
        # fix negative water species
        # -----------------------------------------------------------------------
        if self._mp_namelist.FIX_NEGATIVE:
            self._fix_negative_water_species(
                t=self._gfdl_mp_v3_locals.t,
                dry_dp=self._gfdl_mp_v3_locals.dry_dp,
                vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=self._gfdl_mp_v3_locals.cloud_fraction,
                mppcw=self._gfdl_mp_v3_locals.mppcw,
                mppfr=self._gfdl_mp_v3_locals.mppfr,
            )

        # -----------------------------------------------------------------------
        # fast microphysics loop
        # -----------------------------------------------------------------------
        if self._mp_config.DO_MP_FAST:
            ndsl_log.error(
                "[GFDL1M Microphysics]: NDSL version of GFDLMPV3 fast microphysics with DO_MP_FAST = True has not been implemented, use DO_MP_FULL = True instead. "
                "This value should have been caught by the configuration checker, and this error should never be triggered. Other problems exist."
            )
            raise ValueError(
                "[GFDL1M Microphysics]: NDSL version of GFDLMPV3 fast microphysics with DO_MP_FAST = True has not been implemented, use DO_MP_FULL = True instead. "
                "This value should have been caught by the configuration checker, and this error should never be triggered. Other problems exist."
            )

        # -----------------------------------------------------------------------
        # full microphysics loop
        # -----------------------------------------------------------------------
        if self._mp_config.DO_MP_FULL:
            self._mp_full(state, self._gfdl_mp_v3_locals)

        # -----------------------------------------------------------------------
        # cloud fraction diagnostic
        # -----------------------------------------------------------------------
        if self._mp_namelist.DO_QA and self._mp_config.LAST_STEP:
            self._cloud_fraction(
                t=self._gfdl_mp_v3_locals.t,
                p=self._gfdl_mp_v3_locals.p,
                density=self._gfdl_mp_v3_locals.density,
                vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=self._gfdl_mp_v3_locals.cloud_fraction,
                area=self._gfdl_mp_v3_locals.area,
                h_var=self._gfdl_mp_v3_locals.h_var,
                table_0=self._saturation_tables.table_0,
                table_2=self._saturation_tables.table_2,
                dtable_0=self._saturation_tables.dtable_0,
                dtable_2=self._saturation_tables.dtable_2,
            )

        # -----------------------------------------------------------------------
        # radar reflectivity diagnostic
        # -----------------------------------------------------------------------
        if self._mp_namelist.DO_REF and self._mp_config.LAST_STEP:
            self._radar_reflectivity(
                t=self._gfdl_mp_v3_locals.t,
                density=self._gfdl_mp_v3_locals.density,
                density_factor=self._gfdl_mp_v3_locals.density_factor,
                graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                dbz=self._gfdl_mp_v3_locals.reflectivity,
            )

        # =======================================================================
        # calculation of particle concentration (pc), effective diameter (ed),
        # optical extinction (oe), radar reflectivity factor (rr), and
        # mass-weighted terminal velocity (tv)
        # =======================================================================
        if self._mp_namelist.DO_MP_DIAG:
            ndsl_log.error(
                "[GFDL1M Microphysics]: NDSL version of GFDLMPV3 with DO_MP_DIAG = True option has not been implemented. "
                "This value should have been caught by the configuration checker, and this error should never be triggered. Other problems exist."
            )
            raise ValueError(
                "[GFDL1M Microphysics]: NDSL version of GFDLMPV3 with DO_MP_DIAG = True option has not been implemented. "
                "This value should have been caught by the configuration checker, and this error should never be triggered. Other problems exist."
            )

        # -----------------------------------------------------------------------
        # momentum transportation during sedimentation
        # update temperature before delp and q update
        # -----------------------------------------------------------------------
        if self._mp_namelist.DO_SEDI_UV and self._mp_namelist.DO_SEDI_HEAT:
            self._sedi_momentum_part_1(
                t=self._gfdl_mp_v3_locals.t,
                vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                u=state.u,
                local_u=self._gfdl_mp_v3_locals.u,
                v=state.v,
                local_v=self._gfdl_mp_v3_locals.v,
            )
            self._sedi_momentum_part_2(
                t=self._gfdl_mp_v3_locals.t,
                vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                w=state.w,
                local_w=self._gfdl_mp_v3_locals.w,
            )

        # -----------------------------------------------------------------------
        # total_energy_checker
        # -----------------------------------------------------------------------
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._gfdl_mp_v3_locals.t,
                total_energy=self._gfdl_mp_v3_locals.total_energy.end_d,
                total_water=self._gfdl_mp_v3_locals.total_water.end_d,
                total_energy_b=self._gfdl_mp_v3_locals.total_energy.b_end_d,
                total_water_b=self._gfdl_mp_v3_locals.total_water.b_end_d,
                u=self._gfdl_mp_v3_locals.u,
                v=self._gfdl_mp_v3_locals.v,
                w=self._gfdl_mp_v3_locals.w,
                dp=self._gfdl_mp_v3_locals.dry_dp,
                cloud_vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                cloud_ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                cloud_liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                cloud_rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                cloud_snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                vapor=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                precip_ice=state.precipitation_at_surface.ice,
                precip_liquid=state.precipitation_at_surface.water,
                precip_rain=state.precipitation_at_surface.rain,
                precip_snow=state.precipitation_at_surface.snow,
                precip_graupel=state.precipitation_at_surface.graupel,
                dtotal_energy=self._gfdl_mp_v3_locals.total_energy.delta,
                sen=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                stress=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                moist_q=False,
                save_te_loss=True,
                total_energy_loss=self._gfdl_mp_v3_locals.total_energy.loss,
            )

        # -----------------------------------------------------------------------
        # fix negative water species
        # -----------------------------------------------------------------------
        if self._mp_namelist.FIX_NEGATIVE:
            self._fix_negative_water_species(
                t=self._gfdl_mp_v3_locals.t,
                dry_dp=self._gfdl_mp_v3_locals.dry_dp,
                vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=self._gfdl_mp_v3_locals.cloud_fraction,
                mppcw=self._gfdl_mp_v3_locals.mppcw,
                mppfr=self._gfdl_mp_v3_locals.mppfr,
            )

        # -----------------------------------------------------------------------
        # update state fields
        # -----------------------------------------------------------------------
        self._restore_and_update_humidities(
            t=self._gfdl_mp_v3_locals.t,
            dry_dp=self._gfdl_mp_v3_locals.dry_dp,
            moist_dp_end=self._gfdl_mp_v3_locals.moist_dp_end,
            moist_dp_original=self._gfdl_mp_v3_locals.moist_dp_original,
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
            cloud_fraction=state.radiation_field.cloud_fraction,
            local_vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
            local_ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
            local_liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
            local_graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
            local_rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
            local_snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
            local_cloud_fraction=self._gfdl_mp_v3_locals.cloud_fraction,
            output_reflectivity=gfdl_1m_locals.reflectivity,
            local_reflectivity=self._gfdl_mp_v3_locals.reflectivity,
            dcloud_fraction_dt=gfdl_1m_locals.dcloud_fraction_dt,
            local_condensate=self._gfdl_mp_v3_locals.condensate,
            local_kappa=self._gfdl_mp_v3_locals.kappa,
            local_total_liquid=self._gfdl_mp_v3_locals.total_liquid,
            local_total_solid=self._gfdl_mp_v3_locals.total_solid,
        )

        if self._mp_namelist.DO_SEDI_UV:
            self._copy(input=self._gfdl_mp_v3_locals.u, output=state.u)
            self._copy(input=self._gfdl_mp_v3_locals.v, output=state.v)

        if self._mp_namelist.DO_SEDI_W:
            self._copy(input=self._gfdl_mp_v3_locals.w, output=state.w)

        # -----------------------------------------------------------------------
        # total_energy_checker
        # -----------------------------------------------------------------------
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._gfdl_mp_v3_locals.t,
                total_energy=self._gfdl_mp_v3_locals.total_energy.end_m,
                total_water=self._gfdl_mp_v3_locals.total_water.end_m,
                total_energy_b=self._gfdl_mp_v3_locals.total_energy.b_end_m,
                total_water_b=self._gfdl_mp_v3_locals.total_water.b_end_m,
                u=self._gfdl_mp_v3_locals.u,
                v=self._gfdl_mp_v3_locals.v,
                w=self._gfdl_mp_v3_locals.w,
                dp=self._gfdl_mp_v3_locals.moist_dp_end,
                cloud_vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
                cloud_ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
                cloud_liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
                cloud_rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
                cloud_snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
                vapor=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                precip_ice=state.precipitation_at_surface.ice,
                precip_liquid=state.precipitation_at_surface.water,
                precip_rain=state.precipitation_at_surface.rain,
                precip_snow=state.precipitation_at_surface.snow,
                precip_graupel=state.precipitation_at_surface.graupel,
                dtotal_energy=self._gfdl_mp_v3_locals.total_energy.delta,
                sen=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                stress=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                moist_q=True,
                save_te_loss=False,
                total_energy_loss=self._dummy_field_no_read_no_write_2d_64_bit,
            )

        # -----------------------------------------------------------------------
        # calculate total energy loss or gain
        # -----------------------------------------------------------------------
        self._calculate_total_energy_change(
            t=self._gfdl_mp_v3_locals.t,
            dp=self._gfdl_mp_v3_locals.moist_dp_end,
            vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
            total_energy=self._gfdl_mp_v3_locals.total_energy.magnitude,
        )

        # -----------------------------------------------------------------------
        # conversion of temperature
        # -----------------------------------------------------------------------
        self._convert_temperature_end(
            t_state=state.t,
            t_local=self._gfdl_mp_v3_locals.t,
            vapor=self._gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=self._gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=self._gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=self._gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=self._gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=self._gfdl_mp_v3_locals.mixing_ratio.snow,
            total_liquid=self._gfdl_mp_v3_locals.total_liquid,
            total_solid=self._gfdl_mp_v3_locals.total_solid,
        )

        # -----------------------------------------------------------------------
        # total energy checker
        # -----------------------------------------------------------------------
        if self._mp_namelist.CONSV_CHECKER:
            ndsl_log.debug(
                "[GFDL1M Microphysics]: NDSL version of GFDLMPV3 with CONSV_CHECKER = True has a series of prints at the end of GFDLMPV3Driver (mpdrv in Fortran) "
                "which have not been ported. NDSL has other debug tools which should replace this, but it can also be recreated using DaCe orchestration of "
                "standard Python code. Contact the DSL support team for assistance."
            )
