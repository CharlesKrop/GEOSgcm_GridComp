import dataclasses

from ndsl import Local, LocalState, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, exp, interval, log
from ndsl.dsl.typing import Float, Float64, FloatField, FloatField64, FloatFieldIJ

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.microphysics.config import (
    GFDLMPV3CloudMPConfig,
    GFDLMPV3NamelistConfig,
    GFDLMPV3TableL3xL10,
    GFDLMPV3TableL20,
    GFDLMPV3TableL4,
    GFDLMPV3TableL2,
)
from pyMoist.microphysics.GFDL_1M.microphysics.constants import CFMIN, QCMIN, TICE, QPMIN
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.shared import (
    accretion_2d,
    accretion_3d,
    calc_mhc_lhc_wrapper,
    linear_prof,
    new_ice_condensate,
    new_liquid_condensate,
    p_melt,
    update_hydrometeors,
    update_hydrometeors_and_temperature,
)
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_table_functions import saturation_specific_humidity
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3SaturationTable, GFDLMPV3Tables


def p_graupel_accretion_to_cloud_water_and_rain(
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
    terminal_velocity_graupel: FloatField,
    terminal_velocity_liquid: FloatField,
    terminal_velocity_rain: FloatField,
    mppxg: FloatFieldIJ,
    one_minus_sigma: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
):
    from __externals__ import BLINH, BLING, CGACR, CGACW, CONV_FACTOR, D1_ICE, D1_VAP, DO_3D_ACC_CLIQ, DO_HAIL, DO_QA, DT, MUH, MUG, LI00, LI20, LV00, T_WFR, VDIFFFLAG

    with computation(FORWARD), interval(0, 1):
        scaled_CGACW = CGACW * (1.0e-2 * (1.0 - one_minus_sigma) + one_minus_sigma)

    with computation(PARALLEL), interval(...):
        if t < TICE and graupel > QPMIN:

            tc = t - TICE

            pgacw = 0.0
            if liquid > QCMIN:
                if DO_3D_ACC_CLIQ:
                    pgacw = DT * accretion_3d(
                        v1=terminal_velocity_graupel,
                        v2=terminal_velocity_liquid,
                        condensate_1=liquid,
                        condensate_2=graupel,
                        density=density,
                        c=scaled_CGACW,
                        acc1=ACC.A[16],
                        acc2=ACC.A[17],
                        acco=ACCO,
                        acco_column=8,
                        VDIFFFLAG=VDIFFFLAG,
                    )

                else:
                    graupel_x_density = graupel * density
                    if DO_HAIL:
                        factor = DT * accretion_2d(
                            condensate_x_density=graupel_x_density,
                            density_factor=density_factor,
                            c=scaled_CGACW,
                            blin=BLINH,
                            mu=MUH,
                        )
                    else:
                        factor = DT * accretion_2d(
                            condensate_x_density=graupel_x_density,
                            density_factor=density_factor,
                            c=scaled_CGACW,
                            blin=BLING,
                            mu=MUG,
                        )
                    pgacw = factor / (1.0 + factor) * liquid

            pgacr = 0.0
            if rain > QPMIN:
                pgacr = min(
                    DT
                    * accretion_3d(
                        v1=terminal_velocity_graupel,
                        v2=terminal_velocity_rain,
                        condensate_1=rain,
                        condensate_2=graupel,
                        density=density,
                        c=CGACR,
                        acc1=ACC.A[4],
                        acc2=ACC.A[5],
                        acco=ACCO,
                        acco_column=2,
                        VDIFFFLAG=VDIFFFLAG,
                    ),
                    rain,
                )

            # --- 1. Apply Mass Limits First ---
            if pgacr > QCMIN:
                factor = min(pgacr, rain) / pgacr
                pgacr = factor * pgacr
            else:
                pgacr = 0.0
            if pgacw > QCMIN:
                factor = min(pgacw, liquid) / pgacw
                pgacw = factor * pgacw
            else:
                pgacw = 0.0
            # --- 2. Apply Combined Thermal Limit ---
            sink = pgacr + pgacw
            if sink > QCMIN:
                factor = min(sink, max(0.0, -tc / icpk)) / sink
                pgacr = factor * pgacr
                pgacw = factor * pgacw
                sink = pgacr + pgacw  # Update sink for the final update_qt call

            mpprg = mpprg + sink * dry_dp * CONV_FACTOR

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
                dliquid=-pgacw,
                dgraupel=sink,
                drain=-pgacr,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


def p_graupel_accretion_to_ice(
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
    terminal_velocity_graupel: FloatField,
    terminal_velocity_ice: FloatField,
    mppxg: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
):
    """graupel accretion with cloud ice, Lin et al. (1983)

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
        terminal_velocity_graupel (FloatField)
        terminal_velocity_ice (FloatField)
        mppxg (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
    """
    from __externals__ import BLINH, BLING, CGACI, CONV_FACTOR, DO_3D_ACC_CICE, DO_HAIL, DT, FI2G_FAC, MUH, MUG, VDIFFFLAG

    with computation(PARALLEL), interval(...):
        if t < TICE and ice > QCMIN:
            tc = t - TICE

            sink = 0.0
            graupel_x_density = graupel * density
            if graupel > QPMIN:
                if DO_3D_ACC_CICE:
                    sink = DT * accretion_3d(
                        v1=terminal_velocity_graupel,
                        v2=terminal_velocity_ice,
                        condensate_1=ice,
                        condensate_2=graupel,
                        density=density,
                        c=CGACI,
                        acc1=ACC.A[18],
                        acc2=ACC.A[19],
                        acco=ACCO,
                        acco_column=9,
                        VDIFFFLAG=VDIFFFLAG,
                    )
                else:
                    if DO_HAIL:
                        factor = DT * accretion_2d(
                            condensate_x_density=graupel_x_density,
                            density_factor=density_factor,
                            c=CGACI,
                            blin=BLINH,
                            mu=MUH,
                        )
                    else:
                        factor = DT * accretion_2d(
                            condensate_x_density=graupel_x_density,
                            density_factor=density_factor,
                            c=CGACI,
                            blin=BLING,
                            mu=MUG,
                        )
                    sink = factor / (1.0 + factor) * ice

            sink = min(FI2G_FAC * ice, sink)
            mppxg = mppxg + sink * dry_dp * CONV_FACTOR

            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction,
                vapor,
                ice,
                liquid,
                graupel,
                rain,
                snow,
                dvapor=0.0,
                dice=-sink,
                dliquid=0.0,
                dgraupel=sink,
                drain=0.0,
                dsnow=0.0,
            )


def p_graupel_accretion_to_snow(
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
    terminal_velocity_graupel: FloatField,
    terminal_velocity_snow: FloatField,
    mppxg: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
):
    """graupel accretion with snow, Lin et al. (1983)

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
        terminal_velocity_graupel (FloatField)
        terminal_velocity_snow (FloatField)
        mppxg (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
    """
    from __externals__ import CGACS, CONV_FACTOR, DT, FS2G_FAC, VDIFFFLAG

    with computation(PARALLEL), interval(...):
        if t < TICE and snow > QPMIN and graupel > QPMIN:
            tc = t - TICE

            sink = DT * accretion_3d(
                v1=terminal_velocity_graupel,
                v2=terminal_velocity_snow,
                condensate_1=snow,
                condensate_2=graupel,
                density=density,
                c=CGACS,
                acc1=ACC.A[6],
                acc2=ACC.A[7],
                acco=ACCO,
                acco_column=3,
                VDIFFFLAG=VDIFFFLAG,
            )

            sink = min(FS2G_FAC * snow, sink)
            mppxg = mppxg + sink * dry_dp * CONV_FACTOR

            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction,
                vapor,
                ice,
                liquid,
                graupel,
                rain,
                snow,
                dvapor=0.0,
                dice=0.0,
                dliquid=0.0,
                dgraupel=sink,
                drain=0.0,
                dsnow=-sink,
            )


def p_graupel_melt(
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
    terminal_velocity_graupel: FloatField,
    terminal_velocity_liquid: FloatField,
    terminal_velocity_rain: FloatField,
    mppmg: FloatField,
    one_minus_sigma: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
    CGMLT: GFDLMPV3TableL4,
    table_2: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """graupel melting (includes graupel accretion with cloud water and rain) to form rain Lin et al. (1983)

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
        terminal_velocity_graupel (FloatField)
        terminal_velocity_liquid (FloatField)
        terminal_velocity_rain (FloatField)
        mppmg (FloatField)
        one_minus_sigma (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
        CGMLT (GFDLMPV3TableL4)
        table_2 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)
    """

    from __externals__ import BLINH, BLING, CGACR, CGACW, CONV_FACTOR, D1_ICE, D1_VAP, DO_3D_ACC_CLIQ, DO_HAIL, DO_QA, DT, MUH, MUG, LI00, LI20, LV00, T_WFR, VDIFFFLAG

    with computation(FORWARD), interval(0, 1):
        scaled_CGACW: FloatFieldIJ = CGACW * (1.0e-2 * (1.0 - one_minus_sigma) + one_minus_sigma)

    with computation(PARALLEL), interval(...):
        if t >= TICE and graupel > QPMIN:
            tc = t - TICE

            pgacw = 0.0
            graupel_x_density = graupel * density
            if liquid > QCMIN:
                if DO_3D_ACC_CLIQ:
                    pgacw = accretion_3d(
                        v1=terminal_velocity_graupel,
                        v2=terminal_velocity_liquid,
                        condensate_1=liquid,
                        condensate_2=graupel,
                        density=density,
                        c=scaled_CGACW,
                        acc1=ACC.A[16],
                        acc2=ACC.A[17],
                        acco=ACCO,
                        acco_column=8,
                        VDIFFFLAG=VDIFFFLAG,
                    )
                else:
                    if DO_HAIL:
                        factor = accretion_2d(
                            condensate_x_density=graupel_x_density,
                            density_factor=density_factor,
                            c=scaled_CGACW,
                            blin=BLINH,
                            mu=MUH,
                        )
                    else:
                        factor = accretion_2d(
                            condensate_x_density=graupel_x_density,
                            density_factor=density_factor,
                            c=scaled_CGACW,
                            blin=BLING,
                            mu=MUG,
                        )
                    pgacw = factor / (1.0 + DT * factor) * liquid

            pgacr = 0.0
            if rain > QPMIN:
                pgacr = min(
                    accretion_3d(
                        v1=terminal_velocity_graupel,
                        v2=terminal_velocity_rain,
                        condensate_1=rain,
                        condensate_2=graupel,
                        density=density,
                        c=CGACR,
                        acc1=ACC.A[4],
                        acc2=ACC.A[5],
                        acco=ACCO,
                        acco_column=2,
                        VDIFFFLAG=VDIFFFLAG,
                    ),
                    rain / DT,
                )

            t_in = t
            sat_spec_humidity, dsat_spec_humiditydt = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            dq = sat_spec_humidity - vapor
            if DO_HAIL:
                sink = max(
                    0.0,
                    p_melt(
                        t=t,
                        dcondensate=dq,
                        condensate_x_density=graupel_x_density,
                        pxacw=pgacw,
                        pxacr=pgacr,
                        density=density,
                        density_factor=density_factor,
                        blin=BLINH,
                        mu=MUH,
                        lcpk=lcpk,
                        icpk=icpk,
                        cvm=cvm,
                        c=CGMLT,
                    ),
                )
            else:
                sink = max(
                    0.0,
                    p_melt(
                        t=t,
                        dcondensate=dq,
                        condensate_x_density=graupel_x_density,
                        pxacw=pgacw,
                        pxacr=pgacr,
                        density=density,
                        density_factor=density_factor,
                        blin=BLING,
                        mu=MUG,
                        lcpk=lcpk,
                        icpk=icpk,
                        cvm=cvm,
                        c=CGMLT,
                    ),
                )

            sink = min(graupel, sink * DT, tc / icpk)
            mppmg = mppmg + sink * dry_dp * CONV_FACTOR

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
                dgraupel=-sink,
                drain=sink,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


def p_ice_melt_freeze(
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
    mppfw: FloatFieldIJ,
    mppmi: FloatFieldIJ,
    one_minus_sigma: FloatFieldIJ,
    convection_fraction: FloatFieldIJ,
    surface_type: FloatFieldIJ,
):
    """Estimates two processes: cloud ice melting to form cloud water and rain, Lin et al. (1983),
    and cloud water homogeneous freezing to form cloud ice and snow, Lin et al. (1983)

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
        mppfw (FloatFieldIJ)
        mppmi (FloatFieldIJ)
        one_minus_sigma (FloatFieldIJ)
        convection_fraction (FloatFieldIJ)
        surface_type (FloatFieldIJ)
    """
    from __externals__ import CONV_FACTOR, D1_ICE, D1_VAP, DO_QA, DT, IN_CLOUD_ICE, LI00, LI20, LV00, PSAUT_QI_CRT, QL_MLT, T_WFR, TAU_FREZ, TAU_IMLT

    # psaut_qi_crt (ice to snow conversion) has strong resolution dependence
    # account for this using onemsig to convert more ice to snow at coarser resolutions
    with computation(FORWARD), interval(0, 1):
        critical_ice_factor: FloatFieldIJ = PSAUT_QI_CRT * (1.0e-1 * (1.0 - one_minus_sigma) + one_minus_sigma)

        fac_imlt: FloatFieldIJ = 1.0 - exp(-DT / TAU_IMLT)
        fac_frez: FloatFieldIJ = 1.0 - exp(-DT / TAU_FREZ)

    with computation(PARALLEL), interval(...):
        if t > TICE and ice > QCMIN:
            # Use In-Cloud condensates with scale-aware blending
            if IN_CLOUD_ICE:
                # Enforce minimum bound to prevent vanishing values
                cloud_fraction_bounded = max(cloud_fraction, CFMIN)
            else:
                cloud_fraction_bounded = 1.0
            liquid_internal = liquid / cloud_fraction_bounded
            ice_internal = ice / cloud_fraction_bounded

            tmp = t
            newliq = new_liquid_condensate(t=tmp, liquid=liquid_internal, ice=ice_internal, convection_fraction=convection_fraction, surface_type=surface_type)
            sink = fac_imlt * min(ice_internal, newliq, (t - TICE) / icpk / cloud_fraction_bounded)
            tmp = min(sink, max(QL_MLT / cloud_fraction_bounded - liquid_internal, 0.0))

            tmp = tmp * cloud_fraction_bounded
            sink = sink * cloud_fraction_bounded
            mppmi = mppmi + sink * dry_dp * CONV_FACTOR

            t, vapor, ice, liquid, graupel, rain, snow, cloud_fraction, cvm, total_energy, lcpk, icpk, tcpk, tcp3 = update_hydrometeors_and_temperature(
                cloud_fraction=cloud_fraction,
                vapor=vapor,
                ice=ice,
                liquid=liquid,
                graupel=graupel,
                rain=rain,
                snow=snow,
                dvapor=0.0,
                dice=-sink,
                dliquid=tmp,
                dgraupel=0.0,
                drain=sink - tmp,
                dsnow=0.0,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )

        elif t <= TICE and liquid > QCMIN:
            # Use In-Cloud condensates with scale-aware blending
            if IN_CLOUD_ICE:
                # Enforce minimum bound to prevent vanishing values
                cloud_fraction_bounded = max(cloud_fraction, CFMIN)
            else:
                cloud_fraction_bounded = 1.0
            liquid_internal = liquid / cloud_fraction_bounded
            ice_internal = ice / cloud_fraction_bounded

            tmp = t
            newice = new_ice_condensate(t=tmp, liquid=liquid_internal, ice=ice_internal, convection_fraction=convection_fraction, surface_type=surface_type)
            sink = fac_frez * min(liquid_internal, newice, (TICE - t) / icpk / cloud_fraction_bounded)
            ice_modified = critical_ice_factor / density
            tmp = min(sink, max(ice_modified / cloud_fraction_bounded - ice_internal, 0.0))

            tmp = tmp * cloud_fraction_bounded
            sink = sink * cloud_fraction_bounded
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


def p_ice_to_snow_autoconversion(
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
    dice: FloatField,
    mppas: FloatFieldIJ,
    one_minus_sigma: FloatFieldIJ,
):
    """cloud ice to snow autoconversion, Lin et al. (1983)

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
        dice (FloatField)
        mppas (FloatFieldIJ)
        one_minus_sigma (FloatFieldIJ)
    """
    from __externals__ import CONV_FACTOR, DT, IN_CLOUD_ICE, PSAUT_QI_CRT, TAU_I2S

    with computation(FORWARD), interval(0, 1):
        critical_ice_factor: FloatFieldIJ = PSAUT_QI_CRT * (1.0e-1 * (1.0 - one_minus_sigma) + one_minus_sigma)

        fac_i2s: FloatFieldIJ = 1.0 - exp(-DT / TAU_I2S)

    with computation(PARALLEL), interval(...):

        if t < TICE and ice > QCMIN:

            tc = t - TICE

            # Use In-Cloud condensates with scale-aware blending
            if IN_CLOUD_ICE:
                # Enforce minimum bound to prevent vanishing values
                cloud_fraction_internal = max(cloud_fraction, CFMIN)
            else:
                cloud_fraction_internal = 1.0

            ice_internal = ice / cloud_fraction_internal
            dice_internal = dice / cloud_fraction_internal

            sink = 0.0
            dice_internal = max(dice_internal, QCMIN)
            ice_plus = ice_internal + dice_internal
            ice_modified = critical_ice_factor / density / cloud_fraction_internal
            if ice_plus > (ice_modified + QCMIN):
                if ice_modified > (ice_internal - dice_internal):
                    dq = (0.25 * (ice_plus - ice_modified) ** 2) / dice_internal
                else:
                    dq = ice_internal - ice_modified

                sink = fac_i2s * exp(0.025 * tc) * dq

            sink = min(ice_internal, sink) * cloud_fraction_internal
            mppas = mppas + sink * dry_dp * CONV_FACTOR

            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction,
                vapor,
                ice,
                liquid,
                graupel,
                rain,
                snow,
                dvapor=0.0,
                dice=-sink,
                dliquid=0.0,
                dgraupel=0.0,
                drain=0.0,
                dsnow=sink,
            )


def p_snow_accretion_to_ice(
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
    terminal_velocity_ice: FloatField,
    terminal_velocity_snow: FloatField,
    mppxs: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
):
    """snow accretion with cloud ice, Lin et al. (1983)

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
        terminal_velocity_ice (FloatField)
        terminal_velocity_snow (FloatField)
        mppxs (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
    """
    from __externals__ import BLINS, CONV_FACTOR, CSACI, DO_3D_ACC_CICE, DT, FI2S_FAC, MUS, VDIFFFLAG

    with computation(PARALLEL), interval(...):
        if t < TICE and ice > QCMIN:
            sink = 0.0
            snow_x_density = snow * density
            if snow > QPMIN:
                if DO_3D_ACC_CICE:
                    sink = DT * accretion_3d(
                        v1=terminal_velocity_snow,
                        v2=terminal_velocity_ice,
                        condensate_1=ice,
                        condensate_2=snow,
                        density=density,
                        c=CSACI,
                        acc1=ACC.A[14],
                        acc2=ACC.A[15],
                        acco=ACCO,
                        acco_column=7,
                        VDIFFFLAG=VDIFFFLAG,
                    )
                else:
                    factor = DT * accretion_2d(
                        condensate_x_density=snow_x_density,
                        density_factor=density_factor,
                        c=CSACI,
                        blin=BLINS,
                        mu=MUS,
                    )
                    sink = factor / (1.0 + factor) * ice

            sink = min(FI2S_FAC * ice, sink)
            mppxs = mppxs + sink * dry_dp * CONV_FACTOR

            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction,
                vapor,
                ice,
                liquid,
                graupel,
                rain,
                snow,
                dvapor=0.0,
                dice=-sink,
                dliquid=0.0,
                dgraupel=0.0,
                drain=0.0,
                dsnow=sink,
            )


def p_snow_accretion_to_rain_and_graupel(
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
    terminal_velocity_rain: FloatField,
    terminal_velocity_snow: FloatField,
    mppfr: FloatFieldIJ,
    mpprs: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
    CGFR: GFDLMPV3TableL2,
):
    """now accretion with rain and rain freezing to form graupel, Lin et al. (1983)

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
        terminal_velocity_rain (FloatField)
        terminal_velocity_snow (FloatField)
        mppfr (FloatFieldIJ)
        mpprs (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
        CGFR (GFDLMPV3TableL2)
    """
    from __externals__ import CONV_FACTOR, CSACR, D1_ICE, D1_VAP, DO_QA, DT, MUR, LI00, LI20, LV00, T_WFR, VDIFFFLAG

    with computation(PARALLEL), interval(...):
        if t < TICE and rain > QPMIN:
            tc = t - TICE

            psacr = 0.0
            if snow > QPMIN:
                psacr = DT * accretion_3d(
                    v1=terminal_velocity_snow,
                    v2=terminal_velocity_rain,
                    condensate_1=rain,
                    condensate_2=snow,
                    density=density,
                    c=CSACR,
                    acc1=ACC.A[2],
                    acc2=ACC.A[3],
                    acco=ACCO,
                    acco_column=1,
                    VDIFFFLAG=VDIFFFLAG,
                )

            # Homogeneous freezing threshold (e.g., -40 C)
            if tc < -40.0:
                # Colder than -40C: ALL liquid rain freezes instantaneously.
                # We set pgfr to consume all available qr.
                pgfr = rain
            else:
                # Warmer than -40C: Calculate probabilistic freezing normally.
                pgfr = DT * CGFR.A[0] / density * (exp(-CGFR.A[1] * tc) - 1.0) * exp((6 + MUR) / (MUR + 3) * log(6 * rain * density))

            # --- Apply Mass and Thermal Limits ---
            sink = psacr + pgfr
            if sink > QCMIN:
                factor = min(sink, rain, max(0.0, -tc / icpk)) / sink
                psacr = factor * psacr
                pgfr = factor * pgfr

            sink = psacr + pgfr

            mpprs = mpprs + psacr * dry_dp * CONV_FACTOR
            mppfr = mppfr + pgfr * dry_dp * CONV_FACTOR

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
                dgraupel=pgfr,
                drain=-sink,
                dsnow=psacr,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


def p_snow_melt(
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
    terminal_velocity_liquid: FloatField,
    terminal_velocity_rain: FloatField,
    terminal_velocity_snow: FloatField,
    mppms: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
    CSMLT: GFDLMPV3TableL4,
    table_2: GFDLMPV3SaturationTable,
    dtable_2: GFDLMPV3SaturationTable,
):
    """snow melting (includes snow accretion with cloud water and rain) to form cloud water and rain Lin et al. (1983)

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
        terminal_velocity_liquid (FloatField)
        terminal_velocity_rain (FloatField)
        terminal_velocity_snow (FloatField)
        mppms (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
        CSMLT (GFDLMPV3TableL4)
        table_2 (GFDLMPV3SaturationTable)
        dtable_2 (GFDLMPV3SaturationTable)

    """
    from __externals__ import BLINS, CONV_FACTOR, CRACS, CSACR, CSACW, D1_ICE, D1_VAP, DO_3D_ACC_CLIQ, DO_QA, DT, LI00, LI20, LV00, MUS, QS_MLT, T_WFR, VDIFFFLAG

    with computation(PARALLEL), interval(...):
        if t >= TICE and snow > QPMIN:

            tc = t - TICE

            psacw = 0.0
            snow_x_density = snow * density
            if liquid > QCMIN:
                if DO_3D_ACC_CLIQ:
                    psacw = accretion_3d(
                        v1=terminal_velocity_snow,
                        v2=terminal_velocity_liquid,
                        condensate_1=liquid,
                        condensate_2=snow,
                        density=density,
                        c=CSACW,
                        acc1=ACC.A[12],
                        acc2=ACC.A[13],
                        acco=ACCO,
                        acco_column=6,
                        VDIFFFLAG=VDIFFFLAG,
                    )
                else:
                    factor = accretion_2d(
                        condensate_x_density=snow_x_density,
                        density_factor=density_factor,
                        c=CSACW,
                        blin=BLINS,
                        mu=MUS,
                    )
                    psacw = factor / (1.0 + DT * factor) * liquid

            psacr = 0.0
            pracs = 0.0
            if rain > QPMIN:
                psacr = min(
                    accretion_3d(
                        v1=terminal_velocity_snow,
                        v2=terminal_velocity_rain,
                        condensate_1=rain,
                        condensate_2=snow,
                        density=density,
                        c=CSACR,
                        acc1=ACC.A[2],
                        acc2=ACC.A[3],
                        acco=ACCO,
                        acco_column=1,
                        VDIFFLAG=VDIFFFLAG,
                    ),
                    rain / DT,
                )
                pracs = accretion_3d(
                    v1=terminal_velocity_rain,
                    v2=terminal_velocity_snow,
                    condensate_1=snow,
                    condensate_2=rain,
                    density=density,
                    c=CRACS,
                    acc1=ACC.A[0],
                    acc2=ACC.A[1],
                    acco=ACCO,
                    acco_column=0,
                    VDIFFLAG=VDIFFFLAG,
                )

            t_in = t
            sat_spec_humidity, _ = saturation_specific_humidity(t_in, density, table_2, dtable_2)
            dq = sat_spec_humidity - vapor
            sink = max(
                0.0,
                p_melt(
                    t=t,
                    dcondensate=dq,
                    condensate_x_density=snow_x_density,
                    pxacw=psacw,
                    pxacr=psacr,
                    density=density,
                    density_factor=density_factor,
                    blin=BLINS,
                    mu=MUS,
                    lcpk=lcpk,
                    icpk=icpk,
                    cvm=cvm,
                    c=CSMLT,
                ),
            )

            sink = min(snow, (sink + pracs) * DT, tc / icpk)
            tmp = min(sink, max(QS_MLT - liquid, 0.0))
            mppms = mppms + sink * dry_dp * CONV_FACTOR

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
                dliquid=tmp,
                dgraupel=0.0,
                drain=sink - tmp,
                dsnow=-sink,
                DO_QA=DO_QA,
                D1_VAP=D1_VAP,
                D1_ICE=D1_ICE,
                LI00=LI00,
                LI20=LI20,
                LV00=LV00,
                T_WFR=T_WFR,
            )


def p_snow_to_graupel_autoconversion(
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
    mppag: FloatFieldIJ,
):
    """snow to graupel autoconversion, Lin et al. (1983)

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
        mppag (FloatFieldIJ)
    """
    from __externals__ import CONV_FACTOR, DT, FS2G_FAC, PGAUT_QS_CRT

    with computation(PARALLEL), interval(...):
        if t < TICE and snow > QPMIN:
            tc = t - TICE

            sink = 0
            qsm = PGAUT_QS_CRT / density
            if snow > qsm:
                factor = DT * 1.0e-3 * exp(0.09 * tc)
                sink = factor / (1.0 + factor) * (snow - qsm)

            sink = min(FS2G_FAC * snow, sink)
            mppag = mppag + sink * dry_dp * CONV_FACTOR

            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
                cloud_fraction,
                vapor,
                ice,
                liquid,
                graupel,
                rain,
                snow,
                dvapor=0.0,
                dice=0.0,
                dliquid=0.0,
                dgraupel=sink,
                drain=0.0,
                dsnow=-sink,
            )


@dataclasses.dataclass
class IceCloudLocals(LocalState):
    cvm: Local = dataclasses.field(
        metadata={
            "name": "cvm",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )
    dice: Local = dataclasses.field(
        metadata={
            "name": "dice",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    icpk: Local = dataclasses.field(
        metadata={
            "name": "icpk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lcpk: Local = dataclasses.field(
        metadata={
            "name": "lcpk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    tcpk: Local = dataclasses.field(
        metadata={
            "name": "tcpk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    tcp3: Local = dataclasses.field(
        metadata={
            "name": "tcp3",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_energy: Local = dataclasses.field(
        metadata={
            "name": "total_energy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )
    total_liquid: Local = dataclasses.field(
        metadata={
            "name": "total_liquid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )
    total_solid: Local = dataclasses.field(
        metadata={
            "name": "total_solid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
        }
    )


class IceCloud:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        saturation_tables: GFDLMPV3Tables,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
        DRIVER_DT: Float,
    ):
        # make config visible at runtime
        self._mp_config = mp_config
        self._mp_namelist = mp_namelist
        self._saturation_tables = saturation_tables

        # initialize locals for ice cloud
        self._ice_cloud_locals = IceCloudLocals.make_locals(quantity_factory)

        # construct stencils
        self._calc_mhc_lhc_wrapper = stencil_factory.from_dims_halo(
            func=calc_mhc_lhc_wrapper,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
            },
        )
        self._p_graupel_accretion_to_cloud_water_and_rain = stencil_factory.from_dims_halo(
            func=p_graupel_accretion_to_cloud_water_and_rain,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINH": mp_namelist.BLINH,
                "BLING": mp_namelist.BLING,
                "CGACR": mp_config.CGACR,
                "CGACW": mp_config.CGACW,
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_3D_ACC_CLIQ": mp_namelist.DO_3D_ACC_CLIQ,
                "DO_HAIL": mp_namelist.DO_HAIL,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "MUH": mp_namelist.MUH,
                "MUG": mp_namelist.MUG,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._p_graupel_accretion_to_ice = stencil_factory.from_dims_halo(
            func=p_graupel_accretion_to_ice,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINH": mp_namelist.BLINH,
                "BLING": mp_namelist.BLING,
                "CGACI": mp_config.CGACI,
                "CONV_FACTOR": CONV_FACTOR,
                "DO_3D_ACC_CICE": mp_namelist.DO_3D_ACC_CICE,
                "DO_HAIL": mp_namelist.DO_HAIL,
                "DT": DRIVER_DT,
                "FI2G_FAC": mp_namelist.FI2G_FAC,
                "MUH": mp_namelist.MUH,
                "MUG": mp_namelist.MUG,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._p_graupel_accretion_to_snow = stencil_factory.from_dims_halo(
            func=p_graupel_accretion_to_snow,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CGACS": mp_config.CGACS,
                "CONV_FACTOR": CONV_FACTOR,
                "DT": DRIVER_DT,
                "FS2G_FAC": mp_namelist.FS2G_FAC,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )

        self._p_graupel_melt = stencil_factory.from_dims_halo(
            func=p_graupel_melt,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINH": mp_namelist.BLINH,
                "BLING": mp_namelist.BLING,
                "CGACR": mp_config.CGACR,
                "CGACW": mp_config.CGACW,
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_3D_ACC_CLIQ": mp_namelist.DO_3D_ACC_CLIQ,
                "DO_HAIL": mp_namelist.DO_HAIL,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "MUH": mp_namelist.MUH,
                "MUG": mp_namelist.MUG,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._p_ice_melt_freeze = stencil_factory.from_dims_halo(
            func=p_ice_melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "IN_CLOUD_ICE": mp_namelist.IN_CLOUD_ICE,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "PSAUT_QI_CRT": mp_namelist.PSAUT_QI_CRT,
                "QL_MLT": mp_namelist.QL_MLT,
                "T_WFR": mp_config.T_WFR,
                "TAU_FREZ": mp_namelist.TAU_FREZ,
                "TAU_IMLT": mp_namelist.TAU_IMLT,
            },
        )
        self._p_ice_to_snow_autoconversion = stencil_factory.from_dims_halo(
            func=p_ice_to_snow_autoconversion,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "DT": DRIVER_DT,
                "IN_CLOUD_ICE": mp_namelist.IN_CLOUD_ICE,
                "PSAUT_QI_CRT": mp_namelist.PSAUT_QI_CRT,
                "TSU_I2S": mp_namelist.TAU_I2S,
            },
        )
        self._linear_prof = stencil_factory.from_dims_halo(
            func=linear_prof,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"IRAIN_F": mp_namelist.IRAIN_F, "Z_SLOPE": mp_namelist.Z_SLOPE_LIQ},
        )
        self._p_snow_accretion_to_ice = stencil_factory.from_dims_halo(
            func=p_snow_accretion_to_ice,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINS": mp_namelist.BLINS,
                "CONV_FACTOR": CONV_FACTOR,
                "CSACI": mp_config.CSACI,
                "DO_3D_ACC_CICE": mp_namelist.DO_3D_ACC_CICE,
                "DT": DRIVER_DT,
                "FI2S_FAC": mp_namelist.FI2S_FAC,
                "MUS": mp_namelist.MUS,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._p_snow_accretion_to_rain_and_graupel = stencil_factory.from_dims_halo(
            func=p_snow_accretion_to_rain_and_graupel,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "CSACR": mp_config.CSACR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "MUR": mp_namelist.MUR,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "T_WFR": mp_config.T_WFR,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._p_snow_melt = stencil_factory.from_dims_halo(
            func=p_snow_melt,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINS": mp_namelist.BLINS,
                "CONV_FACTOR": CONV_FACTOR,
                "CRACS": mp_config.CRACS,
                "CSACR": mp_config.CSACR,
                "CSACW": mp_config.CSACW,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_3D_ACC_CLIQ": mp_namelist.DO_3D_ACC_CLIQ,
                "DO_QA": mp_namelist.DO_QA,
                "DT": DRIVER_DT,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "MUS": mp_namelist.MUS,
                "QS_MLT": mp_namelist.QS_MLT,
                "T_WFR": mp_config.T_WFR,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._p_snow_to_graupel_autoconversion = stencil_factory.from_dims_halo(
            func=p_snow_to_graupel_autoconversion,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "DT": DRIVER_DT,
                "FS2G_FAC": mp_namelist.FS2G_FAC,
                "PGAUT_QS_CRT": mp_namelist.PGAUT_QS_CRT,
            },
        )

    def __call__(self, gfdl_mp_v3_locals: GFDLMPV3Locals, mp_full_locals: MPFullLocals):
        # --------------------------------------------------
        # calculate heat capacities and latent heat coefficients
        # --------------------------------------------------
        self._calc_mhc_lhc_wrapper(
            t=gfdl_mp_v3_locals.t,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            total_liquid=self._ice_cloud_locals.total_liquid,
            total_solid=self._ice_cloud_locals.total_solid,
            cvm=self._ice_cloud_locals.cvm,
            total_energy=self._ice_cloud_locals.total_energy,
            lcpk=self._ice_cloud_locals.lcpk,
            icpk=self._ice_cloud_locals.icpk,
            tcpk=self._ice_cloud_locals.tcpk,
            tcp3=self._ice_cloud_locals.tcp3,
        )

        if not self._mp_namelist.DO_WARM_RAIN_MP:
            # -----------------------------------------------------------------------
            # cloud ice/liq melt/freeze to form cloud water/ice and rain/snow
            # -----------------------------------------------------------------------
            self._p_ice_melt_freeze(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                cvm=self._ice_cloud_locals.cvm,
                icpk=self._ice_cloud_locals.icpk,
                lcpk=self._ice_cloud_locals.lcpk,
                tcpk=self._ice_cloud_locals.tcpk,
                tcp3=self._ice_cloud_locals.tcp3,
                total_energy=self._ice_cloud_locals.total_energy,
                mppfw=gfdl_mp_v3_locals.mppfw,
                mppmi=gfdl_mp_v3_locals.mppmi,
                one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
                convection_fraction=gfdl_mp_v3_locals.convection_fraction,
                surface_type=gfdl_mp_v3_locals.surface_type,
            )

            # -----------------------------------------------------------------------
            # vertical subgrid variability
            # -----------------------------------------------------------------------
            self._linear_prof(
                precipitate=gfdl_mp_v3_locals.mixing_ratio.ice,
                dm=self._ice_cloud_locals.dice,
                h_var=gfdl_mp_v3_locals.h_var,
            )

            # -----------------------------------------------------------------------
            # snow melting (includes snow accretion with cloud water and rain) to form cloud water and rain
            # -----------------------------------------------------------------------
            self._p_snow_melt(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                cvm=self._ice_cloud_locals.cvm,
                icpk=self._ice_cloud_locals.icpk,
                lcpk=self._ice_cloud_locals.lcpk,
                tcpk=self._ice_cloud_locals.tcpk,
                tcp3=self._ice_cloud_locals.tcp3,
                total_energy=self._ice_cloud_locals.total_energy,
                terminal_velocity_liquid=mp_full_locals.terminal_velocity.liquid,
                terminal_velocity_rain=mp_full_locals.terminal_velocity.rain,
                terminal_velocity_snow=mp_full_locals.terminal_velocity.snow,
                mppms=gfdl_mp_v3_locals.mppms,
                # tables
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
                CSMLT=self._mp_config.CSMLT,
                table_2=self._saturation_tables.table_2,
                dtable_2=self._saturation_tables.dtable_2,
            )

            # -----------------------------------------------------------------------
            # graupel melting (includes graupel accretion with cloud water and rain) to form rain
            # -----------------------------------------------------------------------
            self._p_graupel_melt(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                cvm=self._ice_cloud_locals.cvm,
                icpk=self._ice_cloud_locals.icpk,
                lcpk=self._ice_cloud_locals.lcpk,
                tcpk=self._ice_cloud_locals.tcpk,
                tcp3=self._ice_cloud_locals.tcp3,
                total_energy=self._ice_cloud_locals.total_energy,
                terminal_velocity_graupel=mp_full_locals.terminal_velocity.graupel,
                terminal_velocity_liquid=mp_full_locals.terminal_velocity.liquid,
                terminal_velocity_rain=mp_full_locals.terminal_velocity.rain,
                mppmg=gfdl_mp_v3_locals.mppmg,
                one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
                CGMLT=self._mp_config.CGMLT,
                table_2=self._saturation_tables.table_2,
                dtable_2=self._saturation_tables.dtable_2,
            )

            # -----------------------------------------------------------------------
            # snow accretion with cloud ice
            # -----------------------------------------------------------------------
            self._p_snow_accretion_to_ice(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                terminal_velocity_ice=mp_full_locals.terminal_velocity.ice,
                terminal_velocity_snow=mp_full_locals.terminal_velocity.snow,
                mppxs=gfdl_mp_v3_locals.mppxs,
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
            )

            # -----------------------------------------------------------------------
            # cloud ice to snow autoconversion
            # -----------------------------------------------------------------------
            self._p_ice_to_snow_autoconversion(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                dice=self._ice_cloud_locals.dice,
                mppas=gfdl_mp_v3_locals.mppas,
                one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
            )

            # -----------------------------------------------------------------------
            # graupel accretion with cloud ice
            # -----------------------------------------------------------------------
            self._p_graupel_accretion_to_ice(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                terminal_velocity_graupel=mp_full_locals.terminal_velocity.graupel,
                terminal_velocity_ice=mp_full_locals.terminal_velocity.ice,
                mppxg=gfdl_mp_v3_locals.mppxg,
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
            )

            # -----------------------------------------------------------------------
            # snow accretion with rain and rain freezing to form graupel
            # -----------------------------------------------------------------------
            self._p_snow_accretion_to_rain_and_graupel(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                cvm=self._ice_cloud_locals.cvm,
                icpk=self._ice_cloud_locals.icpk,
                lcpk=self._ice_cloud_locals.lcpk,
                tcpk=self._ice_cloud_locals.tcpk,
                tcp3=self._ice_cloud_locals.tcp3,
                total_energy=self._ice_cloud_locals.total_energy,
                terminal_velocity_rain=mp_full_locals.terminal_velocity.rain,
                terminal_velocity_snow=mp_full_locals.terminal_velocity.snow,
                mppfr=gfdl_mp_v3_locals.mppfr,
                mpprs=gfdl_mp_v3_locals.mpprs,
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
                CGFR=self._mp_config.CGFR,
            )

            # -----------------------------------------------------------------------
            # graupel accretion with snow
            # -----------------------------------------------------------------------
            self._p_graupel_accretion_to_snow(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                terminal_velocity_graupel=mp_full_locals.terminal_velocity.graupel,
                terminal_velocity_snow=mp_full_locals.terminal_velocity.snow,
                mppxg=gfdl_mp_v3_locals.mppxg,
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
            )

            # -----------------------------------------------------------------------
            # snow to graupel autoconversion
            # -----------------------------------------------------------------------
            self._p_snow_to_graupel_autoconversion(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                mppag=gfdl_mp_v3_locals.mppag,
            )

            # -----------------------------------------------------------------------
            # graupel accretion with cloud water and rain
            # -----------------------------------------------------------------------
            self._p_graupel_accretion_to_cloud_water_and_rain(
                t=gfdl_mp_v3_locals.t,
                dry_dp=gfdl_mp_v3_locals.dry_dp,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
                ice=gfdl_mp_v3_locals.mixing_ratio.ice,
                liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
                graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
                rain=gfdl_mp_v3_locals.mixing_ratio.rain,
                snow=gfdl_mp_v3_locals.mixing_ratio.snow,
                cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
                cvm=self._ice_cloud_locals.cvm,
                icpk=self._ice_cloud_locals.icpk,
                lcpk=self._ice_cloud_locals.lcpk,
                tcpk=self._ice_cloud_locals.tcpk,
                tcp3=self._ice_cloud_locals.tcp3,
                total_energy=self._ice_cloud_locals.total_energy,
                terminal_velocity_graupel=mp_full_locals.terminal_velocity.graupel,
                terminal_velocity_liquid=mp_full_locals.terminal_velocity.liquid,
                terminal_velocity_rain=mp_full_locals.terminal_velocity.rain,
                mppxg=gfdl_mp_v3_locals.mppxg,
                one_minus_sigma=gfdl_mp_v3_locals.one_minus_sigma,
                ACC=self._mp_config.ACC,
                ACCO=self._mp_config.ACCO,
            )
