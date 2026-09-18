import dataclasses

from ndsl import Local, LocalState, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, exp, interval
from ndsl.dsl.typing import Float, Float64, FloatField, FloatField64, FloatFieldIJ

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig, GFDLMPV3TableL3xL10, GFDLMPV3TableL20, GFDLMPV3TableL4
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
    update_hydrometeors_and_temperature,
)
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_table_functions import saturation_specific_humidity
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3SaturationTable, GFDLMPV3Tables


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
                        VDIFFLAG=VDIFFFLAG,
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
            sat_spec_humidity = saturation_specific_humidity(t_in, density, table_2, dtable_2)
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
        gfdl_1m_config: GFDL1MConfig,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
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
        self._p_ice_melt_freeze = stencil_factory.from_dims_halo(
            func=p_ice_melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_QA": mp_namelist.DO_QA,
                "DT": gfdl_1m_config.DT_MOIST,
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
        self._linear_prof = stencil_factory.from_dims_halo(
            func=linear_prof,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"IRAIN_F": mp_namelist.IRAIN_F, "Z_SLOPE": mp_namelist.Z_SLOPE_LIQ},
        )
        self._p_snow_melt = stencil_factory.from_dims_halo(
            func=p_snow_melt,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                BLINS: mp_namelist.BLINS,
                CONV_FACTOR: CONV_FACTOR,
                CRACS: mp_namelist.CRACS,
                CSACR: mp_namelist.CSACR,
                CSACW: mp_namelist.CSACW,
                D1_ICE: mp_config.D1_ICE,
                D1_VAP: mp_config.D1_VAP,
                DO_3D_ACC_CLIQ: mp_namelist.DO_3D_ACC_CLIQ,
                DO_QA: mp_namelist.DO_QA,
                DT: gfdl_1m_config.DT_MOIST,
                LI00: mp_config.LI00,
                LI20: mp_config.LI20,
                LV00: mp_config.LV00,
                MUS: mp_namelist.MUS,
                QS_MLT: mp_namelist.QS_MLT,
                T_WFR: mp_config.T_WFR,
                VDIFFFLAG: mp_namelist.VDIFFFLAG,
            },
        )

    def __call__(self, gfdl_mp_v3_locals: GFDLMPV3Locals, mp_full_locals: MPFullLocals):
        # --------------------------------------------------
        # calculate heat capacities and latent heat coefficients
        # --------------------------------------------------
        self._calc_mhc_lhc_wrapper(
            t=gfdl_mp_v3_locals.t,
            vapor=gfdl_mp_v3_locals.vapor,
            ice=gfdl_mp_v3_locals.ice,
            liquid=gfdl_mp_v3_locals.liquid,
            graupel=gfdl_mp_v3_locals.graupel,
            rain=gfdl_mp_v3_locals.rain,
            snow=gfdl_mp_v3_locals.snow,
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
