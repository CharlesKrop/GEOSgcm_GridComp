import dataclasses

from ndsl import StencilFactory, Local, LocalState, QuantityFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, exp, interval, log
from ndsl.dsl.typing import Float, FloatField, FloatField64, FloatFieldIJ

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig, GFDLMPV3TableL3xL10, GFDLMPV3TableL5, GFDLMPV3TableL20
from pyMoist.microphysics.GFDL_1M.microphysics.constants import CFMIN, QCMIN, QPMIN, RHOW
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_table_functions import saturation_specific_humidity
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3SaturationTable, GFDLMPV3Tables
from pyMoist.microphysics.GFDL_1M.microphysics.shared import (
    accretion_2d,
    accretion_3d,
    calc_mhc_lhc,
    calc_particle_concentration,
    linear_prof,
    moist_heat_capacity_3,
    p_sub,
    update_hydrometeors,
    update_hydrometeors_and_temperature,
)
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState


def evaporation(
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
    h_var: FloatField,
    revap: FloatField,
    mpper: FloatFieldIJ,
    CREVP: GFDLMPV3TableL5,
    table_0: GFDLMPV3SaturationTable,
    dtable_0: GFDLMPV3SaturationTable,
):
    """rain evaporation to form water vapor, Lin et al. (1983)

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
        h_var (FloatField)
        revap (FloatField)
        mpper (FloatFieldIJ)
        CREVP (GFDLMPV3TableL5)
        table_0 (GFDLMPV3SaturationTable)
        dtable_0 (GFDLMPV3SaturationTable)
    """
    from __externals__ import (
        BLINR,
        C1_ICE,
        C1_LIQ,
        C1_VAP,
        CONV_FACTOR,
        D1_ICE,
        D1_VAP,
        DO_QA,
        DT,
        LI00,
        LI20,
        LV00,
        MUR,
        RHC_REVAP,
        T_WFR,
        TAU_REVP,
        USE_ENHANCED_DRY_EVAP,
        USE_RHC_REVAP,
    )

    with computation(FORWARD), interval(0, 1):
        # time-scale factor
        fac_revp: FloatFieldIJ = 1.0
        if TAU_REVP > 1e-6:
            fac_revp = 1.0 - exp(DT / TAU_REVP)

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

    with computation(PARALLEL), interval(...):
        if t > T_WFR and rain > QPMIN:
            mhc = moist_heat_capacity_3(vapor + liquid, rain, total_solid)
            t_in = (t * cvm - LV00 * liquid) / mhc

            # calculate supersaturation and subgrid variability of water
            precipitation = vapor + liquid
            sat_spec_humidity = saturation_specific_humidity(t_in, density, table_0, dtable_0)
            dvapor = sat_spec_humidity - vapor

            dqh = max(liquid, h_var * max(precipitation, QCMIN))
            dqh = min(dqh, 0.2 * precipitation)
            q_minus = precipitation - dqh
            q_plus = precipitation + dqh

            # rain evaporation

            if dvapor > 0.0 and sat_spec_humidity > q_minus:

                if sat_spec_humidity > q_plus:
                    dcondensate = sat_spec_humidity - precipitation
                else:
                    dcondensate = 0.25 * (sat_spec_humidity - q_minus) ** 2 / dqh
                condensate_x_density = rain * density
                t_squared = t_in * t_in
                sink = p_sub(
                    t_squared,
                    dcondensate,
                    condensate_x_density,
                    sat_spec_humidity,
                    density,
                    density_factor,
                    BLINR,
                    MUR,
                    lcpk,
                    cvm,
                    CREVP,
                )
                sink = min(rain, DT * fac_revp * sink, dvapor / (1.0 + lcpk * sat_spec_humidity))

                # -----------------------------------------------------------------------
                # Enhanced scale-aware rain evaporation in dry environmental air.
                # Target grid-mean RH scales with grid resolution via h_var (rhcrit).
                # At 1km (h_var=0.0): evaporates until grid box is 100% saturated.
                # At 100km (h_var=0.3): evaporates until grid box is 70% saturated.
                # -----------------------------------------------------------------------
                if USE_ENHANCED_DRY_EVAP:
                    # True scale-aware target RH based on subgrid moisture variance
                    rh_rain = max(0.70, 1.0 - h_var)
                    # Calculate total mass NEEDED to hit the target RH threshold (Units: kg/kg)
                    tmp = max((rh_rain * saturation_specific_humidity - vapor), 0.0) / (1.0 + lcpk * sat_spec_humidity)
                    # Apply the dimensionless timescale factor so it doesn't evaporate instantly
                    # (Units: dimensionless * kg/kg = kg/kg)
                    tmp = fac_revp * tmp
                    # Bound by the actual rain available (Units: kg/kg)
                    tmp = min(rain, tmp)
                    # Update the final sink amount (Units: kg/kg)
                    sink = max(sink, tmp)

                # use RH cap for rain evaporation
                if USE_RHC_REVAP:
                    rh_tem = precipitation / saturation_specific_humidity
                    if rh_tem >= RHC_REVAP:
                        sink = 0.0

                tmp = sink * dry_dp * CONV_FACTOR

                mpper = mpper + tmp

                # 3D re-evaporation export
                revap = revap + tmp

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


def accretion(
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
    terminal_fall_liquid: FloatField,
    terminal_fall_rain: FloatField,
    mppxr: FloatFieldIJ,
    ACC: GFDLMPV3TableL20,
    ACCO: GFDLMPV3TableL3xL10,
):
    """rain accretion with cloud water, Lin et al. (1983)

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
        terminal_fall_liquid (FloatField)
        terminal_fall_rain (FloatField)
        mppxr (FloatFieldIJ)
        ACC (GFDLMPV3TableL20)
        ACCO (GFDLMPV3TableL3xL10)
    """
    from __externals__ import BLINR, CONV_FACTOR, CRACW, DO_3D_ACC_CLIQ, DO_QA, DT, MUR, T_WFR, VDIFFFLAG

    with computation(PARALLEL), interval(...):
        if t > T_WFR and rain > QPMIN and liquid > QCMIN:

            condensate_x_density = rain * density
            if DO_3D_ACC_CLIQ:
                sink = DT * accretion_3d(
                    terminal_fall_rain,
                    terminal_fall_liquid,
                    liquid,
                    rain,
                    density,
                    CRACW,
                    ACC.A[8],
                    ACC.A[9],
                    ACCO,
                    VDIFFFLAG,
                )
            else:
                sink = DT * accretion_2d(
                    condensate_x_density,
                    density_factor,
                    CRACW,
                    BLINR,
                    MUR,
                )
                sink = sink / (1.0 + sink) * liquid
            mppxr = mppxr + sink * dry_dp * CONV_FACTOR

            vapor, ice, liquid, graupel, rain, snow, cloud_fraction = update_hydrometeors(
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
            )


def autoconversion_part_1(
    liquid: FloatField,
    liquid_internal: FloatField,
    cloud_fraction: FloatField,
    cloud_fraction_internal: FloatField,
):
    from __externals__ import IN_CLOUD_LIQ

    with computation(PARALLEL), interval(...):
        # Use In-Cloud condensates with scale-aware blending
        if IN_CLOUD_LIQ:
            # Enforce minimum bound to prevent vanishing values
            cloud_fraction_internal = max(cloud_fraction, CFMIN)
        else:
            cloud_fraction_internal = 1.0

        liquid_internal = liquid / cloud_fraction_internal


def autoconversion_part_2(
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
    liquid_internal: FloatField,
    cloud_fraction_internal: FloatField,
    h_var: FloatField,
    ccn: FloatField,
    mppar: FloatFieldIJ,
    cpaut: FloatFieldIJ,
    factor_rc: FloatFieldIJ,
):
    """cloud water to rain autoconversion, Hong et al. (2004)

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
        h_var (FloatField)
        ccn (FloatField)
        mppar (FloatFieldIJ)
        cpaut (FloatFieldIJ)
        factor_rc (FloatFieldIJ)
    """
    from __externals__ import CONV_FACTOR, DO_PSD_WATER_NUM, DO_QA, DT, IRAIN_F, MUW, PCAW, PCBW, QL0_MAX, T_WFR

    with computation(FORWARD), interval(0, 1):
        # internal constants
        so1: FloatFieldIJ = -1.0 / 3.0
        so3: FloatFieldIJ = 7.0 / 3.0

    with computation(PARALLEL), interval(...):
        # initialize internal temporary
        dliquid = 0.0

    with computation(PARALLEL), interval(...):
        if IRAIN_F == 0:

            if t > T_WFR and liquid_internal > QCMIN:

                if DO_PSD_WATER_NUM:
                    ccn = calc_particle_concentration(liquid_internal, density, MUW, PCAW, PCBW)
                    ccn = ccn / density

                condensate = factor_rc * ccn
                dliquid = min(max(QCMIN, dliquid), 0.5 * liquid_internal)
                dcondensate = 0.5 * (liquid_internal + dliquid - condensate)

                if dcondensate > 0.0:

                    c_praut = cpaut * exp(so1 * log(ccn * RHOW))
                    sink = min(1.0, dcondensate / dliquid) * DT * c_praut * density * exp(so3 * log(liquid_internal))
                    sink = min(QL0_MAX / cloud_fraction_internal, min(liquid_internal, sink)) * cloud_fraction_internal
                    mppar = mppar + sink * dry_dp * CONV_FACTOR

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

    with computation(PARALLEL), interval(...):
        if IRAIN_F == 1:
            if t > T_WFR and liquid_internal > QCMIN:

                if DO_PSD_WATER_NUM:
                    ccn = calc_particle_concentration(liquid_internal, density, MUW, PCAW, PCBW)
                    ccn = ccn / density

                condensate = factor_rc * ccn
                dcondensate = liquid_internal - condensate

                if dcondensate > 0.0:
                    c_praut = cpaut * exp(so1 * log(ccn * RHOW))
                    sink = min(dcondensate, DT * c_praut * density * exp(so3 * log(liquid_internal)))
                    sink = min(QL0_MAX / cloud_fraction_internal, liquid_internal, sink) * cloud_fraction_internal
                    mppar = mppar + sink * dry_dp * CONV_FACTOR

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


@dataclasses.dataclass
class WarmRainLocals(LocalState):
    cloud_fraction_internal: Local = dataclasses.field(
        metadata={
            "name": "cloud_fraction_internal",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dliquid_internal: Local = dataclasses.field(
        metadata={
            "name": "dliquid_internal",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    liquid_internal: Local = dataclasses.field(
        metadata={
            "name": "liquid_internal",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


class WarmRain:
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
        # make config and tables visible at runtime
        self._mp_config = mp_config
        self._saturation_tables = saturation_tables

        # initialize locals for warm rain
        self._warm_rain_locals = WarmRainLocals.make_locals(quantity_factory)

        # construct stencils
        self._evaporation = stencil_factory.from_dims_halo(
            func=evaporation,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINR": mp_namelist.BLINR,
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "CONV_FACTOR": CONV_FACTOR,
                "D1_ICE": mp_config.D1_ICE,
                "D1_VAP": mp_config.D1_VAP,
                "DO_QA": mp_namelist.DO_QA,
                "DT": gfdl_1m_config.DT_MOIST,
                "LI00": mp_config.LI00,
                "LI20": mp_config.LI20,
                "LV00": mp_config.LV00,
                "MUR": mp_namelist.MUR,
                "RHC_REVAP": mp_namelist.RHC_REVAP,
                "TAU_REVP": mp_namelist.TAU_REVP,
                "T_WFR": mp_config.T_WFR,
                "USE_ENHANCED_DRY_EVAP": mp_namelist.USE_ENHANCED_DRY_EVAP,
                "USE_RHC_REVAP": mp_namelist.USE_RHC_REVAP,
            },
        )
        self._accretion = stencil_factory.from_dims_halo(
            func=accretion,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BLINR": mp_namelist.BLINR,
                "CONV_FACTOR": CONV_FACTOR,
                "CRACW": mp_config.CRACW,
                "DO_3D_ACC_CLIQ": mp_namelist.DO_3D_ACC_CLIQ,
                "DO_QA": mp_namelist.DO_QA,
                "DT": gfdl_1m_config.DT_MOIST,
                "MUR": mp_namelist.MUR,
                "T_WFR": mp_config.T_WFR,
                "VDIFFFLAG": mp_namelist.VDIFFFLAG,
            },
        )
        self._autoconversion_part_1 = stencil_factory.from_dims_halo(
            func=autoconversion_part_1,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"IN_CLOUD_LIQ": mp_namelist.IN_CLOUD_LIQ},
        )
        self._linear_prof = stencil_factory.from_dims_halo(
            func=linear_prof,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"IRAIN_F": mp_namelist.IRAIN_F, "Z_SLOPE": mp_namelist.Z_SLOPE_LIQ},
        )
        self._autoconversion_part_2 = stencil_factory.from_dims_halo(
            func=autoconversion_part_2,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONV_FACTOR": CONV_FACTOR,
                "DO_PSD_WATER_NUM": mp_namelist.DO_PSD_WATER_NUM,
                "DO_QA": mp_namelist.DO_QA,
                "DT": gfdl_1m_config.DT_MOIST,
                "IN_CLOUD_LIQ": mp_namelist.IN_CLOUD_LIQ,
                "IRAIN_F": mp_namelist.IRAIN_F,
                "MUW": mp_namelist.MUW,
                "PCAW": mp_config.PCAW,
                "PCBW": mp_config.PCBW,
                "QL0_MAX": mp_namelist.QL0_MAX,
                "T_WFR": mp_config.T_WFR,
            },
        )

    def __call__(self, state: GFDL1MState, gfdl_mp_v3_locals: GFDLMPV3Locals, mp_full_locals: MPFullLocals):
        # -----------------------------------------------------------------------
        # rain evaporation to form water vapor
        # -----------------------------------------------------------------------
        self._evaporation(
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
            h_var=gfdl_mp_v3_locals.h_var,
            revap=state.non_anvil_large_scale.evaporation,
            mpper=gfdl_mp_v3_locals.mpper,
            # tables
            CREVP=self._mp_config.CREVP,
            table_0=self._saturation_tables.table_0,
            dtable_0=self._saturation_tables.dtable_0,
        )

        # -----------------------------------------------------------------------
        # rain accretion with cloud water
        # -----------------------------------------------------------------------
        self._accretion(
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
            terminal_fall_liquid=mp_full_locals.terminal_velocity.liquid,
            terminal_fall_rain=mp_full_locals.terminal_velocity.rain,
            mppxr=gfdl_mp_v3_locals.mppxr,
            # tables
            ACC=self._mp_config.ACC,
            ACCO=self._mp_config.ACCO,
        )

        # -----------------------------------------------------------------------
        # cloud water to rain autoconversion
        # -----------------------------------------------------------------------
        self._autoconversion_part_1(
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            liquid_internal=self._warm_rain_locals.liquid_internal,
            cloud_fraction=gfdl_mp_v3_locals.cloud_fraction,
            cloud_fraction_internal=self._warm_rain_locals.cloud_fraction_internal,
        )

        self._linear_prof(
            precipitate=self._warm_rain_locals.liquid_internal,
            dm=self._warm_rain_locals.dliquid_internal,
            h_var=gfdl_mp_v3_locals.h_var,
        )

        self._autoconversion_part_2(
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
            h_var=gfdl_mp_v3_locals.h_var,
            ccn=gfdl_mp_v3_locals.ccn,
            mppar=gfdl_mp_v3_locals.mppar,
            cpaut=gfdl_mp_v3_locals.cpaut,
            factor_rc=gfdl_mp_v3_locals.factor_rc,
        )
