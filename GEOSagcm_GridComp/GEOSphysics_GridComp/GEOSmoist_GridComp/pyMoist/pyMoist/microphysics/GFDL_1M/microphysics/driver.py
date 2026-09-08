from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, interval, sqrt
from ndsl.dsl.typing import Bool, Float, Float64, FloatField, FloatField64, FloatFieldIJ, FloatFieldIJ64
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import copy_2d

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.constants import GRAV, ONE_R8, RC, RDGAS, RGRAV, ZVIR
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.shared import (
    calc_mhc_lhc,
    moist_heat_capacity_3,
    moist_heat_capacity_4,
    moist_total_energy,
    update_hydrometeors,
    update_hydrometeors_and_temperature,
)
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.shared.atmos_recipes import compute_estimated_inversion_strength_factor, sigma


def set_value_64_bit(field: FloatField64, value: Float64) -> None:
    """
    Sets every element of a field to a single value.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(PARALLEL), interval(...):
        field = value


def compute_one_minus_sigma(one_minus_sigma: FloatFieldIJ, area: FloatFieldIJ):
    from __externals__ import DO_SCALE_DEP

    with computation(FORWARD), interval(0, 1):
        if DO_SCALE_DEP:
            one_minus_sigma = sigma(sqrt(area))
        else:
            one_minus_sigma = 1.0


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


def convert_temperature(
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


def compute_total_energy(
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


def total_energy_and_water(
    t_local: FloatField64,
    total_energy: FloatField64,
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
    dtotal_energy: FloatFieldIJ64,
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
    local_dp: FloatField,
    local_dry_dp: FloatField,
    dz: FloatField,
    local_dz: FloatField,
    local_density: FloatField,
    local_density_factor: FloatField,
    local_p_thickness: FloatField,
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
        con_r8 = Float64(0.0)

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
        local_dp = dp

        # convert total pressure thickness (dp) to dry air pressure thickness (dry_dp)
        local_dry_dp = local_dp * con_r8

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
        local_density = -local_dp / (GRAV * local_dz)
        local_p_thickness = local_density * RDGAS * local_t

        # for sedi_momentum transport

        local_u = u
        local_v = v
        if not HYDROSTATIC:
            local_w = w


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


class GFDLMPV3Driver(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        gfdl_1m_config: GFDL1MConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        mp_config: GFDLMPV3CloudMPConfig,
    ):
        # initialize NDSLRuntime parent class
        super.__init__(stencil_factory)

        # compute driver specific constants
        # timesteps
        driver_dt = gfdl_1m_config.DT_MOIST / mp_namelist.NTIMES
        dt_inverse = 1 / gfdl_1m_config.DT_MOIST
        # conversion factor to mm/day
        CONV_FACTOR = 86400.0 * RGRAV / gfdl_1m_config.DT_MOIST

        # initialize class specific locals
        self._driver_locals = GFDLMPV3Locals.make_locals(quantity_factory)

        # make config visible at runtime
        self._mp_namelist = mp_namelist
        self._mp_config = mp_config

        # construct stencils
        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._set_value_64_bit = stencil_factory.from_dims_halo(
            func=set_value_64_bit,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._copy_2d = stencil_factory.from_dims_halo(
            func=copy_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._compute_one_minus_sigma = stencil_factory.from_dims_halo(
            func=compute_one_minus_sigma,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DO_SCALE_DEP": mp_namelist.DO_SCALE_DEP},
        )
        self._eis_factor_and_rates = stencil_factory.from_dims_halo(
            func=eis_factor_and_rates,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CPAUT0": mp_config.CPAUT0},
        )
        self._convert_temperature = stencil_factory.from_dims_halo(
            func=convert_temperature,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DO_INLINE_MP": mp_config.DO_INLINE_MP},
        )
        self._compute_total_energy = stencil_factory.from_dims_halo(
            func=compute_total_energy,
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
        self._pressure_derived_fields_mixing_ratio_conversion_copy_state = stencil_factory.from_dims_halo(
            func=pressure_derived_fields_mixing_ratio_conversion_copy_state,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DO_INLINE_MP": mp_config.DO_INLINE_MP,
                "HYDROSTATIC": gfdl_1m_config.LHYDROSTATIC,
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

        # dummy fields that are used as placeholders for optional inputs to stencils that are not called
        # they exist only as a thing to pass to stencils, since all inputs must always be supplied,
        # regardless of whether the associated option is enabled, and Float(0.0) cannot be supplied as an input
        self._dummy_field_no_read_no_write_2d_64_bit = quantity_factory.zeros([I_DIM, J_DIM], units="NOREADWRITE", dtype=Float64)
        self._all_zeros_no_write_3d = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], units="NOWRITE", dtype=Float)

    def __call__(self, state: GFDL1MState, gfdl_1m_locals: GFDL1MLocals):
        # reset mp locals to zero
        self._set_value(field=self._driver_locals.mppcw, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppew, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppe1, value=Float(0.0))
        self._set_value(field=self._driver_locals.mpper, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppdi, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppd1, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppds, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppdg, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppsi, value=Float(0.0))
        self._set_value(field=self._driver_locals.mpps1, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppss, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppsg, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppfw, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppfr, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppar, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppas, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppag, value=Float(0.0))
        self._set_value(field=self._driver_locals.mpprs, value=Float(0.0))
        self._set_value(field=self._driver_locals.mpprg, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppxr, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppxs, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppxg, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppmi, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppms, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppmg, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppm1, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppm2, value=Float(0.0))
        self._set_value(field=self._driver_locals.mppm3, value=Float(0.0))

        # initialization of total energy difference
        self._set_value_64_bit(field=self._driver_locals.total_energy.delta, value=Float64(0.0))
        self._set_value(field=self._driver_locals.tracer_dilution_adjustment, value=Float(1.0))

        # copy convection fraction and surface type, work with copy instead of the original
        self._copy_2d(input=state.convection_fraction, output=self._driver_locals.convection_fraction)
        self._copy_2d(input=state.surface_type, output=self._driver_locals.surface_type)

        # one minus sigma used to control resoluton sensitive parameters
        self._compute_one_minus_sigma(one_minus_sigma=self._driver_locals.one_minus_sigma, area=state.area)

        # Use estimated inversion strength to determine stable vs unstable areas
        self._eis_factor_and_rates(
            estimated_inversion_strength=state.estimated_inversion_strength,
            convection_fraction=self._driver_locals.convection_fraction,
            factor_eis=self._driver_locals.factor_eis,
            factor_rc=self._driver_locals.factor_rc,
            cpaut=self._driver_locals.cpaut,
        )

        # conversion of temperature
        self._convert_temperature(
            t_state=state.t,
            t_local=self._driver_locals.t,
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
        )

        # calculate base total energy
        self._compute_total_energy(
            total_energy=self._driver_locals.total_energy.magnitude,
            t_local=self._driver_locals.t,
            dp=gfdl_1m_locals.dp,
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            snow=state.radiation_field.snow,
            rain=state.radiation_field.rain,
        )

        # total_energy_checker
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._driver_locals.t_local,
                total_energy=self._driver_locals.total_energy_beg_m,
                total_water=self._driver_locals.total_water_beg_m,
                total_energy_b=self._driver_locals.total_energy_b_beg_m,
                total_water_b=self._driver_locals.total_water_b_beg_m,
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
                dtotal_energy=self._driver_locals.dtotal_energy,
                sen=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                stress=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                moist_q=True,
                save_te_loss=False,
                total_energy_loss=self._dummy_field_no_read_no_write_2d_64_bit,
            )

        # initialize radar reflectivity
        self._set_value(field=self._driver_locals.reflectivity, value=Float(-30.0))

        # setup the local state - to be used throughout the rest of microphyscis
        self._pressure_derived_fields_mixing_ratio_conversion_copy_state(
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
            cloud_fraction=state.radiation_field.cloud_fraction,
            local_vapor=self._driver_locals.mixing_ratio.vapor,
            local_ice=self._driver_locals.mixing_ratio.ice,
            local_liquid=self._driver_locals.mixing_ratio.liquid,
            local_graupel=self._driver_locals.mixing_ratio.graupel,
            local_rain=self._driver_locals.mixing_ratio.rain,
            local_snow=self._driver_locals.mixing_ratio.snow,
            local_cloud_fraction=self._driver_locals.cloud_fraction,
            local_t=self._driver_locals.t,
            dp=gfdl_1m_locals.dp,
            local_dp=self._driver_locals.dp,
            local_dry_dp=self._driver_locals.dry_dp,
            dz=gfdl_1m_locals.layer_thickness_negative,
            local_dz=self._driver_locals.dz,
            local_density=self._driver_locals.density,
            local_density_factor=self._driver_locals.density_factor,
            local_p_thickness=self._driver_locals.p_thickness,
            u=state.u,
            local_u=self._driver_locals.u,
            v=state.v,
            local_v=self._driver_locals.v,
            w=state.w,
            local_w=self._driver_locals.w,
        )

        # total_energy_checker
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._driver_locals.t,
                total_energy=self._driver_locals.total_energy.beg_d,
                total_water=self._driver_locals.total_water.beg_d,
                total_energy_b=self._driver_locals.total_energy.b_beg_d,
                total_water_b=self._driver_locals.total_water.b_beg_d,
                u=self._driver_locals.u,
                v=self._driver_locals.v,
                w=self._driver_locals.w,
                dp=self._driver_locals.dry_dp,
                cloud_vapor=self._driver_locals.mixing_ratio.vapor,
                cloud_ice=self._driver_locals.mixing_ratio.ice,
                cloud_liquid=self._driver_locals.mixing_ratio.liquid,
                cloud_rain=self._driver_locals.mixing_ratio.rain,
                cloud_snow=self._driver_locals.mixing_ratio.snow,
                cloud_graupel=self._driver_locals.mixing_ratio.graupel,
                vapor=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                precip_ice=state.precipitation_at_surface.ice,
                precip_liquid=state.precipitation_at_surface.water,
                precip_rain=state.precipitation_at_surface.rain,
                precip_snow=state.precipitation_at_surface.snow,
                precip_graupel=state.precipitation_at_surface.graupel,
                dtotal_energy=self._driver_locals.total_energy.delta,
                sen=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                stress=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                moist_q=False,
                save_te_loss=False,
                total_energy_loss=self._dummy_field_no_read_no_write_2d_64_bit,
            )

        # generate cloud condensation nuclei (CCN), cloud ice nuclei (CIN)
        self._generate_particle_nuclei(
            ccn=self._driver_locals.ccn,
            cin=self._driver_locals.cin,
            concentration_liquid=state.concentration.liquid,
            concentration_ice=state.concentration.ice,
            density=self._driver_locals.density,
            surface_geopotential_height=state.surface_geopotential_height,
        )

        self._horizontal_subgrid_variation(h_var=self._driver_locals.h_var, critical_relative_humidity_for_pdf=state.critical_relative_humidity_for_pdf)

        # fix negative water species from outside
        if self._mp_namelist.FIX_NEGATIVE:
            self._fix_negative_water_species(
                t=self._driver_locals.t,
                dry_dp=self._driver_locals.dry_dp,
                vapor=self._driver_locals.mixing_ratio.vapor,
                ice=self._driver_locals.mixing_ratio.ice,
                liquid=self._driver_locals.mixing_ratio.liquid,
                graupel=self._driver_locals.mixing_ratio.graupel,
                rain=self._driver_locals.mixing_ratio.rain,
                snow=self._driver_locals.mixing_ratio.snow,
                cloud_fraction=self._driver_locals.cloud_fraction,
                mppcw=self._driver_locals.mppcw,
                mppfr=self._driver_locals.mppfr,
            )

        # fast microphysics loop
        if self._mp_config.DO_MP_FAST:
            ndsl_log.error("[GFDL1M Microphysics]: NDSL version of DO_MP_FAST option has not been implemented, please use DO_MP_FULL instead")
            raise ValueError("[GFDL1M Microphysics]: NDSL version of DO_MP_FAST option has not been implemented, please use DO_MP_FULL instead")

        # full microphysics loop
        if self._mp_config.DO_FULL_MP:
