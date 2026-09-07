import dataclasses

from ndsl import NDSLRuntime, StencilFactory, QuantityFactory
from ndsl import Local, LocalState
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Float64, FloatField64, FloatFieldIJ64
from ndsl.dsl.gt4py import computation, interval, PARALLEL, sqrt, FORWARD
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import copy_2d
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3NamelistConfig, GFDLMPV3CloudMPConfig
from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.shared.atmos_recipes import sigma, compute_estimated_inversion_strength_factor
from pyMoist.microphysics.GFDL_1M.microphysics.shared import moist_total_energy, mhc3, mhc4
from pyMoist.microphysics.GFDL_1M.microphysics.constants import RC, ZVIR, GRAV, RGRAV, ONE_R8, RDGAS


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
    from __externals__ import CPAUT0, RTHRESHU, RTHRESHS

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

    from __externals__ import CONSV_TE, HYDROSTATIC, C_AIR, C1_VAP, C1_LIQ, C1_ICE

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
    from __externals__ import DT, HYDROSTATIC, LV00, LI00, C_AIR

    # initialize 64 bit internal fields
    with computation(PARALLEL), interval(...):
        cvm: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        total_liquid = cloud_liquid + precip_rain
        total_solid = cloud_ice + cloud_snow + cloud_graupel
        total_condensate = total_liquid + total_solid
        con_r8 = ONE_R8 - (cloud_vapor + total_condensate)
        if moist_q:
            cvm = mhc4(con_r8, cloud_vapor, total_liquid, total_solid)
        else:
            cvm = mhc3(cloud_vapor, total_liquid, total_solid)

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
    from __externals__ import PROG_CCN, PROG_CIN, CCN_L, CCN_O

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


@dataclasses.dataclass
class GFDLMPV3Locals(LocalState):
    ccn: Local = dataclasses.field(
        metadata={
            "name": "ccn",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    cin: Local = dataclasses.field(
        metadata={
            "name": "cin",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    convection_fraction: Local = dataclasses.field(
        metadata={
            "name": "convection_fraction",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    cpaut: Local = dataclasses.field(
        metadata={
            "name": "cpaut",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dtotal_energy: Local = dataclasses.field(
        metadata={
            "name": "dtotal_energy",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    density: Local = dataclasses.field(
        metadata={
            "name": "density",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    density_factor: Local = dataclasses.field(
        metadata={
            "name": "density_factor",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dry_dp: Local = dataclasses.field(
        metadata={
            "name": "dry_dp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dz: Local = dataclasses.field(
        metadata={
            "name": "dz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    factor_eis: Local = dataclasses.field(
        metadata={
            "name": "factor_eis",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    factor_rc: Local = dataclasses.field(
        metadata={
            "name": "factor_rc",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    h_var: Local = dataclasses.field(
        metadata={
            "name": "h_var",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppag: Local = dataclasses.field(
        metadata={
            "name": "mppag",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppar: Local = dataclasses.field(
        metadata={
            "name": "mppar",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppas: Local = dataclasses.field(
        metadata={
            "name": "mppas",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppcw: Local = dataclasses.field(
        metadata={
            "name": "mppcw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppd1: Local = dataclasses.field(
        metadata={
            "name": "mppd1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppdg: Local = dataclasses.field(
        metadata={
            "name": "mppdg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppdi: Local = dataclasses.field(
        metadata={
            "name": "mppdi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppds: Local = dataclasses.field(
        metadata={
            "name": "mppds",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppe1: Local = dataclasses.field(
        metadata={
            "name": "mppe1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpper: Local = dataclasses.field(
        metadata={
            "name": "mpper",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppew: Local = dataclasses.field(
        metadata={
            "name": "mppew",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppfr: Local = dataclasses.field(
        metadata={
            "name": "mppfr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppfw: Local = dataclasses.field(
        metadata={
            "name": "mppfw",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm1: Local = dataclasses.field(
        metadata={
            "name": "mppm1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm2: Local = dataclasses.field(
        metadata={
            "name": "mppm2",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppm3: Local = dataclasses.field(
        metadata={
            "name": "mppm3",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppmg: Local = dataclasses.field(
        metadata={
            "name": "mppmg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppmi: Local = dataclasses.field(
        metadata={
            "name": "mppmi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppms: Local = dataclasses.field(
        metadata={
            "name": "mppms",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpprg: Local = dataclasses.field(
        metadata={
            "name": "mpprg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpprs: Local = dataclasses.field(
        metadata={
            "name": "mpprs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mpps1: Local = dataclasses.field(
        metadata={
            "name": "mpps1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppsg: Local = dataclasses.field(
        metadata={
            "name": "mppsg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppsi: Local = dataclasses.field(
        metadata={
            "name": "mppsi",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppss: Local = dataclasses.field(
        metadata={
            "name": "mppss",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxg: Local = dataclasses.field(
        metadata={
            "name": "mppxg",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxr: Local = dataclasses.field(
        metadata={
            "name": "mppxr",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    mppxs: Local = dataclasses.field(
        metadata={
            "name": "mppxs",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    one_minus_sigma: Local = dataclasses.field(
        metadata={
            "name": "one_minus_sigma",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    p_thickness: Local = dataclasses.field(
        metadata={
            "name": "p_thickness",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    surface_type: Local = dataclasses.field(
        metadata={
            "name": "surface_type",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    reflectivity: Local = dataclasses.field(
        metadata={
            "name": "reflectivity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "dBZ",
            "dtype": Float,
        }
    )
    t: Local = dataclasses.field(
        metadata={
            "name": "t",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy: Local = dataclasses.field(
        metadata={
            "name": "total_energy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_b_beg_d: Local = dataclasses.field(
        metadata={
            "name": "total_energy_b_beg_d",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_b_beg_m: Local = dataclasses.field(
        metadata={
            "name": "total_energy_b_beg_m",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_b_end_d: Local = dataclasses.field(
        metadata={
            "name": "total_energy_b_end_d",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_b_end_m: Local = dataclasses.field(
        metadata={
            "name": "total_energy_b_end_m",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_beg_d: Local = dataclasses.field(
        metadata={
            "name": "total_energy_beg_d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_beg_m: Local = dataclasses.field(
        metadata={
            "name": "total_energy_beg_m",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_end_d: Local = dataclasses.field(
        metadata={
            "name": "total_energy_end_d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_end_m: Local = dataclasses.field(
        metadata={
            "name": "total_energy_end_m",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_energy_loss: Local = dataclasses.field(
        metadata={
            "name": "total_energy_loss",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_b_beg_d: Local = dataclasses.field(
        metadata={
            "name": "total_water_b_beg_d",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_b_beg_m: Local = dataclasses.field(
        metadata={
            "name": "total_water_b_beg_m",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_b_end_d: Local = dataclasses.field(
        metadata={
            "name": "total_water_b_end_d",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_b_end_m: Local = dataclasses.field(
        metadata={
            "name": "total_water_b_end_m",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_beg_d: Local = dataclasses.field(
        metadata={
            "name": "total_water_beg_d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_beg_m: Local = dataclasses.field(
        metadata={
            "name": "total_water_beg_m",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_end_d: Local = dataclasses.field(
        metadata={
            "name": "total_water_end_d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    total_water_end_m: Local = dataclasses.field(
        metadata={
            "name": "total_water_end_m",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float64,
        }
    )
    tracer_dilution_adjustment: Local = dataclasses.field(
        metadata={
            "name": "tracer_dilution_adjustment",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    u: Local = dataclasses.field(
        metadata={
            "name": "u",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    v: Local = dataclasses.field(
        metadata={
            "name": "v",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    w: Local = dataclasses.field(
        metadata={
            "name": "w",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )


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
        conv_to_mm_per_day = 86400.0 * RGRAV / gfdl_1m_config.DT_MOIST

        # initialize class specific locals
        self._driver_locals = GFDLMPV3Locals.make_locals(quantity_factory)

        # make config visible at runtime
        self._mp_namelist = mp_namelist

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

        # dummy fields that are used as placeholders for optional inputs to stencils that are not called
        # they exist only as a thing to pass to stencils, since all inputs must always be supplied,
        # regardless of whether the associated option is enabled, and Float(0.0) cannot be supplied as an input
        self._dummy_field_no_read_no_write_2d_64_bit = quantity_factory.zeros([I_DIM, J_DIM], units="NOREADWRITE", dtype=Float64)
        self._all_zeros_no_write_3d = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], units="NOWRITE", dtype=Float)

    def __call__(self, state: GFDL1MState, locals: GFDL1MLocals):
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
        self._set_value_64_bit(field=self._driver_locals.dtotal_energy, value=Float64(0.0))
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
            t_local=self._driver_locals.t_local,
            vapor=state.radiation_field.vapor,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
            graupel=state.radiation_field.graupel,
            rain=state.radiation_field.rain,
            snow=state.radiation_field.snow,
        )

        # calculate base total energy
        self._compute_total_energy(
            total_energy=self._driver_locals.total_energy,
            t_local=self._driver_locals.t,
            dp=locals.dp,
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
                dp=locals.dp,
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
            local_vapor=self._driver_locals.local_vapor,
            local_ice=self._driver_locals.local_ice,
            local_liquid=self._driver_locals.local_liquid,
            local_graupel=self._driver_locals.local_graupel,
            local_rain=self._driver_locals.local_rain,
            local_snow=self._driver_locals.local_snow,
            local_cloud_fraction=self._driver_locals.local_cloud_fraction,
            local_t=self._driver_locals.local_t,
            dp=locals.dp,
            local_dp=self._driver_locals.local_dp,
            local_dry_dp=self._driver_locals.local_dry_dp,
            dz=locals.layer_thickness_negative,
            local_dz=self._driver_locals.local_dz,
            local_density=self._driver_locals.local_density,
            local_density_factor=self._driver_locals.local_density_factor,
            local_p_thickness=self._driver_locals.local_p_thickness,
            u=state.u,
            local_u=self._driver_locals.local_u,
            v=state.v,
            local_v=self._driver_locals.local_v,
            w=state.w,
            local_w=self._driver_locals.local_w,
        )

        # total_energy_checker
        if self._mp_namelist.CONSV_CHECKER:
            self._total_energy_and_water(
                t_local=self._driver_locals.t_local,
                total_energy=self._driver_locals.total_energy_beg_d,
                total_water=self._driver_locals.total_water_beg_d,
                total_energy_b=self._driver_locals.total_energy_b_beg_d,
                total_water_b=self._driver_locals.total_water_b_beg_d,
                u=self._driver_locals.u,
                v=self._driver_locals.v,
                w=self._driver_locals.w,
                dp=self._driver_locals.dry_dp,
                cloud_vapor=self._driver_locals.vapor,
                cloud_ice=self._driver_locals.ice,
                cloud_liquid=self._driver_locals.liquid,
                cloud_rain=self._driver_locals.rain,
                cloud_snow=self._driver_locals.snow,
                cloud_graupel=self._driver_locals.graupel,
                vapor=self._all_zeros_no_write_3d,  # NOTE this may break, since the same field is being passed multiple times
                precip_ice=state.precipitation_at_surface.ice,
                precip_liquid=state.precipitation_at_surface.water,
                precip_rain=state.precipitation_at_surface.rain,
                precip_snow=state.precipitation_at_surface.snow,
                precip_graupel=state.precipitation_at_surface.graupel,
                dtotal_energy=self._driver_locals.dtotal_energy,
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
        # if self._mp_namelist.FIX_NEGATIVE:
