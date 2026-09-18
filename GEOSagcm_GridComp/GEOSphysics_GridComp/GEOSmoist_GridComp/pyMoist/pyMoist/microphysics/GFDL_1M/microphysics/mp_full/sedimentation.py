import dataclasses

from ndsl import Local, LocalState, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, exp, function, interval, log, log10, max
from ndsl.dsl.typing import BoolFieldIJ, Float, Float64, FloatField, FloatField64, FloatFieldIJ, FloatFieldIJ64, Int, Bool
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import set_value_2d

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.constants import C_ICE, C_LIQ, CV_AIR, CV_VAP, DZ_MIN, GRAV, QFMIN, RDGAS, TICE
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.shared import calc_mass_weighted_terminal_velocity, calc_mhc_lhc, moist_total_energy
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.shared.cloud_processes import cloud_effective_radius_ice


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


def terminal_velocity_ice(
    t: FloatField64,
    ice: FloatField,
    density: FloatField,
    terminal_velocity_ice: FloatField,
    convection_fraction: FloatFieldIJ,
):
    from __externals__ import CONST_VI, DO_ICE_PRES_SCALING, IFFLAG, VI_FAC, VI_MAX, VI_MIN, aaC, aaL, bbC, bbL, ccC, ccL, ddC, ddL, eeC, eeL

    with computation(PARALLEL), interval(...):
        if CONST_VI:
            terminal_velocity_ice = 0.5 * (VI_MIN + VI_MAX)
        else:
            t_internal = t - TICE

            # 1. Calculate Base Fall Speeds based on chosen formulation
            if IFFLAG == 1:
                # Pure Deng and Mace (2008)
                ice_density = ice * density * 1.0e3
                viLSC = 10.0 ** (log10(ice_density) * (t_internal * (aaL * t_internal + bbL) + ccL) + ddL * t_internal + eeL)
                viCNV = 10.0 ** (log10(ice_density) * (t_internal * (aaC * t_internal + bbC) + ccC) + ddC * t_internal + eeC)
                terminal_velocity_ice = 0.01 * (viLSC * (1.0 - convection_fraction) + viCNV * (convection_fraction))

            elif IFFLAG == 2:
                # Pure Heymsfield and Donner (1990)
                ice_density = ice * density
                terminal_velocity_ice = 3.29 * exp(0.16 * log(ice_density))

            elif IFFLAG == 3:
                # Pure Mishra et al (2014, JGR)
                ice_density = ice * density * 1.0e3
                # Synoptic Vm: a=1.411, b=11.71, c=82.35
                viLSC = max(10.0, (1.411 * t_internal + 11.71 * log10(ice_density * 1.0e3) + 82.35))
                # Anvil Vm: a=1.119, b=14.21, c=68.85
                viCNV = max(10.0, (1.119 * t_internal + 14.21 * log10(ice_density * 1.0e3) + 68.85))
                terminal_velocity_ice = 0.01 * (viLSC * (1.0 - convection_fraction) + viCNV * (convection_fraction))

            elif IFFLAG == 4:
                # Combination: Deng & Mace (2008) LSC + Mishra et al (2014) Anvil CNV
                ice_density = ice * density * 1.0e3
                viLSC = 10.0 ** (log10(ice_density) * (t_internal * (aaL * t_internal + bbL) + ccL) + ddL * t_internal + eeL)
                # Anvil Vm: a=1.119, b=14.21, c=68.85
                viCNV = max(10.0, (1.119 * t_internal + 14.21 * log10(ice_density * 1.0e3) + 68.85))
                terminal_velocity_ice = 0.01 * (viLSC * (1.0 - convection_fraction) + viCNV * (convection_fraction))

            # 2. Apply Universal Pressure Scaling (Accelerates high-alt ice)
            if DO_ICE_PRES_SCALING:
                p_dry = density * RDGAS * t  # dry air pressure
                diam = 2.0 * cloud_effective_radius_ice(p_dry / 100.0, t, ice) * 1.0e6  # microns
                ln_p = log(p_dry / 100.0)
                c0 = -1.04 + 0.298 * ln_p
                c1 = 0.67 - 0.097 * ln_p
                # Apply pressure scaling multiplier
                terminal_velocity_ice = terminal_velocity_ice * (c0 + c1 * log(diam))

            # 3. Apply user multiplier and safety caps
            terminal_velocity_ice = VI_FAC * terminal_velocity_ice
            terminal_velocity_ice = min(VI_MAX, max(VI_MIN, terminal_velocity_ice))


def terminal_velocity_graupel_rain_snow(
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
    with computation(PARALLEL), interval(...):
        if const_v:
            terminal_velocity = 0.5 * (v_min + v_max)
        else:
            if condensate < QFMIN:
                terminal_velocity = calc_mass_weighted_terminal_velocity(condensate, density, mu, tva, tvb, blin)
                terminal_velocity = v_fac * terminal_velocity * density_factor
                terminal_velocity = min(v_max, max(v_min, terminal_velocity))


def set_heights(
    ze: FloatField,
    zs: FloatFieldIJ,
    zt: FloatField,
    dz: FloatField,
    terminal_velocity: FloatField,
):
    from __externals__ import DT

    with computation(FORWARD), interval(1, None):
        half_dt: FloatFieldIJ = 0.5 * DT
        zs = 0.0
        ze[0, 0, 1] = zs
    with computation(BACKWARD), interval(...):
        ze = ze[0, 0, 1] - dz

    with computation(FORWARD), interval(0, 1):
        zt = ze

    with computation(FORWARD), interval(1, None):
        zt = ze - half_dt * (terminal_velocity[0, 0, -1] + terminal_velocity)

    with computation(FORWARD), interval(1, None):
        zt[0, 0, 1] = zs - DT * terminal_velocity

    with computation(FORWARD), interval(...):
        if zt[0, 0, 1] >= zt:
            zt[0, 0, 1] = zt - DZ_MIN


@function
def implicit_fall(
    ze: FloatField,
    terminal_velocity: FloatField,
    dry_dp: FloatField,
    condensate: FloatField,
    precip: FloatField,
    precip_at_surface: FloatFieldIJ,
    DT: Float,
    k_end: Int,
    # temporaries that must be provided by the stencil
    dd: FloatField,
    dz: FloatField,
    condensate_modified: FloatField,
):
    """Compute precipitation throughout the colmun and at the surface.

    This function must be called with a interval(0, 1) statement for the indexing to work properly

    Args:
        ze (FloatField)
        terminal_velocity (FloatField)
        dry_dp (FloatField)
        condensate (FloatField)
        precip_at_surface (FloatFieldIJ)
        precip (FloatField)
        DT (Float)
        k_end (Int)
        dd (FloatField)
        dz (FloatField)
        condensate_modified (FloatField)
    """

    level = 0
    while level <= k_end:
        dz[0, 0, level] = ze[0, 0, level] - ze[0, 0, level + 1]
        dd = DT * terminal_velocity
        condensate[0, 0, level] = condensate[0, 0, level] * dry_dp
        level += 1

    condensate_modified = condensate / (dz + dd)
    level = 1
    while level <= k_end:
        condensate_modified[0, 0, level] = (condensate[0, 0, level] + condensate_modified[0, 0, level - 1] * dd[0, 0, level - 1]) / (dz[0, 0, level] + dd[0, 0, level])
        level += 1

    level = 0
    while level <= k_end:
        condensate_modified[0, 0, level] = condensate_modified[0, 0, level] * dz[0, 0, level]
        level += 1

    precip = condensate - condensate_modified
    level = 1
    while level < k_end:
        precip[0, 0, level] = precip[0, 0, level - 1] + condensate[0, 0, level] - condensate_modified[0, 0, level]
        level += 1
    precip = precip + precip[0, 0, k_end]

    level = 0
    while level <= k_end:
        condensate[0, 0, level] = condensate_modified[0, 0, level] / dry_dp[0, 0, level]
        level += 1

    return condensate, precip, precip_at_surface


def terminal_fall(
    t: FloatField64,
    ze: FloatField,
    zs: FloatFieldIJ,
    zt: FloatField,
    dz: FloatField,
    dry_dp: FloatField,
    vapor: FloatField,
    ice: FloatField,
    liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    terminal_velocity: FloatField,
    precip_at_surface: FloatFieldIJ,
    precip: FloatField,
    dtotal_energy: FloatFieldIJ64,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    mode: Int,
):
    from __externals__ import C1_ICE, C1_LIQ, C1_VAP, C_AIR, DO_SEDI_HEAT, DO_SEDI_UV, DO_SEDI_W, DT, SEDFLAG, k_end

    with computation(FORWARD), interval(0, 1):
        # initialize 2d internals
        precip_fall: BoolFieldIJ = False
        total_energy_1: FloatField64 = 0.0
        total_energy_2: FloatField64 = 0.0
        sum_total_energy_1: FloatField64 = 0.0
        sum_total_energy_2: FloatField64 = 0.0

    with computation(PARALLEL), interval(...):
        # initialize 3d internals
        internal_dd = 0.0
        internal_dz = 0.0
        internal_condensate_modified = 0.0

    with computation(PARALLEL), interval(...):
        # reset 3D precip
        precip = 0.0

    with computation(PARALLEL), interval(...):
        internal_condensate = 0.0
        # determine which precipitate is being considered in this call
        if mode == 1:
            internal_condensate = ice
        if mode == 2:
            internal_condensate = liquid
        if mode == 3:
            internal_condensate = graupel
        if mode == 4:
            internal_condensate = rain
        if mode == 5:
            internal_condensate = snow

    with computation(PARALLEL), interval(...):
        # check if precip will fall anywhere in the column
        # check_column function in original Fortran - inlined here as it was a three line function used only once
        if internal_condensate > QFMIN:
            precip_fall = True

    # all subsequent calculations should only occur if precip_fall is true for a particular column (should have "if precip_fall" guard)

    with computation(PARALLEL), interval(...):
        if precip_fall:
            if DO_SEDI_W:
                # momentum transportation during sedimentation
                dm = dry_dp * (1.0 + vapor + ice + liquid + graupel + rain + snow)

            # energy change during sedimentation
            total_energy_1 = moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dry_dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, False) * GRAV

    # sedimentation

    with computation(PARALLEL), interval(...):
        if precip_fall:
            internal_condensate = 0.0
            # determine which precipitate is being considered in this call
            if mode == 1:
                internal_condensate = ice
            if mode == 2:
                internal_condensate = liquid
            if mode == 3:
                internal_condensate = graupel
            if mode == 4:
                internal_condensate = rain
            if mode == 5:
                internal_condensate = snow

    with computation(FORWARD), interval(0, 1):
        # NOTE once it is possible to call stencils from stencils this section (primarially functions called)
        # can be rewritten to interval(...) instead of manual k loops
        if precip_fall:
            if SEDFLAG == 1:
                implicit_fall(
                    ze,
                    terminal_velocity,
                    dry_dp,
                    internal_condensate,
                    precip,
                    precip_at_surface,
                    DT,
                    k_end,
                    internal_dd,
                    internal_dz,
                    internal_condensate_modified,
                )
            if SEDFLAG == 2:
                option_not_implemented = True
            if SEDFLAG == 3:
                option_not_implemented = True
            if SEDFLAG == 4:
                option_not_implemented = True

    with computation(PARALLEL), interval(...):
        if precip_fall:
            internal_condensate = 0.0
            # determine which precipitate is being considered in this call
            if mode == 1:
                internal_condensate = ice
            if mode == 2:
                internal_condensate = liquid
            if mode == 3:
                internal_condensate = graupel
            if mode == 4:
                internal_condensate = rain
            if mode == 5:
                internal_condensate = snow

    with computation(PARALLEL), interval(...):
        if precip_fall:
            # energy change during sedimentation
            total_energy_2 = moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dry_dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, False) * GRAV

    with computation(FORWARD), interval(...):
        if precip_fall:
            # energy change during sedimentation
            sum_total_energy_1 = sum_total_energy_1 + total_energy_1
            sum_total_energy_2 = sum_total_energy_2 + total_energy_2

    with computation(FORWARD), interval(0, 1):
        if precip_fall:
            # energy change during sedimentation
            dtotal_energy = dtotal_energy + sum_total_energy_1 - sum_total_energy_2

    with computation(FORWARD), interval(0, 1):
        # momentum transportation during sedimentation
        if precip_fall and DO_SEDI_W:
            w = w + precip * terminal_velocity / dm

    with computation(FORWARD), interval(1, None):
        # momentum transportation during sedimentation
        if precip_fall:
            if DO_SEDI_UV:
                u = (dry_dp * u + precip[0, 0, -1] * u[0, 0, -1]) / (dry_dp + precip[0, 0, -1])
                v = (dry_dp * v + precip[0, 0, -1] * v[0, 0, -1]) / (dry_dp + precip[0, 0, -1])

            if DO_SEDI_W:
                w = (dm * w + precip[0, 0, -1] * (w[0, 0, -1] - terminal_velocity[0, 0, -1]) + precip * terminal_velocity) / (dm + precip[0, 0, -1])

    with computation(PARALLEL), interval(...):
        if precip_fall:
            # energy change during sedimentation heating
            total_energy_1 = moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dry_dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, False) * GRAV

    with computation(FORWARD), interval(1, None):
        if precip_fall and DO_SEDI_HEAT:
            dgz = -0.5 * GRAV * (dz[0, 0, -1] + dz)
            cv0 = dm * (CV_AIR + vapor * CV_VAP + (rain + liquid) * C_LIQ + (ice + snow + graupel) * C_ICE) + C_ICE * (precip - precip[0, 0, -1])

            t = (cv0 * t + precip[0, 0, -1] * (C_ICE * t[0, 0, -1] + dgz)) / (cv0 + C_ICE * precip[0, 0, -1])

    with computation(PARALLEL), interval(...):
        if precip_fall and DO_SEDI_HEAT:
            # energy change during sedimentation heating
            total_energy_2 = moist_total_energy(t, vapor, liquid, rain, ice, snow, graupel, dry_dp, C_AIR, C1_VAP, C1_LIQ, C1_ICE, False) * GRAV

    with computation(FORWARD), interval(...):
        if precip_fall:
            # energy change during sedimentation
            sum_total_energy_1 = sum_total_energy_1 + total_energy_1
            sum_total_energy_2 = sum_total_energy_2 + total_energy_2

    with computation(FORWARD), interval(0, 1):
        if precip_fall:
            # energy change during sedimentation
            dtotal_energy = dtotal_energy + sum_total_energy_1 - sum_total_energy_2


def ensure_non_negative_at_toa(field):
    with computation(FORWARD), interval(0, 1):
        field = max(0.0, field)


@dataclasses.dataclass
class SedimentationLocals(LocalState):
    cvm: Local = dataclasses.field(
        metadata={
            "name": "cvm",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float64,
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
    ze: Local = dataclasses.field(
        metadata={
            "name": "ze",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    zs: Local = dataclasses.field(
        metadata={
            "name": "zs",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    zt: Local = dataclasses.field(
        metadata={
            "name": "zt",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )


class Sedimentation:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        gfdl_1m_config: GFDL1MConfig,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
    ):
        self._set_value_2d = stencil_factory.from_dims_halo(
            func=set_value_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._calc_mhc_lhc_wrapper = stencil_factory.from_dims_halo(
            func=calc_mhc_lhc_wrapper,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._terminal_velocity_graupel_rain_snow = stencil_factory.from_dims_halo(
            func=terminal_velocity_graupel_rain_snow,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DT": gfdl_1m_config.DT_MOIST},
        )
        self._terminal_velocity_ice = stencil_factory.from_dims_halo(
            func=terminal_velocity_ice,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONST_VI": mp_namelist.CONST_VI,
                "DO_ICE_PRES_SCALING": mp_namelist.DO_ICE_PRES_SCALING,
                "IFFLAG": mp_namelist.IFFLAG,
                "VI_FAC": mp_namelist.VI_FAC,
                "VI_MAX": mp_namelist.VI_MAX,
                "VI_MIN": mp_namelist.VI_MIN,
                "aaL": Float(-1.70704e-5),
                "bbL": Float(-0.00319109),
                "ccL": Float(-0.0169876),
                "ddL": Float(0.00410839),
                "eeL": Float(1.93644),
                "aaC": Float(-4.18334e-5),
                "bbC": Float(-0.00525867),
                "ccC": Float(-0.0486519),
                "ddC": Float(0.00251197),
                "eeC": Float(1.91523),
            },
        )
        self._set_heights = stencil_factory.from_dims_halo(
            func=set_heights,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DT": gfdl_1m_config.DT_MOIST},
        )
        self._terminal_fall = stencil_factory.from_dims_halo(
            func=terminal_fall,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "C_AIR": mp_config.C_AIR,
                "C1_ICE": mp_config.C1_ICE,
                "C1_LIQ": mp_config.C1_LIQ,
                "C1_VAP": mp_config.C1_VAP,
                "DO_SEDI_HEAT": mp_namelist.DO_SEDI_HEAT,
                "DO_SEDI_UV": mp_namelist.DO_SEDI_UV,
                "DO_SEDI_W": mp_namelist.DO_SEDI_W,
                "DT": gfdl_1m_config.DT_MOIST,
                "SEDFLAG": mp_namelist.SEDFLAG,
            },
        )
        self._ensure_non_negative_at_toa = stencil_factory.from_dims_halo(
            func=ensure_non_negative_at_toa,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # initialize locals for sedimentation
        self._sedimentation_locals = SedimentationLocals.make_locals(quantity_factory)

        # make config visible at runtime
        self._mp_config = mp_config
        self._mp_namelist = mp_namelist

    def __call__(self, state: GFDL1MState, gfdl_1m_locals: GFDL1MLocals, gfdl_mp_v3_locals: GFDLMPV3Locals, mp_full_locals: MPFullLocals):

        # reset locals
        self._set_value_2d(mp_full_locals.surface_precip.ice, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip.liquid, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip.graupel, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip.rain, Float(0.0))
        self._set_value_2d(mp_full_locals.surface_precip.snow, Float(0.0))

        self._set_value(mp_full_locals.precip.ice, Float(0.0))
        self._set_value(mp_full_locals.precip.liquid, Float(0.0))
        self._set_value(mp_full_locals.precip.graupel, Float(0.0))
        self._set_value(mp_full_locals.precip.rain, Float(0.0))
        self._set_value(mp_full_locals.precip.snow, Float(0.0))

        self._set_value_2d(mp_full_locals.terminal_velocity.ice, Float(0.0))
        self._set_value_2d(mp_full_locals.terminal_velocity.liquid, Float(0.0))
        self._set_value_2d(mp_full_locals.terminal_velocity.graupel, Float(0.0))
        self._set_value_2d(mp_full_locals.terminal_velocity.rain, Float(0.0))
        self._set_value_2d(mp_full_locals.terminal_velocity.snow, Float(0.0))

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
            total_liquid=self._sedimentation_locals.total_liquid,
            total_solid=self._sedimentation_locals.total_solid,
            cvm=self._sedimentation_locals.cvm,
            total_energy=self._sedimentation_locals.total_energy,
            lcpk=self._sedimentation_locals.lcpk,
            icpk=self._sedimentation_locals.icpk,
            tcpk=self._sedimentation_locals.tcpk,
            tcp3=self._sedimentation_locals.tcp3,
        )

        # --------------------------------------------------
        # terminal fall and melting of falling cloud ice into rain
        # --------------------------------------------------
        if self._mp_namelist.DO_PSD_ICE_FALL:
            self._terminal_velocity_graupel_rain_snow(
                condensate=gfdl_mp_v3_locals.ice,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                terminal_velocity=mp_full_locals.terminal_velocity.ice,
                tva=self._mp_config.TVAI,
                tvb=self._mp_config.TVBI,
                blin=self._mp_namelist.BLINI,
                mu=self._mp_namelist.MUI,
                v_min=self._mp_namelist.VI_MIN,
                v_max=self._mp_namelist.VI_MAX,
                v_fac=self._mp_namelist.VI_FAC,
                const_v=self._mp_namelist.CONST_VI,
            )
        else:
            self._terminal_velocity_ice(
                t=gfdl_mp_v3_locals.t,
                ice=gfdl_mp_v3_locals.ice,
                density=gfdl_mp_v3_locals.density,
                terminal_velocity_ice=mp_full_locals.terminal_velocity.ice,
                convection_fraction=gfdl_mp_v3_locals.convection_fraction,
            )

        if self._mp_namelist.DO_SEDI_MELT_QI:
            ndsl_log.error(
                "[GFDL1M Microphysics]: NDSL version of DO_SEDI_MELT_QI = True option has not been implemented. "
                "This should have been caught by the configuration checker - this error should never be triggered. There are multiple problems."
            )
            raise ValueError(
                "[GFDL1M Microphysics]: NDSL version of DO_SEDI_MELT_QI = True option has not been implemented. "
                "This should have been caught by the configuration checker - this error should never be triggered. There are multiple problems."
            )

        self._set_heights(
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            terminal_velocity=mp_full_locals.terminal_velocity.ice,
        )

        self._terminal_fall(
            t=gfdl_mp_v3_locals.t,
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            terminal_velocity=mp_full_locals.terminal_velocity.ice,
            precip_at_surface=mp_full_locals.surface_precip.ice,
            precip=mp_full_locals.precip.ice,
            dtotal_energy=gfdl_mp_v3_locals.total_energy.delta,
            u=gfdl_mp_v3_locals.u,
            v=gfdl_mp_v3_locals.v,
            w=gfdl_mp_v3_locals.w,
            mode=Int(1),
        )

        self._ensure_non_negative_at_toa(field=mp_full_locals.precip.ice)

        # --------------------------------------------------
        # terminal fall and melting of falling snow into rain
        # --------------------------------------------------
        self._terminal_velocity_graupel_rain_snow(
            condensate=gfdl_mp_v3_locals.snow,
            density=gfdl_mp_v3_locals.density,
            density_factor=gfdl_mp_v3_locals.density_factor,
            terminal_velocity=mp_full_locals.terminal_velocity.snow,
            tva=self._mp_config.TVAS,
            tvb=self._mp_config.TVBS,
            blin=self._mp_namelist.BLINS,
            mu=self._mp_namelist.MUS,
            v_min=self._mp_namelist.VS_MIN,
            v_max=self._mp_namelist.VS_MAX,
            v_fac=self._mp_namelist.VS_FAC,
            const_v=self._mp_namelist.CONST_VS,
        )

        if self._mp_namelist.DO_SEDI_MELT_QS:
            ndsl_log.error(
                "[GFDL1M Microphysics]: NDSL version of DO_SEDI_MELT_QS = True option has not been implemented. "
                "This should have been caught by the configuration checker - this error should never be triggered. There are multiple problems."
            )
            raise ValueError(
                "[GFDL1M Microphysics]: NDSL version of DO_SEDI_MELT_QS = True option has not been implemented. "
                "This should have been caught by the configuration checker - this error should never be triggered. There are multiple problems."
            )

        self._set_heights(
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            terminal_velocity=mp_full_locals.terminal_velocity.snow,
        )

        self._terminal_fall(
            t=gfdl_mp_v3_locals.t,
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            terminal_velocity=mp_full_locals.terminal_velocity.snow,
            precip_at_surface=mp_full_locals.surface_precip.snow,
            precip=mp_full_locals.precip.snow,
            dtotal_energy=gfdl_mp_v3_locals.total_energy.delta,
            u=gfdl_mp_v3_locals.u,
            v=gfdl_mp_v3_locals.v,
            w=gfdl_mp_v3_locals.w,
            mode=Int(5),
        )

        self._ensure_non_negative_at_toa(field=mp_full_locals.precip.snow)

        # --------------------------------------------------
        # terminal fall and melting of falling graupel into rain
        # --------------------------------------------------
        if self._mp_namelist.DO_HAIL:
            self._terminal_velocity_graupel_rain_snow(
                condensate=gfdl_mp_v3_locals.graupel,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                terminal_velocity=mp_full_locals.terminal_velocity.graupel,
                tva=self._mp_config.TVAG,
                tvb=self._mp_config.TVBG,
                blin=self._mp_namelist.BLING,
                mu=self._mp_namelist.MUG,
                v_min=self._mp_namelist.VG_MIN,
                v_max=self._mp_namelist.VG_MAX,
                v_fac=self._mp_namelist.VG_FAC,
                const_v=self._mp_namelist.CONST_VG,
            )
        else:
            self._terminal_velocity_graupel_rain_snow(
                condensate=gfdl_mp_v3_locals.graupel,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                terminal_velocity=mp_full_locals.terminal_velocity.graupel,
                tva=self._mp_config.TVAG,
                tvb=self._mp_config.TVBG,
                blin=self._mp_namelist.BLING,
                mu=self._mp_namelist.MUG,
                v_min=self._mp_namelist.VG_MIN,
                v_max=self._mp_namelist.VG_MAX,
                v_fac=self._mp_namelist.VG_FAC,
                const_v=self._mp_namelist.CONST_VG,
            )

        if self._mp_namelist.DO_SEDI_MELT_QG:
            ndsl_log.error(
                "[GFDL1M Microphysics]: NDSL version of DO_SEDI_MELT_QG = True option has not been implemented. "
                "This should have been caught by the configuration checker - this error should never be triggered. There are multiple problems."
            )
            raise ValueError(
                "[GFDL1M Microphysics]: NDSL version of DO_SEDI_MELT_QG = True option has not been implemented. "
                "This should have been caught by the configuration checker - this error should never be triggered. There are multiple problems."
            )

        self._set_heights(
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            terminal_velocity=mp_full_locals.terminal_velocity.graupel,
        )

        self._terminal_fall(
            t=gfdl_mp_v3_locals.t,
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            terminal_velocity=mp_full_locals.terminal_velocity.graupel,
            precip_at_surface=mp_full_locals.surface_precip.graupel,
            precip=mp_full_locals.precip.graupel,
            dtotal_energy=gfdl_mp_v3_locals.total_energy.delta,
            u=gfdl_mp_v3_locals.u,
            v=gfdl_mp_v3_locals.v,
            w=gfdl_mp_v3_locals.w,
            mode=Int(3),
        )

        self._ensure_non_negative_at_toa(field=mp_full_locals.precip.graupel)

        # --------------------------------------------------
        # terminal fall of cloud water
        # --------------------------------------------------
        if self._mp_namelist.DO_PSD_WATER_FALL:
            self._terminal_velocity_graupel_rain_snow(
                condensate=gfdl_mp_v3_locals.liquid,
                density=gfdl_mp_v3_locals.density,
                density_factor=gfdl_mp_v3_locals.density_factor,
                terminal_velocity=mp_full_locals.terminal_velocity.liquid,
                tva=self._mp_config.TVAW,
                tvb=self._mp_config.TVBW,
                blin=self._mp_namelist.BLINW,
                mu=self._mp_namelist.MUW,
                v_min=self._mp_namelist.VW_MIN,
                v_max=self._mp_namelist.VW_MAX,
                v_fac=self._mp_namelist.VW_FAC,
                const_v=self._mp_namelist.CONST_VW,
            )

        self._set_heights(
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            terminal_velocity=mp_full_locals.terminal_velocity.liquid,
        )

        self._terminal_fall(
            t=gfdl_mp_v3_locals.t,
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            terminal_velocity=mp_full_locals.terminal_velocity.liquid,
            precip_at_surface=mp_full_locals.surface_precip.liquid,
            precip=mp_full_locals.precip.liquid,
            dtotal_energy=gfdl_mp_v3_locals.total_energy.delta,
            u=gfdl_mp_v3_locals.u,
            v=gfdl_mp_v3_locals.v,
            w=gfdl_mp_v3_locals.w,
            mode=Int(2),
        )

        self._ensure_non_negative_at_toa(field=mp_full_locals.precip.liquid)

        # --------------------------------------------------
        # terminal fall of rain
        # --------------------------------------------------
        self._terminal_velocity_graupel_rain_snow(
            condensate=gfdl_mp_v3_locals.rain,
            density=gfdl_mp_v3_locals.density,
            density_factor=gfdl_mp_v3_locals.density_factor,
            terminal_velocity=mp_full_locals.terminal_velocity.rain,
            tva=self._mp_config.TVAR,
            tvb=self._mp_config.TVBR,
            blin=self._mp_namelist.BLINR,
            mu=self._mp_namelist.MUR,
            v_min=self._mp_namelist.VR_MIN,
            v_max=self._mp_namelist.VR_MAX,
            v_fac=self._mp_namelist.VR_FAC,
            const_v=self._mp_namelist.CONST_VR,
        )

        self._set_heights(
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            terminal_velocity=mp_full_locals.terminal_velocity.rain,
        )

        self._terminal_fall(
            t=gfdl_mp_v3_locals.t,
            ze=self._sedimentation_locals.ze,
            zs=self._sedimentation_locals.zs,
            zt=self._sedimentation_locals.zt,
            dz=gfdl_mp_v3_locals.dz,
            dry_dp=gfdl_mp_v3_locals.dry_dp,
            vapor=gfdl_mp_v3_locals.mixing_ratio.vapor,
            ice=gfdl_mp_v3_locals.mixing_ratio.ice,
            liquid=gfdl_mp_v3_locals.mixing_ratio.liquid,
            graupel=gfdl_mp_v3_locals.mixing_ratio.graupel,
            rain=gfdl_mp_v3_locals.mixing_ratio.rain,
            snow=gfdl_mp_v3_locals.mixing_ratio.snow,
            terminal_velocity=mp_full_locals.terminal_velocity.rain,
            precip_at_surface=mp_full_locals.surface_precip.rain,
            precip=mp_full_locals.precip.rain,
            dtotal_energy=gfdl_mp_v3_locals.total_energy.delta,
            u=gfdl_mp_v3_locals.u,
            v=gfdl_mp_v3_locals.v,
            w=gfdl_mp_v3_locals.w,
            mode=Int(4),
        )

        self._ensure_non_negative_at_toa(field=mp_full_locals.precip.rain)
