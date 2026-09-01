from tkinter.tix import MAX

from pyMoist.shared.atmos_recipes import compute_estimated_inversion_strength_factor
from pyMoist.shared.cloud_processes import fix_up_clouds, hydrostatic_pdf, melt_freeze
from ndsl import Local, LocalState, NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval, sqrt
from ndsl.dsl.typing import BoolFieldIJ, Float, FloatField, FloatFieldIJ, IntFieldIJ, Int
from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.saturation_tables import SaturationVaporPressureTable
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
import dataclasses
from ndsl.stencils.basic_operations import copy


def compute_macrophysics_factors(
    estimated_inversion_strength: FloatFieldIJ,
    surface_type: IntFieldIJ,
    p_mb: FloatField,
    boundary_layer_level_for_uw_shallow_conv: IntFieldIJ,
    minrhcrit: FloatFieldIJ,
    turnrhcrit: FloatFieldIJ,
):
    from __externals__ import MIN_RH_UNSTABLE, MIN_RH_STABLE, TURNRHCRIT_PARAM

    with computation(FORWARD), interval(0, 1):
        fac_eis = compute_estimated_inversion_strength_factor(estimated_inversion_strength)
        minrhcrit = MIN_RH_UNSTABLE * (1.0 - fac_eis) + MIN_RH_STABLE * fac_eis
        minrhcrit = max(0.7, minrhcrit)

        if TURNRHCRIT_PARAM <= 0.0:
            turnrhcrit = p_mb.at(K=boundary_layer_level_for_uw_shallow_conv) - 50
        else:
            turnrhcrit = TURNRHCRIT_PARAM


def compute_critical_relative_humidity(
    area: FloatFieldIJ,
    p_mb: FloatField,
    min_rh_crit: FloatFieldIJ,
    turn_rh_crit: FloatFieldIJ,
    rh_crit: FloatField,
    alpha: FloatField,
    one_minus_alpha: FloatField,
):
    from __externals__ import MAX_RH_CRIT, k_end

    with computation(FORWARD), interval(0, 1):
        # use Slingo-Ritter (1985) formulation for critical relative humidity
        # ensure the max is never lower than the min
        safe_max_rh_crit = MAX(MAX_RH_CRIT, min_rh_crit)
        if p_mb <= turn_rh_crit:
            min_rh_crit = min_rh_crit
        elif K == k_end:
            min_rh_crit = safe_max_rh_crit
        else:
            x_norm = (p_mb - turn_rh_crit) / (p_mb.at(K=k_end) - turn_rh_crit)
            # cubic smoothstep S-curve: x^2 * (3 - 2x)
            min_rh_crit = min_rh_crit + (safe_max_rh_crit - min_rh_crit) * (x_norm * x_norm * (3.0 - 2.0 * x_norm))

    # scale-aware blending for rhcrit
    with computation(PARALLEL), interval(...):
        rh_crit = MAX_RH_CRIT + (min_rh_crit - MAX_RH_CRIT) * sqrt(sqrt(area / 1.0e10))
        # limit alpha to < 30 %
        alpha = max(0.0, min(0.30, (1.0 - rh_crit)))
        one_minus_alpha = 1.0 - alpha


@dataclasses.dataclass
class GFDL1MMacrophysicsLocals(LocalState):
    min_rh_crit: Local = dataclasses.field(
        metadata={
            "name": "min_rh_crit",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    turn_rh_crit: Local = dataclasses.field(
        metadata={
            "name": "turn_rh_crit",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rh_crit: Local = dataclasses.field(
        metadata={
            "name": "rh_crit",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    alpha: Local = dataclasses.field(
        metadata={
            "name": "alpha",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    one_minus_alpha: Local = dataclasses.field(
        metadata={
            "name": "one_minus_alpha",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dummy_field_no_read_write: Local = dataclasses.field(
        metadata={
            "name": "dummy_field_no_read_write",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


class GFDL1MMacrophysics(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
        saturation_tables: SaturationVaporPressureTable,
    ):
        # init NDSLRuntime
        super().__init__(stencil_factory)

        # make saturation tables visible at runtime
        self._saturation_tables = saturation_tables

        # make config visible at runtime
        self._config = config

        # initialize class specific locals
        self._locals = GFDL1MMacrophysicsLocals.make_locals(quantity_factory)

        self._compute_macrophysics_factors = stencil_factory.from_dims_halo(
            func=compute_macrophysics_factors,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "MIN_RH_UNSTABLE": config.MIN_RH_UNSTABLE,
                "MIN_RH_STABLE": config.MIN_RH_STABLE,
                "TURNRHCRIT_PARAM": config.TURNRHCRIT_PARAM,
            },
        )

        self._fix_up_clouds = stencil_factory.from_dims_halo(
            func=fix_up_clouds,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "MIN_CLOUD_FRACTION": config.MIN_CLOUD_FRACTION,
                "MIN_CLOUD_QUANTITY": config.MIN_CLOUD_QUANTITY,
            },
        )

        self._compute_critical_relative_humidity = stencil_factory.from_dims_halo(
            func=compute_critical_relative_humidity,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"MAX_RH_CRIT": config.MAX_RH_CRIT},
        )

        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._hydrostatic_pdf = stencil_factory.from_dims_halo(
            func=hydrostatic_pdf,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "dtime": config.DT_MOIST,
                "PDFSHAPE": config.PDFSHAPE,
                "MIN_CLOUD_FRACTION": config.MIN_CLOUD_FRACTION,
                "USE_BERGERON": config.USE_BERGERON,
            },
        )

        self._melt_freeze = stencil_factory.from_dims_halo(
            func=melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"dtime": config.DT_MOIST},
        )

    def __call__(self, state: GFDL1MState, locals: GFDL1MLocals):

        self._compute_macrophysics_factors(
            estimated_inversion_strength=state.estimated_inversion_strength,
            surface_type=state.surface_type,
            p_mb=locals.p_mb,
            boundary_layer_level_for_uw_shallow_conv=state.boundary_layer_level_for_uw_shallow_conv,
            turnrhcrit=locals.turnrhcrit,
        )

        self._fix_up_clouds(
            t=state.t,
            vapor=state.mixing_ratio.vapor,
            type_one_ice=state.mixing_ratio.large_scale_ice,
            type_one_liquid=state.mixing_ratio.large_scale_liquid,
            type_one_cloud_fraction=state.cloud_fraction.large_scale,
            type_two_ice=state.mixing_ratio.convective_ice,
            type_two_liquid=state.mixing_ratio.convective_liquid,
            type_two_cloud_fraction=state.cloud_fraction.convective,
            lid_level=locals.lid_level,
        )

        self._compute_critical_relative_humidity(
            area=state.area,
            p_mb=locals.p_mb,
            min_rh_crit=self._locals.min_rh_crit,
            turn_rh_crit=self._locals.turn_rh_crit,
            rh_crit=self._locals.rh_crit,
            alpha=self._locals.alpha,
            one_minus_alpha=self._locals.one_minus_alpha,
        )

        if state.critical_relative_humidity_for_pdf is not None:
            self._copy(input=self._locals.one_minus_alpha, output=state.critical_relative_humidity_for_pdf)

        self._hydrostatic_pdf(
            convection_fraction=state.convection_fraction,
            surface_type=state.surface_type,
            p_mb=locals.p_mb,
            layer_height_above_surface=locals.layer_height_above_surface,
            t=state.t,
            alpha=self._locals.alpha,
            vapor=state.mixing_ratio.vapor,
            large_scale_cloud_fraction=state.cloud_fraction.large_scale,
            large_scale_ice=state.mixing_ratio.large_scale_ice,
            large_scale_liquid=state.mixing_ratio.large_scale_liquid,
            convective_cloud_fraction=state.cloud_fraction.convective,
            convective_ice=state.mixing_ratio.convective_ice,
            convective_liquid=state.mixing_ratio.convective_liquid,
            concentration_ice=state.concentration.ice,
            concentration_liquid=state.concentration.liquid,
            liquid_water_static_energy_flux=state.liquid_water_static_energy.flux,
            liquid_water_static_energy_variance=state.liquid_water_static_energy.variance,
            liquid_water_static_energy_third_moment=state.liquid_water_static_energy.third_moment,
            total_water_flux=state.total_water.flux,
            total_water_variance=state.total_water.variance,
            total_water_third_moment=state.total_water.third_moment,
            covariance_liquid_water_static_energy_and_total_water_specific_humidity=state.covariance_liquid_water_static_energy_and_total_water_specific_humidity,
            vertical_motion_variance=state.vertical_motion.variance,
            vertical_motion_third_moment=state.vertical_motion.third_moment,
            pdf_first_plume_fractional_area=state.pdf_first_plume_fractional_area,
            hydrostatic_pdf_iterations=state.hydrostatic_pdf_iterations,
            buoyancy_flux=state.buoyancy_flux,
            liquid_water_flux=state.liquid_water_flux,
            bergeron_needs_preexisting=False,
            use_sc_ice=False,
            sc_ice=self._locals.dummy_field_no_read_write,
            iteration_method=Int(1),
            ese=self._saturation_tables.ese,
            esw=self._saturation_tables.esw,
            esx=self._saturation_tables.esx,
            estfrz=self._saturation_tables.frz,
            estlqu=self._saturation_tables.lqu,
        )

        if self._config.MELT_FREEZE_CLOUD_MACRO:
            self._melt_freeze(
                convection_fraction=Float(1.0),  # since we are explicitly operating on convective types pass convection_fraction always as 1.0
                surface_type=state.surface_type,
                t=state.t,
                liquid=state.mixing_ratio.convective_liquid,
                ice=state.mixing_ratio.convective_ice,
            )

        if self._config.CCW_EVAP_EFF > 0.0:  # else evap done inside GFDL
            self._copy(input=state.mixing_ratio.vapor, output=state.cloud_liquid_evaporation)
            
