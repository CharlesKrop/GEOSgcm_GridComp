import dataclasses

from ndsl import Local, LocalState, NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval, sqrt
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int, IntFieldIJ
from ndsl.stencils.basic_operations import add, copy
from ndsl.stencils.basic_operations_2d import set_value_2d

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.saturation_tables import SaturationVaporPressureTable, GlobalTable_saturation_tables, saturation_specific_humidity
from pyMoist.shared.atmos_recipes import compute_estimated_inversion_strength_factor, fill_negative_q
from pyMoist.shared.cloud_processes import evaporate, fix_up_clouds, hydrostatic_pdf, melt_freeze, sublimate


def compute_macrophysics_factors(
    estimated_inversion_strength: FloatFieldIJ,
    surface_type: IntFieldIJ,
    p_mb: FloatField,
    boundary_layer_level_for_uw_shallow_conv: FloatFieldIJ,
    min_rh_crit: FloatFieldIJ,
    turn_rh_crit: FloatFieldIJ,
):
    """Compute macrophysics factors based on estimated inversion strength and surface conditions.

    Args:
        estimated_inversion_strength (FloatFieldIJ)
        surface_type (IntFieldIJ)
        p_mb (FloatField)
        boundary_layer_level_for_uw_shallow_conv (FloatFieldIJ)
        min_rh_crit (FloatFieldIJ)
        turn_rh_crit (FloatFieldIJ)
    """
    from __externals__ import MIN_RH_STABLE, MIN_RH_UNSTABLE, TURNRHCRIT_PARAM

    with computation(FORWARD), interval(0, 1):
        fac_eis = compute_estimated_inversion_strength_factor(estimated_inversion_strength)
        min_rh_crit = MIN_RH_UNSTABLE * (1.0 - fac_eis) + MIN_RH_STABLE * fac_eis
        min_rh_crit = max(0.7, min_rh_crit)

        if TURNRHCRIT_PARAM <= 0.0:
            # Level indices are positive, so truncation after +0.5 matches NINT.
            turn_rh_crit = p_mb.at(K=int(boundary_layer_level_for_uw_shallow_conv + 0.5)) - 50
        else:
            turn_rh_crit = TURNRHCRIT_PARAM


def compute_critical_relative_humidity(
    area: FloatFieldIJ,
    p_mb: FloatField,
    p_interface_mb: FloatField,
    min_rh_crit: FloatFieldIJ,
    turn_rh_crit: FloatFieldIJ,
    rh_crit: FloatField,
    alpha: FloatField,
    one_minus_alpha: FloatField,
):
    """Compute the critical relative humidity based on area, pressure, and threshold values.

    Args:
        area (FloatFieldIJ)
        p_mb (FloatField)
        p_interface_mb (FloatField)
        min_rh_crit (FloatFieldIJ)
        turn_rh_crit (FloatFieldIJ)
        rh_crit (FloatField)
        alpha (FloatField)
        one_minus_alpha (FloatField)
    """
    from __externals__ import MAX_RH_CRIT, k_end

    with computation(PARALLEL), interval(...):
        # use Slingo-Ritter (1985) formulation for critical relative humidity
        # ensure the max is never lower than the min
        safe_max_rh_crit = max(MAX_RH_CRIT, min_rh_crit)
        if p_mb <= turn_rh_crit:
            minimum_rh = min_rh_crit
        elif K == k_end:
            minimum_rh = safe_max_rh_crit
        else:
            x_norm = (p_mb - turn_rh_crit) / (p_interface_mb.at(K=k_end + 1) - turn_rh_crit)
            # cubic smoothstep S-curve: x^2 * (3 - 2x)
            minimum_rh = min_rh_crit + (safe_max_rh_crit - min_rh_crit) * (x_norm * x_norm * (3.0 - 2.0 * x_norm))

    # scale-aware blending for rhcrit
    with computation(PARALLEL), interval(...):
        rh_crit = MAX_RH_CRIT + (minimum_rh - MAX_RH_CRIT) * sqrt(sqrt(area / 1.0e10))
        # limit alpha to < 30 %
        alpha = max(0.0, min(0.30, (1.0 - rh_crit)))
        one_minus_alpha = 1.0 - alpha


def relative_humidity_after_pdf(t: FloatField, p_mb: FloatField, vapor: FloatField, esx: GlobalTable_saturation_tables, rh: FloatField):
    with computation(PARALLEL), interval(...):
        qsat, dqsat = saturation_specific_humidity(t, p_mb * 100.0, esx)
        rh = vapor / qsat


def update_output_by_dt(input: FloatField, output: FloatField):
    """Update the output field based on the time derivative of the input field.

    Args:
        input (FloatField)
        output (FloatField)
    """
    from __externals__ import DTIME

    with computation(PARALLEL), interval(...):
        output = (input - output) / DTIME


@dataclasses.dataclass
class GFDL1MMacrophysicsLocals(LocalState):
    min_rh_crit: Local = dataclasses.field(
        metadata={
            "name": "min_rh_crit",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    turn_rh_crit: Local = dataclasses.field(
        metadata={
            "name": "turn_rh_crit",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    rh_crit: Local = dataclasses.field(
        metadata={
            "name": "rh_crit",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    alpha: Local = dataclasses.field(
        metadata={
            "name": "alpha",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    one_minus_alpha: Local = dataclasses.field(
        metadata={
            "name": "one_minus_alpha",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    temporary_2d: Local = dataclasses.field(
        metadata={
            "name": "temporary_2d",
            "dims": [I_DIM, J_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    temporary_3d: Local = dataclasses.field(
        metadata={
            "name": "temporary_3d",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )
    dummy_field_no_read_write: Local = dataclasses.field(
        metadata={
            "name": "dummy_field_no_read_write",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "dtype": Float,
        }
    )


class GFDL1MMacrophysics(NDSLRuntime):
    """GFDL 1M Macrophysics

    This class implements the macrophysics component of the GFDL 1M scheme. It includes a hydrostatic PDF solver,
    critical relative humidity calculations, and phase change evaluations.

    NDSL integration: Kropiewnicki September 2026
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
        saturation_tables: SaturationVaporPressureTable,
        lid_level: int,
    ):
        # init NDSLRuntime
        super().__init__(stencil_factory)

        # make saturation tables visible at runtime
        self._saturation_tables = saturation_tables

        # make config visible at runtime
        self._config = config

        # initialize class specific locals
        self._locals = GFDL1MMacrophysicsLocals.make_locals(quantity_factory)

        # construct stencils

        # Fortran runs cloud macro only at/below the pressure lid.
        macro_factory = stencil_factory.restrict_vertical(k_start=lid_level)

        self._add = stencil_factory.from_dims_halo(
            func=add,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._compute_critical_relative_humidity = stencil_factory.from_dims_halo(
            func=compute_critical_relative_humidity,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"MAX_RH_CRIT": config.MAX_RH_CRIT},
        )
        self._compute_macrophysics_factors = stencil_factory.from_dims_halo(
            func=compute_macrophysics_factors,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "MIN_RH_UNSTABLE": config.MIN_RH_UNSTABLE,
                "MIN_RH_STABLE": config.MIN_RH_STABLE,
                "TURNRHCRIT_PARAM": config.TURNRHCRIT,
            },
        )
        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._evaporate = macro_factory.from_dims_halo(
            func=evaporate,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DTIME": config.DT,
                "CCW_EVAP_EFF": config.CCW_EVAP_EFF,
            },
        )
        self._fill_negative_q = stencil_factory.from_dims_halo(
            func=fill_negative_q,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DTIME": config.DT},
        )
        self._fix_up_clouds = stencil_factory.from_dims_halo(
            func=fix_up_clouds,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"LID_LEVEL": lid_level},
        )
        self._relative_humidity_after_pdf = macro_factory.from_dims_halo(
            func=relative_humidity_after_pdf,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._hydrostatic_pdf = macro_factory.from_dims_halo(
            func=hydrostatic_pdf,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DTIME": config.DT,
                "PDFSHAPE": config.PDFSHAPE,
                "USE_BERGERON": config.USE_BERGERON,
            },
        )
        self._melt_freeze = macro_factory.from_dims_halo(
            func=melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DTIME": config.DT},
        )
        self._set_value_2d = stencil_factory.from_dims_halo(
            func=set_value_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._sublimate = macro_factory.from_dims_halo(
            func=sublimate,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "DTIME": config.DT,
                "CCI_EVAP_EFF": config.CCI_EVAP_EFF,
                "ICE_RADII_PARAM": config.ICE_RADII_PARAM,
            },
        )
        self._update_output_by_dt = stencil_factory.from_dims_halo(
            func=update_output_by_dt,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DTIME": config.DT},
        )

    def __call__(self, state: GFDL1MState, locals: GFDL1MLocals):

        # compute miscellaneous factors used later in macrophysics
        self._compute_macrophysics_factors(
            estimated_inversion_strength=state.estimated_inversion_strength,
            surface_type=state.surface_type,
            p_mb=locals.p_mb,
            boundary_layer_level_for_uw_shallow_conv=state.boundary_layer_level_for_uw_shallow_conv,
            min_rh_crit=self._locals.min_rh_crit,
            turn_rh_crit=self._locals.turn_rh_crit,
        )

        # ensure physically reasonable values before macrophysics
        self._fix_up_clouds(
            t=state.t,
            vapor=state.mixing_ratio.vapor,
            type_one_ice=state.mixing_ratio.large_scale_ice,
            type_one_liquid=state.mixing_ratio.large_scale_liquid,
            type_one_cloud_fraction=state.cloud_fraction.large_scale,
            type_two_ice=state.mixing_ratio.convective_ice,
            type_two_liquid=state.mixing_ratio.convective_liquid,
            type_two_cloud_fraction=state.cloud_fraction.convective,
        )

        self._compute_critical_relative_humidity(
            area=state.area,
            p_mb=locals.p_mb,
            p_interface_mb=locals.p_interface_mb,
            min_rh_crit=self._locals.min_rh_crit,
            turn_rh_crit=self._locals.turn_rh_crit,
            rh_crit=self._locals.rh_crit,
            alpha=self._locals.alpha,
            one_minus_alpha=self._locals.one_minus_alpha,
        )

        # export critical relative humidity if the state variable exists
        if state.critical_relative_humidity_for_pdf is not None:
            self._copy(input=self._locals.one_minus_alpha, output=state.critical_relative_humidity_for_pdf)

        # macrophysics core - hyrostatic pdf
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

        self._relative_humidity_after_pdf(
            t=state.t,
            p_mb=locals.p_mb,
            vapor=state.mixing_ratio.vapor,
            esx=self._saturation_tables.esx,
            rh=state.relative_humidity_after_pdf,
        )

        # melt/freeze particles if enabled
        if self._config.MELTFRZ_CLDMACRO:
            # since we are explicitly operating on convective types pass convection_fraction always as 1.0
            self._set_value_2d(field=self._locals.temporary_2d, value=Float(1.0))
            self._melt_freeze(
                convection_fraction=self._locals.temporary_2d,
                surface_type=state.surface_type,
                t=state.t,
                liquid=state.mixing_ratio.convective_liquid,
                ice=state.mixing_ratio.convective_ice,
            )

        # cloud water evaporation - if enabled, else done in the GFDL microphysics core
        if self._config.CCW_EVAP_EFF > 0.0:
            self._copy(input=state.mixing_ratio.vapor, output=state.cloud_liquid_evaporation)

            self._evaporate(
                p_mb=locals.p_mb,
                t=state.t,
                vapor=state.mixing_ratio.vapor,
                rh_crit=self._locals.rh_crit,
                liquid=state.mixing_ratio.convective_liquid,
                ice=state.mixing_ratio.convective_ice,
                cloud_fraction=state.cloud_fraction.convective,
                concentration_liquid=state.concentration.liquid,
                saturation_specific_humidity=locals.saturation_specific_humidity,
            )

            self._update_output_by_dt(input=state.mixing_ratio.vapor, output=state.cloud_liquid_evaporation)

        # cloud ice sublimation - if enabled, else done in the GFDL microphysics core
        if self._config.CCI_EVAP_EFF > 0.0:
            self._copy(input=state.mixing_ratio.vapor, output=state.cloud_ice_sublimation)

            self._sublimate(
                p_mb=locals.p_mb,
                t=state.t,
                vapor=state.mixing_ratio.vapor,
                rh_crit=self._locals.rh_crit,
                liquid=state.mixing_ratio.convective_liquid,
                ice=state.mixing_ratio.convective_ice,
                cloud_fraction=state.cloud_fraction.convective,
                saturation_specific_humidity=locals.saturation_specific_humidity,
            )

            self._update_output_by_dt(input=state.mixing_ratio.vapor, output=state.cloud_ice_sublimation)

        # repair any remaining non-physical values
        self._fix_up_clouds(
            t=state.t,
            vapor=state.mixing_ratio.vapor,
            type_one_ice=state.mixing_ratio.large_scale_ice,
            type_one_liquid=state.mixing_ratio.large_scale_liquid,
            type_one_cloud_fraction=state.cloud_fraction.large_scale,
            type_two_ice=state.mixing_ratio.convective_ice,
            type_two_liquid=state.mixing_ratio.convective_liquid,
            type_two_cloud_fraction=state.cloud_fraction.convective,
        )

        # eliminate any negative values in the following quantities
        self._fill_negative_q(q=state.mixing_ratio.vapor, dqdt=state.fill_negative_tendency_cloud_macro.vapor, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.large_scale_ice, dqdt=state.fill_negative_tendency_cloud_macro.large_scale_ice, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.large_scale_liquid, dqdt=state.fill_negative_tendency_cloud_macro.large_scale_liquid, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.convective_ice, dqdt=state.fill_negative_tendency_cloud_macro.convective_ice, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.convective_liquid, dqdt=state.fill_negative_tendency_cloud_macro.convective_liquid, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.graupel, dqdt=state.fill_negative_tendency_cloud_macro.graupel, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.rain, dqdt=state.fill_negative_tendency_cloud_macro.rain, mass=locals.mass, fill_dqdt=True)
        self._fill_negative_q(q=state.mixing_ratio.snow, dqdt=state.fill_negative_tendency_cloud_macro.snow, mass=locals.mass, fill_dqdt=True)

        # update macrophysics tendencies
        self._update_output_by_dt(input=state.u, output=state.tendencies.du_dt_macro)
        self._update_output_by_dt(input=state.v, output=state.tendencies.dv_dt_macro)
        self._update_output_by_dt(input=state.t, output=state.tendencies.dt_dt_macro)
        self._update_output_by_dt(input=state.mixing_ratio.vapor, output=state.tendencies.dvapor_dt_macro)
        self._add(summand_1=state.mixing_ratio.convective_liquid, summand_2=state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._update_output_by_dt(input=self._locals.temporary_3d, output=state.tendencies.dliquid_dt_macro)
        self._add(summand_1=state.mixing_ratio.convective_ice, summand_2=state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._update_output_by_dt(input=self._locals.temporary_3d, output=state.tendencies.dice_dt_macro)
        self._add(summand_1=state.cloud_fraction.convective, summand_2=state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._update_output_by_dt(input=self._locals.temporary_3d, output=state.tendencies.dcloud_fraction_dt_macro)
        self._update_output_by_dt(input=state.mixing_ratio.graupel, output=state.tendencies.dgraupel_dt_macro)
        self._update_output_by_dt(input=state.mixing_ratio.rain, output=state.tendencies.drain_dt_macro)
        self._update_output_by_dt(input=state.mixing_ratio.snow, output=state.tendencies.dsnow_dt_macro)
