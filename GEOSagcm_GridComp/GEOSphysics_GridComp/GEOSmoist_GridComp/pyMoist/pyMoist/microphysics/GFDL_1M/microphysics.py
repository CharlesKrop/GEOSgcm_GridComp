from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils.basic_operations import add, copy, set_value

from pyMoist.microphysics.gfdl_1m.config import GFDL1MConfig
from pyMoist.microphysics.gfdl_1m.gfdl_mp_v3.gfdl_mp_v3 import GFDLMPV3
from pyMoist.microphysics.gfdl_1m.locals import GFDL1MLocals
from pyMoist.microphysics.gfdl_1m.state import GFDL1MState
from pyMoist.shared.atmos_recipes import fill_negative_q
from pyMoist.shared.cloud_processes import fix_up_clouds, melt_freeze, redistribute_clouds_function
from pyMoist.shared.radiation_coupling import radiation_coupling_scale_aware


def convert_mm_per_day_to_kg_per_m2_per_s(field: FloatField):
    """Convert a field from mm day^-1 to kg m^-2 s^-1."""
    with computation(PARALLEL), interval(...):
        field = field * (1.0 / 86400.0)


def flip_sign(input: FloatField, output: FloatField):
    with computation(PARALLEL), interval(...):
        output = -1.0 * input


def combine_cloud_fractions(large_scale: FloatField, convective: FloatField, combined: FloatField):
    with computation(PARALLEL), interval(...):
        combined = large_scale + convective - (convective * large_scale)
        combined = min(max(combined, 0.0), 1.0)


def negative_adjustment(
    graupel: FloatField,
    ice: FloatField,
    liquid: FloatField,
    rain: FloatField,
    snow: FloatField,
    vapor: FloatField,
):
    with computation(FORWARD), interval(0, 1):
        n_solid_fix = 0
        n_liquid_fix = 0
        n_vapor_col = 0

    with computation(PARALLEL), interval(...):
        if ice < 0.0 and snow > 0.0:
            dq = min(-ice, snow)
            ice = ice + dq
            snow = snow - dq
            n_solid_fix = n_solid_fix + 1

        if snow < 0.0 and graupel > 0.0:
            dq = min(-snow, graupel)
            snow = snow + dq
            graupel = graupel - dq
            n_solid_fix = n_solid_fix + 1

        if graupel < 0.0 and rain > 0.0:
            dq = min(-graupel, rain)
            graupel = graupel + dq
            rain = rain - dq
            n_solid_fix = n_solid_fix + 1

        if rain < 0.0 and liquid > 0.0:
            dq = min(-rain, liquid)
            rain = rain + dq
            liquid = liquid - dq
            n_liquid_fix = n_liquid_fix + 1

        if liquid < 0.0 and vapor > 0.0:
            dq = min(-liquid, vapor)
            liquid = liquid + dq
            vapor = vapor - dq
            n_liquid_fix = n_liquid_fix + 1

    with computation(FORWARD), interval(0, -1):
        if vapor < 0.0:
            vapor[0, 0, 1] = vapor[0, 0, 1] + vapor
            vapor = 0.0
            n_vapor_col = n_vapor_col + 1

    with computation(FORWARD), interval(-1, None):
        if vapor < 0.0 and vapor[0, 0, -1] > 0.0:
            dq = min(-vapor, vapor[0, 0, -1])
            vapor[0, 0, -1] = vapor[0, 0, -1] - dq
            vapor = vapor + dq
            n_vapor_col = n_vapor_col + 1

    with computation(PARALLEL), interval(...):
        graupel = max(graupel, 0.0)
        ice = max(ice, 0.0)
        liquid = max(liquid, 0.0)
        rain = max(rain, 0.0)
        snow = max(snow, 0.0)
        vapor = max(vapor, 1.0e-15)


def post_microphysics_calculations(
    convective_cloud_fraction: FloatField,
    convective_ice: FloatField,
    convective_liquid: FloatField,
    dcloud_fraction_dt: FloatField,
    graupel: FloatField,
    large_scale_cloud_fraction: FloatField,
    large_scale_ice: FloatField,
    large_scale_liquid: FloatField,
    radiation_cloud_fraction: FloatField,
    radiation_graupel: FloatField,
    radiation_ice: FloatField,
    radiation_liquid: FloatField,
    radiation_rain: FloatField,
    radiation_snow: FloatField,
    radiation_vapor: FloatField,
    rain: FloatField,
    snow: FloatField,
    t: FloatField,
    vapor: FloatField,
):
    """Update radiation cloud fraction, redistribute clouds, and update water fields."""
    from __externals__ import DT

    with computation(PARALLEL), interval(...):
        if radiation_liquid + radiation_ice > 0.0:
            radiation_cloud_fraction = min(1.0, max(0.0, radiation_cloud_fraction + dcloud_fraction_dt * DT))
        else:
            radiation_cloud_fraction = 0.0

        (
            radiation_cloud_fraction,
            convective_cloud_fraction,
            convective_ice,
            convective_liquid,
            radiation_ice,
            large_scale_cloud_fraction,
            large_scale_ice,
            large_scale_liquid,
            radiation_liquid,
            t,
            radiation_vapor,
        ) = redistribute_clouds_function(
            cloud_fraction=radiation_cloud_fraction,
            convective_cloud_fraction=convective_cloud_fraction,
            convective_ice=convective_ice,
            convective_liquid=convective_liquid,
            ice=radiation_ice,
            large_scale_cloud_fraction=large_scale_cloud_fraction,
            large_scale_ice=large_scale_ice,
            large_scale_liquid=large_scale_liquid,
            liquid=radiation_liquid,
            t=t,
            vapor=radiation_vapor,
        )

        graupel = radiation_graupel
        rain = radiation_rain
        snow = radiation_snow
        vapor = radiation_vapor


def precip_diagnostics(
    freezing_rainfall: FloatFieldIJ,
    icefall: FloatFieldIJ,
    non_anvil_large_scale_precip: FloatFieldIJ,
    non_anvil_large_scale_snow: FloatFieldIJ,
    precip_at_surface_graupel: FloatFieldIJ,
    precip_at_surface_ice: FloatFieldIJ,
    precip_at_surface_rain: FloatFieldIJ,
    precip_at_surface_snow: FloatFieldIJ,
    precip_at_surface_water: FloatFieldIJ,
):
    """Fill output precipitation diagnostic fields."""
    with computation(FORWARD), interval(0, 1):
        precip_at_surface_graupel = max(precip_at_surface_graupel * (1.0 / 86400), 0.0)
        precip_at_surface_ice = max(precip_at_surface_ice * (1.0 / 86400), 0.0)
        precip_at_surface_rain = max(precip_at_surface_rain * (1.0 / 86400), 0.0)
        precip_at_surface_snow = max(precip_at_surface_snow * (1.0 / 86400), 0.0)
        precip_at_surface_water = max(precip_at_surface_water * (1.0 / 86400), 0.0)

        non_anvil_large_scale_precip = precip_at_surface_rain
        non_anvil_large_scale_snow = precip_at_surface_snow
        icefall = precip_at_surface_ice + precip_at_surface_graupel
        freezing_rainfall = 0.0


def redistribute_precipitation_fluxes(
    anvil_ice_precip_flux: FloatField,
    anvil_liquid_precip_flux: FloatField,
    convective_ice: FloatField,
    convective_liquid: FloatField,
    non_anvil_large_scale_graupel_precip_flux: FloatField,
    non_anvil_large_scale_ice_precip_flux: FloatField,
    non_anvil_large_scale_liquid_precip_flux: FloatField,
    non_anvil_large_scale_rain_precip_flux: FloatField,
    non_anvil_large_scale_snow_precip_flux: FloatField,
    radiation_ice: FloatField,
    radiation_liquid: FloatField,
):
    """Redistribute precipitation fluxes for chemistry."""
    with computation(PARALLEL), interval(1, None):
        tmp_val = min(1.0, max(convective_liquid[0, 0, -1] / max(radiation_liquid[0, 0, -1], 1.0e-8), 0.0))
        anvil_liquid_precip_flux = (non_anvil_large_scale_liquid_precip_flux + non_anvil_large_scale_rain_precip_flux) * tmp_val
        non_anvil_large_scale_liquid_precip_flux = (non_anvil_large_scale_liquid_precip_flux + non_anvil_large_scale_rain_precip_flux) - anvil_liquid_precip_flux

        tmp_val = min(1.0, max(convective_ice[0, 0, -1] / max(radiation_ice[0, 0, -1], 1.0e-8), 0.0))
        anvil_ice_precip_flux = (non_anvil_large_scale_ice_precip_flux + non_anvil_large_scale_snow_precip_flux + non_anvil_large_scale_graupel_precip_flux) * tmp_val
        non_anvil_large_scale_ice_precip_flux = (
            non_anvil_large_scale_ice_precip_flux + non_anvil_large_scale_snow_precip_flux + non_anvil_large_scale_graupel_precip_flux
        ) - anvil_ice_precip_flux


def update_microphysics_tendencies(
    dcloud_fraction_dt_micro: FloatField,
    dgraupel_dt_micro: FloatField,
    dice_dt_micro: FloatField,
    dliquid_dt_micro: FloatField,
    drain_dt_micro: FloatField,
    dsnow_dt_micro: FloatField,
    du_dt_micro: FloatField,
    dv_dt_micro: FloatField,
    dt_dt_micro: FloatField,
    dvapor_dt_micro: FloatField,
    convective_cloud_fraction: FloatField,
    convective_ice: FloatField,
    convective_liquid: FloatField,
    large_scale_cloud_fraction: FloatField,
    large_scale_ice: FloatField,
    large_scale_liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    u: FloatField,
    v: FloatField,
    t: FloatField,
    vapor: FloatField,
):
    """Update microphysics tendencies from the final fields."""
    from __externals__ import DT

    with computation(PARALLEL), interval(...):
        dvapor_dt_micro = (vapor - dvapor_dt_micro) / DT
        dliquid_dt_micro = ((large_scale_liquid + convective_liquid) - dliquid_dt_micro) / DT
        dice_dt_micro = ((large_scale_ice + convective_ice) - dice_dt_micro) / DT
        dcloud_fraction_dt_micro = ((large_scale_cloud_fraction + convective_cloud_fraction) - dcloud_fraction_dt_micro) / DT
        drain_dt_micro = (rain - drain_dt_micro) / DT
        dsnow_dt_micro = (snow - dsnow_dt_micro) / DT
        dgraupel_dt_micro = (graupel - dgraupel_dt_micro) / DT
        du_dt_micro = (u - du_dt_micro) / DT
        dv_dt_micro = (v - dv_dt_micro) / DT
        dt_dt_micro = (t - dt_dt_micro) / DT


class GFDL1MMicrophysics(NDSLRuntime):
    """GFDL 1M cloud microphysics, including precipitation diagnostics and output tendencies."""

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
        state: GFDL1MState,
        locals: GFDL1MLocals,
        lid_level: int,
    ):
        super().__init__(stencil_factory)

        # make config and fields visible at runtime
        self._config = config
        self._state = state
        self._locals = locals

        self._add = stencil_factory.from_dims_halo(func=add, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._post_microphysics_calculations = stencil_factory.from_dims_halo(
            func=post_microphysics_calculations,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DT": config.DT},
        )
        self._convert_mm_per_day_to_kg_per_m2_per_s = stencil_factory.from_dims_halo(
            func=convert_mm_per_day_to_kg_per_m2_per_s,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._convert_mm_per_day_to_kg_per_m2_per_s_interface = stencil_factory.from_dims_halo(
            func=convert_mm_per_day_to_kg_per_m2_per_s,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )
        self._copy = stencil_factory.from_dims_halo(func=copy, compute_dims=[I_DIM, J_DIM, K_DIM])
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
        self._flip_sign = stencil_factory.from_dims_halo(func=flip_sign, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._melt_freeze = stencil_factory.from_dims_halo(
            func=melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DTIME": config.DT},
        )
        self._combine_cloud_fractions = stencil_factory.from_dims_halo(
            func=combine_cloud_fractions,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._negative_adjustment = stencil_factory.from_dims_halo(
            func=negative_adjustment,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._precip_diagnostics = stencil_factory.from_dims_halo(
            func=precip_diagnostics,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._radiation_coupling_scale_aware = stencil_factory.from_dims_halo(
            func=radiation_coupling_scale_aware,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "MIN_RL": config.MIN_RL,
                "MAX_RL": config.MAX_RL,
                "FAC_RL": config.FAC_RL,
                "MIN_RI": config.MIN_RI,
                "MAX_RI": config.MAX_RI,
                "FAC_RI": config.FAC_RI,
                "ICE_RADII_PARAM": config.ICE_RADII_PARAM,
                "LIQ_RADII_PARAM": config.LIQ_RADII_PARAM,
            },
        )
        self._redistribute_precipitation_fluxes = stencil_factory.from_dims_halo(
            func=redistribute_precipitation_fluxes,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )
        self._set_value = stencil_factory.from_dims_halo(func=set_value, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._set_value_k_interface = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )
        self._update_microphysics_tendencies = stencil_factory.from_dims_halo(
            func=update_microphysics_tendencies,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DT": config.DT},
        )
        self._gfdl_microphysics_v3 = GFDLMPV3(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            gfdl_1m_config=config,
            lid_level=lid_level,
        )

    def __call__(self):
        """Run the complete cloud microphysics section, including the GFDLMPV3 driver."""
        self._copy(input=self._state.mixing_ratio.vapor, output=self._state.tendencies.dvapor_dt_micro)
        self._add(summand_1=self._state.mixing_ratio.convective_ice, summand_2=self._state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.tendencies.dice_dt_micro)
        self._add(summand_1=self._state.mixing_ratio.convective_liquid, summand_2=self._state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.tendencies.dliquid_dt_micro)
        self._add(summand_1=self._state.cloud_fraction.convective, summand_2=self._state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.tendencies.dcloud_fraction_dt_micro)
        self._copy(input=self._state.mixing_ratio.graupel, output=self._state.tendencies.dgraupel_dt_micro)
        self._copy(input=self._state.mixing_ratio.rain, output=self._state.tendencies.drain_dt_micro)
        self._copy(input=self._state.mixing_ratio.snow, output=self._state.tendencies.dsnow_dt_micro)
        self._copy(input=self._state.t, output=self._state.tendencies.dt_dt_micro)
        self._copy(input=self._state.u, output=self._state.tendencies.du_dt_micro)
        self._copy(input=self._state.v, output=self._state.tendencies.dv_dt_micro)

        self._flip_sign(input=self._locals.layer_thickness, output=self._locals.layer_thickness_negative)
        self._set_value(field=self._locals.dcloud_fraction_dt, value=Float(0.0))
        self._set_value_k_interface(field=self._state.non_anvil_large_scale.ice_precip_flux, value=Float(0.0))
        self._set_value_k_interface(field=self._state.non_anvil_large_scale.liquid_precip_flux, value=Float(0.0))

        self._combine_cloud_fractions(
            large_scale=self._state.cloud_fraction.large_scale,
            convective=self._state.cloud_fraction.convective,
            combined=self._state.radiation_field.cloud_fraction,
        )
        self._add(summand_1=self._state.mixing_ratio.convective_liquid, summand_2=self._state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.radiation_field.liquid)
        self._add(summand_1=self._state.mixing_ratio.convective_ice, summand_2=self._state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=self._state.radiation_field.ice)
        self._copy(input=self._state.mixing_ratio.vapor, output=self._state.radiation_field.vapor)
        self._copy(input=self._state.mixing_ratio.graupel, output=self._state.radiation_field.graupel)
        self._copy(input=self._state.mixing_ratio.rain, output=self._state.radiation_field.rain)
        self._copy(input=self._state.mixing_ratio.snow, output=self._state.radiation_field.snow)

        if self._config.GFDL_MP3:
            self._gfdl_microphysics_v3(state=self._state, locals=self._locals)
            self._convert_mm_per_day_to_kg_per_m2_per_s(self._state.non_anvil_large_scale.evaporation)
            self._convert_mm_per_day_to_kg_per_m2_per_s(self._state.non_anvil_large_scale.sublimation)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(self._state.non_anvil_large_scale.ice_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(self._state.non_anvil_large_scale.liquid_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(self._state.non_anvil_large_scale.rain_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(self._state.non_anvil_large_scale.snow_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(self._state.non_anvil_large_scale.graupel_precip_flux)
            if self._config.DO_GFDL_REFLECTIVITY:
                option_not_implemented = True

        if self._config.REPORT_GFDL_1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] negative_adjustment requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )
        self._negative_adjustment(
            graupel=self._state.radiation_field.graupel,
            ice=self._state.radiation_field.ice,
            liquid=self._state.radiation_field.liquid,
            rain=self._state.radiation_field.rain,
            snow=self._state.radiation_field.snow,
            vapor=self._state.radiation_field.vapor,
        )

        self._post_microphysics_calculations(
            convective_cloud_fraction=self._state.cloud_fraction.convective,
            convective_ice=self._state.mixing_ratio.convective_ice,
            convective_liquid=self._state.mixing_ratio.convective_liquid,
            dcloud_fraction_dt=self._locals.dcloud_fraction_dt,
            graupel=self._state.mixing_ratio.graupel,
            large_scale_cloud_fraction=self._state.cloud_fraction.large_scale,
            large_scale_ice=self._state.mixing_ratio.large_scale_ice,
            large_scale_liquid=self._state.mixing_ratio.large_scale_liquid,
            radiation_cloud_fraction=self._state.radiation_field.cloud_fraction,
            radiation_graupel=self._state.radiation_field.graupel,
            radiation_ice=self._state.radiation_field.ice,
            radiation_liquid=self._state.radiation_field.liquid,
            radiation_rain=self._state.radiation_field.rain,
            radiation_snow=self._state.radiation_field.snow,
            radiation_vapor=self._state.radiation_field.vapor,
            rain=self._state.mixing_ratio.rain,
            snow=self._state.mixing_ratio.snow,
            t=self._state.t,
            vapor=self._state.mixing_ratio.vapor,
        )

        if self._config.REPORT_GFDL_1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] fill_negative_q requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )
        if self._state.fill_negative_tendency_cloud_micro.vapor is not None:
            self._fill_negative_q(q=self._state.mixing_ratio.vapor, dqdt=self._state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.vapor, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.large_scale_liquid is not None:
            self._fill_negative_q(
                q=self._state.mixing_ratio.large_scale_liquid,
                dqdt=self._state.fill_negative_tendency_cloud_micro.large_scale_liquid,
                mass=self._locals.mass,
                fill_dqdt=True,
            )
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.large_scale_liquid, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.convective_liquid is not None:
            self._fill_negative_q(
                q=self._state.mixing_ratio.convective_liquid,
                dqdt=self._state.fill_negative_tendency_cloud_micro.convective_liquid,
                mass=self._locals.mass,
                fill_dqdt=True,
            )
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.convective_liquid, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.large_scale_ice is not None:
            self._fill_negative_q(
                q=self._state.mixing_ratio.large_scale_ice, dqdt=self._state.fill_negative_tendency_cloud_micro.large_scale_ice, mass=self._locals.mass, fill_dqdt=True
            )
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.large_scale_ice, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.convective_ice is not None:
            self._fill_negative_q(
                q=self._state.mixing_ratio.convective_ice, dqdt=self._state.fill_negative_tendency_cloud_micro.convective_ice, mass=self._locals.mass, fill_dqdt=True
            )
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.convective_ice, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.rain is not None:
            self._fill_negative_q(q=self._state.mixing_ratio.rain, dqdt=self._state.fill_negative_tendency_cloud_micro.rain, mass=self._locals.mass, fill_dqdt=True)
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.rain, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.snow is not None:
            self._fill_negative_q(q=self._state.mixing_ratio.snow, dqdt=self._state.fill_negative_tendency_cloud_micro.snow, mass=self._locals.mass, fill_dqdt=True)
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.snow, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        if self._state.fill_negative_tendency_cloud_micro.graupel is not None:
            self._fill_negative_q(q=self._state.mixing_ratio.graupel, dqdt=self._state.fill_negative_tendency_cloud_micro.graupel, mass=self._locals.mass, fill_dqdt=True)
        else:
            self._fill_negative_q(q=self._state.mixing_ratio.graupel, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)

        self._precip_diagnostics(
            freezing_rainfall=self._state.freezing_rainfall,
            icefall=self._state.icefall,
            non_anvil_large_scale_precip=self._state.non_anvil_large_scale.precip,
            non_anvil_large_scale_snow=self._state.non_anvil_large_scale.snow,
            precip_at_surface_graupel=self._state.precipitation_at_surface.graupel,
            precip_at_surface_ice=self._state.precipitation_at_surface.ice,
            precip_at_surface_rain=self._state.precipitation_at_surface.rain,
            precip_at_surface_snow=self._state.precipitation_at_surface.snow,
            precip_at_surface_water=self._state.precipitation_at_surface.water,
        )
        self._redistribute_precipitation_fluxes(
            anvil_ice_precip_flux=self._state.anvil.ice_precip_flux,
            anvil_liquid_precip_flux=self._state.anvil.liquid_precip_flux,
            convective_ice=self._state.mixing_ratio.convective_ice,
            convective_liquid=self._state.mixing_ratio.convective_liquid,
            non_anvil_large_scale_graupel_precip_flux=self._state.non_anvil_large_scale.graupel_precip_flux,
            non_anvil_large_scale_ice_precip_flux=self._state.non_anvil_large_scale.ice_precip_flux,
            non_anvil_large_scale_liquid_precip_flux=self._state.non_anvil_large_scale.liquid_precip_flux,
            non_anvil_large_scale_rain_precip_flux=self._state.non_anvil_large_scale.rain_precip_flux,
            non_anvil_large_scale_snow_precip_flux=self._state.non_anvil_large_scale.snow_precip_flux,
            radiation_ice=self._state.radiation_field.ice,
            radiation_liquid=self._state.radiation_field.liquid,
        )

        if self._config.MELTFRZ_CLDMICRO:
            self._melt_freeze(
                convection_fraction=self._state.convection_fraction,
                surface_type=self._state.surface_type,
                t=self._state.t,
                liquid=self._state.mixing_ratio.convective_liquid,
                ice=self._state.mixing_ratio.convective_ice,
            )
            self._melt_freeze(
                convection_fraction=self._state.convection_fraction,
                surface_type=self._state.surface_type,
                t=self._state.t,
                liquid=self._state.mixing_ratio.large_scale_liquid,
                ice=self._state.mixing_ratio.large_scale_ice,
            )
            self._fix_up_clouds(
                t=self._state.t,
                vapor=self._state.mixing_ratio.vapor,
                type_one_ice=self._state.mixing_ratio.large_scale_ice,
                type_one_liquid=self._state.mixing_ratio.large_scale_liquid,
                type_one_cloud_fraction=self._state.cloud_fraction.large_scale,
                type_two_ice=self._state.mixing_ratio.convective_ice,
                type_two_liquid=self._state.mixing_ratio.convective_liquid,
                type_two_cloud_fraction=self._state.cloud_fraction.convective,
            )

        self._radiation_coupling_scale_aware(
            t=self._state.t,
            p_mb=self._locals.p_mb,
            cloud_particle_effective_radius_liquid=self._state.cloud_particle_effective_radius.liquid,
            cloud_particle_effective_radius_ice=self._state.cloud_particle_effective_radius.ice,
            concentration_liquid=self._state.concentration.liquid,
            concentration_ice=self._state.concentration.ice,
            convection_fraction=self._state.convection_fraction,
            radiation_cloud_fraction=self._state.radiation_field.cloud_fraction,
            radiation_graupel=self._state.radiation_field.graupel,
            radiation_ice=self._state.radiation_field.ice,
            radiation_liquid=self._state.radiation_field.liquid,
            radiation_rain=self._state.radiation_field.rain,
            radiation_snow=self._state.radiation_field.snow,
            radiation_vapor=self._state.radiation_field.vapor,
            anvil_cloud_fraction=self._state.cloud_fraction.convective,
            anvil_ice=self._state.mixing_ratio.convective_ice,
            anvil_liquid=self._state.mixing_ratio.convective_liquid,
            large_scale_cloud_fraction=self._state.cloud_fraction.large_scale,
            large_scale_ice=self._state.mixing_ratio.large_scale_ice,
            large_scale_liquid=self._state.mixing_ratio.large_scale_liquid,
            graupel=self._state.mixing_ratio.graupel,
            rain=self._state.mixing_ratio.rain,
            snow=self._state.mixing_ratio.snow,
            vapor=self._state.mixing_ratio.vapor,
        )

        if self._config.REPORT_GFDL_1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] fill_negative_q requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )

        self._fill_negative_q(q=self._state.mixing_ratio.convective_ice, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.convective_liquid, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.graupel, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.large_scale_ice, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.large_scale_liquid, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.rain, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.snow, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)
        self._fill_negative_q(q=self._state.mixing_ratio.vapor, dqdt=self._locals._dummy_field_no_read_no_write, mass=self._locals.mass, fill_dqdt=False)

        self._update_microphysics_tendencies(
            dcloud_fraction_dt_micro=self._state.tendencies.dcloud_fraction_dt_micro,
            dgraupel_dt_micro=self._state.tendencies.dgraupel_dt_micro,
            dice_dt_micro=self._state.tendencies.dice_dt_micro,
            dliquid_dt_micro=self._state.tendencies.dliquid_dt_micro,
            drain_dt_micro=self._state.tendencies.drain_dt_micro,
            dsnow_dt_micro=self._state.tendencies.dsnow_dt_micro,
            du_dt_micro=self._state.tendencies.du_dt_micro,
            dv_dt_micro=self._state.tendencies.dv_dt_micro,
            dt_dt_micro=self._state.tendencies.dt_dt_micro,
            dvapor_dt_micro=self._state.tendencies.dvapor_dt_micro,
            convective_cloud_fraction=self._state.cloud_fraction.convective,
            convective_ice=self._state.mixing_ratio.convective_ice,
            convective_liquid=self._state.mixing_ratio.convective_liquid,
            large_scale_cloud_fraction=self._state.cloud_fraction.large_scale,
            large_scale_ice=self._state.mixing_ratio.large_scale_ice,
            large_scale_liquid=self._state.mixing_ratio.large_scale_liquid,
            graupel=self._state.mixing_ratio.graupel,
            rain=self._state.mixing_ratio.rain,
            snow=self._state.mixing_ratio.snow,
            u=self._state.u,
            v=self._state.v,
            t=self._state.t,
            vapor=self._state.mixing_ratio.vapor,
        )
