from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, FORWARD
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils.basic_operations import add, copy, set_value

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.macrophysics import GFDL1MMacrophysics
from pyMoist.microphysics.GFDL_1M.microphysics.gfdl_mp_v3 import GFDLMPV3
from pyMoist.microphysics.GFDL_1M.optimization import get_optimization_config
from pyMoist.microphysics.GFDL_1M.setup import GFDL1MSetup
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.saturation_tables import get_saturation_vapor_pressure_table
from pyMoist.shared.cloud_processes import redistribute_clouds_function, melt_freeze, fix_up_clouds
from pyMoist.shared.atmos_recipes import fill_negative_q, dissipative_kinetic_energy_heating
from pyMoist.shared.radiation_coupling import radiation_coupling_scale_aware


def compute_ice_water_path(
    convective_ice: FloatField,
    large_scale_ice: FloatField,
    graupel: FloatField,
    snow: FloatField,
    mass: FloatField,
    ice_water_path: FloatFieldIJ,
):
    """Compute ice water path throughout a column.

    The ice water path is calculated as the sum of convective ice, large scale ice,
    snow, and graupel, each multiplied by the mass, integrated over the vertical column.

    Args:
        convective_ice (FloatField)
        large_scale_ice (FloatField)
        graupel (FloatField)
        snow (FloatField)
        mass (FloatField)
        ice_water_path (FloatFieldIJ)
    """
    with computation(FORWARD), interval(0, 1):
        # ensure calculation starts at 0
        ice_water_path = 0.0

    with computation(FORWARD), interval(...):
        ice_water_path += (convective_ice + large_scale_ice + snow + graupel) * mass


def compute_liquid_water_path(
    convective_liquid: FloatField,
    large_scale_liquid: FloatField,
    rain: FloatField,
    mass: FloatField,
    liquid_water_path: FloatFieldIJ,
):
    """Compute liquid water path throughout a column.

    The liquid water path is calculated as the sum of convective liquid, large scale liquid,
    and rain, each multiplied by the mass, integrated over the vertical column.

    Args:
        convective_liquid (FloatField)
        large_scale_liquid (FloatField)
        rain (FloatField)
        mass (FloatField)
        liquid_water_path (FloatFieldIJ)
    """
    with computation(FORWARD), interval(0, 1):
        # ensure calculation starts at 0
        liquid_water_path = 0.0

    with computation(FORWARD), interval(...):
        liquid_water_path += (convective_liquid + large_scale_liquid + rain) * mass


def convert_mm_per_day_to_kg_per_m2_per_s(field: FloatField):
    """Convert a field from mm day^-1 to kg m^-2 s^-1

    Args:
        field (FloatField)
    """
    with computation(PARALLEL), interval(...):
        field = field * 1.0e-3 / 86400.0


def flip_sign(input: FloatField, output: FloatField):
    with computation(PARALLEL), interval(...):
        output = -1.0 * input


def min_with_one(field: FloatField):
    with computation(PARALLEL), interval(...):
        field = min(field, 1.0)


def negative_adjustment(
    graupel: FloatField,
    ice: FloatField,
    liquid: FloatField,
    rain: FloatField,
    snow: FloatField,
    vapor: FloatField,
):
    with computation(FORWARD), interval(0, 1):
        # initialize 2d internals
        n_solid_fix = 0
        n_liquid_fix = 0
        n_vapor_col = 0

    # 1. Local phase borrowing (GFDL neg_adj style)
    with computation(PARALLEL), interval(...):
        # Solid phase adjustments
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

        # Liquid phase adjustments
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

    # 2. Column water vapor borrowing
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

    # Final safety clamp
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
    """Compute cloud fraction for radiation, redistribute clouds, and update the primary model state graupel/rain/snow/vapor

    Args:
        convective_cloud_fraction (FloatField): _description_
        convective_ice (FloatField): _description_
        convective_liquid (FloatField): _description_
        dcloud_fraction_dt (FloatField): _description_
        graupel (FloatField): _description_
        large_scale_cloud_fraction (FloatField): _description_
        large_scale_ice (FloatField): _description_
        large_scale_liquid (FloatField): _description_
        radiation_cloud_fraction (FloatField): _description_
        radiation_graupel (FloatField): _description_
        radiation_ice (FloatField): _description_
        radiation_liquid (FloatField): _description_
        radiation_rain (FloatField): _description_
        radiation_snow (FloatField): _description_
        radiation_vapor (FloatField): _description_
        rain (FloatField): _description_
        snow (FloatField): _description_
        t (FloatField): _description_
        vapor (FloatField): _description_
    """
    from __externals__ import DT

    with computation(PARALLEL), interval(...):
        # 1. Update cloud fraction
        if radiation_liquid + radiation_ice > 0.0:
            radiation_cloud_fraction = min(1.0, max(0.0, radiation_cloud_fraction + dcloud_fraction_dt * DT))
        else:
            radiation_cloud_fraction = 0.0

        # 2. Redistribute clouds
        redistribute_clouds_function(
            radiation_cloud_fraction,
            convective_cloud_fraction,
            convective_ice,
            convective_liquid,
            radiation_graupel,
            radiation_ice,
            large_scale_cloud_fraction,
            large_scale_ice,
            large_scale_liquid,
            radiation_liquid,
            radiation_rain,
            radiation_snow,
            t,
            radiation_vapor,
        )

        # 3. Fill vapor/rain/snow/graupel state
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
    """Fill output precipitation diagnostic fields

    Args:
        freezing_rainfall (FloatFieldIJ)
        icefall (FloatFieldIJ)
        non_anvil_large_scale_precip (FloatFieldIJ)
        non_anvil_large_scale_snow (FloatFieldIJ)
        precip_at_surface_graupel (FloatFieldIJ)
        precip_at_surface_ice (FloatFieldIJ)
        precip_at_surface_rain (FloatFieldIJ)
        precip_at_surface_snow (FloatFieldIJ)
        precip_at_surface_water (FloatFieldIJ)
    """
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
    """Resistribute precipitation fluxes for chemistry

    MUST be build with K_INTERFACE_DIM to work properly

    Args:
        anvil_ice_precip_flux (FloatField)
        anvil_liquid_precip_flux (FloatField)
        non_anvil_large_scale_ice_precip_flux (FloatField)
        non_anvil_large_scale_liquid_precip_flux (FloatField)
    """
    with computation(PARALLEL), interval(...):
        tmp_val = min(1.0, max(convective_liquid / max(radiation_liquid, 1.0e-8), 0.0))
        anvil_liquid_precip_flux = (non_anvil_large_scale_ice_precip_flux + non_anvil_large_scale_rain_precip_flux) * tmp_val
        non_anvil_large_scale_ice_precip_flux = (non_anvil_large_scale_ice_precip_flux + non_anvil_large_scale_rain_precip_flux) - anvil_liquid_precip_flux

        tmp_val = min(1.0, max(convective_ice / max(radiation_ice, 1.0e-8), 0.0))
        anvil_ice_precip_flux = (non_anvil_large_scale_liquid_precip_flux + non_anvil_large_scale_snow_precip_flux + non_anvil_large_scale_graupel_precip_flux) * tmp_val
        non_anvil_large_scale_liquid_precip_flux = (
            non_anvil_large_scale_liquid_precip_flux + non_anvil_large_scale_snow_precip_flux + non_anvil_large_scale_graupel_precip_flux
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


class GFDL1M(NDSLRuntime):
    """
    GFDL Single Moment microphysics

    The primary purpose of this code is to compute macro/microphysical tendencies to be applied to state
    variables (p, t, wind, etc.). This code requires all fields to be preloaded with Fortran memory or
    otherwise supplied between the __init__ and __call__ steps.

    Performs the following functions to achieve this goal:
    __init__
        - initialize saturation vapor pressure tables, intialize temporary/output fields, construct stencils
        Arguments: StencilFactory, QuantityFactory, GFDL1MConfig

    __call__
        - setup: compute additional required fields, create pristine copies of input variables
        - phase_change: create new condensates, perform phase change operations
        - driver: precipitate condensates
        - finalize: compute tendencies, prepare fields to be returned to the larger model
        Arguments: none (data needs to be pre-loaded)
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
    ):
        super().__init__(stencil_factory, get_optimization_config(stencil_factory))

        # WARNING - to be removed when 11.10.1 update is complete
        ndsl_log.warning(
            "pyMoist.GFDL_1M: This NDSL version of GFDL_1M was ported from GEOS v11.8.1, and has not yet been updated to v11.10. "
            "Stable execution is not guarenteed, as v11.10 made siginificant changes to the source Fortran."
        )

        # initialize saturation tables
        saturation_tables = get_saturation_vapor_pressure_table(stencil_factory)

        # initialize locals
        self._locals = GFDL1MLocals.make_locals(quantity_factory)

        # make config visible at runtime
        self._config = config

        # build subcomponents and stencils
        self._setup = GFDL1MSetup(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            config=config,
            saturation_tables=saturation_tables,
        )

        self._macrophysics = GFDL1MMacrophysics(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            config=config,
            saturation_tables=saturation_tables,
        )

        self._add = stencil_factory.from_dims_halo(
            func=add,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._post_microphysics_calculations = stencil_factory.from_dims_halo(
            func=post_microphysics_calculations,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DT": config.DT},
        )

        self._compute_ice_water_path = stencil_factory.from_dims_halo(
            func=compute_ice_water_path,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._compute_liquid_water_path = stencil_factory.from_dims_halo(
            func=compute_liquid_water_path,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._convert_mm_per_day_to_kg_per_m2_per_s = stencil_factory.from_dims_halo(
            func=convert_mm_per_day_to_kg_per_m2_per_s,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._convert_mm_per_day_to_kg_per_m2_per_s_interface = stencil_factory.from_dims_halo(
            func=convert_mm_per_day_to_kg_per_m2_per_s,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._copy = stencil_factory.from_dims_halo(
            func=copy,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._dissipative_kinetic_energy_heating = stencil_factory.from_dims_halo(
            func=dissipative_kinetic_energy_heating,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._fill_negative_q = stencil_factory.from_dims_halo(
            func=fill_negative_q,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._fix_up_clouds = stencil_factory.from_dims_halo(
            func=fix_up_clouds,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._flip_sign = stencil_factory.from_dims_halo(
            func=flip_sign,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._melt_freeze = stencil_factory.from_dims_halo(
            func=melt_freeze,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DT": config.DT},
        )

        self._min_with_one = stencil_factory.from_dims_halo(
            func=min_with_one,
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
        )

        self._redistribute_precipitation_fluxes = stencil_factory.from_dims_halo(
            func=redistribute_precipitation_fluxes,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value_k_interface = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._update_microphysics_tendencies = stencil_factory.from_dims_halo(
            func=update_microphysics_tendencies,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._gfdl_microphysics_v3 = GFDLMPV3()

    def __call__(
        self,
        state: GFDL1MState,
    ):
        # miscellaneous setup required for macro and/or microphysics schemes
        self._setup(
            state=state,
            locals=self._locals,
        )

        # --------------------------------------------------
        # MACROPHYSICS
        # --------------------------------------------------
        # compute macrophysical tendencies, use the hydrostatic pdf to distribute particles,
        # then melt, freeze, and evaporate, all according to options defined in namelist
        self._macrophysics(
            state=state,
            locals=self._locals,
        )

        # print a debug warning, if any non-physical values are identified
        if self._config.DEBUG_TQ_ERRORS:
            option_not_implemented = True

        # --------------------------------------------------
        # MICROPHYSICS
        # --------------------------------------------------
        self._copy(input=state.mixing_ratio.vapor, output=state.tendencies.dvapor_dt_micro)
        self._add(summand_1=state.mixing_ratio.convective_ice, summand_2=state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dice_dt_micro)
        self._add(summand_1=state.mixing_ratio.convective_liquid, summand_2=state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dliquid_dt_micro)
        self._add(summand_1=state.cloud_fraction.convective, summand_2=state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.tendencies.dcloud_fraction_dt_micro)
        self._copy(input=state.mixing_ratio.graupel, output=state.tendencies.dgraupel_dt_micro)
        self._copy(input=state.mixing_ratio.rain, output=state.tendencies.drain_dt_micro)
        self._copy(input=state.mixing_ratio.snow, output=state.tendencies.dsnow_dt_micro)
        self._copy(input=state.t, output=state.tendencies.dt_dt_micro)
        self._copy(input=state.u, output=state.tendencies.du_dt_micro)
        self._copy(input=state.v, output=state.tendencies.dv_dt_micro)

        # delta-z layer thickness (gfdl mp v3 expects this to be negative)
        self._flip_sign(input=self._locals.layer_thickness, output=self._locals.layer_thickness_negative)

        # zero out GFDLMPV3 outputs
        self._set_value(field=self._locals.dcloud_fractiondt, value=Float(0.0))
        self._set_value_k_interface(fleid=state.non_anvil_large_scale.ice_precip_flux, value=Float(0.0))
        self._set_value_k_interface(fleid=state.non_anvil_large_scale.liquid_precip_flux, value=Float(0.0))

        # cloud fractions and condensates for radiation
        self._add(summand_1=state.cloud_fraction.convective, summand_2=state.cloud_fraction.large_scale, sum=self._locals.temporary_3d)
        self._min_with_one(field=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.radiation_field.cloud_fraction)
        self._add(summand_1=state.mixing_ratio.convective_liquid, summand_2=state.mixing_ratio.large_scale_liquid, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.radiation_field.liquid)
        self._add(summand_1=state.mixing_ratio.convective_ice, summand_2=state.mixing_ratio.large_scale_ice, sum=self._locals.temporary_3d)
        self._copy(input=self._locals.temporary_3d, output=state.radiation_field.ice)
        self._copy(input=state.mixing_ratio.vapor, output=state.radiation_field.vapor)
        self._copy(input=state.mixing_ratio.graupel, output=state.radiation_field.graupel)
        self._copy(input=state.mixing_ratio.rain, output=state.radiation_field.rain)
        self._copy(input=state.mixing_ratio.snow, output=state.radiation_field.snow)

        if self._config.GFDL_MP3:
            # run the driver
            self._gfdl_microphysics_v3(state=state, locals=self._locals)

            # convert evap/subl/cloud/precipitation flux exports from (mm/day) to (kg m-2 s-1)
            self._convert_mm_per_day_to_kg_per_m2_per_s(state.non_anvil_large_scale.evaporation)
            self._convert_mm_per_day_to_kg_per_m2_per_s(state.non_anvil_large_scale.sublimation)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.ice_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.liquid_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.rain_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.snow_precip_flux)
            self._convert_mm_per_day_to_kg_per_m2_per_s_interface(state.non_anvil_large_scale.graupel_precip_flux)

            if self._config.DO_GFDL_REFLECTIVITY:
                option_not_impemented = True

        if self._config.REPORT_GFDL_1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] negative_adjustment requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )
            self._negative_adjustment(
                graupel=state.radiation_field.graupel,
                ice=state.radiation_field.ice,
                liquid=state.radiation_field.liquid,
                rain=state.radiation_field.rain,
                snow=state.radiation_field.snow,
                vapor=state.radiation_field.vapor,
            )
        else:
            self._negative_adjustment(
                graupel=state.radiation_field.graupel,
                ice=state.radiation_field.ice,
                liquid=state.radiation_field.liquid,
                rain=state.radiation_field.rain,
                snow=state.radiation_field.snow,
                vapor=state.radiation_field.vapor,
            )

        # update cloud fraction, redistribute clouds, and fill vapor/precip states
        self._post_microphysics_calculations(
            cloud_fraction=state.radiation_field.cloud_fraction,
            dcloud_fraction_dt=self._locals.dcloud_fraction_dt,
            ice=state.radiation_field.ice,
            liquid=state.radiation_field.liquid,
        )

        if self._config.REPORT_GFDL_1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] fill_negative_q requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )
            self._fill_negative_q(q=state.mixing_ratio.convective_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.convective_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.graupel, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.rain, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.snow, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.vapor, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
        else:
            self._fill_negative_q(q=state.mixing_ratio.convective_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.convective_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.graupel, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.rain, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.snow, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.vapor, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)

        # 2D precip diagnostics
        self._precip_diagnostics(
            freezing_rainfall=state.freezing_rainfall,
            icefall=state.icefall,
            non_anvil_large_scale_precip=state.non_anvil_large_scale.precip,
            non_anvil_large_scale_snow=state.non_anvil_large_scale.snow,
            precip_at_surface_graupel=state.precipitation_at_surface.graupel,
            precip_at_surface_ice=state.precipitation_at_surface.ice,
            precip_at_surface_rain=state.precipitation_at_surface.rain,
            precip_at_surface_snow=state.precipitation_at_surface.snow,
            precip_at_surface_water=state.precipitation_at_surface.water,
        )

        # Flux Redistribution
        self._redistribute_precipitation_fluxes(
            anvil_ice_precip_flux=state.anvil.ice_precip_flux,
            anvil_liquid_precip_flux=state.anvil.liquid_precip_flux,
            convective_ice=state.mixing_ratio.convective_ice,
            convective_liquid=state.mixing_ratio.convective_liquid,
            non_anvil_large_scale_graupel_precip_flux=state.non_anvil_large_scale.graupel_precip_flux,
            non_anvil_large_scale_ice_precip_flux=state.non_anvil_large_scale.ice_precip_flux,
            non_anvil_large_scale_liquid_precip_flux=state.non_anvil_large_scale.liquid_precip_flux,
            non_anvil_large_scale_rain_precip_flux=state.non_anvil_large_scale.rain_precip_flux,
            non_anvil_large_scale_snow_precip_flux=state.non_anvil_large_scale.snow_precip_flux,
            radiation_ice=state.radiation_field.ice,
            radiation_liquid=state.radiation_field.liquid,
        )

        # MeltFreeze, RadCouple
        if self._config.MELTFRZ_CLDMICRO:
            self._melt_freeze(
                convection_fraction=state.convection_fraction,
                surface_type=state.surface_type,
                t=state.t,
                liquid=state.mixing_ratio.convective_liquid,
                ice=state.mixing_ratio.convective_ice,
            )
            self._melt_freeze(
                convection_fraction=state.convection_fraction,
                surface_type=state.surface_type,
                t=state.t,
                liquid=state.mixing_ratio.large_scale_liquid,
                ice=state.mixing_ratio.large_scale_ice,
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
                lid_level=self._locals.lid_level,
            )

        # Get radiative properties
        self._radiation_coupling_scale_aware(
            t=state.t,
            p_mb=self._locals.p_mb,
            cloud_particle_effective_radius_liquid=state.cloud_particle_effective_radius.liquid,
            cloud_particle_effective_radius_ice=state.cloud_particle_effective_radius.ice,
            concentration_liquid=state.concentration.liquid,
            radiation_cloud_fraction=state.radiation_field.cloud_fraction,
            radiation_graupel=state.radiation_field.graupel,
            radiation_ice=state.radiation_field.ice,
            radiation_liquid=state.radiation_field.liquid,
            radiation_rain=state.radiation_field.rain,
            radiation_snow=state.radiation_field.snow,
            radiation_vapor=state.radiation_field.vapor,
            anvil_cloud_fraction=state.cloud_fraction.convective,
            anvil_ice=state.mixing_ratio.convective_ice,
            anvil_liquid=state.mixing_ratio.convective_liquid,
            large_scale_cloud_fraction=state.cloud_fraction.large_scale,
            large_scale_ice=state.mixing_ratio.large_scale_ice,
            large_scale_liquid=state.mixing_ratio.large_scale_liquid,
            graupel=state.mixing_ratio.graupel,
            rain=state.mixing_ratio.rain,
            snow=state.mixing_ratio.snow,
            vapor=state.mixing_ratio.vapor,
        )

        if self._config.REPORT_GFDL_1M_NEGATIVES:
            ndsl_log.warning(
                "[GFDL_1M] fill_negative_q requested with debug output, but NDSL does not support in-stencil prints. "
                "The math has still been applied as expected, please use NDSL debug tools to see requested values"
            )
            self._fill_negative_q(q=state.mixing_ratio.convective_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.convective_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.graupel, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.rain, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.snow, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.vapor, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
        else:
            self._fill_negative_q(q=state.mixing_ratio.convective_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.convective_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.graupel, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_ice, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.large_scale_liquid, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.rain, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.snow, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)
            self._fill_negative_q(q=state.mixing_ratio.vapor, dqdt=state.fill_negative_tendency_cloud_micro.vapor, mass=self._locals.mass, fill_dqdt=True)

        self._update_microphysics_tendencies(
            dcloud_fraction_dt_micro=state.tendencies.dcloud_fraction_dt_micro,
            dgraupel_dt_micro=state.tendencies.dgraupel_dt_micro,
            dice_dt_micro=state.tendencies.dice_dt_micro,
            dliquid_dt_micro=state.tendencies.dliquid_dt_micro,
            drain_dt_micro=state.tendencies.drain_dt_micro,
            dsnow_dt_micro=state.tendencies.dsnow_dt_micro,
            du_dt_micro=state.tendencies.du_dt_micro,
            dv_dt_micro=state.tendencies.dv_dt_micro,
            dt_dt_micro=state.tendencies.dt_dt_micro,
            dvapor_dt_micro=state.tendencies.dvapor_dt_micro,
            convective_cloud_fraction=state.cloud_fraction.convective,
            convective_ice=state.mixing_ratio.convective_ice,
            convective_liquid=state.mixing_ratio.convective_liquid,
            large_scale_cloud_fraction=state.cloud_fraction.large_scale,
            large_scale_ice=state.mixing_ratio.large_scale_ice,
            large_scale_liquid=state.mixing_ratio.large_scale_liquid,
            graupel=state.mixing_ratio.graupel,
            rain=state.mixing_ratio.rain,
            snow=state.mixing_ratio.snow,
            u=state.u,
            v=state.v,
            t=state.t,
            vapor=state.mixing_ratio.vapor,
        )

        # --------------------------------------------------
        # DIAGNOSTICS
        # --------------------------------------------------
        if state.large_scale_rainwater_source is not None:
            self._add(summand_1=state.tendencies.drain_dt_macro, summand_2=state.tendencies.drain_dt_micro, sum=state.large_scale_rainwater_source)

        if state.tendencies.dt_dt_friction_pressure_weighted is not None:
            self._add(summand_1=state.tendencies.du_dt_macro, summand_2=state.tendencies.du_dt_micro, sum=self._locals.temporary_3d)
            self._add(summand_1=state.tendencies.dv_dt_macro, summand_2=state.tendencies.dv_dt_micro, sum=self._locals.temporary_3d_2)
            self._dissipative_kinetic_energy_heating(
                mass=self._locals.mass,
                u=state.u,
                v=state.v,
                du=self._locals.temporary_3d,
                dv=self._locals.temporary_3d_2,
                dt=state.tendencies.dt_dt_friction_pressure_weighted,
            )

        # call the shared radar diagnostics routine
        option_not_implemented = True

        if state.mass_fraction.suspended_graupel is not None:
            self._copy(input=state.mixing_ratio.graupel, output=state.mass_fraction.suspended_graupel)
        if state.mass_fraction.suspended_rain is not None:
            self._copy(input=state.mixing_ratio.rain, output=state.mass_fraction.suspended_rain)
        if state.mass_fraction.suspended_snow is not None:
            self._copy(input=state.mixing_ratio.snow, output=state.mass_fraction.suspended_snow)

        if state.ice_water_path is not None:
            self._compute_ice_water_path(
                convective_ice=state.mixing_ratio.convective_ice,
                large_scale_ice=state.mixing_ratio.large_scale_ice,
                graupel=state.mixing_ratio.graupel,
                snow=state.mixing_ratio.snow,
                mass=self._locals.mass,
                ice_water_path=state.ice_water_path,
            )

        if state.liquid_water_path is not None:
            self._compute_liquid_water_path(
                convective_liquid=state.mixing_ratio.convective_liquid,
                large_scale_liquid=state.mixing_ratio.large_scale_liquid,
                rain=state.mixing_ratio.rain,
                mass=self._locals.mass,
                liquid_water_path=state.liquid_water_path,
            )
