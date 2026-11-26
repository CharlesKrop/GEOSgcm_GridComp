from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import PARALLEL, computation, interval, FORWARD
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils.basic_operations import add, copy, set_value

from pyMoist.microphysics.gfdl_1m.config import GFDL1MConfig
from pyMoist.microphysics.gfdl_1m.locals import GFDL1MLocals
from pyMoist.microphysics.gfdl_1m.macrophysics import GFDL1MMacrophysics
from pyMoist.microphysics.gfdl_1m.gfdl_mp_v3.gfdl_mp_v3 import GFDLMPV3
from pyMoist.microphysics.gfdl_1m.optimization import get_optimization_config
from pyMoist.microphysics.gfdl_1m.setup import GFDL1MSetup
from pyMoist.microphysics.gfdl_1m.state import GFDL1MState
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
        # One millimeter of liquid water is one kg per square meter.
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

    Must be built with K_INTERFACE_DIM to function properly.

    Args:
        anvil_ice_precip_flux (FloatField)
        anvil_liquid_precip_flux (FloatField)
        non_anvil_large_scale_ice_precip_flux (FloatField)
        non_anvil_large_scale_liquid_precip_flux (FloatField)
    """
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
    """GFDL single-moment microphysics composed from four runtime components."""

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: GFDL1MConfig,
    ):
        super().__init__(stencil_factory, get_optimization_config(stencil_factory))

        saturation_tables = get_saturation_vapor_pressure_table(stencil_factory)
        self._locals = GFDL1MLocals.empty(quantity_factory)
        self._config = config

        # Imported here to avoid a module cycle: the component modules reuse stencil
        # functions defined above in this file.
        from pyMoist.microphysics.gfdl_1m.diagnostics import GFDL1MDiagnostics
        from pyMoist.microphysics.gfdl_1m.microphysics import GFDL1MMicrophysics

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
            lid_level=self._setup.lid_level,
        )
        self._microphysics = GFDL1MMicrophysics(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            config=config,
            locals=self._locals,
            lid_level=self._setup.lid_level,
        )
        self._diagnostics = GFDL1MDiagnostics(
            stencil_factory=stencil_factory,
            locals=self._locals,
        )

    def __call__(self, state: GFDL1MState):
        self._setup(state=state, locals=self._locals)
        self._macrophysics(state=state, locals=self._locals)
        self._microphysics(state=state)
        self._diagnostics(state=state)
