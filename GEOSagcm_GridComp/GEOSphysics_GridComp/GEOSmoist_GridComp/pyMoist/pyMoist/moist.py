from pyMoist.state import MoistState
from pyMoist.convection_tracers import ConvectionTracers
from ndsl.stencils import set_value, copy, add, divide, add_to_self, add_to_self_2d, multiply_2d, subtract_2d, copy_2d, set_value_2D
from pyMoist.saturation_tables import get_saturation_vapor_pressure_table, compute_saturation_specific_humidity, GlobalTable_saturation_tables
from ndsl import StencilFactory, QuantityFactory, NDSLRuntime
from ndsl.dsl.gt4py import computation, PARALLEL, interval, FORWARD, K, BACKWARD
from ndsl.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK, Float, IntFieldIJ
import pyMoist.constants as constants
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyMoist.shared.incloud_processes import fix_mixing_ratio, buoyancy_1, Buoyancy2
from pyMoist.locals import MoistLocals
from pyMoist.config import MoistConfig
from pyMoist.convection import UWConfiguration, UWState, ComputeUwshcuInv, GF2020, GF2020Config, GF2020State, GF2020CumulusParameterizationConfig
from pyMoist.microphysics import GFDL1M, GFDL1MConfig, GFDL1MState


def get_alarm():
    pass


def turn_alarm_off():
    pass


def set_surface_type(
    land_fraction: FloatFieldIJ,
    land_ice_fraction: FloatFieldIJ,
    ice_covered_fraction_of_tile: FloatFieldIJ,
    snow_mass: FloatFieldIJ,
    surface_type: FloatField,
):
    with computation(FORWARD), interval(0, 1):
        if land_ice_fraction > 0.5 or ice_covered_fraction_of_tile > 0.5:
            surface_type = 3.0  # ice
        elif snow_mass > 0.1 and snow_mass != constants.MAPL_UNDEF:
            surface_type = 2.0  # snow
        elif land_fraction > 0.1:
            surface_type = 1.0  # land
        else:
            surface_type = 0.0  # ocean


def compute_derived_state(
    mass: FloatField,
    p_interface: FloatField,
    p_interface_mb: FloatField,
    p_mb: FloatField,
    p_kappa_interface: FloatField,
    p_kappa: FloatField,
    z_interface: FloatField,
    edge_height_above_surface: FloatField,
    layer_height_above_surface: FloatField,
    layer_thickness: FloatField,
    t: FloatField,
    ese: GlobalTable_saturation_tables,
    esx: GlobalTable_saturation_tables,
    saturation_specific_humidity: FloatField,
    dsaturation_specific_humidity: FloatField,
):
    from __externals__ import k_end

    mass = (p_interface[0, 0, 1] - p_interface) / constants.MAPL_GRAV

    p_interface_mb = p_interface * 0.01
    p_kappa_interface = (p_interface / constants.MAPL_P00) ** constants.MAPL_KAPPA
    p_mb = 0.5 * (p_interface_mb + p_interface_mb[0, 0, 1])
    p_kappa = (100.0 * p_mb / constants.MAPL_P00) ** constants.MAPL_KAPPA

    edge_height_above_surface = z_interface - z_interface.at(K=k_end)
    layer_height_above_surface = 0.5 * (edge_height_above_surface + edge_height_above_surface[0, 0, 1])
    layer_thickness = edge_height_above_surface - edge_height_above_surface[0, 0, 1]

    saturation_specific_humidity, dsaturation_specific_humidity = compute_saturation_specific_humidity(t=t, p=p_mb, ese=ese, esx=esx)


def find_highest_level_interface(
    scalar_diffusivity: FloatField,
    value: Float,
    highest_level: IntFieldIJ,
):
    """Find the highest level where a field is greater than a specified value.

    This stencil must be build with K_INTERFACE_DIM to function properly.

    Args:
        field (FloatField): field to be analyzed
        value (Float): threshold for comparison
        highest_level (IntFieldIJ): highest level where field is greater than value
    """
    with computation(FORWARD), interval(0, 1):
        highest_level = -1

    with computation(FORWARD), interval(...):
        if highest_level == -1 and scalar_diffusivity > value:
            highest_level = K


def find_lowest_level_interface(
    scalar_diffusivity: FloatField,
    value: Float,
    lowest_level: IntFieldIJ,
):
    """Find the lowest level where a field is greater than a specified value.

    This stencil must be build with K_INTERFACE_DIM to function properly.

    Args:
        field (FloatField): field to be analyzed
        value (Float): threshold for comparison
        lowest_level (IntFieldIJ): lowest level where field is greater than value
    """
    with computation(FORWARD), interval(0, 1):
        lowest_level = -1

    with computation(BACKWARD), interval(...):
        if lowest_level == -1 and scalar_diffusivity > value:
            lowest_level = K


def export_cbl_level(
    cbl_level_before_moist: IntFieldIJ,
    pbl_level: IntFieldIJ,
    reference_pressure: FloatFieldK,
    p_min_cbl: Float,
):
    from __externals__ import k_end

    with computation(FORWARD), interval(0, 1):
        min_cbl_level: IntFieldIJ = 0

    with computation(FORWARD), interval(...):
        if reference_pressure < p_min_cbl:
            min_cbl_level += 1

    with computation(FORWARD), interval(0, 1):
        if pbl_level != 0:
            cbl_level_before_moist = max(min(pbl_level + 1, k_end - 1), 1)
        else:
            cbl_level_before_moist = k_end - 1
        cbl_level_before_moist = max(cbl_level_before_moist, min_cbl_level)


def compute_convection_fraction(convection_fraction: FloatFieldIJ, cape: FloatFieldIJ):
    from __externals__ import CONVECTION_FRACTION_MAX, CONVECTION_FRACTION_MIN, CONVECTION_FRACTION_EXP

    with computation(FORWARD), interval(0, 1):
        convection_fraction = 0.0
        if CONVECTION_FRACTION_MAX > CONVECTION_FRACTION_MIN:
            if cape != constants.MAPL_UNDEF:
                convection_fraction = max(1.0e-6, min(1.0, (cape - convection_fraction) / (CONVECTION_FRACTION_MAX - CONVECTION_FRACTION_MIN)))

        if CONVECTION_FRACTION_EXP != 1.0:
            convection_fraction = convection_fraction**CONVECTION_FRACTION_


def initialize_convection_tracers():
    pass


def export_concentration(
    input: FloatField,
    factor: Float,
    output: FloatField,
):
    with computation(PARALLEL), interval(...):
        field = field * factor


def update_cloud_fraction(
    total_cloud_fraction: FloatField,
    convective_cloud_fraction: FloatField,
    convective_desired_phase: FloatField,
    convective_other_phase: FloatField,
    large_scale_cloud_fraction: FloatField,
    large_scale_desired_phase: FloatField,
    large_scale_other_phase: FloatField,
):
    with computation(PARALLEL), interval(...):
        total_cloud_fraction = 0.0

    with computation(PARALLEL), interval(...):
        if convective_desired_phase + large_scale_desired_phase > 1.0e-12:
            total_cloud_fraction = (
                (convective_cloud_fraction + large_scale_cloud_fraction)
                * (convective_desired_phase + large_scale_desired_phase)
                / (convective_desired_phase + convective_other_phase + large_scale_desired_phase + large_scale_other_phase)
            )
        total_cloud_fraction


def get_saturation_specific_humidity(
    t: FloatField,
    p_mb: FloatField,
    saturation_specific_humidity: FloatField,
    dsaturation_specific_humidity: FloatField,
    ese: GlobalTable_saturation_tables,
    esx: GlobalTable_saturation_tables,
):
    with computation(PARALLEL), interval(...):
        saturation_specific_humidity, dsaturation_specific_humidity = compute_saturation_specific_humidity(t=t, p=p_mb, ese=ese, esx=esx)


def export_relative_humidity_wrt_ice(relative_humidity_wrt_ice: FloatField, t: FloatField, specific_humidity: FloatField, saturation_specific_humidity: FloatField):
    with computation(PARALLEL), interval(...):
        relative_humidity_wrt_ice = specific_humidity / saturation_specific_humidity

        if t > constants.MAPL_TICE:
            relative_humidity_wrt_ice = 0.0


def export_output_saturation_ratio(
    saturation_ratio: FloatField,
    large_scale_ice_cloud_fraction: FloatField,
    specific_humidity: FloatField,
    saturation_specific_humidity: FloatField,
):
    with computation(PARALLEL), interval(...):
        if large_scale_ice_cloud_fraction < 0.99 and saturation_specific_humidity > 1.0e-20:
            numerator = max((specific_humidity - saturation_specific_humidity * large_scale_ice_cloud_fraction), 0.0) / (1.0 - large_scale_ice_cloud_fraction)
            saturation_ratio = min(numerator / saturation_ratio, 2.0)
        else:
            saturation_ratio = 1.0


def export_relative_humidity_wrt_liquid():
    pass


def rain_out_excessive_rh(
    t: FloatField,
    specific_humidity: FloatField,
    saturation_specific_humidity: FloatField,
    dsaturation_specific_humidity: FloatField,
    mass: FloatField,
    rain_from_large_scale_nonanvil: FloatFieldIJ,
    spurious_rain_from_relative_humidity_cleanup: FloatFieldIJ,
    dt_dt_from_rh_cleanup: FloatField,
    dspecific_humidity_dt_from_rh_cleanup: FloatField,
):
    from __externals__ import DT_MOIST

    with computation(PARALLEL), interval(...):
        if specific_humidity > 1.1 * saturation_specific_humidity:
            excess_water = (specific_humidity - 1.1 * saturation_specific_humidity) / (
                1.0 + 1.1 * dsaturation_specific_humidity * constants.MAPL_ALHL / constants.MAPL_CP
            )
        else:
            excess_water = 0.0

    with computation(FORWARD), interval(...):
        spurious_rain_from_relative_humidity_cleanup = (excess_water * mass) / DT_MOIST

    with computation(FORWARD), interval(0, 1):
        rain_from_large_scale_nonanvil = rain_from_large_scale_nonanvil + spurious_rain_from_relative_humidity_cleanup

    with computation(PARALLEL), interval(...):
        t = t + (constants.MAPL_ALHL / constants.MAPL_CP) * excess_water
        specific_humidity = specific_humidity - spurious_rain_from_relative_humidity_cleanup
        dt_dt_from_rh_cleanup = (t - dt_dt_from_rh_cleanup) / DT_MOIST
        dspecific_humidity_dt_from_rh_cleanup = (specific_humidity - dspecific_humidity_dt_from_rh_cleanup) / DT_MOIST


def divide_by_dt_moist_2d(input: FloatFieldIJ, output: FloatFieldIJ):
    from __external__ import DT_MOIST

    with computation(FORWARD), interval(0, 1):
        output = input / DT_MOIST


def multiply_by_dt_moist_2d(input: FloatFieldIJ, output: FloatFieldIJ):
    from __external__ import DT_MOIST

    with computation(FORWARD), interval(0, 1):
        output = input * DT_MOIST


def update_dlayer_pressure_thickness_dt(dlayer_pressure_thickness_dt: FloatField, field: FloatField):
    with computation(PARALLEL), interval(...):
        dlayer_pressure_thickness_dt = field - field[0, 0, 1]


def ensure_non_negative_2d(field: FloatFieldIJ):
    with computation(FORWARD), interval(0, 1):
        field = max(field, 0.0)


def get_Kuchera_ratios(p_mb: FloatField, t: FloatField, kuchera_ratio: FloatFieldIJ):
    with computation(FORWARD), interval(0, 1):
        t_max: FloatFieldIJ = 0.0

    with computation(BACKWARD), interval(...):
        if p_mb > 500.0:
            t_max = max(t_max, t)

    with computation(FORWARD), interval(0, 1):
        if t_max <= 271.16:
            kuchera_ratio = 12.0 + (271.16 - t_max)
        else:
            kuchera_ratio = 12.0 + 2 * (271.16 - t_max)


def compute_snowfall_total(snowfall_total: FloatFieldIJ, snowfall: FloatFieldIJ, icefall: FloatFieldIJ, kuchera_ratio: FloatFieldIJ):
    from __externals__ import DT_MOSIT

    with computation(FORWARD), interval(0, 1):
        snowfall_total = kuchera_ratio * DT_MOSIT * (snowfall + icefall)


def compute_dry_static_energy(t: FloatField, layer_height_above_surface: FloatField, edge_height_above_surface: FloatField, dry_static_energy: FloatField):
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        dry_static_energy = constants.MAPL_CP * t + constants.MAPL_GRAV * (layer_height_above_surface - edge_height_above_surface.at(K=k_end))


def compute_relative_humidty(
    t: FloatField, p_mb: FloatField, specific_humidity: FloatField, relative_humidity: FloatField, ese: GlobalTable_saturation_tables, esx: GlobalTable_saturation_tables
):
    with computation(PARALLEL), interval(...):
        saturation_specific_humidity, dsaturation_specific_humidity = get_saturation_specific_humidity(t=t, p_mb=p_mb, ese=ese, esx=esx)
        relative_humidity = max(min(specific_humidity / saturation_specific_humidity, 1.02), 0.0)


def compute_condensed_water_path(
    convective_ice: FloatField,
    convective_liquid: FloatField,
    large_scale_ice: FloatField,
    large_scale_liquid: FloatField,
    mass: FloatField,
    condensed_water_path: FloatFieldIJ,
):
    with computation(FORWARD), interval(...):
        condensed_water_path += (convective_ice + convective_liquid + large_scale_ice + large_scale_liquid) * mass


def compute_liquid_water_path(convective_liquid: FloatField, large_scale_liquid: FloatField, mass: FloatField, liquid_water_path: FloatFieldIJ):
    with computation(FORWARD), interval(...):
        liquid_water_path += (convective_liquid + large_scale_liquid) * mass


def compute_ice_water_path(convective_ice: FloatField, large_scale_ice: FloatField, mass: FloatField, ice_water_path: FloatFieldIJ):
    with computation(FORWARD), interval(...):
        ice_water_path += (convective_ice + large_scale_ice) * mass


def compute_total_precipitable_water(specific_humidity: FloatField, mass: FloatField, total_precipitable_water: FloatFieldIJ):
    with computation(FORWARD), interval(...):
        ice_water_path += specific_humidity * mass


class Moist(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory, config: MoistConfig):
        super().__init__(stencil_factory)

        self._config = config

        # initialize saturation vapor pressure tables
        self._saturation_tables = get_saturation_vapor_pressure_table(stencil_factory.backend)

        # initialize locals
        self._locals = MoistLocals.make_locals(quantity_factory)
        self._locals.my_value_is_1_2d.data[:] = Float(1.0)

        # create convection tracers, to be initilalized at runtime
        self.convection_tracers = ConvectionTracers.ones(
            quantity_factory,
            data_dimensions={
                "convection_tracers": config.NUMBER_OF_TRACERS,
                "size_three_dimension": 3,
                "size_four_dimension": 4,
            },
        )

        # construct stencils
        self._set_value = stencil_factory.from_dims_halo(func=set_value, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._set_surface_type = stencil_factory.from_dims_halo(func=set_surface_type, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_derived_state = stencil_factory.from_dims_halo(func=compute_derived_state, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._fix_mixing_ratio = stencil_factory.from_dims_halo(func=fix_mixing_ratio, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._copy = stencil_factory.from_dims_halo(func=copy, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._add = stencil_factory.from_dims_halo(func=add, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._divide = stencil_factory.from_dims_halo(func=divide, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._find_highest_level_interface = stencil_factory.from_dims_halo(func=find_highest_level_interface, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._find_lowest_level_interface = stencil_factory.from_dims_halo(func=find_lowest_level_interface, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._export_cbl_level = stencil_factory.from_dims_halo(func=export_cbl_level, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._buoyancy_2 = Buoyancy2(stencil_factory=stencil_factory, quantity_factory=quantity_factory)
        self._buoyancy_1 = stencil_factory.from_dims_halo(func=buoyancy_1, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_convection_fraction = stencil_factory.from_dims_halo(
            func=compute_convection_fraction,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CONVECTION_FRACTION_MAX": config.CONVECTION_FRACTION_MAX,
                "CONVECTION_FRACTION_MIN": config.CONVECTION_FRACTION_MIN,
                "CONVECTION_FRACTION_EXP": config.CONVECTION_FRACTION_EXP,
            },
        )
        self._export_concentration = stencil_factory.from_dims_halo(func=export_concentration, compute_dims=[I_DIM, J_DIM, K_DIM])

        # initialize convection and microphysics schemes
        if self._config.SHALLOW_MID_DEEP:
            if self._config.SHALLOW_CONVECTION_OPTION == "UW":
                self._uw_config = UWConfiguration(
                    JASON=True if quantity_factory.sizer.nz == 72 else False,
                    NCNST=config.NUMBER_OF_TRACERS,
                    k0=quantity_factory.sizer.nz,
                    windsrcavg=config.WIND_SOURCE_AVERAGE,
                    dotransport=config.USE_TRACER_TRANSPORT_UW,
                    qtsrchgt=config.TOTAL_WATER_INTERPOLATION_HEIGHT,
                    qtsrc_fac=config.TOTAL_WATER_INTERPOLATION_HEIGHT,
                    thlsrc_fac=config.LIQUID_POTENTIAL_TEMPERATURE_SCALING,
                    frc_rasn=config.PRECIP_FRACTION_OF_EXPELLED_CONDENSATE,
                    rbuoy=config.NONHYDRO_PRESSURE_EFFECT_ON_UPDRAFT,
                    epsvarw=config.PBL_TOP_W_VARIANCE_BY_MESO_COMPONENT,
                    use_CINcin=config.USE_IMPLICIT_CIN,
                    mumin1=config.MIN_PBL_TOP_MASSFLUX,
                    rmaxfrac=config.MAX_CORE_UPDRAFT_FRACTION,
                    PGFc=config.PGF_COEFFICIENT,
                    dt=config.DT_MOIST,
                    niter_xc=config.NUMBER_XC_ITERATIONS,
                    criqc=config.MIN_UPDRAFT_CONDENSATE,
                    rle=config.LATERAL_ENTRAINMENT_COEFFICIENT,
                    cridist_opt=config.LATERAL_ENTRAINMENT_MODE,
                    mixscale=config.VERTICAL_MIXING_RATE_STRUCTURE,
                    rdrag=config.DRAG_COEFFICIENT,
                    rkm=config.BUOYANCY_SORTING_PARAMETER,
                    use_self_detrain=config.USE_SELF_DETRAINMENT,
                    detrhgt=config.CRITICAL_MIXING_HEIGHT,
                    use_cumpenent=config.USE_CUMULUS_PENETRATIVE_ENTRAINMENT,
                    rpen=config.PENETRATIVE_ENTRAINMENT_FACTOR,
                    use_momenflx=config.USE_MOMENTUM_FLUX,
                    rdrop=config.LIQUID_DROP_RADIUS,
                    iter_cin=config.NUMBER_IMPLICIT_CIN_ITERATIONS,
                    SCLM_SHALLOW=config.SCLM_SHALLOW,
                )
                self._uw_state = UWState.zeros(quantity_factory, data_dimensions={"ntracers": config.NUMBER_OF_TRACERS})
                self._uw = ComputeUwshcuInv(stencil_factory=stencil_factory, quantity_factory=quantity_factory, config=self._uw_config)
            if self._config.CONVECTION_OPTION == "RAS":
                raise ValueError(f"{self._config.CONVECTION_OPTION} convection not implemented. Please choose a different option.")
            if self._config.CONVECTION_OPTION == "GF":
                self._gf2020_config = GF2020Config(
                    DT_MOIST=config.DT_MOIST,
                    LHYDROSTATIC=config.HYDROSTATIC,
                    STOCHASTIC_CNV=config.STOCHASTIC_CONVECTION,
                    STOCH_TOP=config.STOCH_TOP,
                    STOCH_BOT=config.STOCH_BOT,
                    GF_MIN_AREA=config.GF_MIN_AREA,
                    GF_ENV_SETTING=config.GF_ENV_SETTING,
                    ENTRVERSION=config.ENTRAINMENT_VERSION,
                    CONVECTION_TRACER=config.CONVECTION_TRACER,
                    C1=config.C1,
                    ADV_TRIGGER=config.ADV_TRIGGER,
                    AUTOCONV=config.AUTOCONV,
                    USE_TRACER_TRANSPORT=config.USE_TRACER_TRANSPORT,
                    SCLM_DEEP=config.SCLM_DEEP,
                    FIX_CONVECTIVE_CLOUD=config.FIX_CONVECTIVE_CLOUD,
                    APPLY_SUBSIDENCE_MICROPHYSICS=config.APPLY_SUBSIDENCE_MICROPHYSICS,
                    NUMBER_OF_TRACERS=config.NUMBER_OF_TRACERS,
                    USE_MOMENTUM_TRANSPORT=config.USE_MOMENTUM_TRANSPORT,
                )
                self._gf2020_cumulus_parameterization_config = GF2020CumulusParameterizationConfig(
                    # plume dependent
                    DOWNDRAFT_MAX_HEIGHT_LAND_SHALLOW=config.DOWNDRAFT_MAX_HEIGHT_LAND_SHALLOW,
                    DOWNDRAFT_MAX_HEIGHT_LAND_MID=config.DOWNDRAFT_MAX_HEIGHT_LAND_MID,
                    DOWNDRAFT_MAX_HEIGHT_LAND_DEEP=config.DOWNDRAFT_MAX_HEIGHT_LAND_DEEP,
                    DOWNDRAFT_MAX_HEIGHT_OCEAN_SHALLOW=config.DOWNDRAFT_MAX_HEIGHT_OCEAN_SHALLOW,
                    DOWNDRAFT_MAX_HEIGHT_OCEAN_MID=config.DOWNDRAFT_MAX_HEIGHT_OCEAN_MID,
                    DOWNDRAFT_MAX_HEIGHT_OCEAN_DEEP=config.DOWNDRAFT_MAX_HEIGHT_OCEAN_DEEP,
                    UPDRAFT_MAX_HEIGHT_LAND_SHALLOW=config.UPDRAFT_MAX_HEIGHT_LAND_SHALLOW,
                    UPDRAFT_MAX_HEIGHT_LAND_MID=config.UPDRAFT_MAX_HEIGHT_LAND_MID,
                    UPDRAFT_MAX_HEIGHT_LAND_DEEP=config.UPDRAFT_MAX_HEIGHT_LAND_DEEP,
                    UPDRAFT_MAX_HEIGHT_OCEAN_SHALLOW=config.UPDRAFT_MAX_HEIGHT_OCEAN_SHALLOW,
                    UPDRAFT_MAX_HEIGHT_OCEAN_MID=config.UPDRAFT_MAX_HEIGHT_OCEAN_MID,
                    UPDRAFT_MAX_HEIGHT_OCEAN_DEEP=config.UPDRAFT_MAX_HEIGHT_OCEAN_DEEP,
                    MINIMUM_EVAP_FRACTION_LAND_SHALLOW=config.MINIMUM_EVAP_FRACTION_LAND_SHALLOW,
                    MINIMUM_EVAP_FRACTION_LAND_MID=config.MINIMUM_EVAP_FRACTION_LAND_MID,
                    MINIMUM_EVAP_FRACTION_LAND_DEEP=config.MINIMUM_EVAP_FRACTION_LAND_DEEP,
                    MINIMUM_EVAP_FRACTION_OCEAN_SHALLOW=config.MINIMUM_EVAP_FRACTION_OCEAN_SHALLOW,
                    MINIMUM_EVAP_FRACTION_OCEAN_MID=config.MINIMUM_EVAP_FRACTION_OCEAN_MID,
                    MINIMUM_EVAP_FRACTION_OCEAN_DEEP=config.MINIMUM_EVAP_FRACTION_OCEAN_DEEP,
                    MAXIMUM_EVAP_FRACTION_LAND_SHALLOW=config.MAXIMUM_EVAP_FRACTION_LAND_SHALLOW,
                    MAXIMUM_EVAP_FRACTION_LAND_MID=config.MAXIMUM_EVAP_FRACTION_LAND_MID,
                    MAXIMUM_EVAP_FRACTION_LAND_DEEP=config.MAXIMUM_EVAP_FRACTION_LAND_DEEP,
                    MAXIMUM_EVAP_FRACTION_OCEAN_SHALLOW=config.MAXIMUM_EVAP_FRACTION_OCEAN_SHALLOW,
                    MAXIMUM_EVAP_FRACTION_OCEAN_MID=config.MAXIMUM_EVAP_FRACTION_OCEAN_MID,
                    MAXIMUM_EVAP_FRACTION_OCEAN_DEEP=config.MAXIMUM_EVAP_FRACTION_OCEAN_DEEP,
                    CLOUD_BASE_MASS_FLUX_FACTOR_SHALLOW=config.CLOUD_BASE_MASS_FLUX_FACTOR_SHALLOW,
                    CLOUD_BASE_MASS_FLUX_FACTOR_MID=config.CLOUD_BASE_MASS_FLUX_FACTOR_MID,
                    CLOUD_BASE_MASS_FLUX_FACTOR_DEEP=config.CLOUD_BASE_MASS_FLUX_FACTOR_DEEP,
                    USE_EXCESS_SHALLOW=config.USE_EXCESS_SHALLOW,
                    USE_EXCESS_MID=config.USE_EXCESS_MID,
                    USE_EXCESS_DEEP=config.USE_EXCESS_DEEP,
                    AVERAGE_LAYER_DEPTH_SHALLOW=config.AVERAGE_LAYER_DEPTH_SHALLOW,
                    AVERAGE_LAYER_DEPTH_MID=config.AVERAGE_LAYER_DEPTH_MID,
                    AVERAGE_LAYER_DEPTH_DEEP=config.AVERAGE_LAYER_DEPTH_DEEP,
                    ENABLE_SHALLOW=config.ENABLE_SHALLOW,
                    ENABLE_MID=config.ENABLE_MID,
                    ENABLE_DEEP=config.ENABLE_DEEP,
                    ENTRAINMENT_RATE_SHALLOW=config.ENTRAINMENT_RATE_SHALLOW,
                    ENTRAINMENT_RATE_MID=config.ENTRAINMENT_RATE_MID,
                    ENTRAINMENT_RATE_DEEP=config.ENTRAINMENT_RATE_DEEP,
                    C0_SHAL=config.C0_SHAL,
                    C0_MID=config.C0_MID,
                    C0_DEEP=config.C0_DEEP,
                    TAU_MID=config.TAU_MID,
                    TAU_DEEP=config.TAU_DEEP,
                    CLOSURE_CHOICE_SHALLOW=config.CLOSURE_CHOICE_SHALLOW,
                    CLOSURE_CHOICE_MID=config.CLOSURE_CHOICE_MID,
                    CLOSURE_CHOICE_DEEP=config.CLOSURE_CHOICE_DEEP,
                    # plume independent
                    SHALLOW_MID_DEEP=config.SHALLOW_MID_DEEP,
                    ZERO_DIFF=config.ZERO_DIFF,
                    MOIST_TRIGGER=config.MOIST_TRIGGER,
                    LAMBDA_DEEP=config.LAMBDA_DEEP,
                    LAMBDA_SHALLOW_DOWN=config.LAMBDA_SHALLOW_DOWN,
                    CAP_MAXS=config.CAP_MAXS,
                    OUTPUT_SOUNDING=config.OUTPUT_SOUNDING,
                    USE_SCALE_DEP=config.USE_SCALE_DEP,
                    SATURATION_CALCULATION_CHOICE=config.SATURATION_CALCULATION_CHOICE,
                    CLOUD_LEVEL_GRID=config.CLOUD_LEVEL_GRID,
                    FRAC_MODIS=config.FRAC_MODIS,
                    BOUNDARY_CONDITION_METHOD=config.BOUNDARY_CONDITION_METHOD,
                    OVERSHOOT=config.OVERSHOOT,
                    USE_MEMORY=config.USE_MEMORY,
                    DOWNDRAFT=config.DOWNDRAFT,
                    USE_WETBULB=config.USE_WETBULB,
                    DIURNAL_CYCLE=config.DIURNAL_CYCLE,
                    USE_LINEAR_SUBCLOUD_MOISTURE_FLUXES=config.USE_LINEAR_SUBCLOUD_MOISTURE_FLUXES,
                    CRITICAL_MIXING_RATIO_OVER_OCEAN=config.CRITICAL_MIXING_RATIO_OVER_OCEAN,
                    CRITICAL_MIXING_RATIO_OVER_LAND=config.CRITICAL_MIXING_RATIO_OVER_LAND,
                    BETA_SHALLOW=config.BETA_SHALLOW,
                    EVAP_FIX=config.EVAP_FIX,
                    SGS_W_TIMESCALE=config.SGS_W_TIMESCALE,
                    VERTICAL_DISCRETIZATION_OPTION=config.VERTICAL_DISCRETIZATION_OPTION,
                    ALP1=config.ALP1,
                    USE_FCT=config.USE_FCT,
                    MIN_ENTRAINMENT_RATE=config.MIN_ENTRAINMENT_RATE,
                    USE_SMOOTH_TENDENCIES=config.USE_SMOOTH_TENDENCIES,
                    USE_RAIN_EVAP_BELOW_CLOUD_BASE=config.USE_RAIN_EVAP_BELOW_CLOUD_BASE,
                    USE_CLOUD_DISSIPATION=config.USE_CLOUD_DISSIPATION,
                    LIGHTNING_DIAGNOSTICS=config.LIGHTNING_DIAGNOSTICS,
                    USE_TRACER_SCAVENGE=config.USE_TRACER_SCAVENGE,
                    USE_TRACER_EVAPORATION=config.USE_TRACER_EVAPORATION,
                    USE_FLUX_FORM=config.USE_FLUX_FORM,
                    MAX_TEMP_VAPOR_TENDENCY=config.MAX_TEMP_VAPOR_TENDENCY,
                )
                self._gf2020_state = GF2020State.zeros(quantity_factory)
                self._gf2020 = GF2020(
                    stencil_factory=stencil_factory,
                    quantity_factory=quantity_factory,
                    config=self._gf2020_config,
                    cumulus_parameterization_config=self._gf2020_cumulus_parameterization_config,
                    saturation_tables=self._saturation_tables,
                )
        else:
            if self._config.CONVECTION_OPTION == "RAS":
                raise ValueError(f"{self._config.CONVECTION_OPTION} convection not implemented. Please choose a different option.")
            if self._config.CONVECTION_OPTION == "GF":
                self._gf2020_config = GF2020Config(
                    DT_MOIST=config.DT_MOIST,
                    LHYDROSTATIC=config.HYDROSTATIC,
                    STOCHASTIC_CNV=config.STOCHASTIC_CONVECTION,
                    STOCH_TOP=config.STOCH_TOP,
                    STOCH_BOT=config.STOCH_BOT,
                    GF_MIN_AREA=config.GF_MIN_AREA,
                    GF_ENV_SETTING=config.GF_ENV_SETTING,
                    ENTRVERSION=config.ENTRAINMENT_VERSION,
                    CONVECTION_TRACER=config.CONVECTION_TRACER,
                    C1=config.C1,
                    ADV_TRIGGER=config.ADV_TRIGGER,
                    AUTOCONV=config.AUTOCONV,
                    USE_TRACER_TRANSPORT=config.USE_TRACER_TRANSPORT,
                    SCLM_DEEP=config.SCLM_DEEP,
                    FIX_CONVECTIVE_CLOUD=config.FIX_CONVECTIVE_CLOUD,
                    APPLY_SUBSIDENCE_MICROPHYSICS=config.APPLY_SUBSIDENCE_MICROPHYSICS,
                    NUMBER_OF_TRACERS=config.NUMBER_OF_TRACERS,
                    USE_MOMENTUM_TRANSPORT=config.USE_MOMENTUM_TRANSPORT,
                )
                self._gf2020_cumulus_parameterization_config = GF2020CumulusParameterizationConfig(
                    # plume dependent
                    DOWNDRAFT_MAX_HEIGHT_LAND_SHALLOW=config.DOWNDRAFT_MAX_HEIGHT_LAND_SHALLOW,
                    DOWNDRAFT_MAX_HEIGHT_LAND_MID=config.DOWNDRAFT_MAX_HEIGHT_LAND_MID,
                    DOWNDRAFT_MAX_HEIGHT_LAND_DEEP=config.DOWNDRAFT_MAX_HEIGHT_LAND_DEEP,
                    DOWNDRAFT_MAX_HEIGHT_OCEAN_SHALLOW=config.DOWNDRAFT_MAX_HEIGHT_OCEAN_SHALLOW,
                    DOWNDRAFT_MAX_HEIGHT_OCEAN_MID=config.DOWNDRAFT_MAX_HEIGHT_OCEAN_MID,
                    DOWNDRAFT_MAX_HEIGHT_OCEAN_DEEP=config.DOWNDRAFT_MAX_HEIGHT_OCEAN_DEEP,
                    UPDRAFT_MAX_HEIGHT_LAND_SHALLOW=config.UPDRAFT_MAX_HEIGHT_LAND_SHALLOW,
                    UPDRAFT_MAX_HEIGHT_LAND_MID=config.UPDRAFT_MAX_HEIGHT_LAND_MID,
                    UPDRAFT_MAX_HEIGHT_LAND_DEEP=config.UPDRAFT_MAX_HEIGHT_LAND_DEEP,
                    UPDRAFT_MAX_HEIGHT_OCEAN_SHALLOW=config.UPDRAFT_MAX_HEIGHT_OCEAN_SHALLOW,
                    UPDRAFT_MAX_HEIGHT_OCEAN_MID=config.UPDRAFT_MAX_HEIGHT_OCEAN_MID,
                    UPDRAFT_MAX_HEIGHT_OCEAN_DEEP=config.UPDRAFT_MAX_HEIGHT_OCEAN_DEEP,
                    MINIMUM_EVAP_FRACTION_LAND_SHALLOW=config.MINIMUM_EVAP_FRACTION_LAND_SHALLOW,
                    MINIMUM_EVAP_FRACTION_LAND_MID=config.MINIMUM_EVAP_FRACTION_LAND_MID,
                    MINIMUM_EVAP_FRACTION_LAND_DEEP=config.MINIMUM_EVAP_FRACTION_LAND_DEEP,
                    MINIMUM_EVAP_FRACTION_OCEAN_SHALLOW=config.MINIMUM_EVAP_FRACTION_OCEAN_SHALLOW,
                    MINIMUM_EVAP_FRACTION_OCEAN_MID=config.MINIMUM_EVAP_FRACTION_OCEAN_MID,
                    MINIMUM_EVAP_FRACTION_OCEAN_DEEP=config.MINIMUM_EVAP_FRACTION_OCEAN_DEEP,
                    MAXIMUM_EVAP_FRACTION_LAND_SHALLOW=config.MAXIMUM_EVAP_FRACTION_LAND_SHALLOW,
                    MAXIMUM_EVAP_FRACTION_LAND_MID=config.MAXIMUM_EVAP_FRACTION_LAND_MID,
                    MAXIMUM_EVAP_FRACTION_LAND_DEEP=config.MAXIMUM_EVAP_FRACTION_LAND_DEEP,
                    MAXIMUM_EVAP_FRACTION_OCEAN_SHALLOW=config.MAXIMUM_EVAP_FRACTION_OCEAN_SHALLOW,
                    MAXIMUM_EVAP_FRACTION_OCEAN_MID=config.MAXIMUM_EVAP_FRACTION_OCEAN_MID,
                    MAXIMUM_EVAP_FRACTION_OCEAN_DEEP=config.MAXIMUM_EVAP_FRACTION_OCEAN_DEEP,
                    CLOUD_BASE_MASS_FLUX_FACTOR_SHALLOW=config.CLOUD_BASE_MASS_FLUX_FACTOR_SHALLOW,
                    CLOUD_BASE_MASS_FLUX_FACTOR_MID=config.CLOUD_BASE_MASS_FLUX_FACTOR_MID,
                    CLOUD_BASE_MASS_FLUX_FACTOR_DEEP=config.CLOUD_BASE_MASS_FLUX_FACTOR_DEEP,
                    USE_EXCESS_SHALLOW=config.USE_EXCESS_SHALLOW,
                    USE_EXCESS_MID=config.USE_EXCESS_MID,
                    USE_EXCESS_DEEP=config.USE_EXCESS_DEEP,
                    AVERAGE_LAYER_DEPTH_SHALLOW=config.AVERAGE_LAYER_DEPTH_SHALLOW,
                    AVERAGE_LAYER_DEPTH_MID=config.AVERAGE_LAYER_DEPTH_MID,
                    AVERAGE_LAYER_DEPTH_DEEP=config.AVERAGE_LAYER_DEPTH_DEEP,
                    ENABLE_SHALLOW=config.ENABLE_SHALLOW,
                    ENABLE_MID=config.ENABLE_MID,
                    ENABLE_DEEP=config.ENABLE_DEEP,
                    ENTRAINMENT_RATE_SHALLOW=config.ENTRAINMENT_RATE_SHALLOW,
                    ENTRAINMENT_RATE_MID=config.ENTRAINMENT_RATE_MID,
                    ENTRAINMENT_RATE_DEEP=config.ENTRAINMENT_RATE_DEEP,
                    C0_SHAL=config.C0_SHAL,
                    C0_MID=config.C0_MID,
                    C0_DEEP=config.C0_DEEP,
                    TAU_MID=config.TAU_MID,
                    TAU_DEEP=config.TAU_DEEP,
                    CLOSURE_CHOICE_SHALLOW=config.CLOSURE_CHOICE_SHALLOW,
                    CLOSURE_CHOICE_MID=config.CLOSURE_CHOICE_MID,
                    CLOSURE_CHOICE_DEEP=config.CLOSURE_CHOICE_DEEP,
                    # plume independent
                    SHALLOW_MID_DEEP=config.SHALLOW_MID_DEEP,
                    ZERO_DIFF=config.ZERO_DIFF,
                    MOIST_TRIGGER=config.MOIST_TRIGGER,
                    LAMBDA_DEEP=config.LAMBDA_DEEP,
                    LAMBDA_SHALLOW_DOWN=config.LAMBDA_SHALLOW_DOWN,
                    CAP_MAXS=config.CAP_MAXS,
                    OUTPUT_SOUNDING=config.OUTPUT_SOUNDING,
                    USE_SCALE_DEP=config.USE_SCALE_DEP,
                    SATURATION_CALCULATION_CHOICE=config.SATURATION_CALCULATION_CHOICE,
                    CLOUD_LEVEL_GRID=config.CLOUD_LEVEL_GRID,
                    FRAC_MODIS=config.FRAC_MODIS,
                    BOUNDARY_CONDITION_METHOD=config.BOUNDARY_CONDITION_METHOD,
                    OVERSHOOT=config.OVERSHOOT,
                    USE_MEMORY=config.USE_MEMORY,
                    DOWNDRAFT=config.DOWNDRAFT,
                    USE_WETBULB=config.USE_WETBULB,
                    DIURNAL_CYCLE=config.DIURNAL_CYCLE,
                    USE_LINEAR_SUBCLOUD_MOISTURE_FLUXES=config.USE_LINEAR_SUBCLOUD_MOISTURE_FLUXES,
                    CRITICAL_MIXING_RATIO_OVER_OCEAN=config.CRITICAL_MIXING_RATIO_OVER_OCEAN,
                    CRITICAL_MIXING_RATIO_OVER_LAND=config.CRITICAL_MIXING_RATIO_OVER_LAND,
                    BETA_SHALLOW=config.BETA_SHALLOW,
                    EVAP_FIX=config.EVAP_FIX,
                    SGS_W_TIMESCALE=config.SGS_W_TIMESCALE,
                    VERTICAL_DISCRETIZATION_OPTION=config.VERTICAL_DISCRETIZATION_OPTION,
                    ALP1=config.ALP1,
                    USE_FCT=config.USE_FCT,
                    MIN_ENTRAINMENT_RATE=config.MIN_ENTRAINMENT_RATE,
                    USE_SMOOTH_TENDENCIES=config.USE_SMOOTH_TENDENCIES,
                    USE_RAIN_EVAP_BELOW_CLOUD_BASE=config.USE_RAIN_EVAP_BELOW_CLOUD_BASE,
                    USE_CLOUD_DISSIPATION=config.USE_CLOUD_DISSIPATION,
                    LIGHTNING_DIAGNOSTICS=config.LIGHTNING_DIAGNOSTICS,
                    USE_TRACER_SCAVENGE=config.USE_TRACER_SCAVENGE,
                    USE_TRACER_EVAPORATION=config.USE_TRACER_EVAPORATION,
                    USE_FLUX_FORM=config.USE_FLUX_FORM,
                    MAX_TEMP_VAPOR_TENDENCY=config.MAX_TEMP_VAPOR_TENDENCY,
                )
                self._gf2020_state = GF2020State.zeros(quantity_factory)
                self._gf2020 = GF2020(
                    stencil_factory=stencil_factory,
                    quantity_factory=quantity_factory,
                    config=self._gf2020_config,
                    cumulus_parameterization_config=self._gf2020_cumulus_parameterization_config,
                    saturation_tables=self._saturation_tables,
                )
            if self._config.SHALLOW_CONVECTION_OPTION == "UW":
                self._uw_config = UWConfiguration(
                    JASON=True if quantity_factory.sizer.nz == 72 else False,
                    NCNST=config.NUMBER_OF_TRACERS,
                    k0=quantity_factory.sizer.nz,
                    windsrcavg=config.WIND_SOURCE_AVERAGE,
                    dotransport=config.USE_TRACER_TRANSPORT_UW,
                    qtsrchgt=config.TOTAL_WATER_INTERPOLATION_HEIGHT,
                    qtsrc_fac=config.TOTAL_WATER_INTERPOLATION_HEIGHT,
                    thlsrc_fac=config.LIQUID_POTENTIAL_TEMPERATURE_SCALING,
                    frc_rasn=config.PRECIP_FRACTION_OF_EXPELLED_CONDENSATE,
                    rbuoy=config.NONHYDRO_PRESSURE_EFFECT_ON_UPDRAFT,
                    epsvarw=config.PBL_TOP_W_VARIANCE_BY_MESO_COMPONENT,
                    use_CINcin=config.USE_IMPLICIT_CIN,
                    mumin1=config.MIN_PBL_TOP_MASSFLUX,
                    rmaxfrac=config.MAX_CORE_UPDRAFT_FRACTION,
                    PGFc=config.PGF_COEFFICIENT,
                    dt=config.DT_MOIST,
                    niter_xc=config.NUMBER_XC_ITERATIONS,
                    criqc=config.MIN_UPDRAFT_CONDENSATE,
                    rle=config.LATERAL_ENTRAINMENT_COEFFICIENT,
                    cridist_opt=config.LATERAL_ENTRAINMENT_MODE,
                    mixscale=config.VERTICAL_MIXING_RATE_STRUCTURE,
                    rdrag=config.DRAG_COEFFICIENT,
                    rkm=config.BUOYANCY_SORTING_PARAMETER,
                    use_self_detrain=config.USE_SELF_DETRAINMENT,
                    detrhgt=config.CRITICAL_MIXING_HEIGHT,
                    use_cumpenent=config.USE_CUMULUS_PENETRATIVE_ENTRAINMENT,
                    rpen=config.PENETRATIVE_ENTRAINMENT_FACTOR,
                    use_momenflx=config.USE_MOMENTUM_FLUX,
                    rdrop=config.LIQUID_DROP_RADIUS,
                    iter_cin=config.NUMBER_IMPLICIT_CIN_ITERATIONS,
                    SCLM_SHALLOW=config.SCLM_SHALLOW,
                )
                self._uw_state = UWState.zeros(quantity_factory, data_dimensions={"ntracers": config.NUMBER_OF_TRACERS})
                self._uw = ComputeUwshcuInv(stencil_factory=stencil_factory, quantity_factory=quantity_factory, config=self._uw_config)

        if self._config.CLOUD_MICROPHYSICS_OPTION == "BACM_1M":
            raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")
        if self._config.CLOUD_MICROPHYSICS_OPTION == "GFDL_1M":
            pass
            # self._gfdl1m_config = GFDL1MConfig(
            #     LPHYS_HYDROSTATIC=config.HYDROSTATIC,
            #     LHYDROSTATIC=config.PHYS_HYDROSTATIC,
            #     DT_MOIST=config.DT_MOIST,
            #     MP_TIME=
            #     T_MIN=
            #     T_SUB=
            #     TAU_R2G=
            #     TAU_SMLT=
            #     TAU_G2R=
            #     DW_LAND=
            #     DW_OCEAN=
            #     VI_FAC=
            #     VR_FAC=
            #     VS_FAC=
            #     VG_FAC=
            #     QL_MLT=
            #     DO_QA=
            #     FIX_NEGATIVE=
            #     VI_MAX=
            #     VS_MAX=
            #     VG_MAX=
            #     VR_MAX=
            #     QS_MLT=
            #     QS0_CRT=
            #     QI_GEN=
            #     QL0_MAX=
            #     QI0_MAX=
            #     QI0_CRT=
            #     QR0_CRT=
            #     FAST_SAT_ADJ=
            #     RH_INC=
            #     RH_INS=
            #     RH_INR=
            #     CONST_VI=
            #     CONST_VS=
            #     CONST_VG=
            #     CONST_VR=
            #     USE_CCN=
            #     RTHRESHU=
            #     RTHRESHS=
            #     CCN_L=
            #     CCN_O=
            #     QC_CRT=
            #     TAU_G2V=
            #     TAU_V2G=
            #     TAU_S2V=
            #     TAU_V2S=
            #     TAU_REVP=
            #     TAU_FRZ=
            #     DO_BIGG=
            #     DO_EVAP=
            #     DO_SUBL=
            #     SAT_ADJ0=
            #     C_PIACR=
            #     TAU_IMLT=
            #     TAU_V2L=
            #     TAU_L2V=
            #     TAU_I2V=
            #     TAU_I2S=
            #     TAU_L2R=
            #     QI_LIM=
            #     QL_GEN=
            #     C_PAUT=
            #     C_PSACI=
            #     C_PGACS=
            #     C_PGACI=
            #     Z_SLOPE_LIQ=
            #     Z_SLOPE_ICE=
            #     PROG_CCN=
            #     C_CRACW=
            #     ALIN=
            #     CLIN=
            #     PRECIPRAD=
            #     CLD_MIN=
            #     USE_PPM=
            #     MONO_PROF=
            #     DO_SEDI_HEAT=
            #     SEDI_TRANSPORT=
            #     DO_SEDI_W=
            #     DE_ICE=
            #     ICLOUD_F=
            #     IRAIN_F=
            #     MP_PRINT=
            #     LMELTFRZ=
            #     USE_BERGERON=
            #     TURNRHCRIT_PARAM=
            #     PDFSHAPE=
            #     ANV_ICEFALL=
            #     LS_ICEFALL=
            #     LIQ_RADII_PARAM=
            #     ICE_RADII_PARAM=
            #     FAC_RI=
            #     MIN_RI=
            #     MAX_RI=
            #     FAC_RL=
            #     MIN_RL=
            #     MAX_RL=
            #     CCW_EVAP_EFF=
            #     CCI_EVAP_EFF=
            # )
        if self._config.CLOUD_MICROPHYSICS_OPTION == "THOM_1M":
            raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")
        if self._config.CLOUD_MICROPHYSICS_OPTION == "MGB2_2M":
            raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")

        self._update_cloud_fraction = stencil_factory.from_dims_halo(func=update_cloud_fraction, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._get_saturation_specific_humidity = stencil_factory.from_dims_halo(func=get_saturation_specific_humidity, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._export_relative_humidity_wrt_ice = stencil_factory.from_dims_halo(func=export_relative_humidity_wrt_ice, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._export_output_saturation_ratio = stencil_factory.from_dims_halo(func=export_output_saturation_ratio, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._rain_out_excessive_rh = stencil_factory.from_dims_halo(func=rain_out_excessive_rh, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._divide_by_dt_moist_2d = stencil_factory.from_dims_halo(
            func=divide_by_dt_moist_2d, compute_dims=[I_DIM, J_DIM, K_DIM], externals={"DT_MOIST": config.DT_MOIST}
        )
        self._add_to_self = stencil_factory.from_dims_halo(func=add_to_self, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._update_dlayer_pressure_thickness_dt = stencil_factory.from_dims_halo(func=update_dlayer_pressure_thickness_dt, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._add_to_self_2d = stencil_factory.from_dims_halo(func=add_to_self_2d, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._ensure_non_negative_2d = stencil_factory.from_dims_halo(func=ensure_non_negative_2d, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._subtract_2d = stencil_factory.from_dims_halo(func=subtract_2d, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._multiply_2d = stencil_factory.from_dims_halo(func=multiply_2d, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._get_Kuchera_ratios = stencil_factory.from_dims_halo(func=get_Kuchera_ratios, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._copy_2d = stencil_factory.from_dims_halo(func=copy_2d, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_snowfall_total = stencil_factory.from_dims_halo(func=compute_snowfall_total, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._multiply_by_dt_moist_2d = stencil_factory.from_dims_halo(
            func=multiply_by_dt_moist_2d, compute_dims=[I_DIM, J_DIM, K_DIM], externals={"DT_MOIST": config.DT_MOIST}
        )
        self._compute_dry_static_energy = stencil_factory.from_dims_halo(func=compute_dry_static_energy, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_relative_humidty = stencil_factory.from_dims_halo(func=compute_relative_humidty, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_condensed_water_path = stencil_factory.from_dims_halo(func=compute_condensed_water_path, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_liquid_water_path = stencil_factory.from_dims_halo(func=compute_liquid_water_path, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_ice_water_path = stencil_factory.from_dims_halo(func=compute_ice_water_path, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_total_precipitable_water = stencil_factory.from_dims_halo(func=compute_total_precipitable_water, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._set_value_2d = stencil_factory.from_dims_halo(func=set_value_2D, compute_dims=[I_DIM, J_DIM, K_DIM])

    def __call__(self, state: MoistState, convection_tracers: ConvectionTracers):
        # Get alarm - is on (ringing) when pulled
        ALARM_IS_RINGING = get_alarm()

        if ALARM_IS_RINGING:
            ALARM_IS_RINGING = turn_alarm_off()

            # reset mass fluxes
            self._set_value(state.convective_diagnostics.total_cumulative_mass_flux, 0.0)
            self._set_value(state.convective_diagnostics.total_detraining_mass_flux, 0.0)

            # update surface_type for ice_fraction
            self._set_surface_type(
                land_fraction=state.surface_conditions.land_fraction,
                land_ice_fraction=state.surface_conditions.land_ice_fraction,
                ice_covered_fraction_of_tile=state.surface_conditions.ice_covered_fraction_of_tile,
                snow_mass=state.surface_conditions.snow_mass,
                surface_type=state.surface_conditions.surface_type,
            )

            # compute derived states
            self._compute_derived_state(
                mass=self._locals.mass,
                p_interface=state.atmospheric_state.p_interface,
                p_interface_mb=self._locals.p_interface_mb,
                p_mb=self._locals.p_mb,
                p_kappa_interface=self._locals.p_kappa_interface,
                p_kappa=self._locals.p_kappa,
                z_interface=state.atmospheric_state.z_interface,
                edge_height_above_surface=self._locals.edge_height_above_surface,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                t=state.atmospheric_state.t,
                ese=self._saturation_tables.ese,
                esx=self._saturation_tables.esx,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
            )

            # fill negative specific humidity with zero
            self._fix_mixing_ratio(
                mixing_ratio=state.cloud_condensates.specific_humidity,
                mass=self._locals.mass,
                adjustment=self._locals.temporary_2d,
            )
            if state.diagnostics.negative_vapor_adjustment_start is not None:
                self._copy(self._locals.temporary_2d, state.diagnostics.negative_vapor_adjustment_start)

            # save a copy of the state at the start of the computation
            # all fields are unmodified within Moist at this point
            # one save requires the derived state field p_kappa,
            # forcing this entire block slightly lower than otherwise necessary
            self._copy(state.atmospheric_state.u, state.state_at_input.u)
            self._copy(state.atmospheric_state.v, state.state_at_input.v)
            if state.state_at_input.specific_humidity is not None:
                self._copy(state.cloud_condensates.specific_humidity, state.state_at_input.specific_humidity)
            if state.state_at_input.large_scale_cloud_fraction is not None:
                self._copy(state.cloud_condensates.large_scale_cloud_fraction, state.state_at_input.large_scale_cloud_fraction)
            if state.state_at_input.large_scale_liquid is not None:
                self._copy(state.cloud_condensates.large_scale_liquid, state.state_at_input.large_scale_liquid)
            if state.state_at_input.large_scale_ice is not None:
                self._copy(state.cloud_condensates.large_scale_ice, state.state_at_input.large_scale_ice)
            if state.state_at_input.convective_cloud_fraction is not None:
                self._copy(state.cloud_condensates.convective_cloud_fraction, state.state_at_input.convective_cloud_fraction)
            if state.state_at_input.convective_ice is not None:
                self._copy(state.cloud_condensates.convective_ice, state.state_at_input.convective_ice)
            if state.state_at_input.convective_condensates is not None:
                self._add(state.cloud_condensates.convective_ice, state.cloud_condensates.convective_liquid, state.state_at_input.convective_liquid)
            if state.state_at_input.large_scale_condensates is not None:
                self._add(state.cloud_condensates.large_scale_ice, state.cloud_condensates.large_scale_liquid, state.state_at_input.large_scale_condensates)
            if state.state_at_input.pt is not None:
                self._divide(state.atmospheric_state.t, self._locals.p_kappa, state.state_at_input.pt)
            if state.state_at_input.specific_humidity is not None:
                self._copy(state.cloud_condensates.specific_humidity, state.state_at_input.specific_humidity)
            if state.state_at_input.t_surface is not None:
                self._copy(state.surface_conditions.t_surface, state.state_at_input.t_surface)

            # save scalar diffusivity levels
            if state.diagnostics.highest_level_of_scalar_diffusivity_gt_2 is not None:
                self._find_highest_level_interface(
                    scalar_diffusivity=state.atmospheric_state.scalar_diffusivity_interface,
                    value=Float(2.0),
                    highest_level=state.diagnostics.highest_level_of_scalar_diffusivity_gt_2,
                )

            if state.diagnostics.lowest_level_of_scalar_diffusivity_gt_2 is not None:
                self._find_lowest_level_interface(
                    scalar_diffusivity=state.atmospheric_state.scalar_diffusivity_interface,
                    value=Float(2.0),
                    lowest_level=state.diagnostics.lowest_level_of_scalar_diffusivity_gt_2,
                )

            if state.levels.cbl_level_before_moist is not None:
                self._export_cbl_level(
                    cbl_level_before_moist=state.levels.cbl_level_before_moist,
                    pbl_level=state.levels.pbl_level,
                    reference_pressure=state.atmospheric_state.reference_pressure,
                    p_min_cbl=P_MIN_CBL,
                )

            # compute buoyancy and related parameters
            self._buoyancy_2(
                t=state.atmospheric_state.t,
                specific_humidity=state.cloud_condensates.specific_humidity,
                p_interface_mb=self._locals.p_interface_mb,
                p_mb=self._locals.p_mb,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                buoyancy_surface_parcel=state.convective_diagnostics.buoyancy_surface_parcel,
                sbcape=state.convective_diagnostics.sbcape,
                mlcape=state.convective_diagnostics.mlcape,
                mucape=state.convective_diagnostics.mucape,
                sbcin=state.convective_diagnostics.sbcin,
                mlcin=state.convective_diagnostics.mlcin,
                mucin=state.convective_diagnostics.mucin,
                lfc=state.convective_diagnostics.lfc,
                lnb=state.convective_diagnostics.lnb,
            )

            self._buoyancy_1(
                t=state.atmospheric_state.t,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                specific_humidity=state.cloud_condensates.specific_humidity,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                buoyancy=state.convective_diagnostics.buoyancy_surface_parcel,
                cape=state.convective_diagnostics.cape_surface_parcel,
                cin=state.convective_diagnostics.cin_surface_parcel,
            )

            # initialize diagnosed convective fraction
            self._compute_convection_fraction(convection_fraction=state.convective_diagnostics.convection_fraction, cape=state.convective_diagnostics.cape_surface_parcel)

            # extract convective tracers from the TR bundle
            initialize_convection_tracers()

            # get aerosol activation properties
            if self._config.USE_AEROSOL_NN:
                do_aerosol_activateion = True
            else:
                do_aerosol_activateion = False

            # export concentrations
            self._export_concentration(
                field=state.cloud_condensates.liquid_ccn_concentration, factor=Float(1.0e-1), output=state.cloud_condensates.liquid_ccn_concentration
            )
            self._export_concentration(field=state.cloud_condensates.ice_ccn_concentration, factor=Float(1.0e-1), output=state.cloud_condensates.ice_ccn_concentration)

            # run convection and microphysics
            if self._config.SHALLOW_MID_DEEP:
                if self._config.SHALLOW_CONVECTION_OPTION == "UW":
                    run_UW = True
                if self._config.CONVECTION_OPTION == "RAS":
                    run_RAS = True
                if self._config.CONVECTION_OPTION == "GF":
                    run_GF = True
            else:
                if self._config.CONVECTION_OPTION == "RAS":
                    run_RAS = True
                if self._config.CONVECTION_OPTION == "GF":
                    run_GF = True
                if self._config.SHALLOW_CONVECTION_OPTION == "UW":
                    run_UW = True

            if self._config.CLOUD_MICROPHYSICS_OPTION == "BACM_1M":
                raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")
            if self._config.CLOUD_MICROPHYSICS_OPTION == "GFDL_1M":
                run_GFDL1M = True
            if self._config.CLOUD_MICROPHYSICS_OPTION == "THOM_1M":
                raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")
            if self._config.CLOUD_MICROPHYSICS_OPTION == "MGB2_2M":
                raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")

            # export cloud fractions
            if state.cloud_condensates.large_scale_ice_cloud_fraction is not None:
                self._update_cloud_fraction(
                    total_cloud_fraction=state.cloud_condensates.large_scale_ice_cloud_fraction,
                    convective_cloud_fraction=state.cloud_condensates.convective_cloud_fraction,
                    convective_desired_phase=state.cloud_condensates.convective_ice,
                    convective_other_phase=state.cloud_condensates.convective_liquid,
                    large_scale_cloud_fraction=state.cloud_condensates.large_scale_cloud_fraction,
                    large_scale_desired_phase=state.cloud_condensates.large_scale_ice,
                    large_scale_other_phase=state.cloud_condensates.large_scale_liquid,
                )

            if state.cloud_condensates.large_scale_liquid_cloud_fraction is not None:
                self._update_cloud_fraction(
                    total_cloud_fraction=state.cloud_condensates.large_scale_ice_cloud_fraction,
                    convective_cloud_fraction=state.cloud_condensates.convective_cloud_fraction,
                    convective_desired_phase=state.cloud_condensates.convective_liquid,
                    convective_other_phase=state.cloud_condensates.convective_ice,
                    large_scale_cloud_fraction=state.cloud_condensates.large_scale_cloud_fraction,
                    large_scale_desired_phase=state.cloud_condensates.large_scale_liquid,
                    large_scale_other_phase=state.cloud_condensates.large_scale_ice,
                )

            # rain-out and relative humidity where RH > 110%
            copy(state.atmospheric_state.t, state.tendencies.dt_dt_from_rh_cleanup)
            copy(state.cloud_condensates.specific_humidity, state.tendencies.dspecific_humidity_dt_from_rh_cleanup)

            # compute saturation specific humidity values for current P and T
            self._get_saturation_specific_humidity(
                t=state.atmospheric_state.t,
                p_mb=self._locals.p_mb,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                ese=self._saturation_tables.ese,
                esx=self._saturation_tables.esx,
            )

            if state.cloud_condensates.relative_humidity_wrt_ice is not None:
                self._export_relative_humidity_wrt_ice(
                    t=state.atmospheric_state.t,
                    p_mb=self._locals.p_mb,
                    saturation_specific_humidity=self._locals.saturation_specific_humidity,
                    dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                    ese=self._saturation_tables.ese,
                    esx=self._saturation_tables.esx,
                )

            if state.state_at_output.saturation_ratio is not None:
                self._export_output_saturation_ratio(
                    saturation_ratio=state.state_at_output.saturation_ratio,
                    large_scale_ice_cloud_fraction=state.cloud_condensates.large_scale_ice_cloud_fraction,
                    specific_humidity=state.cloud_condensates.specific_humidity,
                    saturation_specific_humidity=self._locals.saturation_specific_humidity,
                )

            if self._config.CLOUD_MICROPHYSICS_OPTION == "MGB2_2M":
                raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")
            else:
                self._get_saturation_specific_humidity(
                    t=state.atmospheric_state.t,
                    p_mb=self._locals.p_mb,
                    saturation_specific_humidity=self._locals.saturation_specific_humidity,
                    dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                    ese=self._saturation_tables.ese,
                    esx=self._saturation_tables.esx,
                )

            if state.cloud_condensates.relative_humidity_wrt_liquid is not None:
                self._divide(
                    input_1=state.cloud_condensates.specific_humidity,
                    input_2=self._locals.saturation_specific_humidity,
                    output=state.cloud_condensates.relative_humidity_wrt_liquid,
                )

            # rain out excessive RH
            self._rain_out_excessive_rh(
                t=state.atmospheric_state.t,
                specific_humidity=state.cloud_condensates.specific_humidity,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
                mass=self._locals.mass,
                rain_from_large_scale_nonanvil=state.precipitation_at_surface.rain_from_large_scale_nonanvil,
                spurious_rain_from_relative_humidity_cleanup=state.precipitation_at_surface.spurious_rain_from_relative_humidity_cleanup,
                dt_dt_from_rh_cleanup=state.tendencies.dt_dt_from_rh_cleanup,
                dspecific_humidity_dt_from_rh_cleanup=state.tendencies.dspecific_humidity_dt_from_rh_cleanup,
            )

            # cleanup any negative specific_humidity/QC/CF
            self._fix_mixing_ratio(
                mixing_ratio=state.cloud_condensates.specific_humidity,
                mass=self._locals.mass,
                adjustment=self._locals.temporary_2d,
            )

            if state.diagnostics.negative_vapor_adjustment_end is not None:
                self._divide_by_dt_moist_2d(input=self._locals.temporary_2d, output=state.diagnostics.negative_vapor_adjustment_end)

            # export total moist tendencies and fluxes
            # zonal wind
            if state.tendencies.du_dt is not None:
                self._set_value(field=state.tendencies.du_dt, value=Float(0.0))
                if state.tendencies.du_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.du_dt, summand=state.tendencies.du_dt_deep_convection)
                if state.tendencies.du_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.du_dt, summand=state.tendencies.du_dt_shallow_convection)
                if state.tendencies.du_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.du_dt, summand=state.tendencies.du_dt_macrophysics)
                if state.tendencies.du_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.du_dt, summand=state.tendencies.du_dt_microphysics)

            # meridional wind
            if state.tendencies.dv_dt is not None:
                self._set_value(field=state.tendencies.dv_dt, value=Float(0.0))
                if state.tendencies.dv_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.dv_dt, summand=state.tendencies.dv_dt_deep_convection)
                if state.tendencies.dv_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dv_dt, summand=state.tendencies.dv_dt_shallow_convection)
                if state.tendencies.dv_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dv_dt, summand=state.tendencies.dv_dt_macrophysics)
                if state.tendencies.dv_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dv_dt, summand=state.tendencies.dv_dt_microphysics)

            # temperature
            if state.tendencies.dt_dt is not None:
                self._set_value(field=state.tendencies.dt_dt, value=Float(0.0))
                if state.tendencies.dt_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.dt_dt, summand=state.tendencies.dt_dt_deep_convection)
                if state.tendencies.dt_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dt_dt, summand=state.tendencies.dt_dt_shallow_convection)
                if state.tendencies.dt_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dt_dt, summand=state.tendencies.dt_dt_macrophysics)
                if state.tendencies.dt_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dt_dt, summand=state.tendencies.dt_dt_microphysics)
                if state.tendencies.dt_dt_from_rh_cleanup is not None:
                    self._add_to_self(field=state.tendencies.dt_dt, summand=state.tendencies.dt_dt_from_rh_cleanup)

            # specific humidity
            if state.tendencies.dspecific_humidity_dt is not None:
                self._set_value(field=state.tendencies.dspecific_humidity_dt, value=Float(0.0))
                if state.tendencies.dspecific_humidity_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.dspecific_humidity_dt, summand=state.tendencies.dspecific_humidity_dt_deep_convection)
                if state.tendencies.dspecific_humidity_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dspecific_humidity_dt, summand=state.tendencies.dspecific_humidity_dt_shallow_convection)
                if state.tendencies.dspecific_humidity_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dspecific_humidity_dt, summand=state.tendencies.dspecific_humidity_dt_macrophysics)
                if state.tendencies.dspecific_humidity_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dspecific_humidity_dt, summand=state.tendencies.dspecific_humidity_dt_microphysics)
                if state.tendencies.dspecific_humidity_dt_from_rh_cleanup is not None:
                    self._add_to_self(field=state.tendencies.dspecific_humidity_dt, summand=state.tendencies.dspecific_humidity_dt_from_rh_cleanup)

            # liquid mixing ratio
            if state.tendencies.dliquid_dt is not None:
                self._set_value(field=state.tendencies.dliquid_dt, value=Float(0.0))
                if state.tendencies.dliquid_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.dliquid_dt, summand=state.tendencies.dliquid_dt_deep_convection)
                if state.tendencies.dliquid_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dliquid_dt, summand=state.tendencies.dliquid_dt_shallow_convection)
                if state.tendencies.dliquid_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dliquid_dt, summand=state.tendencies.dliquid_dt_macrophysics)
                if state.tendencies.dliquid_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dliquid_dt, summand=state.tendencies.dliquid_dt_microphysics)

            # ice mixing ratio
            if state.tendencies.dice_dt is not None:
                self._set_value(field=state.tendencies.dice_dt, value=Float(0.0))
                if state.tendencies.dice_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.dice_dt, summand=state.tendencies.dice_dt_deep_convection)
                if state.tendencies.dice_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dice_dt, summand=state.tendencies.dice_dt_shallow_convection)
                if state.tendencies.dice_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dice_dt, summand=state.tendencies.dice_dt_macrophysics)
                if state.tendencies.dice_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dice_dt, summand=state.tendencies.dice_dt_microphysics)

            # rain mixing ratio
            if state.tendencies.drain_dt is not None:
                self._set_value(field=state.tendencies.drain_dt, value=Float(0.0))
                if state.tendencies.drain_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.drain_dt, summand=state.tendencies.drain_dt_shallow_convection)
                if state.tendencies.drain_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.drain_dt, summand=state.tendencies.drain_dt_macrophysics)
                if state.tendencies.drain_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.drain_dt, summand=state.tendencies.drain_dt_microphysics)

            # snow specific humidity
            if state.tendencies.dsnow_dt is not None:
                self._set_value(field=state.tendencies.dsnow_dt, value=Float(0.0))
                if state.tendencies.dsnow_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dsnow_dt, summand=state.tendencies.dsnow_dt_shallow_convection)
                if state.tendencies.dsnow_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dsnow_dt, summand=state.tendencies.dsnow_dt_macrophysics)
                if state.tendencies.dsnow_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dsnow_dt, summand=state.tendencies.dsnow_dt_microphysics)

            # graupel mixing ratio
            if state.tendencies.dgraupel_dt is not None:
                self._set_value(field=state.tendencies.dgraupel_dt, value=Float(0.0))
                if state.tendencies.dgraupel_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dgraupel_dt, summand=state.tendencies.dgraupel_dt_macrophysics)
                if state.tendencies.dgraupel_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dgraupel_dt, summand=state.tendencies.dgraupel_dt_microphysics)

            # cloud fraction
            if state.tendencies.dtotal_cloud_fraciton_dt is not None:
                self._set_value(field=state.tendencies.dtotal_cloud_fraciton_dt, value=Float(0.0))
                if state.tendencies.dtotal_cloud_fraciton_dt_deep_convection is not None:
                    self._add_to_self(field=state.tendencies.dtotal_cloud_fraciton_dt, summand=state.tendencies.dtotal_cloud_fraciton_dt_deep_convection)
                if state.tendencies.dtotal_cloud_fraciton_dt_shallow_convection is not None:
                    self._add_to_self(field=state.tendencies.dtotal_cloud_fraciton_dt, summand=state.tendencies.dtotal_cloud_fraciton_dt_shallow_convection)
                if state.tendencies.dtotal_cloud_fraciton_dt_macrophysics is not None:
                    self._add_to_self(field=state.tendencies.dtotal_cloud_fraciton_dt, summand=state.tendencies.dtotal_cloud_fraciton_dt_macrophysics)
                if state.tendencies.dtotal_cloud_fraciton_dt_microphysics is not None:
                    self._add_to_self(field=state.tendencies.dtotal_cloud_fraciton_dt, summand=state.tendencies.dtotal_cloud_fraciton_dt_microphysics)

            # layer pressure thickness
            if state.tendencies.dlayer_pressure_thickness_dt is not None:
                self._set_value(field=state.tendencies.dlayer_pressure_thickness_dt, value=Float(0.0))
                if state.precipitation_flux.ice_convection is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.ice_convection
                    )
                if state.precipitation_flux.ice_shallow_convection is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.ice_shallow_convection
                    )
                if state.precipitation_flux.ice_anvil is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.ice_anvil
                    )
                if state.precipitation_flux.ice_nonanvil_large_scale is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.ice_nonanvil_large_scale
                    )
                if state.precipitation_flux.liquid_convection is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.liquid_convection
                    )
                if state.precipitation_flux.liquid_shallow_convection is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.liquid_shallow_convection
                    )
                if state.precipitation_flux.liquid_anvil is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.liquid_anvil
                    )
                if state.precipitation_flux.liquid_nonanvil_large_scale is not None:
                    self._update_dlayer_pressure_thickness_dt(
                        dlayer_pressure_thickness_dt=state.tendencies.dlayer_pressure_thickness_dt, field=state.precipitation_flux.liquid_nonanvil_large_scale
                    )

            # non-convective liquid flux
            if state.precipitation_flux.liquid_nonconvective is not None:
                self._set_value(field=state.precipitation_flux.liquid_nonconvective, value=Float(0.0))
                if state.precipitation_flux.liquid_anvil is not None:
                    self._add_to_self(field=state.precipitation_flux.liquid_anvil, summand=state.precipitation_flux.liquid_anvil)
                if state.precipitation_flux.liquid_nonanvil_large_scale is not None:
                    self._add_to_self(field=state.precipitation_flux.liquid_anvil, summand=state.precipitation_flux.liquid_nonanvil_large_scale)

            # non-convective ice flux
            if state.precipitation_flux.ice_nonconvective is not None:
                self._set_value(field=state.precipitation_flux.ice_nonconvective, value=Float(0.0))
                if state.precipitation_flux.ice_anvil is not None:
                    self._add_to_self(field=state.precipitation_flux.ice_anvil, summand=state.precipitation_flux.ice_anvil)
                if state.precipitation_flux.ice_nonanvil_large_scale is not None:
                    self._add_to_self(field=state.precipitation_flux.ice_anvil, summand=state.precipitation_flux.ice_nonanvil_large_scale)

            # convective rain
            if state.precipitation_at_surface.rain_from_all_convection is not None:
                self._set_value(field=state.precipitation_at_surface.rain_from_all_convection, value=Float(0.0))
                if state.precipitation_at_surface.rain_from_deep_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_all_convection, summand=state.precipitation_at_surface.rain_from_deep_convection)
                if state.precipitation_at_surface.rain_from_deep_convection is not None:
                    self._add_to_self_2d(
                        field=state.precipitation_at_surface.rain_from_all_convection, summand=state.precipitation_at_surface.rain_from_shallow_convection
                    )
                if state.precipitation_at_surface.rain_from_deep_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_all_convection, summand=state.precipitation_at_surface.rain_from_GF_convection)

            # large scale rain
            if state.precipitation_at_surface.rain_from_all_large_scale is not None:
                self._set_value(field=state.precipitation_at_surface.rain_from_all_large_scale, value=Float(0.0))
                if state.precipitation_at_surface.rain_from_large_scale_nonanvil is not None:
                    self._add_to_self_2d(
                        field=state.precipitation_at_surface.rain_from_all_large_scale, summand=state.precipitation_at_surface.rain_from_large_scale_nonanvil
                    )
                if state.precipitation_at_surface.rain_from_large_scale_anvil is not None:
                    self._add_to_self_2d(
                        field=state.precipitation_at_surface.rain_from_all_large_scale, summand=state.precipitation_at_surface.rain_from_large_scale_anvil
                    )

            # total rain
            if state.precipitation_at_surface.rainfall is not None:
                self._set_value(field=state.precipitation_at_surface.rainfall, value=Float(0.0))
                if state.precipitation_at_surface.rain_from_large_scale_nonanvil is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.rain_from_large_scale_nonanvil)
                if state.precipitation_at_surface.rain_from_large_scale_anvil is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.rain_from_large_scale_nonanvil)
                if state.precipitation_at_surface.rain_from_deep_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.rain_from_deep_convection)
                if state.precipitation_at_surface.rain_from_shallow_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.rain_from_shallow_convection)
                if state.precipitation_at_surface.rain_from_GF_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.rain_from_GF_convection)

            # total snow
            if state.precipitation_at_surface.snowfall is not None:
                self._set_value(field=state.precipitation_at_surface.snowfall, value=Float(0.0))
                if state.precipitation_at_surface.large_scale_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.large_scale_snow)
                if state.precipitation_at_surface.anvil_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.anvil_snow)
                if state.precipitation_at_surface.deep_convection_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.deep_convection_snow)
                if state.precipitation_at_surface.shallow_convection_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rainfall, summand=state.precipitation_at_surface.shallow_convection_snow)

            # all deep convective precip (rain + snow + ice + freezing rain)
            if state.precipitation_at_surface.rain_from_deep_convection is not None:
                if state.precipitation_at_surface.rain_from_GF_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_deep_convection, summand=state.precipitation_at_surface.rain_from_GF_convection)
                if state.precipitation_at_surface.deep_convection_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_deep_convection, summand=state.precipitation_at_surface.deep_convection_snow)

            # all large-scale precip (rain + snow)
            if state.precipitation_at_surface.rain_from_large_scale_nonanvil is not None:
                if state.precipitation_at_surface.large_scale_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_large_scale_nonanvil, summand=state.precipitation_at_surface.large_scale_snow)
                if state.precipitation_at_surface.icefall is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_large_scale_nonanvil, summand=state.precipitation_at_surface.icefall)
                if state.precipitation_at_surface.freezing_rainfall is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_large_scale_nonanvil, summand=state.precipitation_at_surface.freezing_rainfall)

            # all anvil precip (rain + snow)
            if state.precipitation_at_surface.rain_from_large_scale_anvil is not None:
                if state.precipitation_at_surface.anvil_snow is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.rain_from_large_scale_anvil, summand=state.precipitation_at_surface.anvil_snow)

            if state.precipitation_at_surface.rain_from_shallow_convection is not None:
                if state.precipitation_at_surface.shallow_convection_snow is not None:
                    self._add_to_self_2d(
                        field=state.precipitation_at_surface.rain_from_shallow_convection, summand=state.precipitation_at_surface.shallow_convection_snow
                    )

            # total - all precip
            if state.precipitation_at_surface.total_precipitation is not None:
                self._set_value(field=state.precipitation_at_surface.total_precipitation, value=Float(0.0))
                if state.precipitation_at_surface.rain_from_large_scale_nonanvil is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.total_precipitation, summand=state.precipitation_at_surface.rain_from_large_scale_nonanvil)
                if state.precipitation_at_surface.rain_from_large_scale_anvil is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.total_precipitation, summand=state.precipitation_at_surface.rain_from_large_scale_anvil)
                if state.precipitation_at_surface.rain_from_deep_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.total_precipitation, summand=state.precipitation_at_surface.rain_from_deep_convection)
                if state.precipitation_at_surface.rain_from_shallow_convection is not None:
                    self._add_to_self_2d(field=state.precipitation_at_surface.total_precipitation, summand=state.precipitation_at_surface.rain_from_shallow_convection)
                self._ensure_non_negative_2d(field=state.precipitation_at_surface.total_precipitation)

            # diagnosed stratiform precip
            if state.precipitation_at_surface.total_stratiform_precipitation is not None:
                if self._config.CONVECTION_FRACTION_MAX > self._config.CONVECTION_FRACTION_MIN:
                    self._subtract_2d(
                        minuend=self._locals.my_value_is_1_2d, subtrahend=state.convective_diagnostics.convection_fraction, difference=self._locals.temporary_2d
                    )
                    self._multiply_2d(
                        factor_1=self._locals.temporary_2d,
                        factor_2=state.precipitation_at_surface.total_precipitation,
                        product=state.precipitation_at_surface.total_stratiform_precipitation,
                    )
                else:
                    self._set_value(field=state.precipitation_at_surface.total_stratiform_precipitation, value=Float(0.0))
                    if state.precipitation_at_surface.rain_from_large_scale_nonanvil is not None:
                        self._add_to_self_2d(
                            field=state.precipitation_at_surface.total_stratiform_precipitation, summand=state.precipitation_at_surface.rain_from_large_scale_nonanvil
                        )
                    if state.precipitation_at_surface.rain_from_large_scale_anvil is not None:
                        self._add_to_self_2d(
                            field=state.precipitation_at_surface.total_stratiform_precipitation, summand=state.precipitation_at_surface.rain_from_large_scale_anvil
                        )
                        self._ensure_non_negative_2d(field=state.precipitation_at_surface.total_stratiform_precipitation)

            if state.precipitation_at_surface.total_convective_precipitation is not None:
                if self._config.CONVECTION_FRACTION_MAX > self._config.CONVECTION_FRACTION_MIN:
                    self._multiply_2d(
                        factor_1=state.convective_diagnostics.convection_fraction,
                        factor_2=state.precipitation_at_surface.total_precipitation,
                        product=state.precipitation_at_surface.total_convective_precipitation,
                    )
                else:
                    self._set_value(field=state.precipitation_at_surface.total_convective_precipitation, value=Float(0.0))
                    if state.precipitation_at_surface.rain_from_deep_convection is not None:
                        self._add_to_self_2d(
                            field=state.precipitation_at_surface.total_convective_precipitation, summand=state.precipitation_at_surface.rain_from_deep_convection
                        )
                    if state.precipitation_at_surface.rain_from_shallow_convection is not None:
                        self._add_to_self_2d(
                            field=state.precipitation_at_surface.total_convective_precipitation, summand=state.precipitation_at_surface.rain_from_shallow_convection
                        )
                        self._ensure_non_negative_2d(field=state.precipitation_at_surface.total_convective_precipitation)

            # diagnostic precip types
            # TODO need to make sure that icefall and freezing_rainfall remain unallocated at the previous read (in the "all large-scale precip (rain + snow)" section),
            # if they are not already allocated; BUT will be alloced at this line, if not already allocated. AFAIK we do not currently have a way to do this
            if self._config.DIAGNOSE_PRECIP_TYPE or self._config.UPDATE_PRECIP_TYPE:
                raise ValueError(f"Precip type diagnostic has not been implemented. Will implement when needed - it should be easy.")

            # get Kuchera snow:rain ratios
            self._get_Kuchera_ratios(
                p_mb=self._locals.p_mb,
                t=state.atmospheric_state.t,
                kuchera_ratio=state.diagnostics.kuchera_snow_to_liquid_ratio,
            )

            # accumulated precip totals (mm), apply Kuchera ratio for snow
            if state.precipitation_at_surface.snowfall_total is not None:
                self._compute_snowfall_total(
                    snowfall_total=state.precipitation_at_surface.snowfall_total,
                    snowfall=state.precipitation_at_surface.snowfall,
                    icefall=state.precipitation_at_surface.icefall,
                    kuchera_ratio=state.diagnostics.kuchera_snow_to_liquid_ratio,
                )

            if state.precipitation_at_surface.precipitation_total is not None:
                self._multiply_by_dt_moist_2d(input=state.precipitation_at_surface.total_precipitation, output=state.precipitation_at_surface.precipitation_total)

            if state.cloud_condensates.total_liquid is not None:
                self._add(
                    summand_1=state.cloud_condensates.large_scale_liquid, summand_2=state.cloud_condensates.convective_liquid, sum=state.cloud_condensates.total_liquid
                )

            if state.cloud_condensates.total_ice is not None:
                self._add(summand_1=state.cloud_condensates.large_scale_ice, summand_2=state.cloud_condensates.convective_ice, sum=state.cloud_condensates.total_ice)

            if state.cloud_condensates.total_water is not None:
                self._add_to_self(field=state.cloud_condensates.total_liquid, summand=state.cloud_condensates.convective_ice)
                self._add_to_self(field=state.cloud_condensates.total_liquid, summand=state.cloud_condensates.convective_liquid)
                self._add_to_self(field=state.cloud_condensates.total_liquid, summand=state.cloud_condensates.large_scale_ice)
                self._add_to_self(field=state.cloud_condensates.total_liquid, summand=state.cloud_condensates.large_scale_liquid)

            # cloud condensate exports
            if state.state_at_output.large_scale_ice is not None:
                self._copy(input=state.cloud_condensates.large_scale_ice, output=state.state_at_output.large_scale_ice)

            if state.state_at_output.large_scale_liquid is not None:
                self._copy(input=state.cloud_condensates.large_scale_liquid, output=state.state_at_output.large_scale_liquid)

            if state.state_at_output.convective_ice is not None:
                self._copy(input=state.cloud_condensates.convective_ice, output=state.state_at_output.convective_ice)

            if state.state_at_output.convective_liquid is not None:
                self._copy(input=state.cloud_condensates.convective_liquid, output=state.state_at_output.convective_liquid)

            # fill wind, temperature, dry static energy, and relative humidity exports needed for SYNCTQ
            if state.state_at_output.u is not None:
                self._copy(input=state.atmospheric_state.u, output=state.state_at_output.u)

            if state.state_at_output.v is not None:
                self._copy(input=state.atmospheric_state.v, output=state.state_at_output.v)

            if state.state_at_output.t is not None:
                self._copy(input=state.atmospheric_state.t, output=state.state_at_output.t)

            if state.state_at_output.specific_humidity is not None:
                self._copy(input=state.cloud_condensates.specific_humidity, output=state.state_at_output.specific_humidity)

            if state.state_at_output.pt is not None:
                self._divide(dividend=state.atmospheric_state.t, divisor=self._locals.p_kappa, quotient=state.state_at_output.specific_humidity)

            if state.state_at_output.dry_static_energy is not None:
                self._compute_dry_static_energy(
                    t=state.atmospheric_state.t,
                    layer_height_above_surface=self._locals.layer_height_above_surface,
                    edge_height_above_surface=self._locals.edge_height_above_surface,
                    dry_static_energy=state.state_at_output.dry_static_energy,
                )

            if state.state_at_output.relative_humidity is not None:
                self._compute_relative_humidty(
                    t=state.atmospheric_state.t,
                    p_mb=self._locals.p_mb,
                    specific_humidity=state.cloud_condensates.specific_humidity,
                    relative_humidity=state.state_at_output.relative_humidity,
                    ese=self._saturation_tables.ese,
                    esx=self._saturation_tables.esx,
                )

            # other diagnostic outputs
            if state.diagnostics.condensed_water_path is not None:
                self._compute_condensed_water_path(
                    convective_ice=state.cloud_condensates.convective_ice,
                    convective_liquid=state.cloud_condensates.convective_liquid,
                    large_scale_liquid=state.cloud_condensates.large_scale_liquid,
                    large_scale_ice=state.cloud_condensates.large_scale_ice,
                    mass=self._locals.mass,
                    condensed_water_path=state.diagnostics.condensed_water_path,
                )

            if state.diagnostics.liquid_water_path is not None:
                self._compute_liquid_water_path(
                    convective_liquid=state.cloud_condensates.convective_liquid,
                    large_scale_liquid=state.cloud_condensates.large_scale_liquid,
                    liquid_water_path=state.diagnostics.liquid_water_path,
                )

            if state.diagnostics.ice_water_path is not None:
                self._compute_ice_water_path(
                    convective_ice=state.cloud_condensates.convective_ice,
                    large_scale_ice=state.cloud_condensates.large_scale_ice,
                    ice_water_path=state.diagnostics.ice_water_path,
                )

            if state.diagnostics.total_precipitable_water is not None:
                self._compute_total_precipitable_water(
                    specific_humidity=state.cloud_condensates.specific_humidity,
                    mass=self._locals.mass,
                    total_precipitable_water=state.diagnostics.total_precipitable_water,
                )

            # lightning
            if state.diagnostics.lightning_flash_rate is not None:
                self._set_value_2d(field=state.diagnostics.lightning_flash_rate, value=Float(0.0))

        else:  # alarm is NOT ringing
            # compute derived states
            self._compute_derived_state(
                mass=self._locals.mass,
                p_interface=state.atmospheric_state.p_interface,
                p_interface_mb=self._locals.p_interface_mb,
                p_mb=self._locals.p_mb,
                p_kappa_interface=self._locals.p_kappa_interface,
                p_kappa=self._locals.p_kappa,
                z_interface=state.atmospheric_state.z_interface,
                edge_height_above_surface=self._locals.edge_height_above_surface,
                layer_height_above_surface=self._locals.layer_height_above_surface,
                layer_thickness=self._locals.layer_thickness,
                t=state.atmospheric_state.t,
                ese=self._saturation_tables.ese,
                esx=self._saturation_tables.esx,
                saturation_specific_humidity=self._locals.saturation_specific_humidity,
                dsaturation_specific_humidity=self._locals.dsaturation_specific_humidity,
            )

            # fill wind, temperature, dry static energy, and relative humidity exports needed for SYNCTQ
            if state.state_at_output.u is not None:
                self._copy(input=state.atmospheric_state.u, output=state.state_at_output.u)

            if state.state_at_output.v is not None:
                self._copy(input=state.atmospheric_state.v, output=state.state_at_output.v)

            if state.state_at_output.t is not None:
                self._copy(input=state.atmospheric_state.t, output=state.state_at_output.t)

            if state.state_at_output.specific_humidity is not None:
                self._copy(input=state.cloud_condensates.specific_humidity, output=state.state_at_output.specific_humidity)

            if state.state_at_output.pt is not None:
                self._divide(dividend=state.atmospheric_state.t, divisor=self._locals.p_kappa, quotient=state.state_at_output.specific_humidity)

            if state.state_at_output.dry_static_energy is not None:
                self._compute_dry_static_energy(
                    t=state.atmospheric_state.t,
                    layer_height_above_surface=self._locals.layer_height_above_surface,
                    edge_height_above_surface=self._locals.edge_height_above_surface,
                    dry_static_energy=state.state_at_output.dry_static_energy,
                )

            if state.state_at_output.relative_humidity is not None:
                self._compute_relative_humidty(
                    t=state.atmospheric_state.t,
                    p_mb=self._locals.p_mb,
                    specific_humidity=state.cloud_condensates.specific_humidity,
                    relative_humidity=state.state_at_output.relative_humidity,
                    ese=self._saturation_tables.ese,
                    esx=self._saturation_tables.esx,
                )
