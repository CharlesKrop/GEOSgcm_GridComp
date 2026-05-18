from pyMoist.state import MoistState
from pyMoist.convection_tracers import ConvectionTracers
from ndsl.stencils import set_value, copy, add, divide, add_to_self, add_to_self_2d, multiply_2d, subtract_2d, copy_2d, set_value_2D, multiply
from pyMoist.saturation_tables import get_saturation_vapor_pressure_table, compute_saturation_specific_humidity, GlobalTable_saturation_tables
from ndsl import StencilFactory, QuantityFactory, NDSLRuntime
from ndsl.dsl.gt4py import computation, PARALLEL, interval, FORWARD, K, BACKWARD
from ndsl.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK, Float, IntFieldIJ
import pyMoist.constants as constants
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from pyMoist.shared.incloud_processes import fix_mixing_ratio, buoyancy_1, Buoyancy2
from pyMoist.locals import MoistLocals
from pyMoist.config import MoistConfig
from pyMoist.convection import UWConfiguration, UWState, ComputeUwshcuInv, GF2020, GF2020Config, GF2020State, GF2020CumulusParameterizationConfig
from pyMoist.microphysics import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.aerosol_activation import AerosolActivation


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


def compute_vertical_velocity(omega: FloatField, p_mb: FloatField, t: FloatField, w: FloatField):
    with computation(PARALLEL), interval(...):
        w = -omega / (constants.MAPL_GRAV * p_mb * 100.0 / (constants.MAPL_RGAS * t))


def non_neural_network_aerosol_activation(land_fraction: FloatFieldIJ, concentration: FloatField):
    from __externals__ import CCN_LAND, CCN_OCEAN

    with computation(PARALLEL), interval(...):
        concentration = (CCN_LAND * land_fraction + CCN_OCEAN * (1.0 - land_fraction)) * 1.0e6  # number/m3


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
        self._compute_vertical_velocity = stencil_factory.from_dims_halo(func=compute_vertical_velocity, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._non_neural_network_aerosol_activation = stencil_factory.from_dims_halo(
            func=non_neural_network_aerosol_activation, compute_dims=[I_DIM, J_DIM, K_DIM], externals={"CCN_LAND": config.CCN_LAND, "CCN_OCEAN": config.CCN_OCEAN}
        )
        self._multiply = stencil_factory.from_dims_halo(func=multiply, compute_dims=[I_DIM, J_DIM, K_DIM])
        # TODO n_modes probably shouldn't be a constant
        self._aerosol_activation = AerosolActivation(
            stencil_factory=stencil_factory, quantity_factory=quantity_factory, n_modes=constants.N_MODES, nn_ocean=config.CCN_OCEAN, nn_land=config.CCN_LAND
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
            self._gfdl1m_config = GFDL1MConfig(
                LPHYS_HYDROSTATIC=config.HYDROSTATIC,
                LHYDROSTATIC=config.PHYS_HYDROSTATIC,
                DT_MOIST=config.DT_MOIST,
                MP_TIME=config.MAX_MICROPHYSICS_TIMESTEP,
                T_MIN=config.MIN_MELTING_T,
                T_SUB=config.MIN_SUBLIMATION_T,
                TAU_R2G=config.TAU_RAIN_TO_GRAUPEL,
                TAU_SMLT=config.TAU_SNOWMELT,
                TAU_G2R=config.TAU_GRAUPEL_TO_RAIN,
                DW_LAND=config.SUBGRID_VARIABILITY_LAND,
                DW_OCEAN=config.SUBGRID_VARIABILITY_OCEAN,
                VI_FAC=config.fall_velocity_ice_factor,
                VR_FAC=config.fall_velocity_rain_factor,
                VS_FAC=config.fall_velocity_snow_factor,
                VG_FAC=config.fall_velocity_graupel_factor,
                QL_MLT=config.MAX_WATER_FROM_ICE,
                DO_QA=config.DO_INLINE_CLOUD_FRACTION,
                FIX_NEGATIVE=config.FIX_NEGATIVE_WATER_SPECIES,
                VI_MAX=config.MAX_FALL_SPEED_ICE,
                VS_MAX=config.MAX_FALL_SPEED_SNOW,
                VG_MAX=config.MAX_FALL_SPEED_GRAUPEL,
                VR_MAX=config.MAX_FALL_SPEED_RAIN,
                QS_MLT=config.MAX_WATER_FROM_SNOW,
                QS0_CRT=config.SNOW_TO_GRAUPEL_THRESHOLD,
                QI_GEN=config.MAX_CLOUD_ICE_GEN,
                QL0_MAX=config.MAX_CLOUD_WATER,
                QI0_MAX=config.MAX_CLOUD_ICE,
                QI0_CRT=config.ICE_TO_SNOW_THRESHOLD,
                QR0_CRT=config.RAIN_TO_OTHER_THRESHOLD,
                FAST_SAT_ADJ=config.FAST_SAT_ADJ,
                RH_INC=config.RH_INCREMENT_INCLOUD,
                RH_INS=config.RH_INCREMENT_RAIN,
                RH_INR=config.RH_INCREMENT_SNOW,
                CONST_VI=config.USE_CONSTANT_ICE_FALL_SPEED,
                CONST_VS=config.USE_CONSTANT_SNOW_FALL_SPEED,
                CONST_VG=config.USE_CONSTANT_GRAUPEL_FALL_SPEED,
                CONST_VR=config.USE_CONSTANT_RAIN_FALL_SPEED,
                USE_CCN=config.USE_CCN,
                RTHRESHU=config.CRITICAL_CLOUD_DROP_RADIUS_1,
                RTHRESHS=config.CRITICAL_CLOUD_DROP_RADIUS_2,
                CCN_L=config.CCN_LAND,
                CCN_O=config.CCN_OCEAN,
                QC_CRT=config.CRITICAL_PARTIAL_CLOUDY_MIXING_RATIO,
                TAU_G2V=config.TAU_GRAUPEL_TO_VAPOR,
                TAU_V2G=config.TAU_VAPOR_TO_GRAUPEL,
                TAU_S2V=config.TAU_SNOW_TO_VAPOR,
                TAU_V2S=config.TAU_VAPOR_TO_SNOW,
                TAU_REVP=config.TAU_RAIN_EVAPORATION,
                TAU_FRZ=config.TAU_RAIN_FREEZING,
                DO_BIGG=config.DO_BIGG,
                DO_EVAP=config.DO_EVAPORATION,
                DO_SUBL=config.DO_SUBLIMATION,
                SAT_ADJ0=config.SATURATION_ADJUSTMENT_FACTOR,
                C_PIACR=config.ACCRETION_EFF_RAIN_TO_ICE,
                TAU_IMLT=config.TAU_ICE_MELTING,
                TAU_V2L=config.TAU_VAPOR_TO_LIQUID,
                TAU_L2V=config.TAU_LIQUID_TO_VAPOR,
                TAU_I2V=config.TAU_ICE_TO_VAPOR,
                TAU_I2S=config.TAU_ICE_TO_SNOW,
                TAU_L2R=config.TAU_LIQUID_TO_RAIN,
                QI_LIM=config.CLOUD_ICE_LIMITER,
                QL_GEN=config.MAX_CLOUD_LIQUID_GEN,
                C_PAUT=config.LIQUID_TO_RAIN_AUTOCONVERSION,
                C_PSACI=config.ACCRETION_EFF_ICE_TO_SNOW,
                C_PGACS=config.ACCRETION_EFF_SNOW_TO_GRAUPEL,
                C_PGACI=config.ACCRETION_EFF_ICE_TO_GRAUPEL,
                Z_SLOPE_LIQ=config.USE_LINEAR_MONO_SLOPE_LIQUID,
                Z_SLOPE_ICE=config.USE_LINEAR_MONO_SLOPE_ICE,
                PROG_CCN=config.DO_PROGNOSTIC_CCN,
                C_CRACW=config.ACCRETION_EFF_RAIN,
                ALIN=config.ALIN,
                CLIN=config.CLIN,
                PRECIPRAD=config.INCLUDE_PRECIP_IN_CLOUD_FRACTION,
                CLD_MIN=config.MIN_CLOUD_FRACTION,
                USE_PPM=config.USE_PPM,
                MONO_PROF=config.USE_MONO_PROF_PPM,
                DO_SEDI_HEAT=config.DO_SEDI_HEAT_TRANSPORT,
                SEDI_TRANSPORT=config.DO_SEDI_MOMENTUM_TRANSPORT,
                DO_SEDI_W=config.DO_SEDI_W_TRANSPORT,
                DE_ICE=config.DE_ICE,
                ICLOUD_F=config.CLOUD_SCHEME,
                IRAIN_F=config.LIQUID_TO_RAIN_AUTOCONVERSION_SCHEME,
                MP_PRINT=config.DEBUG_PRINT,
                LMELTFRZ=config.MELT_FREEZE,
                USE_BERGERON=config.USE_BERGERON,
                TURNRHCRIT_PARAM=config.TURNRHCRIT_PARAM,
                PDFSHAPE=config.PDFSHAPE,
                ANV_ICEFALL=config.ANVIL_ICEFALL,
                LS_ICEFALL=config.LARGE_SCALE_ICEFALL,
                LIQ_RADII_PARAM=config.LIQUID_RADII_PARAM,
                ICE_RADII_PARAM=config.ICE_RADII_PARAM,
                FAC_RI=config.ICE_RADIUS_FACTOR,
                MIN_RI=config.MIN_ICE_RADIUS,
                MAX_RI=config.MAX_ICE_RADIUS,
                FAC_RL=config.LIQUID_RADIUS_FACTOR,
                MIN_RL=config.MIN_LIQUID_RADIUS,
                MAX_RL=config.MAX_LIQUID_RADIUS,
                CCW_EVAP_EFF=config.CCW_EVAP_EFF,
                CCI_EVAP_EFF=config.CCI_EVAP_EFF,
            )
            self._gfdl1m_state = GFDL1MState.zeros(quantity_factory)
            self._gfdl1m = GFDL1M(stencil_factory=stencil_factory, quantity_factory=quantity_factory, config=self._gfdl1m_config)
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
            if self._config.USE_AEROSOL_NEURAL_NETWORK:
                if self._config.HYDROSTATIC:
                    self._compute_vertical_velocity(omega=state.atmospheric_state.omega, p_mb=self._locals.p_mb, t=state.atmospheric_state.t, w=self._locals.temporary_3d)
                else:
                    self._copy(input=state.atmospheric_state.vertical_motion.velocity, output=self._locals.temporary_3d)

                self._multiply(factor_1=self._locals.p_mb, factor_2=100.0, product=self._locals.p_pascals)
                self._aerosol_activation(
                    aero_dgn=NEED_FLORIANS_FEATURE_FOR_NAMED_DDIM_INDEX,
                    aero_num=NEED_FLORIANS_FEATURE_FOR_NAMED_DDIM_INDEX,
                    aero_sigma=NEED_FLORIANS_FEATURE_FOR_NAMED_DDIM_INDEX,
                    aero_hygroscopicity=NEED_FLORIANS_FEATURE_FOR_NAMED_DDIM_INDEX,
                    t=state.atmospheric_state.t,
                    plo=self._locals.p_mb,
                    qicn=state.cloud_condensates.convective_ice,
                    qils=state.cloud_condensates.large_scale_ice,
                    qlcn=state.cloud_condensates.convective_liquid,
                    qlls=state.cloud_condensates.convective_ice,
                    frland=state.surface_conditions.land_fraction,
                    nwfa=state.cloud_condensates.number_concentration_water_friendly_aerosols,
                    vvel=self._locals.temporary_3d,
                    tke=state.atmospheric_state.turbulent_kinetic_energy,
                    nactl=state.cloud_condensates.convective_liquid,
                    nacti=state.cloud_condensates.ice_concentration,
                )
            else:
                self._non_neural_network_aerosol_activation(land_fraction=state.surface_conditions.land_fraction, concentration=state.cloud_condensates.ice_concentration)
                self._non_neural_network_aerosol_activation(
                    land_fraction=state.surface_conditions.land_fraction, concentration=state.cloud_condensates.liquid_concentration
                )

            # export concentrations
            self._export_concentration(
                field=state.cloud_condensates.liquid_ccn_concentration, factor=Float(1.0e-1), output=state.cloud_condensates.liquid_ccn_concentration
            )
            self._export_concentration(field=state.cloud_condensates.ice_ccn_concentration, factor=Float(1.0e-1), output=state.cloud_condensates.ice_ccn_concentration)

            # run convection and microphysics
            if self._config.SHALLOW_MID_DEEP:
                if self._config.SHALLOW_CONVECTION_OPTION == "UW":
                    moist_to_uw_map = [
                        (lambda m: m.atmospheric_state.p_interface, lambda g: g.input.PLE, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.atmospheric_state.z_interface, lambda g: g.input.ZLE, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.grid_data.area, lambda g: g.input.AREA, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_liquid, lambda g: g.input.QLLS, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_ice, lambda g: g.input.QILS, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_liquid, lambda g: g.input.QLCN, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_ice, lambda g: g.input.QICN, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.levels.pbl_level_shallow_convection, lambda g: g.input.kpbl_inv, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.land_fraction, lambda g: g.input.frland, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.turbulent_kinetic_energy, lambda g: g.input.tke_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.surface_conditions.sensible_heat_flux, lambda g: g.input.shfx, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.surface_evaporation, lambda g: g.input.evap, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.u, lambda g: g.input_output.u0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.v, lambda g: g.input_output.v0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.specific_humidity, lambda g: g.input_output.qv0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.t, lambda g: g.input_output.t0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.cumulus_scale_height_from_shallow_convection, lambda g: g.input_output.cush, [I_DIM, J_DIM]),
                        (lambda m: m.precipitation_at_surface.rain_from_GF_convection, lambda g: g.input_output.cnvtr, [I_DIM, J_DIM]),
                        (lambda m: m.diagnostics.turbulent_kinetic_energy_fraction_from_vertical_velocity, lambda g: g.output.RKFRE, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_shallow_updraft_detrained, lambda g: g.output.MFD_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.entrained_sink_shallow_convection, lambda g: g.output.QLENT_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.entrained_ice_sink_shallow_convection, lambda g: g.output.QIENT_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convection_diagnostics.mass_flux_shallow_updraft_interface, lambda g: g.output.umf_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_mass_detrained_shallow_convection, lambda g: g.output.dcm_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.total_water_flux_shallow_convection_interface, lambda g: g.output.qtflx_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (
                            lambda m: m.convective_diagnostics.liquid_static_energy_flux_shallow_convection_interface,
                            lambda g: g.output.slflx_inv,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (lambda m: m.convective_diagnostics.u_flux_shallow_convection_interface, lambda g: g.output.uflx_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.v_flux_shallow_convection_interface, lambda g: g.output.vflx_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.tendencies.dtotal_cloud_fraciton_dt_shallow_convection, lambda g: g.output.DQADT_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dspecific_humidity_dt_shallow_convection, lambda g: g.output.qvten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dliquid_dt_shallow_convection, lambda g: g.output.qlten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dice_dt_shallow_convection, lambda g: g.output.qiten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_shallow_convection, lambda g: g.output.tten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.du_dt_shallow_convection, lambda g: g.output.uten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dv_dt_shallow_convection, lambda g: g.output.vten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.drain_dt_shallow_convection, lambda g: g.output.qrten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dsnow_dt_shallow_convection, lambda g: g.output.qsten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_fraction_shallow_convection, lambda g: g.output.cufrc_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_shallow_convection, lambda g: g.output.fer_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_detrainment_rate_shallow_convection, lambda g: g.output.fdr_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.number_liquid_droplet_shallow_convection, lambda g: g.output.ndrop_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.number_ice_crystal_shallow_convection, lambda g: g.output.nice_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.subsidence_liquid_shallow_convection_interface, lambda g: g.output.qlsub_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.subsidence_ice_shallow_convection_interface, lambda g: g.output.qisub_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.total_liquid, lambda g: g.output.ql0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.total_ice, lambda g: g.output.qi0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.source_air_t_perturbation_shallow_convection_interface, lambda g: g.output.tpert_out, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.source_air_humidity_perturbation_shallow_convection_interface, lambda g: g.output.qpert_out, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.detrained_ice_shallow_convection_interface, lambda g: g.output.qidet_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.detrained_liquid_shallow_convection_interface, lambda g: g.output.qldet_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.total_cumulative_mass_flux, lambda g: g.output.CNV_MFC, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.total_detraining_mass_flux, lambda g: g.output.CNV_MFD, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.precipitation_flux.shallow_convective_rain, lambda g: g.output.SHLW_PRC3, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.precipitation_flux.shallow_convective_snow, lambda g: g.output.SHLW_SNO3, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.total_column_water_shallow_convection, lambda g: g.output.SC_QT, [I_DIM, J_DIM]),
                        (lambda m: m.tendencies.total_column_moist_static_energy_shallow_convection, lambda g: g.output.SC_MSE, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.scale_height_shallow_convection, lambda g: g.output.CUSH_SC, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensates.convective_cloud_fraction, lambda g: g.input_output.CLCN, [I_DIM, J_DIM, K_DIM]),
                    ]

                    for source_getter, destination_getter, dims in moist_to_uw_map:
                        source = source_getter(state)
                        destination = destination_getter(self._uw_state)

                        if source is None:
                            destination = None
                        else:
                            if K_DIM in dims or K_INTERFACE_DIM in dims:
                                self._copy(input=source, output=destination)
                            else:
                                self._copy_2d(input=source, output=destination)

                    self._uw(self._uw_state)
                if self._config.CONVECTION_OPTION == "RAS":
                    raise ValueError(f"{self._config.CONVECTION_OPTION} convection not implemented. Please choose a different option.")
                if self._config.CONVECTION_OPTION == "GF":
                    moist_to_gf2020_map: list[tuple] = [
                        (lambda m: m.grid_data.area, lambda g: g.area, [I_DIM, J_DIM]),
                        (lambda m: m.grid_data.latitude, lambda g: g.latitude, [I_DIM, J_DIM]),
                        (lambda m: m.grid_data.longitude, lambda g: g.longitude, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.p_interface, lambda g: g.p_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.atmospheric_state.u, lambda g: g.u, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.t, lambda g: g.t, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.v, lambda g: g.v, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.vertical_motion.velocity, lambda g: g.w, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.omega, lambda g: g.omega, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.surface_conditions.t_2m, lambda g: g.t_2m, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.specific_humidity_2m, lambda g: g.specific_humidity_2m, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.t_surface_air, lambda g: g.t_surface, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.specific_humidity_surface, lambda g: g.specific_humidity_surface, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensates.specific_humidity, lambda g: g.vapor, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_liquid, lambda g: g.convective_liquid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_ice, lambda g: g.convective_ice, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_cloud_fraction, lambda g: g.convective_cloud_fraction, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_liquid, lambda g: g.large_scale_liquid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_ice, lambda g: g.large_scale_ice, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_cloud_fraction, lambda g: g.large_scale_cloud_fraction, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.ice_fraction_in_convective_tower, lambda g: g.ice_fraction_in_convective_tower, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.p_interface, lambda g: g.p_interface_timestep_start, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.state_before_dynamics.t, lambda g: g.t_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.u, lambda g: g.u_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.v, lambda g: g.v_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.specific_humidity, lambda g: g.vapor_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.z_interface, lambda g: g.geopotential_height_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.surface_conditions.geopotential_height, lambda g: g.geopotential_height_surface, [I_DIM, J_DIM]),
                        (lambda m: m.levels.pbl_level, lambda g: g.pbl_level, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_fraction, lambda g: g.convection_fraction, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.surface_type, lambda g: g.surface_type, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.stochastic_factor, lambda g: g.seed_convection, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.land_fraction, lambda g: g.land_fraction, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.scalar_diffusivity_interface, lambda g: g.scalar_diffusivity, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.buoyancy_surface_parcel, lambda g: g.buoyancy, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.precipitation_at_surface.rain_from_GF_convection, lambda g: g.convective_precipitation_GF, [I_DIM, J_DIM]),
                        (lambda m: m.precipitation_flux.convective_precipitation_from_RAS, lambda g: g.convective_precipitation_RAS, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_rainwater_source, lambda g: g.convective_rainwater_source, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.surface_conditions.sensible_heat_flux, lambda g: g.sensible_heat_flux, [I_DIM, J_DIM]),
                        (
                            lambda m: m.convective_diagnostics.total_water_flux_deep_convection_interface,
                            lambda g: g.total_water_flux_deep_convection_interface,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (
                            lambda m: m.convective_diagnostics.convective_precipitation_evaporation,
                            lambda g: g.sublimation_of_convective_precipitation,
                            [I_DIM, J_DIM, K_DIM],
                        ),
                        (
                            lambda m: m.convective_diacnostics.convective_precipitation_sublimation,
                            lambda g: g.evaporation_of_convective_precipitation,
                            [I_DIM, J_DIM, K_DIM],
                        ),
                        (lambda m: m.precipitation_flux.ice_convection, lambda g: g.ice_precip_flux_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.precipitation_flux.liquid_convection, lambda g: g.liquid_precip_flux_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.surface_conditions.surface_evaporation, lambda g: g.evaporation, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensated.convective_condensate_source, lambda g: g.convective_condensate_source, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.convective_condensate_grid_mean, lambda g: g.convective_condensate_grid_mean, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.entrainment_parameter, lambda g: g.entrainment_parameter, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate, lambda g: g.lateral_entrainment_rate, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_shallow, lambda g: g.lateral_entrainment_rate_shallow, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_mid, lambda g: g.lateral_entrainment_rate_mid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_deep, lambda g: g.lateral_entrainment_rate_deep, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.updraft_areal_fraction, lambda g: g.updraft_areal_fraction, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.updraft_vertical_velocity, lambda g: g.updraft_vertical_velocity, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_shortwave, lambda g: g.dtdt_shortwave, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_longwave, lambda g: g.dtdt_longwave, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dspecific_humidity_dt_pbl, lambda g: g.dspecific_humiditydt_pbl, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_pbl, lambda g: g.dtdt_pbl, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_from_dynamics, lambda g: g.dtdt_from_dynamics, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dspecific_humidity_dt_from_dynamics, lambda g: g.dvapordt_from_dynamics, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.sigma_mid, lambda g: g.sigma_mid, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.sigma_deep, lambda g: g.sigma_deep, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensated.initial_total_precipitable_water, lambda g: g.total_precipitable_water_initial, [I_DIM, J_DIM]),
                        (
                            lambda m: m.cloud_condensated.initial_total_precipitable_water_saturation,
                            lambda g: g.saturation_total_precipitable_water_initial,
                            [I_DIM, J_DIM],
                        ),
                        (lambda m: m.tendencies.dspecific_humidity_dt_deep_convection, lambda g: g.dvapordt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_deep_convection, lambda g: g.dtdt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.du_dt_deep_convection, lambda g: g.dudt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dv_dt_deep_convection, lambda g: g.dvdt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dliquid_dt_deep_convection, lambda g: g.dliquiddt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dice_dt_deep_convection, lambda g: g.dicedt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dtotal_cloud_fraciton_dt_deep_convection, lambda g: g.dcloudfractiondt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.pressure_shallow_convective_cloud_top, lambda g: g.pressure_shallow_convective_cloud_top, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.pressure_mid_convective_cloud_top, lambda g: g.pressure_mid_convective_cloud_top, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.pressure_deep_convective_cloud_top, lambda g: g.pressure_deep_convective_cloud_top, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_shallow, lambda g: g.mass_flux_shallow, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_mid, lambda g: g.mass_flux_mid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_deep_updraft, lambda g: g.mass_flux_deep_updraft, [I_DIM, J_DIM, K_DIM]),
                        (
                            lambda m: m.convective_diagnostics.mass_flux_deep_updraft_interface,
                            lambda g: g.mass_flux_deep_updraft_interface,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (lambda m: m.convective_diagnostics.mass_flux_deep_updraft_detrained, lambda g: g.mass_flux_deep_updraft_detrained, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_deep_downdraft, lambda g: g.mass_flux_deep_downdraft, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base, lambda g: g.mass_flux_cloud_base, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base_shallow, lambda g: g.mass_flux_cloud_base_shallow, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base_mid, lambda g: g.mass_flux_cloud_base_mid, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base_deep, lambda g: g.mass_flux_cloud_base_deep, [I_DIM, J_DIM]),
                        (
                            lambda m: m.convective_diagnostics.total_cumulative_mass_flux,
                            lambda g: g.total_cumulative_mass_flux_interface,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (lambda m: m.convective_diagnostics.total_detraining_mass_flux, lambda g: g.total_detraining_mass_flux, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.convection_code_shallow, lambda g: g.convection_code_shallow, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_code_mid, lambda g: g.convection_code_mid, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_code_deep, lambda g: g.convection_code_deep, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_0, lambda g: g.cloud_workfunction_0, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_1, lambda g: g.cloud_workfunction_1, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_2, lambda g: g.cloud_workfunction_2, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_3, lambda g: g.cloud_workfunction_3, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_1_pbl, lambda g: g.cloud_workfunction_1_pbl, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_1_cin, lambda g: g.cloud_workfunction_1_cin, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.pbl_time_scale, lambda g: g.pbl_time_scale, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cape_removal_time_scale, lambda g: g.cape_removal_time_scale, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.lightning_density, lambda g: g.lightning_density, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_tracer, lambda g: g.convection_tracer, [I_DIM, J_DIM, K_DIM]),
                    ]

                    for source_getter, destination_getter, dims in moist_to_gf2020_map:
                        source = source_getter(state)
                        destination = destination_getter(self._gf2020_state)

                        if source is None:
                            destination = None
                        else:
                            if K_DIM in dims or K_INTERFACE_DIM in dims:
                                self._copy(input=source, output=destination)
                            else:
                                self._copy_2d(input=source, output=destination)

                    self._gf2020(state=self._gf2020_state, convection_tracers=self._convection_tracers)
            else:
                if self._config.CONVECTION_OPTION == "RAS":
                    run_RAS = True
                if self._config.CONVECTION_OPTION == "GF":
                    moist_to_gf2020_map: list[tuple] = [
                        (lambda m: m.grid_data.area, lambda g: g.area, [I_DIM, J_DIM]),
                        (lambda m: m.grid_data.latitude, lambda g: g.latitude, [I_DIM, J_DIM]),
                        (lambda m: m.grid_data.longitude, lambda g: g.longitude, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.p_interface, lambda g: g.p_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.atmospheric_state.u, lambda g: g.u, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.t, lambda g: g.t, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.v, lambda g: g.v, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.vertical_motion.velocity, lambda g: g.w, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.omega, lambda g: g.omega, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.surface_conditions.t_2m, lambda g: g.t_2m, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.specific_humidity_2m, lambda g: g.specific_humidity_2m, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.t_surface_air, lambda g: g.t_surface, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.specific_humidity_surface, lambda g: g.specific_humidity_surface, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensates.specific_humidity, lambda g: g.vapor, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_liquid, lambda g: g.convective_liquid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_ice, lambda g: g.convective_ice, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_cloud_fraction, lambda g: g.convective_cloud_fraction, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_liquid, lambda g: g.large_scale_liquid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_ice, lambda g: g.large_scale_ice, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_cloud_fraction, lambda g: g.large_scale_cloud_fraction, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.ice_fraction_in_convective_tower, lambda g: g.ice_fraction_in_convective_tower, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.p_interface, lambda g: g.p_interface_timestep_start, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.state_before_dynamics.t, lambda g: g.t_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.u, lambda g: g.u_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.v, lambda g: g.v_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.state_before_dynamics.specific_humidity, lambda g: g.vapor_timestep_start, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.z_interface, lambda g: g.geopotential_height_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.surface_conditions.geopotential_height, lambda g: g.geopotential_height_surface, [I_DIM, J_DIM]),
                        (lambda m: m.levels.pbl_level, lambda g: g.pbl_level, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_fraction, lambda g: g.convection_fraction, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.surface_type, lambda g: g.surface_type, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.stochastic_factor, lambda g: g.seed_convection, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.land_fraction, lambda g: g.land_fraction, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.scalar_diffusivity_interface, lambda g: g.scalar_diffusivity, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.buoyancy_surface_parcel, lambda g: g.buoyancy, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.precipitation_at_surface.rain_from_GF_convection, lambda g: g.convective_precipitation_GF, [I_DIM, J_DIM]),
                        (lambda m: m.precipitation_flux.convective_precipitation_from_RAS, lambda g: g.convective_precipitation_RAS, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_rainwater_source, lambda g: g.convective_rainwater_source, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.surface_conditions.sensible_heat_flux, lambda g: g.sensible_heat_flux, [I_DIM, J_DIM]),
                        (
                            lambda m: m.convective_diagnostics.total_water_flux_deep_convection_interface,
                            lambda g: g.total_water_flux_deep_convection_interface,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (
                            lambda m: m.convective_diagnostics.convective_precipitation_evaporation,
                            lambda g: g.sublimation_of_convective_precipitation,
                            [I_DIM, J_DIM, K_DIM],
                        ),
                        (
                            lambda m: m.convective_diacnostics.convective_precipitation_sublimation,
                            lambda g: g.evaporation_of_convective_precipitation,
                            [I_DIM, J_DIM, K_DIM],
                        ),
                        (lambda m: m.precipitation_flux.ice_convection, lambda g: g.ice_precip_flux_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.precipitation_flux.liquid_convection, lambda g: g.liquid_precip_flux_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.surface_conditions.surface_evaporation, lambda g: g.evaporation, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensated.convective_condensate_source, lambda g: g.convective_condensate_source, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.convective_condensate_grid_mean, lambda g: g.convective_condensate_grid_mean, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.entrainment_parameter, lambda g: g.entrainment_parameter, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate, lambda g: g.lateral_entrainment_rate, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_shallow, lambda g: g.lateral_entrainment_rate_shallow, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_mid, lambda g: g.lateral_entrainment_rate_mid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_deep, lambda g: g.lateral_entrainment_rate_deep, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.updraft_areal_fraction, lambda g: g.updraft_areal_fraction, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.updraft_vertical_velocity, lambda g: g.updraft_vertical_velocity, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_shortwave, lambda g: g.dtdt_shortwave, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_longwave, lambda g: g.dtdt_longwave, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dspecific_humidity_dt_pbl, lambda g: g.dspecific_humiditydt_pbl, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_pbl, lambda g: g.dtdt_pbl, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_from_dynamics, lambda g: g.dtdt_from_dynamics, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dspecific_humidity_dt_from_dynamics, lambda g: g.dvapordt_from_dynamics, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.sigma_mid, lambda g: g.sigma_mid, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.sigma_deep, lambda g: g.sigma_deep, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensated.initial_total_precipitable_water, lambda g: g.total_precipitable_water_initial, [I_DIM, J_DIM]),
                        (
                            lambda m: m.cloud_condensated.initial_total_precipitable_water_saturation,
                            lambda g: g.saturation_total_precipitable_water_initial,
                            [I_DIM, J_DIM],
                        ),
                        (lambda m: m.tendencies.dspecific_humidity_dt_deep_convection, lambda g: g.dvapordt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_deep_convection, lambda g: g.dtdt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.du_dt_deep_convection, lambda g: g.dudt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dv_dt_deep_convection, lambda g: g.dvdt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dliquid_dt_deep_convection, lambda g: g.dliquiddt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dice_dt_deep_convection, lambda g: g.dicedt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dtotal_cloud_fraciton_dt_deep_convection, lambda g: g.dcloudfractiondt_deep_convection, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.pressure_shallow_convective_cloud_top, lambda g: g.pressure_shallow_convective_cloud_top, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.pressure_mid_convective_cloud_top, lambda g: g.pressure_mid_convective_cloud_top, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.pressure_deep_convective_cloud_top, lambda g: g.pressure_deep_convective_cloud_top, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_shallow, lambda g: g.mass_flux_shallow, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_mid, lambda g: g.mass_flux_mid, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_deep_updraft, lambda g: g.mass_flux_deep_updraft, [I_DIM, J_DIM, K_DIM]),
                        (
                            lambda m: m.convective_diagnostics.mass_flux_deep_updraft_interface,
                            lambda g: g.mass_flux_deep_updraft_interface,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (lambda m: m.convective_diagnostics.mass_flux_deep_updraft_detrained, lambda g: g.mass_flux_deep_updraft_detrained, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_deep_downdraft, lambda g: g.mass_flux_deep_downdraft, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base, lambda g: g.mass_flux_cloud_base, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base_shallow, lambda g: g.mass_flux_cloud_base_shallow, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base_mid, lambda g: g.mass_flux_cloud_base_mid, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_cloud_base_deep, lambda g: g.mass_flux_cloud_base_deep, [I_DIM, J_DIM]),
                        (
                            lambda m: m.convective_diagnostics.total_cumulative_mass_flux,
                            lambda g: g.total_cumulative_mass_flux_interface,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (lambda m: m.convective_diagnostics.total_detraining_mass_flux, lambda g: g.total_detraining_mass_flux, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.convection_code_shallow, lambda g: g.convection_code_shallow, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_code_mid, lambda g: g.convection_code_mid, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_code_deep, lambda g: g.convection_code_deep, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_0, lambda g: g.cloud_workfunction_0, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_1, lambda g: g.cloud_workfunction_1, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_2, lambda g: g.cloud_workfunction_2, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_3, lambda g: g.cloud_workfunction_3, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_1_pbl, lambda g: g.cloud_workfunction_1_pbl, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_workfunction_1_cin, lambda g: g.cloud_workfunction_1_cin, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.pbl_time_scale, lambda g: g.pbl_time_scale, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.cape_removal_time_scale, lambda g: g.cape_removal_time_scale, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.lightning_density, lambda g: g.lightning_density, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.convection_tracer, lambda g: g.convection_tracer, [I_DIM, J_DIM, K_DIM]),
                    ]

                    for source_getter, destination_getter, dims in moist_to_gf2020_map:
                        source = source_getter(state)
                        destination = destination_getter(self._gf2020_state)

                        if source is None:
                            destination = None
                        else:
                            if K_DIM in dims or K_INTERFACE_DIM in dims:
                                self._copy(input=source, output=destination)
                            else:
                                self._copy_2d(input=source, output=destination)

                    self._gf2020(state=self._gf2020_state, convection_tracers=self._convection_tracers)
                if self._config.SHALLOW_CONVECTION_OPTION == "UW":
                    moist_to_uw_map = [
                        (lambda m: m.atmospheric_state.p_interface, lambda g: g.input.PLE, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.atmospheric_state.z_interface, lambda g: g.input.ZLE, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.grid_data.area, lambda g: g.input.AREA, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_liquid, lambda g: g.input.QLLS, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.large_scale_ice, lambda g: g.input.QILS, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_liquid, lambda g: g.input.QLCN, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.convective_ice, lambda g: g.input.QICN, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.levels.pbl_level_shallow_convection, lambda g: g.input.kpbl_inv, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.land_fraction, lambda g: g.input.frland, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.turbulent_kinetic_energy, lambda g: g.input.tke_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.surface_conditions.sensible_heat_flux, lambda g: g.input.shfx, [I_DIM, J_DIM]),
                        (lambda m: m.surface_conditions.surface_evaporation, lambda g: g.input.evap, [I_DIM, J_DIM]),
                        (lambda m: m.atmospheric_state.u, lambda g: g.input_output.u0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.v, lambda g: g.input_output.v0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.specific_humidity, lambda g: g.input_output.qv0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.atmospheric_state.t, lambda g: g.input_output.t0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.cumulus_scale_height_from_shallow_convection, lambda g: g.input_output.cush, [I_DIM, J_DIM]),
                        (lambda m: m.precipitation_at_surface.rain_from_GF_convection, lambda g: g.input_output.cnvtr, [I_DIM, J_DIM]),
                        (lambda m: m.diagnostics.turbulent_kinetic_energy_fraction_from_vertical_velocity, lambda g: g.output.RKFRE, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.mass_flux_shallow_updraft_detrained, lambda g: g.output.MFD_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.entrained_sink_shallow_convection, lambda g: g.output.QLENT_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.entrained_ice_sink_shallow_convection, lambda g: g.output.QIENT_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convection_diagnostics.mass_flux_shallow_updraft_interface, lambda g: g.output.umf_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_mass_detrained_shallow_convection, lambda g: g.output.dcm_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.total_water_flux_shallow_convection_interface, lambda g: g.output.qtflx_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (
                            lambda m: m.convective_diagnostics.liquid_static_energy_flux_shallow_convection_interface,
                            lambda g: g.output.slflx_inv,
                            [I_DIM, J_DIM, K_INTERFACE_DIM],
                        ),
                        (lambda m: m.convective_diagnostics.u_flux_shallow_convection_interface, lambda g: g.output.uflx_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.v_flux_shallow_convection_interface, lambda g: g.output.vflx_inv, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.tendencies.dtotal_cloud_fraciton_dt_shallow_convection, lambda g: g.output.DQADT_SC, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dspecific_humidity_dt_shallow_convection, lambda g: g.output.qvten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dliquid_dt_shallow_convection, lambda g: g.output.qlten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dice_dt_shallow_convection, lambda g: g.output.qiten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dt_dt_shallow_convection, lambda g: g.output.tten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.du_dt_shallow_convection, lambda g: g.output.uten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dv_dt_shallow_convection, lambda g: g.output.vten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.drain_dt_shallow_convection, lambda g: g.output.qrten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.dsnow_dt_shallow_convection, lambda g: g.output.qsten_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.cloud_fraction_shallow_convection, lambda g: g.output.cufrc_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_entrainment_rate_shallow_convection, lambda g: g.output.fer_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.lateral_detrainment_rate_shallow_convection, lambda g: g.output.fdr_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.number_liquid_droplet_shallow_convection, lambda g: g.output.ndrop_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensated.number_ice_crystal_shallow_convection, lambda g: g.output.nice_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.subsidence_liquid_shallow_convection_interface, lambda g: g.output.qlsub_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.subsidence_ice_shallow_convection_interface, lambda g: g.output.qisub_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.total_liquid, lambda g: g.output.ql0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.cloud_condensates.total_ice, lambda g: g.output.qi0_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.source_air_t_perturbation_shallow_convection_interface, lambda g: g.output.tpert_out, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.source_air_humidity_perturbation_shallow_convection_interface, lambda g: g.output.qpert_out, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.detrained_ice_shallow_convection_interface, lambda g: g.output.qidet_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.detrained_liquid_shallow_convection_interface, lambda g: g.output.qldet_inv, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.convective_diagnostics.total_cumulative_mass_flux, lambda g: g.output.CNV_MFC, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                        (lambda m: m.convective_diagnostics.total_detraining_mass_flux, lambda g: g.output.CNV_MFD, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.precipitation_flux.shallow_convective_rain, lambda g: g.output.SHLW_PRC3, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.precipitation_flux.shallow_convective_snow, lambda g: g.output.SHLW_SNO3, [I_DIM, J_DIM, K_DIM]),
                        (lambda m: m.tendencies.total_column_water_shallow_convection, lambda g: g.output.SC_QT, [I_DIM, J_DIM]),
                        (lambda m: m.tendencies.total_column_moist_static_energy_shallow_convection, lambda g: g.output.SC_MSE, [I_DIM, J_DIM]),
                        (lambda m: m.convective_diagnostics.scale_height_shallow_convection, lambda g: g.output.CUSH_SC, [I_DIM, J_DIM]),
                        (lambda m: m.cloud_condensates.convective_cloud_fraction, lambda g: g.input_output.CLCN, [I_DIM, J_DIM, K_DIM]),
                    ]

                    for source_getter, destination_getter, dims in moist_to_uw_map:
                        source = source_getter(state)
                        destination = destination_getter(self._uw_state)

                        if source is None:
                            destination = None
                        else:
                            if K_DIM in dims or K_INTERFACE_DIM in dims:
                                self._copy(input=source, output=destination)
                            else:
                                self._copy_2d(input=source, output=destination)

                    self._uw(self._uw_state)

            if self._config.CLOUD_MICROPHYSICS_OPTION == "BACM_1M":
                raise ValueError(f"{self._config.CLOUD_MICROPHYSICS_OPTION} microphysics not implemented. Please choose a different option.")
            if self._config.CLOUD_MICROPHYSICS_OPTION == "GFDL_1M":
                moist_to_gfdl1m_map: list[tuple] = [
                    (lambda m: m.grid_data.area, lambda g: g.area, [I_DIM, J_DIM]),
                    (lambda m: m.atmospheric_state.z_interface, lambda g: g.z_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.atmospheric_state.p_interface, lambda g: g.p_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.atmospheric_state.t, lambda g: g.t, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.atmospheric_state.u, lambda g: g.u, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.atmospheric_state.v, lambda g: g.v, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.surface_conditions.land_fraction, lambda g: g.land_fraction, [I_DIM, J_DIM]),
                    (lambda m: m.atmospheric_state.scalar_diffusivity_interface, lambda g: g.scalar_diffusivity_interface, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.micrphysics_diagnostics.pdf_first_plume_fractional_area, lambda g: g.pdf_first_plume_fractional_area, [I_DIM, J_DIM, K_DIM]),
                    (
                        lambda m: m.micrphysics_diagnostics.covariance_liquid_water_static_energy_and_total_water_specific_humidity,
                        lambda g: g.covariance_liquid_water_static_energy_and_total_water_specific_humidity,
                        [I_DIM, J_DIM, K_DIM],
                    ),
                    (lambda m: m.surface_conditions.t_surface, lambda g: g.surface_temperature, [I_DIM, J_DIM]),
                    (lambda m: m.surface_conditions.sensible_heat_flux, lambda g: g.sensible_heat_flux, [I_DIM, J_DIM]),
                    (lambda m: m.atmospheric_state.omega, lambda g: g.omega, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.convection_diagnostics.convection_fraction, lambda g: g.convection_fraction, [I_DIM, J_DIM]),
                    (lambda m: m.surface_conditions.surface_type, lambda g: g.surface_type, [I_DIM, J_DIM]),
                    (lambda m: m.microphysics_diagnostics.cloud_liquid_evaporation, lambda g: g.cloud_liquid_evaporation, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.microphysics_diagnostics.cloud_ice_evaporation, lambda g: g.cloud_ice_evaporation, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.precipitation_at_surface.icefall, lambda g: g.icefall, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.freezing_rainfall, lambda g: g.freezing_rainfall, [I_DIM, J_DIM]),
                    (lambda m: m.microphysics_diagnostics.relative_humidity_after_pdf, lambda g: g.relative_humidity_after_pdf, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.microphysics_diagnostics.buoyancy_flux, lambda g: g.buoyancy_flux, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.microphysics_diagnostics.liquid_water_flux, lambda g: g.liquid_water_flux, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.microphysics_diagnostics.hydrostatic_pdf_iterations, lambda g: g.hydrostatic_pdf_iterations, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.diagnostics.lower_tropospheric_stability, lambda g: g.lower_tropospheric_stability, [I_DIM, J_DIM]),
                    (lambda m: m.diagnostics.estimated_inversion_strength, lambda g: g.estimated_inversion_strength, [I_DIM, J_DIM]),
                    (lambda m: m.diagnostics.lcl_height, lambda g: g.lcl_height, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_flux.shallow_convective_rain, lambda g: g.shallow_convective_rain, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.precipitation_flux.shallow_convective_snow, lambda g: g.shallow_convective_snow, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.microphysics_diagnostics.critical_relative_humidity_for_pdf, lambda g: g.critical_relative_humidity_for_pdf, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.microphysics_diagnostics.large_scale_rainwater_source, lambda g: g.large_scale_rainwater_source, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.atmospheric_state.vertical_motion.velocity, lambda g: g.vertical_motion.velocity, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.atmospheric_state.vertical_motion.variance, lambda g: g.vertical_motion.variance, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.atmospheric_state.vertical_motion.third_moment, lambda g: g.vertical_motion.third_moment, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.liquid_water_static_energy.flux, lambda g: g.liquid_water_static_energy.flux, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.liquid_water_static_energy.variance, lambda g: g.liquid_water_static_energy.variance, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.liquid_water_static_energy.third_moment, lambda g: g.liquid_water_static_energy.third_moment, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.total_water.flux, lambda g: g.total_water.flux, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.total_water.variance, lambda g: g.total_water.variance, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.total_water.third_moment, lambda g: g.total_water.third_moment, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.specific_humidity, lambda g: g.mixing_ratio.vapor, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.rain, lambda g: g.mixing_ratio.rain, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.snow, lambda g: g.mixing_ratio.snow, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.graupel, lambda g: g.mixing_ratio.graupel, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.large_scale_liquid, lambda g: g.mixing_ratio.large_scale_liquid, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.large_scale_ice, lambda g: g.mixing_ratio.large_scale_ice, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.convective_liquid, lambda g: g.mixing_ratio.convective_liquid, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.convective_ice, lambda g: g.mixing_ratio.convective_ice, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.convective_cloud_fraction, lambda g: g.cloud_fraction.convective, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.large_scale_cloud_fraction, lambda g: g.cloud_fraction.large_scale, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.liquid_concentration, lambda g: g.concentration.liquid, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.ice_concentration, lambda g: g.concentration.ice, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.specific_humidity, lambda g: g.radiation_field.vapor, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.liquid, lambda g: g.radiation_field.liquid, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.ice, lambda g: g.radiation_field.ice, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.rain, lambda g: g.radiation_field.rain, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.graupel, lambda g: g.radiation_field.graupel, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.snow, lambda g: g.radiation_field.snow, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.cloud_fraction, lambda g: g.radiation_field.cloud_fraction, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.ice_particle_effective_radius, lambda g: g.cloud_particle_effective_radius.ice, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.radiation_state.liquid_particle_effective_radius, lambda g: g.cloud_particle_effective_radius.liquid, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.precipitation_at_surface.falling_ice, lambda g: g.precipitation_at_surface.ice, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.falling_graupel, lambda g: g.precipitation_at_surface.graupel, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.falling_rain, lambda g: g.precipitation_at_surface.rain, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.falling_snow, lambda g: g.precipitation_at_surface.snow, [I_DIM, J_DIM]),
                    (
                        lambda m: m.precipitation_at_surface.rain_from_shallow_convection,
                        lambda g: g.precipitation_at_surface.shallow_convective_precipitation,
                        [I_DIM, J_DIM],
                    ),
                    (lambda m: m.precipitation_at_surface.rain_from_deep_convection, lambda g: g.precipitation_at_surface.deep_convective_precipitation, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.rain_from_large_scale_anvil, lambda g: g.precipitation_at_surface.anvil_precipitation, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.shallow_convection_snow, lambda g: g.precipitation_at_surface.shallow_convective_snow, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.deep_convection_snow, lambda g: g.precipitation_at_surface.deep_convective_snow, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.anvil_snow, lambda g: g.precipitation_at_surface.anvil_snow, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.rain_from_large_scale_nonanvil, lambda g: g.non_anvil_large_scale.precip, [I_DIM, J_DIM]),
                    (lambda m: m.precipitation_at_surface.large_scale_snow, lambda g: g.non_anvil_large_scale.snow, [I_DIM, J_DIM]),
                    (
                        lambda m: m.microphysics_diagnostics.nonanvil_large_scale_precipitation_evaporation,
                        lambda g: g.non_anvil_large_scale.evaporation,
                        [I_DIM, J_DIM, K_DIM],
                    ),
                    (
                        lambda m: m.microphysics_diagnostics.nonanvil_large_scale_precipitation_sublimation,
                        lambda g: g.non_anvil_large_scale.precip,
                        [I_DIM, J_DIM, K_DIM],
                    ),
                    (lambda m: m.precipitation_flux.liquid_nonanvil_large_scale, lambda g: g.non_anvil_large_scale.precip, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.precipitation_flux.ice_nonanvil_large_scale, lambda g: g.non_anvil_large_scale.ice_precip_flux, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.precipitation_flux.liquid_anvil, lambda g: g.anvil.liquid_precip_flux, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.precipitation_flux.ice_anvil, lambda g: g.anvil.ice_precip_flux, [I_DIM, J_DIM, K_INTERFACE_DIM]),
                    (lambda m: m.tendencies.dtotal_cloud_fraciton_dt_macrophysics, lambda g: g.tendencies.dcloud_fractiondt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dspecific_humidity_dt_macrophysics, lambda g: g.tendencies.dvapordt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dice_dt_macrophysics, lambda g: g.tendencies.dicedt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dliquid_dt_macrophysics, lambda g: g.tendencies.dliquiddt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.drain_dt_macrophysics, lambda g: g.tendencies.draindt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dgraupel_dt_macrophysics, lambda g: g.tendencies.dgraupeldt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dsnow_dt_macrophysics, lambda g: g.tendencies.dsnowdt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.du_dt_macrophysics, lambda g: g.tendencies.dudt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dv_dt_macrophysics, lambda g: g.tendencies.dvdt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dt_dt_macrophysics, lambda g: g.tendencies.dtdt_macro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dtotal_cloud_fraciton_dt_microphysics, lambda g: g.tendencies.dcloud_fractiondt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dspecific_humidity_dt_microphysics, lambda g: g.tendencies.dvapordt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dice_dt_microphysics, lambda g: g.tendencies.dicedt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dliquid_dt_microphysics, lambda g: g.tendencies.dliquiddt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.drain_dt_microphysics, lambda g: g.tendencies.draindt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dgraupel_dt_microphysics, lambda g: g.tendencies.dgraupeldt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dsnow_dt_microphysics, lambda g: g.tendencies.dsnowdt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.du_dt_microphysics, lambda g: g.tendencies.dudt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dv_dt_microphysics, lambda g: g.tendencies.dvdt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dt_dt_microphysics, lambda g: g.tendencies.dtdt_micro, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.tendencies.dt_dt_friction_pressure_weighted, lambda g: g.tendencies.dtdt_friction_pressure_weighted, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.mass_fraction_suspended_rain, lambda g: g.mass_fraction.suspended_rain, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.mass_fraction_suspended_graupel, lambda g: g.mass_fraction.suspended_graupel, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.cloud_condensates.mass_fraction_suspended_snow, lambda g: g.mass_fraction.suspended_snow, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.diagnostics.radar.simulated_reflectivity, lambda g: g.radar.simulated_reflectivity, [I_DIM, J_DIM, K_DIM]),
                    (lambda m: m.diagnostics.radar.maximum_composite_reflectivity, lambda g: g.radar.maximum_composite_reflectivity, [I_DIM, J_DIM]),
                    (lambda m: m.diagnostics.radar.base_1km_agl_reflectivity, lambda g: g.radar.base_1km_agl_reflectivity, [I_DIM, J_DIM]),
                    (lambda m: m.diagnostics.radar.echo_top_reflectivity, lambda g: g.radar.echo_top_reflectivity, [I_DIM, J_DIM]),
                    (lambda m: m.diagnostics.radar.minus_10c_reflectivity, lambda g: g.radar.minus_10c_reflectivity, [I_DIM, J_DIM]),
                ]

                for source_getter, destination_getter, dims in moist_to_gfdl1m_map:
                    source = source_getter(state)
                    destination = destination_getter(self._gfdl1m_state)

                    if source is None:
                        destination = None
                    else:
                        if K_DIM in dims or K_INTERFACE_DIM in dims:
                            self._copy(input=source, output=destination)
                        else:
                            self._copy_2d(input=source, output=destination)

                self._gfdl1m(self._gfdl1m_state)
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
