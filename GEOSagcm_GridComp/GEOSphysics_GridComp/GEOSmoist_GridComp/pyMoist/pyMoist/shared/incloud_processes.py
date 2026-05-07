"""Stencils and functions called by multiple pyMoist modules.
These functions evaluate various in-cloud microphysical
processes/quantities."""

from ndsl.dsl.gt4py import PARALLEL, GlobalTable, computation, exp, float32, float64, floor, function, interval, log10, round_away_from_zero, sin, FORWARD, BACKWARD, K
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, IntFieldIJ, BoolFieldIJ

import pyMoist.constants as constants
from pyMoist.shared.atmos_recipes import air_density
from ndsl import StencilFactory, QuantityFactory, Quantity, NDSLRuntime
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyMoist.shared.find_levels import find_t_lcl
from pyMoist.saturation_tables import compute_saturation_specific_humidity


@function
def ice_fraction_modis(
    temp: Float,
):
    # Use MODIS polynomial from Hu et al, DOI: (10.1029/2009JD012384)
    tc = max(-46.0, min(temp - constants.MAPL_TICE, 46.0))  # convert to celcius and limit range from -46:46 C
    ptc = 7.6725 + 1.0118 * tc + 0.1422 * tc**2 + 0.0106 * tc**3 + 0.000339 * tc**4 + 0.00000395 * tc**5
    ice_frct = 1.0 - (1.0 / (1.0 + exp(-1 * ptc)))
    return ice_frct


@function
def ice_fraction(
    temp,
    cnv_frc,
    srf_type,
):
    """Determine the ice/liquid fraction.

    Args:
        temp (Float): temperature (Kelvin)
        cnv_frc (Float): convection fraction within the column
        srf_type (Float): surface type

    Returns:
        Float: ice fraction
    """
    # Anvil clouds
    # Anvil-Convective sigmoidal function like figure 6(right)
    # Sigmoidal functions Hu et al 2010, doi:10.1029/2009JD012384
    if constants.ICE_RADII_PARAM == 1:
        # Jason formula
        if temp <= constants.JaT_ICE_ALL:
            icefrct_c = 1.000
        elif temp > constants.JaT_ICE_ALL and temp <= constants.JaT_ICE_MAX:
            icefrct_c = sin(0.5 * constants.MAPL_PI * (1.00 - (temp - constants.JaT_ICE_ALL) / (constants.JaT_ICE_MAX - constants.JaT_ICE_ALL)))
        else:
            icefrct_c = 0.00
    else:
        # Default formula
        if temp <= constants.aT_ICE_ALL:
            icefrct_c = 1.000
        elif temp > constants.aT_ICE_ALL and temp <= constants.aT_ICE_MAX:
            icefrct_c = sin(0.5 * constants.MAPL_PI * (1.00 - (temp - constants.aT_ICE_ALL) / (constants.aT_ICE_MAX - constants.aT_ICE_ALL)))
        else:
            icefrct_c = 0.00
    icefrct_c = max(min(icefrct_c, 1.00), 0.00) ** constants.aICEFRPWR

    # Sigmoidal functions like figure 6b/6c of Hu et al 2010, doi:10.1029/2009JD012384
    srf_type_int = round(srf_type)

    if srf_type_int == 2 or srf_type_int == 3:  # 2 = snow, 3 = ice
        if temp <= constants.iT_ICE_ALL:
            icefrct_m = 1.000
        elif temp > constants.iT_ICE_ALL and temp <= constants.iT_ICE_MAX:
            icefrct_m = sin(0.5 * constants.MAPL_PI * (1.00 - (temp - constants.iT_ICE_ALL) / (constants.iT_ICE_MAX - constants.iT_ICE_ALL)))
        else:
            icefrct_m = 0.00
        icefrct_m = max(min(icefrct_m, 1.00), 0.00) ** constants.iICEFRPWR

    elif srf_type_int == 1:  # land
        if temp <= constants.lT_ICE_ALL:
            icefrct_m = 1.000
        elif temp > constants.lT_ICE_ALL and temp <= constants.lT_ICE_MAX:
            icefrct_m = sin(0.5 * constants.MAPL_PI * (1.00 - (temp - constants.lT_ICE_ALL) / (constants.lT_ICE_MAX - constants.lT_ICE_ALL)))
        else:
            icefrct_m = 0.00
        icefrct_m = max(min(icefrct_m, 1.00), 0.00) ** constants.lICEFRPWR

    elif srf_type_int == 0:  # ocean
        if temp <= constants.oT_ICE_ALL:
            icefrct_m = 1.000
        elif temp > constants.oT_ICE_ALL and temp <= constants.oT_ICE_MAX:
            icefrct_m = sin(0.5 * constants.MAPL_PI * (1.00 - (temp - constants.oT_ICE_ALL) / (constants.oT_ICE_MAX - constants.oT_ICE_ALL)))
        else:
            icefrct_m = 0.00
        icefrct_m = max(min(icefrct_m, 1.00), 0.00) ** constants.oICEFRPWR

    else:
        # unknown surface type detected - you should not be here
        icefrct_m = -999

    ice_frac = icefrct_m * (1.0 - cnv_frc) + icefrct_c * cnv_frc
    return ice_frac


@function
def cloud_effective_radius_liquid(
    pressure: Float,
    temperature: Float,
    liquid_mixing_ratio: Float,
    liquid_concentration: Float,
) -> Float:
    """
    Calculate the effective radius of liquid droplets clouds

    Arguments:
        pressure (in): pressure (millibars)
        temperature (in): temperature (Kelvin)
        liquid_mixing_ratio (in): liquid mixing ratio (kg/kg)
        liquid_concentration (in): liquid cloud droplet concentration (m^-3)

    Returns:
        radius (Float): drop radius
    """
    # Calculate liquid water content
    wc = 1.0e3 * air_density(pressure, temperature) * liquid_mixing_ratio  # air density [g/m3] * liquid cloud mixing ratio [kg/kg]
    # Calculate cloud drop number concentration from the aerosol model + ....
    nnx = max(liquid_concentration * 1.0e-6, 10.0)
    # Calculate Radius in meters [m]
    if constants.LIQ_RADII_PARAM == 1:
        # Jason Version
        radius = min(
            60.0e-6,
            max(
                2.5e-6,
                1.0e-6 * constants.BX * (wc / nnx) ** constants.R13BBETA * constants.ABETA * 6.92,
            ),
        )
    else:
        # [liu&daum, 2000 and 2005. liu et al 2008]
        radius = min(
            60.0e-6,
            max(2.5e-6, 1.0e-6 * constants.LBX * (wc / nnx) ** constants.LBE),
        )
    return radius


@function
def cloud_effective_radius_ice(
    pressure: Float,
    temperature: Float,
    ice_mixing_ratio: Float,
) -> Float:
    """
    Calculate the effective radius of ice particles in clouds

    Arguments:
        pressure (in): pressure (millibars)
        temperature (in): temperature (Kelvin)
        ice_mixing_ratio (in): liquid mixing ratio (kg/kg)

    Returns:
        radius (Float): ice particle radius
    """
    # Calculate ice water content
    wc = 1.0e3 * air_density(pressure, temperature) * ice_mixing_ratio  # air density [g/m3] * ice cloud mixing ratio [kg/kg]
    # Calculate radius in meters [m]
    if constants.ICE_RADII_PARAM == 1:
        # Ice cloud effective radius -- [klaus wyser, 1998]
        if temperature > constants.MAPL_TICE or ice_mixing_ratio <= 0.0:
            bb = -2.0
        else:
            bb = -2.0 + log10(wc / 50.0) * (1.0e-3 * (constants.MAPL_TICE - temperature) ** 1.5)
            # NOTE: there is an issue in this line which causes differences between Fortran and Python
            # the multiplication "-2.0 * log'd result" is performed differently (~60 ULP), despite the log
            # being correct. Needs to be looked into at some point, but not critical for overall performance.
        bb = min(max(bb, -6.0), -2.0)
        radius = 377.4 + 203.3 * bb + 37.91 * bb**2 + 2.3696 * bb**3
        radius = min(150.0e-6, max(5.0e-6, 1.0e-6 * radius))
    else:
        # Ice cloud effective radius ----- [Sun, 2001]
        tc = temperature - constants.MAPL_TICE
        zfsr = 1.2351 + 0.0105 * tc
        aa = 45.8966 * (wc**0.2214)
        bb = 0.79570 * (wc**0.2535)
        radius = zfsr * (aa + bb * (temperature - 83.15))
        radius = min(150.0e-6, max(5.0e-6, 1.0e-6 * radius * 0.64952))
    return radius


def fix_up_clouds(
    mixing_ratio_vapor: FloatField,
    t: FloatField,
    mixing_ratio_large_scale_liquid: FloatField,
    mixing_ratio_large_scale_ice: FloatField,
    large_scale_cloud_fraction: FloatField,
    mixing_ratio_convective_liquid: FloatField,
    mixing_ratio_convective_ice: FloatField,
    convective_cloud_fraction: FloatField,
) -> None:
    """
    Modify various cloud variables to ensure physical consistency.

    Performed in this order:
        If cloud fraction is too low, move all liquid and frozen water to vapor form
            and remove cloud.
        If liquid water is too low, move all liquid water to vapor form.
        If frozen water is too low, move all frozen water to vapor form.
        If total liquid + frozen water is too low, move all water to vapor form
            and remove cloud.

    Parameters:
    mixing_ratio_vapor (inout): water vapor mixing ratio
    t (inout): temperature.
    mixing_ratio_large_scale_liquid (inout): large scale cloud liquid water mixing ratio
    mixing_ratio_large_scale_ice (inout): large scale cloud frozen water mixing ratio
    large_scale_cloud_fraction (inout): large scale cloud fraction
    mixing_ratio_convective_liquid (inout): convective cloud liquid water mixing ratio
    mixing_ratio_convective_ice (inout): convective cloud frozen water mixing ratio
    convective_cloud_fraction (inout): convective cloud fraction
    """
    with computation(PARALLEL), interval(...):
        # fix small convective cloud fraction
        if convective_cloud_fraction < 1.0e-5:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_convective_liquid + mixing_ratio_convective_ice
            t = (
                t
                - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_convective_liquid
                - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_convective_ice
            )
            convective_cloud_fraction = 0.0
            mixing_ratio_convective_liquid = 0.0
            mixing_ratio_convective_ice = 0.0
        # fix small large scale cloud fraction
        if large_scale_cloud_fraction < 1.0e-5:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_large_scale_liquid + mixing_ratio_large_scale_ice
            t = (
                t
                - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_large_scale_liquid
                - (constants.MAPL_LATENT_HEAT_SUBLIMATION / constants.MAPL_CP) * mixing_ratio_large_scale_ice
            )
            large_scale_cloud_fraction = 0.0
            mixing_ratio_large_scale_liquid = 0.0
            mixing_ratio_large_scale_ice = 0.0
        # if large scale liquid water concentration is too low
        if mixing_ratio_large_scale_liquid < 1.0e-8:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_large_scale_liquid
            t = t - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_large_scale_liquid
            mixing_ratio_large_scale_liquid = 0.0
        # if large scale frozen water concentration is too low
        if mixing_ratio_large_scale_ice < 1.0e-8:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_large_scale_ice
            t = t - (constants.MAPL_LATENT_HEAT_SUBLIMATION / constants.MAPL_CP) * mixing_ratio_large_scale_ice
            mixing_ratio_large_scale_ice = 0.0
        # if convective liquid water concentration is too low
        if mixing_ratio_convective_liquid < 1.0e-8:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_convective_liquid
            t = t - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_convective_liquid
            mixing_ratio_convective_liquid = 0.0
        # if convective frozen water concentration is too low
        if mixing_ratio_convective_ice < 1.0e-8:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_convective_ice
            t = t - (constants.MAPL_LATENT_HEAT_SUBLIMATION / constants.MAPL_CP) * mixing_ratio_convective_ice
            mixing_ratio_convective_ice = 0.0
        # if total convective water is too low
        if (mixing_ratio_convective_liquid + mixing_ratio_convective_ice) < 1.0e-8:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_convective_liquid + mixing_ratio_convective_ice
            t = (
                t
                - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_convective_liquid
                - (constants.MAPL_LATENT_HEAT_SUBLIMATION / constants.MAPL_CP) * mixing_ratio_convective_ice
            )
            convective_cloud_fraction = 0.0
            mixing_ratio_convective_liquid = 0.0
            mixing_ratio_convective_ice = 0.0
        # if total large scale water is too low
        if (mixing_ratio_large_scale_liquid + mixing_ratio_large_scale_ice) < 1.0e-8:
            mixing_ratio_vapor = mixing_ratio_vapor + mixing_ratio_large_scale_liquid + mixing_ratio_large_scale_ice
            t = (
                t
                - (constants.MAPL_LATENT_HEAT_VAPORIZATION / constants.MAPL_CP) * mixing_ratio_large_scale_liquid
                - (constants.MAPL_LATENT_HEAT_SUBLIMATION / constants.MAPL_CP) * mixing_ratio_large_scale_ice
            )
            large_scale_cloud_fraction = 0.0
            mixing_ratio_large_scale_liquid = 0.0
            mixing_ratio_large_scale_ice = 0.0


# able of lookup values of radiative effective radius of ice crystals as a function of temperature from
# -94C to 0C for make_ice_number. Taken from WRF RRTMG radiation code where it is attributed to
# Jon Egill Kristjansson and coauthors. This must be built into a custom shape off-grid quantity,
# passed into the stencil which calls make_ice_number with a custom field type (defined below), then
# passed into the function make_ice_number and accessed with .A[index] to function properly
RADIATIVE_EFFECTIVE_RADIUS = [
    5.92779,
    6.26422,
    6.61973,
    6.99539,
    7.39234,
    7.81177,
    8.25496,
    8.72323,
    9.21800,
    9.74075,
    10.2930,
    10.8765,
    11.4929,
    12.1440,
    12.8317,
    13.5581,
    14.2319,
    15.0351,
    15.8799,
    16.7674,
    17.6986,
    18.6744,
    19.6955,
    20.7623,
    21.8757,
    23.0364,
    24.2452,
    25.5034,
    26.8125,
    27.7895,
    28.6450,
    29.4167,
    30.1088,
    30.7306,
    31.2943,
    31.8151,
    32.3077,
    32.7870,
    33.2657,
    33.7540,
    34.2601,
    34.7892,
    35.3442,
    35.9255,
    36.5316,
    37.1602,
    37.8078,
    38.4720,
    39.1508,
    39.8442,
    40.5552,
    41.2912,
    42.0635,
    42.8876,
    43.7863,
    44.7853,
    45.9170,
    47.2165,
    48.7221,
    50.4710,
    52.4980,
    54.8315,
    57.4898,
    60.4785,
    63.7898,
    65.5604,
    71.2885,
    75.4113,
    79.7368,
    84.2351,
    88.8833,
    93.6658,
    98.5739,
    103.603,
    108.752,
    114.025,
    119.424,
    124.954,
    130.630,
    136.457,
    142.446,
    148.608,
    154.956,
    161.503,
    168.262,
    175.248,
    182.473,
    189.952,
    197.699,
    205.728,
    214.055,
    222.694,
    231.661,
    240.971,
    250.639,
]
RADIATIVE_EFFECTIVE_RADIUS_Table_Type = GlobalTable[(Float, len(RADIATIVE_EFFECTIVE_RADIUS))]


@function
def make_ice_number(
    cloud_ice_mixing_ratio,
    t,
    RADIATIVE_EFFECTIVE_RADIUS,
):
    """
    Get the ice crystal number given cloud ice mixing ratio and temperature.
    Returns the number of droplets per kg per m3.

    Args:
        cloud_ice_mixing_ratio (in): units kg/m3
        t (in): units K
        RADIATIVE_EFFECTIVE_RADIUS: table used for calculations

    Returns:
        crystal_number: units number/(kg*m3)

    Developed by H. Barnes @ NOAA/OAR/ESRL/GSL Earth Prediction Advancement Division
    """

    # DEBUG
    crystal_number = 0.0
    # internal constant
    ice_density = 890.0

    if cloud_ice_mixing_ratio == 0.0:
        crystal_number = 0.0

    else:
        # From the model 3D temperature field, subtract 180K for which
        # index value of RADIATIVE_EFFECTIVE_RADIUS as a start.  Value of corr is for
        # interpolating between neighboring values in the table.

        idx_rei = int(t - 180.0)
        idx_rei = min(max(idx_rei, 0), 93)
        corr = t - floor(t)
        reice = RADIATIVE_EFFECTIVE_RADIUS.A[idx_rei] * (1.0 - corr) + RADIATIVE_EFFECTIVE_RADIUS.A[idx_rei + 1] * corr
        deice = 2.0 * reice * 1.0e-6

        internal_lambda = float64(3.0 / deice)

        # value of the dispersion parameter according to Heymsfield et al 2002, Table3.
        t_celcius = t - 273.15

        t_celcius = min(max(t_celcius, -70.0), -15.0)

        if t_celcius > -27.0:
            lambdai = 6.8 * exp(-0.096 * t_celcius)
        else:
            lambdai = 24.8 * exp(-0.049 * t_celcius)

        mui = (0.13 * (lambdai**0.64)) - 2.0

        k = (mui + 3) * (mui * 3) / (mui + 2) / (mui + 1)

        crystal_number = k * cloud_ice_mixing_ratio * internal_lambda * internal_lambda * internal_lambda / (constants.MAPL_PI * ice_density)

    return crystal_number


# table of constants for make_droplet_number, this must be built into a custom shape off-grid quantity,
# passed into the stencil which calls make_droplet_number with a custom field type (defined below), then
# passed into the function make_droplet_number and accessed with .A[index] to function properly
G_RATIO = [24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184, 2730, 3360, 4080, 4896]
G_RATIO_Table_Type = GlobalTable[(Float, len(G_RATIO))]


@function
def make_droplet_number(
    cloud_water_mixing_ratio,
    num_water_friendly_aerosols,
    G_RATIO,
):
    """
    Get the droplet number given cloud water mixing ratio and number of water-friendly aerosols.
    Returns the number of droplets per kg per m3.

    Args:
        cloud_water_mixing_ratio (in): units kg/m3
        num_water_friendly_aerosols (in): units number/kg
        G_RATIO: table used for calculations

    Returns:
        droplet_number: units number/(kg*m3)

    Developed by H. Barnes @ NOAA/OAR/ESRL/GSL Earth Prediction Advancement Division
    """
    am_r = constants.MAPL_PI * 1000.0 / 6.0

    if cloud_water_mixing_ratio <= 0.0:
        droplet_number = 0.0
    else:
        internal_num_water_friendly_aerosols = max(99.0e6, min(num_water_friendly_aerosols, 5.0e10))
        nu_c = max(2, min(round_away_from_zero(2.5e10 / internal_num_water_friendly_aerosols), 15))

        x1 = max(1.0, min(internal_num_water_friendly_aerosols * 1.0e-9, 10.0)) - 1.0
        xDc = (30.0 - x1 * 20.0 / 9.0) * 1.0e-6

        internal_lambda = (float64(4.0) + nu_c) / xDc

        qnc = cloud_water_mixing_ratio / G_RATIO.A[int(nu_c - 1)] * internal_lambda * internal_lambda * internal_lambda / am_r
        droplet_number = float32(qnc)

    return droplet_number


def fix_mixing_ratio(
    mixing_ratio: FloatField,
    mass: FloatField,
    adjustment: FloatFieldIJ,
):
    # predefine two FloatFieldIJ internal fields
    with computation(FORWARD), interval(0, 1):
        k_sum_1: FloatFieldIJ = 0.0
        k_sum_2: FloatFieldIJ = 0.0

    with computation(FORWARD), interval(...):
        k_sum_1 = k_sum_1 + (mixing_ratio * mass)

    with computation(PARALLEL), interval(...):
        if mixing_ratio < 0.0:
            mixing_ratio = 0.0

    with computation(FORWARD), interval(...):
        k_sum_2 = k_sum_2 + (mixing_ratio * mass)

    with computation(PARALLEL), interval(...):
        if k_sum_2 > 0.0:
            factor = (k_sum_2 - k_sum_1) / k_sum_2
            # reduce Q proportionally to the increase in TPW
            mixing_ratio = mixing_ratio * (1.0 - factor)

    with computation(FORWARD), interval(0, 1):
        adjustment = k_sum_2 - k_sum_1


def buoyancy_1(
    t: FloatField,
    layer_height_above_surface: FloatField,
    layer_thickness: FloatField,
    specific_humidity: FloatField,
    saturation_specific_humidity: FloatField,
    dsaturation_specific_humidity: FloatField,
    buoyancy: FloatField,
    cape: FloatFieldIJ,
    cin: FloatFieldIJ,
):
    from __externals__ import k_end

    with computation(FORWARD), interval(-1, None):
        buoyancy = t + (constants.MAPL_GRAV / constants.MAPL_CP) * layer_height_above_surface + (constants.MAPL_ALHL / constants.MAPL_CP) * specific_humidity

    with computation(BACKWARD), interval(...):
        buoyancy = buoyancy.at(K=k_end) - (
            t + (constants.MAPL_GRAV / constants.CP) * layer_height_above_surface + (constants.MAPL_ALHL / constants.MAPL_CP) * saturation_specific_humidity
        )
        buoyancy = constants.MAPL_GRAV * buoyancy / ((1.0 + constants.MAPL_ALHL * dsaturation_specific_humidity) * t)

    with computation(FORWARD), interval(-1, None):
        buoyancy = 0.0

        cape = 0.0
        cin = 0.0

    with computation(FORWARD), interval(0, -1):
        if buoyancy > 0.0:
            cape = cape + buoyancy * layer_thickness
        if buoyancy < 0.0:
            cin = cin + buoyancy * layer_thickness

    with computation(FORWARD), interval(0, 1):
        if cape <= 0.0:
            cape = constants.MAPL_UNDEF
            cin = constants.MAPL_UNDEF


class Buoyancy2(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory):
        super().__init__(stencil_factory)

        self._stencil_factory = stencil_factory

        # functions and stencils to be used only in this class
        def setup(
            t: FloatField,
            specific_humidity: FloatField,
            environment_virtual_t: FloatField,
            parcel_moist_static_energy: FloatFieldIJ,
            parcel_specific_humidity: FloatFieldIJ,
        ):

            with computation(PARALLEL), interval(...):
                environment_virtual_t = t * (1 + constants.MAPL_VIREPS * specific_humidity)

            with computation(FORWARD), interval(0, 1):
                parcel_moist_static_energy = 0.0
                parcel_specific_humidity = 0.0

        def get_cape_cin(
            critical_level: IntFieldIJ,
            t: FloatField,
            p_mb: FloatField,
            layer_height_above_surface: FloatField,
            layer_thickness: FloatField,
            saturation_specific_humidity: FloatField,
            dsaturation_specific_humidity: FloatField,
            environment_virtual_t: FloatField,
            parcel_moist_static_energy: FloatFieldIJ,
            parcel_specific_humidity: FloatFieldIJ,
            buoyancy: FloatField,
            cape: FloatFieldIJ,
            cin: FloatFieldIJ,
            lfc: FloatFieldIJ,
            lnb: FloatFieldIJ,
        ):
            from __externals__ import k_end

            with computation(PARALLEL), interval(...):
                if K <= critical_level:
                    buoyancy = 0.0

            with computation(FORWARD), interval(0, 1):
                parcel_specific_humidity_new = parcel_specific_humidity
                cape = 0.0
                cin = 0.0
                lfc = constants.MAPL_UNDEF
                lnb = constants.MAPL_UNDEF

                # initial parcel temperature at source level (k_end)
                parcel_t: FloatFieldIJ = (
                    parcel_specific_humidity
                    - (constants.MAPL_GRAV / constants.MAPL_CP) * layer_height_above_surface.at(K=k_end)
                    - (constants.MAPL_ALHL / constants.MAPL_CP) * parcel_specific_humidity
                )

            with computation(FORWARD), interval(0, 1):
                above_lcl: BoolFieldIJ = False
                t_lcl = find_t_lcl(t=t, rh=100.0 * parcel_specific_humidity / saturation_specific_humidity.at(K=k_end))
                if parcel_t < t_lcl:
                    above_lcl = True

            with computation(BACKWARD), interval(0, -1):
                if K <= critical_level:
                    # start at level above source air

                    # determine parcel specific_humidity and temperature
                    if above_lcl == False:
                        # new parcel temperature w/o condensation
                        parcel_t = parcel_t - (constants.MAPL_GRAV / constants.MAPL_CP) * (layer_height_above_surface - layer_height_above_surface[0, 0, 1]) / (
                            1.0 + (constants.MAPL_ALHL / constants.MAPL_CP) * dsaturation_specific_humidity
                        )
                        if parcel_t < t_lcl:
                            parcel_t = parcel_t + (constants.MAPL_GRAV / constants.MAPL_CP) * (layer_height_above_surface - layer_height_above_surface[0, 0, 1])
                            above_lcl = True

                    if above_lcl == True and parcel_specific_humidity_new * (constants.MAPL_ALHL / constants.MAPL_CP) > 0.01:
                        # initial guess including condenssation
                        parcel_t = parcel_t - (constants.MAPL_GRAV / constants.MAPL_CP) * (layer_height_above_surface - layer_height_above_surface[0, 0, 1]) / (
                            1.0 + (constants.MAPL_ALHL / constants.MAPL_CP) * dsaturation_specific_humidity
                        )
                        # iterate until parcel is saturated
                        iteration = 1
                        while iteration <= 10:
                            dspecific_humidity = parcel_specific_humidity_new - compute_saturation_specific_humidity(t=parcel_t, p=p_mb)
                            if abs(dsaturation_specific_humidity - (constants.MAPL_ALHL / constants.MAPL_CP)) < 0.01:
                                iteration = 11  # exit loop
                            parcel_t = parcel_t + dspecific_humidity * (constants.MAPL_ALHL / constants.MAPL_CP) / (
                                1.0 + (constants.MAPL_ALHL / constants.MAPL_CP) * dsaturation_specific_humidity
                            )
                            parcel_specific_humidity_new = parcel_specific_humidity_new - dspecific_humidity / (
                                1.0 + (constants.MAPL_ALHL / constants.MAPL_CP) * dsaturation_specific_humidity
                            )
                            iteration += 1

                    parcel_t = (
                        parcel_moist_static_energy
                        - (constants.MAPL_GRAV / constants.CP) * layer_height_above_surface
                        - (constants.MAPL_ALHL / constants.MAPL_CP) * parcel_specific_humidity_new
                    )

                    parcel_virtual_t = parcel_t * (1.0 * constants.MAPL_VIREPS * parcel_specific_humidity_new)

                    # parcel buoyancy
                    buoyancy = constants.MAPL_GRAV * (parcel_virtual_t - environment_virtual_t) / environment_virtual_t

            with computation(FORWARD), interval(0, 1):
                lfc_level: IntFieldIJ = k_end
                lnb_level: IntFieldIJ = k_end
                above_lfc: BoolFieldIJ = False
                stop_computation: BoolFieldIJ = False

            # if surface parcel is immediately buoyant, scan upward to find the first elevated buoyancy > 0
            # level above a buoyancy < 0 level. label it LFC
            with computation(BACKWARD), interval(0, -2):
                if K <= critical_level:
                    if buoyancy.at(K=k_end - 1) > 0.0:
                        if buoyancy > 0.0 and buoyancy[0, 0, 1] <= 0.0 and stop_computation == False:
                            lfc_level = K
                            above_lfc = True
                        if above_lfc == True and buoyancy < 0.0 and stop_computation == False:
                            lnb_level = K
                            stop_computation = True

            # if no such level is found in the previous block, set lfc as surface
            with computation(BACKWARD), interval(0, -1):
                if K <= critical_level:
                    if buoyancy.at(K=k_end - 1) <= 0.0:
                        if buoyancy > 0.0 and above_lfc == False and stop_computation == False:
                            lfc_level = K
                            above_lfc = True
                        if above_lfc == True and buoyancy < 0.0 and stop_computation == False:
                            lnb_level = K
                            stop_computation = True

            with computation(FORWARD), interval(0, 1):
                lfc = layer_height_above_surface.at(K=lfc_level)
                lnb = layer_height_above_surface.at(K=lnb_level)

            with computation(FORWARD), interval(...):
                if K <= critical_level:
                    cape = cape + max(0.0, buoyancy * layer_thickness)
                    if K >= lfc_level:
                        cin = cin + min(0.0, buoyancy * layer_thickness)

        def reset_to_undef_2d(field: FloatFieldIJ):
            with computation(FORWARD), interval(0, 1):
                if field <= 0.0:
                    field = constants.MAPL_UNDEF

        def max_along_k_dim(in_field: FloatField, out_field: FloatFieldIJ):
            with computation(FORWARD), interval(0, 1):
                out_field = constants.MAPL_UNDEF

            with computation(FORWARD), interval(...):
                if in_field > out_field:
                    out_field = in_field

        def mixed_layer_parcel(
            t: FloatField,
            specific_humidity: FloatField,
            p_interface_mb: FloatField,
            p_mb: FloatField,
            layer_height_above_surface: FloatField,
            layer_thickness: FloatField,
            lfc: IntFieldIJ,
            lnb: IntFieldIJ,
            parcel_moist_static_energy: FloatFieldIJ,
            parcel_specific_humidity: FloatFieldIJ,
            critical_level: IntFieldIJ,
        ):
            from __externals__ import k_end

            with computation(PARALLEL), interval(...):
                buoyancy = constants.MAPL_UNDEF
                aggregated_height = 0.0

            with computation(FORWARD), interval(0, 1):
                critical_level = 0
                lfc = 0
                lnb = 0

            with computation(BACKWARD), interval(...):
                if p_interface_mb - p_mb < 90.0:
                    parcel_moist_static_energy = (
                        parcel_moist_static_energy
                        + (t + (constants.MAPL_GRAV / constants.MAPL_CP) * layer_height_above_surface + (constants.MAPL_ALHL / constants.MAPL_CP) * specific_humidity)
                        * layer_thickness
                    )
                    parcel_specific_humidity = parcel_specific_humidity + specific_humidity * layer_thickness
                    aggregated_height = aggregated_height + layer_thickness
                    critical_level = K

            with computation(FORWARD), interval(0, 1):
                if aggregated_height > 0:
                    # average
                    parcel_moist_static_energy = parcel_moist_static_energy / aggregated_height
                    parcel_specific_humidity = parcel_specific_humidity / aggregated_height

        def most_unstable_parcel(
            t: FloatField,
            specific_humidity: FloatField,
            p_interface_mb: FloatField,
            p_mb: FloatField,
            layer_height_above_surface: FloatField,
            layer_thickness: FloatField,
            lfc: IntFieldIJ,
            lnb: IntFieldIJ,
            mucape: FloatFieldIJ,
            mucin: FloatFieldIJ,
            parcel_moist_static_energy: FloatFieldIJ,
            parcel_specific_humidity: FloatFieldIJ,
            critical_level: IntFieldIJ,
        ):
            from __externals__ import k_end

            with computation(PARALLEL), interval(...):
                buoyancy = constants.MAPL_UNDEF

            with computation(FORWARD), interval(0, 1):
                mucape = 0.0
                mucin = 0.0
                lfc = constants.MAPL_UNDEF
                lnb = constants.MAPL_UNDEF
                stop_computation: BoolFieldIJ = False
                critical_level = k_end

            with computation(BACKWARD), interval(...):
                if p_interface_mb - p_mb > 255.0:
                    stop_computation = True
                parcel_moist_static_energy = (
                    t + (constants.MAPL_GRAV / constants.MAPL_CP) * layer_height_above_surface + (constants.MAPL_ALHL / constants.MAPL_CP) * specific_humidity
                )
                parcel_specific_humidity = specific_humidity

        def surface_based_parcel(
            t: FloatField,
            specific_humidity: FloatField,
            layer_height_above_surface: FloatField,
            parcel_moist_static_energy: FloatFieldIJ,
            parcel_specific_humidity: FloatFieldIJ,
            critical_level: IntFieldIJ,
        ):
            from __externals__ import k_end

            with computation(FORWARD), interval(0, 1):
                critical_level = k_end

            with computation(FORWARD), interval(-1, None):
                parcel_moist_static_energy = (
                    t + (constants.MAPL_GRAV / constants.MAPL_CP) * layer_height_above_surface + (constants.MAPL_ALHL / constants.MAPL_CP) * specific_humidity
                )
                parcel_specific_humidity = specific_humidity

        self._setup = stencil_factory.from_dims_halo(func=setup, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._get_cape_cin = stencil_factory.from_dims_halo(func=get_cape_cin, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._reset_to_undef_2d = stencil_factory.from_dims_halo(func=reset_to_undef_2d, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._max_along_k_dim = stencil_factory.from_dims_halo(func=max_along_k_dim, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._mixed_layer_parcel = stencil_factory.from_dims_halo(func=mixed_layer_parcel, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._most_unstable_parcel = stencil_factory.from_dims_halo(func=most_unstable_parcel, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._surface_based_parcel = stencil_factory.from_dims_halo(func=surface_based_parcel, compute_dims=[I_DIM, J_DIM, K_DIM])

        # initialize locals
        self._environment_virtual_t = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM], Float)
        self._parcel_moist_static_energy = self.make_local(quantity_factory, [I_DIM, J_DIM], Float)
        self._parcel_specific_humidity = self.make_local(quantity_factory, [I_DIM, J_DIM], Float)
        self._critical_level = self.make_local(quantity_factory, [I_DIM, J_DIM], Float)
        self._mucape_3d = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM], Float)
        self._mucin_3d = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM], Float)

    def __call__(
        self,
        t: Quantity,
        specific_humidity: Quantity,
        p_interface_mb: Quantity,
        p_mb: Quantity,
        layer_height_above_surface: Quantity,
        layer_thickness: Quantity,
        saturation_specific_humidity: Quantity,
        dsaturation_specific_humidity: Quantity,
        buoyancy_surface_parcel: Quantity,
        sbcape: Quantity,
        mlcape: Quantity,
        mucape: Quantity,
        sbcin: Quantity,
        mlcin: Quantity,
        mucin: Quantity,
        lfc: Quantity,
        lnb: Quantity,
    ):
        self._setup(
            t=t,
            specific_humidity=specific_humidity,
            environment_virtual_t=self._environment_virtual_t,
            parcel_moist_static_energy=self._parcel_moist_static_energy,
            parcel_specific_humidity=self._parcel_specific_humidity,
        )

        # Mixed-layer calculation. Parcel properties averaged over lowest 90 hPa
        if mlcape is not None and mlcin is not None:
            self._mixed_layer_parcel(
                t=t,
                specific_humidity=specific_humidity,
                p_interface_mb=p_interface_mb,
                p_mb=p_mb,
                layer_height_above_surface=layer_height_above_surface,
                layer_thickness=layer_thickness,
                lfc=lfc,
                lnb=lnb,
                parcel_moist_static_energy=self._parcel_moist_static_energy,
                parcel_specific_humidity=self._parcel_specific_humidity,
                critica_level=self._critical_level,
            )

            self._get_cape_cin(
                critica_level=self._critical_level,
                t=t,
                p_mb=p_mb,
                layer_height_above_surface=layer_height_above_surface,
                layer_thickness=layer_thickness,
                saturation_specific_humidity=saturation_specific_humidity,
                dsaturation_specific_humidity=dsaturation_specific_humidity,
                environment_virtual_t=self._environment_virtual_t,
                parcel_moist_static_energy=self._parcel_moist_static_energy,
                parcel_specific_humidity=self._parcel_specific_humidity,
                buoyancy=buoyancy_surface_parcel,
                cape=mlcape,
                cin=mlcin,
                lfc=lfc,
                lnb=lnb,
            )

            self._reset_to_undef_2d(field=mlcape)
            self._reset_to_undef_2d(field=mlcin)

        # Most unstable calculation. Parcel in lowest 255 hPa with largest CAPE
        if mucape is not None and mucin is not None:
            self._most_unstable_parcel(
                t=t,
                specific_humidity=specific_humidity,
                p_interface_mb=p_interface_mb,
                p_mb=p_mb,
                layer_height_above_surface=layer_height_above_surface,
                layer_thickness=layer_thickness,
                lfc=lfc,
                lnb=lnb,
                mucape=mucape,
                mucin=mucin,
                parcel_moist_static_energy=self._parcel_moist_static_energy,
                parcel_specific_humidity=self._parcel_specific_humidity,
                critica_level=self._critical_level,
            )

            for k in range(self._stencil_factory.grid_indexing.domain_compute()[2]):
                self._critical_level = k
                self._get_cape_cin(
                    critica_level=self._critical_level,
                    t=t,
                    p_mb=p_mb,
                    layer_height_above_surface=layer_height_above_surface,
                    layer_thickness=layer_thickness,
                    saturation_specific_humidity=saturation_specific_humidity,
                    dsaturation_specific_humidity=dsaturation_specific_humidity,
                    environment_virtual_t=self._environment_virtual_t,
                    parcel_moist_static_energy=self._parcel_moist_static_energy,
                    parcel_specific_humidity=self._parcel_specific_humidity,
                    buoyancy=buoyancy_surface_parcel,
                    cape=mucape,
                    cin=mucin,
                    lfc=lfc,
                    lnb=lnb,
                )

                # TODO need a better solution for this, but the design and use of get_cape_cin
                # forces a compromise so that it can work with 2d and 3d situations.
                # may need new feature(s) to implement this properly
                self._mucape_3d[:, :, k] = mucape[:]
                self._mucin_3d[:, :, k] = mucin[:]

            self._max_along_k_dim(in_field=self._mucape_3d, out_field=mucape)
            self._max_along_k_dim(in_field=self._mucin_3d, out_field=mucin)

            self._reset_to_undef_2d(field=mucape)
            self._reset_to_undef_2d(field=mucin)

        # Surface-based calculation
        self._surface_based_parcel(
            t=t,
            specific_humidity=specific_humidity,
            layer_height_above_surface=layer_height_above_surface,
            parcel_moist_static_energy=self._parcel_moist_static_energy,
            parcel_specific_humidity=self._parcel_specific_humidity,
            critical_level=self._critical_level,
        )

        self._get_cape_cin(
            critica_level=self._critical_level,
            t=t,
            p_mb=p_mb,
            layer_height_above_surface=layer_height_above_surface,
            layer_thickness=layer_thickness,
            saturation_specific_humidity=saturation_specific_humidity,
            dsaturation_specific_humidity=dsaturation_specific_humidity,
            environment_virtual_t=self._environment_virtual_t,
            parcel_moist_static_energy=self._parcel_moist_static_energy,
            parcel_specific_humidity=self._parcel_specific_humidity,
            buoyancy=buoyancy_surface_parcel,
            cape=sbcape,
            cin=sbcin,
            lfc=lfc,
            lnb=lnb,
        )

        self._reset_to_undef_2d(field=sbcape)
        self._reset_to_undef_2d(field=sbcin)
