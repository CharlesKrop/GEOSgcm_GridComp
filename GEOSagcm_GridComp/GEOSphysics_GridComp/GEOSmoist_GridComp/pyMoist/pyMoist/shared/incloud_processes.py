"""Stencils and functions called by multiple pyMoist modules.
These functions evaluate various in-cloud microphysical
processes/quantities."""

from ndsl.dsl.gt4py import PARALLEL, GlobalTable, computation, exp, float32, float64, floor, function, interval, log10, round_away_from_zero, sin, K, FORWARD
from ndsl.dsl.typing import Float, FloatField, BoolFieldIJ

import pyMoist.constants as constants
from pyMoist.shared.atmos_recipes import air_density


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

    if srf_type_int == 2 or srf_type_int == 3 or srf_type_int == 4:  # 2 = snow, 3 = ice, 4 = landice
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
    t: FloatField,
    vapor: FloatField,
    type_one_ice: FloatField,
    type_one_liquid: FloatField,
    type_one_cloud_fraction: FloatField,
    type_two_ice: FloatField,
    type_two_liquid: FloatField,
    type_two_cloud_fraction: FloatField,
    lid_level: Int,
):
    """Fix up cloud variables to ensure physical consistency.

    Args:
        t (FloatField): temperature (Kelvin)
        vapor (FloatField): water vapor mixing ratio
        type_one_ice (FloatField): ice mixing ratio (type one, e.g. convective/large scale)
        type_one_liquid (FloatField): liquid mixing ratio (type one, e.g. convective/large scale)
        type_one_cloud_fraction (FloatField): cloud fraction (type one, e.g. convective/large scale)
        type_two_ice (FloatField): ice mixing ratio (type two, e.g. convective/large scale)
        type_two_liquid (FloatField): liquid mixing ratio (type two, e.g. convective/large scale)
        type_two_cloud_fraction (FloatField): cloud fraction (type two, e.g. convective/large scale)
        lid_level (Int): the level above which clouds are removed (note 0 = TOA). set to a number < 0 to disable
    """

    from __externals__ import MIN_CLOUD_FRACTION, MIN_CLOUD_QUANTITY

    with computation(FORWARD), interval(0, 1):
        if lid_level < 0:
            remove_clouds: BoolFieldIJ = False
        else:
            remove_clouds: BoolFieldIJ = True

    with computation(PARALLEL), interval(...):
        if remove_clouds:
            # remove all cloud quantities above the lid level
            if K < lid_level:
                vapor = vapor + type_one_ice + type_one_liquid + type_two_ice + type_two_liquid
                t = (
                    t
                    - (constants.MAPL_ALHL / constants.MAPL_CP) * (type_one_liquid + type_two_liquid)
                    - (constants.MAPL_ALHS / constants.MAPL_CP) * (type_one_ice + type_two_ice)
                )
                type_one_ice = 0.0
                type_one_liquid = 0.0
                type_one_cloud_fraction = 0.0
                type_two_ice = 0.0
                type_two_liquid = 0.0
                type_two_cloud_fraction = 0.0

        else:
            # ensure physical values for cloud quantities

            # ensure total cloud fraction <= 1.0
            total_cloud_fraction = type_one_cloud_fraction + type_two_cloud_fraction
            if total_cloud_fraction > 1.0:
                type_one_cloud_fraction = type_one_cloud_fraction * (1.0 / total_cloud_fraction)
                type_two_cloud_fraction = type_two_cloud_fraction * (1.0 / total_cloud_fraction)

            # fix if type one cloud fraction too small
            if type_one_cloud_fraction < MIN_CLOUD_FRACTION:
                vapor = vapor + type_one_liquid + type_two_liquid
                t = t - (constants.MAPL_ALHL / constants.MAPL_CP) * type_one_liquid - (constants.MAPL_ALHS / constants.MAPL_CP) * type_two_liquid
                type_one_ice = 0.0
                type_one_liquid = 0.0
                type_one_cloud_fraction = 0.0

            # fix if type two cloud fraction too small
            if type_two_cloud_fraction < MIN_CLOUD_FRACTION:
                vapor = vapor + type_two_liquid + type_two_ice
                t = t - (constants.MAPL_ALHL / constants.MAPL_CP) * type_two_liquid - (constants.MAPL_ALHS / constants.MAPL_CP) * type_two_ice
                type_two_ice = 0.0
                type_two_liquid = 0.0
                type_two_cloud_fraction = 0.0

            # fix if type one liquid is too small
            if type_one_liquid < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_one_liquid
                t = t - (constants.MAPL_ALHL / constants.MAPL_CP) * type_one_liquid
                type_one_liquid = 0.0

            # fix if type one ice is too small
            if type_one_ice < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_one_ice
                t = t - (constants.MAPL_ALHS / constants.MAPL_CP) * type_one_ice
                type_one_ice = 0.0

            # fix if type two liquid is too small
            if type_two_liquid < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_two_liquid
                t = t - (constants.MAPL_ALHL / constants.MAPL_CP) * type_two_liquid
                type_two_liquid = 0.0

            # fix if type two ice is too small
            if type_two_ice < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_two_ice
                t = t - (constants.MAPL_ALHS / constants.MAPL_CP) * type_two_ice
                type_two_ice = 0.0

            # fix all type one quantities if liquid + ice is too small
            if (type_one_liquid + type_two_liquid) < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_one_liquid + type_two_liquid
                t = t - (constants.MAPL_ALHL / constants.MAPL_CP) * type_one_liquid - (constants.MAPL_ALHS / constants.MAPL_CP) * type_two_liquid
                type_two_ice = 0.0
                type_one_liquid = 0.0
                type_one_cloud_fraction = 0.0

            # fix all type two quantities if liquid + ice is too small
            if (type_two_liquid + type_two_ice) < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_two_liquid + type_two_ice
                t = t - (constants.MAPL_ALHL / constants.MAPL_CP) * type_two_liquid - (constants.MAPL_ALHS / constants.MAPL_CP) * type_two_ice
                type_two_cloud_fraction = 0.0
                type_two_liquid = 0.0
                type_two_ice = 0.0


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
