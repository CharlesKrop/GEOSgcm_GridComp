"""Stencils and functions called by multiple pyMoist modules.
These functions evaluate various in-cloud microphysical
processes/quantities."""

from ndsl.dsl.gt4py import (
    PARALLEL,
    GlobalTable,
    computation,
    exp,
    float32,
    float64,
    floor,
    function,
    interval,
    log10,
    round_away_from_zero,
    sin,
    K,
    FORWARD,
    BACKWARD,
    log,
    sqrt,
)
from ndsl.dsl.typing import Float, FloatField, BoolFieldIJ, Bool, Int, IntFieldIJ, FloatFieldIJ

from pyMoist.shared.atmos_recipes import air_density
from pyMoist.saturation_tables import (
    saturation_specific_humidity,
    GlobalTable_saturation_tables,
    saturation_specific_humidity_liquid_surface,
    saturation_specific_humidity_frozen_surface,
)
from pyMoist.constants import (
    LBE,
    MAPL_RVAP,
    MAPL_RGAS,
    MAPL_CPVAP,
    MAPL_CPDRY,
    MAPL_TICE,
    ICE_RADII_PARAM,
    JaT_ICE_ALL,
    JaT_ICE_MAX,
    MAPL_PI,
    aT_ICE_ALL,
    aT_ICE_MAX,
    aICEFRPWR,
    iT_ICE_MAX,
    iT_ICE_ALL,
    iICEFRPWR,
    lT_ICE_MAX,
    lT_ICE_ALL,
    lICEFRPWR,
    oT_ICE_MAX,
    oT_ICE_ALL,
    oICEFRPWR,
    LIQ_RADII_PARAM,
    BX,
    R13BBETA,
    ABETA,
    LBX,
    LBE,
    MAPL_CP,
    MAPL_ALHL,
    MAPL_ALHS,
    MAPL_ALHF,
    R_AIR,
    TAUFRZ,
    TAUMLT,
    EPSILON,
    RHO_I,
    RHO_W,
    K_COND,
    DIFFU,
)


@function
def ice_fraction_modis(
    temp: Float,
):
    # Use MODIS polynomial from Hu et al, DOI: (10.1029/2009JD012384)
    tc = max(-46.0, min(temp - MAPL_TICE, 46.0))  # convert to celcius and limit range from -46:46 C
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
    if ICE_RADII_PARAM == 1:
        # Jason formula
        if temp <= JaT_ICE_ALL:
            icefrct_c = 1.000
        elif temp > JaT_ICE_ALL and temp <= JaT_ICE_MAX:
            icefrct_c = sin(0.5 * MAPL_PI * (1.00 - (temp - JaT_ICE_ALL) / (JaT_ICE_MAX - JaT_ICE_ALL)))
        else:
            icefrct_c = 0.00
    else:
        # Default formula
        if temp <= aT_ICE_ALL:
            icefrct_c = 1.000
        elif temp > aT_ICE_ALL and temp <= aT_ICE_MAX:
            icefrct_c = sin(0.5 * MAPL_PI * (1.00 - (temp - aT_ICE_ALL) / (aT_ICE_MAX - aT_ICE_ALL)))
        else:
            icefrct_c = 0.00
    icefrct_c = max(min(icefrct_c, 1.00), 0.00) ** aICEFRPWR

    # Sigmoidal functions like figure 6b/6c of Hu et al 2010, doi:10.1029/2009JD012384
    srf_type_int = round(srf_type)

    if srf_type_int == 2 or srf_type_int == 3 or srf_type_int == 4:  # 2 = snow, 3 = ice, 4 = landice
        if temp <= iT_ICE_ALL:
            icefrct_m = 1.000
        elif temp > iT_ICE_ALL and temp <= iT_ICE_MAX:
            icefrct_m = sin(0.5 * MAPL_PI * (1.00 - (temp - iT_ICE_ALL) / (iT_ICE_MAX - iT_ICE_ALL)))
        else:
            icefrct_m = 0.00
        icefrct_m = max(min(icefrct_m, 1.00), 0.00) ** iICEFRPWR

    elif srf_type_int == 1:  # land
        if temp <= lT_ICE_ALL:
            icefrct_m = 1.000
        elif temp > lT_ICE_ALL and temp <= lT_ICE_MAX:
            icefrct_m = sin(0.5 * MAPL_PI * (1.00 - (temp - lT_ICE_ALL) / (lT_ICE_MAX - lT_ICE_ALL)))
        else:
            icefrct_m = 0.00
        icefrct_m = max(min(icefrct_m, 1.00), 0.00) ** lICEFRPWR

    elif srf_type_int == 0:  # ocean
        if temp <= oT_ICE_ALL:
            icefrct_m = 1.000
        elif temp > oT_ICE_ALL and temp <= oT_ICE_MAX:
            icefrct_m = sin(0.5 * MAPL_PI * (1.00 - (temp - oT_ICE_ALL) / (oT_ICE_MAX - oT_ICE_ALL)))
        else:
            icefrct_m = 0.00
        icefrct_m = max(min(icefrct_m, 1.00), 0.00) ** oICEFRPWR

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
    if LIQ_RADII_PARAM == 1:
        # Jason Version
        radius = min(
            60.0e-6,
            max(
                2.5e-6,
                1.0e-6 * BX * (wc / nnx) ** R13BBETA * ABETA * 6.92,
            ),
        )
    else:
        # [liu&daum, 2000 and 2005. liu et al 2008]
        radius = min(
            60.0e-6,
            max(2.5e-6, 1.0e-6 * LBX * (wc / nnx) ** LBE),
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
    if ICE_RADII_PARAM == 1:
        # Ice cloud effective radius -- [klaus wyser, 1998]
        if temperature > MAPL_TICE or ice_mixing_ratio <= 0.0:
            bb = -2.0
        else:
            bb = -2.0 + log10(wc / 50.0) * (1.0e-3 * (MAPL_TICE - temperature) ** 1.5)
        bb = min(max(bb, -6.0), -2.0)
        radius = 377.4 + 203.3 * bb + 37.91 * bb**2 + 2.3696 * bb**3
        radius = min(150.0e-6, max(5.0e-6, 1.0e-6 * radius))
    else:
        # Ice cloud effective radius ----- [Sun, 2001]
        tc = temperature - MAPL_TICE
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
                t = t - (MAPL_ALHL / MAPL_CP) * (type_one_liquid + type_two_liquid) - (MAPL_ALHS / MAPL_CP) * (type_one_ice + type_two_ice)
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
                t = t - (MAPL_ALHL / MAPL_CP) * type_one_liquid - (MAPL_ALHS / MAPL_CP) * type_two_liquid
                type_one_ice = 0.0
                type_one_liquid = 0.0
                type_one_cloud_fraction = 0.0

            # fix if type two cloud fraction too small
            if type_two_cloud_fraction < MIN_CLOUD_FRACTION:
                vapor = vapor + type_two_liquid + type_two_ice
                t = t - (MAPL_ALHL / MAPL_CP) * type_two_liquid - (MAPL_ALHS / MAPL_CP) * type_two_ice
                type_two_ice = 0.0
                type_two_liquid = 0.0
                type_two_cloud_fraction = 0.0

            # fix if type one liquid is too small
            if type_one_liquid < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_one_liquid
                t = t - (MAPL_ALHL / MAPL_CP) * type_one_liquid
                type_one_liquid = 0.0

            # fix if type one ice is too small
            if type_one_ice < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_one_ice
                t = t - (MAPL_ALHS / MAPL_CP) * type_one_ice
                type_one_ice = 0.0

            # fix if type two liquid is too small
            if type_two_liquid < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_two_liquid
                t = t - (MAPL_ALHL / MAPL_CP) * type_two_liquid
                type_two_liquid = 0.0

            # fix if type two ice is too small
            if type_two_ice < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_two_ice
                t = t - (MAPL_ALHS / MAPL_CP) * type_two_ice
                type_two_ice = 0.0

            # fix all type one quantities if liquid + ice is too small
            if (type_one_liquid + type_two_liquid) < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_one_liquid + type_two_liquid
                t = t - (MAPL_ALHL / MAPL_CP) * type_one_liquid - (MAPL_ALHS / MAPL_CP) * type_two_liquid
                type_two_ice = 0.0
                type_one_liquid = 0.0
                type_one_cloud_fraction = 0.0

            # fix all type two quantities if liquid + ice is too small
            if (type_two_liquid + type_two_ice) < MIN_CLOUD_QUANTITY:
                vapor = vapor + type_two_liquid + type_two_ice
                t = t - (MAPL_ALHL / MAPL_CP) * type_two_liquid - (MAPL_ALHS / MAPL_CP) * type_two_ice
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

        crystal_number = k * cloud_ice_mixing_ratio * internal_lambda * internal_lambda * internal_lambda / (MAPL_PI * ice_density)

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
    am_r = MAPL_PI * 1000.0 / 6.0

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


@function
def find_t_lcl(
    t: Float,
    rh: Float,
):
    """
    Computes the LCL temperature

    Arguments:
        t (Float): temperature at surface (K)
        rh (Float): relative humidity at surface

    Returns:
        tlcl: LCL temperature
    """
    term1 = 1.0 / (t - 55.0)
    term2 = log(max(0.1, rh) / 100.0) / 2840.0
    denom = term1 - term2
    tlcl = (1.0 / denom) + 55.0
    return tlcl


def find_lcl_level(
    t: FloatField,
    p_mb: FloatField,
    vapor: FloatField,
    esx: GlobalTable_saturation_tables,
    lcl_level: IntFieldIJ,
):
    """
    Find the level of the lifted condensation level (LCL).

    Arguments:
        t (FloatField): (in) Atmospheric temperature (K)
        p_mb (FloatField): (in) pressure (mb)
        vapor (FloatField): (in) water vapor mixing radio (kg/kg)
        esx (GlobalTable_saturation_tables): (in) saturation vapor pressure table, details unknown
        lcl_level (IntFieldIJ): (out) LCL level
    """
    from __externals__ import k_end

    # set up mask to stop computation
    with computation(FORWARD), interval(0, 1):
        found_level: BoolFieldIJ = False

    # get LCL pressure
    with computation(PARALLEL), interval(-1, None):
        qsat, _ = saturation_specific_humidity(t=t, p=p_mb * 100.0, esx=esx)
        rhsfc = 100.0 * vapor / qsat
        qsat, _ = saturation_specific_humidity(t=t, p=p_mb * 100.0, esx=esx)
        rhsfc = 100.0 * vapor / qsat
        tlcl = find_t_lcl(t=t, rh=rhsfc)
        rm = (1.0 - vapor) * MAPL_RGAS + vapor * MAPL_RVAP
        rm = (1.0 - vapor) * MAPL_RGAS + vapor * MAPL_RVAP
        cpm = (1.0 - vapor) * MAPL_CPDRY + vapor * MAPL_CPVAP
        plcl = p_mb * ((tlcl / t) ** (cpm / rm))

    # find nearest level <= LCL pressure
    with computation(BACKWARD), interval(...):
        if found_level == False:  # noqa
            lcl_level = K
        if p_mb <= plcl.at(K=k_end):
            found_level = True


@function
def pdffrac(
    pdfshape: Int,
    qtmean: Float,
    sigmaqt1: Float,
    sigmaqt2: Float,
    qstar: Float,
):
    """Determine cloud fraction

    Args:
        pdfshape (Int)
        qtmean (Float)
        sigmaqt1 (Float)
        sigmaqt2 (Float)
        qstar (Float)

    Returns:
        Float: cloud_fraction
    """
    if pdfshape == 1:
        if (qtmean + sigmaqt1) < qstar:
            cloud_fraction = 0.0
        else:
            if sigmaqt1 > 0.0:
                cloud_fraction = min(qtmean + sigmaqt1 - qstar, 2.0 * sigmaqt1) / (2.0 * sigmaqt1)
            else:
                cloud_fraction = 1.0
    elif pdfshape == 2:
        qtmode = qtmean + (sigmaqt1 - sigmaqt2) / 3.0
        qtmin = max(qtmode - sigmaqt1, 0.0)
        qtmax = qtmode + sigmaqt2
        if qtmax <= qstar:
            cloud_fraction = 0.0
        elif qtmode <= qstar and qstar < qtmax:
            cloud_fraction = (qtmax - qstar) * (qtmax - qstar) / ((qtmax - qtmin) * (qtmax - qtmode))
        elif qtmin <= qstar and qstar < qtmode:
            cloud_fraction = 1.0 - ((qstar - qtmin) * (qstar - qtmin) / ((qtmax - qtmin) * (qtmode - qtmin)))
        elif qstar <= qtmin:
            cloud_fraction = 1.0

    return cloud_fraction


@function
def pdfcondensate(
    pdfshape: Int,
    qtmean: Float,
    sigmaqt1: Float,
    sigmaqt2: Float,
    qstar: Float,
):
    """Quantify new condensate

    Args:
        pdfshape (Int)
        qtmean (Float)
        sigmaqt1 (Float)
        sigmaqt2 (Float)
        qstar (Float)

    Returns:
        Float: condensate
    """
    qtmean_64: float64 = qtmean
    sigmaqt1_64: float64 = sigmaqt1
    sigmaqt2_64: float64 = sigmaqt2
    qstar_64: float64 = qstar

    condensate: float64 = 0.0
    if pdfshape == 1:
        if (qtmean_64 + sigmaqt1_64) < qstar_64:
            condensate = float64(0.0)
        elif qstar_64 > (qtmean_64 - sigmaqt1_64):
            if sigmaqt1_64 > 0.0:
                condensate = (min(qtmean_64 + sigmaqt1_64 - qstar_64, float64(2.0) * sigmaqt1_64) ** 2) / (float64(4.0) * sigmaqt1_64)
            else:
                condensate = qtmean_64 - qstar_64
        else:
            condensate = qtmean_64 - qstar_64
    elif pdfshape == 2:
        qtmode = qtmean_64 + (sigmaqt1_64 - sigmaqt2_64) / float64(2.0)
        qtmin = max(qtmode - sigmaqt1_64, float64(0.0))
        qtmax = qtmode + sigmaqt2_64
        if qtmax <= qstar_64:
            condensate = float64(0.0)
        elif qtmode <= qstar_64 and qstar_64 < qtmax:
            constB = float64(2.0) / ((qtmax - qtmin) * (qtmax - qtmode))
            cloudf = (qtmax - qstar_64) * (qtmax - qstar_64) / ((qtmax - qtmin) * (qtmax - qtmode))
            term1 = (qstar_64 * qstar_64 * qstar_64) / float64(3.0)
            term2 = (qtmax * qstar_64 * qstar_64) / float64(2.0)
            term3 = (qtmax * qtmax * qtmax) / float64(6.0)
            condensate = constB * (term1 - term2 + term3) - qstar_64 * cloudf
        elif qtmin <= qstar_64 and qstar_64 < qtmode:
            constA = float64(2.0) / ((qtmax - qtmin) * (qtmode - qtmin))
            cloudf = float64(1.0) - ((qstar_64 - qtmin) * (qstar_64 - qtmin) / ((qtmax - qtmin) * (qtmode - qtmin)))
            term1 = (qstar_64 * qstar_64 * qstar_64) / float64(3.0)
            term2 = (qtmin * qstar_64 * qstar_64) / float64(2.0)
            term3 = (qtmin * qtmin * qtmin) / float64(6.0)
            condensate = qtmean_64 - (constA * (term1 - term2 + term3)) - qstar_64 * cloudf
        elif qstar_64 <= qtmin:
            condensate = qtmean_64 - qstar_64
    return condensate


@function
def bergeron_partition(
    dtime: Float,
    p_mb: Float,
    t: Float,
    vapor: Float,
    large_scale_ice: Float,
    large_scale_liquid: Float,
    convective_ice: Float,
    convective_liquid: Float,
    concentration_ice: Float,
    delta_condensate: Float,
    fraction_ice: Float,
    convection_fraction: Float,
    surface_type: Float,
    ese: GlobalTable_saturation_tables,
    esw: GlobalTable_saturation_tables,
    frz: Float,
    lqu: Float,
    needs_preexisting: Bool,
):
    """Partition the new condensates. Follows Barahona et al. GMD. 2014

    Args:
        dtime (Float)
        p_mb (Float)
        t (Float)
        vapor (Float)
        large_scale_ice (Float)
        large_scale_liquid (Float)
        convective_ice (Float)
        convective_liquid (Float)
        concentration_ice (Float)
        delta_condensate (Float)
        fraction_ice (Float)
        convection_fraction (Float)
        surface_type (Float)
        ese (GlobalTable_saturation_tables)
        esw (GlobalTable_saturation_tables)
        frz (Float)
        lqu (Float)
        needs_preexisting (Bool)

    Returns:
        None
    """

    # NDSL can only have one return, this replicates the multi-return mechanics in the source Fortran
    stop_calculation = False

    # PHASE 1: Initialization & Temperature Bounds
    t_celsius = t - MAPL_TICE
    delta_condensate_rate = delta_condensate = dtime  # convert total mass change to a rate

    # combine resolved and parameterized masses to get a bulk view of the grid box
    tot_ice = large_scale_ice + convective_ice
    tot_liquid = large_scale_liquid + convective_liquid
    tot_mass = tot_ice + tot_liquid

    # calculate active ice crystal number concentration.
    # (concentration_ice is total. We scale it by the liquid mass fraction to estimate
    # how many crystals are interacting with the liquid environment)
    f_mass_ice = 0.0
    if tot_mass > 0.0:
        f_mass_ice = tot_ice / tot_mass
    n_ice_active = (1.0 - f_mass_ice) * concentration_ice

    # PHASE 2: Mixed-Phase Regime & Deposition Physics
    # calculate how fast water vapor deposits onto existing ice crystals.
    fraction_ice = 0.0

    # if no resolved ice exists, check if we are allowed to spontaneously nucleate
    if large_scale_ice <= 0.0:
        if needs_preexisting:
            # explicit microphysics handles nucleation; no deposition growth allowed here
            stop_calculation = True
    else:
        # fall back to temperature-based diagnostic fraction
        fraction_ice = ice_fraction(t, convection_fraction, surface_type)
        stop_calculation = True

    if not stop_calculation:
        # calculate saturation thresholds
        qsat_liquid, _ = saturation_specific_humidity_liquid_surface(esw, lqu, t, p_mb * 100.0)
        qsat_ice, dqsat_ice = saturation_specific_humidity_frozen_surface(ese, frz, t, p_mb * 100.0)

        # limit available vapor to liquid saturation (droplets cap the vapor pressure)
        qv_inc = min(vapor, qsat_liquid)

        # diffusivity of water vapor in air (Seinfeld and Pandis 2006)
        diff = (0.211 * 1013.25 / (p_mb + 0.1)) * (((t + 0.1) / MAPL_TICE) ** 1.94) * 1e-4
        den_air = (p_mb * 100.0) / (MAPL_RGAS * t)
        den_ice = 1000.0 * (0.9167 - 1.75e-4 * t_celsius - 5.0e-7 * t_celsius * t_celsius)  # (Pruppacher & Klett 1997)
        lh_corr = 1.0 + dqsat_ice * (MAPL_ALHS / MAPL_CP)

        # estimate ice crystal diameter (assumes a monodisperse size distribution)
        if n_ice_active > 1.0 and large_scale_ice > 1.0e-10:
            d_crystal = max((large_scale_ice / (n_ice_active * den_ice * MAPL_PI)) ** (0.333), 20.0e-6)
        else:
            d_crystal = 20.0e-6

        # deposition time scale inverse (1/Tau)
        time_eff_inv = n_ice_active * den_air * 2.0 * MAPL_PI * diff * d_crystal / lh_corr

        # calculate final deposition rate using analytical integration of the relaxation equation
        deposition_rate = 0.0
        if time_eff_inv > 0.0 and large_scale_ice > 1.0e-14:
            aux = max(min(dtime * time_eff_inv, 20.0), 0.0)
            deposition_rate = (qv_inc - qsat_ice) * (1.0 - exp(-aux)) / dtime

        # ice can sublimate, but only up to the amount of existing resolved ice
        deposition_rate = max(deposition_rate, -large_scale_ice / dtime)

        # PHASE 3: Condensate Partitioning
        # apply the Bergeron-Findeisen process based on the calculated deposition_rate
        dice_rate = 0.0
        dliquid_rate = 0.0

        if delta_condensate_rate >= 0.0:
            # --- NET CONDENSATION ---
            if deposition_rate > 0.0:
                # ice grows by deposition. It can consume the new condensate (delta_condensate_rate)
                # PLUS the evaporation of existing resolved liquid (large_scale_liquid/dtime).
                dice_rate = min(deposition_rate, delta_condensate_rate + (large_scale_liquid / dtime))
                dliquid_rate = delta_condensate_rate - dice_rate
            else:
                # deposition is negative/zero; PDF allows condensation in subsaturated conditions
                dliquid_rate = delta_condensate_rate
                dice_rate = 0.0

        else:
            # --- NET EVAPORATION ---
            # liquid droplets evaporate much faster than ice crystals sublimate.
            # therefore, liquid evaporates first, regardless of the deposition calculation.
            dliquid_rate = max(delta_condensate_rate, -large_scale_liquid / dtime)
            dice_rate = max(delta_condensate_rate - dliquid_rate, -large_scale_ice / dtime)

        # calculate the final diagnostic ice fraction
        if delta_condensate_rate != 0.0:
            fraction_ice = max(min(dice_rate / delta_condensate_rate, 1.0), 0.0)

    return fraction_ice


def hydrostatic_pdf(
    convection_fraction: FloatFieldIJ,
    surface_type: FloatFieldIJ,
    p_mb: FloatField,
    layer_height_above_surface: FloatField,
    t: FloatField,
    alpha: FloatField,
    vapor: FloatField,
    large_scale_cloud_fraction: FloatField,
    large_scale_ice: FloatField,
    large_scale_liquid: FloatField,
    convective_cloud_fraction: FloatField,
    convective_ice: FloatField,
    convective_liquid: FloatField,
    concentration_ice: FloatField,
    concentration_liquid: FloatField,
    liquid_water_static_energy_flux: FloatField,
    liquid_water_static_energy_variance: FloatField,
    liquid_water_static_energy_third_moment: FloatField,
    total_water_flux: FloatField,
    total_water_variance: FloatField,
    total_water_third_moment: FloatField,
    covariance_liquid_water_static_energy_and_total_water_specific_humidity: FloatField,
    vertical_motion_variance: FloatField,
    vertical_motion_third_moment: FloatField,
    pdf_first_plume_fractional_area: FloatField,
    hydrostatic_pdf_iterations: FloatField,
    buoyancy_flux: FloatField,
    liquid_water_flux: FloatField,
    bergeron_needs_preexisting: Bool,
    use_sc_ice: Bool,
    sc_ice: FloatField,
    iteration_method: Int,
    ese: GlobalTable_saturation_tables,
    esw: GlobalTable_saturation_tables,
    esx: GlobalTable_saturation_tables,
    estfrz: Float,
    estlqu: Float,
):
    """Apply hydrostatic PDF cloud closure and thermodynamic adjustment.

    The routine isolates the environmental portion of a grid box, diagnoses cloud fraction and condensate from the configured subgrid PDF,
    iterates temperature and moisture to equilibrium, then updates resolved vapor, liquid water, ice, and temperature in place.

    Args:
        convection_fraction (FloatFieldIJ)
        surface_type (FloatFieldIJ)
        p_mb (FloatField)
        layer_height_above_surface (FloatField)
        t (FloatField)
        alpha (FloatField)
        vapor (FloatField)
        large_scale_cloud_fraction (FloatField)
        large_scale_ice (FloatField)
        large_scale_liquid (FloatField)
        convective_cloud_fraction (FloatField)
        convective_ice (FloatField)
        convective_liquid (FloatField)
        concentration_ice (FloatField)
        concentration_liquid (FloatField)
        liquid_water_static_energy_flux (FloatField)
        liquid_water_static_energy_variance (FloatField)
        liquid_water_static_energy_third_moment (FloatField)
        total_water_flux (FloatField)
        total_water_variance (FloatField)
        total_water_third_moment (FloatField)
        covariance_liquid_water_static_energy_and_total_water_specific_humidity (FloatField)
        vertical_motion_variance (FloatField)
        vertical_motion_third_moment (FloatField)
        pdf_first_plume_fractional_area (FloatField)
        hydrostatic_pdf_iterations (FloatField)
        buoyancy_flux (FloatField)
        liquid_water_flux (FloatField)
        bergeron_needs_preexisting (Bool): option for bergeron partition; True will end the partitioning early
        use_sc_ice (Bool): controls sub-grid ice supersaturation scaling
        sc_ice (FloatField): sub-grid ice supersaturation numerical input field, only read when use_sc_ice is true
        iteration_method (Int): controls execution method, set to -999 to use default
        esx (GlobalTable_saturation_tables)
        ese (GlobalTable_saturation_tables)
        esw (GlobalTable_saturation_tables)
        estfrz (Float)
        estlqu (Float)
    """
    from __externals__ import dtime, PDFSHAPE, MIN_CLOUD_FRACTION, USE_BERGERON

    # PHASE 1: setup & environmental isolation
    with computation(FORWARD), interval(0, 1):
        if iteration_method != -999:
            iteration_method_internal: IntFieldIJ = iteration_method
        else:
            # default value
            iteration_method_internal: IntFieldIJ = 1

        inverse_large_scale_cloud_fraction: FloatFieldIJ = 0.0
        if large_scale_cloud_fraction != 0.0:
            inverse_large_scale_cloud_fraction = 1.0 / (1.0 - large_scale_cloud_fraction)

        large_scale_cloud_fraction_internal = large_scale_cloud_fraction * inverse_large_scale_cloud_fraction
        large_scale_condensate_internal = (large_scale_liquid + large_scale_ice) * inverse_large_scale_cloud_fraction
        large_scale_ice_internal = large_scale_ice * inverse_large_scale_cloud_fraction
        t_internal = t

        # calculate available environmental water vapor by subtracting vapor assumed to be locked inside the parameterized cloud
        qsat, dqsat = saturation_specific_humidity(t_internal, p_mb, esx)
        vapor_internal = (vapor - qsat * large_scale_cloud_fraction) * inverse_large_scale_cloud_fraction

        # total environmental water (can be negative due to CN saturation assumptions)
        total_water_internal = large_scale_condensate_internal + vapor_internal

        # PHASE 2: pre-calculate Double Gaussian PDF (if applicable)
        if PDFSHAPE == 6:
            option_not_implemented = True

        # PHASE 3: Iteration to Thermodynamic Equilibrium
        # adjust temperature, saturation, and condensate until stable
        max_iterations = 20
        count = 0
        while count < max_iterations:
            # store state from previous iteration
            vapor_internal_old = vapor_internal
            large_scale_condensate_internal_old = large_scale_condensate_internal
            large_scale_cloud_fraction_internal_old = large_scale_cloud_fraction_internal
            t_internal_old = t_internal

            qsat_iteration, dqsat_iteration = saturation_specific_humidity(t_internal, p_mb, esx)

            if use_sc_ice:
                sc_ice_internal = min(max(sc_ice, 1.0), 1.7)
                qsxsc = qsat_iteration * sc_ice_internal
                if large_scale_ice_internal >= 0.0 and qsat_iteration > total_water_internal:
                    qsat_iteration = qsxsc

            # setup PDF width (sigma) based on shape
            if PDFSHAPE < 3:
                # Top-hat or Triangular
                sigmaqt1 = alpha * qsat_iteration
                sigmaqt2 = alpha * qsat_iteration
            elif PDFSHAPE == 4:
                # Lognormal
                sigmaqt1 = max(alpha / sqrt(3.0), 0.001)

            # evaluate PDF to find new condensate (large_scale_condensate_internal) and fraction (large_scale_cloud_fraction_internal)

            if PDFSHAPE < 5:
                large_scale_cloud_fraction_internal = pdffrac(PDFSHAPE, total_water_internal, sigmaqt1, sigmaqt2, qsat_iteration)
                large_scale_condensate_internal = pdfcondensate(PDFSHAPE, total_water_internal, sigmaqt1, sigmaqt2, qsat_iteration)

            elif PDFSHAPE == 5:
                option_not_implemented = True
            elif PDFSHAPE == 6:
                option_not_implemented = True

            if USE_BERGERON:
                delta_condensate = large_scale_condensate_internal - large_scale_condensate_internal_old
                large_scale_cloud_fraction = large_scale_cloud_fraction_internal * (1.0 - convective_cloud_fraction)
                n_fac = 100.0 * p_mb * R_AIR / t_internal

                fraction_ice = bergeron_partition(
                    dtime=dtime,
                    p_mb=p_mb,
                    t=t,
                    vapor=vapor,
                    large_scale_ice=large_scale_ice,
                    large_scale_liquid=large_scale_liquid,
                    convective_ice=convective_ice,
                    convective_liquid=convective_liquid,
                    concentration_ice=concentration_ice,
                    delta_condensate=delta_condensate,
                    fraction_ice=fraction_ice,
                    convection_fraction=convection_fraction,
                    surface_type=surface_type,
                    ese=ese,
                    esw=esw,
                    frz=estfrz,
                    lqu=estlqu,
                    needs_preexisting=bergeron_needs_preexisting,
                )

            # relax the condensate update to prevent oscillation during iteration
            latent_heeat_factor = (1.0 - fraction_ice) * (MAPL_ALHL / MAPL_CP) + fraction_ice * (MAPL_ALHS / MAPL_CP)
            if PDFSHAPE == 1:
                large_scale_condensate_internal = large_scale_condensate_internal_old + (large_scale_condensate_internal - large_scale_condensate_internal_old) / (
                    1.0 - (large_scale_cloud_fraction_internal * (alpha - 1.0) - (large_scale_condensate_internal / qsat)) * dqsat * latent_heeat_factor
                )
            elif PDFSHAPE == 2:
                large_scale_condensate_internal = large_scale_condensate_internal_old + 0.5 * (large_scale_condensate_internal - large_scale_condensate_internal_old) / (
                    1.0 - (large_scale_cloud_fraction_internal * (alpha - 1.0) - (large_scale_condensate_internal / qsat)) * dqsat * latent_heeat_factor
                )
            elif PDFSHAPE >= 5:
                large_scale_condensate_internal = large_scale_condensate_internal_old + 0.7 * (large_scale_condensate_internal - large_scale_condensate_internal_old)

            # update vapor based on condensation changes
            vapor_internal = vapor_internal_old - (large_scale_condensate_internal - large_scale_condensate_internal_old)

            # temperature update methods
            if iteration_method_internal == 1:
                # use fixed-point iteration
                t_internal = (
                    t_internal_old
                    + (1.0 - fraction_ice)
                    * (MAPL_ALHL / MAPL_CP)
                    * (large_scale_condensate_internal - large_scale_condensate_internal_old)
                    * (1.0 - large_scale_cloud_fraction)
                    + fraction_ice * (MAPL_ALHS / MAPL_CP) * (large_scale_condensate_internal - large_scale_condensate_internal_old) * (1.0 - large_scale_cloud_fraciton)
                )

                PDFITERS = count
                if abs(t_internal - t_internal_old) < 0.00001:
                    count += max_iterations + 1  # stop the count

            else:
                # secant method
                f_t_internal = (
                    t_internal_old
                    + (1.0 - fraction_ice)
                    * (MAPL_ALHL / MAPL_CP)
                    * (large_scale_condensate_internal - large_scale_condensate_internal_old)
                    * (1.0 - large_scale_cloud_fraction)
                    + fraction_ice * (MAPL_ALHS / MAPL_CP) * (large_scale_condensate_internal - large_scale_condensate_internal_old) * (1.0 - large_scale_cloud_fraciton)
                )

                PDFITERS = count
                if abs(f_t_internal - t_internal_old) < 1.0e-6:
                    t_internal = f_t_internal
                    count = max_iterations + 1  # stop the count

                if count > 1:
                    # prevent division by zero
                    denom = (f_t_internal - t_internal_old) - (t_internal_previous - f_t_internal_previous)
                    if abs(denom) > 1.0e-10:
                        t_internal = t_internal_old - (f_t_internal - t_internal_old) * (t_internal_old - t_internal_previous) / denom
                    else:
                        t_internal = f_t_internal
                        count = max_iterations + 1  # stop the count
                else:
                    t_internal = t_internal_old  # first iteration, use initial guess

                # store values for next secant iteration step
                t_internal_previous = t_internal_old
                f_t_internal_previous = f_t_internal

            count += 1

        if PDFSHAPE == 6:
            option_not_implemented = True

        # PHASE 4: Finalization & Mapping back to Absolute Grid Box
        # scale the environmental values back down to grid-box absolutes,
        # partition into ice/liquid, and update prognostic variables.

        large_scale_cloud_fraction = large_scale_cloud_fraction_internal * (1.0 - convective_cloud_fraction)
        large_scale_condensate_internal = large_scale_condensate_internal * (1.0 - convective_cloud_fraction)

        # determine net change in total resolved condensate
        excess_condensate = large_scale_condensate_internal - (QLLS + large_scale_ice)

        if excess_condensate < 0.0:
            # net evaporation: liquid evaporates first, then ice
            dliquid = max(excess_condensate, -QLLS)
            dice = max(excess_condensate - dliquid, -large_scale_ice)
        else:
            # net condensation: partition based on ice fraction
            dliquid = (1.0 - fraction_ice) * excess_condensate
            dice = fraction_ice * excess_condensate

        # clean up residual clouds if fraction is negligibly small
        if large_scale_cloud_fraction < MIN_CLOUD_FRACTION:
            dice = -large_scale_ice
            dliquid = -large_scale_liquid

        # update global arrays
        large_scale_ice = large_scale_ice + dice
        large_scale_liquid = large_scale_liquid + dliquid
        vapor = vapor - (dice + dliquid)

        # final temperature update (applies latent heat of vaporization and fusion)
        t = t + (MAPL_ALHL / MAPL_CP) * (dice + dliquid) + MAPL_ALHF / MAPL_CP * (dice)


def melt_freeze(
    convection_fraction: FloatFieldIJ,
    surface_type: FloatFieldIJ,
    t: FloatField,
    liquid: FloatField,
    ice: FloatField,
):
    """_summary_

    Args:
        convection_fraction (FloatFieldIJ): _description_
        surface_type (FloatFieldIJ): _description_
        t (FloatField): _description_
        liquid (FloatField): _description_
        ice (FloatField): _description_
    """

    from __externals__ import dtime

    with computation(FORWARD), interval(0, 1):
        latent_heat_fusion: FloatFieldIJ = MAPL_ALHS - MAPL_ALHL

    with computation(PARALLEL), interval(...):
        if t <= MAPL_TICE:
            # FREEZING REGIME (TE <= TICE)

            # 1. Target ice deficit (new_ice_condensate)
            fraction_ice = ice_fraction(t, convection_fraction, surface_type)
            target_ice = min(max(0.0, fraction_ice * (liquid + ice) - ice), liquid)

            # 2. Thermodynamic limit (prevent latent heating above freezing point)
            max_phase_change = max(0.0, (MAPL_TICE - t) * MAPL_CP / latent_heat_fusion)

            # 3. Apply relaxation timescale
            condensate_phase_changed = (1.0 - exp(-dtime / max(dtime, TAUFRZ))) * min(target_ice, max_phase_change)

            # 4. Update states (liquid -> ice, temp warms)
            ice = ice + condensate_phase_changed
            liquid = liquid - condensate_phase_changed
            t = t + (latent_heat_fusion * condensate_phase_changed) / MAPL_CP

        else:
            # MELTING REGIME (TE > TICE)

            # 1. Target melt (assuming 0% ice fraction above freezing)
            target_melt = ice

            # 2. Thermodynamic limit (prevent latent cooling below freezing point)
            max_phase_change = max(0.0, (t - MAPL_TICE) * MAPL_CP / latent_heat_fusion)

            # 3. Apply relaxation timescale
            condensate_phase_changed = (1.0 - exp(-dtime / max(dtime, TAUMLT))) * min(target_melt, max_phase_change)

            # 4. Update states (ice -> liquid, temp cools)
            ice = ice - condensate_phase_changed
            liquid = liquid + condensate_phase_changed
            t = t - (latent_heat_fusion * condensate_phase_changed) / MAPL_CP


def evaporate(
    p_mb: FloatField,
    t: FloatField,
    vapor: FloatField,
    rh_crit: FloatField,
    liquid: FloatField,
    ice: FloatField,
    cloud_fraction: FloatField,
    concentration_liquid: FloatField,
    saturation_specific_humidity: FloatField,
):
    from __externals__ import DTIME, CCW_EVAP_EFF

    # EVAPORATION OF CLOUD WATER - DelGenio et al (1996, J. Clim., 9, 270-303) formulation (Eq.s 15-17)

    with computation(PARALLEL), interval(...):
        es = 100.0 * p_mb * saturation_specific_humidity / (EPSILON + (1.0 - EPSILON) * saturation_specific_humidity)  # (100's <-^ convert from mbar to Pa)

        rh_limited = min(vapor / saturation_specific_humidity, 1.00)

        k1 = (MAPL_ALHL**2) * RHO_W / (K_COND * MAPL_RVAP * (t**2))

        # DIFFU is given for 1000 mb, so 1000.0/p_mb accounts for increased diffusivity at lower pressure
        k2 = MAPL_RVAP * t * RHO_W / (DIFFU * (1000.0 / p_mb) * es)

        if cloud_fraction > 0.0 and liquid > 0.0:
            liquid_modified = liquid / cloud_fraction
        else:
            liquid_modified = 0.0

        radius = cloud_effective_radius_liquid(p_mb, t, liquid_modified, concentration_liquid)

        if rh_limited < rh_crit and radius > 0.0:
            evap = CCW_EVAP_EFF * liquid * DTIME * (rh_crit - rh_limited) / ((k1 + k2) * radius**2)
            evap = max(0.0, min(evap, liquid))
        else:
            evap = 0.0

        total_condensate = ice + liquid
        if total_condensate > 0.0:
            cloud_fraction = cloud_fraction * (total_condensate - evap) / total_condensate

        vapor = vapor + evap
        liquid = liquid - evap
        t = t - (MAPL_ALHL / MAPL_CP) * evap


def sublimate(
    p_mb: FloatField,
    t: FloatField,
    vapor: FloatField,
    rh_crit: FloatField,
    liquid: FloatField,
    ice: FloatField,
    cloud_fraction: FloatField,
    saturation_specific_humidity: FloatField,
):
    from __externals__ import DTIME, CCI_EVAP_EFF

    # SUBLIMATION OF CLOUD WATER - DelGenio et al (1996, J. Clim., 9, 270-303) formulation (Eq.s 15-17)

    with computation(PARALLEL), interval(...):
        es = 100.0 * p_mb * saturation_specific_humidity / (EPSILON + (1.0 - EPSILON) * saturation_specific_humidity)  # (100's <-^ convert from mbar to Pa)

        rh_limited = min(vapor / saturation_specific_humidity, 1.00)

        # NOTE MAPL_ALHS is MAPL_ALHL in the fortran. this is a bug. has been fixed in the NDSL
        # but IS STILL WRONG IN THE FORTRAN. this will lead to numerical differences
        k1 = (MAPL_ALHS**2) * RHO_I / (K_COND * MAPL_RVAP * (t**2))

        # DIFFU is given for 1000 mb, so 1000.0/p_mb accounts for increased diffusivity at lower pressure
        k2 = MAPL_RVAP * t * RHO_I / (DIFFU * (1000.0 / p_mb) * es)

        if cloud_fraction > 0.0 and ice > 0.0:
            ice_modified = ice / cloud_fraction
        else:
            ice_modified = 0.0

        radius = cloud_effective_radius_ice(p_mb, t, ice_modified)

        if rh_limited < rh_crit and radius > 0.0:
            subl = CCI_EVAP_EFF * ice * DTIME * (rh_crit - rh_limited) / ((k1 + k2) * radius**2)
            subl = max(0.0, min(subl, ice))
        else:
            subl = 0.0

        total_condensate = ice + liquid
        if total_condensate > 0.0:
            cloud_fraction = cloud_fraction * (total_condensate - subl) / total_condensate

        vapor = vapor + subl
        ice = ice - subl
        t = t - (MAPL_ALHS / MAPL_CP) * subl
