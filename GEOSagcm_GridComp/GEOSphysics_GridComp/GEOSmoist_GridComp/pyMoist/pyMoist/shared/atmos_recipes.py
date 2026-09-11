"""Stencils and functions called by multiple pyMoist modules.
These functions perform basic math and calculate fundamental
meteorological quantities"""

from ndsl.dsl.gt4py import exp, function
from ndsl.dsl.typing import Float

from pyMoist.constants import MAPL_GRAV, SIGMA_DX, SIGMA_EXP


@function
def air_density(PL: Float, TE: Float) -> Float:
    """
    Calculate air density [kg/m^3]

    Parameters:
    PL (Float): Pressure level.
    TE (Float): Temperature.

    Returns:
    Float: Calculated air density.
    """
    air_density = (100.0 * PL) / (MAPL_GRAV * TE)
    return air_density


@function
def sigma(dx, custom_dx: Float = SIGMA_DX, custom_exp: Float = SIGMA_EXP) -> Float:
    """Arakawa 2011 based sigma function"""
    sigma = (1.0 - 0.9839 * exp(-0.09835 * (dx / custom_dx))) ** custom_exp

    return sigma


@function
def compute_estimated_inversion_strength_factor(estimated_inversion_strength) -> Float:
    if estimated_inversion_strength >= 10.0:
        # Very stable regime
        eis_factor = 1.0
    elif estimated_inversion_strength <= 0.0:
        # Very unstable regime
        eis_factor = 0.0
    else:
        # Smooth function from 0 to 1
        eis_factor = (estimated_inversion_strength / 10.0) ** 2

    return eis_factor
