"""Stencils and functions called by multiple pyMoist modules.
These functions perform basic math and calculate fundamental
meteorological quantities"""

from ndsl.dsl.gt4py import exp, function
from ndsl.dsl.typing import Float

from pyMoist.constants import MAPL_GRAV, SIGMA_EXP, SIGMA_DX


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
def sigma(dx, custom_dx: Float = -9e10, custom_exp: Float = -9e10) -> Float:
    """Arakawa 2011 based sigma function"""
    internal_exp = SIGMA_EXP
    if custom_exp != -9e10:
        internal_exp = custom_exp
    if custom_dx != -9e10:
        sigma = (1.0 - 0.9839 * exp(-0.09835 * (dx / custom_dx))) ** internal_exp
    else:
        sigma = (1.0 - 0.9839 * exp(-0.09835 * (dx / SIGMA_DX))) ** internal_exp

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
