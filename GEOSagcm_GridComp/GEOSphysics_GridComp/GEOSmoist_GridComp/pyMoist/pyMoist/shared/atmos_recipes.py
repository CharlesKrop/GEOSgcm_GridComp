"""Stencils and functions called by multiple pyMoist modules.
These functions perform basic math and calculate fundamental
meteorological quantities"""

from ndsl.dsl.gt4py import exp, function, computation, FORWARD, PARALLEL, interval, sqrt
from ndsl.dsl.typing import Float, Bool, FloatField, FloatFieldIJ

from pyMoist.constants import MAPL_GRAV, MAPL_CP
from pyMoist.shared.constants import SIGMA_DX, SIGMA_EXP


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


def dissipative_kinetic_energy_heating(
    mass: FloatField,
    u: FloatField,
    v: FloatField,
    du: FloatField,
    dv: FloatField,
    dt: FloatField,
):
    """Dissapate kinetic energy into heat

    Args:
        mass (FloatField)
        u (FloatField)
        v (FloatField)
        du (FloatField)
        dv (FloatField)
        dt (FloatField)
    """
    # since kinetic energy is being dissipated, add heating accordingly (from ECMWF)
    with computation(FORWARD), interval(0, 1):
        dts: FloatFieldIJ = 0.0
        fpi: FloatFieldIJ = 0.0

    with computation(FORWARD), interval(...):
        dt = 0.0
        # total KE dissiptaion estimate
        dts = dts - (du * u + dv * v) * mass
        # fpi needed for calcualtion of conversion to potential energy integrated
        ke = sqrt(du * du + dv * dv)
        fpi = fpi + ke * mass

        if fpi > 0.0:
            dt = (ke / fpi) * dts * (1.0 / MAPL_CP)


def fill_negative_q(
    q: FloatField,
    dqdt: FloatField,
    mass: FloatField,
    fill_dqdt: Bool,
):
    """Fill negative values of a water species (mixing ratio)

    Args:
        q (FloatField): water species/mixing ratio
        dqdt (FloatField): tendency do to fill - only written if fill_dqdt is True
        mass (FloatField): mass of the air parcel
        fill_dqdt (Bool): controls read/write of dqdt
    """
    from __externals__ import DTIME

    with computation(PARALLEL), interval(...):
        # save original q if tendency is requested
        if fill_dqdt:
            dqdt = q

    with computation(FORWARD), interval(0, 1):
        # fill internal temporaries
        total_precipitable_water_before: FloatFieldIJ = 0.0
        total_precipitable_water_after: FloatFieldIJ = 0.0

    with computation(FORWARD), interval(...):
        # moisture limited: per column mass conserving q fix
        total_precipitable_water_before = total_precipitable_water_before + q * mass

    with computation(PARALLEL), interval(...):
        # remove negative values
        if q < 0.0:
            q = 0.0

    with computation(FORWARD), interval(...):
        # compute total precipitable water after removing negative values
        total_precipitable_water_after = total_precipitable_water_after + q * mass

    with computation(FORWARD), interval(0, 1):
        dtpw = total_precipitable_water_before - total_precipitable_water_after  # > 0 means mass was removed

    # redistribute delta TPW to positive layers only
    with computation(FORWARD), interval(...):
        if abs(dtpw) > 1.0e-15:
            positive_mass = 0.0
            if q > 0.0:
                positive_mass = positive_mass + mass

    with computation(FORWARD), interval(...):
        if abs(dtpw) > 1.0e-15:
            if positive_mass > 0.0:
                if q > 0.0:
                    q = q + dtpw * (mass / positive_mass)
                    if q < 0.0:
                        q = 0.0  # safety

    with computation(PARALLEL), interval(...):
        # update dqdt if requested
        if fill_dqdt:
            dqdt = (q - dqdt) / DTIME
