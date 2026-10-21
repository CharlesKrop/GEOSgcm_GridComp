from ndsl.dsl.gt4py import floor, function, int32

from pyMoist.saturation_tables.constants import DEGSUBS, ERFAC, ESFAC, MAPL_TICE, MAX_MIXING_RATIO, TABLESIZE, TMAXTBL, TMINLQU, TMINTBL


@function
def saturation_specific_humidity_frozen_surface(
    ese,
    frz,
    t,
    p=-999.0,
):
    """Computes saturation specific humidity over liquid water surface, using
    data from saturation pressure tables.

    Tables must be initialized before use.

    Args:
        ese (Float): saturation pressure table in Pascals, specifics unknown
        frz (Float): saturation pressure at reference temperature (273.16 K)
        t (Float): temperature in Kelvin
        p (Float, optional): pressure in Pascals. Defaults to -999.

    Returns:
        qsat (Float): saturation specific humidity
        dqsat (Float): derivative saturation specific humidity with respect to temperature
    """
    dqsat = 0.0
    if t <= TMINTBL:
        qsat = ese.A[0]  # type: ignore
        ddq = 0.0
    elif t >= MAPL_TICE:
        qsat = frz
        ddq = 0.0
    else:
        t = (t - TMINTBL) * DEGSUBS + 1.0
        t_integer = int(floor(t))
        ddq = ese.A[t_integer] - ese.A[t_integer - 1]  # type: ignore
        qsat = (t - t_integer) * ddq + ese.A[t_integer - 1]  # type: ignore

    if p > 0:
        # apply pressure correction
        if p > qsat:
            dd = ESFAC / (p - (1.0 - ESFAC) * qsat)
            qsat = qsat * dd
            dqsat = ddq * ERFAC * p * dd * dd
        else:
            qsat = MAX_MIXING_RATIO
            dqsat = 0.0
    else:
        dqsat = ddq

    return qsat, dqsat


@function
def saturation_specific_humidity_liquid_surface(
    esw,
    lqu,
    t,
    p=-999.0,
):
    """Computes saturation specific humidity over liquid water surface, using
    data from saturation pressure tables.

    Tables must be initialized before use.

    Arguments:
        esw (Float): saturation pressure table in Pascals, specifics unknown
        lqu (Float): saturation pressure at reference temperature (233.16 K)
        t (Float): temperature in Kelvin
        p (Float, optional): pressure in Pascals. Defaults to -999.

    Returns:
        qsat (Float): saturation specific humidity
        dqsat (Float): derivative saturation specific humidity with respect to temperature
    """
    dqsat = 0.0
    if t <= TMINLQU:
        qsat = lqu
        ddq = 0.0
    elif t >= TMAXTBL:
        TABLESIZE_MINUS_1: int32 = TABLESIZE - 1
        qsat = esw.A[TABLESIZE_MINUS_1]  # type: ignore
        ddq = 0.0
    else:
        t = (t - TMINTBL) * DEGSUBS + 1.0
        t_integer = int(floor(t))
        ddq = esw.A[t_integer] - esw.A[t_integer - 1]  # type: ignore
        qsat = (t - t_integer) * ddq + esw.A[t_integer - 1]  # type: ignore

    if p > 0:
        # apply pressure correction
        if p > qsat:
            dd = ESFAC / (p - (1.0 - ESFAC) * qsat)
            qsat = qsat * dd
            dqsat = ddq * ERFAC * p * dd * dd
        else:
            qsat = MAX_MIXING_RATIO
            dqsat = 0.0
    else:
        dqsat = ddq

    return qsat, dqsat


@function
def saturation_specific_humidity(t, p, esx, use_dqsat_order=False):
    """Compute saturation specific humidity and derivative saturation specific humidity
    with respect to temperature from saturation pressure tables.

    Tables must be initialized before use.

    Arguments:
        t (Float): temperature in Kelvin
        p (Float): pressure in Pascals
        esx (Float): saturation pressure table in Pascals, specifics unknown
        use_dqsat_order (bool): match GEOS_DQSAT's derivative evaluation order

    Returns:
        qsat (Float): saturation specific humidity
        dqsat (Float): derivative saturation specific humidity with respect to temperature
    """

    if t <= TMINTBL:
        t = TMINTBL
    elif t >= TMAXTBL - 0.001:
        t = TMAXTBL - 0.001

    t = (t - TMINTBL) * DEGSUBS + 1.0
    t_integer = int32(floor(t))
    IT_MINUS_1 = t_integer - 1

    dq = esx.A[t_integer] - esx.A[IT_MINUS_1]  # type: ignore
    qsat = (t - t_integer) * dq + esx.A[IT_MINUS_1]  # type: ignore

    if p <= qsat:
        qsat = MAX_MIXING_RATIO
        dqsat = 0.0
    else:
        dd = 1.0 / (p - (1.0 - ESFAC) * qsat)
        qsat = ESFAC * qsat * dd
        # Match the selected Fortran entry point's evaluation order. A few ULPs
        # in this derivative can change a PDF equilibrium iteration's outcome.
        if use_dqsat_order:
            dqsat = (ESFAC * DEGSUBS) * dq * p * (dd * dd)
        else:
            dqsat = dq * DEGSUBS
            dqsat = ESFAC * dqsat * p * (dd * dd)

    return qsat, dqsat
