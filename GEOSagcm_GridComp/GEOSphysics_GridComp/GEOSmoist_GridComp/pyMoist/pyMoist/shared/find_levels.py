from ndsl.dsl.gt4py import log, function
from ndsl.dsl.typing import Float


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
    term_1 = 1.0 / (t - 55.0)
    term_2 = log(max(0.1, rh) / 100.0) / 2840.0
    denom = term_1 - term_2
    t_lcl = (1.0 / denom) + 55.0
    return t_lcl
