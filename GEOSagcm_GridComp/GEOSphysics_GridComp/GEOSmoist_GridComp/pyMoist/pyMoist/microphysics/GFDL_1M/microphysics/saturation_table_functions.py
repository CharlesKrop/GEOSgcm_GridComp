from ndsl.dsl.gt4py import function
from ndsl.dsl.typing import Float, Int
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3SaturationTable
from pyMoist.microphysics.GFDL_1M.microphysics.constants import RDELT, RDGAS, RVGAS, SATURATION_TABLE_LENGTH, SATURATION_TABLE_TMIN, ZVIR


@function
def saturation_specific_humidity(t: Float, density: Float, table: GFDLMPV3SaturationTable, dtable: GFDLMPV3SaturationTable):
    """Compute the saturated specific humidity based on a desired table.

    table_0/dtable_0 considers only water: useful for idealized experiments (it can also only be used in warm rain microphysics)
    table_1/dtable_1 considers water and ice: most realistic saturation water vapor pressure for the full temperature range
    table_2/dtable_2 considers water and ice: it is not designed for mixed-phase cloud microphysics. used for ice microphysics (< 0C) or warm rain microphysics (> 0 C)

    Args:
        t (Float): temperature
        density (Float): air density
        table (GFDLMPV3SaturationTable): saturation table
        dtable (GFDLMPV3SaturationTable): derivative of the saturation table
    """
    ap1 = RDELT * max(t - SATURATION_TABLE_TMIN, 0.0) + 1.0
    ap1 = min(Float(SATURATION_TABLE_LENGTH), ap1)
    sat_p = saturation_water_pressure(t, table, dtable) / (RVGAS * t * density)
    index: Int = Int(ap1 - 0.5)
    # Apply protections (bounds checking)
    # Ensures index is >= 0 AND it+1 <= es_table_length - 1
    index = max(0, min(index, SATURATION_TABLE_LENGTH - 2))
    return RDELT * (dtable.A[index] + (ap1 - index) * (dtable.A[index + 1] - dtable.A[index])) / (RVGAS * t * density)


@function
def saturation_specific_humidity_no_density(t: Float, p: Float, vapor: Float, table: GFDLMPV3SaturationTable, dtable: GFDLMPV3SaturationTable):
    """Compute the saturated specific humidity based on a desired table when density is not available.

    table_0/dtable_0 considers only water: useful for idealized experiments (it can also only be used in warm rain microphysics)
    table_1/dtable_1 considers water and ice: most realistic saturation water vapor pressure for the full temperature range
    table_2/dtable_2 considers water and ice: it is not designed for mixed-phase cloud microphysics. used for ice microphysics (< 0C) or warm rain microphysics (> 0 C)

    Args:
        t (Float): temperature
        p (Float): air pressure
        vapor (Float): water vapor mixing ratio
        table (GFDLMPV3SaturationTable): saturation table
        dtable (GFDLMPV3SaturationTable): derivative of the saturation table
    """

    density = p / (RDGAS * t * (1. + ZVIR * vapor))
    return saturation_specific_humidity(t, density, table, dtable)


@function
def saturation_water_pressure(t: Float, table: GFDLMPV3SaturationTable, dtable: GFDLMPV3SaturationTable):
    """Compute the saturation water pressure based on a desired table.

    table_0/dtable_0 considers only water: useful for idealized experiments (it can also only be used in warm rain microphysics)
    table_1/dtable_1 considers water and ice: most realistic saturation water vapor pressure for the full temperature range
    table_2/dtable_2 considers water and ice: it is not designed for mixed-phase cloud microphysics. used for ice microphysics (< 0C) or warm rain microphysics (> 0 C)

    Args:
        t (Float): temperature
        table (GFDLMPV3SaturationTable): saturation table
        dtable (GFDLMPV3SaturationTable): derivative of the saturation table
    """
    ap1 = RDELT * max(t - SATURATION_TABLE_TMIN, 0.0) + 1.0
    ap1 = min(Float(SATURATION_TABLE_LENGTH), ap1)
    index: Int = Int(ap1)
    # Apply protections (bounds checking)
    # Ensures index is >= 0 AND index+1 <= es_table_length - 1
    index = max(0, min(index, SATURATION_TABLE_LENGTH - 2))
    return table.A[index] + (ap1 - index) * dtable.A[index]
