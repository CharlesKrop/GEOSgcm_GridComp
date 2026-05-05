import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int


@dataclasses.dataclass
class AtmosphericState:
    """Core 3D atmospheric fields"""

    t: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    u: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    v: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    w: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    omega: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    turbulent_kinetic_energy: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    p_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    z_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    scalar_diffusivity_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    reference_pressure: Quantity = dataclasses.field(
        metadata={
            "dims": [K_INTERFACE_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class SurfaceConditions:
    """Core 2D surface condtions"""

    t_surface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    sensible_heat_flux: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_evaporation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    land_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    land_ice_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    ice_covered_fraction_of_tile: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    snow_mass: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_type: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )


class Levels:
    """2D fields which specify a particular level"""

    pbl_level: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Int,
        }
    )
    cbl_level_before_moist: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Int,
        }
    )


@dataclasses.dataclass
class CloudCondensates:
    """3D cloud vapor/liquid/ice mixing ratios (total, large scale, and convective)."""

    specific_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_ice_cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_liquid_cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_water: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    liquid_concentration: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    ice_concentration: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    relative_humidity_wrt_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    relative_humidity_wrt_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    number_concentration_water_friendly_aerosols: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    ice_ccn_concentration: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    liquid_ccn_concentration: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class ConvectiveDiagnostics:
    convection_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    cape_surface_parcel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    cin_surface_parcel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    buoyancy_surface_parcel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    sbcape: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    sbcin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mlcape: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mlcin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mucape: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mucin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lfc_level: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lnb_level: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_cumulative_mass_flux: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_detraining_mass_flux: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class Tendencies:
    # zonal wind
    du_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    du_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    du_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    du_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    du_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # meridional wind
    dv_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dv_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dv_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dv_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dv_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # temperature
    dt_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_from_rh_cleanup: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # specific humidity
    dspecific_humidity_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dspecific_humidity_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dspecific_humidity_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dspecific_humidity_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dspecific_humidity_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dspecific_humidity_dt_from_rh_cleanup: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # liquid
    dliquid_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dliquid_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dliquid_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dliquid_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dliquid_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # ice
    dice_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dice_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dice_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dice_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dice_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # rain
    drain_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    drain_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    drain_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    drain_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # snow
    dsnow_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dsnow_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dsnow_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dsnow_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # graupel
    dgraupel_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dgraupel_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dgraupel_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # cloud fraction
    dtotal_cloud_fraciton_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dtotal_cloud_fraciton_dt_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dtotal_cloud_fraciton_dt_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dtotal_cloud_fraciton_dt_macrophysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dtotal_cloud_fraciton_dt_microphysics: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    #
    dlayer_pressure_thickness_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class PrecipitationAtSurface:
    total_precipitation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    precipitation_total: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rainfall: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    spurious_rain_from_relative_humidity_cleanup: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    total_convective_precipitation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_all_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_deep_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_GF_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    total_stratiform_precipitation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_all_large_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_large_scale_nonanvil: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    rain_from_large_scale_anvil: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    snowfall: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    snowfall_total: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    large_scale_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    anvil_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    deep_convection_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    shallow_convection_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    kuchera_snow_to_liquid_ratio: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    icefall: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    freezing_rainfall: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_precipitation_type: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class PrecipitationFlux:
    flux_ice_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_ice_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_ice_anvil: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_ice_nonanvil_large_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_ice_nonconvective: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_liquid_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_liquid_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_liquid_anvil: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_liquid_nonanvil_large_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    flux_liquid_nonconvective: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class StateAtInput:
    """Copy of various other state fields saved at the start of Moist"""

    pt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    t_surface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    u: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    v: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    specific_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    vapor: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_condensates: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_condensates: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class StateAtOutput:
    t: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    pt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    u: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    v: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    specific_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    relative_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    saturation_ratio: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dry_static_energy: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class Diagnostics:
    negative_vapor_adjustment_start = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    negative_vapor_adjustment_end = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    highest_level_of_scalar_diffusivity_gt_2 = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lowest_level_of_scalar_diffusivity_gt_2 = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    condensed_water_path = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_liquid_water_path = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    liquid_water_path = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    ice_water_path = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    total_precipitable_water = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lightning_flash_rate = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class MoistState(State):
    """
    State for the GEOS moist physics package (MOIST GridComp).

    Covers every field fetched via MAPL_GetPointer / ESMF_StateGet in the
    moist RUN subroutine, organised by the physical role of each field rather
    than by the ESMF state bucket (IMPORT / INTERNAL / EXPORT) it lives in.

    Fields that are optional in the Fortran (fetched without ALLOC=.TRUE. and
    guarded by `if (associated(...))`) are typed as `Quantity`.
    """

    atmospheric_state: AtmosphericState
    surface_conditions: SurfaceConditions
    levels: Levels
    cloud_condensates: CloudCondensates
    convective_diagnostics: ConvectiveDiagnostics
    tendencies: Tendencies
    precipitation: PrecipitationAtSurface
    precipitation_flux: PrecipitationFlux
    state_at_input: StateAtInput
    state_at_output: StateAtOutput
    diagnostics: Diagnostics
