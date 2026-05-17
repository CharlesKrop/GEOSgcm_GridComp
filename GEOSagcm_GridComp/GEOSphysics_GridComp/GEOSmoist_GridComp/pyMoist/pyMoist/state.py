import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int


@dataclasses.dataclass
class GridData:
    """Grid Information"""

    area: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lattiude: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    longitude: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )


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
    class VerticalMotion:
        velocity: Quantity = dataclasses.field(
            metadata={
                "name": "velocity",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m s-1",
                "intent": "?",
                "dtype": Float,
            }
        )
        variance: Quantity = dataclasses.field(
            metadata={
                "name": "variance",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m2 s-2",
                "intent": "?",
                "dtype": Float,
            }
        )
        third_moment: Quantity = dataclasses.field(
            metadata={
                "name": "third_moment",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m3 s-3",
                "intent": "?",
                "dtype": Float,
            }
        )

    vertical_motion: VerticalMotion


@dataclasses.dataclass
class SurfaceConditions:
    """Core 2D surface condtions"""

    t_surface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    t_surface_air: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    t_2m: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    specific_humidity_surface_air: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    specific_humidity_2m: Quantity = dataclasses.field(
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
    geopotential_height: Quantity = dataclasses.field(
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
    pbl_level_shallow_convection: Quantity = dataclasses.field(
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
    rain: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    graupel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    snow: Quantity = dataclasses.field(
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
    ice_particle_effective_radius: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    liquid_particle_effective_radius: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    ice_fraction_in_convective_tower: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_rainwater_source: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_condensate_source: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_condensate_grid_mean: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    initial_total_precipitable_water: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    initial_total_precipitable_water_saturation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    entrained_ice_sink_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    entrained_liquid_sink_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    cloud_fraction_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    number_liquid_droplet_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    number_ice_crystal_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )

    @dataclasses.dataclass
    class LiquidWaterStaticEnergy:
        """
        Units:
            flux: K m s-1
            variance: K+2
            third_moment: K+3
        """

        flux: Quantity = dataclasses.field(
            metadata={
                "name": "flux",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "K m s-1",
                "intent": "?",
                "dtype": Float,
            }
        )
        variance: Quantity = dataclasses.field(
            metadata={
                "name": "variance",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "K+2",
                "intent": "?",
                "dtype": Float,
            }
        )
        third_moment: Quantity = dataclasses.field(
            metadata={
                "name": "third_moment",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "K+3",
                "intent": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class TotalWater:
        """
        Optional outputs of Hydrostatic PDF for PDF_Shape=5

        Units:
            flux: kg kg-1 m s-1
            variance: 1
            third_moment: 1
        """

        flux: Quantity = dataclasses.field(
            metadata={
                "name": "flux",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 m s-1",
                "intent": "?",
                "dtype": Float,
            }
        )
        variance: Quantity = dataclasses.field(
            metadata={
                "name": "variance",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "1",
                "intent": "?",
                "dtype": Float,
            }
        )
        third_moment: Quantity = dataclasses.field(
            metadata={
                "name": "third_moment",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "1",
                "intent": "?",
                "dtype": Float,
            }
        )

    liquid_water_static_energy: LiquidWaterStaticEnergy
    total_water: TotalWater


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
    dt_dt_friction_pressure_weighted: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_shortwave: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_longwave: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_pbl: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dt_dt_from_dynamics: Quantity = dataclasses.field(
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
    dspecific_humidity_dt_pbl: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    dspecific_humidity_dt_from_dynamics: Quantity = dataclasses.field(
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
    # pressure related
    dlayer_pressure_thickness_dt: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # mass fraction
    mass_fraction_suspended_rain: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_fraction_suspended_graupel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_fraction_suspended_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    # shallow convection
    subsidence_liquid_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    subsidence_ice_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_column_water_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    total_column_moist_static_energy_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
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
    ice_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    ice_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    ice_anvil: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    ice_nonanvil_large_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    ice_nonconvective: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    liquid_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    liquid_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    liquid_anvil: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    liquid_nonanvil_large_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    liquid_nonconvective: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    shallow_convective_rain: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    shallow_convective_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    falling_graupel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    falling_ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    falling_rain: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    falling_snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_precipitation_from_RAS: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class RadiationState:
    """Copy of various fields for use in radiation"""

    specific_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    liquid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    ice: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    rain: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    graupel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    snow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    cloud_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class StateBeforeDynamics:
    """State fields at the start of the timestep (before dynamics)"""

    p_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
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
    specific_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
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
class ConvectiveDiagnostics:
    convection_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cape_surface_parcel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cin_surface_parcel: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
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
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    sbcin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    mlcape: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    mlcin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    mucape: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    mucin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lfc: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lnb: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
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
    stochastic_factor: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    convective_precipitation_evaporation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convective_precipitation_sublimation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    entrainment_parameter: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lateral_entrainment_rate: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lateral_entrainment_rate_shallow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lateral_entrainment_rate_mid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lateral_entrainment_rate_deep: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lateral_entrainment_rate_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    lateral_detrainment_rate_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    updraft_areal_fraction: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    updraft_vertical_velocity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    sigma_mid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    sigma_mid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    sigma_deep: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    pressure_shallow_convective_cloud_top: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    pressure_mid_convective_cloud_top: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    pressure_deep_convective_cloud_top: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    mass_flux_shallow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_shallow_updraft_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_shallow_updraft_detrained: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_mid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_deep_updraft: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_deep_updraft_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    mass_flux_deep_updraft_detrained: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_deep_downdraft: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_cloud_base: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_cloud_base_shallow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_cloud_base_mid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    mass_flux_cloud_base_deep: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    convection_code_shallow: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    convection_code_mid: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    convection_code_deep: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_workfunction_0: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_workfunction_1: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_workfunction_2: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_workfunction_3: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_workfunction_1_pbl: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_workfunction_1_cin: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    pbl_time_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cape_removal_time_scale: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lightning_density: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    convection_tracer: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cumulus_scale_height_from_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_mass_detrained_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    total_water_flux_deep_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    total_water_flux_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    liquid_static_energy_flux_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    u_flux_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    v_flux_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "dtype": Float,
        }
    )
    source_air_t_perturbation_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    source_air_humidity_perturbation_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    detrained_ice_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    detrained_liquid_shallow_convection_interface: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    scale_height_shallow_convection: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class MicrophysicsDiagnostics:
    pdf_first_plume_fractional_area: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    covariance_liquid_water_static_energy_and_total_water_specific_humidity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    cloud_liquid_evaporation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    cloud_ice_evaporation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    relative_humidity_after_pdf: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    buoyancy_flux: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    liquid_water_flux: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    hydrostatic_pdf_iterations: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    critical_relative_humidity_for_pdf: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    large_scale_rainwater_source: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    nonanvil_large_scale_precipitation_evaporation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    nonanvil_large_scale_precipitation_sublimation: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class Diagnostics:
    negative_vapor_adjustment_start: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    negative_vapor_adjustment_end: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    highest_level_of_scalar_diffusivity_gt_2: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lowest_level_of_scalar_diffusivity_gt_2: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    condensed_water_path: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    cloud_liquid_water_path: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    liquid_water_path: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    ice_water_path: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    total_precipitable_water: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lightning_flash_rate: Quantity = dataclasses.field(
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
    lower_tropospheric_stability: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    estimated_inversion_strength: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    lcl_height: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    turbulent_kinetic_energy_fraction_from_vertical_velocity: Quantity = dataclasses.field(
        metadata={
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )

    @dataclasses.dataclass
    class Radar:
        simulated_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "dims": [I_DIM, J_DIM, K_DIM],
                "dtype": Float,
            }
        )
        maximum_composite_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )
        base_1km_agl_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )
        echo_top_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )
        minus_10c_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "dims": [I_DIM, J_DIM],
                "dtype": Float,
            }
        )

    radar: Radar


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

    grid_data: GridData
    atmospheric_state: AtmosphericState
    surface_conditions: SurfaceConditions
    levels: Levels
    cloud_condensates: CloudCondensates
    tendencies: Tendencies
    precipitation_at_surface: PrecipitationAtSurface
    precipitation_flux: PrecipitationFlux
    radiation_state: RadiationState
    state_before_dynamics: StateBeforeDynamics
    state_at_input: StateAtInput
    state_at_output: StateAtOutput
    convective_diagnostics: ConvectiveDiagnostics
    microphysics_diagnostics: MicrophysicsDiagnostics
    diagnostics: Diagnostics
