import dataclasses

from ndsl import Quantity, State
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float


@dataclasses.dataclass
class GFDL1MState(State):
    area: Quantity = dataclasses.field(
        metadata={
            "name": "area",
            "dims": [I_DIM, J_DIM],
            "units": "m2",
            "dtype": Float,
        }
    )
    z_interface: Quantity = dataclasses.field(
        metadata={
            "name": "geopotential_height_interface",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "m",
            "dtype": Float,
        }
    )
    p_interface: Quantity = dataclasses.field(
        metadata={
            "name": "p_interface",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "Pa",
            "dtype": Float,
        }
    )
    t: Quantity = dataclasses.field(
        metadata={
            "name": "t",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "K",
            "dtype": Float,
        }
    )
    u: Quantity = dataclasses.field(
        metadata={
            "name": "u",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m s-1",
            "dtype": Float,
        }
    )
    v: Quantity = dataclasses.field(
        metadata={
            "name": "v",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m s-1",
            "dtype": Float,
        }
    )
    land_fraction: Quantity = dataclasses.field(
        metadata={
            "name": "land_fraction",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    scalar_diffusivity_interface: Quantity | None = dataclasses.field(
        metadata={
            "name": "scalar_diffusivity_interface",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "m2 s-1",
            "dtype": Float,
        }
    )
    pdf_first_plume_fractional_area: Quantity = dataclasses.field(
        metadata={
            "name": "pdf_first_plume_fractional_area",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    covariance_liquid_water_static_energy_and_total_water_specific_humidity: Quantity = dataclasses.field(
        metadata={
            "name": "covariance_liquid_water_static_energy_and_total_water_specific_humudity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "K",
            "dtype": Float,
        }
    )
    surface_temperature: Quantity | None = dataclasses.field(
        metadata={
            "name": "surface_temperature",
            "dims": [I_DIM, J_DIM],
            "units": "K",
            "dtype": Float,
        }
    )
    sensible_heat_flux: Quantity | None = dataclasses.field(
        metadata={
            "name": "sensible_heat_flux",
            "dims": [I_DIM, J_DIM],
            "units": "W m-2",
            "dtype": Float,
        }
    )
    omega: Quantity = dataclasses.field(
        metadata={
            "name": "omega",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa s-1",
            "dtype": Float,
        }
    )
    convection_fraction: Quantity = dataclasses.field(
        metadata={
            "name": "convection_fraction",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    surface_type: Quantity = dataclasses.field(
        metadata={
            "name": "surface_type",
            "dims": [I_DIM, J_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    evaporation: Quantity = dataclasses.field(
        metadata={
            "name": "evaporation",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    surface_geopotential_height: Quantity = dataclasses.field(
        metadata={
            "name": "surface_geopotential_height",
            "dims": [I_DIM, J_DIM],
            "units": "m2 s-2",
            "dtype": Float,
        }
    )
    cloud_liquid_evaporation: Quantity = dataclasses.field(
        metadata={
            "name": "cloud_liquid_evaporation",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    cloud_ice_sublimation: Quantity = dataclasses.field(
        metadata={
            "name": "cloud_ice_sublimation",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    icefall: Quantity = dataclasses.field(
        metadata={
            "name": "icefall",
            "dims": [I_DIM, J_DIM],
            "units": "kg m-2 s-1",
            "dtype": Float,
        }
    )
    freezing_rainfall: Quantity = dataclasses.field(
        metadata={
            "name": "freezing_rainfall",
            "dims": [I_DIM, J_DIM],
            "units": "kg m-2 s-1",
            "dtype": Float,
        }
    )
    relative_humidity_after_pdf: Quantity = dataclasses.field(
        metadata={
            "name": "relative_humidity_after_pdf",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    buoyancy_flux: Quantity = dataclasses.field(
        metadata={
            "name": "buoyancy_flux",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    liquid_water_flux: Quantity = dataclasses.field(
        metadata={
            "name": "liquid_water_flux",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 m s-1",
            "dtype": Float,
        }
    )
    hydrostatic_pdf_iterations: Quantity = dataclasses.field(
        metadata={
            "name": "hydrostatic_pdf_iterations",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    lower_tropospheric_stability: Quantity = dataclasses.field(
        metadata={
            "name": "lower_tropospheric_stability",
            "dims": [I_DIM, J_DIM],
            "units": "K",
            "dtype": Float,
        }
    )
    estimated_inversion_strength: Quantity = dataclasses.field(
        metadata={
            "name": "estimated_inversion_strength",
            "dims": [I_DIM, J_DIM],
            "units": "K",
            "dtype": Float,
        }
    )
    shallow_convection_rain: Quantity | None = dataclasses.field(
        metadata={
            "name": "shallow_convection_rain",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    shallow_convection_snow: Quantity | None = dataclasses.field(
        metadata={
            "name": "shallow_convection_snow",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    critical_relative_humidity_for_pdf: Quantity = dataclasses.field(
        metadata={
            "name": "critical_relative_humidity_for_pdf",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "1",
            "dtype": Float,
        }
    )
    large_scale_rainwater_source: Quantity | None = dataclasses.field(
        metadata={
            "name": "large_scale_rainwater_source",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    ice_water_path: Quantity | None = dataclasses.field(
        metadata={
            "name": "ice_water_path",
            "dims": [I_DIM, J_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    liquid_water_path: Quantity | None = dataclasses.field(
        metadata={
            "name": "liquid_water_path",
            "dims": [I_DIM, J_DIM],
            "units": "kg kg-1 s-1",
            "dtype": Float,
        }
    )
    boundary_layer_level_for_uw_shallow_conv: Quantity | None = dataclasses.field(
        metadata={
            "name": "boundary_layer_level_for_uw_shallow_conv",
            "dims": [I_DIM, J_DIM],
            "units": "1",
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
                "dtype": Float,
            }
        )
        variance: Quantity = dataclasses.field(
            metadata={
                "name": "variance",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m2 s-2",
                "dtype": Float,
            }
        )
        third_moment: Quantity = dataclasses.field(
            metadata={
                "name": "third_moment",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m3 s-3",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class MixingRatio:
        vapor: Quantity = dataclasses.field(
            metadata={
                "name": "vapor",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        large_scale_liquid: Quantity = dataclasses.field(
            metadata={
                "name": "large_scale_liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        large_scale_ice: Quantity = dataclasses.field(
            metadata={
                "name": "large_scale_ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        convective_liquid: Quantity = dataclasses.field(
            metadata={
                "name": "convective_liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        convective_ice: Quantity = dataclasses.field(
            metadata={
                "name": "convective_ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class CloudFraction:
        large_scale: Quantity = dataclasses.field(
            metadata={
                "name": "large_scale_cloud_fraction",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "1",
                "dtype": Float,
            }
        )
        convective: Quantity = dataclasses.field(
            metadata={
                "name": "convective_cloud_fraction",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class Concentration:
        liquid: Quantity = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m-3",
                "dtype": Float,
            }
        )
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m-3",
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
                "dtype": Float,
            }
        )
        variance: Quantity = dataclasses.field(
            metadata={
                "name": "variance",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "K+2",
                "dtype": Float,
            }
        )
        third_moment: Quantity = dataclasses.field(
            metadata={
                "name": "third_moment",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "K+3",
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
                "dtype": Float,
            }
        )
        variance: Quantity = dataclasses.field(
            metadata={
                "name": "variance",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "1",
                "dtype": Float,
            }
        )
        third_moment: Quantity = dataclasses.field(
            metadata={
                "name": "third_moment",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class RadiationField:
        cloud_fraction: Quantity = dataclasses.field(
            metadata={
                "name": "cloud_fraction",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        vapor: Quantity = dataclasses.field(
            metadata={
                "name": "vapor",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        liquid: Quantity = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "cloud_fraction",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "cloud_fraction",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "cloud_fraction",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class CloudParticleEffectiveRadius:
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "dtype": Float,
            }
        )
        liquid: Quantity = dataclasses.field(
            metadata={
                "name": "liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class PrecipitationAtSurface:
        water: Quantity = dataclasses.field(
            metadata={
                "name": "water",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        shallow_convective_precipitation: Quantity = dataclasses.field(
            metadata={
                "name": "shallow_convective_precipitation",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        deep_convective_precipitation: Quantity = dataclasses.field(
            metadata={
                "name": "deep_convective_precipitation",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        anvil_precipitation: Quantity = dataclasses.field(
            metadata={
                "name": "anvil_precipitation",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        shallow_convective_snow: Quantity = dataclasses.field(
            metadata={
                "name": "shallow_convective_snow",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        deep_convective_snow: Quantity = dataclasses.field(
            metadata={
                "name": "deep_convective_snow",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        anvil_snow: Quantity = dataclasses.field(
            metadata={
                "name": "anvil_snow",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class NonAnvilLargeScale:
        precip: Quantity = dataclasses.field(
            metadata={
                "name": "precip",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        evaporation: Quantity = dataclasses.field(
            metadata={
                "name": "evaporation",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        sublimation: Quantity = dataclasses.field(
            metadata={
                "name": "sublimation",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        graupel_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "graupel_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        ice_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "ice_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        liquid_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "liquid_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        rain_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "rain_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        snow_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "snow_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class Anvil:
        liquid_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "liquid_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        ice_precip_flux: Quantity = dataclasses.field(
            metadata={
                "name": "ice_precip_flux",
                "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class Tendencies:
        dcloud_fractiondt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dsurface_specific_humuditydt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dvapordt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dvapordt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dicedt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dicedt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dliquiddt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dliquiddt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        draindt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "draindt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dgraupeldt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dgraupeldt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dsnowdt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dsnowdt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dudt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dudt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dvdt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dvdt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dtdt_macro: Quantity = dataclasses.field(
            metadata={
                "name": "dtdt_macro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dcloud_fractiondt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dsurface_specific_humuditydt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dvapordt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dvapordt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dicedt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dicedt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dliquiddt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dliquiddt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        draindt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "draindt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dgraupeldt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dgraupeldt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dsnowdt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dsnowdt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dudt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dudt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dvdt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dvdt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dtdt_micro: Quantity = dataclasses.field(
            metadata={
                "name": "dtdt_micro",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1 s-1",
                "dtype": Float,
            }
        )
        dtdt_friction_pressure_weighted: Quantity | None = dataclasses.field(
            metadata={
                "name": "dtdt_friction_pressure_weighted",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "Pa K s-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class Radar:
        simulated_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "name": "simulated_reflectivity",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "dBZ",
                "dtype": Float,
            }
        )
        maximum_composite_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "name": "maximum_composite_reflectivity",
                "dims": [I_DIM, J_DIM],
                "units": "dBZ",
                "dtype": Float,
            }
        )
        base_1km_agl_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "name": "base_1km_agl_reflectivity",
                "dims": [I_DIM, J_DIM],
                "units": "dBZ",
                "dtype": Float,
            }
        )
        echo_top_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "name": "echo_top_reflectivity",
                "dims": [I_DIM, J_DIM],
                "units": "dBZ",
                "dtype": Float,
            }
        )
        minus_10c_reflectivity: Quantity | None = dataclasses.field(
            metadata={
                "name": "minus_10c_reflectivity",
                "dims": [I_DIM, J_DIM],
                "units": "dBZ",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class MassFraction:
        suspended_graupel: Quantity | None = dataclasses.field(
            metadata={
                "name": "suspended_graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        suspended_rain: Quantity | None = dataclasses.field(
            metadata={
                "name": "suspended_rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )
        suspended_snow: Quantity | None = dataclasses.field(
            metadata={
                "name": "suspended_snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg kg-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class FallSpeed:
        graupel: Quantity = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m s-1",
                "dtype": Float,
            }
        )
        ice: Quantity = dataclasses.field(
            metadata={
                "name": "ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m s-1",
                "dtype": Float,
            }
        )
        rain: Quantity = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m s-1",
                "dtype": Float,
            }
        )
        snow: Quantity = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "m s-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class FillNegativeTendencyCloudMacro:

        graupel: Quantity | None = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        convective_ice: Quantity | None = dataclasses.field(
            metadata={
                "name": "convective_ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        large_scale_ice: Quantity | None = dataclasses.field(
            metadata={
                "name": "large_scale_ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        convective_liquid: Quantity | None = dataclasses.field(
            metadata={
                "name": "convective_liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        large_scale_liquid: Quantity | None = dataclasses.field(
            metadata={
                "name": "large_scale_liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        rain: Quantity | None = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        snow: Quantity | None = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        vapor: Quantity | None = dataclasses.field(
            metadata={
                "name": "vapor",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class FillNegativeTendencyCloudMicro:

        graupel: Quantity | None = dataclasses.field(
            metadata={
                "name": "graupel",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        convective_ice: Quantity | None = dataclasses.field(
            metadata={
                "name": "convective_ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        large_scale_ice: Quantity | None = dataclasses.field(
            metadata={
                "name": "large_scale_ice",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        convective_liquid: Quantity | None = dataclasses.field(
            metadata={
                "name": "convective_liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        large_scale_liquid: Quantity | None = dataclasses.field(
            metadata={
                "name": "large_scale_liquid",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        rain: Quantity | None = dataclasses.field(
            metadata={
                "name": "rain",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        snow: Quantity | None = dataclasses.field(
            metadata={
                "name": "snow",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )
        vapor: Quantity | None = dataclasses.field(
            metadata={
                "name": "vapor",
                "dims": [I_DIM, J_DIM, K_DIM],
                "units": "kg m-2 s-1",
                "dtype": Float,
            }
        )

    vertical_motion: VerticalMotion
    mixing_ratio: MixingRatio
    cloud_fraction: CloudFraction
    concentration: Concentration
    liquid_water_static_energy: LiquidWaterStaticEnergy
    total_water: TotalWater
    radiation_field: RadiationField
    cloud_particle_effective_radius: CloudParticleEffectiveRadius
    precipitation_at_surface: PrecipitationAtSurface
    non_anvil_large_scale: NonAnvilLargeScale
    anvil: Anvil
    tendencies: Tendencies
    radar: Radar
    mass_fraction: MassFraction
    fall_speed: FallSpeed
    fill_negative_tendency_cloud_macro: FillNegativeTendencyCloudMacro
    fill_negative_tendency_cloud_micro: FillNegativeTendencyCloudMicro
