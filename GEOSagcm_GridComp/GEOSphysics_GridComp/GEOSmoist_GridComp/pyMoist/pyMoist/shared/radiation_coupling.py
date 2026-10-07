from ndsl.dsl.gt4py import computation, FORWARD, PARALLEL, interval
from pyMoist.constants import MAPL_UNDEF
from ndsl.dsl.typing import FloatField
from pyMoist.shared.constants import CFMIN, QCMIN, QPMIN
from pyMoist.shared.cloud_processes import cloud_effective_radius_ice, cloud_effective_radius_liquid


def radiation_coupling_scale_aware(
    t: FloatField,
    p_mb: FloatField,
    cloud_particle_effective_radius_liquid: FloatField,
    cloud_particle_effective_radius_ice: FloatField,
    concentration_liquid: FloatField,
    radiation_cloud_fraction: FloatField,
    radiation_graupel: FloatField,
    radiation_ice: FloatField,
    radiation_liquid: FloatField,
    radiation_rain: FloatField,
    radiation_snow: FloatField,
    radiation_vapor: FloatField,
    anvil_cloud_fraction: FloatField,
    anvil_ice: FloatField,
    anvil_liquid: FloatField,
    large_scale_cloud_fraction: FloatField,
    large_scale_ice: FloatField,
    large_scale_liquid: FloatField,
    graupel: FloatField,
    rain: FloatField,
    snow: FloatField,
    vapor: FloatField,
):
    from __externals__ import MIN_RL, MAX_RL, FAC_RL, MIN_RI, MAX_RI, FAC_RI

    with computation(PARALLEL), interval(...):
        # 1. Pass through water vapor
        radiation_vapor = vapor

        # 2. Combine inputs for Total Cloud Fraction and Cloud Condensates
        total_liquid = large_scale_liquid + anvil_liquid
        total_ice = large_scale_ice + anvil_ice
        radiation_cloud_fraction = max(min(large_scale_cloud_fraction + anvil_cloud_fraction, 1.0), 0.0)

        # 3. Process clouds if there is fraction OR resolved condensate
        if radiation_cloud_fraction >= CFMIN or (total_liquid + total_ice) >= QCMIN:
            # Calculate in-cloud specific humidities and cap at 0.01 max
            if total_liquid >= QCMIN:
                radiation_liquid = min(total_liquid / radiation_cloud_fraction, 0.01)
            else:
                radiation_liquid = 0.0

            if total_ice >= QCMIN:
                radiation_ice = min(total_ice / radiation_cloud_fraction, 0.01)
            else:
                radiation_ice = 0.0

            if rain >= QPMIN:
                radiation_rain = min(rain / radiation_cloud_fraction, 0.01)
            else:
                radiation_rain = 0.0

            if snow >= QPMIN:
                radiation_snow = min(snow / radiation_cloud_fraction, 0.01)
            else:
                radiation_snow = 0.0

            if graupel >= QPMIN:
                radiation_graupel = min(graupel / radiation_cloud_fraction, 0.01)
            else:
                radiation_graupel = 0.0
        else:
            # Clear sky
            radiation_cloud_fraction = 0.0
            radiation_liquid = 0.0
            radiation_ice = 0.0
            radiation_rain = 0.0
            radiation_snow = 0.0
            radiation_graupel = 0.0

        # 4. LIQUID RADII (BRAMS formulation with limits)
        if radiation_liquid > 0.0:
            cloud_particle_effective_radius_liquid = max(MIN_RL, min(cloud_effective_radius_liquid(p_mb, t, radiation_liquid, concentration_liquid) * FAC_RL, MAX_RL))
        else:
            cloud_particle_effective_radius_liquid = MAPL_UNDEF

        # 5. ICE RADII (BRAMS formulation with limits)
        if radiation_ice > 0.0:
            cloud_particle_effective_radius_ice = max(MIN_RI, min(cloud_effective_radius_ice(p_mb, t, radiation_ice) * FAC_RI, MAX_RI))
        else:
            radiation_ice = MAPL_UNDEF
