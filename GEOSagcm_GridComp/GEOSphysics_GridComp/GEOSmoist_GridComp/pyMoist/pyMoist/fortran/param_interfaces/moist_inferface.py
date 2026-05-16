from typing import Any

from MAPL_PythonBridge import UserCode, get_MAPLPy
from MAPL_PythonBridge.types import CVoidPointer
from mpi4py import MPI
from ndsl.constants import I_DIM, J_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int
from ndsl.utils import safe_assign_array

from pyMoist.constants import NUMBER_OF_TRACERS
from pyMoist.state import MoistState
from pyMoist.moist import Moist
from pyMoist.convection_tracers import ConvectionTracers
from pyMoist.fortran import get_NDSL_physics
from pyMoist.fortran.build_helper import StencilBackendCompilerOverride
from pyMoist.fortran.managed_state import MAPLManagedState
from pyMoist.fortran.memory_factory import MAPLMemoryRepository
from pyMoist.fortran.moist_workarounds import MOIST_WORKAROUNDS
from pyMoist.fortran.profiler import TimedCUDAProfiler
from pyMoist.saturation_tables import SaturationVaporPressureTable


def _default_or_get_from_namelist(default, name_in_namelist: str, namelist: dict[str, Any]) -> Any:
    return default if name_in_namelist not in namelist else namelist[name_in_namelist]


class MoistInterface(UserCode):
    def __init__(self) -> None:
        pass

    def init(self, mapl_state: CVoidPointer, import_state: CVoidPointer, export_state: CVoidPointer):
        maplpy = get_MAPLPy()
        ndsl_stack = get_NDSL_physics(mapl_state)

        # Initialize configuration

        # Initialize the module
        with StencilBackendCompilerOverride(
            MPI.COMM_WORLD,
            ndsl_stack.stencil_factory.config.dace_config,
        ):
            self._mosit = Moist(
                stencil_factory=ndsl_stack.stencil_factory,
                quantity_factory=ndsl_stack.quantity_factory,
            )

        # Make the state
        self._managed_state = MAPLManagedState(
            MoistState.empty(ndsl_stack.quantity_factory),
            ndsl_stack.interface_type,
        )

        self._managed_convection_tracers = MAPLManagedState(
            ConvectionTracers.empty(
                ndsl_stack.quantity_factory,
                data_dimensions={
                    "convection_tracers": config.NUMBER_OF_TRACERS,
                    "size_three_dimension": 3,
                    "size_four_dimension": 4,
                },
            ),
            ndsl_stack.interface_type,
        )

    def run(
        self,
        mapl_state: CVoidPointer,
        import_state: CVoidPointer,
        export_state: CVoidPointer,
    ):
        pass

    def run_with_internal(
        self,
        mapl_state: CVoidPointer,
        import_state: CVoidPointer,
        export_state: CVoidPointer,
        internal_state: CVoidPointer,
    ):
        ndsl_stack = get_NDSL_physics(mapl_state)
        import_repository = MAPLMemoryRepository(import_state, ndsl_stack.quantity_factory)
        internal_repository = MAPLMemoryRepository(internal_state, ndsl_stack.quantity_factory)
        export_repository = MAPLMemoryRepository(export_state, ndsl_stack.quantity_factory)

        # note which components will be run - used to allocate the correct set of fields
        # TODO need to solve this
        run_gfdl1m_microphysics = False
        run_gf2020_convection = False
        run_uw_shallow_convection = False

        # Fill the MoistState with data

        # GridData
        self._managed_state.register("grid_data.area", "AREA", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("grid_data.latitude", "DSL__GF2020_LATS", internal_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("grid_data.longitude", "DSL__GF2020_LONS", internal_repository, dims=[I_DIM, J_DIM])

        # AtmosphericState
        self._managed_state.register("atmospheric_state.t", "T", import_repository)
        self._managed_state.register("atmospheric_state.u", "U", import_repository)
        self._managed_state.register("atmospheric_state.v", "V", import_repository)
        self._managed_state.register("atmospheric_state.omega", "OMEGA", import_repository)
        self._managed_state.register("atmospheric_state.turbulent_kinetic_energy", "TKE", import_repository)
        self._managed_state.register("atmospheric_state.p_interface", "PLE", import_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("atmospheric_state.z_interface", "ZLE", import_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("atmospheric_state.scalar_diffusivity_interface", "KH", import_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("atmospheric_state.reference_pressure", "PREF", import_repository, dims=[K_INTERFACE_DIM])
        self._managed_state.register("atmospheric_state.vertical_motion.velocity", "W", import_repository)
        self._managed_state.register("atmospheric_state.vertical_motion.variance", "W2", import_repository)
        self._managed_state.register("atmospheric_state.vertical_motion.third_moment", "W3", import_repository)

        # SurfaceConditions
        self._managed_state.register("surface_conditions.t_surface", "TS", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.t_surface_air", "TA", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.t_2m", "T2M", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.specific_humidity_surface_air", "QA", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.specific_humidity_2m", "Q2M", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.sensible_heat_flux", "SH", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.surface_evaporation", "EVAP", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.land_fraction", "FRLAND", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.land_ice_fraction", "FRLANDICE", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.ice_covered_fraction_of_tile", "FRACI", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.snow_mass", "SNOMAS", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("surface_conditions.surface_type", "SRF_TYPE", import_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("surface_conditions.geopotential_height", "PHIS", import_repository, dims=[I_DIM, J_DIM])

        # Levels
        self._managed_state.register("levels.pbl_level", "KPBL", import_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("levels.cbl_level_before_moist", "KCBL_moist", import_repository, dims=[I_DIM, J_DIM])

        # CloudCondensates
        self._managed_state.register("cloud_condensates.specific_humidity", "Q", internal_repository)
        self._managed_state.register("cloud_condensates.rain", "QRAIN", internal_repository)
        self._managed_state.register("cloud_condensates.graupel", "QGRAUPEL", internal_repository)
        self._managed_state.register("cloud_condensates.snow", "QSNOW", internal_repository)
        self._managed_state.register("cloud_condensates.convective_cloud_fraction", "CLCN", internal_repository)
        self._managed_state.register("cloud_condensates.convective_ice", "QICN", internal_repository)
        self._managed_state.register("cloud_condensates.convective_liquid", "QLCN", internal_repository)
        self._managed_state.register("cloud_condensates.large_scale_cloud_fraction", "CLLS", internal_repository)
        self._managed_state.register("cloud_condensates.large_scale_ice", "QILS", internal_repository)
        self._managed_state.register("cloud_condensates.large_scale_liquid", "QLLS", internal_repository)
        self._managed_state.register("cloud_condensates.large_scale_ice_cloud_fraction", "CFICE", export_repository, alloc=True)
        self._managed_state.register("cloud_condensates.large_scale_liquid_cloud_fraction", "CFLIQ", export_repository)
        self._managed_state.register("cloud_condensates.total_liquid", "QLTOT", export_repository)
        self._managed_state.register("cloud_condensates.total_ice", "QITOT", export_repository)
        self._managed_state.register("cloud_condensates.total_water", "QCTOT", export_repository)
        self._managed_state.register("cloud_condensates.liquid_concentration", "NACTL", internal_repository)
        self._managed_state.register("cloud_condensates.ice_concentration", "NACTI", internal_repository)
        self._managed_state.register("cloud_condensates.relative_humidity_wrt_ice", "RHICE", internal_repository)
        self._managed_state.register("cloud_condensates.relative_humidity_wrt_liquid", "RHLIQ", internal_repository)
        self._managed_state.register("cloud_condensates.number_concentration_water_friendly_aerosols", "NWFA", internal_repository, alloc=True)
        self._managed_state.register("cloud_condensates.ice_ccn_concentration", "NCCN_ICE", internal_repository)
        self._managed_state.register("cloud_condensates.liquid_ccn_concentration", "NCCN_LIQ", internal_repository)
        self._managed_state.register("cloud_condensates.liquid_water_static_energy.flux", "WSL", import_repository)
        self._managed_state.register("cloud_condensates.liquid_water_static_energy.variance", "SL2", import_repository)
        self._managed_state.register("cloud_condensates.liquid_water_static_energy.third_moment", "SL3", import_repository)
        self._managed_state.register("cloud_condensates.total_water.flux", "WQT", import_repository)
        self._managed_state.register("cloud_condensates.total_water.variance", "QT2", import_repository)
        self._managed_state.register("cloud_condensates.total_water.third_moment", "QT3", import_repository)
        self._managed_state.register("cloud_condensates.ice_particle_effective_radius", "RI", internal_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("cloud_condensates.liquid_particle_effective_radius", "RL", internal_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("cloud_condensates.mass_fraction_suspended_rain", "QRTOT", internal_repository)
        self._managed_state.register("cloud_condensates.mass_fraction_suspended_graupel", "QGTOT", internal_repository)
        self._managed_state.register("cloud_condensates.mass_fraction_suspended_snow", "QSTOT", internal_repository)
        self._managed_state.register("cloud_condensates.ice_fraction_in_convective_tower", "CNV_FICE", export_repository)
        self._managed_state.register("cloud_condensates.convective_rainwater_source", "DQRC", export_repository)
        self._managed_state.register("cloud_condensated.convective_condensate_source", "CNV_DQCDT", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("cloud_condensated.convective_condensate_grid_mean", "CNV_QC", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("cloud_condensated.initial_total_precipitable_water", "TPWI", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("cloud_condensated.initial_total_precipitable_water_saturation", "TPWI_star", export_repository, dims=[I_DIM, J_DIM], alloc=True)

        # Tendencies
        self._managed_state.register("tendencies.du_dt", "DUDT", export_repository)
        self._managed_state.register("tendencies.du_dt_deep_convection", "DUDT_DC", export_repository)
        self._managed_state.register("tendencies.du_dt_shallow_convection", "DUDT_SC", export_repository)
        self._managed_state.register("tendencies.du_dt_macrophysics", "DUDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.du_dt_microphysics", "DUDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dv_dt", "DVDT", export_repository)
        self._managed_state.register("tendencies.dv_dt_deep_convection", "DVDT_DC", export_repository)
        self._managed_state.register("tendencies.dv_dt_shallow_convection", "DVDT_SC", export_repository)
        self._managed_state.register("tendencies.dv_dt_macrophysics", "DVDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dv_dt_microphysics", "DVDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dt_dt", "DTDT", export_repository)
        self._managed_state.register("tendencies.dt_dt_deep_convection", "DTDT_DC", export_repository)
        self._managed_state.register("tendencies.dt_dt_shallow_convection", "DTDT_SC", export_repository)
        self._managed_state.register("tendencies.dt_dt_macrophysics", "DTDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dt_dt_microphysics", "DTDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dt_dt_from_rh_cleanup", "DTDT_ER", export_repository, alloc=True)
        self._managed_state.register("tendencies.dt_dt_friction_pressure_weighted", "DTDTFRIC", export_repository)
        self._managed_state.register("tendencies.dt_dt_shortwave", "RADSW", import_repository)
        self._managed_state.register("tendencies.dt_dt_longwave", "RADLW", import_repository)
        self._managed_state.register("tendencies.dt_dt_pbl", "DTDT_BL", import_repository)
        self._managed_state.register("tendencies.dt_dt_from_dynamics", "DTDTDYN", import_repository)
        self._managed_state.register("tendencies.dspecific_humidity_dt", "DQDT", export_repository)
        self._managed_state.register("tendencies.dspecific_humidity_dt_deep_convection", "DQVDT_DC", export_repository)
        self._managed_state.register("tendencies.dspecific_humidity_dt_shallow_convection", "DQVDT_SC", export_repository)
        self._managed_state.register("tendencies.dspecific_humidity_dt_macrophysics", "DQVDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dspecific_humidity_dt_microphysics", "DQVDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dspecific_humidity_dt_from_rh_cleanup", "DQVDT_ER", export_repository, alloc=True)
        self._managed_state.register("tendencies.dspecific_humidity_dt_pbl", "DQDT_BL", import_repository)
        self._managed_state.register("tendencies.dspecific_humidity_dt_from_dynamics", "DQVDTDYN", import_repository)
        self._managed_state.register("tendencies.dliquid_dt", "DQLDT", export_repository)
        self._managed_state.register("tendencies.dliquid_dt_deep_convection", "DQLDT_DC", export_repository)
        self._managed_state.register("tendencies.dliquid_dt_shallow_convection", "DQLDT_SC", export_repository)
        self._managed_state.register("tendencies.dliquid_dt_macrophysics", "DQLDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dliquid_dt_microphysics", "DQLDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dice_dt", "DQIDT", export_repository)
        self._managed_state.register("tendencies.dice_dt_deep_convection", "DQIDT_DC", export_repository)
        self._managed_state.register("tendencies.dice_dt_shallow_convection", "DQIDT_SC", export_repository)
        self._managed_state.register("tendencies.dice_dt_macrophysics", "DQIDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dice_dt_microphysics", "DQIDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.drain_dt", "DQRDT", export_repository)
        self._managed_state.register("tendencies.drain_dt_shallow_convection", "DQRDT_SC", export_repository)
        self._managed_state.register("tendencies.drain_dt_macrophysics", "DQRDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.drain_dt_microphysics", "DQRDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dsnow_dt", "DQSDT", export_repository)
        self._managed_state.register("tendencies.dsnow_dt_shallow_convection", "DQSDT_SC", export_repository)
        self._managed_state.register("tendencies.dsnow_dt_macrophysics", "DQSDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dsnow_dt_microphysics", "DQSDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dgraupel_dt", "DQGDT", export_repository)
        self._managed_state.register("tendencies.dgraupel_dt_macrophysics", "DQGDT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dgraupel_dt_microphysics", "DQGDT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dtotal_cloud_fraciton_dt", "DQADT", export_repository)
        self._managed_state.register("tendencies.dtotal_cloud_fraciton_dt_deep_convection", "DQADT_DC", export_repository)
        self._managed_state.register("tendencies.dtotal_cloud_fraciton_dt_shallow_convection", "DQADT_SC", export_repository)
        self._managed_state.register("tendencies.dtotal_cloud_fraciton_dt_macrophysics", "DQADT_macro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dtotal_cloud_fraciton_dt_microphysics", "DQADT_micro", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("tendencies.dlayer_pressure_thickness_dt", "DPDTMST", export_repository)

        # PrecipitationAtSurface
        self._managed_state.register("precipitation_at_surface.total_precipitation", "TPREC", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.precipitation_total", "PRECTOTAL", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("precipitation_at_surface.rainfall", "RAIN", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register(
            "precipitation_at_surface.spurious_rain_from_relative_humidity_cleanup", "ER_PRCP", export_repository, dims=[I_DIM, J_DIM], alloc=True
        )
        self._managed_state.register("precipitation_at_surface.total_convective_precipitation", "PREC_CONV", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.rain_from_all_convection", "PCU", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("precipitation_at_surface.rain_from_deep_convection", "CN_PRCP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.rain_from_shallow_convection", "SC_PRCP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.rain_from_GF_convection", "CNPCPRATE", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("precipitation_at_surface.total_stratiform_precipitation", "PREC_STRAT", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.rain_from_all_large_scale", "PLS", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("precipitation_at_surface.rain_from_large_scale_nonanvil", "LS_PRCP", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.rain_from_large_scale_anvil", "AN_PRCP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.snowfall", "SNO", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.snowfall_total", "SNOWTOTAL", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("precipitation_at_surface.large_scale_snow", "LS_SNR", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.anvil_snow", "AN_SNR", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.deep_convection_snow", "CN_SNR", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.shallow_convection_snow", "SC_SNR", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.icefall", "ICE", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.freezing_rainfall", "FRZR", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.surface_precipitation_type", "PTYPE", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("precipitation_at_surface.falling_graupel", "PRCP_GRAUPEL", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.falling_ice", "PRCP_ICE", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.falling_rain", "PRCP_RAIN", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_at_surface.falling_snow", "PRCP_SNOW", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)

        # PrecipitationFlux
        self._managed_state.register("precipitation_flux.ice_convection", "PFI_CN", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("precipitation_flux.ice_shallow_convection", "PFI_SC", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("precipitation_flux.ice_anvil", "PFI_AN", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_flux.ice_nonanvil_large_scale", "PFI_LS", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_flux.ice_nonconvective", "PFI_LSAN", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("precipitation_flux.liquid_convection", "PFL_CN", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("precipitation_flux.liquid_shallow_convection", "PFL_SC", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("precipitation_flux.liquid_anvil", "PFL_AN", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_flux.liquid_nonanvil_large_scale", "PFL_LS", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("precipitation_flux.liquid_nonconvective", "PFL_LSAN", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("precipitation_flux.shallow_convective_rain", "SHLW_PRC3", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics or run_uw_shallow_convection)
        self._managed_state.register("precipitation_flux.shallow_convective_snow", "SHLW_SNO3", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics or run_uw_shallow_convection)
        self._managed_state.register("precipitation_flux.convective_precipitation_from_RAS", "CNV_PRC3", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("precipitation_flux.total_water_flux_deep_convection_interface", "WQT_DC", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], alloc=run_gf2020_convection)

        # RadiationState
        self._managed_state.register("radiation_state.specific_humidity" "FCLD", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("radiation_state.liquid", "QV", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("radiation_state.ice", "QL", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("radiation_state.rain", "QI", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("radiation_state.graupel", "QR", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("radiation_state.snow", "QS", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("radiation_state.cloud_fraction", "QG", export_repository, alloc=run_gfdl1m_microphysics)

        # StateBeforeDynamics
        self._managed_state.register("state_before_dynamics.p_interface", "PLE_DYN_IN", import_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM])
        self._managed_state.register("state_before_dynamics.t", "T_DYN_IN", import_repository)
        self._managed_state.register("state_before_dynamics.u", "U_DYN_IN", import_repository)
        self._managed_state.register("state_before_dynamics.v", "V_DYN_IN", import_repository)
        self._managed_state.register("state_before_dynamics.specific_humidity", "QV_DYN_IN", import_repository)

        # StateAtInput
        self._managed_state.register("state_at_input.pt", "TH_moist", export_repository)
        self._managed_state.register("state_at_input.t_surface", "TS_moist", export_repository)
        self._managed_state.register("state_at_input.u", "UMST0", export_repository, alloc=True)
        self._managed_state.register("state_at_input.v", "VMST0", export_repository, alloc=True)
        self._managed_state.register("state_at_input.specific_humidity", "QX0", export_repository)
        self._managed_state.register("state_at_input.specific_humidity", "Q_moist", export_repository)
        self._managed_state.register("state_at_input.convective_cloud_fraction", "CLCNX0", export_repository)
        self._managed_state.register("state_at_input.convective_condensates", "QCCNX0", export_repository)
        self._managed_state.register("state_at_input.convective_ice", "QICNX0", export_repository)
        self._managed_state.register("state_at_input.convective_liquid", "QLCNX0", export_repository)
        self._managed_state.register("state_at_input.large_scale_cloud_fraction", "CLLSX0", export_repository)
        self._managed_state.register("state_at_input.large_scale_condensates", "QCLSX0", export_repository)
        self._managed_state.register("state_at_input.large_scale_ice", "QILSX0", export_repository)
        self._managed_state.register("state_at_input.large_scale_liquid", "QLLSX0", export_repository)

        # StateAtOutput
        self._managed_state.register("state_at_output.t", "TAFMOIST", export_repository, alloc=True)
        self._managed_state.register("state_at_output.pt", "THAFMOIST", export_repository, alloc=True)
        self._managed_state.register("state_at_output.u", "UAFMOIST", export_repository, alloc=True)
        self._managed_state.register("state_at_output.v", "VAFMOIST", export_repository, alloc=True)
        self._managed_state.register("state_at_output.specific_humidity", "QAFMOIST", export_repository, alloc=True)
        self._managed_state.register("state_at_output.relative_humidity", "RH2", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("state_at_output.saturation_ratio", "SAT_RAT", export_repository)
        self._managed_state.register("state_at_output.dry_static_energy", "SAFMOIST", export_repository, alloc=True)
        self._managed_state.register("state_at_output.convective_ice", "QICNX1", export_repository)
        self._managed_state.register("state_at_output.convective_liquid", "QLCNX1", export_repository)
        self._managed_state.register("state_at_output.large_scale_ice", "QILSX1", export_repository)
        self._managed_state.register("state_at_output.large_scale_liquid", "QLLSX1", export_repository)

        # ConvectiveDiagnostics
        self._managed_state.register("convective_diagnostics.convection_fraction", "CNV_FRC", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.cape_surface_parcel", "CAPE", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.cin_surface_parcel", "INHB", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.buoyancy_surface_parcel", "BYNCY", export_repository, alloc=True)
        self._managed_state.register("convective_diagnostics.sbcape", "SBCAPE", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.sbcin", "SBCIN", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.mlcape", "MLCAPE", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.mlcin", "MLCIN", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.mucape", "MUCAPE", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.mucin", "MUCIN", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.lfc_level", "ZLFC", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.lnb_level", "ZLNB", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.total_cumulative_mass_flux", "CNV_MFC", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.total_detraining_mass_flux", "CNV_MFD", export_repository, dims=[I_DIM, J_DIM], alloc=True)
        self._managed_state.register("convective_diagnostics.stochastic_factor", "STOCH_CNV", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("convective_diagnostics.convective_precipitation_evaporation", "REV_CN", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.convective_precipitation_sublimation", "RSU_CN", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.entrainment_parameter", "ENTLAM", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.lateral_entrainment_rate", "ENTR", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.lateral_entrainment_rate_shallow", "ENTR_SH", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.lateral_entrainment_rate_mid", "ENTR_MD", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.lateral_entrainment_rate_deep", "ENTR_DP", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.updraft_areal_fraction", "CNV_UPDF", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.updraft_vertical_velocity", "CNV_CVW", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.sigma_mid", "SIGMA_MID", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.sigma_deep", "SIGMA_DEEP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.pressure_shallow_convective_cloud_top", "CNV_TOPP_SH", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.pressure_mid_convective_cloud_top", "CNV_TOPP_MD", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.pressure_deep_convective_cloud_top", "CNV_TOPP_DP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_shallow", "MUPSH", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_mid", "MUPMD", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_deep_updraft", "MUPDP", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_deep_updraft_interface", "UMF_DC", export_repository, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_deep_updraft_detrained", "MFD_DC", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_deep_downdraft", "MDNDP", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_cloud_base", "CNV_MF0", export_repository, alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_cloud_base_shallow", "MFSH", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_cloud_base_mid", "MFMD", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.mass_flux_cloud_base_deep", "MFDP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.convection_code_shallow", "ERRSH", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.convection_code_mid", "ERRMD", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.convection_code_deep", "ERRDP", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cloud_workfunction_0", "AA0", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cloud_workfunction_1", "AA1", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cloud_workfunction_2", "AA2", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cloud_workfunction_3", "AA3", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cloud_workfunction_1_pbl", "AA1_BL", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cloud_workfunction_1_cin", "AA1_CIN", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.pbl_time_scale", "TAU_BL", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.cape_removal_time_scale", "TAU_EC", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.lightning_density", "LFR_GF", export_repository, dims=[I_DIM, J_DIM], alloc=run_gf2020_convection)
        self._managed_state.register("convective_diagnostics.convection_tracer", "CNV_TR", internal_repository)

        # MicrophysicsDiagnostics
        self._managed_state.register("microphysics_diagnostics.pdf_first_plume_fractional_area", "PDF_A", import_repository)
        self._managed_state.register("microphysics_diagnostics.covariance_liquid_water_static_energy_and_total_water_specific_humidity", "SLQT", import_repository)
        self._managed_state.register("microphysics_diagnostics.cloud_liquid_evaporation", "EVAPC", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.cloud_ice_evaporation", "SUBLC", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.relative_humidity_after_pdf", "RHX", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.buoyancy_flux", "WTHV2", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.liquid_water_flux", "WQL", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.hydrostatic_pdf_iterations", "PDFITERS", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.critical_relative_humidity_for_pdf", "RHCRIT", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.large_scale_rainwater_source", "DQRL", export_repository)
        self._managed_state.register("microphysics_diagnostics.nonanvil_large_scale_precipitation_evaporation", "REV_LS", export_repository, alloc=run_gfdl1m_microphysics)
        self._managed_state.register("microphysics_diagnostics.nonanvil_large_scale_precipitation_sublimation", "RSU_LS", export_repository, alloc=run_gfdl1m_microphysics)

        # Diagnostics
        self._managed_state.register("diagnostics.negative_vapor_adjustment_start", "FILLNQV_IN", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.negative_vapor_adjustment_end", "FILLNQV", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.highest_level_of_scalar_diffusivity_gt_2", "KHu_moist", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.lowest_level_of_scalar_diffusivity_gt_2", "KHl_moist", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.condensed_water_path", "CWP", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.cloud_liquid_water_path", "CLWP", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.liquid_water_path", "LWP", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.ice_water_path", "IWP", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.total_precipitable_water", "TPW", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.lightning_flash_rate", "LFR_GCC", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.kuchera_snow_to_liquid_ratio", "KUCHERA_RATIO", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.lower_tropospheric_stability", "LTS", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("diagnostics.estimated_inversion_strength", "EIS", export_repository, dims=[I_DIM, J_DIM], alloc=run_gfdl1m_microphysics)
        self._managed_state.register("diagnostics.lcl_height", "ZLCL", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.radar.simulated_reflectivity", "DBZ", export_repository)
        self._managed_state.register("diagnostics.radar.maximum_composite_reflectivity", "DBZ_MAX", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.radar.base_1km_agl_reflectivity", "DBZ_1KM", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.radar.echo_top_reflectivity", "DBZ_TOP", export_repository, dims=[I_DIM, J_DIM])
        self._managed_state.register("diagnostics.radar.minus_10c_reflectivity", "DBZ_M10C", export_repository, dims=[I_DIM, J_DIM])

        if self._moist is None:
            raise RuntimeError("Moist Runtime called before initialization was done. Abort.")

        with TimedCUDAProfiler("Moist Physics", {}):
            with TimedCUDAProfiler("Moist Physics - State copy", {}):
                self._managed_state.fortran_to_ndsl()

            with TimedCUDAProfiler("Moist Physics Numerics", {}):
                self._moist()

            with TimedCUDAProfiler("Moist Physics - State copy-back", {}):
                self._managed_state.ndsl_to_fortran()

    def finalize(
        self,
        mapl_state: CVoidPointer,
        import_state: CVoidPointer,
        export_state: CVoidPointer,
    ):
        self._managed_state.save_recorded()


CODE = MoistInterface()
