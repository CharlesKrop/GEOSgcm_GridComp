import os

from MAPL_PythonBridge import UserCode, get_MAPLPy
from MAPL_PythonBridge.types import CVoidPointer
from mpi4py import MPI
from ndsl.dsl.typing import Float, Int

from pyMoist.fortran import get_NDSL_physics
from pyMoist.fortran.build_helper import StencilBackendCompilerOverride
from pyMoist.fortran.cuda_profiler import TimedCUDAProfiler
from pyMoist.fortran.managed_state import MAPLManagedState
from pyMoist.fortran.memory_factory import MAPLMemoryRepository
from pyMoist.microphysics.GFDL_1M import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.shared.constants import V12_ICE_POLYNOMIAL, USE_AEROSOL_NN


class GFDL1MInterface(UserCode):
    def __init__(self) -> None:
        self._gfdl_1m = None

    def init(self, mapl_state: CVoidPointer, import_state: CVoidPointer, export_state: CVoidPointer):
        maplpy = get_MAPLPy()
        ndsl_stack = get_NDSL_physics(mapl_state)
        lid = maplpy.get_resource("GFDL_MP_KLID:", mapl_state, default=Float(-999.0))

        config = GFDL1MConfig(
            ANV_ICEFALL=maplpy.get_resource("ANV_ICEFALL:", mapl_state, default=Float(1.0)),
            CCI_EVAP_EFF=maplpy.get_resource("CCI_EVAP_EFF:", mapl_state, default=Float(4.0e-3)),
            CCW_EVAP_EFF=maplpy.get_resource("CCW_EVAP_EFF:", mapl_state, default=Float(4.0e-3)),
            CNV_FRACTION_MAX=maplpy.get_resource("CNV_FRACTION_MAX:", mapl_state, default=Float(2500.0)),
            CNV_FRACTION_MIN=maplpy.get_resource("CNV_FRACTION_MIN:", mapl_state, default=Float(500.0)),
            CWD=os.getcwd(),  # this is a placeholder. need to pass the true path from fortran to python
            DO_GFDL_REFLECTIVITY=maplpy.get_resource("DO_GFDL_REFLECTIVITY:", mapl_state, default=True),
            DT=maplpy.get_resource("DSL__GFLD1M_DT", mapl_state, default=Float(0.0)),
            FAC_RI=maplpy.get_resource("FAC_RI:", mapl_state, default=Float(1.0)),
            FAC_RL=maplpy.get_resource("FAC_RL:", mapl_state, default=Float(1.0)),
            GFDL_MP3=maplpy.get_resource("GFDL_MP3:", mapl_state, default=True),
            GFDL_MP_KLID=lid - Int(1) if lid > 0 else Int(0),
            ICE_CNV_VFALL_PARAM=maplpy.get_resource("ICE_CNV_VFALL_PARAM:", mapl_state, default=Int(1)),
            ICE_FRACTION_POLYNOMIAL=maplpy.get_resource("ICE_FRACTION_POLYNOMIAL:", mapl_state, default=V12_ICE_POLYNOMIAL),
            ICE_LSC_VFALL_PARAM=maplpy.get_resource("ICE_LSC_VFALL_PARAM:", mapl_state, default=Int(1)),
            ICE_RADII_PARAM=maplpy.get_resource("ICE_RADII_PARAM:", mapl_state, default=Int(2)),
            LIQ_RADII_PARAM=maplpy.get_resource("LIQ_RADII_PARAM:", mapl_state, default=Int(1)),
            LS_ICEFALL=maplpy.get_resource("LS_ICEFALL:", mapl_state, default=Float(1.0)),
            MAX_RH_CRIT=maplpy.get_resource("MAX_RH_CRIT:", mapl_state, default=Float(1.0)),
            MAX_RI=maplpy.get_resource("MAX_RI:", mapl_state, default=Float(100.0e-6)),
            MAX_RL=maplpy.get_resource("MAX_RL:", mapl_state, default=Float(60.0e-6)),
            MELTFRZ_CLDMACRO=maplpy.get_resource("MELTFRZ_CLDMACRO:", mapl_state, default=True),
            MELTFRZ_CLDMICRO=maplpy.get_resource("MELTFRZ_CLDMICRO:", mapl_state, default=False),
            MIN_RH_STABLE=maplpy.get_resource("MIN_RH_STABLE:", mapl_state, default=Float(0.875)),
            MIN_RH_UNSTABLE=maplpy.get_resource("MIN_RH_UNSTABLE:", mapl_state, default=Float(0.975)),
            MIN_RI=maplpy.get_resource("MIN_RI:", mapl_state, default=Float(5.0e-6)),
            MIN_RL=maplpy.get_resource("MIN_RL:", mapl_state, default=Float(2.5e-6)),
            PDFSHAPE=maplpy.get_resource("PDFSHAPE:", mapl_state, default=Int(1)),
            PHYS_HYDROSTATIC=maplpy.get_resource("PHYS_HYDROSTATIC:", mapl_state, default=True),
            REPORT_GFDL_1M_NEGATIVES=maplpy.get_resource("REPORT_GFDL_1M_NEGATIVES:", mapl_state, default=False),
            SH_MD_DP=maplpy.get_resource("SH_MD_DP:", mapl_state, default=True),
            TURNRHCRIT=maplpy.get_resource("TURNRHCRIT:", mapl_state, default=Float(-9999.0)),
            USE_AEROSOL_NN=maplpy.get_resource("USE_AEROSOL_NN:", mapl_state, default=USE_AEROSOL_NN),
            USE_BERGERON=maplpy.get_resource("USE_BERGERON:", mapl_state, default=USE_AEROSOL_NN),
        )

        # Initialize the module
        with StencilBackendCompilerOverride(
            MPI.COMM_WORLD,
            ndsl_stack.stencil_factory.config.dace_config,
        ):
            self._gfdl_1m = GFDL1M(ndsl_stack.stencil_factory, ndsl_stack.quantity_factory, config)

        # Make the state
        self._managed_state = MAPLManagedState(
            GFDL1MState.empty(ndsl_stack.quantity_factory),
            ndsl_stack.interface_type,
        )

        self._first_run = True

    def _get_code(self) -> GFDL1M:
        if not self._gfdl_1m:
            raise RuntimeError("GFDL1M Runtime called before initialization was finished. Abort.")
        return self._gfdl_1m  # type: ignore[unreachable]

    def _first_run_init(
        self,
        mapl_state: CVoidPointer,
        import_state: CVoidPointer,
        export_state: CVoidPointer,
        internal_state: CVoidPointer,
    ):
        if not self._first_run:
            return
        self._first_run = False

        ndsl_stack = get_NDSL_physics(mapl_state)
        import_repository = MAPLMemoryRepository(import_state, ndsl_stack.quantity_factory)
        internal_repository = MAPLMemoryRepository(internal_state, ndsl_stack.quantity_factory)
        export_repository = MAPLMemoryRepository(export_state, ndsl_stack.quantity_factory)

        # INTERNAL STATE
        self._managed_state.register("mixing_ratio.vapor", "Q", internal_repository)
        self._managed_state.register("mixing_ratio.rain", "QRAIN", internal_repository)
        self._managed_state.register("mixing_ratio.snow", "QSNOW", internal_repository)
        self._managed_state.register("mixing_ratio.graupel", "QGRAUPEL", internal_repository)
        self._managed_state.register("mixing_ratio.large_scale_liquid", "QLLS", internal_repository)
        self._managed_state.register("mixing_ratio.large_scale_ice", "QILS", internal_repository)
        self._managed_state.register("mixing_ratio.convective_liquid", "QLCN", internal_repository)
        self._managed_state.register("mixing_ratio.convective_ice", "QICN", internal_repository)
        self._managed_state.register("cloud_fraction.large_scale", "CLLS", internal_repository)
        self._managed_state.register("cloud_fraction.convective", "CLCN", internal_repository)
        self._managed_state.register("concentration.liquid", "NACTL", internal_repository)
        self._managed_state.register("concentration.ice", "NACTI", internal_repository)

        # IMPORT STATE
        self._managed_state.register_2D("area", "AREA", import_repository)
        self._managed_state.register_K_interface("p_interface", "PLE", import_repository)
        self._managed_state.register_K_interface("z_interface", "ZLE", import_repository)
        self._managed_state.register("t", "T", import_repository)
        self._managed_state.register("u", "U", import_repository)
        self._managed_state.register("v", "V", import_repository)
        self._managed_state.register_2D("land_fraction", "FRLAND", import_repository)
        self._managed_state.register_K_interface("scalar_diffusivity_interface", "KH", import_repository)
        self._managed_state.register("covariance_liquid_water_static_energy_and_total_water_specific_humidity", "SLQT", import_repository)
        self._managed_state.register_2D("surface_temperature", "TS", import_repository)
        self._managed_state.register_2D("sensible_heat_flux", "SH", import_repository)
        self._managed_state.register("omega", "OMEGA", import_repository)
        self._managed_state.register("pdf_first_plume_fractional_area", "PDF_A", import_repository)
        self._managed_state.register_2D("evaporation", "EVAP", import_repository)
        self._managed_state.register_2D("surface_geopotential_height", "PHIS", import_repository)
        self._managed_state.register_2D("boundary_layer_level_for_uw_shallow_conv", "KPBL_SC", import_repository)

        self._managed_state.register("vertical_motion.velocity", "W", import_repository)
        self._managed_state.register("vertical_motion.variance", "W2", import_repository)
        self._managed_state.register("vertical_motion.third_moment", "W3", import_repository)

        self._managed_state.register("liquid_water_static_energy.flux", "WSL", import_repository)
        self._managed_state.register("liquid_water_static_energy.variance", "SL2", import_repository)
        self._managed_state.register("liquid_water_static_energy.third_moment", "SL3", import_repository)

        self._managed_state.register("total_water.flux", "WQT", import_repository)
        self._managed_state.register("total_water.variance", "QT2", import_repository)
        self._managed_state.register("total_water.third_moment", "QT3", import_repository)

        # EXPORT STATE
        self._managed_state.register_2D("convection_fraction", "CNV_FRC", export_repository, alloc=True)
        self._managed_state.register_2D("surface_type", "SRF_TYPE", export_repository, alloc=True)
        self._managed_state.register("cloud_liquid_evaporation", "EVAPC", export_repository, alloc=True)
        self._managed_state.register("cloud_ice_sublimation", "SUBLC", export_repository, alloc=True)
        self._managed_state.register_2D("icefall", "ICE", export_repository, alloc=True)
        self._managed_state.register_2D("freezing_rainfall", "FRZR", export_repository, alloc=True)
        self._managed_state.register("relative_humidity_after_pdf", "RHX", export_repository, alloc=True)
        self._managed_state.register("buoyancy_flux", "WTHV2", export_repository, alloc=True)
        self._managed_state.register("liquid_water_flux", "WQL", export_repository, alloc=True)
        self._managed_state.register("hydrostatic_pdf_iterations", "PDFITERS", export_repository, alloc=True)
        self._managed_state.register_2D("lower_tropospheric_stability", "LTS", export_repository, alloc=True)
        self._managed_state.register_2D("estimated_inversion_strength", "EIS", export_repository, alloc=True)
        self._managed_state.register("shallow_convection_rain", "SHLW_PRC3", export_repository, alloc=True)
        self._managed_state.register("shallow_convection_snow", "SHLW_SNO3", export_repository, alloc=True)
        self._managed_state.register("critical_relative_humidity_for_pdf", "RHCRIT", export_repository, alloc=True)
        self._managed_state.register("large_scale_rainwater_source", "DQRL", export_repository)
        self._managed_state.register_2D("ice_water_path", "IWP", export_repository)
        self._managed_state.register_2D("liquid_water_path", "LWP", export_repository)

        self._managed_state.register("radiation_field.cloud_fraction", "FCLD", export_repository, alloc=True)
        self._managed_state.register("radiation_field.vapor", "QV", export_repository, alloc=True)
        self._managed_state.register("radiation_field.liquid", "QL", export_repository, alloc=True)
        self._managed_state.register("radiation_field.ice", "QI", export_repository, alloc=True)
        self._managed_state.register("radiation_field.rain", "QR", export_repository, alloc=True)
        self._managed_state.register("radiation_field.snow", "QS", export_repository, alloc=True)
        self._managed_state.register("radiation_field.graupel", "QG", export_repository, alloc=True)

        self._managed_state.register("cloud_particle_effective_radius.liquid", "RL", export_repository, alloc=True)
        self._managed_state.register("cloud_particle_effective_radius.ice", "RI", export_repository, alloc=True)

        self._managed_state.register_2D("precipitation_at_surface.water", "PRCP_WATER", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.rain", "PRCP_RAIN", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.snow", "PRCP_SNOW", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.ice", "PRCP_ICE", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.graupel", "PRCP_GRAUPEL", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.shallow_convective_precipitation", "SC_PRCP", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.deep_convective_precipitation", "CN_PRCP", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.anvil_precipitation", "AN_PRCP", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.shallow_convective_snow", "SC_SNR", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.deep_convective_snow", "CN_SNR", export_repository, alloc=True)
        self._managed_state.register_2D("precipitation_at_surface.anvil_snow", "AN_SNR", export_repository, alloc=True)

        self._managed_state.register_2D("non_anvil_large_scale.precip", "LS_PRCP", export_repository, alloc=True)
        self._managed_state.register_2D("non_anvil_large_scale.snow", "LS_SNR", export_repository, alloc=True)
        self._managed_state.register("non_anvil_large_scale.evaporation", "REV_LS", export_repository, alloc=True)
        self._managed_state.register("non_anvil_large_scale.sublimation", "RSU_LS", export_repository, alloc=True)
        self._managed_state.register_K_interface("non_anvil_large_scale.ice_precip_flux", "PFI_LS", export_repository, alloc=True)
        self._managed_state.register_K_interface("non_anvil_large_scale.liquid_precip_flux", "PFL_LS", export_repository, alloc=True)
        self._managed_state.register_K_interface("non_anvil_large_scale.rain_precip_flux", "PFR_LS", export_repository, alloc=True)
        self._managed_state.register_K_interface("non_anvil_large_scale.snow_precip_flux", "PFS_LS", export_repository, alloc=True)
        self._managed_state.register_K_interface("non_anvil_large_scale.graupel_precip_flux", "PFG_LS", export_repository, alloc=True)

        self._managed_state.register_K_interface("anvil.liquid_precip_flux", "PFL_AN", export_repository, alloc=True)
        self._managed_state.register_K_interface("anvil.ice_precip_flux", "PFI_AN", export_repository, alloc=True)

        self._managed_state.register("tendencies.dcloud_fraction_dt_macro", "DQADT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dvapor_dt_macro", "DQVDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dice_dt_macro", "DQIDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dliquid_dt_macro", "DQLDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.drain_dt_macro", "DQRDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dgraupel_dt_macro", "DQGDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dsnow_dt_macro", "DQSDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.du_dt_macro", "DUDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dv_dt_macro", "DVDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dt_dt_macro", "DTDT_macro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dcloud_fraction_dt_micro", "DQADT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dvapor_dt_micro", "DQVDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dice_dt_micro", "DQIDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dliquid_dt_micro", "DQLDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.drain_dt_micro", "DQRDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dgraupel_dt_micro", "DQGDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dsnow_dt_micro", "DQSDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.du_dt_micro", "DUDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dv_dt_micro", "DVDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dt_dt_micro", "DTDT_micro", export_repository, alloc=True)
        self._managed_state.register("tendencies.dtdt_friction_pressure_weighted", "DTDTFRIC", export_repository)

        self._managed_state.register("radar.simulated_reflectivity", "REF_DBZ", export_repository)
        self._managed_state.register_2D("radar.maximum_composite_reflectivity", "REF_DBZ_MAX", export_repository)
        self._managed_state.register_2D("radar.base_1km_agl_reflectivity", "REF_DBZ_1KM", export_repository)
        self._managed_state.register_2D("radar.echo_top_reflectivity", "REF_DBZ_TOP", export_repository)
        self._managed_state.register_2D("radar.minus_10c_reflectivity", "REF_DBZ_M10C", export_repository)

        self._managed_state.register("mass_fraction.suspended_graupel", "QGTOT", export_repository)
        self._managed_state.register("mass_fraction.suspended_rain", "QRTOT", export_repository)
        self._managed_state.register("mass_fraction.suspended_snow", "QSTOT", export_repository)

        self._managed_state.register("fall_speed.graupel", "VFALL_GRAUPEL", export_repository)
        self._managed_state.register("fall_speed.ice", "VFALL_ICE", export_repository)
        self._managed_state.register("fall_speed.rain", "VFALL_RAIN", export_repository)
        self._managed_state.register("fall_speed.snow", "VFALL_SNOW", export_repository)

        self._managed_state.register("fill_negative_tendency_cloud_macro.graupel", "DQGDT_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.convective_ice", "DQICNDT_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.large_scale_ice", "DQILSSDT_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.convective_liquid", "DQLCNDT_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.large_scale_liquid", "DQLLSDT_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.rain", "DQR_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.snow", "DQS_FILL_CLDMACRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_macro.vapor", "DQVDT_FILL_CLDMACRO", export_repository)

        self._managed_state.register("fill_negative_tendency_cloud_micro.graupel", "DQGDT_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.convective_ice", "DQICNDT_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.large_scale_ice", "DQILSSDT_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.convective_liquid", "DQLCNDT_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.large_scale_liquid", "DQLLSDT_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.rain", "DQR_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.snow", "DQS_FILL_CLDMICRO", export_repository)
        self._managed_state.register("fill_negative_tendency_cloud_micro.vapor", "DQVDT_FILL_CLDMICRO", export_repository)

        # system to alert that radar diag is requested, but not yet implemented. remove when implemented
        # do_radar_diagnostic = (
        #     export_repository.associated("DBZ")
        #     or export_repository.associated("DBZ_MAX")
        #     or export_repository.associated("DBZ_1KM")
        #     or export_repository.associated("DBZ_TOP")
        #     or export_repository.associated("DBZ_M10C")
        # )

        # self._get_code().post_init(do_radar_diagnostic)

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
        self._first_run_init(mapl_state, import_state, export_state, internal_state)

        with TimedCUDAProfiler("GFDL 1M", {}):
            with TimedCUDAProfiler("GFDL 1M - State copy", {}):
                self._managed_state.fortran_to_ndsl()
                # KPBL_SC is Fortran-indexed. Convert to Python index while Python numerics run, then restore before state copy-back.
                self._managed_state.ndsl_state.boundary_layer_level_for_uw_shallow_conv.field[:] -= 1

            with TimedCUDAProfiler("GFDL 1M Numerics", {}):
                self._get_code()(self._managed_state.ndsl_state)

            with TimedCUDAProfiler("GFDL 1M - State copy-back", {}):
                self._managed_state.ndsl_state.boundary_layer_level_for_uw_shallow_conv.field[:] += 1
                self._managed_state.ndsl_to_fortran()

    def finalize(
        self,
        mapl_state: CVoidPointer,
        import_state: CVoidPointer,
        export_state: CVoidPointer,
    ):
        pass


CODE = GFDL1MInterface()
