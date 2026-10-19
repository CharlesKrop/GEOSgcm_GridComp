import copy
import time

from f90nml import Namelist
from ndsl import StencilFactory, ndsl_log
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyMoist import GFDL1M, GFDL1MConfig, GFDL1MState
from pyMoist.saturation_tables.tables.main import SaturationVaporPressureTable

# ⚠️ Careful those are _hot caches_ bench - do are NOT
# expect the same timings online in the model.
MICRO_BENCH = False


class TranslateGFDL1M(TranslateFortranData2Py):
    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.quantity_factory = grid.quantity_factory

        if MICRO_BENCH:
            self._ts = 10
            stencil_factory.config.dace_config.set_timer(stencil_factory.comm)
        else:
            self._ts = 0

        # NOTE not all fields are associated in v11.5.2. fields which are not associated are not serialized,
        # and not output by the parameterization, and have therefore been manually disabled within this test
        self.in_vars["data_vars"] = {
            "AN_PRCP": {},
            "AN_SNR": {},
            "AREA": {},
            "CLCN": {},
            "CLDREFFI": {},
            "CLDREFFL": {},
            "CLLS": {},
            "CNV_FRC": {},
            "CN_PRCP": {},
            "CN_SNR": {},
            "DQADT_macro": {},
            "DQADT_micro": {},
            "DQGDT_macro": {},
            "DQGDT_micro": {},
            "DQIDT_macro": {},
            "DQIDT_micro": {},
            "DQLDT_macro": {},
            "DQLDT_micro": {},
            "DQRDT_macro": {},
            "DQRDT_micro": {},
            "DQSDT_macro": {},
            "DQSDT_micro": {},
            "DQVDT_macro": {},
            "DQVDT_micro": {},
            "DTDTFRIC": {},
            "DTDT_macro": {},
            "DTDT_micro": {},
            "DUDT_macro": {},
            "DUDT_micro": {},
            "DVDT_macro": {},
            "DVDT_micro": {},
            "EIS": {},
            "EVAP": {},
            "EVAPC": {},
            "FRLAND": {},
            "FRZR": {},
            "ICE": {},
            "KH": {},
            "KPBLSC": {},
            "LS_PRCP": {},
            "LS_SNR": {},
            "LTS": {},
            "LWP": {},
            "NACTI": {},
            "NACTL": {},
            "OMEGA": {},
            "PDFITERS": {},
            "PDF_A": {},
            "PFG_LS": {},
            "PFI_AN": {},
            "PFI_LS": {},
            "PFL_AN": {},
            "PFL_LS": {},
            "PFR_LS": {},
            "PFS_LS": {},
            "PHIS": {},
            "PLE": {},
            "PRCP_GRAUPEL": {},
            "PRCP_ICE": {},
            "PRCP_RAIN": {},
            "PRCP_SNOW": {},
            "PRCP_WATER": {},
            "Q": {},
            "QGRAUPEL": {},
            "QGTOT": {},
            "QICN": {},
            "QILS": {},
            "QLCN": {},
            "QLLS": {},
            "QRAIN": {},
            "QRTOT": {},
            "QSNOW": {},
            "QSTOT": {},
            "QT2": {},
            "QT3": {},
            "RAD_CF": {},
            "RAD_QG": {},
            "RAD_QI": {},
            "RAD_QL": {},
            "RAD_QR": {},
            "RAD_QS": {},
            "RAD_QV": {},
            "REV_LS": {},
            "RHCRIT": {},
            "RHX": {},
            "RSU_LS": {},
            "SC_PRCP": {},
            "SC_SNR": {},
            "SH": {},
            "SHLW_PRC3": {},
            "SHLW_SNO3": {},
            "SL2": {},
            "SL3": {},
            "SLQT": {},
            "SRF_TYPE": {},
            "SUBLC": {},
            "T": {},
            "TS": {},
            "U": {},
            "V": {},
            "VFALL_GRAUPEL": {},
            "VFALL_ICE": {},
            "VFALL_RAIN": {},
            "VFALL_SNOW": {},
            "W": {},
            "W2": {},
            "W3": {},
            "WQL": {},
            "WQT": {},
            "WSL": {},
            "WTHV2": {},
            "ZLE": {},
        }

        self.out_vars = self.in_vars["data_vars"].copy()

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GFDL1M-Config")

    def compute(self, inputs):
        # initialize constants
        config = GFDL1MConfig(CWD="/Users/ckropiew/GEOS/v11.10.1/run/experiments/GCM_C24_L72_NX1_NY1/scratch", **self.constants)

        # initialize state
        state = GFDL1MState.zeros(self.quantity_factory)

        # fill state with input data
        # Top-level state fields
        safe_assign_array(state.area.field[:], inputs["AREA"])
        safe_assign_array(state.z_interface.field[:], inputs["ZLE"])
        safe_assign_array(state.p_interface.field[:], inputs["PLE"])
        safe_assign_array(state.t.field[:], inputs["T"])
        safe_assign_array(state.u.field[:], inputs["U"])
        safe_assign_array(state.v.field[:], inputs["V"])
        safe_assign_array(state.land_fraction.field[:], inputs["FRLAND"])
        safe_assign_array(state.scalar_diffusivity_interface.field[:], inputs["KH"])
        safe_assign_array(state.pdf_first_plume_fractional_area.field[:], inputs["PDF_A"])
        safe_assign_array(state.covariance_liquid_water_static_energy_and_total_water_specific_humidity.field[:], inputs["SLQT"])
        safe_assign_array(state.surface_temperature.field[:], inputs["TS"])
        safe_assign_array(state.sensible_heat_flux.field[:], inputs["SH"])
        safe_assign_array(state.omega.field[:], inputs["OMEGA"])
        safe_assign_array(state.convection_fraction.field[:], inputs["CNV_FRC"])
        safe_assign_array(state.surface_type.field[:], inputs["SRF_TYPE"])
        safe_assign_array(state.evaporation.field[:], inputs["EVAP"])
        safe_assign_array(state.surface_geopotential_height.field[:], inputs["PHIS"])
        safe_assign_array(state.cloud_liquid_evaporation.field[:], inputs["EVAPC"])
        safe_assign_array(state.cloud_ice_sublimation.field[:], inputs["SUBLC"])
        safe_assign_array(state.icefall.field[:], inputs["ICE"])
        safe_assign_array(state.freezing_rainfall.field[:], inputs["FRZR"])
        safe_assign_array(state.relative_humidity_after_pdf.field[:], inputs["RHX"])
        safe_assign_array(state.buoyancy_flux.field[:], inputs["WTHV2"])
        safe_assign_array(state.liquid_water_flux.field[:], inputs["WQL"])
        safe_assign_array(state.hydrostatic_pdf_iterations.field[:], inputs["PDFITERS"])
        safe_assign_array(state.lower_tropospheric_stability.field[:], inputs["LTS"])
        safe_assign_array(state.estimated_inversion_strength.field[:], inputs["EIS"])
        safe_assign_array(state.shallow_convection_rain.field[:], inputs["SHLW_PRC3"])
        safe_assign_array(state.shallow_convection_snow.field[:], inputs["SHLW_SNO3"])
        safe_assign_array(state.critical_relative_humidity_for_pdf.field[:], inputs["RHCRIT"])
        safe_assign_array(state.liquid_water_path.field[:], inputs["LWP"])
        safe_assign_array(state.boundary_layer_level_for_uw_shallow_conv.field[:], inputs["KPBLSC"])

        # state.vertical_motion
        safe_assign_array(state.vertical_motion.velocity.field[:], inputs["W"])
        safe_assign_array(state.vertical_motion.variance.field[:], inputs["W2"])
        safe_assign_array(state.vertical_motion.third_moment.field[:], inputs["W3"])

        # state.mixing_ratio
        safe_assign_array(state.mixing_ratio.vapor.field[:], inputs["Q"])
        safe_assign_array(state.mixing_ratio.rain.field[:], inputs["QRAIN"])
        safe_assign_array(state.mixing_ratio.snow.field[:], inputs["QSNOW"])
        safe_assign_array(state.mixing_ratio.graupel.field[:], inputs["QGRAUPEL"])
        safe_assign_array(state.mixing_ratio.large_scale_liquid.field[:], inputs["QLLS"])
        safe_assign_array(state.mixing_ratio.large_scale_ice.field[:], inputs["QILS"])
        safe_assign_array(state.mixing_ratio.convective_liquid.field[:], inputs["QLCN"])
        safe_assign_array(state.mixing_ratio.convective_ice.field[:], inputs["QICN"])

        # state.cloud_fraction
        safe_assign_array(state.cloud_fraction.large_scale.field[:], inputs["CLLS"])
        safe_assign_array(state.cloud_fraction.convective.field[:], inputs["CLCN"])

        # state.concentration
        safe_assign_array(state.concentration.liquid.field[:], inputs["NACTL"])
        safe_assign_array(state.concentration.ice.field[:], inputs["NACTI"])

        # state.liquid_water_static_energy
        safe_assign_array(state.liquid_water_static_energy.flux.field[:], inputs["WSL"])
        safe_assign_array(state.liquid_water_static_energy.variance.field[:], inputs["SL2"])
        safe_assign_array(state.liquid_water_static_energy.third_moment.field[:], inputs["SL3"])

        # state.total_water
        safe_assign_array(state.total_water.flux.field[:], inputs["WQT"])
        safe_assign_array(state.total_water.variance.field[:], inputs["QT2"])
        safe_assign_array(state.total_water.third_moment.field[:], inputs["QT3"])

        # state.radiation_field
        safe_assign_array(state.radiation_field.cloud_fraction.field[:], inputs["RAD_CF"])
        safe_assign_array(state.radiation_field.vapor.field[:], inputs["RAD_QV"])
        safe_assign_array(state.radiation_field.liquid.field[:], inputs["RAD_QL"])
        safe_assign_array(state.radiation_field.ice.field[:], inputs["RAD_QI"])
        safe_assign_array(state.radiation_field.rain.field[:], inputs["RAD_QR"])
        safe_assign_array(state.radiation_field.snow.field[:], inputs["RAD_QS"])
        safe_assign_array(state.radiation_field.graupel.field[:], inputs["RAD_QG"])

        # state.cloud_particle_effective_radius
        safe_assign_array(state.cloud_particle_effective_radius.liquid.field[:], inputs["CLDREFFL"])
        safe_assign_array(state.cloud_particle_effective_radius.ice.field[:], inputs["CLDREFFI"])

        # state.precipitation_at_surface
        safe_assign_array(state.precipitation_at_surface.water.field[:], inputs["PRCP_WATER"])
        safe_assign_array(state.precipitation_at_surface.rain.field[:], inputs["PRCP_RAIN"])
        safe_assign_array(state.precipitation_at_surface.snow.field[:], inputs["PRCP_SNOW"])
        safe_assign_array(state.precipitation_at_surface.ice.field[:], inputs["PRCP_ICE"])
        safe_assign_array(state.precipitation_at_surface.graupel.field[:], inputs["PRCP_GRAUPEL"])
        safe_assign_array(state.precipitation_at_surface.shallow_convective_precipitation.field[:], inputs["SC_PRCP"])
        safe_assign_array(state.precipitation_at_surface.deep_convective_precipitation.field[:], inputs["CN_PRCP"])
        safe_assign_array(state.precipitation_at_surface.anvil_precipitation.field[:], inputs["AN_PRCP"])
        safe_assign_array(state.precipitation_at_surface.shallow_convective_snow.field[:], inputs["SC_SNR"])
        safe_assign_array(state.precipitation_at_surface.deep_convective_snow.field[:], inputs["CN_SNR"])
        safe_assign_array(state.precipitation_at_surface.anvil_snow.field[:], inputs["AN_SNR"])

        # state.non_anvil_large_scale
        safe_assign_array(state.non_anvil_large_scale.precip.field[:], inputs["LS_PRCP"])
        safe_assign_array(state.non_anvil_large_scale.snow.field[:], inputs["LS_SNR"])
        safe_assign_array(state.non_anvil_large_scale.evaporation.field[:], inputs["REV_LS"])
        safe_assign_array(state.non_anvil_large_scale.sublimation.field[:], inputs["RSU_LS"])
        safe_assign_array(state.non_anvil_large_scale.ice_precip_flux.field[:], inputs["PFI_LS"])
        safe_assign_array(state.non_anvil_large_scale.liquid_precip_flux.field[:], inputs["PFL_LS"])
        safe_assign_array(state.non_anvil_large_scale.rain_precip_flux.field[:], inputs["PFR_LS"])
        safe_assign_array(state.non_anvil_large_scale.snow_precip_flux.field[:], inputs["PFS_LS"])
        safe_assign_array(state.non_anvil_large_scale.graupel_precip_flux.field[:], inputs["PFG_LS"])

        # state.anvil
        safe_assign_array(state.anvil.liquid_precip_flux.field[:], inputs["PFL_AN"])
        safe_assign_array(state.anvil.ice_precip_flux.field[:], inputs["PFI_AN"])

        # state.tendencies
        safe_assign_array(state.tendencies.dcloud_fraction_dt_macro.field[:], inputs["DQADT_macro"])
        safe_assign_array(state.tendencies.dvapor_dt_macro.field[:], inputs["DQVDT_macro"])
        safe_assign_array(state.tendencies.dice_dt_macro.field[:], inputs["DQIDT_macro"])
        safe_assign_array(state.tendencies.dliquid_dt_macro.field[:], inputs["DQLDT_macro"])
        safe_assign_array(state.tendencies.drain_dt_macro.field[:], inputs["DQRDT_macro"])
        safe_assign_array(state.tendencies.dgraupel_dt_macro.field[:], inputs["DQGDT_macro"])
        safe_assign_array(state.tendencies.dsnow_dt_macro.field[:], inputs["DQSDT_macro"])
        safe_assign_array(state.tendencies.du_dt_macro.field[:], inputs["DUDT_macro"])
        safe_assign_array(state.tendencies.dv_dt_macro.field[:], inputs["DVDT_macro"])
        safe_assign_array(state.tendencies.dt_dt_macro.field[:], inputs["DTDT_macro"])

        safe_assign_array(state.tendencies.dcloud_fraction_dt_micro.field[:], inputs["DQADT_micro"])
        safe_assign_array(state.tendencies.dvapor_dt_micro.field[:], inputs["DQVDT_micro"])
        safe_assign_array(state.tendencies.dice_dt_micro.field[:], inputs["DQIDT_micro"])
        safe_assign_array(state.tendencies.dliquid_dt_micro.field[:], inputs["DQLDT_micro"])
        safe_assign_array(state.tendencies.drain_dt_micro.field[:], inputs["DQRDT_micro"])
        safe_assign_array(state.tendencies.dgraupel_dt_micro.field[:], inputs["DQGDT_micro"])
        safe_assign_array(state.tendencies.dsnow_dt_micro.field[:], inputs["DQSDT_micro"])
        safe_assign_array(state.tendencies.du_dt_micro.field[:], inputs["DUDT_micro"])
        safe_assign_array(state.tendencies.dv_dt_micro.field[:], inputs["DVDT_micro"])
        safe_assign_array(state.tendencies.dt_dt_micro.field[:], inputs["DTDT_micro"])

        safe_assign_array(state.tendencies.dt_dt_friction_pressure_weighted.field[:], inputs["DTDTFRIC"])

        # state.mass_fraction
        safe_assign_array(state.mass_fraction.suspended_graupel.field[:], inputs["QGTOT"])
        safe_assign_array(state.mass_fraction.suspended_rain.field[:], inputs["QRTOT"])
        safe_assign_array(state.mass_fraction.suspended_snow.field[:], inputs["QSTOT"])

        # state.fall_speed
        safe_assign_array(state.fall_speed.graupel.field[:], inputs["VFALL_GRAUPEL"])
        safe_assign_array(state.fall_speed.ice.field[:], inputs["VFALL_ICE"])
        safe_assign_array(state.fall_speed.rain.field[:], inputs["VFALL_RAIN"])
        safe_assign_array(state.fall_speed.snow.field[:], inputs["VFALL_SNOW"])

        # initialize test class
        code = GFDL1M(
            stencil_factory=self.stencil_factory,
            quantity_factory=self.quantity_factory,
            config=config,
        )

        # execute test code
        code(state=state)

        outputs = {
            "AREA": state.area.field[:],
            "ZLE": state.z_interface.field[:],
            "PLE": state.p_interface.field[:],
            "T": state.t.field[:],
            "U": state.u.field[:],
            "V": state.v.field[:],
            "FRLAND": state.land_fraction.field[:],
            "KH": state.scalar_diffusivity_interface.field[:],
            "PDF_A": state.pdf_first_plume_fractional_area.field[:],
            "SLQT": state.covariance_liquid_water_static_energy_and_total_water_specific_humidity.field[:],
            "TS": state.surface_temperature.field[:],
            "SH": state.sensible_heat_flux.field[:],
            "OMEGA": state.omega.field[:],
            "CNV_FRC": state.convection_fraction.field[:],
            "SRF_TYPE": state.surface_type.field[:],
            "EVAP": state.evaporation.field[:],
            "PHIS": state.surface_geopotential_height.field[:],
            "EVAPC": state.cloud_liquid_evaporation.field[:],
            "SUBLC": state.cloud_ice_sublimation.field[:],
            "ICE": state.icefall.field[:],
            "FRZR": state.freezing_rainfall.field[:],
            "RHX": state.relative_humidity_after_pdf.field[:],
            "WTHV2": state.buoyancy_flux.field[:],
            "WQL": state.liquid_water_flux.field[:],
            "PDFITERS": state.hydrostatic_pdf_iterations.field[:],
            "LTS": state.lower_tropospheric_stability.field[:],
            "EIS": state.estimated_inversion_strength.field[:],
            "SHLW_PRC3": state.shallow_convection_rain.field[:],
            "SHLW_SNO3": state.shallow_convection_snow.field[:],
            "RHCRIT": state.critical_relative_humidity_for_pdf.field[:],
            "LWP": state.liquid_water_path.field[:],
            "KPBLSC": state.boundary_layer_level_for_uw_shallow_conv.field[:],
            "W": state.vertical_motion.velocity.field[:],
            "W2": state.vertical_motion.variance.field[:],
            "W3": state.vertical_motion.third_moment.field[:],
            "Q": state.mixing_ratio.vapor.field[:],
            "QRAIN": state.mixing_ratio.rain.field[:],
            "QSNOW": state.mixing_ratio.snow.field[:],
            "QGRAUPEL": state.mixing_ratio.graupel.field[:],
            "QLLS": state.mixing_ratio.large_scale_liquid.field[:],
            "QILS": state.mixing_ratio.large_scale_ice.field[:],
            "QLCN": state.mixing_ratio.convective_liquid.field[:],
            "QICN": state.mixing_ratio.convective_ice.field[:],
            "CLLS": state.cloud_fraction.large_scale.field[:],
            "CLCN": state.cloud_fraction.convective.field[:],
            "NACTL": state.concentration.liquid.field[:],
            "NACTI": state.concentration.ice.field[:],
            "WSL": state.liquid_water_static_energy.flux.field[:],
            "SL2": state.liquid_water_static_energy.variance.field[:],
            "SL3": state.liquid_water_static_energy.third_moment.field[:],
            "WQT": state.total_water.flux.field[:],
            "QT2": state.total_water.variance.field[:],
            "QT3": state.total_water.third_moment.field[:],
            "RAD_CF": state.radiation_field.cloud_fraction.field[:],
            "RAD_QV": state.radiation_field.vapor.field[:],
            "RAD_QL": state.radiation_field.liquid.field[:],
            "RAD_QI": state.radiation_field.ice.field[:],
            "RAD_QR": state.radiation_field.rain.field[:],
            "RAD_QS": state.radiation_field.snow.field[:],
            "RAD_QG": state.radiation_field.graupel.field[:],
            "CLDREFFL": state.cloud_particle_effective_radius.liquid.field[:],
            "CLDREFFI": state.cloud_particle_effective_radius.ice.field[:],
            "PRCP_WATER": state.precipitation_at_surface.water.field[:],
            "PRCP_RAIN": state.precipitation_at_surface.rain.field[:],
            "PRCP_SNOW": state.precipitation_at_surface.snow.field[:],
            "PRCP_ICE": state.precipitation_at_surface.ice.field[:],
            "PRCP_GRAUPEL": state.precipitation_at_surface.graupel.field[:],
            "SC_PRCP": state.precipitation_at_surface.shallow_convective_precipitation.field[:],
            "CN_PRCP": state.precipitation_at_surface.deep_convective_precipitation.field[:],
            "AN_PRCP": state.precipitation_at_surface.anvil_precipitation.field[:],
            "SC_SNR": state.precipitation_at_surface.shallow_convective_snow.field[:],
            "CN_SNR": state.precipitation_at_surface.deep_convective_snow.field[:],
            "AN_SNR": state.precipitation_at_surface.anvil_snow.field[:],
            "LS_PRCP": state.non_anvil_large_scale.precip.field[:],
            "LS_SNR": state.non_anvil_large_scale.snow.field[:],
            "REV_LS": state.non_anvil_large_scale.evaporation.field[:],
            "RSU_LS": state.non_anvil_large_scale.sublimation.field[:],
            "PFI_LS": state.non_anvil_large_scale.ice_precip_flux.field[:],
            "PFL_LS": state.non_anvil_large_scale.liquid_precip_flux.field[:],
            "PFR_LS": state.non_anvil_large_scale.rain_precip_flux.field[:],
            "PFS_LS": state.non_anvil_large_scale.snow_precip_flux.field[:],
            "PFG_LS": state.non_anvil_large_scale.graupel_precip_flux.field[:],
            "PFL_AN": state.anvil.liquid_precip_flux.field[:],
            "PFI_AN": state.anvil.ice_precip_flux.field[:],
            "DQADT_macro": state.tendencies.dcloud_fraction_dt_macro.field[:],
            "DQVDT_macro": state.tendencies.dvapor_dt_macro.field[:],
            "DQIDT_macro": state.tendencies.dice_dt_macro.field[:],
            "DQLDT_macro": state.tendencies.dliquid_dt_macro.field[:],
            "DQRDT_macro": state.tendencies.drain_dt_macro.field[:],
            "DQGDT_macro": state.tendencies.dgraupel_dt_macro.field[:],
            "DQSDT_macro": state.tendencies.dsnow_dt_macro.field[:],
            "DUDT_macro": state.tendencies.du_dt_macro.field[:],
            "DVDT_macro": state.tendencies.dv_dt_macro.field[:],
            "DTDT_macro": state.tendencies.dt_dt_macro.field[:],
            "DQADT_micro": state.tendencies.dcloud_fraction_dt_micro.field[:],
            "DQVDT_micro": state.tendencies.dvapor_dt_micro.field[:],
            "DQIDT_micro": state.tendencies.dice_dt_micro.field[:],
            "DQLDT_micro": state.tendencies.dliquid_dt_micro.field[:],
            "DQRDT_micro": state.tendencies.drain_dt_micro.field[:],
            "DQGDT_micro": state.tendencies.dgraupel_dt_micro.field[:],
            "DQSDT_micro": state.tendencies.dsnow_dt_micro.field[:],
            "DUDT_micro": state.tendencies.du_dt_micro.field[:],
            "DVDT_micro": state.tendencies.dv_dt_micro.field[:],
            "DTDT_micro": state.tendencies.dt_dt_micro.field[:],
            "DTDTFRIC": state.tendencies.dt_dt_friction_pressure_weighted.field[:],
            "QGTOT": state.mass_fraction.suspended_graupel.field[:],
            "QRTOT": state.mass_fraction.suspended_rain.field[:],
            "QSTOT": state.mass_fraction.suspended_snow.field[:],
            "VFALL_GRAUPEL": state.fall_speed.graupel.field[:],
            "VFALL_ICE": state.fall_speed.ice.field[:],
            "VFALL_RAIN": state.fall_speed.rain.field[:],
            "VFALL_SNOW": state.fall_speed.snow.field[:],
        }

        validation_outputs = copy.deepcopy(outputs)

        if self._ts > 0:
            s = time.perf_counter()
            for _ in range(self._ts):
                code(state=state)
            e = time.perf_counter()
            ndsl_log.info(f"GFDL1M Driver micro bench: {(e - s) / self._ts:.4f}s")
            self.stencil_factory.config.dace_config.performance_collector.write_out_rank_0(self.stencil_factory.backend, float(config.DT_MOIST), "done")

        return validation_outputs
