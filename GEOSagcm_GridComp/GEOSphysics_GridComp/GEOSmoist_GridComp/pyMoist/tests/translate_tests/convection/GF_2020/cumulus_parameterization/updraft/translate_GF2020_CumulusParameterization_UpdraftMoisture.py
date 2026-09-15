from f90nml import Namelist
from ndsl import StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py

from pyMoist.convection.GF_2020.config import GF2020Config
from pyMoist.convection.GF_2020.cumulus_parameterization.config import GF2020CumulusParameterizationConfig
from pyMoist.convection.GF_2020.cumulus_parameterization.constants import MAXENS1, MAXENS2, MAXENS3, NUMBER_OF_PLUMES
from pyMoist.convection.GF_2020.cumulus_parameterization.locals import GF2020CumulusParameterizationLocals
from pyMoist.convection.GF_2020.cumulus_parameterization.plume_dependent_constants import GF2020PlumeDependentConstants
from pyMoist.convection.GF_2020.cumulus_parameterization.setup.set_constants import set_constants
from pyMoist.convection.GF_2020.cumulus_parameterization.state import GF2020CumulusParameterizationState
from pyMoist.convection.GF_2020.cumulus_parameterization.updraft import updraft_moisture


class TestCore:
    def __init__(
        self,
        grid: Grid,
        stencil_factory: StencilFactory,
        in_vars: dict,
        out_vars: dict,
    ):
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        in_vars["data_vars"] = {
            "local_start_level": {},
            "error_code": {},
            "local_geopotential_height_cloud_levels_forced": {},
            "local_cloud_total_water_after_entrainment_forced": {},
            "cloud_liquid_after_rain_forced": {},
            "condensate_to_fall_forced": {},
            "total_normalized_integrated_condensate_forced": {},
            "local_cloud_moist_static_energy_forced": {},
            "local_updraft_column_temperature_forced": {},
            "ocean_fraction": {},
            "convection_fraction": {},
            "surface_type": {},
            "p_forced": {},
            "cloud_top_level": {},
            "local_d_buoyancy_forced": {},
            "local_cloud_liquid_before_rain_forced": {},
            "local_t_cloud_levels": {},
            "local_vapor_forced": {},
            "local_gamma_cloud_levels_forced": {},
            "normalized_massflux_updraft_forced": {},
            "local_env_saturation_mixing_ratio_cloud_levels_forced": {},
            "updraft_origin_level": {},
            "local_vapor_cloud_levels_forced": {},
            "local_vapor_excess": {},
            "ccn": {},
            "local_mass_entrainment_updraft": {},
            "local_mass_detrainment_updraft": {},
            "local_psum": {},
            "local_psumh": {},
            "local_c1d": {},
            "local_add_buoyancy": {},
            "local_vertical_velocity_3d": {},
        }

        out_vars.update(in_vars["data_vars"])

    def __call__(self, constants: dict, cu_param_constants: dict, plume: str, **inputs):
        # initialize constants
        config = GF2020Config(**constants)
        cumulus_parameterization_config = GF2020CumulusParameterizationConfig(**cu_param_constants)
        plume_dependent_constants = GF2020PlumeDependentConstants()
        plume_dependent_constants = set_constants(cumulus_parameterization_config, plume_dependent_constants, plume)

        # initialize dataclasses
        state = GF2020CumulusParameterizationState.zeros(
            self.quantity_factory,
            data_dimensions={
                "plumes": NUMBER_OF_PLUMES,
                "convection_tracers": config.NUMBER_OF_TRACERS,
            },
        )

        locals = GF2020CumulusParameterizationLocals.zeros(
            self.quantity_factory,
            data_dimensions={
                "ensemble_1": MAXENS1,
                "ensemble_2": MAXENS2,
                "ensemble_3": MAXENS3,
                "ensemble_members": MAXENS1 * MAXENS2 * MAXENS3,
                "convection_tracers": config.NUMBER_OF_TRACERS,
            },
        )

        # fill relevant parts of dataclasses
        locals.start_level[:] = inputs["local_start_level"] - 1
        state.output.error_code[:, :, plume_dependent_constants.PLUME_INDEX] = inputs["error_code"]
        locals.geopotential_height_cloud_levels_forced[:] = inputs["local_geopotential_height_cloud_levels_forced"]
        locals.cloud_total_water_after_entrainment_forced[:] = inputs["local_cloud_total_water_after_entrainment_forced"]
        state.output.cloud_liquid_after_rain_forced[:, :, :, plume_dependent_constants.PLUME_INDEX] = inputs["cloud_liquid_after_rain_forced"]
        state.output.condensate_to_fall_forced[:, :, :, plume_dependent_constants.PLUME_INDEX] = inputs["condensate_to_fall_forced"]
        state.output.total_normalized_integrated_condensate_forced[:, :, plume_dependent_constants.PLUME_INDEX] = inputs["total_normalized_integrated_condensate_forced"]
        locals.cloud_moist_static_energy_forced[:] = inputs["local_cloud_moist_static_energy_forced"]
        locals.updraft_column_temperature_forced[:] = inputs["local_updraft_column_temperature_forced"]
        state.input.ocean_fraction[:] = inputs["ocean_fraction"]
        state.input.convection_fraction[:] = inputs["convection_fraction"]
        state.input.surface_type[:] = inputs["surface_type"]
        state.input_output.p_forced[:] = inputs["p_forced"]
        state.output.cloud_top_level[:, :, plume_dependent_constants.PLUME_INDEX] = inputs["cloud_top_level"] - 1
        locals.d_buoyancy_forced[:] = inputs["local_d_buoyancy_forced"]
        locals.cloud_liquid_before_rain_forced[:] = inputs["local_cloud_liquid_before_rain_forced"]
        locals.t_cloud_levels[:] = inputs["local_t_cloud_levels"]
        locals.vapor_forced[:] = inputs["local_vapor_forced"]
        locals.gamma_cloud_levels_forced[:] = inputs["local_gamma_cloud_levels_forced"]
        state.output.normalized_massflux_updraft_forced[:, :, :, plume_dependent_constants.PLUME_INDEX] = inputs["normalized_massflux_updraft_forced"]
        locals.environment_saturation_mixing_ratio_cloud_levels_forced[:] = inputs["local_env_saturation_mixing_ratio_cloud_levels_forced"]
        state.output.updraft_origin_level[:, :, plume_dependent_constants.PLUME_INDEX] = inputs["updraft_origin_level"] - 1
        locals.vapor_cloud_levels_forced[:] = inputs["local_vapor_cloud_levels_forced"]
        locals.vapor_excess[:] = inputs["local_vapor_excess"]
        state.input_output.ccn[:] = inputs["ccn"]
        locals.mass_entrainment_updraft[:] = inputs["local_mass_entrainment_updraft"]
        locals.mass_detrainment_updraft[:] = inputs["local_mass_detrainment_updraft"]
        locals.psum[:] = inputs["local_psum"]
        locals.psumh[:] = inputs["local_psumh"]
        locals.c1d[:] = inputs["local_c1d"]
        locals.add_buoyancy[:] = inputs["local_add_buoyancy"]
        locals.vertical_velocity_3d[:] = inputs["local_vertical_velocity_3d"]

        # initialize test code
        code = self.stencil_factory.from_dims_halo(
            func=updraft_moisture,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "BOUNDARY_CONDITION_METHOD": cumulus_parameterization_config.BOUNDARY_CONDITION_METHOD,
                "USE_LINEAR_SUBCLOUD_MOISTURE_FLUXES": cumulus_parameterization_config.USE_LINEAR_SUBCLOUD_MOISTURE_FLUXES,
                "AUTOCONV": config.AUTOCONV,
                "CRITICAL_MIXING_RATIO_OVER_OCEAN": cumulus_parameterization_config.CRITICAL_MIXING_RATIO_OVER_OCEAN,
                "CRITICAL_MIXING_RATIO_OVER_LAND": cumulus_parameterization_config.CRITICAL_MIXING_RATIO_OVER_LAND,
                "FRAC_MODIS": cumulus_parameterization_config.FRAC_MODIS,
                "ZERO_DIFF": cumulus_parameterization_config.ZERO_DIFF,
            },
        )

        # call test code
        if plume_dependent_constants.ENABLE_PLUME == 1:
            code(
                start_level=locals.start_level,
                error_code=state.output.error_code,
                geopotential_height_cloud_levels_forced=locals.geopotential_height_cloud_levels_forced,
                cloud_total_water_after_entrainment_forced=locals.cloud_total_water_after_entrainment_forced,
                cloud_liquid_after_rain_forced=state.output.cloud_liquid_after_rain_forced,
                condensate_to_fall_forced=state.output.condensate_to_fall_forced,
                total_normalized_integrated_condensate_forced=state.output.total_normalized_integrated_condensate_forced,
                cloud_moist_static_energy_forced=locals.cloud_moist_static_energy_forced,
                updraft_column_temperature_forced=locals.updraft_column_temperature_forced,
                ocean_fraction=state.input.ocean_fraction,
                convection_fraction=state.input.convection_fraction,
                surface_type=state.input.surface_type,
                p_forced=state.input_output.p_forced,
                cloud_top_level=state.output.cloud_top_level,
                d_buoyancy_forced=locals.d_buoyancy_forced,
                cloud_liquid_before_rain_forced=locals.cloud_liquid_before_rain_forced,
                t_cloud_levels=locals.t_cloud_levels,
                vapor_forced=locals.vapor_forced,
                gamma_cloud_levels_forced=locals.gamma_cloud_levels_forced,
                normalized_massflux_updraft_forced=state.output.normalized_massflux_updraft_forced,
                environment_saturation_mixing_ratio_cloud_levels_forced=locals.environment_saturation_mixing_ratio_cloud_levels_forced,
                updraft_origin_level=state.output.updraft_origin_level,
                vapor_cloud_levels_forced=locals.vapor_cloud_levels_forced,
                vapor_excess=locals.vapor_excess,
                ccn=state.input_output.ccn,
                mass_entrainment_updraft=locals.mass_entrainment_updraft,
                mass_detrainment_updraft=locals.mass_detrainment_updraft,
                psum=locals.psum,
                psumh=locals.psumh,
                c1d=locals.c1d,
                add_buoyancy=locals.add_buoyancy,
                vertical_velocity_3d=locals.vertical_velocity_3d,
                C0=plume_dependent_constants.C0,
                AVERAGE_LAYER_DEPTH=plume_dependent_constants.AVERAGE_LAYER_DEPTH,
                plume=plume_dependent_constants.PLUME_INDEX,
            )

        # write output
        outputs = {
            "local_start_level": locals.start_level.field[:] + 1,
            "error_code": state.output.error_code.field[:, :, plume_dependent_constants.PLUME_INDEX],
            "local_geopotential_height_cloud_levels_forced": locals.geopotential_height_cloud_levels_forced.field[:],
            "local_cloud_total_water_after_entrainment_forced": locals.cloud_total_water_after_entrainment_forced.field[:],
            "cloud_liquid_after_rain_forced": state.output.cloud_liquid_after_rain_forced.field[:, :, :, plume_dependent_constants.PLUME_INDEX],
            "condensate_to_fall_forced": state.output.condensate_to_fall_forced.field[:, :, :, plume_dependent_constants.PLUME_INDEX],
            "total_normalized_integrated_condensate_forced": state.output.total_normalized_integrated_condensate_forced.field[
                :, :, plume_dependent_constants.PLUME_INDEX
            ],
            "local_cloud_moist_static_energy_forced": locals.cloud_moist_static_energy_forced.field[:],
            "local_updraft_column_temperature_forced": locals.updraft_column_temperature_forced.field[:],
            "ocean_fraction": state.input.ocean_fraction.field[:],
            "convection_fraction": state.input.convection_fraction.field[:],
            "surface_type": state.input.surface_type.field[:],
            "p_forced": state.input_output.p_forced.field[:],
            "cloud_top_level": state.output.cloud_top_level.field[:, :, plume_dependent_constants.PLUME_INDEX] + 1,
            "local_d_buoyancy_forced": locals.d_buoyancy_forced.field[:],
            "local_cloud_liquid_before_rain_forced": locals.cloud_liquid_before_rain_forced.field[:],
            "local_t_cloud_levels": locals.t_cloud_levels.field[:],
            "local_vapor_forced": locals.vapor_forced.field[:],
            "local_gamma_cloud_levels_forced": locals.gamma_cloud_levels_forced.field[:],
            "normalized_massflux_updraft_forced": state.output.normalized_massflux_updraft_forced.field[:, :, :, plume_dependent_constants.PLUME_INDEX],
            "local_env_saturation_mixing_ratio_cloud_levels_forced": locals.environment_saturation_mixing_ratio_cloud_levels_forced.field[:],
            "updraft_origin_level": state.output.updraft_origin_level.field[:, :, plume_dependent_constants.PLUME_INDEX] + 1,
            "local_vapor_cloud_levels_forced": locals.vapor_cloud_levels_forced.field[:],
            "local_vapor_excess": locals.vapor_excess.field[:],
            "ccn": state.input_output.ccn.field[:],
            "local_mass_entrainment_updraft": locals.mass_entrainment_updraft.field[:],
            "local_mass_detrainment_updraft": locals.mass_detrainment_updraft.field[:],
            "local_psum": locals.psum.field[:],
            "local_psumh": locals.psumh.field[:],
            "local_c1d": locals.c1d.field[:],
            "local_add_buoyancy": locals.add_buoyancy.field[:],
            "local_vertical_velocity_3d": locals.vertical_velocity_3d.field[:],
        }

        return outputs


class TranslateGF2020_CumulusParameterization_UpdraftMoisture_shallow(TranslateFortranData2Py):
    def __init__(
        self,
        grid: Grid,
        _namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)

        self.test_core = TestCore(grid, stencil_factory, self.in_vars, self.out_vars)

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GF2020-constants")
        self.cu_param_constants = data_loader.load("GF2020_CumulusParameterization-constants")

    def compute_func(self, **inputs):
        outputs = self.test_core(self.constants, self.cu_param_constants, "shallow", **inputs)

        return outputs


class TranslateGF2020_CumulusParameterization_UpdraftMoisture_mid(TranslateFortranData2Py):
    def __init__(
        self,
        grid: Grid,
        _namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)

        self.test_core = TestCore(grid, stencil_factory, self.in_vars, self.out_vars)

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GF2020-constants")
        self.cu_param_constants = data_loader.load("GF2020_CumulusParameterization-constants")

    def compute_func(self, **inputs):
        outputs = self.test_core(self.constants, self.cu_param_constants, "mid", **inputs)

        return outputs


class TranslateGF2020_CumulusParameterization_UpdraftMoisture_deep(TranslateFortranData2Py):
    def __init__(
        self,
        grid: Grid,
        _namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)

        self.test_core = TestCore(grid, stencil_factory, self.in_vars, self.out_vars)

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("GF2020-constants")
        self.cu_param_constants = data_loader.load("GF2020_CumulusParameterization-constants")

    def compute_func(self, **inputs):
        outputs = self.test_core(self.constants, self.cu_param_constants, "deep", **inputs)

        return outputs
