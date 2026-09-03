import copy

from f90nml import Namelist
from ndsl import Quantity, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.stencils.testing.translate import TranslateFortranData2Py

from pyMoist.convection.GF_2020.config import GF2020Config
from pyMoist.convection.GF_2020.cumulus_parameterization.config import (
    GF2020CumulusParameterizationConfig,
)
from pyMoist.convection.GF_2020.cumulus_parameterization.constants import (
    MAXENS1,
    MAXENS2,
    MAXENS3,
    NUMBER_OF_PLUMES,
    Plumes,
)
from pyMoist.convection.GF_2020.cumulus_parameterization.environment import (
    environment_cloud_levels,
)
from pyMoist.convection.GF_2020.cumulus_parameterization.locals import (
    GF2020CumulusParameterizationLocals,
)
from pyMoist.convection.GF_2020.cumulus_parameterization.plume_dependent_constants import (
    GF2020PlumeDependentConstants,
)
from pyMoist.convection.GF_2020.cumulus_parameterization.setup.set_constants import (
    set_constants,
)
from pyMoist.convection.GF_2020.cumulus_parameterization.state import (
    GF2020CumulusParameterizationState,
)


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
            "error_code": {},
            "p_forced": {},
            "p_cloud_levels_forced": {},
            "p_surface": {},
            "local_t_modified": {},
            "local_t_cloud_levels_modified": {},
            "t_surface": {},
            "local_vapor_modified": {},
            "local_vapor_cloud_levels_modified": {},
            "local_env_saturation_mixing_ratio_modified": {},
            "local_env_saturation_mixing_ratio_cloud_levels_modified": {},
            "u": {},
            "v": {},
            "local_u_cloud_levels": {},
            "local_v_cloud_levels": {},
            "local_env_moist_static_energy_modified": {},
            "local_env_moist_static_energy_cloud_levels_modified": {},
            "local_env_saturation_moist_static_energy_modified": {},
            "local_env_saturation_moist_static_energy_cloud_levels_modified": {},
            "local_geopotential_height_modified": {},
            "local_geopotential_height_cloud_levels_modified": {},
            "topography_height_no_negative": {},
            "local_gamma_cloud_levels": {},
        }

        out_vars.update(in_vars["data_vars"])

    def __call__(self, constants: dict, cu_param_constants: dict, plume: str, **inputs):
        # initialize constants
        config = GF2020Config(**constants)
        cumulus_parameterization_config = GF2020CumulusParameterizationConfig(
            **cu_param_constants
        )
        plume_dependent_constants = GF2020PlumeDependentConstants()
        plume_dependent_constants = set_constants(
            cumulus_parameterization_config, plume_dependent_constants, plume
        )

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
        state.output.error_code[:, :, plume_dependent_constants.PLUME_INDEX] = inputs[
            "error_code"
        ]
        state.input_output.p_forced[:] = inputs["p_forced"]
        state.output.p_cloud_levels_forced[
            :, :, :, plume_dependent_constants.PLUME_INDEX
        ] = inputs["p_cloud_levels_forced"]
        state.input_output.p_surface[:] = inputs["p_surface"]
        locals.t_modified[:] = inputs["local_t_modified"]
        locals.t_cloud_levels_modified[:] = inputs["local_t_cloud_levels_modified"]
        state.input_output.t_surface[:] = inputs["t_surface"]
        locals.vapor_modified[:] = inputs["local_vapor_modified"]
        locals.vapor_cloud_levels_modified[:] = inputs[
            "local_vapor_cloud_levels_modified"
        ]
        locals.environment_saturation_mixing_ratio_modified[:] = inputs[
            "local_env_saturation_mixing_ratio_modified"
        ]
        locals.environment_saturation_mixing_ratio_cloud_levels_modified[:] = inputs[
            "local_env_saturation_mixing_ratio_cloud_levels_modified"
        ]
        state.input_output.u[:] = inputs["u"]
        state.input_output.v[:] = inputs["v"]
        locals.u_cloud_levels[:] = inputs["local_u_cloud_levels"]
        locals.v_cloud_levels[:] = inputs["local_v_cloud_levels"]
        locals.environment_moist_static_energy_modified[:] = inputs[
            "local_env_moist_static_energy_modified"
        ]
        locals.environment_moist_static_energy_cloud_levels_modified[:] = inputs[
            "local_env_moist_static_energy_cloud_levels_modified"
        ]
        locals.environment_saturation_moist_static_energy_modified[:] = inputs[
            "local_env_saturation_moist_static_energy_modified"
        ]
        locals.environment_saturation_moist_static_energy_cloud_levels_modified[:] = (
            inputs["local_env_saturation_moist_static_energy_cloud_levels_modified"]
        )
        locals.geopotential_height_modified[:] = inputs[
            "local_geopotential_height_modified"
        ]
        locals.geopotential_height_cloud_levels_modified[:] = inputs[
            "local_geopotential_height_cloud_levels_modified"
        ]
        state.input_output.topography_height_no_negative[:] = inputs[
            "topography_height_no_negative"
        ]
        locals.gamma_cloud_levels[:] = inputs["local_gamma_cloud_levels"]

        code = self.stencil_factory.from_dims_halo(
            func=environment_cloud_levels,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "CLOUD_LEVEL_GRID": cumulus_parameterization_config.CLOUD_LEVEL_GRID
            },
        )

        if plume_dependent_constants.ENABLE_PLUME == 1:
            p_3d: Quantity = copy.deepcopy(state.input_output.u)

            p_3d.field[:] = state.output.p_cloud_levels_forced.field[
                :, :, :, plume_dependent_constants.PLUME_INDEX
            ]
            # print(p_3d.shape)
            # print(locals.geopotential_height_modified.shape)
            # print(
            #     state.output.p_cloud_levels_forced.field[
            #         :, :, :, plume_dependent_constants.PLUME_INDEX
            #     ].shape
            # )
            code(
                p=state.input_output.p_forced,
                p_surface=state.input_output.p_surface,
                p_cloud_levels=p_3d,
                topography_height_no_negative=state.input_output.topography_height_no_negative,
                geopotential_height=locals.geopotential_height_modified,
                geopotential_height_cloud_levels=locals.geopotential_height_cloud_levels_modified,
                t=locals.t_modified,
                t_surface=state.input_output.t_surface,
                t_cloud_levels=locals.t_cloud_levels_modified,
                vapor=locals.vapor_modified,
                vapor_cloud_levels=locals.vapor_cloud_levels_modified,
                u=state.input_output.u,
                v=state.input_output.v,
                u_cloud_levels=locals.u_cloud_levels,
                v_cloud_levels=locals.v_cloud_levels,
                environment_saturation_mixing_ratio=locals.environment_saturation_mixing_ratio_modified,
                environment_saturation_mixing_ratio_cloud_levels=locals.environment_saturation_mixing_ratio_cloud_levels_modified,
                environment_moist_static_energy=locals.environment_moist_static_energy_modified,
                environment_moist_static_energy_cloud_levels=locals.environment_moist_static_energy_cloud_levels_modified,
                environment_saturation_moist_static_energy=locals.environment_saturation_moist_static_energy_modified,
                environment_saturation_moist_static_energy_cloud_levels=locals.environment_saturation_moist_static_energy_cloud_levels_modified,
                gamma_cloud_levels=locals.gamma_cloud_levels,
                error_code=state.output.error_code,
                plume=plume_dependent_constants.PLUME_INDEX,
            )
            state.output.p_cloud_levels_forced.field[
                :, :, :, plume_dependent_constants.PLUME_INDEX
            ] = p_3d.field

        outputs = {
            "error_code": state.output.error_code.field[
                :, :, plume_dependent_constants.PLUME_INDEX
            ],
            "p_forced": state.input_output.p_forced.field[:],
            "p_cloud_levels_forced": state.output.p_cloud_levels_forced.field[
                :, :, :, plume_dependent_constants.PLUME_INDEX
            ],
            "p_surface": state.input_output.p_surface.field[:],
            "local_t_modified": locals.t_modified.field[:],
            "local_t_cloud_levels_modified": locals.t_cloud_levels_modified.field[:],
            "t_surface": state.input_output.t_surface.field[:],
            "local_vapor_modified": locals.vapor_modified.field[:],
            "local_vapor_cloud_levels_modified": locals.vapor_cloud_levels_modified.field[
                :
            ],
            "local_env_saturation_mixing_ratio_modified": locals.environment_saturation_mixing_ratio_modified.field[
                :
            ],
            "local_env_saturation_mixing_ratio_cloud_levels_modified": locals.environment_saturation_mixing_ratio_cloud_levels_modified.field[
                :
            ],
            "u": state.input_output.u.field[:],
            "v": state.input_output.v.field[:],
            "local_u_cloud_levels": locals.u_cloud_levels.field[:],
            "local_v_cloud_levels": locals.v_cloud_levels.field[:],
            "local_env_moist_static_energy_modified": locals.environment_moist_static_energy_modified.field[
                :
            ],
            "local_env_moist_static_energy_cloud_levels_modified": locals.environment_moist_static_energy_cloud_levels_modified.field[
                :
            ],
            "local_env_saturation_moist_static_energy_modified": locals.environment_saturation_moist_static_energy_modified.field[
                :
            ],
            "local_env_saturation_moist_static_energy_cloud_levels_modified": locals.environment_saturation_moist_static_energy_cloud_levels_modified.field[
                :
            ],
            "local_geopotential_height_modified": locals.geopotential_height_modified.field[
                :
            ],
            "local_geopotential_height_cloud_levels_modified": locals.geopotential_height_cloud_levels_modified.field[
                :
            ],
            "topography_height_no_negative": state.input_output.topography_height_no_negative.field[
                :
            ],
            "local_gamma_cloud_levels": locals.gamma_cloud_levels.field[:],
        }

        return outputs


class TranslateGF2020_CumulusParameterization_EnvironmentCloudLevels_3_shallow(
    TranslateFortranData2Py
):
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
        self.cu_param_constants = data_loader.load(
            "GF2020_CumulusParameterization-constants"
        )

    def compute_func(self, **inputs):
        outputs = self.test_core(
            self.constants, self.cu_param_constants, Plumes.SHALLOW.value, **inputs
        )

        return outputs


class TranslateGF2020_CumulusParameterization_EnvironmentCloudLevels_3_mid(
    TranslateFortranData2Py
):
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
        self.cu_param_constants = data_loader.load(
            "GF2020_CumulusParameterization-constants"
        )

    def compute_func(self, **inputs):
        outputs = self.test_core(
            self.constants, self.cu_param_constants, Plumes.MID.value, **inputs
        )

        return outputs


class TranslateGF2020_CumulusParameterization_EnvironmentCloudLevels_3_deep(
    TranslateFortranData2Py
):
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
        self.cu_param_constants = data_loader.load(
            "GF2020_CumulusParameterization-constants"
        )

    def compute_func(self, **inputs):
        outputs = self.test_core(
            self.constants, self.cu_param_constants, Plumes.DEEP.value, **inputs
        )

        return outputs
