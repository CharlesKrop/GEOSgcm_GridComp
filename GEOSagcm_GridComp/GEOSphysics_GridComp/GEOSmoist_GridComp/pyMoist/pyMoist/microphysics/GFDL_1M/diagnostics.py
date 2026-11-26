from ndsl import NDSLRuntime, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.stencils.basic_operations import add, copy

from pyMoist.microphysics.gfdl_1m.optimization import get_optimization_config
from pyMoist.microphysics.gfdl_1m.state import GFDL1MState
from pyMoist.microphysics.gfdl_1m.locals import GFDL1MLocals
from pyMoist.shared.atmos_recipes import dissipative_kinetic_energy_heating


def compute_ice_water_path(
    convective_ice: FloatField,
    large_scale_ice: FloatField,
    graupel: FloatField,
    snow: FloatField,
    mass: FloatField,
    ice_water_path: FloatFieldIJ,
):
    """Compute ice water path throughout a column."""
    with computation(FORWARD), interval(0, 1):
        ice_water_path = 0.0

    with computation(FORWARD), interval(...):
        ice_water_path += (convective_ice + large_scale_ice + snow + graupel) * mass


def compute_liquid_water_path(
    convective_liquid: FloatField,
    large_scale_liquid: FloatField,
    rain: FloatField,
    mass: FloatField,
    liquid_water_path: FloatFieldIJ,
):
    """Compute liquid water path throughout a column."""
    with computation(FORWARD), interval(0, 1):
        liquid_water_path = 0.0

    with computation(FORWARD), interval(...):
        liquid_water_path += (convective_liquid + large_scale_liquid + rain) * mass


class GFDL1MDiagnostics(NDSLRuntime):
    """Compute GFDL 1M tendencies and state diagnostics."""

    def __init__(self, stencil_factory: StencilFactory, locals: GFDL1MLocals):
        super().__init__(stencil_factory)
        self._locals = locals

        self._add = stencil_factory.from_dims_halo(func=add, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._compute_ice_water_path = stencil_factory.from_dims_halo(
            func=compute_ice_water_path,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._compute_liquid_water_path = stencil_factory.from_dims_halo(
            func=compute_liquid_water_path,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        self._copy = stencil_factory.from_dims_halo(func=copy, compute_dims=[I_DIM, J_DIM, K_DIM])
        self._dissipative_kinetic_energy_heating = stencil_factory.from_dims_halo(
            func=dissipative_kinetic_energy_heating,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

    def __call__(self, state: GFDL1MState):
        """Compute various diagnostic terms outputs."""
        if state.large_scale_rainwater_source is not None:
            self._add(
                summand_1=state.tendencies.drain_dt_macro,
                summand_2=state.tendencies.drain_dt_micro,
                sum=state.large_scale_rainwater_source,
            )

        if state.tendencies.dt_dt_friction_pressure_weighted is not None:
            self._add(
                summand_1=state.tendencies.du_dt_macro,
                summand_2=state.tendencies.du_dt_micro,
                sum=self._locals.temporary_3d,
            )
            self._add(
                summand_1=state.tendencies.dv_dt_macro,
                summand_2=state.tendencies.dv_dt_micro,
                sum=self._locals.temporary_3d_2,
            )
            self._dissipative_kinetic_energy_heating(
                mass=self._locals.mass,
                u=self._locals.u_unmodified,
                v=self._locals.v_unmodified,
                du=self._locals.temporary_3d,
                dv=self._locals.temporary_3d_2,
                dt=state.tendencies.dt_dt_friction_pressure_weighted,
            )

        # Radar diagnostics are not implemented in the NDSL port yet.
        option_not_implemented = True

        if state.mass_fraction.suspended_graupel is not None:
            self._copy(input=state.mixing_ratio.graupel, output=state.mass_fraction.suspended_graupel)
        if state.mass_fraction.suspended_rain is not None:
            self._copy(input=state.mixing_ratio.rain, output=state.mass_fraction.suspended_rain)
        if state.mass_fraction.suspended_snow is not None:
            self._copy(input=state.mixing_ratio.snow, output=state.mass_fraction.suspended_snow)

        if state.ice_water_path is not None:
            self._compute_ice_water_path(
                convective_ice=state.mixing_ratio.convective_ice,
                large_scale_ice=state.mixing_ratio.large_scale_ice,
                graupel=state.mixing_ratio.graupel,
                snow=state.mixing_ratio.snow,
                mass=self._locals.mass,
                ice_water_path=state.ice_water_path,
            )

        if state.liquid_water_path is not None:
            self._compute_liquid_water_path(
                convective_liquid=state.mixing_ratio.convective_liquid,
                large_scale_liquid=state.mixing_ratio.large_scale_liquid,
                rain=state.mixing_ratio.rain,
                mass=self._locals.mass,
                liquid_water_path=state.liquid_water_path,
            )
