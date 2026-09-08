import dataclasses

from ndsl import StencilFactory, Local, LocalState, QuantityFactory
from ndsl.dsl.typing import FloatFieldIJ, FloatField, Float
from ndsl.dsl.gt4py import computation, PARALLEL, interval
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.sedimentation import Sedimentation
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.ice_cloud import IceCloud
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.warm_rain import WarmRain
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.subgrid_processes import SubgridProcesses
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals


def update_precip_fluxes(
    surface_precip_ice: FloatFieldIJ,
    surface_precip_liquid: FloatFieldIJ,
    surface_precip_graupel: FloatFieldIJ,
    surface_precip_rain: FloatFieldIJ,
    surface_precip_snow: FloatFieldIJ,
    internal_surface_precip_ice: FloatFieldIJ,
    internal_surface_precip_liquid: FloatFieldIJ,
    internal_surface_precip_graupel: FloatFieldIJ,
    internal_surface_precip_rain: FloatFieldIJ,
    internal_surface_precip_snow: FloatFieldIJ,
    internal_precip_ice: FloatFieldIJ,
    internal_precip_liquid: FloatFieldIJ,
    internal_precip_graupel: FloatFieldIJ,
    internal_precip_rain: FloatFieldIJ,
    internal_precip_snow: FloatFieldIJ,
    # the following are k-interface fields
    non_anvil_large_scale_ice_precip_flux: FloatField,
    non_anvil_large_scale_liquid_precip_flux: FloatField,
    non_anvil_large_scale_graupel_precip_flux: FloatField,
    non_anvil_large_scale_rain_precip_flux: FloatField,
    non_anvil_large_scale_snow_precip_flux: FloatField,
):
    from __externals__ import CONV_FACTOR

    with computation(PARALLEL), interval(...):
        surface_precip_ice = surface_precip_ice + internal_surface_precip_ice * CONV_FACTOR
        surface_precip_liquid = surface_precip_liquid + internal_surface_precip_liquid * CONV_FACTOR
        surface_precip_graupel = surface_precip_graupel + internal_surface_precip_graupel * CONV_FACTOR
        surface_precip_rain = surface_precip_rain + internal_surface_precip_rain * CONV_FACTOR
        surface_precip_snow = surface_precip_snow + internal_surface_precip_snow * CONV_FACTOR

        non_anvil_large_scale_ice_precip_flux = non_anvil_large_scale_ice_precip_flux + internal_precip_ice * CONV_FACTOR
        non_anvil_large_scale_liquid_precip_flux = non_anvil_large_scale_liquid_precip_flux + internal_precip_liquid * CONV_FACTOR
        non_anvil_large_scale_graupel_precip_flux = non_anvil_large_scale_graupel_precip_flux + internal_precip_graupel * CONV_FACTOR
        non_anvil_large_scale_rain_precip_flux = non_anvil_large_scale_rain_precip_flux + internal_precip_rain * CONV_FACTOR
        non_anvil_large_scale_snow_precip_flux = non_anvil_large_scale_snow_precip_flux + internal_precip_snow * CONV_FACTOR


@dataclasses.dataclass
class MPFullLocals(LocalState):
    precip_ice: Local = dataclasses.field(
        metadata={
            "name": "precip_ice",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    precip_liquid: Local = dataclasses.field(
        metadata={
            "name": "precip_liquid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    precip_graupel: Local = dataclasses.field(
        metadata={
            "name": "precip_graupel",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    precip_rain: Local = dataclasses.field(
        metadata={
            "name": "precip_rain",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    precip_snow: Local = dataclasses.field(
        metadata={
            "name": "precip_snow",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    surface_precip_ice: Local = dataclasses.field(
        metadata={
            "name": "surface_precip_ice",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_precip_liquid: Local = dataclasses.field(
        metadata={
            "name": "surface_precip_liquid",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_precip_graupel: Local = dataclasses.field(
        metadata={
            "name": "surface_precip_graupel",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_precip_rain: Local = dataclasses.field(
        metadata={
            "name": "surface_precip_rain",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    surface_precip_snow: Local = dataclasses.field(
        metadata={
            "name": "surface_precip_snow",
            "dims": [I_DIM, J_DIM],
            "dtype": Float,
        }
    )
    terminal_velocity_ice: Local = dataclasses.field(
        metadata={
            "name": "terminal_velocity_ice",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    terminal_velocity_liquid: Local = dataclasses.field(
        metadata={
            "name": "terminal_velocity_liquid",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    terminal_velocity_graupel: Local = dataclasses.field(
        metadata={
            "name": "terminal_velocity_graupel",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    terminal_velocity_rain: Local = dataclasses.field(
        metadata={
            "name": "terminal_velocity_rain",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )
    terminal_velocity_snow: Local = dataclasses.field(
        metadata={
            "name": "terminal_velocity_snow",
            "dims": [I_DIM, J_DIM, K_DIM],
            "dtype": Float,
        }
    )


class MPFull:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
    ):

        # initialize subcomponents
        self._sedimentation = Sedimentation(stencil_factory, quantity_factory, mp_config, mp_namelist, CONV_FACTOR)
        self._update_precip_fluxes = stencil_factory.from_dims_halo(
            func=update_precip_fluxes,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._warm_rain = WarmRain(stencil_factory, mp_config, mp_namelist)
        self._ice_cloud = IceCloud(stencil_factory, mp_config, mp_namelist)
        self._subgrid_processes = SubgridProcesses(stencil_factory, mp_config, mp_namelist)

        # initialize MPFull locals
        self._mp_full_locals = MPFullLocals.make_locals(quantity_factory)

        # make namelist visible at runtime
        self._mp_namelist = mp_namelist

    def __call__(self, state: GFDL1MState, gfdl_1m_locals: GFDL1MLocals, gfdl_mp_v3_locals: GFDLMPV3Locals):
        for f in self._mp_namelist.NTIMES:
            # sedimentation of cloud ice, snow, graupel or hail, and rain
            self._sedimentation(state, gfdl_1m_locals, gfdl_mp_v3_locals, self._mp_full_locals)

            self._update_precip_fluxes(
                surface_precip_ice=state.precipitation_at_surface.ice,
                surface_precip_liquid=state.precipitation_at_surface.water,
                surface_precip_graupel=state.precipitation_at_surface.graupel,
                surface_precip_rain=state.precipitation_at_surface.rain,
                surface_precip_snow=state.precipitation_at_surface.snow,
                internal_surface_precip_ice=self._mp_full_locals.internal_surface_precip_ice,
                internal_surface_precip_liquid=self._mp_full_locals.internal_surface_precip_liquid,
                internal_surface_precip_graupel=self._mp_full_locals.internal_surface_precip_graupel,
                internal_surface_precip_rain=self._mp_full_locals.internal_surface_precip_rain,
                internal_surface_precip_snow=self._mp_full_locals.internal_surface_precip_snow,
                internal_precip_ice=self._mp_full_locals.internal_precip_ice,
                internal_precip_liquid=self._mp_full_locals.internal_precip_liquid,
                internal_precip_graupel=self._mp_full_locals.internal_precip_graupel,
                internal_precip_rain=self._mp_full_locals.internal_precip_rain,
                internal_precip_snow=self._mp_full_locals.internal_precip_snow,
                non_anvil_large_scale_ice_precip_flux=state.non_anvil_large_scale.ice_precip_flux,
                non_anvil_large_scale_liquid_precip_flux=state.non_anvil_large_scale.liquid_precip_flux,
                non_anvil_large_scale_graupel_precip_flux=state.non_anvil_large_scale.graupel_precip_flux,
                non_anvil_large_scale_rain_precip_flux=state.non_anvil_large_scale.rain_precip_flux,
                non_anvil_large_scale_snow_precip_flux=state.non_anvil_large_scale.snow_precip_flux,
            )
