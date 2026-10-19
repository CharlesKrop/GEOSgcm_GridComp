from ndsl import QuantityFactory, StencilFactory, NDSLRuntime
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval, FORWARD
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ

from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.mp_full_locals import MPFullLocals
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.ice_cloud import IceCloud
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.sedimentation import Sedimentation
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.subgrid_processes import SubgridProcesses
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.warm_rain import WarmRain
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import GFDLMPV3Tables
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.optimization import get_optimization_config


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
    internal_precip_ice: FloatField,
    internal_precip_liquid: FloatField,
    internal_precip_graupel: FloatField,
    internal_precip_rain: FloatField,
    internal_precip_snow: FloatField,
    # the following are k-interface fields
    non_anvil_large_scale_ice_precip_flux: FloatField,
    non_anvil_large_scale_liquid_precip_flux: FloatField,
    non_anvil_large_scale_graupel_precip_flux: FloatField,
    non_anvil_large_scale_rain_precip_flux: FloatField,
    non_anvil_large_scale_snow_precip_flux: FloatField,
):
    from __externals__ import CONV_FACTOR

    with computation(FORWARD), interval(0, 1):
        surface_precip_ice = surface_precip_ice + internal_surface_precip_ice * CONV_FACTOR
        surface_precip_liquid = surface_precip_liquid + internal_surface_precip_liquid * CONV_FACTOR
        surface_precip_graupel = surface_precip_graupel + internal_surface_precip_graupel * CONV_FACTOR
        surface_precip_rain = surface_precip_rain + internal_surface_precip_rain * CONV_FACTOR
        surface_precip_snow = surface_precip_snow + internal_surface_precip_snow * CONV_FACTOR

    with computation(FORWARD), interval(...):
        # all LHS fields are k-interface fields, while RHS (internal_precip_*) are k-center fields
        # do not update bottom layer (TOA), offset all read/writes up one
        non_anvil_large_scale_ice_precip_flux[0, 0, 1] = non_anvil_large_scale_ice_precip_flux[0, 0, 1] + internal_precip_ice * CONV_FACTOR
        non_anvil_large_scale_liquid_precip_flux[0, 0, 1] = non_anvil_large_scale_liquid_precip_flux[0, 0, 1] + internal_precip_liquid * CONV_FACTOR
        non_anvil_large_scale_graupel_precip_flux[0, 0, 1] = non_anvil_large_scale_graupel_precip_flux[0, 0, 1] + internal_precip_graupel * CONV_FACTOR
        non_anvil_large_scale_rain_precip_flux[0, 0, 1] = non_anvil_large_scale_rain_precip_flux[0, 0, 1] + internal_precip_rain * CONV_FACTOR
        non_anvil_large_scale_snow_precip_flux[0, 0, 1] = non_anvil_large_scale_snow_precip_flux[0, 0, 1] + internal_precip_snow * CONV_FACTOR


class MPFull(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        saturation_tables: GFDLMPV3Tables,
        mp_config: GFDLMPV3CloudMPConfig,
        mp_namelist: GFDLMPV3NamelistConfig,
        CONV_FACTOR: Float,
        DRIVER_DT: Float,
    ):
        # initialize NDSLRuntime parent class
        super().__init__(stencil_factory, get_optimization_config(stencil_factory))

        # initialize subcomponents
        self._sedimentation = Sedimentation(stencil_factory, quantity_factory, mp_config, mp_namelist, DRIVER_DT)
        self._update_precip_fluxes = stencil_factory.from_dims_halo(
            func=update_precip_fluxes,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"CONV_FACTOR": CONV_FACTOR},
        )
        self._warm_rain = WarmRain(stencil_factory, quantity_factory, saturation_tables, mp_config, mp_namelist, CONV_FACTOR, DRIVER_DT)
        self._ice_cloud = IceCloud(stencil_factory, quantity_factory, saturation_tables, mp_config, mp_namelist, CONV_FACTOR, DRIVER_DT)
        self._subgrid_processes = SubgridProcesses(stencil_factory, quantity_factory, saturation_tables, mp_config, mp_namelist, CONV_FACTOR, DRIVER_DT)

        # initialize MPFull locals
        self._mp_full_locals = MPFullLocals.make_locals(quantity_factory)

        # make namelist visible at runtime
        self._mp_namelist = mp_namelist

    def __call__(self, state: GFDL1MState, gfdl_mp_v3_locals: GFDLMPV3Locals):
        for f in range(self._mp_namelist.NTIMES):
            # --------------------------------------------------
            # sedimentation of cloud ice, snow, graupel or hail, and rain
            # --------------------------------------------------
            self._sedimentation(gfdl_mp_v3_locals, self._mp_full_locals)

            self._update_precip_fluxes(
                surface_precip_ice=state.precipitation_at_surface.ice,
                surface_precip_liquid=state.precipitation_at_surface.water,
                surface_precip_graupel=state.precipitation_at_surface.graupel,
                surface_precip_rain=state.precipitation_at_surface.rain,
                surface_precip_snow=state.precipitation_at_surface.snow,
                internal_surface_precip_ice=self._mp_full_locals.surface_precip.ice,
                internal_surface_precip_liquid=self._mp_full_locals.surface_precip.liquid,
                internal_surface_precip_graupel=self._mp_full_locals.surface_precip.graupel,
                internal_surface_precip_rain=self._mp_full_locals.surface_precip.rain,
                internal_surface_precip_snow=self._mp_full_locals.surface_precip.snow,
                internal_precip_ice=self._mp_full_locals.precip.ice,
                internal_precip_liquid=self._mp_full_locals.precip.liquid,
                internal_precip_graupel=self._mp_full_locals.precip.graupel,
                internal_precip_rain=self._mp_full_locals.precip.rain,
                internal_precip_snow=self._mp_full_locals.precip.snow,
                non_anvil_large_scale_ice_precip_flux=state.non_anvil_large_scale.ice_precip_flux,
                non_anvil_large_scale_liquid_precip_flux=state.non_anvil_large_scale.liquid_precip_flux,
                non_anvil_large_scale_graupel_precip_flux=state.non_anvil_large_scale.graupel_precip_flux,
                non_anvil_large_scale_rain_precip_flux=state.non_anvil_large_scale.rain_precip_flux,
                non_anvil_large_scale_snow_precip_flux=state.non_anvil_large_scale.snow_precip_flux,
            )

            # --------------------------------------------------
            # warm rain cloud microphysics
            # --------------------------------------------------
            self._warm_rain(state, gfdl_mp_v3_locals, self._mp_full_locals)

            # --------------------------------------------------
            # ice cloud microphysics
            # --------------------------------------------------
            self._ice_cloud(gfdl_mp_v3_locals, self._mp_full_locals)

            # --------------------------------------------------
            # temperature sensitive high vertical resolution processes
            # --------------------------------------------------
            if self._mp_namelist.DO_SUBGRID_PROC:
                self._subgrid_processes(state, gfdl_mp_v3_locals)
