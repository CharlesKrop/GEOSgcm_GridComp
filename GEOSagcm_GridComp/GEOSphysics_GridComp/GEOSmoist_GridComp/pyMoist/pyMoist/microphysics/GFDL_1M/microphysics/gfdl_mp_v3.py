import os
import dataclasses

import f90nml

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3Config
from pyMoist.microphysics.GFDL_1M.microphysics import constants
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import get_saturation_vapor_pressure_tables
from ndsl import StencilFactory, ndsl_log, NDSLRuntime, QuantityFactory
from ndsl.dsl.typing import Float, Float64
from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import set_value_2d
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.driver import GFDLMPV3Driver
from pyMoist.microphysics.GFDL_1M.microphysics.locals import GFDLMPV3Locals


@dataclasses.dataclass
class GFDLMPV3HeatCapacities:
    C_AIR: Float
    C_VAP: Float
    D0_VAP: Float
    D1_ICE: Float
    D1_VAP: Float
    LV00: Float64
    LI00: Float64
    LI20: Float64
    C1_VAP: Float64
    C1_LIQ: Float64
    C1_ICE: Float64

    @classmethod
    def init_to_none(cls) -> "GFDLMPV3HeatCapacities":
        """Create an all-None instance, meant to be fully populated afterward."""
        return cls(**{f.name: None for f in dataclasses.fields(cls)})


class GFDLMPV3(NDSLRuntime):
    """GFDL Cloud Microphysics Package (GFDL MP) Version 3
    The algorithms are originally derived from Lin et al. (1983).
    Most of the key elements have been simplified / improved.
    This code at this stage bears little to no similarity to the original Lin MP in ZETAC.
    Developers: Linjiong Zhou and the GFDL FV3 Team
    References:
    Version 0: Chen and Lin (2011 doi: 10.1029/2011GL047629, 2013 doi: 10.1175/JCLI-D-12-00061.1)
    Version 1: Zhou et al. (2019 doi: 10.1175/BAMS-D-17-0246.1)
    Version 2: Harris et al. (2020 doi: 10.1029/2020MS002223), Zhou et al. (2022 doi: 10.25923/pz3c-8b96)
    Version 3: Zhou et al. (2022 doi: 10.1029/2021MS002971)
    NASA integration: Putman April 2025
    NDSL integration: Kropiewnicki September 2026
    """

    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory, gfdl_1m_config: GFDL1MConfig, namelist: str = "input.nml"):
        # initialize NDSLRuntime parent class
        super.__init__(stencil_factory)

        # make overatching GFDL1M config visible throughout the class
        self._gfdl_1m_config = gfdl_1m_config

        # initialize data classes - config will be updated in read_namelist and heat_capacities will be updated in setup_heat_capacities
        self._mp_config = GFDLMPV3Config.init_to_none()
        self._mp_heat_capacities = GFDLMPV3HeatCapacities.init_to_none()

        # read namelist and initialize configuration - must be done before heat capacities are initialized
        full_nml_path = os.path.join(self._gfdl_1m_config.CWD, namelist)
        self._read_namelist(full_nml_path)

        # initialize heat capacities
        self._setup_heat_capacities()

        # initialize saturation tables
        self._saturation_tables = get_saturation_vapor_pressure_tables(stencil_factory=stencil_factory)

        # initialize locals
        self._mp_locals = GFDLMPV3Locals.make_locals(quantity_factory)

        # construct stencil
        self._set_value_k_interface = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._set_value = stencil_factory.from_dims_halo(
            func=set_value,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._set_value_2d = stencil_factory.from_dims_halo(
            func=set_value_2d,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # initialize the driver
        self._driver = GFDLMPV3Driver(stencil_factory)

    def _read_namelist(self, nml_path: str):
        """Populate config from the gfdl_mp_nml group of a Fortran namelist file.

        For every field defined on GFDLMPV3Config:
          - if the field is present in the gfdl_mp_nml namelist group, use that value
          - otherwise, fall back to the default constant of the same name in constants.py

        This mirrors the Fortran behavior where namelist variables that aren't set in
        the file simply retain their pre-initialized default value.
        """
        if not os.path.isfile(nml_path):
            ndsl_log.error(f"[GFDL1M Microphysics] namelist file: {nml_path} does not exist")
            # NOTE can this error message be incorporated directly into exc_info?
            raise FileNotFoundError(f"namelist file: {nml_path} does not exist")

        try:
            full_nml = f90nml.read(nml_path)
        except Exception as e:
            ndsl_log.error(f"[GFDL1M Microphysics]: namelist exists at {nml_path} but read failed, bailing out", exc_info=e)

        # f90nml lowercases group and key names by default
        mp_nml = full_nml.get("gfdl_mp_nml", {})
        ndsl_log.info(f"[GFDL1M Microphysics]: full microphysics namelist:\n{mp_nml}")

        for field in dataclasses.fields(GFDLMPV3Config):
            name = field.name
            key = name.lower()

            if key in mp_nml:
                # value came from the namelist file - cast to the declared field type
                value = mp_nml[key]
                setattr(self._mp_config, name, field.type(value))
            else:
                # not overridden in the namelist - fall back to the default in constants.py
                if not hasattr(constants, "_" + name):
                    ndsl_log.error(f"[GFDL1M Microphysics]: '{name}' does not have a fallback value specified, it must be included in the namelist")
                    # NOTE can this error message be incorporated directly into exc_info?
                    raise AttributeError(f"[GFDL1M Microphysics]: '{name}' does not have a fallback value specified, it must be included in the namelist")
                setattr(self._mp_config, name, getattr(constants, "_" + name))

    def _setup_heat_capacities(self):
        if self._gfdl_1m_config.LHYDROSTATIC:
            self._mp_heat_capacities.C_AIR = constants.CP_AIR
            self._mp_heat_capacities.C_VAP = constants.CP_VAP
            self._mp_config.DO_SEDI_W = False
        else:
            self._mp_heat_capacities.C_AIR = constants.CV_AIR
            self._mp_heat_capacities.C_VAP = constants.CV_VAP

        # scaled constants (to reduce float point errors for 32-bit)

        self._mp_heat_capacities.D1_VAP = self._mp_heat_capacities.D0_VAP / self._mp_heat_capacities.C_AIR
        self._mp_heat_capacities.D1_ICE = constants.DC_ICE / self._mp_heat_capacities.C_AIR

        if self._gfdl_1m_config.LHYDROSTATIC:
            LV00 = (constants.HLV - self._mp_heat_capacities.D0_VAP * constants.TICE) / self._mp_heat_capacities.C_AIR
        else:
            if self._gfdl_1m_config.ENG_CNV_OLD:
                LV00 = (constants.HLV - self._mp_heat_capacities.D0_VAP * constants.TICE) / self._mp_heat_capacities.C_AIR
            else:
                LV00 = (constants.HLV - self._mp_heat_capacities.D0_VAP * constants.TICE - constants.RVGAS * constants.TICE) / self._mp_heat_capacities.C_AIR

        self._mp_heat_capacities.LI00 = (constants.HLF - constants.DC_ICE * constants.TICE) / self._mp_heat_capacities.C_AIR
        self._mp_heat_capacities.LI20 = LV00 + self._mp_heat_capacities.LI00

        self._mp_heat_capacities.C1_VAP = self._mp_heat_capacities.C_VAP / self._mp_heat_capacities.C_AIR
        self._mp_heat_capacities.C1_LIQ = constants.C_LIQ / self._mp_heat_capacities.C_AIR
        self._mp_heat_capacities.C1_ICE = constants.C_ICE / self._mp_heat_capacities.C_AIR

    def __call__(self, state: GFDL1MState, locals: GFDL1MLocals):
        # reset state fields to zero
        self._set_value_2d(field=state.precipitation_at_surface.water, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.rain, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.snow, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.ice, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.graupel, value=Float(0.0))

        # reset gfdl locals to zero
        self._set_value_2d(field=locals.dcondensatedt, value=Float(0.0))
        self._set_value(field=state.non_anvil_large_scale.evaporation, value=Float(0.0))
        self._set_value(field=state.non_anvil_large_scale.sublimation, value=Float(0.0))
        self._set_value_k_interface

        # reset mp locals to zero
        self._set_value(field=self._mp_locals.mppcw, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppew, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppe1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpper, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppdi, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppd1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppds, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppdg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppsi, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpps1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppss, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppsg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppfw, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppfr, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppar, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppas, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppag, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpprs, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mpprg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppxr, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppxs, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppxg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppmi, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppms, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppmg, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppm1, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppm2, value=Float(0, 0))
        self._set_value(field=self._mp_locals.mppm3, value=Float(0, 0))
