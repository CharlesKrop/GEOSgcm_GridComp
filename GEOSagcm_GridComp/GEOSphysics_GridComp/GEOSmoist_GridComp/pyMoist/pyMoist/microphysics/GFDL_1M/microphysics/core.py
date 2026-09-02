import os
import dataclasses

import f90nml

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3Config
from pyMoist.microphysics.GFDL_1M.microphysics import constants
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import get_saturation_vapor_pressure_tables
from ndsl import StencilFactory, ndsl_log


class GFDLMPV3:
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

    def __init__(self, stencil_factory: StencilFactory, config: GFDLMPV3Config, namelist: str = "input.nml"):
        # make config visible at runtime
        self.config = config

        # read namelist
        full_nml_path = os.path.join(self.config.cwd, namelist)
        self._read_namelist(full_nml_path)

        # initialize saturation tables
        self.saturation_tables = get_saturation_vapor_pressure_tables(stencil_factory=stencil_factory)

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
                setattr(self.config, name, field.type(value))
            else:
                # not overridden in the namelist - fall back to the default in constants.py
                if not hasattr(constants, "_" + name):
                    ndsl_log.error(f"[GFDL1M Microphysics]: '{name}' does not have a fallback value specified, it must be included in the namelist")
                    # NOTE can this error message be incorporated directly into exc_info?
                    raise AttributeError(f"[GFDL1M Microphysics]: '{name}' does not have a fallback value specified, it must be included in the namelist")
                setattr(self.config, name, getattr(constants, "_" + name))

    def __call__(self, *args, **kwds):
        pass
