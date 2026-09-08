from ndsl import StencilFactory

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig


class WarmRain:
    def __init__(self, stencil_factory: StencilFactory, mp_config: GFDLMPV3CloudMPConfig, mp_namelist: GFDLMPV3NamelistConfig):
        pass

    def __call__(self, *args, **kwds):
        pass
