from ndsl import StencilFactory
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.sedimentation import Sedimentation
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.ice_cloud import IceCloud
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.warm_rain import WarmRain
from pyMoist.microphysics.GFDL_1M.microphysics.mp_full.subgrid_processes import SubgridProcesses
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.microphysics.driver import GFDLMPV3Locals

def 


class MPFull:
    def __init__(self, stencil_factory: StencilFactory, mp_config: GFDLMPV3CloudMPConfig, mp_namelist: GFDLMPV3NamelistConfig):
        # initialize subclasses
        self._sedimentation = Sedimentation(stencil_factory, mp_config, mp_namelist)
        self._warm_rain = WarmRain(stencil_factory, mp_config, mp_namelist)
        self._ice_cloud = IceCloud(stencil_factory, mp_config, mp_namelist)
        self._subgrid_processes = SubgridProcesses(stencil_factory, mp_config, mp_namelist)

    def __call__(self, ):
        pass
