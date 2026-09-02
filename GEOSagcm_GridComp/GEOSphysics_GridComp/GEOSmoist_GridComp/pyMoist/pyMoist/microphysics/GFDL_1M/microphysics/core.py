from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3Config


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

    def __init__(self, config: GFDLMPV3Config):
        # make config visible at runtime
        self.config = config

        # initialize saturation tables


    def __call__(self, *args, **kwds):
        pass
