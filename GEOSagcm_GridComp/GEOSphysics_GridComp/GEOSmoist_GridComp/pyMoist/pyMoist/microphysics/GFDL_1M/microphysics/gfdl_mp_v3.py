import os
from math import exp, gamma, log, sqrt

import numpy as np
from ndsl import NDSLRuntime, QuantityFactory, StencilFactory, ndsl_log
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import Float, Int
from ndsl.stencils.basic_operations import set_value
from ndsl.stencils.basic_operations_2d import set_value_2d
from ndsl.dsl.gt4py import float64

from pyMoist.microphysics.GFDL_1M.config import GFDL1MConfig
from pyMoist.microphysics.GFDL_1M.locals import GFDL1MLocals
from pyMoist.microphysics.GFDL_1M.optimization import get_optimization_config
from pyMoist.microphysics.GFDL_1M.microphysics import constants
from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3CloudMPConfig, GFDLMPV3NamelistConfig
from pyMoist.microphysics.GFDL_1M.microphysics.driver import GFDLMPV3Driver
from pyMoist.microphysics.GFDL_1M.microphysics.saturation_tables import get_saturation_vapor_pressure_tables
from pyMoist.microphysics.GFDL_1M.state import GFDL1MState


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

    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory, gfdl_1m_config: GFDL1MConfig, lid_level: Int, namelist: str = "input.nml"):
        # initialize NDSLRuntime parent class
        super().__init__(stencil_factory, get_optimization_config(stencil_factory))

        # make overatching GFDL1M config visible throughout the class
        self._gfdl_1m_config = gfdl_1m_config

        # initialize namelist parameters. will be read from namelist where available, otherwise use defaults
        self._mp_namelist = GFDLMPV3NamelistConfig.init_from_namelist(os.path.join(gfdl_1m_config.CWD, namelist), dt=gfdl_1m_config.DT)

        # initialize cloud microphysics configuration
        # NOTE currently a two step process so that it is abundantly clear that the calculations required to fully initialize
        # this class are dependent on the namelist and the GFDL1M configuration (i.e. they must be initialized first)
        # open to discussion on merging into a one step initialization
        self._mp_config = GFDLMPV3CloudMPConfig.init_to_none()
        self._setup_cloud_mp_config(quantity_factory, gfdl_1m_config, self._mp_namelist, self._mp_config)

        # initialize saturation tables
        _saturation_tables = get_saturation_vapor_pressure_tables(stencil_factory=stencil_factory)

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
        self._driver = GFDLMPV3Driver(
            # if GFDL_MP_KLID is non-negative (default value is -999), microphysics will be restricted to start at that vertical level
            stencil_factory=stencil_factory.restrict_vertical(k_start=lid_level),
            quantity_factory=quantity_factory,
            saturation_tables=_saturation_tables,
            gfdl_1m_config=self._gfdl_1m_config,
            mp_namelist=self._mp_namelist,
            mp_config=self._mp_config,
        )

    def _setup_cloud_mp_config(
        self, quantity_factory: QuantityFactory, gfdl_1m_config: GFDL1MConfig, mp_namelist: GFDLMPV3NamelistConfig, mp_config: GFDLMPV3CloudMPConfig
    ):

        # construct quantities for the tables: these must be given a data dimension and defined as a
        # quantity so that they can be brought into stencils as GlobalTables
        quantity_factory.update_data_dimensions({"LEN2_TABLE": 2})
        quantity_factory.update_data_dimensions({"LEN3_TABLE": 3})
        quantity_factory.update_data_dimensions({"LEN4_TABLE": 4})
        quantity_factory.update_data_dimensions({"LEN5_TABLE": 5})
        quantity_factory.update_data_dimensions({"LEN10_TABLE": 10})
        quantity_factory.update_data_dimensions({"LEN20_TABLE": 20})

        # NOTE this was taken from the old microphysics, which worked on GPU
        # but how did this work? isn't np.array incompatable? shouldn't this be cupy/xumpy?
        # did the init always happen on cpu?
        mp_config.ACC = quantity_factory.from_array(np.zeros(20, dtype=Float), ["LEN20_TABLE"], "n/a")
        mp_config.ACCO = quantity_factory.from_array(np.zeros((3, 10), dtype=Float), ["LEN3_TABLE", "LEN10_TABLE"], "n/a")
        mp_config.CGFR = quantity_factory.from_array(np.zeros(2, dtype=Float), ["LEN2_TABLE"], "n/a")
        mp_config.CGMLT = quantity_factory.from_array(np.zeros(4, dtype=Float), ["LEN4_TABLE"], "n/a")
        mp_config.CGSUB = quantity_factory.from_array(np.zeros(5, dtype=Float), ["LEN5_TABLE"], "n/a")
        mp_config.CREVP = quantity_factory.from_array(np.zeros(5, dtype=Float), ["LEN5_TABLE"], "n/a")
        mp_config.CSMLT = quantity_factory.from_array(np.zeros(4, dtype=Float), ["LEN4_TABLE"], "n/a")
        mp_config.CSSUB = quantity_factory.from_array(np.zeros(5, dtype=Float), ["LEN5_TABLE"], "n/a")

        # generate numpy arrays for internal tables
        ace = np.zeros(20, dtype=Float)
        act = np.zeros(20, dtype=Float)
        occ = np.zeros(3, dtype=Float)

        # these options are inputs in the original GFDL version
        # GEOS does not support them, so set them manually here
        mp_config.CONSV_TE = False
        mp_config.DO_INLINE_MP = False
        mp_config.DO_MP_FAST = False
        mp_config.DO_MP_FULL = True
        mp_config.LAST_STEP = True
        mp_config.USE_COND = True
        mp_config.MOIST_KAPPA = True

        # --------------------------------------------------
        # heat capacities and related parameters
        # --------------------------------------------------
        if gfdl_1m_config.PHYS_HYDROSTATIC:
            mp_config.C_AIR = constants.CP_AIR
            mp_config.C_VAP = constants.CP_VAP
            mp_namelist.DO_SEDI_W = False
        else:
            mp_config.C_AIR = constants.CV_AIR
            mp_config.C_VAP = constants.CV_VAP
        d0_vap = mp_config.C_VAP - constants.C_LIQ

        # scaled constants (to reduce float point errors for 32-bit)

        mp_config.D1_VAP = float64(d0_vap / mp_config.C_AIR)
        mp_config.D1_ICE = float64(constants.DC_ICE / mp_config.C_AIR)

        if gfdl_1m_config.PHYS_HYDROSTATIC:
            mp_config.LV00 = float64((constants.HLV - d0_vap * constants.TICE) / mp_config.C_AIR)
        else:
            mp_config.LV00 = float64((constants.HLV - d0_vap * constants.TICE - constants.RVGAS * constants.TICE) / mp_config.C_AIR)

        mp_config.LI00 = float64((constants.HLF - constants.DC_ICE * constants.TICE) / mp_config.C_AIR)
        mp_config.LI20 = float64(mp_config.LV00 + mp_config.LI00)

        mp_config.C1_VAP = float64(mp_config.C_VAP / mp_config.C_AIR)
        mp_config.C1_LIQ = float64(constants.C_LIQ / mp_config.C_AIR)
        mp_config.C1_ICE = float64(constants.C_ICE / mp_config.C_AIR)

        # --------------------------------------------------
        # all other internal configuration parameters
        # --------------------------------------------------

        # complete freezing temperature

        if mp_namelist.DO_WARM_RAIN_MP:
            mp_config.T_WFR = mp_namelist.T_MIN
        else:
            mp_config.T_WFR = constants.TICE - Float(40.0)

        # cloud water autoconversion, Hong et al. (2004)

        aone = Float(2.0) / Float(9.0) * (Float(3.0) / Float(4.0)) ** (Float(4.0) / Float(3.0)) / constants.PI ** (Float(1.0) / Float(3.0))
        mp_config.CPAUT0 = float64(mp_namelist.C_PAUT * aone * constants.GRAV / constants.VISD)

        # terminal velocities parameters, Lin et al. (1983)

        gcon = (Float(4.0) * constants.GRAV * constants.RHOG / (Float(3.0) * constants.CDG * constants.RHO0)) ** Float(0.5)
        hcon = (Float(4.0) * constants.GRAV * constants.RHOH / (Float(3.0) * constants.CDH * constants.RHO0)) ** Float(0.5)

        # part of the slope parameters

        mp_config.NORMW = float64(constants.PI * constants.RHOW * mp_namelist.N0W_SIG * gamma(mp_namelist.MUW + Int(3)))
        mp_config.NORMI = float64(constants.PI * constants.RHOI * mp_namelist.N0I_SIG * gamma(mp_namelist.MUI + Int(3)))
        mp_config.NORMR = float64(constants.PI * constants.RHOR * mp_namelist.N0R_SIG * gamma(mp_namelist.MUR + Int(3)))
        mp_config.NORMS = float64(constants.PI * constants.RHOS * mp_namelist.N0S_SIG * gamma(mp_namelist.MUS + Int(3)))
        mp_config.NORMG = float64(constants.PI * constants.RHOG * mp_namelist.N0G_SIG * gamma(mp_namelist.MUG + Int(3)))
        mp_config.NORMH = float64(constants.PI * constants.RHOH * mp_namelist.N0H_SIG * gamma(mp_namelist.MUH + Int(3)))

        mp_config.EXPOW = float64(exp(mp_namelist.N0W_EXP / (mp_namelist.MUW + Int(3)) * log(Float(10.0))))
        mp_config.EXPOI = float64(exp(mp_namelist.N0I_EXP / (mp_namelist.MUI + Int(3)) * log(Float(10.0))))
        mp_config.EXPOR = float64(exp(mp_namelist.N0R_EXP / (mp_namelist.MUR + Int(3)) * log(Float(10.0))))
        mp_config.EXPOS = float64(exp(mp_namelist.N0S_EXP / (mp_namelist.MUS + Int(3)) * log(Float(10.0))))
        mp_config.EXPOG = float64(exp(mp_namelist.N0G_EXP / (mp_namelist.MUG + Int(3)) * log(Float(10.0))))
        mp_config.EXPOH = float64(exp(mp_namelist.N0H_EXP / (mp_namelist.MUH + Int(3)) * log(Float(10.0))))

        # parameters for particle concentration (pc), effective diameter (ed), optical extinction (oe),
        # radar reflectivity factor (rr), and mass-weighted terminal velocity (tv)

        mp_config.PCAW = float64(
            exp(Int(3) / (mp_namelist.MUW + Int(3)) * log(mp_namelist.N0W_SIG))
            * gamma(mp_namelist.MUW)
            * exp(Int(3) * mp_namelist.N0W_EXP / (mp_namelist.MUW + Int(3)) * log(Float(10.0)))
        )
        mp_config.PCAI = float64(
            exp(Int(3) / (mp_namelist.MUI + Int(3)) * log(mp_namelist.N0I_SIG))
            * gamma(mp_namelist.MUI)
            * exp(Int(3) * mp_namelist.N0I_EXP / (mp_namelist.MUI + Int(3)) * log(Float(10.0)))
        )
        mp_config.PCAR = float64(
            exp(Int(3) / (mp_namelist.MUR + Int(3)) * log(mp_namelist.N0R_SIG))
            * gamma(mp_namelist.MUR)
            * exp(Int(3) * mp_namelist.N0R_EXP / (mp_namelist.MUR + Int(3)) * log(Float(10.0)))
        )
        mp_config.PCAS = float64(
            exp(Int(3) / (mp_namelist.MUS + Int(3)) * log(mp_namelist.N0S_SIG))
            * gamma(mp_namelist.MUS)
            * exp(Int(3) * mp_namelist.N0S_EXP / (mp_namelist.MUS + Int(3)) * log(Float(10.0)))
        )
        mp_config.PCAG = float64(
            exp(Int(3) / (mp_namelist.MUG + Int(3)) * log(mp_namelist.N0G_SIG))
            * gamma(mp_namelist.MUG)
            * exp(Int(3) * mp_namelist.N0G_EXP / (mp_namelist.MUG + Int(3)) * log(Float(10.0)))
        )
        mp_config.PCAH = float64(
            exp(Int(3) / (mp_namelist.MUH + Int(3)) * log(mp_namelist.N0H_SIG))
            * gamma(mp_namelist.MUH)
            * exp(Int(3) * mp_namelist.N0H_EXP / (mp_namelist.MUH + Int(3)) * log(Float(10.0)))
        )

        mp_config.PCBW = float64(exp(mp_namelist.MUW / (mp_namelist.MUW + Int(3)) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + Int(3)))))
        mp_config.PCBI = float64(exp(mp_namelist.MUI / (mp_namelist.MUI + Int(3)) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + Int(3)))))
        mp_config.PCBR = float64(exp(mp_namelist.MUR / (mp_namelist.MUR + Int(3)) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + Int(3)))))
        mp_config.PCBS = float64(exp(mp_namelist.MUS / (mp_namelist.MUS + Int(3)) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + Int(3)))))
        mp_config.PCBG = float64(exp(mp_namelist.MUG / (mp_namelist.MUG + Int(3)) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + Int(3)))))
        mp_config.PCBH = float64(exp(mp_namelist.MUH / (mp_namelist.MUH + Int(3)) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + Int(3)))))

        mp_config.EDAW = float64(
            exp(-Float(1.0) / (mp_namelist.MUW + Int(3)) * log(mp_namelist.N0W_SIG))
            * (mp_namelist.MUW + Int(2))
            * exp(-mp_namelist.N0W_EXP / (mp_namelist.MUW + Int(3)) * log(Float(10.0)))
        )
        mp_config.EDAI = float64(
            exp(-Float(1.0) / (mp_namelist.MUI + Int(3)) * log(mp_namelist.N0I_SIG))
            * (mp_namelist.MUI + Int(2))
            * exp(-mp_namelist.N0I_EXP / (mp_namelist.MUI + Int(3)) * log(Float(10.0)))
        )
        mp_config.EDAR = float64(
            exp(-Float(1.0) / (mp_namelist.MUR + Int(3)) * log(mp_namelist.N0R_SIG))
            * (mp_namelist.MUR + Int(2))
            * exp(-mp_namelist.N0R_EXP / (mp_namelist.MUR + Int(3)) * log(Float(10.0)))
        )
        mp_config.EDAS = float64(
            exp(-Float(1.0) / (mp_namelist.MUS + Int(3)) * log(mp_namelist.N0S_SIG))
            * (mp_namelist.MUS + Int(2))
            * exp(-mp_namelist.N0S_EXP / (mp_namelist.MUS + Int(3)) * log(Float(10.0)))
        )
        mp_config.EDAG = float64(
            exp(-Float(1.0) / (mp_namelist.MUG + Int(3)) * log(mp_namelist.N0G_SIG))
            * (mp_namelist.MUG + Int(2))
            * exp(-mp_namelist.N0G_EXP / (mp_namelist.MUG + Int(3)) * log(Float(10.0)))
        )
        mp_config.EDAH = float64(
            exp(-Float(1.0) / (mp_namelist.MUH + Int(3)) * log(mp_namelist.N0H_SIG))
            * (mp_namelist.MUH + Int(2))
            * exp(-mp_namelist.N0H_EXP / (mp_namelist.MUH + Int(3)) * log(Float(10.0)))
        )

        mp_config.EDBW = float64(exp(Float(1.0) / (mp_namelist.MUW + Int(3)) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + Int(3)))))
        mp_config.EDBI = float64(exp(Float(1.0) / (mp_namelist.MUI + Int(3)) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + Int(3)))))
        mp_config.EDBR = float64(exp(Float(1.0) / (mp_namelist.MUR + Int(3)) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + Int(3)))))
        mp_config.EDBS = float64(exp(Float(1.0) / (mp_namelist.MUS + Int(3)) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + Int(3)))))
        mp_config.EDBG = float64(exp(Float(1.0) / (mp_namelist.MUG + Int(3)) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + Int(3)))))
        mp_config.EDBH = float64(exp(Float(1.0) / (mp_namelist.MUH + Int(3)) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + Int(3)))))

        mp_config.OEAW = float64(
            exp(Float(1.0) / (mp_namelist.MUW + Int(3)) * log(mp_namelist.N0W_SIG))
            * constants.PI
            * gamma(mp_namelist.MUW + Int(2))
            * exp(mp_namelist.N0W_EXP / (mp_namelist.MUW + Int(3)) * log(10.0))
        )
        mp_config.OEAI = float64(
            exp(Float(1.0) / (mp_namelist.MUI + Int(3)) * log(mp_namelist.N0I_SIG))
            * constants.PI
            * gamma(mp_namelist.MUI + Int(2))
            * exp(mp_namelist.N0I_EXP / (mp_namelist.MUI + Int(3)) * log(10.0))
        )
        mp_config.OEAR = float64(
            exp(Float(1.0) / (mp_namelist.MUR + Int(3)) * log(mp_namelist.N0R_SIG))
            * constants.PI
            * gamma(mp_namelist.MUR + Int(2))
            * exp(mp_namelist.N0R_EXP / (mp_namelist.MUR + Int(3)) * log(10.0))
        )
        mp_config.OEAS = float64(
            exp(Float(1.0) / (mp_namelist.MUS + Int(3)) * log(mp_namelist.N0S_SIG))
            * constants.PI
            * gamma(mp_namelist.MUS + Int(2))
            * exp(mp_namelist.N0S_EXP / (mp_namelist.MUS + Int(3)) * log(10.0))
        )
        mp_config.OEAG = float64(
            exp(Float(1.0) / (mp_namelist.MUG + Int(3)) * log(mp_namelist.N0G_SIG))
            * constants.PI
            * gamma(mp_namelist.MUG + Int(2))
            * exp(mp_namelist.N0G_EXP / (mp_namelist.MUG + Int(3)) * log(10.0))
        )
        mp_config.OEAH = float64(
            exp(Float(1.0) / (mp_namelist.MUH + Int(3)) * log(mp_namelist.N0H_SIG))
            * constants.PI
            * gamma(mp_namelist.MUH + Int(2))
            * exp(mp_namelist.N0H_EXP / (mp_namelist.MUH + Int(3)) * log(10.0))
        )

        mp_config.OEBW = float64(
            Int(2) * exp((mp_namelist.MUW + Int(2)) / (mp_namelist.MUW + Int(3)) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + Int(3))))
        )
        mp_config.OEBI = float64(
            Int(2) * exp((mp_namelist.MUI + Int(2)) / (mp_namelist.MUI + Int(3)) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + Int(3))))
        )
        mp_config.OEBR = float64(
            Int(2) * exp((mp_namelist.MUR + Int(2)) / (mp_namelist.MUR + Int(3)) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + Int(3))))
        )
        mp_config.OEBS = float64(
            Int(2) * exp((mp_namelist.MUS + Int(2)) / (mp_namelist.MUS + Int(3)) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + Int(3))))
        )
        mp_config.OEBG = float64(
            Int(2) * exp((mp_namelist.MUG + Int(2)) / (mp_namelist.MUG + Int(3)) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + Int(3))))
        )
        mp_config.OEBH = float64(
            Int(2) * exp((mp_namelist.MUH + Int(2)) / (mp_namelist.MUH + Int(3)) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + Int(3))))
        )

        mp_config.RRAW = float64(
            exp(-Int(3) / (mp_namelist.MUW + Int(3)) * log(mp_namelist.N0W_SIG))
            * gamma(mp_namelist.MUW + Int(6))
            * exp(-Int(3) * mp_namelist.N0W_EXP / (mp_namelist.MUW + Int(3)) * log(Float(10.0)))
        )
        mp_config.RRAI = float64(
            exp(-Int(3) / (mp_namelist.MUI + Int(3)) * log(mp_namelist.N0I_SIG))
            * gamma(mp_namelist.MUI + Int(6))
            * exp(-Int(3) * mp_namelist.N0I_EXP / (mp_namelist.MUI + Int(3)) * log(Float(10.0)))
        )
        mp_config.RRAR = float64(
            exp(-Int(3) / (mp_namelist.MUR + Int(3)) * log(mp_namelist.N0R_SIG))
            * gamma(mp_namelist.MUR + Int(6))
            * exp(-Int(3) * mp_namelist.N0R_EXP / (mp_namelist.MUR + Int(3)) * log(Float(10.0)))
        )
        mp_config.RRAS = float64(
            exp(-Int(3) / (mp_namelist.MUS + Int(3)) * log(mp_namelist.N0S_SIG))
            * gamma(mp_namelist.MUS + Int(6))
            * exp(-Int(3) * mp_namelist.N0S_EXP / (mp_namelist.MUS + Int(3)) * log(Float(10.0)))
        )
        mp_config.RRAG = float64(
            exp(-Int(3) / (mp_namelist.MUG + Int(3)) * log(mp_namelist.N0G_SIG))
            * gamma(mp_namelist.MUG + Int(6))
            * exp(-Int(3) * mp_namelist.N0G_EXP / (mp_namelist.MUG + Int(3)) * log(Float(10.0)))
        )
        mp_config.RRAH = float64(
            exp(-Int(3) / (mp_namelist.MUH + Int(3)) * log(mp_namelist.N0H_SIG))
            * gamma(mp_namelist.MUH + Int(6))
            * exp(-Int(3) * mp_namelist.N0H_EXP / (mp_namelist.MUH + Int(3)) * log(Float(10.0)))
        )

        mp_config.RRBW = float64(exp((mp_namelist.MUW + Int(6)) / (mp_namelist.MUW + Int(3)) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + Int(3)))))
        mp_config.RRBI = float64(exp((mp_namelist.MUI + Int(6)) / (mp_namelist.MUI + Int(3)) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + Int(3)))))
        mp_config.RRBR = float64(exp((mp_namelist.MUR + Int(6)) / (mp_namelist.MUR + Int(3)) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + Int(3)))))
        mp_config.RRBS = float64(exp((mp_namelist.MUS + Int(6)) / (mp_namelist.MUS + Int(3)) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + Int(3)))))
        mp_config.RRBG = float64(exp((mp_namelist.MUG + Int(6)) / (mp_namelist.MUG + Int(3)) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + Int(3)))))
        mp_config.RRBH = float64(exp((mp_namelist.MUH + Int(6)) / (mp_namelist.MUH + Int(3)) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + Int(3)))))

        mp_config.TVAW = float64(
            Float(exp(-mp_namelist.BLINW / (mp_namelist.MUW + Float(3.0)) * Float(log(mp_namelist.N0W_SIG))))
            * mp_namelist.ALINW
            * Float(gamma(mp_namelist.MUW + mp_namelist.BLINW + Float(3.0)))
            * Float(exp(-mp_namelist.BLINW * mp_namelist.N0W_EXP / (mp_namelist.MUW + Float(3.0)) * Float(log(Float(10.0)))))
        )
        mp_config.TVAI = float64(
            Float(exp(-mp_namelist.BLINI / (mp_namelist.MUI + Float(3.0)) * Float(log(mp_namelist.N0I_SIG))))
            * mp_namelist.ALINI
            * Float(gamma(mp_namelist.MUI + mp_namelist.BLINI + Float(3.0)))
            * Float(exp(-mp_namelist.BLINI * mp_namelist.N0I_EXP / (mp_namelist.MUI + Float(3.0)) * Float(log(Float(10.0)))))
        )
        mp_config.TVAR = float64(
            Float(exp(-mp_namelist.BLINR / (mp_namelist.MUR + Float(3.0)) * Float(log(mp_namelist.N0R_SIG))))
            * mp_namelist.ALINR
            * Float(gamma(mp_namelist.MUR + mp_namelist.BLINR + Float(3.0)))
            * Float(exp(-mp_namelist.BLINR * mp_namelist.N0R_EXP / (mp_namelist.MUR + Float(3.0)) * Float(log(Float(10.0)))))
        )
        mp_config.TVAS = float64(
            Float(exp(-mp_namelist.BLINS / (mp_namelist.MUS + Float(3.0)) * Float(log(mp_namelist.N0S_SIG))))
            * mp_namelist.ALINS
            * Float(gamma(mp_namelist.MUS + mp_namelist.BLINS + Float(3.0)))
            * Float(exp(-mp_namelist.BLINS * mp_namelist.N0S_EXP / (mp_namelist.MUS + Float(3.0)) * Float(log(Float(10.0)))))
        )
        mp_config.TVAG = float64(
            Float(exp(-mp_namelist.BLING / (mp_namelist.MUG + Float(3.0)) * Float(log(mp_namelist.N0G_SIG))))
            * mp_namelist.ALING
            * Float(gamma(mp_namelist.MUG + mp_namelist.BLING + Float(3.0)))
            * Float(exp(-mp_namelist.BLING * mp_namelist.N0G_EXP / (mp_namelist.MUG + Float(3.0)) * Float(log(Float(10.0)))))
            * gcon
        )
        mp_config.TVAH = float64(
            Float(exp(-mp_namelist.BLINH / (mp_namelist.MUH + Float(3.0)) * Float(log(mp_namelist.N0H_SIG))))
            * mp_namelist.ALINH
            * Float(gamma(mp_namelist.MUH + mp_namelist.BLINH + Float(3.0)))
            * Float(exp(-mp_namelist.BLINH * mp_namelist.N0H_EXP / (mp_namelist.MUH + Float(3.0)) * Float(log(Float(10.0)))))
            * hcon
        )

        mp_config.TVBW = float64(
            Float(exp(mp_namelist.BLINW / (mp_namelist.MUW + Float(3.0)) * Float(log(constants.PI * constants.RHOW * Float(gamma(mp_namelist.MUW + Float(3.0)))))))
            * Float(gamma(mp_namelist.MUW + Float(3.0)))
        )
        mp_config.TVBI = float64(
            Float(exp(mp_namelist.BLINI / (mp_namelist.MUI + Float(3.0)) * Float(log(constants.PI * constants.RHOI * Float(gamma(mp_namelist.MUI + Float(3.0)))))))
            * Float(gamma(mp_namelist.MUI + Float(3.0)))
        )
        mp_config.TVBR = float64(
            Float(exp(mp_namelist.BLINR / (mp_namelist.MUR + Float(3.0)) * Float(log(constants.PI * constants.RHOR * Float(gamma(mp_namelist.MUR + Float(3.0)))))))
            * Float(gamma(mp_namelist.MUR + Float(3.0)))
        )
        mp_config.TVBS = float64(
            Float(exp(mp_namelist.BLINS / (mp_namelist.MUS + Float(3.0)) * Float(log(constants.PI * constants.RHOS * Float(gamma(mp_namelist.MUS + Float(3.0)))))))
            * Float(gamma(mp_namelist.MUS + Float(3.0)))
        )
        mp_config.TVBG = float64(
            Float(exp(mp_namelist.BLING / (mp_namelist.MUG + Float(3.0)) * Float(log(constants.PI * constants.RHOG * Float(gamma(mp_namelist.MUG + Float(3.0)))))))
            * Float(gamma(mp_namelist.MUG + Float(3.0)))
        )
        mp_config.TVBH = float64(
            Float(exp(mp_namelist.BLINH / (mp_namelist.MUH + Float(3.0)) * Float(log(constants.PI * constants.RHOH * Float(gamma(mp_namelist.MUH + Float(3.0)))))))
            * Float(gamma(mp_namelist.MUH + Float(3.0)))
        )

        # Schmidt number, Sc ** (1 / 3) in Lin et al. (1983)

        scm3 = exp(Float(1.0) / Float(3.0) * log(constants.VISK / constants.VDIFU))

        pisq = constants.PI * constants.PI

        # accretion between cloud water, cloud ice, rain, snow, and graupel or hail, Lin et al. (1983)

        mp_config.CRACW = (
            constants.PI
            * mp_namelist.N0R_SIG
            * mp_namelist.ALINR
            * gamma(Int(2) + mp_namelist.MUR + mp_namelist.BLINR)
            / (Float(4.0) * exp((Int(2) + mp_namelist.MUR + mp_namelist.BLINR) / (mp_namelist.MUR + Int(3)) * log(mp_config.NORMR)))
            * exp((Int(1) - mp_namelist.BLINR) * log(mp_config.EXPOR))
        )
        mp_config.CRACI = (
            constants.PI
            * mp_namelist.N0R_SIG
            * mp_namelist.ALINR
            * gamma(Int(2) + mp_namelist.MUR + mp_namelist.BLINR)
            / (Float(4.0) * exp((Int(2) + mp_namelist.MUR + mp_namelist.BLINR) / (mp_namelist.MUR + Int(3)) * log(mp_config.NORMR)))
            * exp((Int(1) - mp_namelist.BLINR) * log(mp_config.EXPOR))
        )
        mp_config.CSACW = (
            constants.PI
            * mp_namelist.N0S_SIG
            * mp_namelist.ALINS
            * gamma(Int(2) + mp_namelist.MUS + mp_namelist.BLINS)
            / (Float(4.0) * exp((Int(2) + mp_namelist.MUS + mp_namelist.BLINS) / (mp_namelist.MUS + Int(3)) * log(mp_config.NORMS)))
            * exp((Int(1) - mp_namelist.BLINS) * log(mp_config.EXPOS))
        )
        mp_config.CSACI = (
            constants.PI
            * mp_namelist.N0S_SIG
            * mp_namelist.ALINS
            * gamma(Int(2) + mp_namelist.MUS + mp_namelist.BLINS)
            / (Float(4.0) * exp((Int(2) + mp_namelist.MUS + mp_namelist.BLINS) / (mp_namelist.MUS + Int(3)) * log(mp_config.NORMS)))
            * exp((Int(1) - mp_namelist.BLINS) * log(mp_config.EXPOS))
        )
        if mp_namelist.DO_HAIL:
            mp_config.CGACW = (
                constants.PI
                * mp_namelist.N0H_SIG
                * mp_namelist.ALINH
                * gamma(Int(2) + mp_namelist.MUH + mp_namelist.BLINH)
                * hcon
                / (Float(4.0) * exp((Int(2) + mp_namelist.MUH + mp_namelist.BLINH) / (mp_namelist.MUH + Int(3)) * log(mp_config.NORMH)))
                * exp((Int(1) - mp_namelist.BLINH) * log(mp_config.EXPOH))
            )
            mp_config.CGACI = (
                constants.PI
                * mp_namelist.N0H_SIG
                * mp_namelist.ALINH
                * gamma(Int(2) + mp_namelist.MUH + mp_namelist.BLINH)
                * hcon
                / (Float(4.0) * exp((Int(2) + mp_namelist.MUH + mp_namelist.BLINH) / (mp_namelist.MUH + Int(3)) * log(mp_config.NORMH)))
                * exp((Int(1) - mp_namelist.BLINH) * log(mp_config.EXPOH))
            )
        else:
            mp_config.CGACW = (
                constants.PI
                * mp_namelist.N0G_SIG
                * mp_namelist.ALING
                * gamma(Int(2) + mp_namelist.MUG + mp_namelist.BLING)
                * gcon
                / (Float(4.0) * exp((Int(2) + mp_namelist.MUG + mp_namelist.BLING) / (mp_namelist.MUG + Int(3)) * log(mp_config.NORMG)))
                * exp((Int(1) - mp_namelist.BLING) * log(mp_config.EXPOG))
            )
            mp_config.CGACI = (
                constants.PI
                * mp_namelist.N0G_SIG
                * mp_namelist.ALING
                * gamma(Int(2) + mp_namelist.MUG + mp_namelist.BLING)
                * gcon
                / (Float(4.0) * exp((Int(2) + mp_namelist.MUG + mp_namelist.BLING) / (mp_namelist.MUG + Int(3)) * log(mp_config.NORMG)))
                * exp((Int(1) - mp_namelist.BLING) * log(mp_config.EXPOG))
            )

        if mp_namelist.DO_3D_ACC_CLIQ:

            mp_config.CRACW = pisq * mp_namelist.N0R_SIG * mp_namelist.N0W_SIG * constants.RHOW / Float(24.0)
            mp_config.CSACW = pisq * mp_namelist.N0S_SIG * mp_namelist.N0W_SIG * constants.RHOW / Float(24.0)
            if mp_namelist.DO_HAIL:
                mp_config.CGACW = pisq * mp_namelist.N0H_SIG * mp_namelist.N0W_SIG * constants.RHOW / Float(24.0)
            else:
                mp_config.CGACW = pisq * mp_namelist.N0G_SIG * mp_namelist.N0W_SIG * constants.RHOW / Float(24.0)

        if mp_namelist.DO_3D_ACC_CICE:

            craci = pisq * mp_namelist.N0R_SIG * mp_namelist.N0I_SIG * constants.RHOI / Float(24.0)
            csaci = pisq * mp_namelist.N0S_SIG * mp_namelist.N0I_SIG * constants.RHOI / Float(24.0)
            if mp_namelist.DO_HAIL:
                cgaci = pisq * mp_namelist.N0H_SIG * mp_namelist.N0I_SIG * constants.RHOI / Float(24.0)
            else:
                cgaci = pisq * mp_namelist.N0G_SIG * mp_namelist.N0I_SIG * constants.RHOI / Float(24.0)

        mp_config.CRACW = mp_config.CRACW * mp_namelist.C_PRACW
        mp_config.CRACI = mp_config.CRACI * mp_namelist.C_PRACI
        mp_config.CSACW = mp_config.CSACW * mp_namelist.C_PSACW
        mp_config.CSACI = mp_config.CSACI * mp_namelist.C_PSACI
        mp_config.CGACW = mp_config.CGACW * mp_namelist.C_PGACW
        mp_config.CGACI = mp_config.CGACI * mp_namelist.C_PGACI

        # accretion between cloud water, cloud ice, rain, snow, and graupel or hail, Lin et al. (1983)

        mp_config.CRACS = pisq * mp_namelist.N0R_SIG * mp_namelist.N0S_SIG * constants.RHOS / Float(24.0)
        mp_config.CSACR = pisq * mp_namelist.N0S_SIG * mp_namelist.N0R_SIG * constants.RHOR / Float(24.0)
        if mp_namelist.DO_HAIL:
            mp_config.CGACR = pisq * mp_namelist.N0H_SIG * mp_namelist.N0R_SIG * constants.RHOR / Float(24.0)
            mp_config.CGACS = pisq * mp_namelist.N0H_SIG * mp_namelist.N0S_SIG * constants.RHOS / Float(24.0)
        else:
            mp_config.CGACR = pisq * mp_namelist.N0G_SIG * mp_namelist.N0R_SIG * constants.RHOR / Float(24.0)
            mp_config.CGACS = pisq * mp_namelist.N0G_SIG * mp_namelist.N0S_SIG * constants.RHOS / Float(24.0)

        mp_config.CRACS = mp_config.CRACS * mp_namelist.C_PRACS
        mp_config.CSACR = mp_config.CSACR * mp_namelist.C_PSACR
        mp_config.CGACR = mp_config.CGACR * mp_namelist.C_PGACR
        mp_config.CGACS = mp_config.CGACS * mp_namelist.C_PGACS

        act[0] = mp_config.NORMS
        act[1] = mp_config.NORMR
        act[2] = act[1]
        act[3] = act[0]
        act[4] = act[1]
        if mp_namelist.DO_HAIL:
            act[5] = mp_config.NORMH
        else:
            act[5] = mp_config.NORMG
        act[6] = act[0]
        act[7] = act[5]
        act[8] = mp_config.NORMW
        act[9] = act[1]
        act[10] = mp_config.NORMI
        act[11] = act[1]
        act[12] = act[8]
        act[13] = act[0]
        act[14] = act[10]
        act[15] = act[0]
        act[16] = act[8]
        act[17] = act[5]
        act[18] = act[10]
        act[19] = act[5]

        ace[0] = mp_config.EXPOS
        ace[1] = mp_config.EXPOR
        ace[2] = ace[1]
        ace[3] = ace[0]
        ace[4] = ace[1]
        if mp_namelist.DO_HAIL:
            ace[5] = mp_config.EXPOH
        else:
            ace[5] = mp_config.EXPOG
        ace[6] = ace[0]
        ace[7] = ace[5]
        ace[8] = mp_config.EXPOW
        ace[9] = ace[1]
        ace[10] = mp_config.EXPOI
        ace[11] = ace[1]
        ace[12] = ace[8]
        ace[13] = ace[0]
        ace[14] = ace[10]
        ace[15] = ace[0]
        ace[16] = ace[8]
        ace[17] = ace[5]
        ace[18] = ace[10]
        ace[19] = ace[5]

        mp_config.ACC[0] = mp_namelist.MUS
        mp_config.ACC[1] = mp_namelist.MUR
        mp_config.ACC[2] = mp_config.ACC[1]
        mp_config.ACC[3] = mp_config.ACC[0]
        mp_config.ACC[4] = mp_config.ACC[1]
        if mp_namelist.DO_HAIL:
            mp_config.ACC[5] = mp_namelist.MUH
        else:
            mp_config.ACC[5] = mp_namelist.MUG
        mp_config.ACC[6] = mp_config.ACC[0]
        mp_config.ACC[7] = mp_config.ACC[5]
        mp_config.ACC[8] = mp_namelist.MUW
        mp_config.ACC[9] = mp_config.ACC[1]
        mp_config.ACC[10] = mp_namelist.MUI
        mp_config.ACC[11] = mp_config.ACC[1]
        mp_config.ACC[12] = mp_config.ACC[8]
        mp_config.ACC[13] = mp_config.ACC[0]
        mp_config.ACC[14] = mp_config.ACC[10]
        mp_config.ACC[15] = mp_config.ACC[0]
        mp_config.ACC[16] = mp_config.ACC[8]
        mp_config.ACC[17] = mp_config.ACC[5]
        mp_config.ACC[18] = mp_config.ACC[10]
        mp_config.ACC[19] = mp_config.ACC[5]

        occ[0] = 1.0
        occ[1] = 2.0
        occ[2] = 1.0

        for i in range(3):
            fi = i + 1  # add one to index to replicate fortran indexing
            for k in range(10):
                fk = k + 1  # add one to index to replicate fortran indexing

                mp_config.ACCO[i, k] = (
                    occ[i]
                    * gamma(Int(6) + mp_config.ACC[2 * fk - 1 - 1] - fi)
                    * gamma(mp_config.ACC[2 * fk - 1] + fi - Int(1))
                    / (
                        exp((Int(6) + mp_config.ACC[2 * fk - 1 - 1] - fi) / (mp_config.ACC[2 * fk - 1 - 1] + Int(3)) * log(act[2 * fk - 1 - 1]))
                        * exp((mp_config.ACC[2 * fk - 1] + fi - Int(1)) / (mp_config.ACC[2 * fk - 1] + Int(3)) * log(act[2 * fk - 1]))
                    )
                    * exp((fi - Int(3)) * log(ace[2 * fk - 1 - 1]))
                    * exp((Int(4) - fi) * log(ace[2 * fk - 1]))
                )

        # rain evaporation, snow sublimation, and graupel or hail sublimation, Lin et al. (1983)

        mp_config.CREVP[0] = (
            Float(2.0)
            * constants.PI
            * constants.VDIFU
            * constants.TCOND
            * constants.RVGAS
            * mp_namelist.N0R_SIG
            * gamma(Int(1) + mp_namelist.MUR)
            / exp((Int(1) + mp_namelist.MUR) / (mp_namelist.MUR + Int(3)) * log(mp_config.NORMR))
            * exp(Float(2.0) * log(mp_config.EXPOR))
        )
        mp_config.CREVP[1] = Float(0.78)
        mp_config.CREVP[2] = (
            Float(0.31)
            * scm3
            * sqrt(mp_namelist.ALINR / constants.VISK)
            * gamma((Int(3) + Int(2) * mp_namelist.MUR + mp_namelist.BLINR) / Int(2))
            / exp((Int(3) + Int(2) * mp_namelist.MUR + mp_namelist.BLINR) / (mp_namelist.MUR + Int(3)) / Int(2) * log(mp_config.NORMR))
            * exp((Int(1) + mp_namelist.MUR) / (mp_namelist.MUR + Int(3)) * log(mp_config.NORMR))
            / gamma(Int(1) + mp_namelist.MUR)
            * exp((-Int(1) - mp_namelist.BLINR) / Float(2.0) * log(mp_config.EXPOR))
        )
        mp_config.CREVP[3] = constants.TCOND * constants.RVGAS
        mp_config.CREVP[4] = constants.VDIFU

        mp_config.CSSUB[0] = (
            Float(2.0)
            * constants.PI
            * constants.VDIFU
            * constants.TCOND
            * constants.RVGAS
            * mp_namelist.N0S_SIG
            * gamma(Int(1) + mp_namelist.MUS)
            / exp((Int(1) + mp_namelist.MUS) / (mp_namelist.MUS + Int(3)) * log(mp_config.NORMS))
            * exp(Float(2.0) * log(mp_config.EXPOS))
        )
        mp_config.CSSUB[1] = Float(0.78)
        mp_config.CSSUB[2] = (
            Float(0.31)
            * scm3
            * sqrt(mp_namelist.ALINS / constants.VISK)
            * gamma((Int(3) + Int(2) * mp_namelist.MUS + mp_namelist.BLINS) / Int(2))
            / exp((Int(3) + Int(2) * mp_namelist.MUS + mp_namelist.BLINS) / (mp_namelist.MUS + Int(3)) / Int(2) * log(mp_config.NORMS))
            * exp((Int(1) + mp_namelist.MUS) / (mp_namelist.MUS + Int(3)) * log(mp_config.NORMS))
            / gamma(Int(1) + mp_namelist.MUS)
            * exp((-Int(1) - mp_namelist.BLINS) / Float(2.0) * log(mp_config.EXPOS))
        )
        mp_config.CSSUB[3] = constants.TCOND * constants.RVGAS
        mp_config.CSSUB[4] = constants.VDIFU

        if mp_namelist.DO_HAIL:
            mp_config.CGSUB[0] = (
                Float(2.0)
                * constants.PI
                * constants.VDIFU
                * constants.TCOND
                * constants.RVGAS
                * mp_namelist.N0H_SIG
                * gamma(Int(1) + mp_namelist.MUH)
                / exp((Int(1) + mp_namelist.MUH) / (mp_namelist.MUH + Int(3)) * log(mp_config.NORMH))
                * exp(Float(2.0) * log(mp_config.EXPOH))
            )
            mp_config.CGSUB[1] = Float(0.78)
            mp_config.CGSUB[2] = (
                Float(0.31)
                * scm3
                * sqrt(mp_namelist.ALINH * hcon / constants.VISK)
                * gamma((Int(3) + Int(2) * mp_namelist.MUH + mp_namelist.BLINH) / Int(2))
                / exp(Float(1.0) / (mp_namelist.MUH + Int(3)) * (Int(3) + Int(2) * mp_namelist.MUH + mp_namelist.BLINH) / Int(2) * log(mp_config.NORMH))
                * exp(Float(1.0) / (mp_namelist.MUH + Int(3)) * (Int(1) + mp_namelist.MUH) * log(mp_config.NORMH))
                / gamma(Int(1) + mp_namelist.MUH)
                * exp((-Int(1) - mp_namelist.BLINH) / Float(2.0) * log(mp_config.EXPOH))
            )
        else:
            mp_config.CGSUB[0] = (
                Float(2.0)
                * constants.PI
                * constants.VDIFU
                * constants.TCOND
                * constants.RVGAS
                * mp_namelist.N0G_SIG
                * gamma(Int(1) + mp_namelist.MUG)
                / exp((Int(1) + mp_namelist.MUG) / (mp_namelist.MUG + Int(3)) * log(mp_config.NORMG))
                * exp(Float(2.0) * log(mp_config.EXPOG))
            )
            mp_config.CGSUB[1] = Float(0.78)
            mp_config.CGSUB[2] = (
                Float(0.31)
                * scm3
                * sqrt(mp_namelist.ALING * gcon / constants.VISK)
                * gamma((Int(3) + Int(2) * mp_namelist.MUG + mp_namelist.BLING) / Int(2))
                / exp((Int(3) + Int(2) * mp_namelist.MUG + mp_namelist.BLING) / (mp_namelist.MUG + Int(3)) / Int(2) * log(mp_config.NORMG))
                * exp((Int(1) + mp_namelist.MUG) / (mp_namelist.MUG + Int(3)) * log(mp_config.NORMG))
                / gamma(Int(1) + mp_namelist.MUG)
                * exp((-Int(1) - mp_namelist.BLING) / Float(2.0) * log(mp_config.EXPOG))
            )
        mp_config.CGSUB[3] = constants.TCOND * constants.RVGAS
        mp_config.CGSUB[4] = constants.VDIFU

        # snow melting, Lin et al. (1983)

        mp_config.CSMLT[0] = (
            Float(2.0)
            * constants.PI
            * constants.TCOND
            * mp_namelist.N0S_SIG
            * gamma(Int(1) + mp_namelist.MUS)
            / exp((Int(1) + mp_namelist.MUS) / (mp_namelist.MUS + Int(3)) * log(mp_config.NORMS))
            * exp(Float(2.0) * log(mp_config.EXPOS))
        )
        mp_config.CSMLT[1] = (
            Float(2.0)
            * constants.PI
            * constants.VDIFU
            * mp_namelist.N0S_SIG
            * gamma(Int(1) + mp_namelist.MUS)
            / exp((Int(1) + mp_namelist.MUS) / (mp_namelist.MUS + Int(3)) * log(mp_config.NORMS))
            * exp(Float(2.0) * log(mp_config.EXPOS))
        )
        mp_config.CSMLT[2] = mp_config.CSSUB[1]
        mp_config.CSMLT[3] = mp_config.CSSUB[2]

        # graupel or hail melting, Lin et al. (1983)

        if mp_namelist.DO_HAIL:
            mp_config.CGMLT[0] = (
                Float(2.0)
                * constants.PI
                * constants.TCOND
                * mp_namelist.N0H_SIG
                * gamma(Int(1) + mp_namelist.MUH)
                / exp((Int(1) + mp_namelist.MUH) / (mp_namelist.MUH + Int(3)) * log(mp_config.NORMH))
                * exp(Float(2.0) * log(mp_config.EXPOH))
            )
            mp_config.CGMLT[1] = (
                Float(2.0)
                * constants.PI
                * constants.VDIFU
                * mp_namelist.N0H_SIG
                * gamma(Int(1) + mp_namelist.MUH)
                / exp((Int(1) + mp_namelist.MUH) / (mp_namelist.MUH + Int(3)) * log(mp_config.NORMH))
                * exp(Float(2.0) * log(mp_config.EXPOH))
            )
        else:
            mp_config.CGMLT[0] = (
                Float(2.0)
                * constants.PI
                * constants.TCOND
                * mp_namelist.N0G_SIG
                * gamma(Int(1) + mp_namelist.MUG)
                / exp((Int(1) + mp_namelist.MUG) / (mp_namelist.MUG + Int(3)) * log(mp_config.NORMG))
                * exp(Float(2.0) * log(mp_config.EXPOG))
            )
            mp_config.CGMLT[1] = (
                Float(2.0)
                * constants.PI
                * constants.VDIFU
                * mp_namelist.N0G_SIG
                * gamma(Int(1) + mp_namelist.MUG)
                / exp((Int(1) + mp_namelist.MUG) / (mp_namelist.MUG + Int(3)) * log(mp_config.NORMG))
                * exp(Float(2.0) * log(mp_config.EXPOG))
            )
        mp_config.CGMLT[2] = mp_config.CGSUB[1]
        mp_config.CGMLT[3] = mp_config.CGSUB[2]

        # rain freezing, Lin et al. (1983)

        mp_config.CGFR[0] = (
            Float(1.0e2)
            / Int(36)
            * pisq
            * mp_namelist.N0R_SIG
            * constants.RHOR
            * gamma(Int(6) + mp_namelist.MUR)
            / exp((Int(6) + mp_namelist.MUR) / (mp_namelist.MUR + Int(3)) * log(mp_config.NORMR))
            * exp(Float(-3.0) * log(mp_config.EXPOR))
        )
        mp_config.CGFR[1] = Float(0.66)

    def check_config(self, namelist: GFDLMPV3NamelistConfig, config: GFDLMPV3CloudMPConfig):
        """Checks for any flags that are no meeting the expected value.
        Failing flags are likely not implemented,
        or at the very least not fully implemented

        Args:
            namelist (GFDLMPV3NamelistConfig): namelist containing user-specified GFDLMPV3 configuration
            config (GFDLMPV3CloudMPConfig): dataclass of all non-constant (i.e. computed from constants or namelist values) parameters

        Raises:
            ValueError: list of non-compliant constants with actual and expected values
        """
        failures = []

        # Helper function to evaluate and record non-compliant flags
        def check_param(param_name, actual, expected):
            if actual != expected:
                failures.append({"parameter": param_name, "actual": actual, "expected": expected})

        # Helper function to evaluate and record flags whose value must be one of a set of allowed values.
        # `allowed` is the full set of technically valid values.
        # `implemented` (optional) is the subset that is actually supported right now.
        # If omitted, `implemented` defaults to `allowed` (i.e. everything allowed is implemented).
        def check_param_in(param_name, actual, allowed, implemented=None):
            if implemented is None:
                implemented = allowed

            if actual not in allowed:
                failures.append({"parameter": param_name, "actual": actual, "expected": f"one of {allowed}"})
            elif actual not in implemented:
                failures.append(
                    {
                        "parameter": param_name,
                        "actual": actual,
                        "expected": f"one of {implemented} (values {sorted(set(allowed) - set(implemented))} are valid options but not yet implemented)",
                    }
                )

        # Helper function to evaluate and record parameters that must fall within (or outside) a threshold.
        # Provide min_value and/or max_value; each bound is inclusive unless the corresponding *_exclusive flag is set to True.
        def check_param_threshold(param_name, actual, min_value=None, max_value=None, min_exclusive=False, max_exclusive=False):
            ok = True
            bounds = []

            if min_value is not None:
                ok = ok and (actual > min_value if min_exclusive else actual >= min_value)
                bounds.append(f"{'>' if min_exclusive else '>='} {min_value}")
            if max_value is not None:
                ok = ok and (actual < max_value if max_exclusive else actual <= max_value)
                bounds.append(f"{'<' if max_exclusive else '<='} {max_value}")

            if not ok:
                # checking against a descriptive "expected" string ensures the system prints
                # a human-readable requirement rather than a boolean
                failures.append({"parameter": param_name, "actual": actual, "expected": " and ".join(bounds)})

        # Flag checks (Boolean flags)
        check_param("GFDLMPV3CloudMPConfig.DO_MP_FAST", config.DO_MP_FAST, False)
        check_param("GFDLMPV3CloudMPConfig.DO_MP_FULL", config.DO_MP_FULL, True)
        check_param("GFDLMPV3NamelistConfig.DO_PSD_WATER_FALL", namelist.DO_PSD_WATER_FALL, False)
        check_param("GFDLMPV3NamelistConfig.DO_SEDI_MELT_QG", namelist.DO_SEDI_MELT_QG, False)
        check_param("GFDLMPV3NamelistConfig.DO_SEDI_MELT_QI", namelist.DO_SEDI_MELT_QI, False)
        check_param("GFDLMPV3NamelistConfig.DO_SEDI_MELT_QS", namelist.DO_SEDI_MELT_QS, False)
        check_param("GFDLMPV3NamelistConfig.DO_MP_DIAG", namelist.DO_MP_DIAG, False)
        check_param("GFDLMPV3NamelistConfig.DO_HAIL", namelist.DO_HAIL, False)
        check_param_in("GFDLMPV3NamelistConfig.IFFLAG", namelist.IFFLAG, [1, 2, 3, 4])
        check_param_in("GFDLMPV3NamelistConfig.SEDFLAG", namelist.SEDFLAG, [1, 2, 3, 4], implemented=[1])

        # Threshold checks
        # if config_dependent_constants.DTS >= 300.0:
        #     # checking against a "expected" string will always fail, but this ensures the system prints the correct value in the error
        #     failures.append({"parameter": "DTS", "actual": config_dependent_constants.DTS, "expected": "< 300.0"})

        if failures:
            formatted_failures = "\n".join(f"  - Parameter: {f['parameter']}, Actual: {f['actual']}, Expected: {f['expected']}" for f in failures)
            raise ValueError(f"One or more namelist parameters do not meet expected values:\n{formatted_failures}")

    def __call__(self, state: GFDL1MState, locals: GFDL1MLocals):
        # reset state fields to zero
        self._set_value_2d(field=state.precipitation_at_surface.water, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.rain, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.snow, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.ice, value=Float(0.0))
        self._set_value_2d(field=state.precipitation_at_surface.graupel, value=Float(0.0))

        # reset gfdl locals to zero
        self._set_value(field=locals.dcloud_fraction_dt, value=Float(0.0))
        self._set_value(field=state.non_anvil_large_scale.evaporation, value=Float(0.0))
        self._set_value(field=state.non_anvil_large_scale.sublimation, value=Float(0.0))

        self._driver(state, locals)
