import os
import dataclasses

import f90nml

from pyMoist.microphysics.GFDL_1M.microphysics.config import GFDLMPV3NamelistConfig, GFDLMPV3CloudMPConfig
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
from math import gamma, exp, log, sqrt
import numpy as np


@dataclasses.dataclass
class GFDLMPV3HeatCapacities:

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

        # initialize namelist parameters. will be read from namelist where available, otherwise use defaults
        self._mp_namelist = GFDLMPV3NamelistConfig.init_from_namelist(os.path.join(gfdl_1m_config.CWD, namelist))

        # initialize cloud microphysics configuration
        # NOTE currently a two step process so that it is abundantly clear that the calculations required to fully initialize
        # this class are dependent on the namelist and the GFDL1M configuration (i.e. they must be initialized first)
        # open to discussion on merging into a one step initialization
        self._mp_config = GFDLMPV3CloudMPConfig.init_to_none()
        self._setup_cloud_mp_config(quantity_factory, gfdl_1m_config, self._mp_namelist, self._mp_config)

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
        mp_config.ACC = quantity_factory.from_array(np.array(20, dtype=Float), ["LEN20_TABLE"], "n/a")
        mp_config.ACCO = quantity_factory.from_array(np.array((3, 20), dtype=Float), ["LEN3_TABLE", "LEN20_TABLE"], "n/a")
        mp_config.CGFR = quantity_factory.from_array(np.array(2, dtype=Float), ["LEN2_TABLE"], "n/a")
        mp_config.CGMLT = quantity_factory.from_array(np.array(4, dtype=Float), ["LEN4_TABLE"], "n/a")
        mp_config.CGSUB = quantity_factory.from_array(np.array(5, dtype=Float), ["LEN5_TABLE"], "n/a")
        mp_config.CREVP = quantity_factory.from_array(np.array(5, dtype=Float), ["LEN5_TABLE"], "n/a")
        mp_config.CSMLT = quantity_factory.from_array(np.array(4, dtype=Float), ["LEN4_TABLE"], "n/a")
        mp_config.CSSUB = quantity_factory.from_array(np.array(5, dtype=Float), ["LEN5_TABLE"], "n/a")

        # generate numpy arrays for internal tables
        ace = np.array(20, dtype=Float)
        act = np.array(20, dtype=Float)
        occ = np.array(3, dtype=Float)
        crevp = np.array(5, dtype=Float)

        # --------------------------------------------------
        # heat capacities and related parameters
        # --------------------------------------------------
        if gfdl_1m_config.LHYDROSTATIC:
            mp_config.C_AIR = constants.CP_AIR
            mp_config.C_VAP = constants.CP_VAP
            mp_namelist.DO_SEDI_W = False
        else:
            mp_config.C_AIR = constants.CV_AIR
            mp_config.C_VAP = constants.CV_VAP

        # scaled constants (to reduce float point errors for 32-bit)

        mp_config.D1_VAP = mp_config.D0_VAP / mp_config.C_AIR
        mp_config.D1_ICE = constants.DC_ICE / mp_config.C_AIR

        if gfdl_1m_config.LHYDROSTATIC:
            LV00 = (constants.HLV - mp_config.D0_VAP * constants.TICE) / mp_config.C_AIR
        else:
            if gfdl_1m_config.ENG_CNV_OLD:
                LV00 = (constants.HLV - mp_config.D0_VAP * constants.TICE) / mp_config.C_AIR
            else:
                LV00 = (constants.HLV - mp_config.D0_VAP * constants.TICE - constants.RVGAS * constants.TICE) / mp_config.C_AIR

        mp_config.LI00 = (constants.HLF - constants.DC_ICE * constants.TICE) / mp_config.C_AIR
        mp_config.LI20 = LV00 + mp_config.LI00

        mp_config.C1_VAP = mp_config.C_VAP / mp_config.C_AIR
        mp_config.C1_LIQ = constants.C_LIQ / mp_config.C_AIR
        mp_config.C1_ICE = constants.C_ICE / mp_config.C_AIR

        # --------------------------------------------------
        # all other internal configuration parameters
        # --------------------------------------------------

        # complete freezing temperature

        if mp_namelist.DO_WARM_RAIN_MP:
            mp_config.T_WFR = mp_namelist.T_MIN
        else:
            mp_config.T_WFR = constants.TICE - 40.0

        # cloud water autoconversion, Hong et al. (2004)

        aone = 2.0 / 9.0 * (3.0 / 4.0) ** (4.0 / 3.0) / constants.PI ** (1.0 / 3.0)
        mp_config.CPAUT0 = mp_namelist.C_PAUT * aone * constants.GRAV / constants.VISD

        # terminal velocities parameters, Lin et al. (1983)

        gcon = (4.0 * constants.GRAV * constants.RHOG / (3.0 * constants.CDG * constants.RHO0)) ** 0.5
        hcon = (4.0 * constants.GRAV * constants.RHOH / (3.0 * constants.CDH * constants.RHO0)) ** 0.5

        # part of the slope parameters

        mp_config.NORMW = constants.PI * constants.RHOW * mp_namelist.N0W_SIG * gamma(mp_namelist.MUW + 3)
        mp_config.NORMI = constants.PI * constants.RHOI * mp_namelist.N0I_SIG * gamma(mp_namelist.MUI + 3)
        mp_config.NORMR = constants.PI * constants.RHOR * mp_namelist.N0R_SIG * gamma(mp_namelist.MUR + 3)
        mp_config.NORMS = constants.PI * constants.RHOS * mp_namelist.N0S_SIG * gamma(mp_namelist.MUS + 3)
        mp_config.NORMG = constants.PI * constants.RHOG * mp_namelist.N0G_SIG * gamma(mp_namelist.MUG + 3)
        mp_config.NORMH = constants.PI * constants.RHOH * mp_namelist.N0H_SIG * gamma(mp_namelist.MUH + 3)

        mp_config.EXPOW = exp(mp_namelist.N0W_EXP / (mp_namelist.MUW + 3) * log(10.0))
        mp_config.EXPOI = exp(mp_namelist.N0I_EXP / (mp_namelist.MUI + 3) * log(10.0))
        mp_config.EXPOR = exp(mp_namelist.N0R_EXP / (mp_namelist.MUR + 3) * log(10.0))
        mp_config.EXPOS = exp(mp_namelist.N0S_EXP / (mp_namelist.MUS + 3) * log(10.0))
        mp_config.EXPOG = exp(mp_namelist.N0G_EXP / (mp_namelist.MUG + 3) * log(10.0))
        mp_config.EXPOH = exp(mp_namelist.N0H_EXP / (mp_namelist.MUH + 3) * log(10.0))

        # parameters for particle concentration (pc), effective diameter (ed), optical extinction (oe), radar reflectivity factor (rr), and mass-weighted terminal velocity (tv)

        mp_config.PCAW = exp(3 / (mp_namelist.MUW + 3) * log(mp_namelist.N0W_SIG)) * gamma(mp_namelist.MUW) * exp(3 * mp_namelist.N0W_EXP / (muw + 3) * log(10.0))
        mp_config.PCAI = exp(3 / (mp_namelist.MUI + 3) * log(mp_namelist.N0I_SIG)) * gamma(mp_namelist.MUI) * exp(3 * mp_namelist.N0I_EXP / (mui + 3) * log(10.0))
        mp_config.PCAR = (
            exp(3 / (mp_namelist.MUR + 3) * log(mp_namelist.N0R_SIG)) * gamma(mp_namelist.MUR) * exp(3 * mp_namelist.N0R_EXP / (mp_namelist.MUR + 3) * log(10.0))
        )
        mp_config.PCAS = (
            exp(3 / (mp_namelist.MUS + 3) * log(mp_namelist.N0S_SIG)) * gamma(mp_namelist.MUS) * exp(3 * mp_namelist.N0S_EXP / (mp_namelist.MUS + 3) * log(10.0))
        )
        mp_config.PCAG = (
            exp(3 / (mp_namelist.MUG + 3) * log(mp_namelist.N0G_SIG)) * gamma(mp_namelist.MUG) * exp(3 * mp_namelist.N0G_EXP / (mp_namelist.MUG + 3) * log(10.0))
        )
        mp_config.PCAH = (
            exp(3 / (mp_namelist.MUH + 3) * log(mp_namelist.N0H_SIG)) * gamma(mp_namelist.MUH) * exp(3 * mp_namelist.N0H_EXP / (mp_namelist.MUH + 3) * log(10.0))
        )

        mp_config.PCBW = exp(mp_namelist.MUW / (mp_namelist.MUW + 3) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + 3)))
        mp_config.PCBI = exp(mp_namelist.MUI / (mp_namelist.MUI + 3) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + 3)))
        mp_config.PCBR = exp(mp_namelist.MUR / (mp_namelist.MUR + 3) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + 3)))
        mp_config.PCBS = exp(mp_namelist.MUS / (mp_namelist.MUS + 3) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + 3)))
        mp_config.PCBG = exp(mp_namelist.MUG / (mp_namelist.MUG + 3) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + 3)))
        mp_config.PCBH = exp(mp_namelist.MUH / (mp_namelist.MUH + 3) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + 3)))

        mp_config.EDAW = (
            exp(-1.0 / (mp_namelist.MUW + 3) * log(mp_namelist.N0W_SIG)) * (mp_namelist.MUW + 2) * exp(-mp_namelist.N0W_EXP / (mp_namelist.MUW + 3) * log(10.0))
        )
        mp_config.EDAI = (
            exp(-1.0 / (mp_namelist.MUI + 3) * log(mp_namelist.N0I_SIG)) * (mp_namelist.MUI + 2) * exp(-mp_namelist.N0I_EXP / (mp_namelist.MUI + 3) * log(10.0))
        )
        mp_config.EDAR = (
            exp(-1.0 / (mp_namelist.MUR + 3) * log(mp_namelist.N0R_SIG)) * (mp_namelist.MUR + 2) * exp(-mp_namelist.N0R_EXP / (mp_namelist.MUR + 3) * log(10.0))
        )
        mp_config.EDAS = (
            exp(-1.0 / (mp_namelist.MUS + 3) * log(mp_namelist.N0S_SIG)) * (mp_namelist.MUS + 2) * exp(-mp_namelist.N0S_EXP / (mp_namelist.MUS + 3) * log(10.0))
        )
        mp_config.EDAG = (
            exp(-1.0 / (mp_namelist.MUG + 3) * log(mp_namelist.N0G_SIG)) * (mp_namelist.MUG + 2) * exp(-mp_namelist.N0G_EXP / (mp_namelist.MUG + 3) * log(10.0))
        )
        mp_config.EDAH = (
            exp(-1.0 / (mp_namelist.MUH + 3) * log(mp_namelist.N0H_SIG)) * (mp_namelist.MUH + 2) * exp(-mp_namelist.N0H_EXP / (mp_namelist.MUH + 3) * log(10.0))
        )

        mp_config.EDBW = exp(1.0 / (mp_namelist.MUW + 3) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + 3)))
        mp_config.EDBI = exp(1.0 / (mp_namelist.MUI + 3) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + 3)))
        mp_config.EDBR = exp(1.0 / (mp_namelist.MUR + 3) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + 3)))
        mp_config.EDBS = exp(1.0 / (mp_namelist.MUS + 3) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + 3)))
        mp_config.EDBG = exp(1.0 / (mp_namelist.MUG + 3) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + 3)))
        mp_config.EDBH = exp(1.0 / (mp_namelist.MUH + 3) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + 3)))

        mp_config.OEAW = (
            exp(1.0 / (mp_namelist.MUW + 3) * log(mp_namelist.N0W_SIG))
            * constants.PI
            * gamma(mp_namelist.MUW + 2)
            * exp(mp_namelist.N0W_EXP / (mp_namelist.MUW + 3) * log(10.0))
        )
        mp_config.OEAI = (
            exp(1.0 / (mp_namelist.MUI + 3) * log(mp_namelist.N0I_SIG))
            * constants.PI
            * gamma(mp_namelist.MUI + 2)
            * exp(mp_namelist.N0I_EXP / (mp_namelist.MUI + 3) * log(10.0))
        )
        mp_config.OEAR = (
            exp(1.0 / (mp_namelist.MUR + 3) * log(mp_namelist.N0R_SIG))
            * constants.PI
            * gamma(mp_namelist.MUR + 2)
            * exp(mp_namelist.N0R_EXP / (mp_namelist.MUR + 3) * log(10.0))
        )
        mp_config.OEAS = (
            exp(1.0 / (mp_namelist.MUS + 3) * log(mp_namelist.N0S_SIG))
            * constants.PI
            * gamma(mp_namelist.MUS + 2)
            * exp(mp_namelist.N0S_EXP / (mp_namelist.MUS + 3) * log(10.0))
        )
        mp_config.OEAG = (
            exp(1.0 / (mp_namelist.MUG + 3) * log(mp_namelist.N0G_SIG))
            * constants.PI
            * gamma(mp_namelist.MUG + 2)
            * exp(mp_namelist.N0G_EXP / (mp_namelist.MUG + 3) * log(10.0))
        )
        mp_config.OEAH = (
            exp(1.0 / (mp_namelist.MUH + 3) * log(mp_namelist.N0H_SIG))
            * constants.PI
            * gamma(mp_namelist.MUH + 2)
            * exp(mp_namelist.N0H_EXP / (mp_namelist.MUH + 3) * log(10.0))
        )

        mp_config.OEBW = 2 * exp((mp_namelist.MUW + 2) / (mp_namelist.MUW + 3) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + 3)))
        mp_config.OEBI = 2 * exp((mp_namelist.MUI + 2) / (mp_namelist.MUI + 3) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + 3)))
        mp_config.OEBR = 2 * exp((mp_namelist.MUR + 2) / (mp_namelist.MUR + 3) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + 3)))
        mp_config.OEBS = 2 * exp((mp_namelist.MUS + 2) / (mp_namelist.MUS + 3) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + 3)))
        mp_config.OEBG = 2 * exp((mp_namelist.MUG + 2) / (mp_namelist.MUG + 3) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + 3)))
        mp_config.OEBH = 2 * exp((mp_namelist.MUH + 2) / (mp_namelist.MUH + 3) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + 3)))

        mp_config.RRAW = (
            exp(-3 / (mp_namelist.MUW + 3) * log(mp_namelist.N0W_SIG)) * gamma(mp_namelist.MUW + 6) * exp(-3 * mp_namelist.N0W_EXP / (mp_namelist.MUW + 3) * log(10.0))
        )
        mp_config.RRAI = (
            exp(-3 / (mp_namelist.MUI + 3) * log(mp_namelist.N0I_SIG)) * gamma(mp_namelist.MUI + 6) * exp(-3 * mp_namelist.N0I_EXP / (mp_namelist.MUI + 3) * log(10.0))
        )
        mp_config.RRAR = (
            exp(-3 / (mp_namelist.MUR + 3) * log(mp_namelist.N0R_SIG)) * gamma(mp_namelist.MUR + 6) * exp(-3 * mp_namelist.N0R_EXP / (mp_namelist.MUR + 3) * log(10.0))
        )
        mp_config.RRAS = (
            exp(-3 / (mp_namelist.MUS + 3) * log(mp_namelist.N0S_SIG)) * gamma(mp_namelist.MUS + 6) * exp(-3 * mp_namelist.N0S_EXP / (mp_namelist.MUS + 3) * log(10.0))
        )
        mp_config.RRAG = (
            exp(-3 / (mp_namelist.MUG + 3) * log(mp_namelist.N0G_SIG)) * gamma(mp_namelist.MUG + 6) * exp(-3 * mp_namelist.N0G_EXP / (mp_namelist.MUG + 3) * log(10.0))
        )
        mp_config.RRAH = (
            exp(-3 / (mp_namelist.MUH + 3) * log(mp_namelist.N0H_SIG)) * gamma(mp_namelist.MUH + 6) * exp(-3 * mp_namelist.N0H_EXP / (mp_namelist.MUH + 3) * log(10.0))
        )

        mp_config.RRBW = exp((mp_namelist.MUW + 6) / (mp_namelist.MUW + 3) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + 3)))
        mp_config.RRBI = exp((mp_namelist.MUI + 6) / (mp_namelist.MUI + 3) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + 3)))
        mp_config.RRBR = exp((mp_namelist.MUR + 6) / (mp_namelist.MUR + 3) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + 3)))
        mp_config.RRBS = exp((mp_namelist.MUS + 6) / (mp_namelist.MUS + 3) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + 3)))
        mp_config.RRBG = exp((mp_namelist.MUG + 6) / (mp_namelist.MUG + 3) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + 3)))
        mp_config.RRBH = exp((mp_namelist.MUH + 6) / (mp_namelist.MUH + 3) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + 3)))

        mp_config.TVAW = (
            exp(-mp_namelist.BLINW / (mp_namelist.MUW + 3) * log(mp_namelist.N0W_SIG))
            * mp_namelist.ALINW
            * gamma(mp_namelist.MUW + mp_namelist.BLINW + 3)
            * exp(-mp_namelist.BLINW * mp_namelist.N0W_EXP / (mp_namelist.MUW + 3) * log(10.0))
        )
        mp_config.TVAI = (
            exp(-mp_namelist.BLINI / (mp_namelist.MUI + 3) * log(mp_namelist.N0I_SIG))
            * mp_namelist.ALINI
            * gamma(mp_namelist.MUI + mp_namelist.BLINI + 3)
            * exp(-mp_namelist.BLINI * mp_namelist.N0I_EXP / (mp_namelist.MUI + 3) * log(10.0))
        )
        mp_config.TVAR = (
            exp(-mp_namelist.BLINR / (mp_namelist.MUR + 3) * log(mp_namelist.N0R_SIG))
            * mp_namelist.ALINR
            * gamma(mp_namelist.MUR + mp_namelist.BLINR + 3)
            * exp(-mp_namelist.BLINR * mp_namelist.N0R_EXP / (mp_namelist.MUR + 3) * log(10.0))
        )
        mp_config.TVAS = (
            exp(-mp_namelist.BLINS / (mp_namelist.MUS + 3) * log(mp_namelist.N0S_SIG))
            * mp_namelist.ALINS
            * gamma(mp_namelist.MUS + mp_namelist.BLINS + 3)
            * exp(-mp_namelist.BLINS * mp_namelist.N0S_EXP / (mp_namelist.MUS + 3) * log(10.0))
        )
        mp_config.TVAG = (
            exp(-mp_namelist.BLING / (mp_namelist.MUG + 3) * log(mp_namelist.N0G_SIG))
            * mp_namelist.ALING
            * gamma(mp_namelist.MUG + mp_namelist.BLING + 3)
            * exp(-mp_namelist.BLING * mp_namelist.N0G_EXP / (mp_namelist.MUG + 3) * log(10.0))
            * gcon
        )
        mp_config.TVAH = (
            exp(-mp_namelist.BLINH / (mp_namelist.MUH + 3) * log(mp_namelist.N0H_SIG))
            * mp_namelist.ALINH
            * gamma(mp_namelist.MUH + mp_namelist.BLINH + 3)
            * exp(-mp_namelist.BLINH * mp_namelist.N0H_EXP / (mp_namelist.MUH + 3) * log(10.0))
            * hcon
        )

        mp_config.TVBW = exp(mp_namelist.BLINW / (mp_namelist.MUW + 3) * log(constants.PI * constants.RHOW * gamma(mp_namelist.MUW + 3))) * gamma(mp_namelist.MUW + 3)
        mp_config.TVBI = exp(mp_namelist.BLINI / (mp_namelist.MUI + 3) * log(constants.PI * constants.RHOI * gamma(mp_namelist.MUI + 3))) * gamma(mp_namelist.MUI + 3)
        mp_config.TVBR = exp(mp_namelist.BLINR / (mp_namelist.MUR + 3) * log(constants.PI * constants.RHOR * gamma(mp_namelist.MUR + 3))) * gamma(mp_namelist.MUR + 3)
        mp_config.TVBS = exp(mp_namelist.BLINS / (mp_namelist.MUS + 3) * log(constants.PI * constants.RHOS * gamma(mp_namelist.MUS + 3))) * gamma(mp_namelist.MUS + 3)
        mp_config.TVBG = exp(mp_namelist.BLING / (mp_namelist.MUG + 3) * log(constants.PI * constants.RHOG * gamma(mp_namelist.MUG + 3))) * gamma(mp_namelist.MUG + 3)
        mp_config.TVBH = exp(mp_namelist.BLINH / (mp_namelist.MUH + 3) * log(constants.PI * constants.RHOH * gamma(mp_namelist.MUH + 3))) * gamma(mp_namelist.MUH + 3)

        # Schmidt number, Sc ** (1 / 3) in Lin et al. (1983)

        scm3 = exp(1.0 / 3.0 * log(constants.VISK / constants.VDIFU))

        pisq = constants.PI * constants.PI

        # accretion between cloud water, cloud ice, rain, snow, and graupel or hail, Lin et al. (1983)

        mp_config.CRACW = (
            constants.PI
            * mp_namelist.N0R_SIG
            * mp_namelist.ALINR
            * gamma(2 + mp_namelist.MUR + mp_namelist.BLINR)
            / (4.0 * exp((2 + mp_namelist.MUR + mp_namelist.BLINR) / (mp_namelist.MUR + 3) * log(mp_config.NORMR)))
            * exp((1 - mp_namelist.BLINR) * log(mp_config.EXPOR))
        )
        mp_config.CRACI = (
            constants.PI
            * mp_namelist.N0R_SIG
            * mp_namelist.ALINR
            * gamma(2 + mp_namelist.MUR + mp_namelist.BLINR)
            / (4.0 * exp((2 + mp_namelist.MUR + mp_namelist.BLINR) / (mp_namelist.MUR + 3) * log(mp_config.NORMR)))
            * exp((1 - mp_namelist.BLINR) * log(mp_config.EXPOR))
        )
        mp_config.CSACW = (
            constants.PI
            * mp_namelist.N0S_SIG
            * mp_namelist.ALINS
            * gamma(2 + mp_namelist.MUS + mp_namelist.BLINS)
            / (4.0 * exp((2 + mp_namelist.MUS + mp_namelist.BLINS) / (mp_namelist.MUS + 3) * log(mp_config.NORMS)))
            * exp((1 - mp_namelist.BLINS) * log(mp_config.EXPOS))
        )
        mp_config.CSACI = (
            constants.PI
            * mp_namelist.N0S_SIG
            * mp_namelist.ALINS
            * gamma(2 + mp_namelist.MUS + mp_namelist.BLINS)
            / (4.0 * exp((2 + mp_namelist.MUS + mp_namelist.BLINS) / (mp_namelist.MUS + 3) * log(mp_config.NORMS)))
            * exp((1 - mp_namelist.BLINS) * log(mp_config.EXPOS))
        )
        if mp_namelist.DO_HAIL:
            mp_config.CGACW = (
                constants.PI
                * mp_namelist.N0H_SIG
                * mp_namelist.ALINH
                * gamma(2 + mp_namelist.MUH + mp_namelist.BLINH)
                * hcon
                / (4.0 * exp((2 + mp_namelist.MUH + mp_namelist.BLINH) / (mp_namelist.MUH + 3) * log(mp_namelist.NORMH)))
                * exp((1 - mp_namelist.BLINH) * log(mp_config.EXPOH))
            )
            mp_config.CGACI = (
                constants.PI
                * mp_namelist.N0H_SIG
                * mp_namelist.ALINH
                * gamma(2 + mp_namelist.MUH + mp_namelist.BLINH)
                * hcon
                / (4.0 * exp((2 + mp_namelist.MUH + mp_namelist.BLINH) / (mp_namelist.MUH + 3) * log(mp_namelist.NORMH)))
                * exp((1 - mp_namelist.BLINH) * log(mp_config.EXPOH))
            )
        else:
            mp_config.CGACW = (
                constants.PI
                * mp_namelist.N0G_SIG
                * mp_namelist.ALING
                * gamma(2 + mp_namelist.MUG + mp_namelist.BLING)
                * gcon
                / (4.0 * exp((2 + mp_namelist.MUG + mp_namelist.BLING) / (mp_namelist.MUG + 3) * log(mp_namelist.NORMG)))
                * exp((1 - mp_namelist.BLING) * log(mp_config.EXPOG))
            )
            mp_config.CGACI = (
                constants.PI
                * mp_namelist.N0G_SIG
                * mp_namelist.ALING
                * gamma(2 + mp_namelist.MUG + mp_namelist.BLING)
                * gcon
                / (4.0 * exp((2 + mp_namelist.MUG + mp_namelist.BLING) / (mp_namelist.MUG + 3) * log(mp_namelist.NORMG)))
                * exp((1 - mp_namelist.BLING) * log(mp_config.EXPOG))
            )

        if mp_namelist.DO_3D_ACC_CLIQ:

            mp_config.CRACW = pisq * mp_namelist.N0R_SIG * mp_namelist.N0W_SIG * constants.RHOW / 24.0
            mp_config.CSACW = pisq * mp_namelist.N0S_SIG * mp_namelist.N0W_SIG * constants.RHOW / 24.0
            if mp_namelist.DO_HAIL:
                mp_config.CGACW = pisq * mp_namelist.N0H_SIG * mp_namelist.N0W_SIG * constants.RHOW / 24.0
            else:
                mp_config.CGACW = pisq * mp_namelist.N0G_SIG * mp_namelist.N0W_SIG * constants.RHOW / 24.0

        if mp_namelist.DO_3D_ACC_CICE:

            craci = pisq * mp_namelist.N0R_SIG * mp_namelist.N0I_SIG * constants.RHOI / 24.0
            csaci = pisq * mp_namelist.N0S_SIG * mp_namelist.N0I_SIG * constants.RHOI / 24.0
            if mp_namelist.DO_HAIL:
                cgaci = pisq * mp_namelist.N0H_SIG * mp_namelist.N0I_SIG * constants.RHOI / 24.0
            else:
                cgaci = pisq * mp_namelist.N0G_SIG * mp_namelist.N0I_SIG * constants.RHOI / 24.0

        mp_config.CRACW = mp_config.CRACW * mp_namelist.C_PRACW
        mp_config.CRACI = mp_config.CRACI * mp_namelist.C_PRACI
        mp_config.CSACW = mp_config.CSACW * mp_namelist.C_PSACW
        mp_config.CSACI = mp_config.CSACI * mp_namelist.C_PSACI
        mp_config.CGACW = mp_config.CGACW * mp_namelist.C_PGACW
        mp_config.CGACI = mp_config.CGACI * mp_namelist.C_PGACI

        # accretion between cloud water, cloud ice, rain, snow, and graupel or hail, Lin et al. (1983)

        mp_config.CRACS = pisq * mp_namelist.N0R_SIG * mp_namelist.N0S_SIG * constants.RHOS / 24.0
        mp_config.CSACR = pisq * mp_namelist.N0S_SIG * mp_namelist.N0R_SIG * constants.RHOR / 24.0
        if mp_namelist.DO_HAIL:
            mp_config.CGACR = pisq * mp_namelist.N0H_SIG * mp_namelist.N0R_SIG * constants.RHOR / 24.0
            mp_config.CGACS = pisq * mp_namelist.N0H_SIG * mp_namelist.N0S_SIG * constants.RHOS / 24.0
        else:
            mp_config.CGACR = pisq * mp_namelist.N0G_SIG * mp_namelist.N0R_SIG * constants.RHOR / 24.0
            mp_config.CGACS = pisq * mp_namelist.N0G_SIG * mp_namelist.N0S_SIG * constants.RHOS / 24.0

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
        mp_config.ACC[8] = mp_config.ACC[5]
        mp_config.ACC[9] = mp_namelist.MUW
        mp_config.ACC[10] = mp_config.ACC[1]
        mp_config.ACC[11] = mp_namelist.MUI
        mp_config.ACC[12] = mp_config.ACC[1]
        mp_config.ACC[13] = mp_config.ACC[8]
        mp_config.ACC[14] = mp_config.ACC[0]
        mp_config.ACC[15] = mp_config.ACC[10]
        mp_config.ACC[16] = mp_config.ACC[0]
        mp_config.ACC[17] = mp_config.ACC[8]
        mp_config.ACC[18] = mp_config.ACC[5]
        mp_config.ACC[19] = mp_config.ACC[10]
        mp_config.ACC[20] = mp_config.ACC[5]

        occ[0] = 1.0
        occ[1] = 2.0
        occ[2] = 1.0

        for i in range(mp_config.ACCO.shape[0]):
            for k in range(mp_config.ACCO.shape[1]):
                mp_config.ACCO[i, k] = (
                    occ[i]
                    * gamma(6 + mp_config.ACC(2 * (k + 1) - 1) - i + 1)
                    * gamma(mp_config.ACC(2 * (k + 1)) + i)
                    / (
                        exp((6 + mp_config.ACC(2 * (k + 1) - 1) - i + 1) / (mp_config.ACC(2 * (k + 1) - 1) + 3) * log(mp_config.ACC(2 * (k + 1) - 1)))
                        * exp((mp_config.ACC(2 * (k + 1)) + i) / (mp_config.ACC(2 * (k + 1)) + 3) * log(mp_config.ACC(2 * (k + 1))))
                    )
                    * exp((i - 3) * log(ace(2 * (k + 1) - 1)))
                    * exp((4 - i) * log(ace(2 * (k + 1))))
                )

        # rain evaporation, snow sublimation, and graupel or hail sublimation, Lin et al. (1983)

        crevp[0] = (
            2.0
            * constants.PI
            * constants.VDIFU
            * constants.TCOND
            * constants.RVGAS
            * mp_namelist.N0R_SIG
            * gamma(1 + mp_namelist.MUR)
            / exp((1 + mp_namelist.MUR) / (mp_namelist.MUR + 3) * log(mp_config.NORMR))
            * exp(2.0 * log(mp_config.EXPOR))
        )
        crevp[1] = 0.78
        crevp[2] = (
            0.31
            * scm3
            * sqrt(mp_namelist.ALINR / constants.VISK)
            * gamma((3 + 2 * mp_namelist.MUR + mp_namelist.BLINR) / 2)
            / exp((3 + 2 * mp_namelist.MUR + mp_namelist.BLINR) / (mp_namelist.MUR + 3) / 2 * log(mp_config.NORMR))
            * exp((1 + mp_namelist.MUR) / (mp_namelist.MUR + 3) * log(mp_config.NORMR))
            / gamma(1 + mp_namelist.MUR)
            * exp((-1 - mp_namelist.BLINR) / 2.0 * log(mp_config.EXPOR))
        )
        crevp[3] = constants.TCOND * constants.RVGAS
        crevp[4] = constants.VDIFU

        mp_config.CSSUB[0] = (
            2.0
            * constants.PI
            * constants.VDIFU
            * constants.TCOND
            * constants.RVGAS
            * mp_namelist.N0S_SIG
            * gamma(1 + mp_namelist.MUS)
            / exp((1 + mp_namelist.MUS) / (mp_namelist.MUS + 3) * log(mp_config.NORMS))
            * exp(2.0 * log(mp_config.EXPOS))
        )
        mp_config.CSSUB[1] = 0.78
        mp_config.CSSUB[2] = (
            0.31
            * scm3
            * sqrt(mp_namelist.ALINS / constants.VISK)
            * gamma((3 + 2 * mp_namelist.MUS + mp_namelist.BLINS) / 2)
            / exp((3 + 2 * mp_namelist.MUS + mp_namelist.BLINS) / (mp_namelist.MUS + 3) / 2 * log(mp_config.NORMS))
            * exp((1 + mp_namelist.MUS) / (mp_namelist.MUS + 3) * log(mp_config.NORMS))
            / gamma(1 + mp_namelist.MUS)
            * exp((-1 - mp_namelist.BLINS) / 2.0 * log(mp_config.EXPOS))
        )
        mp_config.CSSUB[3] = constants.TCOND * constants.RVGAS
        mp_config.CSSUB[4] = constants.VDIFU

        if mp_namelist.DO_HAIL:
            mp_config.CGSUB[0] = (
                2.0
                * constants.PI
                * constants.VDIFU
                * constants.TCOND
                * constants.RVGAS
                * mp_namelist.N0H_SIG
                * gamma(1 + mp_namelist.MUH)
                / exp((1 + mp_namelist.MUH) / (mp_namelist.MUH + 3) * log(mp_config.NORMH))
                * exp(2.0 * log(mp_config.EXPOH))
            )
            mp_config.CGSUB[1] = 0.78
            mp_config.CGSUB[2] = (
                0.31
                * scm3
                * sqrt(mp_namelist.ALINH * hcon / constants.VISK)
                * gamma((3 + 2 * mp_namelist.MUH + mp_namelist.BLINH) / 2)
                / exp(1.0 / (mp_namelist.MUH + 3) * (3 + 2 * mp_namelist.MUH + mp_namelist.BLINH) / 2 * log(mp_config.NORMH))
                * exp(1.0 / (mp_namelist.MUH + 3) * (1 + mp_namelist.MUH) * log(mp_config.NORMH))
                / gamma(1 + mp_namelist.MUH)
                * exp((-1 - mp_namelist.BLINH) / 2.0 * log(mp_config.EXPOH))
            )
        else:
            mp_config.CGSUB[0] = (
                2.0
                * constants.PI
                * constants.VDIFU
                * constants.TCOND
                * constants.RVGAS
                * mp_namelist.N0G_SIG
                * gamma(1 + mp_namelist.MUG)
                / exp((1 + mp_namelist.MUG) / (mp_namelist.MUG + 3) * log(mp_config.NORMG))
                * exp(2.0 * log(mp_config.EXPOG))
            )
            mp_config.CGSUB[1] = 0.78
            mp_config.CGSUB[2] = (
                0.31
                * scm3
                * sqrt(mp_namelist.ALING * gcon / constants.VISK)
                * gamma((3 + 2 * mp_namelist.MUG + mp_namelist.BLING) / 2)
                / exp((3 + 2 * mp_namelist.MUG + mp_namelist.BLING) / (mp_namelist.MUG + 3) / 2 * log(mp_config.NORMG))
                * exp((1 + mp_namelist.MUG) / (mp_namelist.MUG + 3) * log(mp_config.NORMG))
                / gamma(1 + mp_namelist.MUG)
                * exp((-1 - mp_namelist.BLING) / 2.0 * log(mp_config.EXPOG))
            )
        mp_config.CGSUB[3] = constants.TCOND * constants.RVGAS
        mp_config.CGSUB[4] = constants.VDIFU

        # snow melting, Lin et al. (1983)

        mp_config.CSMLT[0] = (
            2.0
            * constants.PI
            * constants.TCOND
            * mp_namelist.N0S_SIG
            * gamma(1 + mp_namelist.MUS)
            / exp((1 + mp_namelist.MUS) / (mp_namelist.MUS + 3) * log(mp_config.NORMS))
            * exp(2.0 * log(mp_config.EXPOS))
        )
        mp_config.CSMLT[1] = (
            2.0
            * constants.PI
            * constants.VDIFU
            * mp_namelist.N0S_SIG
            * gamma(1 + mp_namelist.MUS)
            / exp((1 + mp_namelist.MUS) / (mp_namelist.MUS + 3) * log(mp_config.NORMS))
            * exp(2.0 * log(mp_config.EXPOS))
        )
        mp_config.CSMLT[2] = mp_config.CGSUB[1]
        mp_config.CSMLT[3] = mp_config.CGSUB[2]

        # graupel or hail melting, Lin et al. (1983)

        if mp_namelist.DO_HAIL:
            mp_config.CGMLT[0] = (
                2.0
                * constants.PI
                * constants.TCOND
                * mp_namelist.N0H_SIG
                * gamma(1 + mp_namelist.MUH)
                / exp((1 + mp_namelist.MUH) / (mp_namelist.MUH + 3) * log(mp_config.NORMH))
                * exp(2.0 * log(mp_config.EXPOH))
            )
            mp_config.CGMLT[1] = (
                2.0
                * constants.PI
                * constants.VDIFU
                * mp_namelist.N0H_SIG
                * gamma(1 + mp_namelist.MUH)
                / exp((1 + mp_namelist.MUH) / (mp_namelist.MUH + 3) * log(mp_config.NORMH))
                * exp(2.0 * log(mp_config.EXPOH))
            )
        else:
            mp_config.CGMLT[0] = (
                2.0
                * constants.PI
                * constants.TCOND
                * mp_namelist.N0G_SIG
                * gamma(1 + mp_namelist.MUG)
                / exp((1 + mp_namelist.MUG) / (mp_namelist.MUG + 3) * log(mp_config.NORMG))
                * exp(2.0 * log(mp_config.EXPOG))
            )
            mp_config.CGMLT[1] = (
                2.0
                * constants.PI
                * constants.VDIFU
                * mp_namelist.N0G_SIG
                * gamma(1 + mp_namelist.MUG)
                / exp((1 + mp_namelist.MUG) / (mp_namelist.MUG + 3) * log(mp_config.NORMG))
                * exp(2.0 * log(mp_config.EXPOG))
            )
        mp_config.CGMLT[2] = mp_config.CGSUB[1]
        mp_config.CGMLT[3] = mp_config.CGSUB[2]

        # rain freezing, Lin et al. (1983)

        mp_config.CGFR[0] = (
            1.0e2
            / 36
            * pisq
            * mp_namelist.N0R_SIG
            * constants.RHOR
            * gamma(6 + mp_namelist.MUR)
            / exp((6 + mp_namelist.MUR) / (mp_namelist.MUR + 3) * log(mp_config.NORMR))
            * exp(-3.0 * log(mp_config.EXPOR))
        )
        mp_config.CGFR[1] = 0.66

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
