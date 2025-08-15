from ndsl import StencilFactory
from ndsl.dsl.gt4py import PARALLEL, interval, computation, FORWARD, sqrt, max, min, abs, floor
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from pyMoist.convective_parameterization.GF_2020.config import GF2020Config
import pyMoist.constants as constants
from pyMoist.convective_parameterization.GF_2020.temporaries import Temporaries
from pyMoist.convective_parameterization.GF_2020.state import MixingRatios
from pyMoist.saturation_tables.qsat_functions import saturation_specific_humidity
from pyMoist.field_types import GlobalTable_saturaion_tables
from pyMoist.saturation_tables.tables.main import SaturationVaporPressureTable


def setup(
    dx2d: FloatFieldIJ,
    stochastic_sig: FloatFieldIJ,
    zm: FloatField,
    zt: FloatField,
    dm: FloatField,
    lons: FloatFieldIJ,
    lats: FloatFieldIJ,
    aot500: FloatFieldIJ,
    temp2m: FloatFieldIJ,
    sflux_r: FloatFieldIJ,
    sflux_t: FloatFieldIJ,
    topt: FloatFieldIJ,
    xland: FloatFieldIJ,
    sfc_press: FloatFieldIJ,
    kpbl: FloatFieldIJ,
    cnvfrc: FloatFieldIJ,
    srftype: FloatFieldIJ,
    col_sat: FloatFieldIJ,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    entr_c: FloatField,
    temp: FloatField,
    press: FloatField,
    rvap: FloatField,
    mp_ice_ls: FloatField,
    mp_liq_ls: FloatField,
    mp_cf_ls: FloatField,
    mp_ice_cn: FloatField,
    mp_liq_cn: FloatField,
    mp_cf_cn: FloatField,
    curr_rvap: FloatField,
    # forcings
    buoy_exc: FloatField,
    rthften: FloatField,
    rqvften: FloatField,
    rth_advten: FloatField,
    rthblten: FloatField,
    rqvblten: FloatField,
    # output
    conprr: FloatFieldIJ,
    lightn_dens: FloatFieldIJ,
    rthcuten: FloatField,
    rqvcuten: FloatField,
    rqccuten: FloatField,
    rucuten: FloatField,
    rvcuten: FloatField,
    sub_mpqi_1: FloatField,
    sub_mpql_1: FloatField,
    sub_mpcf_1: FloatField,
    sub_mpqi_2: FloatField,
    sub_mpql_2: FloatField,
    sub_mpcf_2: FloatField,
    rbuoycuten: FloatField,
    # rchemcuten: FloatField, # NOTE I AM A PROBLEM. TRACERS CRAP.
    revsu_gf: FloatField,
    prfil_gf: FloatField,
    # # NOTE ALL OF THESE ARE PROBLEMS
    # do_this_column: FloatFieldIJ,
    # ierr4d: FloatField,
    # jmin4d: FloatField,
    # klcl4d: FloatField,
    # k224d: FloatField,
    # kbcon4d: FloatField,
    # ktop4d: FloatField,
    # kstabi4d: FloatField,
    # kstabm4d: FloatField,
    # cprr4d: FloatField,
    # xmb4d: FloatField,
    # edt4d: FloatField,
    # pwav4d: FloatField,
    # sigma4d: FloatField,
    # pcup5d: FloatField,
    # entr5d: FloatField,
    # up_massentr5d: FloatField,
    # up_massdetr5d: FloatField,
    # dd_massentr5d: FloatField,
    # dd_massdetr5d: FloatField,
    # zup5d: FloatField,
    # zdn5d: FloatField,
    # prup5d: FloatField,
    # prdn5d: FloatField,
    # clwup5d: FloatField,
    # tup5d: FloatField,
    # conv_cld_fr5d: FloatField,
):
    from __externals__ import C1, ADV_TRIGGER

    with computation(PARALLEL), interval(...):
        if C1 > 0:
            USE_C1D = True
        if ADV_TRIGGER == 2:
            option_not_implemented_placeholder = True

    with computation(PARALLEL), interval(...):
        # initalization
        rtgt = 1.0

        ztexec = 0.0
        zqexec = 0.0
        last_ierr = -999
        fixout_qv = 1.0
        conprr = 0.0
        lightn_dens = 0.0
        revsu_gf_internal = 0.0
        prfil_gf_internal = 0.0
        Tpert_internal = 0.0
        temp_tendqv = 0.0
        !- tendencies (w/ maxiens)
        outt   (i,:,:)=0.0
        outu   (i,:,:)=0.0
        outv   (i,:,:)=0.0
        outq   (i,:,:)=0.0
        outqc  (i,:,:)=0.0
        outnice(i,:,:)=0.0
        outnliq(i,:,:)=0.0
        outbuoy(i,:,:)=0.0



class GF2020DriverSetup:
    def __init__(self, stencil_factory: StencilFactory, GF_2020_config: GF2020Config):
        self.setup = stencil_factory.from_dims_halo(
            func=setup,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            externals={
                "C1": GF_2020_config.C1,
                "ADV_TRIGGER": GF_2020_config.ADV_TRIGGER,
            },
        )

    def __call__(self, *args, **kwds):
        pass
