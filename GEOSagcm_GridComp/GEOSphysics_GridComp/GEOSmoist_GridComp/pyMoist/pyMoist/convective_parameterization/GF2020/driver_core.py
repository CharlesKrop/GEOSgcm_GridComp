import numpy as np
import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    THIS_K,
)

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatFieldIJ, FloatField, Int, IntFieldIJ


def driver(
    dt: Float,
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
    p_sfc: FloatFieldIJ,
    kpbl: IntFieldIJ,
    cnvfrc: FloatFieldIJ,
    srftype: FloatFieldIJ,
    col_sat: FloatFieldIJ,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    entr_c: FloatFieldIJ,
    temp: FloatFieldIJ,
    pres: FloatFieldIJ,
    rvap: FloatFieldIJ,
    qils: FloatField,
    qlls: FloatField,
    clls: FloatField,
    qicn: FloatField,
    qlcn: FloatField,
    clcn: FloatField,
    curr_rvap: FloatField,
    # ---- forcings---
    buoy_exc: FloatField,
    rthften: FloatField,
    rqvften: FloatField,
    rth_advten: FloatField,
    rthblten: FloatField,
    rqvblten: FloatField,
    # ---- output ----
    conprr: FloatFieldIJ,
    lightn_dens: FloatFieldIJ,
    rthcuten: FloatField,
    rqvcuten: FloatField,
    rqccuten: FloatField,
    rucuten: FloatField,
    rvcuten: FloatField,
    sub_mpqi_1: FloatField,
    sub_mpqi_2: FloatField,
    sub_mpql_1: FloatField,
    sub_mpql_2: FloatField,
    sub_mpcf_1: FloatField,
    sub_mpcf_2: FloatField,
    rbuoycuten: FloatField,
    # rchemcuten, # NOTE THIS CONATINS ALL THE TRACERS, will handle this separately when needed
    revsu_gf: FloatField,
    prfil_gf: FloatField,
    do_this_column: IntFieldIJ,
    ierr_deep: IntFieldIJ,
    ierr_mid: IntFieldIJ,
    ierr_shal: IntFieldIJ,
    jmin_deep: IntFieldIJ,
    jmin_mid: IntFieldIJ,
    jmin_shal: IntFieldIJ,
    klcl_deep: IntFieldIJ,
    klcl_mid: IntFieldIJ,
    klcl_shal: IntFieldIJ,
    k22_deep: IntFieldIJ,
    k22_mid: IntFieldIJ,
    k22_shal: IntFieldIJ,
    kbcon_deep: IntFieldIJ,
    kbcon_mid: IntFieldIJ,
    kbcon_shal: IntFieldIJ,
    ktop_deep: IntFieldIJ,
    ktop_mid: IntFieldIJ,
    ktop_shal: IntFieldIJ,
    kstabi_deep: IntFieldIJ,
    kstabi_mid: IntFieldIJ,
    kstabi_shal: IntFieldIJ,
    kstabm_deep: IntFieldIJ,
    kstabm_mid: IntFieldIJ,
    kstabm_shal: IntFieldIJ,
    cprr_deep: FloatFieldIJ,
    cprr_mid: FloatFieldIJ,
    cprr_shal: FloatFieldIJ,
    xmb_deep: FloatFieldIJ,
    xmb_mid: FloatFieldIJ,
    xmb_shal: FloatFieldIJ,
    edt_deep: FloatFieldIJ,
    edt_mid: FloatFieldIJ,
    edt_shal: FloatFieldIJ,
    pwav_deep: FloatFieldIJ,
    pwav_mid: FloatFieldIJ,
    pwav_shal: FloatFieldIJ,
    sigma_deep: FloatFieldIJ,
    sigma_mid: FloatFieldIJ,
    sigma_shal: FloatFieldIJ,
    pcup_deep: FloatField,
    pcup_mid: FloatField,
    pcup_shal: FloatField,
    entr_deep: FloatField,
    entr_mid: FloatField,
    entr_shal: FloatField,
    up_massentr_deep: FloatField,
    up_massentr_mid: FloatField,
    up_massentr_shal: FloatField,
    up_massdetr_deep: FloatField,
    up_massdetr_mid: FloatField,
    up_massdetr_shal: FloatField,
    dd_massentr_deep: FloatField,
    dd_massentr_mid: FloatField,
    dd_massentr_shal: FloatField,
    dd_massdetr_deep: FloatField,
    dd_massdetr_mid: FloatField,
    dd_massdetr_shal: FloatField,
    zup_deep: FloatField,
    zup_mid: FloatField,
    zup_shal: FloatField,
    zdn_deep: FloatField,
    zdn_mid: FloatField,
    zdn_shal: FloatField,
    prup_deep: FloatField,
    prup_mid: FloatField,
    prup_shal: FloatField,
    prdn_deep: FloatField,
    prdn_mid: FloatField,
    prdn_shal: FloatField,
    clwup_deep: FloatField,
    clwup_mid: FloatField,
    clwup_shal: FloatField,
    tup_deep: FloatField,
    tup_mid: FloatField,
    tup_shal: FloatField,
    conv_cld_fr_deep: FloatField,
    conv_cld_fr_mid: FloatField,
    conv_cld_fr_shal: FloatField,
    # for debug/diagnostic
    AA0: FloatFieldIJ,
    AA1: FloatFieldIJ,
    AA2: FloatFieldIJ,
    AA3: FloatFieldIJ,
    AA1_BL: FloatFieldIJ,
    AA1_CIN: FloatFieldIJ,
    TAU_BL: FloatFieldIJ,
    TAU_EC: FloatFieldIJ,
):
    from __externals__ import USE_TRACER_TRANSP, AUTOCONV

    with computation(FORWARD), interval(0, 1):
        ztexec = 0.0
        zqexec = 0.0
        last_ierr = -999
        fixout_qv = 1.0

        conprr = 0.0
        lightn_dens = 0.0

        if USE_TRACER_TRANSP == 1:
            out_chem_1_deep = 0.0
            out_chem_2_deep = 0.0
            out_chem_1_mid = 0.0
            out_chem_2_mid = 0.0
            out_chem_1_shal = 0.0
            out_chem_2_shal = 0.0

    with computation(PARALLEL), interval(...):
        revsu_gf_2d = 0.0
        prfil_gf_2d = 0.0
        Tpert_2d = 0.0
        temp_tendqv = 0.0
        # tendencies (w/ maxiens)
        outt_deep = 0.0
        outt_mid = 0.0
        outt_shal = 0.0
        outu_deep = 0.0
        outu_mid = 0.0
        outu_shal = 0.0
        outv_deep = 0.0
        outv_mid = 0.0
        outv_shal = 0.0
        outq_deep = 0.0
        outq_mid = 0.0
        outq_shal = 0.0
        outqc_deep = 0.0
        outqc_mid = 0.0
        outqc_shal = 0.0
        outnice_deep = 0.0
        outnice_mid = 0.0
        outnice_shal = 0.0
        outnliq_deep = 0.0
        outnliq_mid = 0.0
        outnliq_shal = 0.0
        outbuoy_deep = 0.0
        outbuoy_mid = 0.0
        outbuoy_shal = 0.0
        omeg_deep = 0.0
        omeg_mid = 0.0
        omeg_shal = 0.0

        if AUTOCONV == 2:
            ccn = max(100.0, (370.37 * (0.01 + max(0.0, aot500))) ** 1.555)
        else:
            ccn = 100.0



