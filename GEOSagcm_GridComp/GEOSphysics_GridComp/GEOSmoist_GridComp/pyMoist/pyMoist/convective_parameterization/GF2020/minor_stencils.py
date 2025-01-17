import numpy as np
import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    computation,
    interval,
    PARALLEL,
    FORWARD,
    BACKWARD,
    THIS_K,
    sqrt,
    round,
    max,
)

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int
import pyMoist.constants as constants


def zero_output(
    WQT_DC: FloatField,
    CNV_MFC: FloatField,
    CNV_MF0: FloatField,
    CNV_PRC3: FloatField,
    CNV_MFD: FloatField,
    CNV_DQCDT: FloatField,
    CNV_UPDF: FloatField,
    CNV_CVW: FloatField,
    CNV_QC: FloatField,
    ENTLAM: FloatField,
    CNPCPRATE: FloatField,
    LIGHTN_DENS: FloatField,
    REVSU: FloatField,
    PRFIL: FloatField,
    DQDT_GF: FloatField,
    DTDT_GF: FloatField,
    DUDT_GF: FloatField,
    DVDT_GF: FloatField,
    SIGMA_DEEP: FloatField,
    SIGMA_MID: FloatField,
    MUPDP: FloatField,
    MDNDP: FloatField,
    MUPSH: FloatField,
    MUPMD: FloatField,
    MFDP: FloatField,
    MFSH: FloatField,
    MFMD: FloatField,
    ERRDP: FloatField,
    ERRSH: FloatField,
    ERRMD: FloatField,
    AA0: FloatField,
    AA1: FloatField,
    AA2: FloatField,
    AA3: FloatField,
    AA1_BL: FloatField,
    AA1_CIN: FloatField,
    TAU_BL: FloatField,
    TAU_EC: FloatField,
):
    with computation(PARALLEL), interval(...):
        wqt_dc = 0.0
        cnv_mfc = 0.0
        cnv_mf0 = 0.0
        cnv_prc3 = 0.0
        cnv_mfd = 0.0
        cnv_dqcdt = 0.0
        cnv_updf = 0.0
        cnv_cvw = 0.0
        cnv_qc = 0.0
        entlam = 0.0
        cnpcprate = 0.0
        lightn_dens = 0.0
        revsu = 0.0
        prfil = 0.0

        dqdt_gf = 0.0
        dtdt_gf = 0.0
        dudt_gf = 0.0
        dvdt_gf = 0.0

        sigma_deep = 0.0
        sigma_mid = 0.0
        mupdp = 0.0
        mdndp = 0.0
        mupsh = 0.0
        mupmd = 0.0
        mfdp = 0.0
        mfsh = 0.0
        mfmd = 0.0
        errdp = 0.0
        errsh = 0.0
        errmd = 0.0
        aa0 = 0.0
        aa1 = 0.0
        aa2 = 0.0
        aa3 = 0.0
        aa1_bl = 0.0
        aa1_cin = 0.0
        tau_bl = 0.0
        tau_ec = 0.0


def flipz(flip: FloatField):



@gtscript.function
def setup_driver(
    flip:FloatField,
    t2m: FloatFieldIJ,
    temp2m: FloatFieldIJ,
    evap: FloatFieldIJ,
    sflux_r: FloatFieldIJ,
    sh: FloatFieldIJ,
    ple: FloatField,
    t: FloatField,
    q: FloatField,
    sflux_t: FloatFieldIJ,
    phis: FloatFieldIJ,
    surface_height: FloatFieldIJ,
    frland: FloatFieldIJ,
    xland: FloatFieldIJ,
    area: FloatFieldIJ,
    dz2d: FloatFieldIJ,
    kpblin: FloatFieldIJ,
    pbl_top_level: FloatFieldIJ,
):
    from __externals__ import (
        kend,
        USE_SCALE_DEP,
        N_TRACERS,
    )  # if pep8 states that constants should be capitals why is kend not capitalized?

    with computation(PARALLEL), interval(...):
        flip = kend - THIS_K

    with computation(FORWARD), interval(0, 1):

        # 2-d input data
        aot500 = 0.1 
        # as moist is called before surface, at the 1st time step all arrays
        # from surface are zero
        if max(t2m) < 1.e-6:
            temp2m = t.at(K=kend) # Kelvin
        else:
            temp2m = t2m # or TA(:,:) ! Kelvin

        # moisture flux from sfc
        sflux_r = evap  # kg m-2 s-1

        # sensible heat flux (sh) comes in W m-2, below it is converted to K m s-1
        # (air_dens_sfc = ple(:,:,mzp)/( 287.04*TA(:,:)*(1.+0.608*QA(:,:)))))
        sflux_t = sh / (
            1004.0
            * ple.at(K=kend)
            / (287.04 * t.at(K=kend))
            * (1.0 + 0.608 * q.at(K=kend))
        )  # K m s-1
        # topography height  (m)
        surface_height = phis / constants.MAPL_GRAV
        # land/ocean fraction: land if < 1 ,ocean if = 1
        xland = 1.0 - frland

        # grid length for the scale awareness (in the future, pass the dx2d array instead
        # of the 0-D real number "dx" for the case of non-uniform grid resolution)
        if USE_SCALE_DEP == 0:
            dx2d = 100000.0  # meters
        else:
            dx2d = sqrt(area)  # meters

        # pbl heigth index
        if round(kpblin) != 0:
            pbl_top_level = max(0, flip.at(K=min( round(kpblin), kend)))
        else:
            pbl_top_level = 0

        # NOTE NOTE NOTE NEED TO HANDLE THE TRIMMING STUFF NOTE NOTE NOTE