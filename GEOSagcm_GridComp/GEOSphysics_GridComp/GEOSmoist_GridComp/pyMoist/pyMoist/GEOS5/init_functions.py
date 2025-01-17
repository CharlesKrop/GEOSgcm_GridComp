import numpy as np
from gt4py.cartesian.gtscript import i32

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int


class outputs:
    def __init__(
        self,
        quantity_factory
    ):
        cnv_mfc = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")
        cnv_mf0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnv_prc3 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnv_mfd = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnv_dqcdt = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnv_updf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnv_cvw = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnv_qc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        entlam = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        revsu = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        prfil = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        dqdt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        dtdt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        dudt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        dvdt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        mupdp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        mupsh = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        mupmd = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        mfdp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        mfsh = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        mfmd = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        errdp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        errsh = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        errmd = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        aa0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        aa1 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        aa2 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        aa3 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        aa1_bl = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        aa1_cin = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        tau_bl = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        tau_ec = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        sigma_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        sigma_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        cnpcprate = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

class temporaries:
    def __init__(
        self,
        quantity_factory
    ):
        
