import numpy as np
from gt4py.cartesian.gtscript import i32

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int


class outputs:
    def __init__(self, quantity_factory):
        self.cnv_mfc = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")
        self.wqt_dc = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")

        self.cnv_mf0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.cnv_prc3 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.cnv_mfd = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.cnv_dqcdt = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.cnv_updf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.cnv_cvw = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.cnv_qc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.entlam = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        self.cnpcprate = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.lightn_dens = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")

        self.revsu = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.entr3d = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.entr_dp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.entr_md = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.entr_sh = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        self.prfil = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")

        # Tendencies
        self.dqdt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.dtdt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.dudt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.dvdt_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        # for debug/diagnostoc purposes
        self.sigma_deep = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.sigma_mid = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.cnv_topp_dp = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.cnv_topp_md = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.cnv_topp_sh = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.mupdp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.mdndp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.mupsh = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.mupmd = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.mfdp = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.mfsh = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.mfmd = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.errdp = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.errsh = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.errmd = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.aa0 = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.aa1 = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.aa2 = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.aa3 = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.aa1_bl = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.aa1_cin = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.tau_bl = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.tau_ec = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")


class temporaries:
    def __init__(self, quantity_factory):
        self.up = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.vp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.wp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.rvap = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.press = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zm3d = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zt3d = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.dm3d = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.ec3d = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.curr_rvap = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.buoy_exc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.khloc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        self.gsf_t = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # grid-scale forcing for temp
        self.gsf_q = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # advection forcing for rv
        self.advf_t = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # advection forcing for temp
        self.sgsf_t = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # sub-grid scale forcing for temp
        self.sgsf_q = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # sub-grid scale forcing for rv
        self.src_t = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # temp tendency      from convection
        self.src_q = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # rv tendency        from convection
        self.src_ci = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # cloud/ice tendency from convection
        self.src_u = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # U tendency         from convection
        self.src_v = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # V tendency         from convection
        self.src_ni = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # Ice     number tendency from convection
        self.src_nl = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # Droplet number tendency from convection
        self.src_buoy = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # buoyancy tendency from downdrafts
        self.revsu_gf = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # evaporation_or_sublimation of_convective_precipitation kg kg-1 s-1
        self.prfil_gf = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM], "n/a"
        )  # ice_or_liq convective_precipitation flux: kg m2 s-1 (deep only)

    REAL,  DIMENSION(nmp, mzp , mxp, myp ) ::     &
                                         mp_ice   &
                                        ,mp_liq   &
                                        ,mp_cf

    REAL,  DIMENSION(nmp, mzp , mxp, myp ) ::     &
                                         SUB_MPQI & ! subsidence transport applied to ice mix ratio
                                        ,SUB_MPQL & ! subsidence transport applied to cloud mix ratio
                                        ,SUB_MPCF   ! subsidence transport applied to cloud fraction


class namelist_constants:
    def __init__(self, maxiens):

        # following values correspond to [deep, shallow, congestus]
        self.CUM_HEI_DOWN_LAND = [
            0.30,
            0.20,
            0.20,
        ]  # [0.2,0.8] height of the max Z Downdraft, default = 0.50
        self.CUM_HEI_DOWN_OCEAN = [
            0.30,
            0.20,
            0.20,
        ]  # [0.2,0.8] height of the max Z Downdraft, default = 0.50

        self.CUM_HEI_UPDF_LAND = [
            0.35,
            0.10,
            0.10,
        ]  # [0.2,0.8] height of the max Z Updraft, default = 0.35
        self.CUM_HEI_UPDF_OCEAN = [
            0.35,
            0.10,
            0.10,
        ]  # [0.2,0.8] height of the max Z Updraft, default = 0.35

        self.CUM_MIN_EDT_LAND = [
            0.10,
            0.00,
            0.10,
        ]  # minimum evap fraction allowed over the land, default= 0.1
        self.CUM_MIN_EDT_OCEAN = [
            0.10,
            0.00,
            0.10,
        ]  # minimum evap fraction allowed over the ocean, default= 0.1

        self.CUM_MAX_EDT_LAND = [
            0.90,
            0.00,
            0.90,
        ]  # maximum evap fraction allowed over the land, default= 0.9
        self.CUM_MAX_EDT_OCEAN = [
            0.90,
            0.00,
            0.90,
        ]  # maximum evap fraction allowed over the ocean, default= 0.9

        self.CUM_FADJ_MASSFLX = [
            1.00,
            1.00,
            1.00,
        ]  # multiplicative factor for tunning the mass flux at cloud base, default = 1.0
        self.CUM_USE_EXCESS = [1, 1, 1]  # use T,Q excess sub-grid scale variability
