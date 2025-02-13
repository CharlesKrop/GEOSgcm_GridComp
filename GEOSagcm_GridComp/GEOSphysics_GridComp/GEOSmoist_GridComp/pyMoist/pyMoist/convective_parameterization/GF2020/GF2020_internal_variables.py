import numpy as np
from gt4py.cartesian.gtscript import i32

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, Int


class outputs:
    def __init__(self, quantity_factory):
        self.cnv_tr = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")

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
        self.temp2m = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.temp_old = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp_new = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp_new_bl = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp_new_adv = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.qv_old = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.qv_new = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.qv_new_bl = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.qv_new_adv = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.sflux_r = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.sflux_t = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.topo_height = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.xland = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.dx2d = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.pbl_top_level = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.dz = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.air_density = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.ec3d = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.p_sfc = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.gsf_t = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.gsf_q = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.sgsf_t = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.sgsf_q = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.advf_t = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zle_dyn = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")
        self.mass_dyn = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.buoyancy_excess = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.press = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.rvap = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.up = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.vp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.wp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zt = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.dm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.khloc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.curr_rvap = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.ocean_fraction = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.zws = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.last_ierr = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.fixout_qv = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.conprr = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.out_chem_1_deep = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.out_chem_2_deep = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.out_chem_1_mid = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.out_chem_2_mid = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.out_chem_1_shal = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.out_chem_2_shal = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.topo_height_no_neg = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.lons_degrees = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.lats_degrees = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.revsu_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.prfil_gf = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp_tendqv = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outt_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outt_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outt_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outu_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outu_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outu_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outv_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outv_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outv_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outq_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outq_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outq_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outqc_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outqc_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outqc_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outnice_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outnice_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outnice_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outnliq_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outnliq_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outnliq_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outbuoy_deep = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outbuoy_mid = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.outbuoy_shal = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.omega = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.ccn = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.sensible_heat_sfc_flux = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.latent_heat_sfc_flux = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        self.dz = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.air_density = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.temp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.pres = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.rvap = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.up = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.vp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.wp = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zt = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.zm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.dm = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.khloc = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        self.curr_rvap = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")


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
