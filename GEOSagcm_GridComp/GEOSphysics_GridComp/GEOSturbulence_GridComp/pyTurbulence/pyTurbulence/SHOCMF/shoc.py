import dace
from ndsl import NDSLRuntime, OptimizationConfig, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, K, computation, erfc, exp, float32, int32, int64, interval, isnan, log, sqrt, tanh
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int

from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyTurbulence.SHOCMF.locals import SHOCMFLocals
from pyTurbulence.SHOCMF.state import SHOCMFState
import pyTurbulence.constants as constants
from pyMoist.saturation_tables.tables.liquid_exact import liquid_exact
from pyMoist.saturation_tables.tables.ice_exact import ice_exact
from pyMoist.saturation_tables.saturation_specific_humidity_functions import saturation_specific_humidity_frozen_surface
from pyMoist.saturation_tables.formulation import SaturationFormulation
from pyMoist.saturation_tables.tables.constants import IceExactConstants, LiquidExactConstants


def invert_interface_vars(
    zi: FloatField,
    phii_inv: FloatField,
):
    """
    Map GEOS interface variables to those of SHOC
    """
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kinv = k_end-K
        zi = phii_inv.at(K=kinv)-phii_inv.at(K=k_end)

def invert_inputs(
    zl: FloatField,
    phil_inv: FloatField,
    phii_inv: FloatField,
    tkh: FloatField,
    tkh_inv: FloatField,
    prsl: FloatField,
    prsl_inv: FloatField,
    u: FloatField,
    v: FloatField,
    u_inv: FloatField,
    v_inv: FloatField,
    omega: FloatField,
    omega_inv: FloatField,
    tabs: FloatField,
    tabs_inv: FloatField,
    qwv: FloatField,
    qwv_inv: FloatField,
    qcl: FloatField,
    qc_inv: FloatField,
    qci: FloatField,
    qi_inv: FloatField,
    cld_sgs: FloatField,
    cld_sgs_inv: FloatField,
    tke: FloatField,
    tke_inv: FloatField,
    wthv_sec: FloatField,
    wthv_sec_inv: FloatField,
    wthv_mf: FloatField,
    wthv_mf_inv: FloatField,
):
    """
    Map GEOS variables to those of SHOC
    """
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kinv = k_end-K
        zl = phil_inv.at(K=kinv)-phii_inv.at(K=k_end+1)
        tkh = tkh_inv.at(K=kinv)
        prsl = prsl_inv.at(K=kinv)
        u = u_inv.at(K=kinv)
        v = v_inv.at(K=kinv)
        omega = omega_inv.at(K=kinv)
        tabs = tabs_inv.at(K=kinv)
        qwv = qwv_inv.at(K=kinv)
        qcl = qc_inv.at(K=kinv)
        qci = qi_inv.at(K=kinv)
        cld_sgs = cld_sgs_inv.at(K=kinv)
        tke = tke_inv.at(K=kinv)
        wthv_sec = wthv_sec_inv.at(K=kinv)
        wthv_mf  = wthv_mf_inv.at(K=kinv)
    

def setup_derived_inputs(
    prsl: FloatField,
    qv: FloatField,
    qwv: FloatField,
    thv: FloatField,
    tabs: FloatField,
    qcl: FloatField,
    qci: FloatField,
    omega: FloatField,
    qpl: FloatField,
    qpi: FloatField,
    total_water: FloatField,
    gamaz: FloatField,
    zl: FloatField,
    hl: FloatField,
    bet: FloatField,
):
    """
    Calculate derived variables
    """
    with computation(PARALLEL), interval(...):
        wrk = 1.0 / prsl
        qv = max(qwv, 0.0)
        thv = tabs * (1.0+constants.epsv*qv-qcl-qci)
        qpl = 0.0  # comment or remove when using with prognostic rain/snow
        qpi = 0.0  # comment or remove when using with prognostic rain/snow
        total_water = qcl + qci + qv
        prespot = (constants.MAPL_P00*wrk) ** constants.kapa        
        bet = constants.ggr/(tabs*prespot)    
        thv = thv*prespot       

        gamaz = constants.gocp * zl

        hl = tabs + gamaz - constants.fac_cond*(qcl+qpl) - constants.fac_fus *(qci+qpi)

def define_vertical_grid_increments(
    adzi: FloatField,
    zl: FloatField,
    adzl: FloatField,
    zi: FloatField,
):
    """
    Define vertical grid increments for later use in the vertical differentiation
    """
    from __externals__ import k_end

    with computation(FORWARD), interval(1,None):
        adzi = (zl - zl[0,0,-1])
        adzl[0,0,-1] = (zi - zi[0,0,-1])

    with computation(FORWARD), interval(0,1):
        adzi   = zl-zi

    with computation(FORWARD), interval(-1,None):
        adzi[0,0,1]  = zi[0,0,1]-zl.at(K=k_end)
        adzl = adzi

def tke_shear_prod(
    def2: FloatField,
    adzi: FloatField,
    u: FloatField,
    v: FloatField,
):
    """
    Calculate shear production of TKE
    """

    with computation(PARALLEL), interval(...):
        def2 = 0.0

    with computation(FORWARD), interval(0,1):
        rdzw_up = 1.0/adzi[0,0,1]
        wrku1 = (u[0,0,1]-u)*rdzw_up
        wrkv1 = (v[0,0,1]-v)*rdzw_up
        def2 = wrku1*wrku1 + wrkv1*wrkv1
        txd: FloatFieldIJ = rdzw_up

    with computation(FORWARD), interval(1,-1):
        rdzw_up = 1./adzi[0,0,1]
        rdzw_dn = txd
        wrku1 = (u[0,0,1]-u)*rdzw_up
        wrku2 = (u-u[0,0,-1])*rdzw_dn
        wrkv1 = (v[0,0,1]-v)*rdzw_up
        wrkv2 = (v-v[0,0,-1])*rdzw_dn
        def2 = 0.5 * ((wrku1*wrku1) + (wrku2*wrku2) + (wrkv1*wrkv1) + (wrkv2*wrkv2))
        txd = rdzw_up

    with computation(FORWARD), interval(-1,None):
        rdzw_dn = txd
        wrku2 = (u-u[0,0,-1])*rdzw_dn
        wrkv2 = (v-v[0,0,-1])*rdzw_dn
        def2 = (wrku2*wrku2) + (wrkv2*wrkv2)

def calc_numbers(
    u: FloatField,
    v: FloatField,
    adzi: FloatField,
    RI: FloatField,
    prnum: FloatField,
    thv: FloatField,
    tke_mf: FloatField,
):
    """
    Defines Richardson number and Prandtl number on edges
    """
    from __externals__ import k_end, PRNUMBER

    with computation(PARALLEL), interval(0,-1):
        DU = (u - u[0,0,1])**2 + (v - v[0,0,1])**2
        DU = max( sqrt(DU) / adzi, 0.005 )

    with computation(PARALLEL), interval(...):
        RI = 0.0

    with computation(FORWARD), interval(...):
        RI[0,0,1] = 0.0

    with computation(FORWARD), interval(0,-1):
        RI[0,0,1] = constants.ggr*( (thv[0,0,1] - thv) / adzi ) / ( 0.5*( thv+thv[0,0,1] ) * (DU**2) )
    
    with computation(FORWARD), interval(...):
        kinv = k_end + 1 - K
        if PRNUMBER < 0.0:
            if RI <= 0.0 or tke_mf.at(K=kinv) > 1e-4:
                prnum = -1.*PRNUMBER
                prnum[0,0,1] = -1.*PRNUMBER
            else:
                prnum = -1.*PRNUMBER+2.1*min(10.,RI)
                prnum[0,0,1] = -1.*PRNUMBER+2.1*min(10.,RI[0,0,1])
        else:
            prnum = PRNUMBER
            prnum[0,0,1] = PRNUMBER

def reset_tke(
    tke: FloatField,
    tkesbdiss: FloatField,
    tkesbshear: FloatField,
    tkesbbuoy: FloatField,
):
    from __externals__ import min_tke

    with computation(PARALLEL), interval(...):
        tke = max(min_tke,tke)
        tkesbdiss = 0.
        tkesbshear = 0.
        tkesbbuoy  = 0.
 

def eddy_length2(
    adzi: FloatField,
    bet: FloatField,
    qcl: FloatField,
    qci: FloatField,
    tabs: FloatField,
    prsl: FloatField,
    dtqw: FloatField,
    dtqi: FloatField,
    qpl: FloatField,
    qpi: FloatField,
    cld_sgs: FloatField,
    hl: FloatField,
    total_water: FloatField,
    brunt: FloatField,
    brunt2: FloatField,
    brunt_edge: FloatField,
    qv: FloatField,
    zl: FloatField,
    dryzpbl: FloatFieldIJ,
    formulation: Int,
):
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kb = 0
        kc = 1

    with computation(FORWARD), interval(0,1):
        thedz = adzi.at(K=kc)

    with computation(PARALLEL), interval(1,-1):
        thedz = adzi[0,0,1] + adzi
        betdze = 0.5*(bet-bet[0,0,-1]) / adzi
    
    with computation(PARALLEL), interval(-1,None):
        thedz = adzi
        betdze = 0.5*(bet-bet.at(K=k_end-1)) / adzi

    with computation(PARALLEL), interval(...):
        betdz = bet / thedz
        wrk = qcl + qci
        omn = qcl / (wrk+1.e-20)
        lstarn = constants.fac_cond + (1.-omn)*constants.fac_fus
        qsatt_liq, _ = liquid_exact(
            tabs,
            formulation,
            LiquidExactConstants.B6,
            LiquidExactConstants.B5,
            LiquidExactConstants.B4,
            LiquidExactConstants.B3,
            LiquidExactConstants.B2,
            LiquidExactConstants.B1,
            LiquidExactConstants.B0,
            IceExactConstants.BI6,
            IceExactConstants.BI5,
            IceExactConstants.BI4,
            IceExactConstants.BI3,
            IceExactConstants.BI2,
            IceExactConstants.BI1,
            IceExactConstants.BI0,
            IceExactConstants.S16,
            IceExactConstants.S15,
            IceExactConstants.S14,
            IceExactConstants.S13,
            IceExactConstants.S12,
            IceExactConstants.S11,
            IceExactConstants.S10,
            IceExactConstants.S26,
            IceExactConstants.S25,
            IceExactConstants.S24,
            IceExactConstants.S23,
            IceExactConstants.S22,
            IceExactConstants.S21,
            IceExactConstants.S20,
            LiquidExactConstants.DL_0,
            LiquidExactConstants.DL_1,
            LiquidExactConstants.DL_2,
            LiquidExactConstants.DL_3,
            LiquidExactConstants.DL_4,
            LiquidExactConstants.DL_5,
            LiquidExactConstants.TS,
            LiquidExactConstants.LOGPS,
            LiquidExactConstants.CL_0,
            LiquidExactConstants.CL_1,
            LiquidExactConstants.CL_2,
            LiquidExactConstants.CL_3,
            LiquidExactConstants.CL_4,
            LiquidExactConstants.CL_5,
            LiquidExactConstants.CL_6,
            LiquidExactConstants.CL_7,
            LiquidExactConstants.CL_8,
            LiquidExactConstants.CL_9,
            prsl,
        )
        qsatt_ice, _ = ice_exact(
            tabs,
            formulation,
            IceExactConstants.TMINSTR,
            IceExactConstants.TMINICE,
            IceExactConstants.TSTARR1,
            IceExactConstants.TSTARR2,
            IceExactConstants.TSTARR3,
            IceExactConstants.TSTARR4,
            IceExactConstants.TMAXSTR,
            IceExactConstants.DI_0,
            IceExactConstants.DI_1,
            IceExactConstants.DI_2,
            IceExactConstants.DI_3,
            IceExactConstants.CI_0,
            IceExactConstants.CI_1,
            IceExactConstants.CI_2,
            IceExactConstants.CI_3,
            IceExactConstants.S16,
            IceExactConstants.S15,
            IceExactConstants.S14,
            IceExactConstants.S13,
            IceExactConstants.S12,
            IceExactConstants.S11,
            IceExactConstants.S10,
            IceExactConstants.S26,
            IceExactConstants.S25,
            IceExactConstants.S24,
            IceExactConstants.S23,
            IceExactConstants.S22,
            IceExactConstants.S21,
            IceExactConstants.S20,
            IceExactConstants.BI6,
            IceExactConstants.BI5,
            IceExactConstants.BI4,
            IceExactConstants.BI3,
            IceExactConstants.BI2,
            IceExactConstants.BI1,
            IceExactConstants.BI0,
            prsl
        )
        brunt = qsatt_ice
        qsatt = omn  * qsatt_liq + (1.-omn) * qsatt_ice
        dqsat =  omn * dtqw + (1.-omn) * dtqi
        bbb = (1. + constants.epsv*qsatt-wrk-qpl-qpi + 1.61*tabs*dqsat) / (1.+lstarn*dqsat)

    # with computation(PARALLEL), interval(...):
    #     brunt = cld_sgs*betdz*(bbb*(hl.at(K=kc)-hl.at(K=kb))) + (bbb*lstarn - (1.+lstarn*dqsat)*tabs) * (total_water.at(K=kc)-total_water.at(K=kb)) + (bbb*constants.fac_cond - (1.+constants.fac_cond*dqsat)*tabs)*(qpl.at(K=kc)-qpl.at(K=kb)) + (bbb*constants.fac_sub  - (1.+constants.fac_sub*dqsat)*tabs)*(qpi.at(K=kc)-qpi.at(K=kb))
    
    # with computation(PARALLEL), interval(1,None):
    #     bbb = 0.5*(bbb + (1. + constants.epsv*qsatt-wrk-qpl[0,0,-1]-qpi[0,0,-1] + 1.61*tabs[0,0,-1]*dqsat) / (1.+lstarn*dqsat) )
    #     brunt_edge = 0.5*(cld_sgs+cld_sgs[0,0,-1])*betdz*(bbb*(hl-hl[0,0,-1]) + (bbb*lstarn - (1.+lstarn*dqsat)*tabs) * (total_water-total_water[0,0,-1]) + (bbb*constants.fac_cond - (1.+constants.fac_cond*dqsat)*tabs)*(qpl-qpl[0,0,-1]) + (bbb*constants.fac_sub  - (1.+constants.fac_sub*dqsat)*tabs)*(qpi-qpi[0,0,-1]) )

    # with computation(PARALLEL), interval(...):
    #     bbb = 1. + constants.epsv*qv - qpl - qpi
    #     brunt = brunt + (1.-cld_sgs)*betdz*( bbb*(hl.at(K=kc)-hl.at(K=kb)) + constants.epsv*tabs*(total_water.at(K=kc)-total_water.at(K=kb)) + (bbb*constants.fac_cond-tabs)*(qpl.at(K=kc)-qpl.at(K=kb)) + (bbb*constants.fac_sub -tabs)*(qpi.at(K=kc)-qpi.at(K=kb)) )

    # with computation(PARALLEL), interval(1,None):
    #     bbb = 0.5*(bbb + 1. + constants.epsv*qv[0,0,-1] - qpl[0,0,-1] - qpi[0,0,-1])
    #     brunt_edge = brunt_edge + (1.-0.5*(cld_sgs+cld_sgs[0,0,-1]))*betdz*( bbb*(hl-hl[0,0,-1]) + constants.epsv*tabs*(total_water-total_water[0,0,-1]) + (bbb*constants.fac_cond-tabs)*(qpl-qpl[0,0,-1]) + (bbb*constants.fac_sub -tabs)*(qpi-qpi[0,0,-1]) )

    # with computation(PARALLEL), interval(...):
    #     if (brunt < 1e-5 or zl < 0.75*dryzpbl):
    #         brunt2 = constants.bruntmin
    #     else:
    #         brunt2 = brunt

    # with computation(FORWARD), interval(0,1):
    #     brunt_edge = brunt_edge[0,0,1]
    #     brunt2 = brunt2[0,0,1]

    # with computation(FORWARD), interval(-1,None):
    #     brunt_edge = brunt_edge.at(K=k_end-1)
    #     brunt2 = brunt2.at(K=k_end-1)


def eddy_length3(
    tke: FloatField,
    thv: FloatField,
    zl: FloatField,
    dryzpbl: FloatFieldIJ,
    brunt2: FloatField,
    smixt: FloatField,
    smixt1: FloatField,
    smixt2: FloatField,
    smixt3: FloatField,
):
    from __externals__ import k_end, LENOPT, LENFAC1, LENFAC2, LENFAC3

    with computation(PARALLEL), interval(0,-1):
        tkes = sqrt(tke)
        kk = K
        wrk = thv+0.2
        while kk < k_end and wrk > thv.at(K=kk):
            kk = kk+1
        kk = kk-1

        if abs(thv.at(K=kk+1)-thv.at(K=kk)) > 0.01:
            l_par = zl.at(K=kk) + max(0.,(wrk-thv.at(K=kk))* (zl.at(K=kk+1)-zl.at(K=kk)) / (thv.at(K=kk+1)-thv.at(K=kk)))
        else:
            l_par = zl.at(K=kk)
    
    with computation(PARALLEL), interval(0,-1):
        kk = K
        wrk = thv-0.2 
        while kk > 0 and wrk < thv.at(K=kk):
            kk = kk-1
        if kk == 0 and wrk < thv.at(K=0): 
            kk = kk-1
        kk = kk+1

        if abs(thv.at(K=kk+1)-thv.at(K=kk)) > 0.01:
            l_par = l_par - zl.at(K=kk) + max(0.,(thv.at(K=kk)-wrk)* (zl.at(K=kk+1)-zl.at(K=kk))/(thv.at(K=kk+1)-thv.at(K=kk)))
        else:
            l_par = l_par - zl.at(K=kk)

    with computation(PARALLEL), interval(0,-1):
        l_par = max(min(l_par,1500.),25.)  

    with computation(PARALLEL), interval(0,-1):
        if LENOPT < 4:
            smixt1 = constants.vonk*zl*LENFAC1
            smixt2 = sqrt(l_par*400.*tkes)*LENFAC2
            if ( zl < 0.75*dryzpbl or zl < 500. ):
                smixt3 = max(0.05,tkes)*4.*LENFAC3/(sqrt(brunt2))
            else:
                smixt3 = max(0.05,tkes)*LENFAC3/(sqrt(brunt2))
            
    with computation(PARALLEL), interval(0,-1):
        if LENOPT == 1:
            wrk1 = sqrt(3./(1./smixt2**2+1./smixt3**2))
            if (zl < 300.):
                smixt = wrk1 + (smixt1-wrk1)*exp(-(zl/60.))
            else:
                smixt = wrk1
        elif (LENOPT == 2):
            smixt = 3./(1./smixt1+1./smixt2+1./smixt3)
        elif LENOPT == 3:
            smixt = sqrt(3.)/sqrt(1./smixt1**2+1./smixt2**2+1./smixt3**2)

    with computation(PARALLEL), interval(0,-1):
        if LENOPT == 4: 
            wrk2 = 1.0/(400.*tkes)
            wrk3 = sqrt(brunt2)/(0.7*tkes)
            wrk1 = 1.0/(wrk2+wrk3)
            smixt = 3.3*LENFAC1*(wrk1 + (constants.vonk*zl-wrk1)*exp(-zl/(0.1*dryzpbl)))
            smixt1 = 3.3*LENFAC1/wrk2
            smixt2 = 3.3*LENFAC1/wrk3
            smixt3 = 3.3*LENFAC1*constants.vonk*zl

    with computation(PARALLEL), interval(0,-1):
        smixt = min(constants.max_eddy_length_scale,max(constants.min_eddy_length_scale,smixt))
    
    with computation(PARALLEL), interval(0,-1):    
        if zl > dryzpbl:
            smixt = smixt*(0.025+0.975*exp(-(zl-dryzpbl)/3000.))
      
    with computation(PARALLEL), interval(-1,None):  
        smixt = smixt[0,0,-1]
        smixt1 = smixt1[0,0,-1]
        smixt2 = smixt2[0,0,-1]
        smixt3 = smixt3[0,0,-1]


def solve_tke(
    adzl: FloatField,
    tkh: FloatField,
    wthv_sec: FloatField,
    wthv_mf: FloatField,
    thv: FloatField,
    brunt: FloatField,
    tke: FloatField,
    prnum: FloatField,
    smixt: FloatField,
    def2: FloatField,
    zl: FloatField,
    tke_mf: FloatField,
    tkesbbuoy: FloatField,
    tkesbshear: FloatField,
    tkesbdiss: FloatField,
    tscale1: FloatField,
):
    from __externals__ import k_end, BUOYOPT, Ce, Ces, dtn, nitr, min_tke, max_tke

    with computation(PARALLEL), interval(...):
        Cek = Ce/0.7

        if K == 0:
            ku = 1
            kd = 1
            Cek = Ces
        elif K == k_end:
            ku = K
            kd = K
            Cek = Ces
        
        grd = adzl
        wrk  = 0.5 * (tkh.at(K=ku)+tkh.at(K=kd))

        if BUOYOPT == 2:
            a_prod_bu = (constants.ggr / thv) * wthv_sec
        else:
            a_prod_bu = -1.*wrk*brunt + (constants.ggr / thv)*wthv_mf

        buoy_sgs = brunt

        if buoy_sgs <= 0.0:
            smix = grd
        else:
            smix = min(grd,max(0.1*grd, 0.76*sqrt(tke/(buoy_sgs+1.e-10))))

        Cee = Cek* (constants.pt19 + constants.pt51*smix/grd)
        wrk   = 0.5 * wrk * (prnum.at(K=ku) + prnum.at(K=kd))

        a_prod_sh = min(min(constants.tkhmax,wrk)*def2,0.1)  
        wtke = tke
        wtk2 = wtke
        wrk  = (dtn*Cee)/smixt
        wrk1 = wtke + dtn*(a_prod_sh+a_prod_bu)

        wrk2 = min_tke*(1.+9.*exp(-zl/100.))+0.5*(tke_mf+tke_mf)
        itr=1
        while itr <= nitr:                    
            wtke   = min(max(wrk2, wtke), max_tke)
            a_diss = wrk*sqrt(wtke)           
            wtke   = wrk1 / (1.+a_diss)
            wtke   = constants.tkef1*wtke + constants.tkef2*wtk2   
            wtk2   = wtke
            itr = itr + 1

        tke = min(max(wrk2, wtke), max_tke)
        tscale1 = (dtn+dtn) / a_diss        
        a_diss = (a_diss/dtn)*tke  

        tkesbdiss = -a_diss
        tkesbshear = a_prod_sh
        tkesbbuoy = a_prod_bu
          

def environmental_tke(
    tscale1: FloatField,
    zl: FloatField,
    tke: FloatField,
    dryzpbl: FloatFieldIJ,
    brunt_edge: FloatField,
    prnum:FloatField,
    tkh: FloatField,
    isotropy: FloatField,
):
    from __externals__ import shoc_lambda, ck

    with computation(PARALLEL), interval(1,None):
        wrk = 0.5*(tscale1+tscale1[0,0,-1])
        lambda_zfac = 0.5+0.5*tanh((zl-0.75*dryzpbl-100.)/100) 

        if brunt_edge <= 1e-5:
            isotropy = max(30.,min(constants.max_eddy_dissipation_time_scale,wrk))
        else:
            isotropy = max(30.,min(constants.max_eddy_dissipation_time_scale,wrk/(1.0+shoc_lambda*lambda_zfac*brunt_edge*wrk*wrk)))
        
        if tke < 2e-4: 
            isotropy = 30.

        wrk1 = ck / prnum
        tkh = wrk1*isotropy*0.5*(tke+tke[0,0,-1]) 
        tkh = min(tkh,constants.tkhmax)
    
    with computation(FORWARD), interval(0,1):
        isotropy = isotropy[0,0,1]


def flip_and_export(
    tkh_inv: FloatField,
    tkm_inv: FloatField,
    isotropy_inv: FloatField,
    tke_inv: FloatField,
    tkesbdiss: FloatField,
    tkh: FloatField,
    prnum: FloatField,
    isotropy: FloatField,
    tke: FloatField,
    tkesbdiss_inv: FloatField,
):
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kinv = k_end-K
        tkh_inv = tkh.at(K=kinv)
        tkm_inv = min(constants.tkhmax,tkh.at(K=kinv)*prnum.at(K=kinv))
        isotropy_inv = isotropy.at(K=kinv)
        tke_inv = tke.at(K=kinv)
        tkesbdiss_inv = tkesbdiss.at(K=kinv)


def flip_output_kinterface(
    input: FloatField,
    output: FloatField,
):
    from __externals__ import k_end

    with computation(FORWARD), interval(0,-1):
        kinv = k_end - K
        output[0,0,1] = input.at(K=kinv)

def flip_output(
    input: FloatField,
    output: FloatField,
):
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kinv = k_end - K
        output = input.at(K=kinv)



class RUN_SHOC(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: SHOCMFConfiguration,
        formulation: SaturationFormulation = SaturationFormulation.Staars,
    ) -> None:
        """
        RUN_SHOC

        Arguments:
            stencil_factory (StencilFactory): Factory for creating stencil computations.
            quantity_factory (QuantityFactory): Factory for creating quantities.
            config (dataclass): Data class containing configuration dependent
            constants.
        """

        oconfig = OptimizationConfig(stree=OptimizationConfig.Tree(enabled=False))
        super().__init__(stencil_factory, oconfig)

        self.config = config
        self.locals = SHOCMFLocals.make(self, quantity_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory


        self._invert_interface_vars = self.stencil_factory.from_dims_halo(
            func=invert_interface_vars,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        self._invert_inputs = self.stencil_factory.from_dims_halo(
            func=invert_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._setup_derived_inputs = self.stencil_factory.from_dims_halo(
            func=setup_derived_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._define_vertical_grid_increments = self.stencil_factory.from_dims_halo(
            func=define_vertical_grid_increments,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._tke_shear_prod = self.stencil_factory.from_dims_halo(
            func=tke_shear_prod,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._calc_numbers = self.stencil_factory.from_dims_halo(
            func=calc_numbers,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"PRNUMBER": config.PRNUMBER}
        )

        self._reset_tke = self.stencil_factory.from_dims_halo(
            func=reset_tke,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"min_tke": config.min_tke},
        )

        self._eddy_length2 = self.stencil_factory.from_dims_halo(
            func=eddy_length2,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._eddy_length3 = self.stencil_factory.from_dims_halo(
            func=eddy_length3,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"LENOPT":config.LENOPT, "LENFAC1":config.LENFAC1, "LENFAC2":config.LENFAC2,"LENFAC3":config.LENFAC3}
        )

        self._solve_tke = self.stencil_factory.from_dims_halo(
            func=solve_tke,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"max_tke":config.max_tke,"min_tke":config.min_tke,"nitr":config.nitr,"BUOYOPT":config.BUOYOPT, "Ce":config.Ce, "Ces":config.Ces, "dtn":config.dtn}
        )

        self._environmental_tke = self.stencil_factory.from_dims_halo(
            func=environmental_tke,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"shoc_lambda":config.shoc_lambda, "ck": config.ck}
        )

        self._flip_and_export = self.stencil_factory.from_dims_halo(
            func=flip_and_export,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._flip_output = self.stencil_factory.from_dims_halo(
            func=flip_output,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._flip_output_kinterface = self.stencil_factory.from_dims_halo(
            func=flip_output_kinterface,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        if formulation == SaturationFormulation.Staars:
            self.formulation_int = Int(1)
        elif formulation == SaturationFormulation.CAM:
            self.formulation_int = Int(2)
        elif formulation == SaturationFormulation.MurphyAndKoop:
            self.formulation_int = Int(3)


    def __call__(
        self,
        state: SHOCMFState,
        ):
        """
        RUN_SHOC 
        For NDSL-specific questions, email katrina.fandrich@nasa.gov

        ##############################################################################

        Arguments:
            state: SHOCMFState
        """

    
        # Reset locals

        self._invert_interface_vars(
            zi=self.locals.zi,
            phii_inv=state.input.ZL0,
        )

        self._invert_inputs(
            zl=self.locals.zl,
            phil_inv=state.input.Z,
            phii_inv=state.input.ZL0,
            tkh=self.locals.tkh,
            tkh_inv=state.input_output.TKH,
            prsl=self.locals.prsl,
            prsl_inv=state.input.PLO,
            u=self.locals.u,
            v=self.locals.v,
            u_inv=state.input.U,
            v_inv=state.input.V,
            omega=self.locals.omega,
            omega_inv=state.input.OMEGA,
            tabs=self.locals.tabs,
            tabs_inv=state.input.T,
            qwv=self.locals.qwv,
            qwv_inv=state.input.Q,
            qcl=self.locals.qcl,
            qc_inv=state.input.QL,
            qci=self.locals.qci,
            qi_inv=state.input.QI,
            cld_sgs=self.locals.cld_sgs,
            cld_sgs_inv=state.input.QA,
            tke=self.locals.tke,
            tke_inv=state.input_output.TKESHOC,
            wthv_sec=self.locals.wthv_sec,
            wthv_sec_inv=state.input.WTHV2,
            wthv_mf=self.locals.wthv_mf,
            wthv_mf_inv=state.input.BUOYF,
        )

        self._setup_derived_inputs(
            prsl=self.locals.prsl,
            qv=self.locals.qv,
            qwv=self.locals.qwv,
            thv=self.locals.thv,
            tabs=self.locals.tabs,
            qcl=self.locals.qcl,
            qci=self.locals.qci,
            omega=state.input.OMEGA,
            qpl=self.locals.qpl,
            qpi=self.locals.qpi,
            total_water=self.locals.total_water,
            gamaz=self.locals.gamaz,
            zl=self.locals.zl,
            hl=self.locals.hl,
            bet=self.locals.bet,
        )


        self._define_vertical_grid_increments(
            adzi=self.locals.adzi,
            zl=self.locals.zl,
            adzl=self.locals.adzl,
            zi=self.locals.zi,
        )

        self._tke_shear_prod(
            def2=self.locals.def2,
            adzi=self.locals.adzi,
            u=self.locals.u,
            v=self.locals.v,
        )

        self._calc_numbers(
            u=self.locals.u,
            v=self.locals.v,
            adzi=self.locals.adzi,
            RI=self.locals.RI,
            prnum=self.locals.prnum,
            thv=self.locals.thv,
            tke_mf=state.input.MFTKE,
        )

        self._reset_tke(
            tke=self.locals.tke,
            tkesbdiss=self.locals.tkesbdiss,
            tkesbshear=self.locals.tkesbshear,
            tkesbbuoy=self.locals.tkesbbuoy,
        )


        self._eddy_length2(
            adzi=self.locals.adzi,
            bet=self.locals.bet,
            qcl=self.locals.qcl,
            qci=self.locals.qci,
            tabs=self.locals.tabs,
            prsl=self.locals.prsl,
            dtqw=self.locals.dtqw,
            dtqi=self.locals.dtqi,
            qpl=self.locals.qpl,
            qpi=self.locals.qpi,
            cld_sgs=self.locals.cld_sgs,
            hl=self.locals.hl,
            total_water=self.locals.total_water,
            brunt=self.locals.brunt,
            brunt2=self.locals.brunt2,
            brunt_edge=self.locals.brunt_edge,
            qv=self.locals.qv,
            zl=self.locals.zl,
            dryzpbl=state.input.DRYCBLH,
            formulation=self.formulation_int,
        )

        self._eddy_length3(
            tke=self.locals.tke,
            thv=self.locals.thv,
            zl=self.locals.zl,
            dryzpbl=state.input.DRYCBLH,
            brunt2=self.locals.brunt2,
            smixt=self.locals.smixt,
            smixt1=self.locals.smixt1,
            smixt2=self.locals.smixt2,
            smixt3=self.locals.smixt3,
        )

        self._solve_tke(
            adzl=self.locals.adzl,
            tkh=self.locals.tkh,
            wthv_sec=self.locals.wthv_sec,
            wthv_mf=self.locals.wthv_mf,
            thv=self.locals.thv,
            brunt=self.locals.brunt,
            tke=self.locals.tke,
            prnum=self.locals.prnum,
            smixt=self.locals.smixt,
            def2=self.locals.def2,
            zl=self.locals.zl,
            tke_mf=state.input.MFTKE,
            tkesbbuoy=self.locals.tkesbbuoy,
            tkesbshear=self.locals.tkesbshear,
            tkesbdiss=self.locals.tkesbdiss,
            tscale1=self.locals.tscale1,
        )

        self._environmental_tke(  
            tscale1=self.locals.tscale1,
            zl=self.locals.zl,
            tke=self.locals.tke,
            dryzpbl=state.input.DRYCBLH,
            brunt_edge=self.locals.brunt_edge,
            prnum=self.locals.prnum,
            tkh=self.locals.tkh,
            isotropy=self.locals.isotropy,
        )
        
        self._flip_and_export(
            tkh_inv=state.output.TKH,
            tkm_inv=state.output.KM,
            isotropy_inv=state.output.ISOTROPY,
            tke_inv=state.output.TKESHOC,
            tkesbdiss=self.locals.tkesbdiss,
            tkh=self.locals.tkh,
            prnum=self.locals.prnum,
            isotropy=self.locals.isotropy,
            tke=self.locals.tke,
            tkesbdiss_inv=state.output.TKEDISS,
        )

        # Flip diagnostic outputs only if requested
        if state.output.TKEBUOY is not None:
            self._flip_output(
                input=self.locals.tkesbbuoy,
                output=state.output.TKEBUOY,
            )
        
        if state.output.TKESHEAR is not None:
            self._flip_output(
                input=self.locals.tkesbshear,
                output=state.output.TKESHEAR
            )

        if state.output.LSHOC is not None:
            self._flip_output(
                input=self.locals.smixt,
                output=state.output.LSHOC,
            )
        
        if state.output.LSHOC1 is not None:
            self._flip_output(
                input=self.locals.smixt1,
                output=state.output.LSHOC1,
            )
        
        if state.output.LSHOC2 is not None:
            self._flip_output(
                input=self.locals.smixt2,
                output=state.output.LSHOC2,
            )

        if state.output.LSHOC3 is not None:
            self._flip_output(
                input=self.locals.smixt3,
                output=state.output.LSHOC3,
            )

        if state.output.BRUNTSHOC is not None:
            self._flip_output(
                input=self.locals.brunt,
                output=state.output.BRUNTSHOC
            )
        
        if state.output.SHOCPRNUM is not None:
            self._flip_output_kinterface(
                input=self.locals.prnum,
                output=state.output.SHOCPRNUM,
            )

        if state.output.RI is not None:
            self._flip_output_kinterface(
                input=self.locals.RI,
                output=state.output.RI,
            )



        










