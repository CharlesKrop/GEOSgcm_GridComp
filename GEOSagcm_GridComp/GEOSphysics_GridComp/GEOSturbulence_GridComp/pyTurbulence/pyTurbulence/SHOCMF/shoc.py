import dace
from ndsl import NDSLRuntime, OptimizationConfig, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, K, computation, erfc, exp, float32, int32, int64, interval, isnan, log, sqrt, tanh
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int

from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
#from pyTurbulence.SHOCMF.locals import SHOCMFLocals
#from pyTurbulence.SHOCMF.state import SHOCMFState
import pyTurbulence.constants as constants
from pyMoist.saturation_tables.tables.liquid_exact import liquid_exact
from pyMoist.saturation_tables.tables.ice_exact import ice_exact
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
    w: FloatField,
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
        w = - constants.rog * omega * thv * wrk
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
    from __externals__ import dtn

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
    PRNUM: FloatField,
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
                PRNUM = -1.*PRNUMBER
                PRNUM[0,0,1] = -1.*PRNUMBER
            else:
                PRNUM = -1.*PRNUMBER+2.1*min(10.,RI)
                PRNUM[0,0,1] = -1.*PRNUMBER+2.1*min(10.,RI[0,0,1])
        else:
            PRNUM = PRNUMBER
            PRNUM[0,0,1] = PRNUMBER

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
        #qsatt = omn  * liquid_exact_no_stencil(tabs,prsl,SaturationFormulation.Staars, dtqw) + (1.-omn) * ice_exact_no_stencil(tabs,prsl,SaturationFormulation.Staars,dtqi)
        qsatt, _ = liquid_exact(
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
        #dqsat =  omn * dtqw + (1.-omn) * dtqi
        #bbb = (1. + constants.epsv*qsatt-wrk-qpl-qpi + 1.61*tabs*dqsat) / (1.+lstarn*dqsat)
        brunt=qsatt

    #with computation(PARALLEL), interval(...):
        #brunt = cld_sgs*betdz*(bbb*(hl.at(K=kc)-hl.at(K=kb))) + (bbb*lstarn - (1.+lstarn*dqsat)*tabs) * (total_water.at(K=kc)-total_water.at(K=kb)) + (bbb*constants.fac_cond - (1.+constants.fac_cond*dqsat)*tabs)*(qpl.at(K=kc)-qpl.at(K=kb)) + (bbb*constants.fac_sub  - (1.+constants.fac_sub*dqsat)*tabs)*(qpi.at(K=kc)-qpi.at(K=kb))
    # with computation(PARALLEL), interval(1,None):
    #     bbb = 0.5*(bbb + (1. + constants.epsv*qsatt-wrk-qpl[0,0,-1]-qpi[0,0,-1] + 1.61*tabs[0,0,-1]*dqsat) / (1.+lstarn*dqsat) )
    #     brunt_edge = 0.5*(cld_sgs+cld_sgs[0,0,-1])*betdz*(bbb*(hl-hl[0,0,-1]) + (bbb*lstarn - (1.+lstarn*dqsat)*tabs) * (total_water-total_water[0,0,-1]) + (bbb*fac_cond - (1.+fac_cond*dqsat)*tabs)*(qpl-qpl[0,0,-1]) + (bbb*fac_sub  - (1.+fac_sub*dqsat)*tabs)*(qpi-qpi[0,0,-1]) )

    # with computation(PARALLEL), interval(...):
    #     bbb = 1. + constants.epsv*qv - qpl - qpi
    #     brunt = brunt + (1.-cld_sgs)*betdz*( bbb*(hl.at(K=kc)-hl.at(K=kb)) + constants.epsv*tabs*(total_water.at(K=kc)-total_water.at(K=kb)) + (bbb*constants.fac_cond-tabs)*(qpl.at(K=kc)-qpl.at(K=kb)) + (bbb*constants.fac_sub -tabs)*(qpi.at(K=kc)-qpi.at(K=kb)) )

    # with computation(PARALLEL), interval(1,None):
    #     bbb = 0.5*(bbb + 1. + epsv*qv(i,j,k-1) - qpl(i,j,k-1) - qpi(i,j,k-1))
    #     brunt_edge(i,j,k) = brunt_edge(i,j,k) + (1.-0.5*(cld_sgs(i,j,k)+cld_sgs(i,j,k-1)))*betdz*( bbb*(hl(i,j,k)-hl(i,j,k-1)) + epsv*tabs(i,j,k)*(total_water(i,j,k)-total_water(i,j,k-1)) + (bbb*fac_cond-tabs(i,j,k))*(qpl(i,j,k)-qpl(i,j,k-1)) + (bbb*fac_sub -tabs(i,j,k))*(qpi(i,j,k)-qpi(i,j,k-1)) )

    # with computation(PARALLEL), interval(...):
    #     if (brunt < 1e-5 or zl < 0.75*dryzpbl):
    #         brunt2 = bruntmin
    #     else:
    #         brunt2 = brunt

    # with computation(PARALLEL), interval(...):
    #     brunt_edge(:,:,1) = brunt_edge(:,:,2)
    #     brunt_edge(:,:,nz) = brunt_edge(:,:,nzm)
    #     brunt2(:,:,1) = brunt2(:,:,2)
    #     brunt2(:,:,nzm) = brunt2(:,:,nzm-1)


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
            externals={"dtn": config.dtn},
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





    def __call__(self,):
        """
        RUN_SHOC 
        For NDSL-specific questions, email katrina.fandrich@nasa.gov

        ##############################################################################

        Arguments:
            state: SHOCMFState
        """

        # self._invert_interface_vars(
        #     zi=,
        #     phii_inv=,
        # )

        # self._invert_inputs(
        #     zl=,
        #     phil_inv=,
        #     phii_inv=,
        #     tkh=,
        #     tkh_inv=,
        #     prsl=,
        #     prsl_inv=,
        #     u=,
        #     v=,
        #     omega=,
        #     omega_inv=,
        #     tabs=,
        #     tabs_inv=,
        #     qwv=,
        #     qwv_inv=,
        #     qcl=,
        #     qc_inv=,
        #     qci=,
        #     qi_inv=,
        #     cld_sgs=,
        #     cld_sgs_inv=,
        #     tke=,
        #     tke_inv=,
        #     wthv_sec=,
        #     wthv_sec_inv=,
        #     wthv_mf=,
        #     wthv_mf_inv=,
        # )

        # self._setup_derived_inputs(
        #     wrk=,
        #     prsl=,
        #     qv=,
        #     qwv=,
        #     thv=,
        #     tabs=,
        #     qcl=,
        #     qci=,
        #     w=,
        #     omega=,
        #     qpl=,
        #     qpi=,
        #     total_water=,
        #     qcl=,
        #     qci=,
        #     prespot=,
        #     tabs=,
        #     gamaz=,
        #     zl=,
        #     hl=,
        # )

        # self._define_vertical_grid_increments(
        #     adzi=,
        #     zl=,
        #     adzl=,
        #     zi=,
        # )

        # # The three stencils below solve the TKE equation
        # self._tke_shear_prod(
        #     rdtn=,
        #     def2=,
        #     adzi=,
        #     u=,
        #     v=,
        # )

        # self._calc_numbers(
        #     u=,
        #     v=,
        #     adzi=,
        #     RI=,
        #     PRNUM=,
        #     thv=,
        # )

        # self._reset_tke(
        #     tke=,
        #     min_tke=,
        #     tkesbdiss=,
        #     tkebshear=,
        #     tkesbbuoy=,
        # )

        #self._eddy_length2()

        # self._eddy_length3(
        #     tke=tke,
        #     thv=thv,
        #     zl=zl,
        #     dryzpbl=dryzpbl,
        #     brunt2=brunt2,
        #     smixt=smixt,
        #     smixt1=smixt1,
        #     smixt2=smixt2,
        #     smixt3=smixt3,
        # )

        # self._solve_tke(
        #     adzl=,
        #     tkh=,
        #     wthv_sec=,
        #     wthv_mf=,
        #     thv=,
        #     brunt=,
        #     tke=,
        #     prnum=,
        #     smixt=,
        #     def2=,
        #     zl=,
        #     tke_mf=,
        #     tkesbbuoy=,
        #     tkesbshear=,
        #     tkesbdiss=,
        #     tscale1=,
        # )

        #self._environmental_tke()
        
        #self._flip_and_export()

        # Flip optional exports only if requested
        # if tkesbbuoy_inv is not None:
        #     self._flip_output()
        
        # if tkesbshear_inv is not None:
        #     self._flip_output()

        # if smixt_inv is not None:
        #     self._flip_output()
        
        # if smixt1_inv is not None:
        #     self._flip_output()
        
        # if smixt2_inv is not None:
        #     self._flip_output()

        # if smixt3_inv is not None:
        #     self._flip_output()

        # if bruntmst_inv is not None:
        #     self._flip_output()
        
        # if prnum_inv is not None:
        #     self._flip_output()

        # if ri_inv is not None:
        #     self._flip_output()

        










