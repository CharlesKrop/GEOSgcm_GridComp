import dace
from ndsl import NDSLRuntime, OptimizationConfig, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, K, computation, erfc, exp, float32, int32, int64, interval, isnan, log, sqrt, tanh
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int

from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyTurbulence.SHOCMF.locals import SHOCMFLocals
from pyTurbulence.SHOCMF.state import SHOCMFState
import pyTurbulence.constants as constants
from pyTurbulence.field_types import FloatField_UpdraftProperties
from pyMoist.saturation_tables.tables.liquid_exact import liquid_exact
from pyMoist.saturation_tables.tables.ice_exact import ice_exact
from pyMoist.saturation_tables.formulation import SaturationFormulation
from pyMoist.saturation_tables.tables.constants import IceExactConstants, LiquidExactConstants
from pyMoist.saturation_tables import GlobalTable_saturation_tables
from pyMoist.saturation_tables.saturation_specific_humidity_functions import saturation_specific_humidity

def setup_inputs(
    pblh2: FloatFieldIJ,
    thv3: FloatField,
    tke3: FloatField,
    wqt2: FloatFieldIJ,
    wthl2: FloatFieldIJ,
    zlo3: FloatField,
    zw3: FloatField,
    entx: FloatField,
    pblh: FloatFieldIJ,
    tmp: FloatFieldIJ,
    wqt: FloatFieldIJ,
    wthl: FloatFieldIJ,
    wthv: FloatFieldIJ,
):
    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        # NOTE: Need to double check how to do this
        # if (associated(entx) then 
        entx = constants.MAPL_UNDEF

    with computation(FORWARD), interval(0,1):
        ae3 = 1.0

        wthl=wthl2/constants.cp
        wqt=wqt2
        pblh=pblh2
        pblh=max(pblh,constants.pblhmin)
        wthv=wthl+constants.MAPL_EPSILON*thv3.at(K=k_end)*wqt

        tmp = 0.
        tmp2 = 0.
        k_idx = k_end

        while zlo3.at(K=k_idx)<100. and k_idx>1:
            tmp2 = tmp2 + (zw3.at(K=k_idx-1)-zw3.at(K=k_idx))
            tmp = tmp+tke3.at(K=k_idx)*(zw3.at(K=k_idx-1)-zw3.at(K=k_idx))
            k_idx = k_idx-1

        tmp = tmp/tmp2  

def estimate_scale_height(
    wthv: FloatFieldIJ,
    tmp: FloatFieldIJ,
    phis: FloatFieldIJ,
    UPW: FloatField,
    UPTHL: FloatField,
    UPTHV: FloatField,
    UPQT: FloatField,
    UPA: FloatField,
    UPU: FloatField,
    UPV: FloatField,
    UPQI: FloatField,
    UPQL: FloatField,
    ENT: FloatField,
    QR: FloatField,
    QS: FloatField,
    pw3: FloatField,
    t3: FloatField,
    wqt: FloatFieldIJ,
    qv3: FloatField,
    zlo3: FloatField,
    nup2: IntFieldIJ,
    L0: FloatFieldIJ,
    pmid: FloatField,
    esx: GlobalTable_saturation_tables,
    ztop: FloatFieldIJ,
    zw3: FloatField,
    z: FloatField,
):
    from __externals__ import NUP, ET, L0_EDMF, L0fac, k_end

    with computation(FORWARD), interval(...):
        stop_loop: BoolFieldIJ = False
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            nup2 = NUP
            # NOTE: THESE ARE FLOATFIELD BUNDLES shape (24,24,73,10)
            UPW=0.
            UPTHL=0.
            UPTHV=0.
            UPQT=0.
            UPA=0.
            UPU=0.
            UPV=0.
            UPQI=0.
            UPQL=0.
            ENT=0.
            QR = 0.
            QS = 0.

            if ET == 2:
                pmid = 0.5*(pw3+pw3[0,0,1])
                z = zlo3-zw3.at(K=k_end+1)
                wstar: FloatFieldIJ=max(0.1,(constants.MAPL_GRAV*wthv*1e3/t3.at(K=k_end))**(1./3.)) 
                qstar: FloatFieldIJ=max(0.,wqt)/wstar
                thstar: FloatFieldIJ=max(0.,wthv)/wstar

                sigmaQT: FloatFieldIJ=2.0*qstar
                sigmaTH: FloatFieldIJ=2.0*thstar

                tep: FloatFieldIJ  = t3.at(K=k_end)+max(0.1,sigmaTH) 
                qp: FloatFieldIJ   = qv3.at(K=k_end)+sigmaQT

    with computation(FORWARD), interval(0,1):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ET == 2:
                t1: FloatFieldIJ   = t3.at(K=k_end)
                z1: FloatFieldIJ   = z.at(K=k_end)
                ztop: FloatFieldIJ = z.at(K=k_end)

            else: 
                L0 = L0_EDMF

    with computation(BACKWARD), interval(1,-1):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ET == 2 and stop_loop == False:
                z2: FloatFieldIJ = z
                t2: FloatFieldIJ = t3
                pp: FloatFieldIJ = pmid

                tep   = tep - constants.MAPL_GRAV*( z2-z1 )/constants.MAPL_CP
                qp    = qp  + (0.7/1000.)*(z2-z1)*(qv3-qp)  
                tep   = tep + (0.7/1000.)*(z2-z1)*(t3-tep)

                qsp, dqsp = saturation_specific_humidity(tep , pp , esx)

                dqp   = max( qp - qsp, 0. )/(1.+(constants.MAPL_ALHL/constants.MAPL_CP)*dqsp )
                qp    = qp - dqp
                tep   = tep  + constants.MAPL_ALHL * dqp/constants.MAPL_CP

                if ( t2*(1.+constants.MAPL_VIREPS*qv3) >= tep*(1.+constants.MAPL_VIREPS*qp)+0.2 ):
                    ztop = 0.5*(z2+z1)
                    stop_loop = True

                if stop_loop == False:
                    z1 = z2
                    t1 = t2

    with computation(FORWARD), interval(0,1):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ET == 2:
                L0 = max(min(ztop,2500.),500.) / L0fac

                #NOTE: Need to fix this
                #if (associated(mfdepth)) mfdepth(IH,JH) = ztop
    
    

def flip_variables(
    wthv: FloatFieldIJ,
    tmp: FloatFieldIJ,
    phis: FloatFieldIJ,
    ztop: FloatFieldIJ,
    zlo3: FloatField,
    zw3: FloatField,
    zlo: FloatField,
    u3: FloatField,
    u: FloatField,
    v3: FloatField,
    v: FloatField,
    thl3: FloatField,
    thl: FloatField,
    thv3: FloatField,
    thv: FloatField,
    qv3: FloatField,
    qv: FloatField,
    ql3: FloatField,
    ql: FloatField,
    qi3: FloatField,
    qi: FloatField,
    ui: FloatField,
    vi: FloatField,
    thli: FloatField,
    qvi: FloatField,
    qli: FloatField,
    qii:FloatField,
    qt: FloatField,
    qti: FloatField,
    rhoe: FloatField,
    zw: FloatField,
    p: FloatField,
    dp: FloatField,
    rhoe3: FloatField,
    pw3: FloatField,
):

    from __externals__ import k_end, DISCRETE

    with computation(PARALLEL), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                k_inv=k_end-K
                zlo=zlo3.at(K=k_inv)-zw3.at(K=k_end+1)
                u=u3.at(K=k_inv)
                v=v3.at(K=k_inv)
                thl=thl3.at(K=k_inv)
                thv=thv3.at(K=k_inv)
                qv=qv3.at(K=k_inv)
                ql=ql3.at(K=k_inv)
                qi=qi3.at(K=k_inv)

    with computation(FORWARD), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                k_inv=k_end-K+1
                if DISCRETE == 0:
                    ui   = 0.5*( u3.at(K=k_inv)   + u3.at(K=k_inv-1) )
                    vi   = 0.5*( v3.at(K=k_inv)   + v3.at(K=k_inv-1) )
                    thli = 0.5*( thl3.at(K=k_inv) + thl3.at(K=k_inv-1) )
                    qvi  = 0.5*( qv3.at(K=k_inv)  + qv3.at(K=k_inv-1) )
                    qli = 0.5*( ql3.at(K=k_inv)  + ql3.at(K=k_inv-1) )
                    qii  = 0.5*( qi3.at(K=k_inv)  + qi3.at(K=k_inv-1) )
                else:
                    ui   = u3.at(K=k_inv-1)
                    vi   = v3.at(K=k_inv-1)
                    thli = thl3.at(K=k_inv-1)
                    qvi = qv3.at(K=k_inv-1)
                    qli  = ql3.at(K=k_inv-1)
                    qii  = qi3.at(K=k_inv-1)

    with computation(FORWARD), interval(-1,None):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                ui[0,0,1]     = u
                vi[0,0,1]    = v
                thli[0,0,1]   = thl
                qvi[0,0,1]    = qv
                qli[0,0,1]    = ql
                qii[0,0,1]    = qi

    with computation(PARALLEL), interval(0,1):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                ui   = u
                vi   = v
                thli = thl
                qvi  = qv
                qli  = ql
                qii  = qi

    with computation(FORWARD), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                k_inv=k_end-K+1

                qt  = qv+ql+qi
                qti = qvi+qli+qii
                qti[0,0,1] = qvi[0,0,1]+qli[0,0,1]+qii[0,0,1]

    with computation(FORWARD), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                k_inv=k_end-K+1
                rhoe = rhoe3.at(K=k_inv)
                zw  = zw3.at(K=k_inv)-zw3.at(K=k_end+1)
                p   = pw3.at(K=k_inv)

    with computation(FORWARD), interval(-1,None):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                k_inv=k_end-K+1
                rhoe[0,0,1] = rhoe3.at(K=k_inv-1)
                zw[0,0,1]  = zw3.at(K=k_inv-1)-zw3.at(K=k_end+1)
                p[0,0,1]= pw3.at(K=k_inv-1)

    with computation(FORWARD), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                dp = p-p[0,0,1]


def surface_conditions(
    wmin: FloatFieldIJ,
    wmax: FloatFieldIJ,
    p: FloatField,
    exfh: FloatField,
    exf: FloatField,
    wthv: FloatFieldIJ,
    pblh: FloatFieldIJ,
    tmp: FloatFieldIJ,
    phis: FloatFieldIJ,
    ztop: FloatFieldIJ,
    wqt: FloatFieldIJ,
    wthl: FloatFieldIJ,
):
    from __externals__ import AlphaW, AlphaQT, AlphaTH, pwmin, pwmax

    with computation(PARALLEL), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                exfh=(p/constants.MAPL_P00)**constants.MAPL_KAPPA
                exf=(0.5*(p[0,0,1]+p)/constants.MAPL_P00)**constants.MAPL_KAPPA

    with computation(FORWARD), interval(-1,None):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                exfh[0,0,1]=(p[0,0,1]/constants.MAPL_P00)**constants.MAPL_KAPPA
            
    with computation(FORWARD), interval(0,1):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                wstar: FloatFieldIJ=max(constants.wstarmin,(constants.MAPL_GRAV*wthv*pblh/300.)**(1./3.))
                qstar: FloatFieldIJ=max(0.,wqt)/wstar
                thstar: FloatFieldIJ=max(0.,wthl)/wstar

                sigmaW: FloatFieldIJ=AlphaW*wstar
                sigmaQT: FloatFieldIJ=AlphaQT*qstar
                sigmaTH: FloatFieldIJ=AlphaTH*thstar

                wmin=sigmaW*pwmin
                wmax=sigmaW*pwmax

def identify_inversions(
    wthv: FloatFieldIJ,
    tmp: FloatFieldIJ,
    tmp2: FloatFieldIJ,
    phis: FloatFieldIJ,
    ztop: FloatFieldIJ,
    zlo: FloatField,
    t3: FloatField,
    thv: FloatField,
    wcfac: FloatField,
):
    from __externals__ import k_end, k_start, WCTHRESH

    with computation(FORWARD), interval(0,1):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                stop_loop: BoolFieldIJ = False
                tmp2 = 0.
                kidx: IntFieldIJ = k_start

    with computation(PARALLEL), interval(...):
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                wcfac = 0.

    with computation(FORWARD), interval(0,1):
         if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                while zlo.at(K=kidx) < 1500. and stop_loop==False:
                    if t3.at(K=k_end-kidx-1) > t3.at(K=k_end-kidx):
                        tmp2 = thv.at(K=kidx)  
                        stop_loop=True
                    kidx = kidx+1
                    ktmp: IntFieldIJ = kidx

    with computation(FORWARD), interval(...):
         if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            if ztop > 100.0:
                if tmp2 != 0.:
                    while zlo.at(K=ktmp-1) < zlo.at(K=kidx-1)+1e3:
                        ktmp = ktmp+1
                    if K <= kidx-1:
                        wcfac = min(10.,max(0.,thv.at(K=ktmp-1)-thv.at(K=kidx-1)-WCTHRESH))*exp(-(zlo.at(K=kidx-1)-zlo)/200. )

def define_surface_properties(
    nup2: IntFieldIJ,
    qt: FloatField,
    sigmaQT: FloatFieldIJ,
    sigmaTH: FloatFieldIJ,
    sigmaW: FloatFieldIJ,
    thv: FloatField,
    u: FloatField,
    v: FloatField,
    wmax: FloatFieldIJ,
    wmin: FloatFieldIJ,
    wthv: FloatFieldIJ,
    UPA: FloatField_UpdraftProperties,
    UPQT: FloatField_UpdraftProperties,
    UPTHV: FloatField_UpdraftProperties,
    UPU: FloatField_UpdraftProperties,
    UPV: FloatField_UpdraftProperties,
    UPW: FloatField_UpdraftProperties,
):
    from __externals__ import UPABUOYDEP

    with computation(FORWARD), interval(0,1):
        idx: IntFieldIJ = 1
        while idx <= nup2:
            wlv: FloatFieldIJ=wmin+(wmax-wmin)/(float32(nup2))*(float32(idx)-1.)
            wtv: FloatFieldIJ=wmin+(wmax-wmin)/(float32(nup2))*float32(idx)

            UPW[0,0,0][idx-1]=min(0.5*(wlv+wtv), 5.)

            if UPABUOYDEP!=0:
                UPA[0,0,0][idx-1]=(0.5+0.5*tanh((wthv-0.02)/0.09))*(0.5*erfc(wlv/(sqrt(2.)*sigmaW))-0.5*erfc(wtv/(sqrt(2.)*sigmaW)))
            else:
                UPA[0,0,0][idx-1]=(0.5*erfc(wlv/(sqrt(2.)*sigmaW))-0.5*erfc(wtv/(sqrt(2.)*sigmaW)))

            UPU[0,0,0][idx-1]=u
            UPV[0,0,0][idx-1]=v

            UPQT[0,0,0][idx-1]=qt+0.32*UPW[0,0,0][idx-1]*sigmaQT/sigmaW
            UPTHV[0,0,0][idx-1]=thv+0.58*UPW[0,0,0][idx-1]*sigmaTH/sigmaW

            idx = idx + 1





class RUN_EDMF(NDSLRuntime):
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: SHOCMFConfiguration,
        formulation: SaturationFormulation = SaturationFormulation.Staars,
    ) -> None:
        """
        RUN_EDMF

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


        self._setup_inputs = self.stencil_factory.from_dims_halo(
            func=setup_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._estimate_scale_height = self.stencil_factory.from_dims_halo(
            func=estimate_scale_height,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        self._flip_variables = self.stencil_factory.from_dims_halo(
            func=flip_variables,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"DISCRETE":config.DISCRETE}
        )

        self._surface_conditions = self.stencil_factory.from_dims_halo(
            func=surface_conditions,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"AlphaW":config.AlphaW, "AlphaQT":config.AlphaQT, "AlphaTH": config.AlphaTH, "pwmin": config.pwmin, "pwmax":config.pwmax}
        )

        self._identify_inversions = self.stencil_factory.from_dims_halo(
            func=identify_inversions,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"WCTHRESH":config.WCTHRESH}
        )

        self._define_surface_properties = self.stencil_factory.from_dims_halo(
            func=define_surface_properties,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"UPABUOYDEP":config.UPABUOYDEP}
        )




    def __call__(
        self,
        state: SHOCMFState,
        ):
        """
        RUN_EDMF 
        For NDSL-specific questions, email katrina.fandrich@nasa.gov

        ##############################################################################

        Arguments:
            state: SHOCMFState
        """

    
        # Reset locals

        # self._setup_inputs(
        #     pblh2=,
        #     thv3=,
        #     tke3=,
        #     wqt2=,
        #     wthl2=,
        #     zlo3=,
        #     zw3=,
        #     entx=,
        #     pblh=,
        #     tmp=,
        #     wqt=,
        #     wthl=,
        #     wthv=,
        # )

        #self._estimate_scale_height(
        #     wthv=,
        #     tmp=,
        #     phis=,
        #     UPW=,
        #     UPTHL=,
        #     UPTHV=,
        #     UPQT=,
        #     UPA=,
        #     UPU=,
        #     UPV=,
        #     UPQI=,
        #     UPQL=,
        #     ENT=,
        #     QR=,
        #     QS=,
        #     pw3=,
        #     t3=,
        #     wqt=,
        #     qv3=,
        #     zlo3=,
        #     nup2=,
        #     L0=,
        #     pmid=,
        #     esx=,
        #     ztop=,
        #     zw3=,
        #     z=,
        # )

        #self._flip_variables(
        #     wthv=,
        #     tmp=,
        #     phis=,
        #     ztop=,
        #     zlo3=,
        #     zw3=,
        #     zlo=,
        #     u3=,
        #     u=,
        #     v3=,
        #     v=,
        #     thl3=,
        #     thl=,
        #     thv3=,
        #     thv=,
        #     qv3=,
        #     qv=,
        #     ql3=,
        #     ql=,
        #     qi3=,
        #     qi=,
        #     ui=,
        #     vi=,
        #     thli=,
        #     qvi=,
        #     qli=,
        #     qii=,
        #     qt=,
        #     qti=,
        #     rhoe=,
        #     zw=,
        #     p=,
        #     dp=,
        #     rhoe3=,
        #     pw3=,
        # )

        #Poisson

        #self._surface_conditions(
        #     wmin=,
        #     wmax=,
        #     p=,
        #     exfh=,
        #     exf=,
        #     wthv=,
        #     pblh=,
        #     tmp=,
        #     phis=,
        #     ztop=,
        #     wqt=,
        #     wthl=,
        # )

        # self._identify_inversions(  
        #     wthv=,
        #     tmp=,
        #     tmp2=,
        #     phis=,
        #     ztop=,
        #     zlo=,
        #     t3=,
        #     thv=,
        #     wcfac=,
        # )

        # self._define_surface_properties(
        #     nup2=,
        #     qt=,
        #     sigmaQT=,
        #     sigmaTH=,
        #     sigmaW=,
        #     thv=,
        #     u=,
        #     v=,
        #     wmax=,
        #     wmin=,
        #     wthv=,
        #     UPA=,
        #     UPQT=,
        #     UPTHV=,
        #     UPU=,
        #     UPV=,
        #     UPW=,
        # )

