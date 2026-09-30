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
from pyMoist.saturation_tables.formulation import SaturationFormulation
from pyMoist.saturation_tables.tables.constants import IceExactConstants, LiquidExactConstants


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

):
    from __externals__ import NUP, ET, L0_EDMF, L0fac, k_end

    with computation(PARALLEL), interval(...):
        stop_loop = False
        if wthv > 0.0 and tmp>0.05 and phis < 3e4:
            nup2 = NUP
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

    #     if ET == 2:
    #         pmid = 0.5*(pw3[0,0,-1]+pw3)
    #         call calc_mf_depth(t3,zlo3(IH,JH,:)-zw3(IH,JH,kte),qv3(IH,JH,:),pmid,ztop,wthv,wqt)
    #         wstar=max(0.1,(mapl_grav*wthv*1e3/t(kte))**(1./3.)) 
    #         thstar=max(0.,wthv)/wstar

    #         sigmaQT=2.0*qstar
    #         sigmaTH=2.0*thstar

    #         tep  = t3.at(K=k_end)+max(0.1,sigmaTH) 
    #         qp   = qv3.at(K=k_end)+sigmaQT

    #         t1   = t3.at(K=k_end)
    #         z1   = zlo.at(K=k_end)
    #         ztop = zlo.at(K=k_end)
    #     else: 
    #         L0 = L0_EDMF

    # with computation(BACKWARD), interval(1,-1):
    #     if ET == 2 and stop_loop == False:
    #         z2 = zlo
    #         t2 = t3
    #         pp = pmid

    #         tep   = tep - constants.MAPL_GRAV*( z2-z1 )/constants.MAPL_CP

    #         qp    = qp  + (0.7/1000.)*(z2-z1)*(qv3-qp)  
    #         tep   = tep + (0.7/1000.)*(z2-z1)*(t3-tep)

    #         _, dqsp  = GEOS_DQSAT(tep , pp , qsat=qsp,  pascals=.true. )

    #         dqp   = max( qp - qsp, 0. )/(1.+(MAPL_ALHL/MAPL_CP)*dqsp )
    #         qp    = qp - dqp
    #         tep   = tep  + MAPL_ALHL * dqp/MAPL_CP

    #         if ( t2*(1.+MAPL_VIREPS*q(k)) .ge. tep*(1.+MAPL_VIREPS*qp)+0.2 ) then
    #             ztop = 0.5*(z2+z1)
    #             stop_loop = True

    #         if stop_loop == False:
    #             z1 = z2
    #             t1 = t2

    # with computation(PARALLEL), interval(...):
    #     if ET == 2:
    #         L0 = max(min(ztop,2500.),500.) / L0fac

    #         #NOTE: Need to fix this
    #         #if (associated(mfdepth)) mfdepth(IH,JH) = ztop
    
    




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

        #self._estimate_scale_height()