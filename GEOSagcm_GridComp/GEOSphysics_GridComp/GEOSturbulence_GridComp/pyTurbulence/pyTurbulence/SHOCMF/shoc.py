import dace
from ndsl import NDSLRuntime, OptimizationConfig, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, K, computation, erfc, exp, float32, int32, int64, interval, isnan, log, sqrt
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ

from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyTurbulence.SHOCMF.locals import SHOCMFLocals
from pyTurbulence.SHOCMF.UW.state import SHOCMFState
import pyTurbulence.constants as constants


def invert_interface_inputs(
    zi: FloatField,
    phii_inv: FloatField,
):

    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kinv = k_end-K+1
        zi = phii_inv.at(k=kinv)-phii_inv.at(K=k_end)

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

    from __externals__ import k_end

    with computation(PARALLEL), interval(...):
        kinv = k_end-K+1
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
    wrk: FloatField,
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
    qcl: FloatField,
    qci: FloatField,
    prespot: FloatField,
    tabs: FloatField,
    gamaz: FloatField,
    zl: FloatField,
    hl: FloatField,
):
    with computation(PARALLEL), interval(...):
        wrk = 1.0 / prsl
        qv = max(qwv, 0.0)
        thv = tabs * (1.0+constants.epsv*qv-qcl-qci)
        w = - constants.rog * omega * thv * wrk
        qpl = 0.0  # comment or remove when using with prognostic rain/snow
        qpi = 0.0  # comment or remove when using with prognostic rain/snow
        total_water = qcl + qci + qv
        prespot = (constants.MAPL_P00*wrk) ** constants.kapa        # Exner function
        bet = constants.ggr/(tabs*prespot)     # Moorthi
        thv = thv*prespot            # Moorthi

        # Lapse rate * height = reference temperature
        gamaz = constants.gocp * zl

        # Liquid/ice water static energy - ! Note the the units are degrees K
        hl = tabs + gamaz - constants.fac_cond*(qcl+qpl) - constants.fac_fus *(qci+qpi)

def define_vertical_grid_increments(
    adzi: FloatField,
    zl: FloatField,
    adzl: FloatField,
    zi: FloatField,
):
    from __externals__ import k_end

    # NOT SURE ABOUT THIS, NEEDS TO BE TESTED
    with computation(FORWARD), interval(1,None):
        adzi[0,0,1] = (zl - zl[0,0,-1])
        adzl[0,0,-1] = (zi[0,0,1] - zi)

    with computation(FORWARD), interval(0,1):
        adzi[0,0,1]   = (zl-zi[0,0,1]) 

    with computation(FORWARD), interval(-1,None):
        adzi[0,0,1]  = zi-zl[0,0,-1]
        adzl = adzi



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


        self._invert_interface_inputs = self.stencil_factory.from_dims_halo(
            func=invert_interface_inputs,
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



    def __call__(self, state: SHOCMFState):
        """
        RUN_SHOC 
        For NDSL-specific questions, email katrina.fandrich@nasa.gov

        ##############################################################################

        Arguments:
            state: SHOCMFState
        """

        self._invert_interface_inputs(
            zi=,
            phii_inv=,
        )

        self._invert_inputs(
            zl=,
            phil_inv=,
            phii_inv=,
            tkh=,
            tkh_inv=,
            prsl=,
            prsl_inv=,
            u=,
            v=,
            omega=,
            omega_inv=,
            tabs=,
            tabs_inv=,
            qwv=,
            qwv_inv=,
            qcl=,
            qc_inv=,
            qci=,
            qi_inv=,
            cld_sgs=,
            cld_sgs_inv=,
            tke=,
            tke_inv=,
            wthv_sec=,
            wthv_sec_inv=,
            wthv_mf=,
            wthv_mf_inv=,
        )

        self._setup_derived_inputs(
            wrk=,
            prsl=,
            qv=,
            qwv=,
            thv=,
            tabs=,
            qcl=,
            qci=,
            w=,
            omega=,
            qpl=,
            qpi=,
            total_water=,
            qcl=,
            qci=,
            prespot=,
            tabs=,
            gamaz=,
            zl=,
            hl=,
        )

        self._define_vertical_grid_increments(
            adzi=,
            zl=,
            adzl=,
            zi=,
        )


        









