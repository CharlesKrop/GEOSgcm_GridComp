import dace
from ndsl import NDSLRuntime, OptimizationConfig, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, K, computation, erfc, exp, float32, int32, int64, interval, isnan, log, sqrt, tanh
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int, Float

from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyTurbulence.SHOCMF.locals import SHOCMFLocals
from pyTurbulence.SHOCMF.state import SHOCMFState
import pyTurbulence.constants as constants
from pyMoist.saturation_tables.tables.liquid_exact import liquid_exact
from pyMoist.saturation_tables.tables.ice_exact import ice_exact
from pyMoist.saturation_tables.saturation_specific_humidity_functions import saturation_specific_humidity_frozen_surface,saturation_specific_humidity_liquid_surface, saturation_specific_humidity
from pyMoist.saturation_tables.formulation import SaturationFormulation
from pyMoist.saturation_tables import GlobalTable_saturation_tables
from pyMoist.saturation_tables.tables.constants import IceExactConstants, LiquidExactConstants


def qsat_test(
    t: FloatFieldIJ,
    p: FloatFieldIJ,
    qsat: FloatFieldIJ,
    dqsat: FloatFieldIJ,
    qsat_liq: FloatFieldIJ,
    qsat_ice: FloatFieldIJ,
    dqsat_liq: FloatFieldIJ,
    dqsat_ice: FloatFieldIJ,
    frz: Float,
    lqu: Float,
    ese: GlobalTable_saturation_tables,
    esw: GlobalTable_saturation_tables,
    esx: GlobalTable_saturation_tables,
    formulation: Int,
):
    from __externals__ import k_end

    with computation(FORWARD), interval(0,1):
        qsat, dqsat = saturation_specific_humidity(t,p,esx)

        qsat_liq, dqsat_liq = saturation_specific_humidity_liquid_surface(esw,lqu,t,p)
        qsat_ice, dqsat_ice = saturation_specific_humidity_frozen_surface(ese,frz,t,p)

        # qsat_liq, dqsat_liq = liquid_exact(
        #     t,
        #     formulation,
        #     LiquidExactConstants.B6,
        #     LiquidExactConstants.B5,
        #     LiquidExactConstants.B4,
        #     LiquidExactConstants.B3,
        #     LiquidExactConstants.B2,
        #     LiquidExactConstants.B1,
        #     LiquidExactConstants.B0,
        #     IceExactConstants.BI6,
        #     IceExactConstants.BI5,
        #     IceExactConstants.BI4,
        #     IceExactConstants.BI3,
        #     IceExactConstants.BI2,
        #     IceExactConstants.BI1,
        #     IceExactConstants.BI0,
        #     IceExactConstants.S16,
        #     IceExactConstants.S15,
        #     IceExactConstants.S14,
        #     IceExactConstants.S13,
        #     IceExactConstants.S12,
        #     IceExactConstants.S11,
        #     IceExactConstants.S10,
        #     IceExactConstants.S26,
        #     IceExactConstants.S25,
        #     IceExactConstants.S24,
        #     IceExactConstants.S23,
        #     IceExactConstants.S22,
        #     IceExactConstants.S21,
        #     IceExactConstants.S20,
        #     LiquidExactConstants.DL_0,
        #     LiquidExactConstants.DL_1,
        #     LiquidExactConstants.DL_2,
        #     LiquidExactConstants.DL_3,
        #     LiquidExactConstants.DL_4,
        #     LiquidExactConstants.DL_5,
        #     LiquidExactConstants.TS,
        #     LiquidExactConstants.LOGPS,
        #     LiquidExactConstants.CL_0,
        #     LiquidExactConstants.CL_1,
        #     LiquidExactConstants.CL_2,
        #     LiquidExactConstants.CL_3,
        #     LiquidExactConstants.CL_4,
        #     LiquidExactConstants.CL_5,
        #     LiquidExactConstants.CL_6,
        #     LiquidExactConstants.CL_7,
        #     LiquidExactConstants.CL_8,
        #     LiquidExactConstants.CL_9,
        #     p,
        # )
        # qsat_ice, dqsat_ice = ice_exact(
        #         t,
        #         formulation,
        #         IceExactConstants.TMINSTR,
        #         IceExactConstants.TMINICE,
        #         IceExactConstants.TSTARR1,
        #         IceExactConstants.TSTARR2,
        #         IceExactConstants.TSTARR3,
        #         IceExactConstants.TSTARR4,
        #         IceExactConstants.TMAXSTR,
        #         IceExactConstants.DI_0,
        #         IceExactConstants.DI_1,
        #         IceExactConstants.DI_2,
        #         IceExactConstants.DI_3,
        #         IceExactConstants.CI_0,
        #         IceExactConstants.CI_1,
        #         IceExactConstants.CI_2,
        #         IceExactConstants.CI_3,
        #         IceExactConstants.S16,
        #         IceExactConstants.S15,
        #         IceExactConstants.S14,
        #         IceExactConstants.S13,
        #         IceExactConstants.S12,
        #         IceExactConstants.S11,
        #         IceExactConstants.S10,
        #         IceExactConstants.S26,
        #         IceExactConstants.S25,
        #         IceExactConstants.S24,
        #         IceExactConstants.S23,
        #         IceExactConstants.S22,
        #         IceExactConstants.S21,
        #         IceExactConstants.S20,
        #         IceExactConstants.BI6,
        #         IceExactConstants.BI5,
        #         IceExactConstants.BI4,
        #         IceExactConstants.BI3,
        #         IceExactConstants.BI2,
        #         IceExactConstants.BI1,
        #         IceExactConstants.BI0,
        #         p,
        #     )



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


        self.qsat_test = self.stencil_factory.from_dims_halo(
            func=qsat_test,
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


        # self._qsat_test(
        #     t=,
        #     p: FloatField,
        #     qsat_liq: FloatField,
        #     qsat_ice: FloatField,
        #     formulation: Int,
        # )











