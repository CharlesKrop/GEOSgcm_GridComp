from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import setup_preliminary_inputs
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateSetupPreliminaryInputs(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # FloatField Inputs
        self.in_vars["data_vars"] = {
            "FCLD": {},
            "PLE": {},
            "QGTOT": {},
            "QITOT": {},
            "QLTOT": {},
            "QRTOT": {},
            "QSTOT": {},
            "ZLE": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "PLO": self.grid.compute_dict(),
            "QA": self.grid.compute_dict(),
            "QI": self.grid.compute_dict(),
            "QL": self.grid.compute_dict(),
            "Z": self.grid.compute_dict(),
            "ZL0": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_shoc)

        _setup_preliminary_inputs = self.stencil_factory.from_dims_halo(
            func=setup_preliminary_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        FCLD = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(FCLD.view[:, :], inputs["FCLD"])
        PLE = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(PLE.view[:, :, :], inputs["PLE"])
        QGTOT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(QGTOT.view[:, :, :], inputs["QGTOT"])
        QITOT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(QITOT.view[:, :], inputs["QITOT"])
        QLTOT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(QLTOT.view[:, :], inputs["QLTOT"])
        QRTOT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(QRTOT.view[:, :, :], inputs["QRTOT"])
        QSTOT = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(QSTOT.view[:, :, :], inputs["QSTOT"])
        ZLE = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(ZLE.view[:, :, :], inputs["ZLE"])


        # Outputs
        PLO = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        QA = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_DIM], units="n/a")
        QI = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_DIM], units="n/a")
        QL = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_DIM], units="n/a")
        Z = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_DIM], units="n/a")
        ZL0 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM,K_INTERFACE_DIM], units="n/a")
   
        _setup_preliminary_inputs(
            ZLE=ZLE,
            ZL0=ZL0,
            QLTOT=QLTOT,
            QRTOT=QRTOT,
            QL=QL,
            QI=QI,
            QITOT=QITOT,
            QSTOT=QSTOT,
            QGTOT=QGTOT,
            QA=QA,
            FCLD=FCLD,
            Z=Z,
            PLE=PLE,
            PLO=PLO,
        )

        return {
            "PLO": PLO.view[:],
            "QA": QA.view[:],
            "QI": QI.view[:],
            "QL": QL.view[:],
            "Z": Z.view[:],
            "ZL0": ZL0.view[:],
        }
