from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import flip_output
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateDiagnostics(TranslateFortranData2Py):
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
            "brunt": {},
            "ri": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "ri_inv": self.grid.compute_dict(),
            "bruntmst_inv": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _flip_output = self.stencil_factory.from_dims_halo(
            func=flip_output,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        brunt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(brunt.view[:, :, :], inputs["brunt"])
        ri = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(ri.view[:, :,:], inputs["ri"])

        # Outputs
        bruntmst_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        ri_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
      
        _flip_output(
            input=brunt,
            output=bruntmst_inv,
        )

        _flip_output(
            input=ri,
            output=ri_inv,
        )

        return {
            "ri_inv": ri_inv.view[:],
            "bruntmst_inv": bruntmst_inv.view[:],
        }
