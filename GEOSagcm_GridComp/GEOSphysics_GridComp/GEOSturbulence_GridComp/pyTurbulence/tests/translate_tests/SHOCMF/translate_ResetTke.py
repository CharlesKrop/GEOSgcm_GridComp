from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import calc_numbers
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateResetTke(TranslateFortranData2Py):
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
            "tke": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "tke": self.grid.compute_dict(),
            "tkesbbuoy": self.grid.compute_dict(),
            "tkesbdiss": self.grid.compute_dict(),
            "tkesbshear": self.grid.compute_dict(),
        }

        def extra_data_load(self, data_loader: DataLoader):
            self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _reset_tke = self.stencil_factory.from_dims_halo(
            func=reset_tke,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"min_tke": config.min_tke}
        )

        # Inputs
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke.view[:, :, :], inputs["tke"])

        # Outputs
        tkesbdiss = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkesbshear = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkesbbuoy = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
   
        _reset_tke(
            tke=tke,
            tkesbdiss=tkesbdiss,
            tkesbshear=tkesbshear,
            tkesbbuoy=tkesbbuoy,
        )

        return {
            "tke": tke.view[:],
            "tkesbbuoy": tkesbbuoy.view[:],
            "tkesbdiss": tkesbdiss.view[:],
            "tkesbshear": tkesbshear.view[:],
        }
