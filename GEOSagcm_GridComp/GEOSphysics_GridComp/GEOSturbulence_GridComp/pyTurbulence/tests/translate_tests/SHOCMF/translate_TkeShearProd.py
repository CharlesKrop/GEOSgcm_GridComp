from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import tke_shear_prod
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateTkeShearProd(TranslateFortranData2Py):
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
            "adzi": {},
            "u": {},
            "v": {},
        }


        # FloatField Outputs
        self.out_vars = {
            "def2": self.grid.compute_dict(),
        }

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)
        
        _tke_shear_prod = self.stencil_factory.from_dims_halo(
            func=tke_shear_prod,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"dtn": config.dtn}
        )

        # Inputs
        adzi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(adzi.view[:, :, :], inputs["adzi"])
        u = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(u.view[:, :, :], inputs["u"])
        v = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(v.view[:, :, :], inputs["v"])

        # Outputs
        def2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        rdtn = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
   
        _tke_shear_prod(
            rdtn=rdtn,
            def2=def2,
            adzi=adzi,
            u=u,
            v=v,
        )

        return {
            "def2": def2.view[:],
        }
