from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import invert_interface_vars


class TranslateInvertInterfaceVars(TranslateFortranData2Py):
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
            "phii_inv": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "zi": self.grid.compute_dict(),
        }

    def compute(self, inputs):

        _invert_interface_vars = self.stencil_factory.from_dims_halo(
            func=invert_interface_vars,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        phii_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(phii_inv.view[:, :, :], inputs["phii_inv"])

        # Outputs
        zi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
   
        _invert_interface_vars(
            zi=zi,
            phii_inv=phii_inv,
        )

        return {
            "zi": zi.view[:],
        }
