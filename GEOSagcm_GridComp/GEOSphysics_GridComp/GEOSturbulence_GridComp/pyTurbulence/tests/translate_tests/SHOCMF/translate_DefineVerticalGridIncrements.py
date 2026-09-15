from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import define_vertical_grid_increments


class TranslateDefineVerticalGridIncrements(TranslateFortranData2Py):
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
            "zi": {},
            "zl": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "adzi": self.grid.compute_dict(),
            "adzl": self.grid.compute_dict(),
        }

    def compute(self, inputs):

        _define_vertical_grid_increments = self.stencil_factory.from_dims_halo(
            func=define_vertical_grid_increments,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        zi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(zi.view[:, :, :], inputs["zi"])
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zl.view[:, :, :], inputs["zl"])

        # Outputs
        adzi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        adzl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
   
        _define_vertical_grid_increments(
            adzi=adzi,
            zl=zl,
            adzl=adzl,
            zi=zi,
        )

        return {
            "adzi": adzi.view[:],
            "adzl": adzl.view[:],
        }
