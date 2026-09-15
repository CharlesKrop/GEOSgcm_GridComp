from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import eddy_length1
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateEddyLength1(TranslateFortranData2Py):
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
            "bet": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "betdz": self.grid.compute_dict(),
            "brunt": self.grid.compute_dict(),
            "smixt": self.grid.compute_dict(),
            "thedz": self.grid.compute_dict(),
        }

        def extra_data_load(self, data_loader: DataLoader):
            self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _eddy_length1 = self.stencil_factory.from_dims_halo(
            func=eddy_length1,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        adzi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(adzi.view[:, :, :], inputs["adzi"])
        bet = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(bet.view[:, :, :], inputs["bet"])

        # Outputs
        betdz = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        brunt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        smixt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        thedz = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
   
        _eddy_length1(
            bet=bet,
            betdz=betdz,
            smixt=smixt,
            brunt=brunt,
        )

        return {
            "betdz": betdz.view[:],
            "brunt": brunt.view[:],
            "smixt": smixt.view[:],
        }
