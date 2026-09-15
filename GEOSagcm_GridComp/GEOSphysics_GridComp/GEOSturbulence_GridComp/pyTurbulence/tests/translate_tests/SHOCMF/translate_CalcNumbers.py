from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import calc_numbers
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateCalcNumbers(TranslateFortranData2Py):
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
            "thv": {},
            "tke_mf": {},
            "u": {},
            "v": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "PRNUM": self.grid.compute_dict(),
            "RI": self.grid.compute_dict(),
        }

        def extra_data_load(self, data_loader: DataLoader):
            self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _calc_numbers = self.stencil_factory.from_dims_halo(
            func=calc_numbers,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"PRNUMBER": config.PRNUMBER}
        )

        # Inputs
        adzi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(adzi.view[:, :, :], inputs["adzi"])
        thv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(thv.view[:, :, :], inputs["thv"])
        tke_mf = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(tke_mf.view[:, :, :], inputs["tke_mf"])
        u = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(u.view[:, :, :], inputs["u"])
        v = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(v.view[:, :, :], inputs["v"])

        # Outputs
        PRNUM = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        RI = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
   
        _calc_numbers(
            u=u,
            v=v,
            adzi=adzi,
            RI=RI,
            PRNUM=PRNUM,
            thv=thv,
            tke_mf=tke_mf,
        )

        return {
            "PRNUM": PRNUM.view[:],
            "RI": RI.view[:],
        }
