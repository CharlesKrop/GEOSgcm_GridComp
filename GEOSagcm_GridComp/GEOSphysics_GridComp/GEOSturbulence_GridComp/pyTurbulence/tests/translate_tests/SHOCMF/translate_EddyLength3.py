from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import eddy_length3
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateEddyLength3(TranslateFortranData2Py):
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
            "brunt2": {},
            "dryzpbl": {},
            "thv": {},
            "tke": {},
            "zl": {},
            "smixt": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "smixt": self.grid.compute_dict(),
            "smixt1": self.grid.compute_dict(),
            "smixt2": self.grid.compute_dict(),
            "smixt3": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _eddy_length3 = self.stencil_factory.from_dims_halo(
            func=eddy_length3,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"LENOPT":config.LENOPT, "LENFAC1":config.LENFAC1, "LENFAC2":config.LENFAC2,"LENFAC3":config.LENFAC3}
        )

        # Inputs
        brunt2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(brunt2.view[:, :, :], inputs["brunt2"])
        dryzpbl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(dryzpbl.view[:, :], inputs["dryzpbl"])
        thv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(thv.view[:, :, :], inputs["thv"])
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke.view[:, :, :], inputs["tke"])
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zl.view[:, :,:], inputs["zl"]) 
        smixt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(smixt.view[:, :,:], inputs["smixt"]) 

        # Outputs
        smixt1 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        smixt2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        smixt3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")


        _eddy_length3(
            tke=tke,
            thv=thv,
            zl=zl,
            dryzpbl=dryzpbl,
            brunt2=brunt2,
            smixt=smixt,
            smixt1=smixt1,
            smixt2=smixt2,
            smixt3=smixt3,
        )

        return {
            "smixt": smixt.view[:],
            "smixt1": smixt1.view[:],
            "smixt2": smixt2.view[:],
            "smixt3": smixt3.view[:],
        }
