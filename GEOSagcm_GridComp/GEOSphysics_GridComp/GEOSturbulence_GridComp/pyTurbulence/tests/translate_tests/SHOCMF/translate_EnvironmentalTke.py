from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import environmental_tke
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateEnvironmentalTke(TranslateFortranData2Py):
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
            "tscale1": {},
            "zl": {},
            "tke": {},
            "dryzpbl": {},
            "brunt_edge": {},
            "prnum": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "isotropy": self.grid.compute_dict(),
            "tkh": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _environmental_tke = self.stencil_factory.from_dims_halo(
            func=environmental_tke,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"shoc_lambda":config.shoc_lambda, "ck": config.ck}
        )

        # Inputs
        dryzpbl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(dryzpbl.view[:, :], inputs["dryzpbl"])
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke.view[:, :, :], inputs["tke"])
        tscale1 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tscale1.view[:, :, :], inputs["tscale1"])
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zl.view[:, :, :], inputs["zl"])
        prnum = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(prnum.view[:, :, :], inputs["prnum"])
        brunt_edge = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(brunt_edge.view[:, :, :], inputs["brunt_edge"])
        
        # Outputs
        isotropy = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkh = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
    
        _environmental_tke(
            tscale1=tscale1,
            zl=zl,
            tke=tke,
            dryzpbl=dryzpbl,
            brunt_edge=brunt_edge,
            prnum=prnum,
            tkh=tkh,
            isotropy=isotropy,
        )

        return {
            "isotropy": isotropy.view[:],
            "tkh": tkh.view[:],
        }
