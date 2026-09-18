from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import flip_and_export
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateFlipAndExport(TranslateFortranData2Py):
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
            "isotropy": {},
            "prnum": {},
            "tke": {},
            "tkesbdiss": {},
            "tkh": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "tke_inv": self.grid.compute_dict(),
            "isotropy_inv": self.grid.compute_dict(),
            "tkesbdiss_inv": self.grid.compute_dict(),
            "tkh_inv": self.grid.compute_dict(),
            "tkm_inv": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _flip_and_export = self.stencil_factory.from_dims_halo(
            func=flip_and_export,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        isotropy = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(isotropy.view[:, :, :], inputs["isotropy"])
        prnum = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(prnum.view[:, :,:], inputs["prnum"])
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke.view[:, :, :], inputs["tke"])
        tkesbdiss = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tkesbdiss.view[:, :, :], inputs["tkesbdiss"])
        tkh = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tkh.view[:, :,:], inputs["tkh"]) 

        # Outputs
        tke_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkesbdiss_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkh_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkm_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        isotropy_inv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")

        _flip_and_export(
            tkh_inv=tkh_inv,
            tkm_inv=tkm_inv,
            isotropy_inv=isotropy_inv,
            tke_inv=tke_inv,
            tkesbdiss=tkesbdiss,
            tkh=tkh,
            prnum=prnum,
            isotropy=isotropy,
            tke=tke,
            tkesbdiss_inv=tkesbdiss_inv,
        )

        return {
            "tke_inv": tke_inv.view[:],
            "tkesbdiss_inv": tkesbdiss_inv.view[:],
            "tkh_inv": tkh_inv.view[:],
            "tkm_inv": tkm_inv.view[:],
            "isotropy_inv": isotropy_inv.view[:],
        }
