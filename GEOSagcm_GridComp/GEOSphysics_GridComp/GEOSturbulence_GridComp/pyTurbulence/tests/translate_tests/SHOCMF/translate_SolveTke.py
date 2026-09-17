from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import solve_tke
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateSolveTke(TranslateFortranData2Py):
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
            "adzl": {},
            "brunt": {},
            "def2": {},
            "prnum": {},
            "smixt": {},
            "thv": {},
            "tke": {},
            "tke_mf": {},
            "tkh": {},
            "wthv_mf": {},
            "wthv_sec": {},
            "zl": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "tke": self.grid.compute_dict(),
            "tkesbbuoy": self.grid.compute_dict(),
            "tkesbdiss": self.grid.compute_dict(),
            "tkesbshear": self.grid.compute_dict(),
            "tscale1": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _solve_tke = self.stencil_factory.from_dims_halo(
            func=solve_tke,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"max_tke":config.max_tke,"min_tke":config.min_tke,"nitr":config.nitr,"BUOYOPT":config.BUOYOPT, "Ce":config.Ce, "Ces":config.Ces, "dtn":config.dtn}
        )

        # Inputs
        adzl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(adzl.view[:, :, :], inputs["adzl"])
        brunt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(brunt.view[:, :, :], inputs["brunt"])
        def2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(def2.view[:, :, :], inputs["def2"])
        prnum = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(prnum.view[:, :, :], inputs["prnum"])
        smixt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(smixt.view[:, :], inputs["smixt"])
        thv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(thv.view[:, :, :], inputs["thv"])
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke.view[:, :, :], inputs["tke"])
        tke_mf = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(tke_mf.view[:, :, :], inputs["tke_mf"])
        tkh = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tkh.view[:, :, :], inputs["tkh"])
        wthv_mf = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(wthv_mf.view[:, :, :], inputs["wthv_mf"])
        wthv_sec = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(wthv_sec.view[:, :, :], inputs["wthv_sec"])
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zl.view[:, :, :], inputs["zl"])
 

        # Outputs
        tkesbbuoy = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkesbdiss = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tkesbshear = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        tscale1 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")

        _solve_tke(
            adzl=adzl,
            tkh=tkh,
            wthv_sec=wthv_sec,
            wthv_mf=wthv_mf,
            thv=thv,
            brunt=brunt,
            tke=tke,
            prnum=prnum,
            smixt=smixt,
            def2=def2,
            zl=zl,
            tke_mf=tke_mf,
            tkesbbuoy=tkesbbuoy,
            tkesbshear=tkesbshear,
            tkesbdiss=tkesbdiss,
            tscale1=tscale1,
        )

        return {
            "tke": tke.view[:],
            "tkesbbuoy": tkesbbuoy.view[:],
            "tkesbdiss": tkesbdiss.view[:],
            "tkesbshear": tkesbshear.view[:],
            "tscale1": tscale1.view[:],
        }
