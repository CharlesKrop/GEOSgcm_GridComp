from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.edmf import setup_inputs
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateSetupInputs(TranslateFortranData2Py):
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
            "pblh2": {},
            "thv3": {},
            "tke3": {},
            "wqt2": {},
            "wthl2": {},
            "zlo3": {},
            "zw3": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "entx": self.grid.compute_dict(),
            "pblh": self.grid.compute_dict(),
            "tmp": self.grid.compute_dict(),
            "wqt": self.grid.compute_dict(),
            "wthl": self.grid.compute_dict(),
            "wthv": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)

        _setup_inputs = self.stencil_factory.from_dims_halo(
            func=setup_inputs,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        pblh2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(pblh2.view[:, :], inputs["pblh2"])
        thv3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(thv3.view[:, :, :], inputs["thv3"])
        tke3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke3.view[:, :, :], inputs["tke3"])
        wqt2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wqt2.view[:, :], inputs["wqt2"])
        wthl2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wthl2.view[:, :], inputs["wthl2"])
        zlo3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zlo3.view[:, :, :], inputs["zlo3"])
        zw3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zw3.view[:, :, :], inputs["zw3"])


        # Outputs
        entx = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        pblh = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        tmp = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        wqt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        wthl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        wthv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
   
        _setup_inputs(
            pblh2=pblh2,
            thv3=thv3,
            tke3=tke3,
            wqt2=wqt2,
            wthl2=wthl2,
            zlo3=zlo3,
            zw3=zw3,
            entx=entx,
            pblh=pblh,
            tmp=tmp,
            wqt=wqt,
            wthl=wthl,
            wthv=wthv,
        )

        return {
            "entx": entx.view[:],
            "pblh": pblh.view[:],
            "tmp": tmp.view[:],
            "wqt": wqt.view[:],
            "wthl": wthl.view[:],
            "wthv": wthv.view[:],
        }
