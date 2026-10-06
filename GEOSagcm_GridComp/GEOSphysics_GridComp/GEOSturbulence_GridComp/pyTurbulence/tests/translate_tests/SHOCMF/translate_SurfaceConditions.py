from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.edmf import surface_conditions
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateSurfaceConditions(TranslateFortranData2Py):
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
            "pblh": {},
            "phis": {},
            "tmp": {},
            "wqt": {},
            "wthl": {},
            "wthv": {},
            "ztop": {},
            "p": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "exf": self.grid.compute_dict(),
            "exfh": self.grid.compute_dict(),
            "wmax": self.grid.compute_dict(),
            "wmin": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)

        self._surface_conditions = self.stencil_factory.from_dims_halo(
            func=surface_conditions,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"AlphaW":config.AlphaW, "AlphaQT":config.AlphaQT, "AlphaTH": config.AlphaTH, "pwmin": config.pwmin, "pwmax":config.pwmax}
        )

        # Inputs
        pblh = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(pblh.view[:, :], inputs["pblh"])

        phis = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(phis.view[:, :], inputs["phis"])


        tmp = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(tmp.view[:, :], inputs["tmp"])

        wqt = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wqt.view[:, :], inputs["wqt"])

        wthl = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wthl.view[:, :], inputs["wthl"])

        wthv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wthv.view[:, :], inputs["wthv"])

        ztop = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(ztop.view[:, :], inputs["ztop"])

        p = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(p.view[:, :, :], inputs["p"])

        

        # Outputs
        exf = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        exfh = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )

        qstar = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        sigmaQT = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )

        sigmaTH = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        sigmaW = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )

        thstar = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        wmax = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )

        wmin = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )

        wstar = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
      

        self._surface_conditions(
            pblh=pblh,
            wmin=wmin,
            wmax=wmax,
            p=p,
            exfh=exfh,
            exf=exf,
            wthv=wthv,
            tmp=tmp,
            phis=phis,
            ztop=ztop,
            wqt=wqt,
            wthl=wthl,
        )

        return {
            "wmin": wmin.view[:],
            "wmax": wmax.view[:],
            "exf": exf.view[:],
            "exfh": exfh.view[:],
        }