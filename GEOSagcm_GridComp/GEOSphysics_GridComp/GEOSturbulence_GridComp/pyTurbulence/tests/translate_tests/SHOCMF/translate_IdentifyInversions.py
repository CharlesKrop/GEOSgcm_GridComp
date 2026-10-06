from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.edmf import identify_inversions
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateIdentifyInversions(TranslateFortranData2Py):
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
            "phis": {},
            "t3": {},
            "thv": {},
            "tmp_in": {},
            "wthv": {},
            "zlo": {},
            "ztop": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "tmp_out": self.grid.compute_dict(),
            "wcfac": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)

        self._identify_inversions = self.stencil_factory.from_dims_halo(
            func=identify_inversions,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"WCTHRESH":config.WCTHRESH}
        )

        # Inputs
        phis = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(phis.view[:, :], inputs["phis"])

        t3 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(t3.view[:, :,:], inputs["t3"])


        thv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(thv.view[:, :,:], inputs["thv"])

        tmp_in = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(tmp_in.view[:, :], inputs["tmp_in"])

        wthv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wthv.view[:, :], inputs["wthv"])

        zlo = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_DIM],
            units="n/a",
        )
        safe_assign_array(zlo.view[:, :], inputs["zlo"])

        ztop = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(ztop.view[:, :], inputs["ztop"])


        # Outputs
        wcfac = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        tmp = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
      

        self._identify_inversions(
            wthv=wthv,
            tmp=tmp,
            tmp_in=tmp_in,
            phis=phis,
            ztop=ztop,
            zlo=zlo,
            t3=t3,
            thv=thv,
            wcfac=wcfac,
        )

        return {
            "tmp_out": tmp.view[:],
            "wcfac": wcfac.view[:],
        }