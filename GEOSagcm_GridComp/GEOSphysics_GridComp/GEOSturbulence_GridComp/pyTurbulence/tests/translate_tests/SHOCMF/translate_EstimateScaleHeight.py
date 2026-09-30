from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.edmf import estimate_scale_height
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateEstimateScaleHeight(TranslateFortranData2Py):
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
            "qv3": {},
            "t3": {},
            "tmp": {},
            "wqt": {},
            "wthv": {},
            "zlo3": {},
            "pw3": {},
            "zw3_ESH": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "L0_ESH": self.grid.compute_dict(),
            "nup2": self.grid.compute_dict(),
            "pmid": self.grid.compute_dict(),
            "ztop": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)

        _estimate_scale_height = self.stencil_factory.from_dims_halo(
            func=estimate_scale_height,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        phis = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(phis.view[:, :], inputs["phis"])
        qv3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qv3.view[:, :, :], inputs["qv3"])
        t3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(t3.view[:, :, :], inputs["t3"])
        tmp = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(tmp.view[:, :], inputs["tmp"])
        wqt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wqt.view[:, :], inputs["wqt"])
        wthv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(wthv.view[:, :], inputs["wthv"])
        zlo3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zlo3.view[:, :, :], inputs["zlo3"])
        pw3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(pw3.view[:, :, :], inputs["pw3"])
        zw3 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(zw3.view[:, :, :], inputs["zw3_ESH"])



        # Outputs
        L0 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        nup2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        pmid = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        ztop = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
   
        _estimate_scale_height(
            
        )

        return {
            "L0": L0.view[:],
            "nup2": nup2.view[:],
            "pmid": pmid.view[:],
            "ztop": ztop.view[:],
        }
