from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Float, Int
from ndsl.dsl.gt4py import int32

from pyTurbulence.SHOCMF.edmf import scale_updraft
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration


class TranslateScaleUpdraft(TranslateFortranData2Py):
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
            "UPA": {},
            "UPW": {},
            "dp": {},
            "dt": {},
            "rhoe": {},
            "nup2": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "UPW": self.grid.compute_dict(),

        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants_edmf,**self.constants_shoc)


        self.quantity_factory.add_data_dimensions(
            {
                "number_of_properties": int(config.NUP),
            }
        )

        _scale_updraft = self.stencil_factory.from_dims_halo(
            func=scale_updraft,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        UPA = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        safe_assign_array(UPA.view[:], inputs["UPA"])
        UPW = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM, "number_of_properties"], units="n/a")
        safe_assign_array(UPW.view[:], inputs["UPW"])
        dp = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(dp.view[:, :], inputs["dp"])
        dt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(dt.view[:, :], inputs["dt"])
        rhoe = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(rhoe.view[:, :, :], inputs["rhoe"])
        nup2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a", dtype=Int)
        safe_assign_array(nup2.view[:, :], inputs["nup2"])

        _scale_updraft(
            rhoe=rhoe,
            UPA=UPA,
            UPW=UPW,
            dt=dt,
            dp=dp,
            nup2=nup2,
        )

        return {
            "UPW": UPW.view[:],
        }
