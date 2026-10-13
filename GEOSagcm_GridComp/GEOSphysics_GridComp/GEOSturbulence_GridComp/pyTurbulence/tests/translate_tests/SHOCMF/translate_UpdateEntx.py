from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyTurbulence.SHOCMF.edmf import update_entx
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table


class TranslateUpdateEntx(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # Inputs
        self.in_vars["data_vars"] = {
            "ENT": {},
            "UPA": {},
            "entx": {},
            "nup2": {},
            "wthv": {},
            "phis": {},
            "tmp": {},
        }

        # Outputs
        self.out_vars = {
            "entx": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants_edmf = data_loader.load("EDMF-constants")
        self.constants_shoc = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(
            **self.constants_edmf,
            **self.constants_shoc
        )

        self.quantity_factory.add_data_dimensions(
            {
                "number_of_properties": int(config.NUP),
            }
        )

        _update_entx = self.stencil_factory.from_dims_halo(
            func=update_entx,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        ENT = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(ENT.view[:], inputs["ENT"])

        UPA = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPA.view[:], inputs["UPA"])

        nup2 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
            dtype=Int,
        )
        safe_assign_array(nup2.view[:, :], inputs["nup2"])

        entx = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(entx.view[:, :], inputs["entx"])

        tmp = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(tmp.view[:, :], inputs["tmp"])

        wthv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wthv.view[:, :], inputs["wthv"])

        phis = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(phis.view[:, :], inputs["phis"])


        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esx =  saturation_vapor_pressure_table.esx

        total = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        total1 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )


        _update_entx(
            nup2=nup2,
            UPA=UPA,
            entx=entx,
            ENT=ENT,
            total=total,
            total1=total1,
        )

        return {
            "entx": entx.view[:],
        }