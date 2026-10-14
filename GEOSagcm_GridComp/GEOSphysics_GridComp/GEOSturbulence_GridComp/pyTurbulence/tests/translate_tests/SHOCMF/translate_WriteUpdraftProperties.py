from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyTurbulence.SHOCMF.edmf import write_updraft_properties
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table


class TranslateWriteUpdraftProperties(TranslateFortranData2Py):
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
            "UPA": {},
            "UPQL": {},
            "UPQI": {},
            "nup2": {},
        }

        # Outputs
        self.out_vars = {
            "moist_a": self.grid.compute_dict(),
            "dry_a": self.grid.compute_dict(),
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

        _write_updraft_properties = self.stencil_factory.from_dims_halo(
            func=write_updraft_properties,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
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

        UPQL = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPQL.view[:], inputs["UPQL"])

        UPQI = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPQI.view[:], inputs["UPQI"])

        nup2 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
            dtype=Int,
        )
        safe_assign_array(nup2.view[:, :], inputs["nup2"])

    

        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esx =  saturation_vapor_pressure_table.esx

        moist_a = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_INTERFACE_DIM],
            units="n/a",
        )
        dry_a = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_INTERFACE_DIM],
            units="n/a",
        )


        _write_updraft_properties(
            dry_a=dry_a,
            moist_a=moist_a,
            nup2=nup2,
            UPQL=UPQL,
            UPQI=UPQI,
            UPA=UPA,
        )

        return {
            "moist_a": moist_a.view[:],
            "dry_a": dry_a.view[:],
        }