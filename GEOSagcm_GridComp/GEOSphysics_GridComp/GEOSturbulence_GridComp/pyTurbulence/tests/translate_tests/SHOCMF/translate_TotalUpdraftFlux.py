from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyTurbulence.SHOCMF.edmf import total_updraft_flux
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table


class TranslateTotalUpdraftFlux(TranslateFortranData2Py):
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
            "UPW": {},
            "UPA": {},
            "UPQT": {},
            "UPTHV": {},
            "nup2": {},
            "wthv": {},
            "wqt": {},
            "qt": {},
            "thv": {},
            "p": {},
        }

        # Outputs
        self.out_vars = {
            "UPTHV": self.grid.compute_dict(),
            "UPQT": self.grid.compute_dict(),
            "UPQL": self.grid.compute_dict(),
            "UPQI": self.grid.compute_dict(),
            "UPTHL": self.grid.compute_dict(),
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

        _total_updraft_flux = self.stencil_factory.from_dims_halo(
            func=total_updraft_flux,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"ICERAMP":config.ICERAMP},
        )

        # Inputs
        UPW = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPW.view[:], inputs["UPW"])

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

        UPQT = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPQT.view[:], inputs["UPQT"])

        UPTHV = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )
        safe_assign_array(UPTHV.view[:], inputs["UPTHV"])

        nup2 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
            dtype=Int,
        )
        safe_assign_array(nup2.view[:, :], inputs["nup2"])

        wthv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wthv.view[:, :], inputs["wthv"])

        wqt = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(wqt.view[:, :], inputs["wqt"])

        qt = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(qt.view[:], inputs["qt"])

        thv = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(thv.view[:], inputs["thv"])

        p = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(p.view[:], inputs["p"])

        # Outputs
        UPTHL = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
                "number_of_properties",
            ],
            units="n/a",
        )

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

        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esx =  saturation_vapor_pressure_table.esx

        _total_updraft_flux(
            UPW=UPW,
            UPA=UPA,
            UPQT=UPQT,
            UPTHV=UPTHV,
            nup2=nup2,
            wthv=wthv,
            wqt=wqt,
            qt=qt,
            thv=thv,
            p=p,
            UPTHL=UPTHL,
            UPQL=UPQL,
            UPQI=UPQI,
            esx=self.esx,
        )

        return {
            "UPTHV": UPTHV.view[:],
            "UPQT": UPQT.view[:],
            "UPQL": UPQL.view[:],
            "UPQI": UPQI.view[:],
            "UPTHL": UPTHL.view[:],
        }