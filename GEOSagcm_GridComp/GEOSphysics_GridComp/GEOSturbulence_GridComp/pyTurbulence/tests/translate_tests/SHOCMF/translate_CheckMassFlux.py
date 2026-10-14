from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array
from ndsl.dsl.typing import Int

from pyTurbulence.SHOCMF.edmf import check_mass_flux
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table


class TranslateCheckMassFlux(TranslateFortranData2Py):
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
            "UPW": {},
            "nup2": {},
            "dp": {},
            "rhoe": {},
            "dt": {},
            "qr": {},
            "qs": {},
        }

        # Outputs
        self.out_vars = {
            "UPA": self.grid.compute_dict(),
            "qr": self.grid.compute_dict(),
            "qs": self.grid.compute_dict(),
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

        _check_mass_flux = self.stencil_factory.from_dims_halo(
            func=check_mass_flux,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={"MFLIMFAC":config.MFLIMFAC}, 
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

        nup2 = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
            dtype=Int,
        )
        safe_assign_array(nup2.view[:, :], inputs["nup2"])

        dp = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_DIM],
            units="n/a",
        )
        safe_assign_array(dp.view[:, :], inputs["dp"])

        rhoe= QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            units="n/a",
        )
        safe_assign_array(rhoe.view[:, :,:], inputs["rhoe"])

        dt = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM],
            units="n/a",
        )
        safe_assign_array(dt.view[:, :], inputs["dt"])

        qr = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_DIM],
            units="n/a",
        )
        safe_assign_array(qr.view[:, :], inputs["qr"])

        qs = QuantityFactory.zeros(
            self.quantity_factory,
            dims=[I_DIM, J_DIM,K_DIM],
            units="n/a",
        )
        safe_assign_array(qs.view[:, :], inputs["qs"]) 


        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esx =  saturation_vapor_pressure_table.esx


        _check_mass_flux(
            nup2=nup2,
            rhoe=rhoe,
            UPA=UPA,
            UPW=UPW,
            dp=dp,
            dt=dt,
            qr=qr,
            qs=qs,
        )

        return {
            "UPA": UPA.view[:],
            "qr": qr.view[:],
            "qs": qs.view[:],
        }