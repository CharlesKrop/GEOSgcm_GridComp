from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.qsat_test import qsat_test
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables.formulation import SaturationFormulation
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int


class TranslateQSatTest(TranslateFortranData2Py):
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
            "p": {},
            "t": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "qsatliq_test": self.grid.compute_dict(),
            "qsatice_test": self.grid.compute_dict(),
            "qsat_test": self.grid.compute_dict(),
            "dqsat_test": self.grid.compute_dict(),
            "dtqw": self.grid.compute_dict(),
            "dtqi": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _qsat_test = self.stencil_factory.from_dims_halo(
            func=qsat_test,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        p_test = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(p_test.view[:, :], inputs["p"])
        t_test = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(t_test.view[:, :], inputs["t"])


        # Outputs
        qsat = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        dqsat= QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        qsat_ice_test = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        qsat_liq_test = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        dqsat_ice_test = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        dqsat_liq_test = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")

        formulation = Int(1) # Fix this later

        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.ese = saturation_vapor_pressure_table.ese
        self.frz = saturation_vapor_pressure_table.frz
        self.esw = saturation_vapor_pressure_table.esw
        self.lqu = saturation_vapor_pressure_table.lqu
        self.esx =  saturation_vapor_pressure_table.esx
   
        _qsat_test(
            t=t_test,
            p=p_test,
            qsat=qsat,
            dqsat=dqsat,
            qsat_liq=qsat_liq_test,
            qsat_ice=qsat_ice_test,
            dqsat_liq=dqsat_liq_test,
            dqsat_ice=dqsat_ice_test,
            ese=self.ese,
            esw=self.esw,
            esx=self.esx,
            frz=self.frz,
            lqu=self.lqu,
            formulation=formulation,
        )

        return {
            "qsat_test": qsat.view[:],
            "dqsat_test": dqsat.view[:],
            "qsatliq_test": qsat_liq_test.view[:],
            "qsatice_test": qsat_ice_test.view[:],
            "dtqw": dqsat_liq_test.view[:],
            "dtqi": dqsat_ice_test.view[:],
        }
