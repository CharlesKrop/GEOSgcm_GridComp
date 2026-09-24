from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import eddy_length2
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyMoist.saturation_tables.formulation import SaturationFormulation
from pyMoist.saturation_tables import GlobalTable_saturation_tables, get_saturation_vapor_pressure_table
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int


class TranslateEddyLength2(TranslateFortranData2Py):
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
            "adzi": {},
            "bet": {},
            "brunt2": {},
            "cld_sgs": {},
            "dryzpbl": {},
            "hl": {},
            "prsl": {},
            "qci": {},
            "qcl": {},
            "qpi": {},
            "qpl": {},
            "tabs": {},
            "thv": {},
            "tke": {},
            "total_water": {},
            "zl": {},
            "dtqw": {},
            "dtqi": {},
            "qv": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "brunt": self.grid.compute_dict(),
            "brunt2": self.grid.compute_dict(),
            "brunt_edge": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        _eddy_length2 = self.stencil_factory.from_dims_halo(
            func=eddy_length2,
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )

        # Inputs
        adzi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")
        safe_assign_array(adzi.view[:, :, :], inputs["adzi"])
        bet = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(bet.view[:, :, :], inputs["bet"])
        brunt2 = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(brunt2.view[:, :, :], inputs["brunt2"])
        cld_sgs = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(cld_sgs.view[:, :, :], inputs["cld_sgs"])
        dryzpbl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM], units="n/a")
        safe_assign_array(dryzpbl.view[:, :], inputs["dryzpbl"])
        hl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(hl.view[:, :, :], inputs["hl"])
        prsl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(prsl.view[:, :, :], inputs["prsl"])
        qci = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qci.view[:, :, :], inputs["qci"])
        qcl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qcl.view[:, :, :], inputs["qcl"])
        qpi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qpi.view[:, :, :], inputs["qpi"])
        qpl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qpl.view[:, :, :], inputs["qpl"])
        tabs = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tabs.view[:, :, :], inputs["tabs"])
        thv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(thv.view[:, :, :], inputs["thv"])
        tke = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(tke.view[:, :, :], inputs["tke"])
        total_water = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(total_water.view[:, :, :], inputs["total_water"])
        zl = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(zl.view[:, :, :], inputs["zl"])
        dtqw = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(dtqw.view[:, :, :], inputs["dtqw"])
        dtqi = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(dtqi.view[:, :, :], inputs["dtqi"])
        qv = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        safe_assign_array(qv.view[:, :, :], inputs["qv"])
          

        # Outputs
        brunt = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_DIM], units="n/a")
        brunt_edge = QuantityFactory.zeros(self.quantity_factory, dims=[I_DIM, J_DIM, K_INTERFACE_DIM], units="n/a")

        formulation = Int(1)

        saturation_vapor_pressure_table = get_saturation_vapor_pressure_table(self.stencil_factory)
        self.esw = saturation_vapor_pressure_table.esw

   
        _eddy_length2(
            adzi=adzi,
            bet=bet,
            qcl=qcl,
            qci=qci,
            tabs=tabs,
            prsl=prsl,
            dtqw=dtqw,
            dtqi=dtqi,
            qpl=qpl,
            qpi=qpi,
            cld_sgs=cld_sgs,
            hl=hl,
            total_water=total_water,
            brunt=brunt,
            brunt2=brunt2,
            brunt_edge=brunt_edge,
            qv=qv,
            zl=zl,
            dryzpbl=dryzpbl,
            formulation=formulation,
        )

        return {
            "brunt": brunt.view[:],
            "brunt2": brunt2.view[:],
            "brunt_edge": brunt_edge.view[:],
        }
