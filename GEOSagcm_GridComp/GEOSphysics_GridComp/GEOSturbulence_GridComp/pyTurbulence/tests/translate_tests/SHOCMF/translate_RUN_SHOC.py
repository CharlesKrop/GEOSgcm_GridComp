from f90nml import Namelist
from ndsl import QuantityFactory, StencilFactory
from ndsl.stencils.testing.savepoint import DataLoader
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from ndsl.utils import safe_assign_array

from pyTurbulence.SHOCMF.shoc import RUN_SHOC
from pyTurbulence.SHOCMF.config import SHOCMFConfiguration
from pyTurbulence.SHOCMF.state import SHOCMFState
from pyTurbulence.SHOCMF.locals import SHOCMFLocals
from pyMoist.saturation_tables.formulation import SaturationFormulation
from ndsl.dsl.typing import Bool, BoolFieldIJ, FloatField, FloatFieldIJ, IntField, IntFieldIJ, Int


class TranslateRUN_SHOC(TranslateFortranData2Py):
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
            "BUOYF": {},
            "DRYCBLH": {},
            "FCLD": {},
            "MFTKE": {},
            "OMEGA": {},
            "PLE": {},
            "Q": {},
            "QGTOT": {},
            "QITOT": {},
            "QLTOT": {},
            "QPI": {},
            "QPL": {},
            "QRTOT": {},
            "QSTOT": {},
            "SH": {},
            "T": {},
            "TKESHOC": {},
            "TKH": {},
            "U": {},
            "V": {},
            "WTHV2": {},
            "Z": {},
            "ZLE": {},
        }

        # FloatField Outputs
        self.out_vars = {
            "ISOTROPY": self.grid.compute_dict(),
            "KM": self.grid.compute_dict(),
            "TKEDISS": self.grid.compute_dict(),
            "TKESHOC": self.grid.compute_dict(),
            "TKH": self.grid.compute_dict(),
            "RI": self.grid.compute_dict(),
            "BRUNTSHOC": self.grid.compute_dict(),
        }

    def extra_data_load(self, data_loader: DataLoader):
        self.constants = data_loader.load("SHOCMF-constants")

    def compute(self, inputs):
        config = SHOCMFConfiguration(**self.constants)

        state = SHOCMFState.zeros(
            self.quantity_factory,
        )

        run_shoc = RUN_SHOC(
            self.stencil_factory,
            self.quantity_factory,
            config,
            
        )

        # Inputs
        state.input.BUOYF.field[:] = inputs["BUOYF"]
        state.input.DRYCBLH.field[:] = inputs["DRYCBLH"]
        state.input.FCLD.field[:] = inputs["FCLD"]
        state.input.MFTKE.field[:] = inputs["MFTKE"]
        state.input.OMEGA.field[:] = inputs["OMEGA"]
        state.input.PLE.field[:] = inputs["PLE"]
        state.input.Q.field[:] = inputs["Q"]
        state.input.QGTOT.field[:] = inputs["QGTOT"]
        state.input.QITOT.field[:] = inputs["QITOT"]
        state.input.QLTOT.field[:] = inputs["QLTOT"]
        state.input.QPL.field[:] = inputs["QPL"]
        state.input.QPI.field[:] = inputs["QPI"]
        state.input.QRTOT.field[:] = inputs["QRTOT"]
        state.input.QSTOT.field[:] = inputs["QSTOT"]
        state.input.SH.field[:] = inputs["SH"]
        state.input.T.field[:] = inputs["T"]
        state.input.U.field[:] = inputs["U"]
        state.input.V.field[:] = inputs["V"]
        state.input.WTHV2.field[:] = inputs["WTHV2"]
        state.input.Z.field[:] = inputs["Z"]
        state.input.ZLE.field[:] = inputs["ZLE"]
        
        # In/outs
        state.input_output.TKESHOC.field[:] = inputs["TKESHOC"]
        state.input_output.TKH.field[:] = inputs["TKH"]

        run_shoc(
            state,
            )

        return {
            "ISOTROPY": state.output.ISOTROPY.view[:],
            "KM": state.output.KM.view[:],
            "TKEDISS": state.output.TKEDISS.view[:],
            "TKESHOC": state.input_output.TKESHOC.view[:],
            "TKH": state.input_output.TKH.view[:],
            "RI": state.output.RI.view[:],
            "BRUNTSHOC": state.output.BRUNTSHOC.view[:],
        }
