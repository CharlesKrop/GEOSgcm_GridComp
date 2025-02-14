import numpy as np

from ndsl import Namelist, Quantity, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from pyMoist.convective_parameterization.GF2020.


class TranslateGF2020(TranslateFortranData2Py):
    def __init__(self, grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory
        self._grid = grid

        # FloatField Inputs
        self.in_vars["data_vars"] = {
            # inputs to GF2020
            "JM": {},
            "IM": {},
            "LM": {},
            "LONS": {},
            "LATS": {},
            "GF_DT": {},
            "PLE": {},
            "PL": {},
            "ZLE0": {},
            "ZL0": {},
            "PK": {},
            "MASS": {},
            "KH"       : {},
            "T": {},
            "TH": {},
            "Q": {},
            "U": {},
            "V": {},
            "TMP3D": {},
            "BYNCY": {},
            "QLCN": {},
            "QICN": {},
            "QLLS": {},
            "QILS": {},
            "CNPCPRATE": {},
            "CNV_MF0": {},
            "CNV_PRC3": {},
            "MFD_DC": {},
            "CNV_DQCDT": {},
            "ENTLAM": {},
            "UMF_DC": {},
            "CNV_UPDF": {},
            "CNV_CVW": {},
            "CNV_QC": {},
            "CLCN": {},
            "CLLS": {},
            "QV_DYN_IN": {},
            "PLE_DYN_IN": {},
            "U_DYN_IN": {},
            "V_DYN_IN": {},
            "T_DYN_IN": {},
            "RADSW": {},
            "RADLW": {},
            "DQDT_BL": {},
            "DTDT_BL": {},
            "FRLAND": {},
            "TMP2D": {},
            "T2M": {},
            "Q2M": {},
            "TA": {},
            "QA": {},
            "SH": {},
            "EVAP": {},
            "PHIS": {},
            "KPBL": {},
            "CNV_FRC": {},
            "SRF_TYPE": {},
            "SEEDCNV": {},
            "SIGMA_DEEP": {},
            "SIGMA_MID": {},
            "DQVDT_DC": {},
            "DTDT_DC": {},
            "DUDT_DC": {},
            "DVDT_DC": {},
            "CNV_TOPP_DP": {},
            "CNV_TOPP_MD": {},
            "CNV_TOPP_SH": {},
            "MUPDP": {},
            "MUPSH": {},
            "MUPMD": {},
            "MDNDP": {},
            "MFDP": {},
            "MFSH": {},
            "MFMD": {},
            "ERRDP": {},
            "ERRSH": {},
            "ERRMD": {},
            "WQT_DC": {},
            "AA0": {},
            "AA1": {},
            "AA2": {},
            "AA3": {},
            "AA1_BL": {},
            "AA1_CIN": {},
            "TAU_BL": {},
            "TAU_EC": {},
            "DTDTDYN": {},
            "DQVDTDYN": {},
            "REVSU": {},
            "ENTR": {},
            "ENTR_DP": {},
            "ENTR_MD": {},
            "ENTR_SH": {},
            "PRFIL": {},
            "TPWI": {},
            "TPWI_star": {},
            "LFR_GF": {},
            "CNV_TR": {},
            # namelist paramters
        }

        # FloatField Outputs
        self.out_vars = {

        }

    def make_ij_field(self, data) -> Quantity:
        qty = self.quantity_factory.empty(
            [X_DIM, Y_DIM],
            "n/a",
        )
        qty.view[:, :] = qty.np.asarray(data[:, :])
        return qty

    def make_ijk_field(self, data) -> Quantity:
        qty = self.quantity_factory.empty(
            [X_DIM, Y_DIM, Z_DIM],
            "n/a",
        )
        qty.view[:, :, :] = qty.np.asarray(data[:, :, :])
        return qty

    def compute(self, inputs):
        # FloatField Variables
        t1_icloud = self.make_ijk_field(inputs["t1_icloud"])

        # Float Variables
        # Namelist options
        mp_time = Float(inputs["mp_time_icloud"][0])

        return {
            "t1_icloud": t1_icloud.view[:],
            "qv_icloud": qv_icloud.view[:],
            "ql_icloud": ql_icloud.view[:],
            "qr_icloud": qr_icloud.view[:],
            "qi_icloud": qi_icloud.view[:],
            "qs_icloud": qs_icloud.view[:],
            "qg_icloud": qg_icloud.view[:],
            "qa_icloud": qa_icloud.view[:],
            "subl1_icloud": self.subl1.view[:],
        }
