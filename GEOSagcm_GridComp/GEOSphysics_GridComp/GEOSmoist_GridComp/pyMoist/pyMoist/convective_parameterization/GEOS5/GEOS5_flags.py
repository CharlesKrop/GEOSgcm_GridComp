"""Temporary file with GEOS5 namelist parameters"""

from gt4py.cartesian.gtscript import i32, f32


class GEOS5_flags:
    def __init__(
        self,
        ICUMULUS_GF,
        CLOSURE_CHOICE,
        CUM_ENTR_RATE,
        USE_TRACER_TRANSP,
        USE_TRACER_SCAVEN,
        USE_FLUX_FORM,
        USE_FCT,
        USE_TRACER_EVAP,
        USE_SCALE_DEP,
        DICYCLE,
        ALP1,
        BC_METH,
        CUM_AVE_LAYER,
        AVE_LAYER,
        TAU_DEEP,
        TAU_MID,
        C0_DEEP,
        C0_MID,
        C0_SHAL,
        QRC_CRIT,
        QRC_CRIT_LND,
        QRC_CRIT_OCN,
        C1,
    ):
        self.ICUMULUS_GF = ICUMULUS_GF
        self.CLOSURE_CHOICE = CLOSURE_CHOICE
        self.CUM_ENTR_RATE = CUM_ENTR_RATE
        self.USE_TRACER_TRANSP = USE_TRACER_TRANSP
        self.USE_TRACER_SCAVEN = USE_TRACER_SCAVEN
        self.USE_FLUX_FORM = USE_FLUX_FORM
        self.USE_FCT = USE_FCT
        self.USE_TRACER_EVAP = USE_TRACER_EVAP
        self.USE_SCALE_DEP = USE_SCALE_DEP
        self.DICYCLE = DICYCLE
        self.ALP1 = ALP1
        self.BC_METH = BC_METH
        self.CUM_AVE_LAYER = CUM_AVE_LAYER
        self.AVE_LAYER = AVE_LAYER
        self.TAU_DEEP = TAU_DEEP
        self.TAU_MID = TAU_MID
        self.C0_DEEP = C0_DEEP
        self.C0_MID = C0_MID
        self.C0_SHAL = C0_SHAL
        self.QRC_CRIT = QRC_CRIT
        self.QRC_CRIT_LND = QRC_CRIT_LND
        self.QRC_CRIT_OCN = QRC_CRIT_OCN
        self.C1 = C1
