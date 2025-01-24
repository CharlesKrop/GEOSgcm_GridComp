"""Temporary file with GF2020 global & namelist parameters"""

from gt4py.cartesian.gtscript import i32, f32


class GF2020_flags:
    def __init__(
        self,
        HYDROSTATIC,
        ZERO_DIFF,
        DEEP,
        SHALLOW,
        CONGESTUS,
        CLOSURE_DEEP,
        CLOSURE_SHALLOW,
        CLOSURE_CONGESTUS,
        ENTRVERSION,
        MIN_ENTR_RATE,
        ENTR_DP,
        ENTR_MD,
        ENTR_SH,
        SGS_W_TIMESCALE,
        FADJ_MASSFLX_DP,
        FADJ_MASSFLX_SH,
        FADJ_MASSFLX_MD,
        USE_TRACER_TRANSP,
        USE_TRACER_SCAVEN,
        USE_SCALE_DEP,
        USE_MOMENTUM_TRANSP,
        DICYCLE,
        OUTPUT_SOUND,
        USE_MEMORY,
        TAU_OCEA_CP,
        TAU_LAND_CP,
        DOWNDRAFT,
        USE_FLUX_FORM,
        USE_TRACER_EVAP,
        APPLY_SUB_MP,
        ALP1,
        LIGHTNING_DIAG,
        OVERSHOOT,
        USE_WETBULB,
        LAMBAU_SHDN,
        MAX_TQ_TEND,
        USE_CLOUD_DISSIPATION,
        USE_SMOOTH_TEND,
        BETA_SH,
        USE_LINEAR_SUBCL_MF,
        CAP_MAXS,
        GF_ENV_SETTING,
        STOCH_TOP,
        STOCH_BOT,
        STOCHASTIC_CNV,
        GF_MIN_AREA,
        TAU_MID,
        TAU_DEEP,
        CLEV_GRID,
        VERT_DISCR,
        USE_FCT,
        SATUR_CALC,
        BC_METH,
        USE_REBCB,
        AUTOCONV,
        LAMBAU_DEEP,
        MOIST_TRIGGER,
        FRAC_MODIS,
        EVAP_FIX,
        ADV_TRIGGER,
        AVE_LAYER_DP,
        AVE_LAYER_SH,
        AVE_LAYER_MD,
        USE_EXCESS_DP,
        USE_EXCESS_SH,
        USE_EXCESS_MD,
        C0_DEEP,
        C0_MID,
        C0_SHAL,
        QRC_CRIT_OCN,
        QRC_CRIT_LND,
        C1,
        HEI_DOWN_LAND_DP,
        HEI_DOWN_LAND_SH,
        HEI_DOWN_LAND_MD,
        HEI_DOWN_OCEAN_DP,
        HEI_DOWN_OCEAN_SH,
        HEI_DOWN_OCEAN_MD,
        HEI_UPDF_LAND_DP,
        HEI_UPDF_LAND_SH,
        HEI_UPDF_LAND_MD,
        HEI_UPDF_OCEAN_DP,
        HEI_UPDF_OCEAN_SH,
        HEI_UPDF_OCEAN_MD,
        MIN_EDT_LAND_DP,
        MIN_EDT_LAND_SH,
        MIN_EDT_LAND_MD,
        MIN_EDT_OCEAN_DP,
        MIN_EDT_OCEAN_SH,
        MIN_EDT_OCEAN_MD,
        MAX_EDT_LAND_DP,
        MAX_EDT_LAND_SH,
        MAX_EDT_LAND_MD,
        MAX_EDT_OCEAN_DP,
        MAX_EDT_OCEAN_SH,
        MAX_EDT_OCEAN_MD,
        SCLM_DEEP,
        FIX_CNV_CLOUD,
        MAXIENS,
        N_TRACERS,
    ):
        # namelist flags
        self.HYDROSTATIC = HYDROSTATIC
        self.ZERO_DIFF = ZERO_DIFF
        self.DEEP = DEEP
        self.SHALLOW = SHALLOW
        self.CONGESTUS = CONGESTUS
        self.CLOSURE_DEEP = CLOSURE_DEEP
        self.CLOSURE_SHALLOW = CLOSURE_SHALLOW
        self.CLOSURE_CONGESTUS = CLOSURE_CONGESTUS
        self.ENTRVERSION = ENTRVERSION
        self.MIN_ENTR_RATE = MIN_ENTR_RATE
        self.ENTR_DP = ENTR_DP
        self.ENTR_MD = ENTR_MD
        self.ENTR_SH = ENTR_SH
        self.SGS_W_TIMESCALE = SGS_W_TIMESCALE
        self.FADJ_MASSFLX_DP = FADJ_MASSFLX_DP
        self.FADJ_MASSFLX_SH = FADJ_MASSFLX_SH
        self.FADJ_MASSFLX_MD = FADJ_MASSFLX_MD
        self.USE_TRACER_TRANSP = USE_TRACER_TRANSP
        self.USE_TRACER_SCAVEN = USE_TRACER_SCAVEN
        self.USE_SCALE_DEP = USE_SCALE_DEP
        self.USE_MOMENTUM_TRANSP = USE_MOMENTUM_TRANSP
        self.DICYCLE = DICYCLE
        self.OUTPUT_SOUND = OUTPUT_SOUND
        self.USE_MEMORY = USE_MEMORY
        self.TAU_OCEA_CP = TAU_OCEA_CP
        self.TAU_LAND_CP = TAU_LAND_CP
        self.DOWNDRAFT = DOWNDRAFT
        self.USE_FLUX_FORM = USE_FLUX_FORM
        self.USE_TRACER_EVAP = USE_TRACER_EVAP
        self.APPLY_SUB_MP = APPLY_SUB_MP
        self.ALP1 = ALP1
        self.LIGHTNING_DIAG = LIGHTNING_DIAG
        self.OVERSHOOT = OVERSHOOT
        self.USE_WETBULB = USE_WETBULB
        self.LAMBAU_SHDN = LAMBAU_SHDN
        self.MAX_TQ_TEND = MAX_TQ_TEND
        self.USE_CLOUD_DISSIPATION = USE_CLOUD_DISSIPATION
        self.USE_SMOOTH_TEND = USE_SMOOTH_TEND
        self.BETA_SH = BETA_SH
        self.USE_LINEAR_SUBCL_MF = USE_LINEAR_SUBCL_MF
        self.CAP_MAXS = CAP_MAXS
        self.GF_ENV_SETTING = GF_ENV_SETTING
        self.STOCH_TOP = STOCH_TOP
        self.STOCH_BOT = STOCH_BOT
        self.STOCHASTIC_CNV = STOCHASTIC_CNV
        self.GF_MIN_AREA = GF_MIN_AREA
        self.TAU_MID = TAU_MID
        self.TAU_DEEP = TAU_DEEP
        self.CLEV_GRID = CLEV_GRID
        self.VERT_DISCR = VERT_DISCR
        self.USE_FCT = USE_FCT
        self.SATUR_CALC = SATUR_CALC
        self.BC_METH = BC_METH
        self.USE_REBCB = USE_REBCB
        self.AUTOCONV = AUTOCONV
        self.LAMBAU_DEEP = LAMBAU_DEEP
        self.MOIST_TRIGGER = MOIST_TRIGGER
        self.FRAC_MODIS = FRAC_MODIS
        self.EVAP_FIX = EVAP_FIX
        self.ADV_TRIGGER = ADV_TRIGGER
        self.AVE_LAYER_DP = AVE_LAYER_DP
        self.AVE_LAYER_SH = AVE_LAYER_SH
        self.AVE_LAYER_MD = AVE_LAYER_MD
        self.USE_EXCESS_DP = USE_EXCESS_DP
        self.USE_EXCESS_SH = USE_EXCESS_SH
        self.USE_EXCESS_MD = USE_EXCESS_MD
        self.C0_DEEP = C0_DEEP
        self.C0_MID = C0_MID
        self.C0_SHAL = C0_SHAL
        self.QRC_CRIT_OCN = QRC_CRIT_OCN
        self.QRC_CRIT_LND = QRC_CRIT_LND
        self.C1 = C1
        self.HEI_DOWN_LAND_DP = HEI_DOWN_LAND_DP
        self.HEI_DOWN_LAND_SH = HEI_DOWN_LAND_SH
        self.HEI_DOWN_LAND_MD = HEI_DOWN_LAND_MD
        self.HEI_DOWN_OCEAN_DP = HEI_DOWN_OCEAN_DP
        self.HEI_DOWN_OCEAN_SH = HEI_DOWN_OCEAN_SH
        self.HEI_DOWN_OCEAN_MD = HEI_DOWN_OCEAN_MD
        self.HEI_UPDF_LAND_DP = HEI_UPDF_LAND_DP
        self.HEI_UPDF_LAND_SH = HEI_UPDF_LAND_SH
        self.HEI_UPDF_LAND_MD = HEI_UPDF_LAND_MD
        self.HEI_UPDF_OCEAN_DP = HEI_UPDF_OCEAN_DP
        self.HEI_UPDF_OCEAN_SH = HEI_UPDF_OCEAN_SH
        self.HEI_UPDF_OCEAN_MD = HEI_UPDF_OCEAN_MD
        self.MIN_EDT_LAND_DP = MIN_EDT_LAND_DP
        self.MIN_EDT_LAND_SH = MIN_EDT_LAND_SH
        self.MIN_EDT_LAND_MD = MIN_EDT_LAND_MD
        self.MIN_EDT_OCEAN_DP = MIN_EDT_OCEAN_DP
        self.MIN_EDT_OCEAN_SH = MIN_EDT_OCEAN_SH
        self.MIN_EDT_OCEAN_MD = MIN_EDT_OCEAN_MD
        self.MAX_EDT_LAND_DP = MAX_EDT_LAND_DP
        self.MAX_EDT_LAND_SH = MAX_EDT_LAND_SH
        self.MAX_EDT_LAND_MD = MAX_EDT_LAND_MD
        self.MAX_EDT_OCEAN_DP = MAX_EDT_OCEAN_DP
        self.MAX_EDT_OCEAN_SH = MAX_EDT_OCEAN_SH
        self.MAX_EDT_OCEAN_MD = MAX_EDT_OCEAN_MD
        self.SCLM_DEEP = SCLM_DEEP
        self.FIX_CNV_CLOUD = FIX_CNV_CLOUD

        # other miscelaneous flags
        self.MAXIENS = MAXIENS
        self.N_TRACERS = N_TRACERS
