from dataclasses import dataclass

from ndsl.dsl.typing import Float, Int, Bool


@dataclass
class GFDL1MConfig:
    ANV_ICEFALL: Float
    CCI_EVAP_EFF: Float
    CCW_EVAP_EFF: Float
    CNV_FRACTION_MAX: Float
    CNV_FRACTION_MIN: Float
    DO_GFDL_REFLECTIVITY: Bool
    DT: Int
    FAC_RI: Float
    FAC_RL: Float
    GFDL_MP3: Bool
    GFDL_MP_KLID: Float
    ICE_CNV_VFALL_PARAM: Int
    ICE_FRACTION_POLYNOMIAL: Int
    ICE_LSC_VFALL_PARAM: Int
    ICE_RADII_PARAM: Int
    LIQ_RADII_PARAM: Int
    LS_ICEFALL: Float
    MAX_RH_CRIT: Float
    MAX_RI: Float
    MAX_RL: Float
    MELTFRZ_CLDMACRO: Bool
    MELTFRZ_CLDMICRO: Bool
    MIN_RH_STABLE: Float
    MIN_RH_UNSTABLE: Float
    MIN_RI: Float
    MIN_RL: Float
    PDFSHAPE: Int
    PHYS_HYDROSTATIC: Bool
    REPORT_GFDL_1M_NEGATIVES: Bool
    SH_MD_DP: Bool
    TURNRHCRIT: Float
    USE_AEROSOL_NN: Bool
    USE_BERGERON: Bool
