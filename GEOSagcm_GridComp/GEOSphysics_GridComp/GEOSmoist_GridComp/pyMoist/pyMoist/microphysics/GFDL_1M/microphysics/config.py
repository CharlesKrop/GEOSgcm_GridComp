import dataclasses
from ndsl.dsl.typing import Float, Int, Bool


@dataclasses.dataclass
class GFDLMPV3Config:
    """Configuration for the GFDL MP V3 microphysics scheme."""

    NTIMES: Int  # cloud microphysics sub cycles
    NCONDS: Int  # condensation sub cycles
    CFFLAG: Int  # cloud fraction scheme
    ICLOUD_F: Int  # GFDL cloud scheme
    IRAIN_F: Int  # cloud water to rain autoconversion scheme
    INFLAG: Int  # ice nucleation scheme
    IGFLAG: Int  # ice generation scheme
    IFFLAG: Int  # ice fall scheme
    REWFLAG: Int  # cloud water effective radius scheme
    REIFLAG: Int  # cloud ice effective radius scheme
    RERFLAG: Int  # rain effective radius scheme
    RESFLAG: Int  # snow effective radius scheme
    REGFLAG: Int  # graupel effective radius scheme
    RADR_FLAG: Int  # radar reflectivity for rain
    RADS_FLAG: Int  # radar reflectivity for snow
    RADG_FLAG: Int  # radar reflectivity for graupel
    SEDFLAG: Int  # sedimentation scheme
    VDIFFFLAG: Int  # wind difference scheme in accretion
    DO_SCALE_DEP: Bool  # impose scale dependence using sigma function
    DO_SEDI_UV: Bool  # transport of horizontal momentum in sedimentation
    DO_SEDI_W: Bool  # transport of vertical momentum in sedimentation
    DO_SEDI_HEAT: Bool  # transport of heat in sedimentation
    DO_SEDI_MELT_QI: Bool  # melt cloud ice, snow, and graupel during sedimentation
    DO_SEDI_MELT_QS: Bool  # melt cloud ice, snow, and graupel during sedimentation
    DO_SEDI_MELT_QG: Bool  # melt cloud ice, snow, and graupel during sedimentation
    DO_QA: Bool  # do inline cloud fraction
    RAD_SNOW: Bool  # include snow in cloud fraciton calculation
    RAD_GRAUPEL: Bool  # include graupel in cloud fraction calculation
    RAD_RAIN: Bool  # include rain in cloud fraction calculation
    DO_CLD_ADJ: Bool  # do cloud fraction adjustment
    DO_REF: Bool  # do radar calculations
    Z_SLOPE_LIQ: Bool  # use linear mono slope for autocconversions
    Z_SLOPE_ICE: Bool  # use linear mono slope for autocconversions
    USE_RHC_CEVAP: Bool  # cap of rh for cloud water evaporation
    USE_RHC_REVAP: Bool  # cap of rh for rain evaporation
    USE_ENHANCED_DRY_EVAP: Bool  # Alternative minimum evaporation formula
    CONST_VW: Bool  # if True, the constants are specified by v * _fac
    CONST_VI: Bool  # if True, the constants are specified by v * _fac
    CONST_VS: Bool  # if True, the constants are specified by v * _fac
    CONST_VG: Bool  # if True, the constants are specified by v * _fac
    CONST_VR: Bool  # if True, the constants are specified by v * _fac
    LIQ_ICE_COMBINE: Bool  # combine all liquid water, combine all solid water
    SNOW_GRAUPLE_COMBINE: Bool  # combine snow and graupel
    PROG_CCN: Bool  # do prognostic ccn (Yi Ming's method)
    PROG_CIN: Bool  # do prognostic cin
    FIX_NEGATIVE: Bool  # fix negative water species
    DO_EVAP_TIMESCALE: Bool  # whether to apply a timescale to evaporation
    DO_COND_TIMESCALE: Bool  # whether to apply a timescale to condensation
    DO_HAIL: Bool  # use hail parameters instead of graupel
    CONSV_CHECKER: Bool  # turn on energy and water conservation checker
    DO_WARM_RAIN_MP: Bool  # do warm rain cloud microphysics only
    DO_WBF: Bool  # do Wegener Bergeron Findeisen process
    DO_BIGG: Bool  # do Bigg process
    DO_PSD_WATER_FALL: Bool  # calculate cloud water terminal velocity based on PSD
    DO_PSD_ICE_FALL: Bool  # calculate cloud ice terminal velocity based on PSD
    DO_PSD_WATER_NUM: Bool  # calculate cloud water number concentration based on PSD
    DO_PSD_ICE_NUM: Bool  # calculate cloud ice number concentration based on PSD
    CP_HEATING: Bool  # update temperature based on constant pressure
    DELAY_COND_EVAP: Bool  # do condensation evaporation only at the last time step
    DO_SUBGRID_PROC: Bool  # do temperature sentive high vertical resolution processes
    FAST_FR_MLT: Bool  # do freezing and melting in fast microphysics
    FAST_DEP_SUB: Bool  # do deposition and sublimation in fast microphysics
    DO_MP_DIAG: Bool  # enable microphysical quantities diagnostic
    MP_TIME: Float  # maximum microphysics time step (s)
    N0W_SIG: Float  # intercept parameter (significant) of cloud water (Lin et al. 1983) (1/m^4) (Martin et al. 1994)
    N0I_SIG: Float  # intercept parameter (significant) of cloud ice (Lin et al. 1983) (1/m^4) (McFarquhar et al. 2015)
    N0R_SIG: Float  # intercept parameter (significant) of rain (Lin et al. 1983) (1/m^4) (Marshall and Palmer 1948)
    N0S_SIG: Float  # intercept parameter (significant) of snow (Lin et al. 1983) (1/m^4) (Gunn and Marshall 1958)
    N0G_SIG: Float  # intercept parameter (significant) of graupel (Rutledge and Hobbs 1984) (1/m^4) (Houze et al. 1979)
    N0H_SIG: Float  # intercept parameter (significant) of hail (Lin et al. 1983) (1/m^4) (Federer and Waldvogel 1975)
    N0W_EXP: Float  # intercept parameter (exponent) of cloud water (Lin et al. 1983) (1/m^4) (Martin et al. 1994)
    N0I_EXP: Float  # intercept parameter (exponent) of cloud ice (Lin et al. 1983) (1/m^4) (McFarquhar et al. 2015)
    N0R_EXP: Float  # intercept parameter (exponent) of rain (Lin et al. 1983) (1/m^4) (Marshall and Palmer 1948)
    N0S_EXP: Float  # intercept parameter (exponent) of snow (Lin et al. 1983) (1/m^4) (Gunn and Marshall 1958)
    N0G_EXP: Float  # intercept parameter (exponent) of graupel (Rutledge and Hobbs 1984) (1/m^4) (Houze et al. 1979)
    N0H_EXP: Float  # intercept parameter (exponent) of hail (Lin et al. 1983) (1/m^4) (Federer and Waldvogel 1975)
    MUW: Float  # shape parameter of cloud water in Gamma distribution (Martin et al. 1994)
    MUI: Float  # shape parameter of cloud ice in Gamma distribution (McFarquhar et al. 2015)
    MUR: Float  # shape parameter of rain in Gamma distribution (Marshall and Palmer 1948)
    MUS: Float  # shape parameter of snow in Gamma distribution (Gunn and Marshall 1958)
    MUG: Float  # shape parameter of graupel in Gamma distribution (Houze et al. 1979)
    MUH: Float  # shape parameter of hail in Gamma distribution (Federer and Waldvogel 1975)
    ALINW: Float  # "a" in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)
    ALINI: Float  # "a" in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)
    ALINR: Float  # "a" in Lin et al. (1983) for rain (Liu and Orville 1969)
    ALINS: Float  # "a" in Lin et al. (1983) for snow (straka 2009)
    ALING: Float  # "a" in Lin et al. (1983), similar to a, but for graupel (Pruppacher and Klett 2010)
    ALINH: Float  # "a" in Lin et al. (1983), similar to a, but for hail (Pruppacher and Klett 2010)
    BLINW: Float  # "b" in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)
    BLINI: Float  # "b" in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)
    BLINR: Float  # "b" in Lin et al. (1983) for rain (Liu and Orville 1969)
    BLINS: Float  # "b" in Lin et al. (1983) for snow (straka 2009)
    BLING: Float  # "b" in Lin et al. (1983), similar to b, but for graupel (Pruppacher and Klett 2010)
    BLINH: Float  # "b" in Lin et al. (1983), similar to b, but for hail (Pruppacher and Klett 2010)
    TICE_MLT: Float  # can set ice melting temperature to 268 based on observation (Kay et al. 2016) (K)
    T_MIN: Float  # minimum temperature to freeze - dry all water vapor (K)
    T_SUB: Float  # minimum temperature for sublimation of cloud ice (K)
    RH_INC: Float  # rh increment for complete evaporation of cloud water and cloud ice
    RH_INR: Float  # rh increment for minimum evaporation of rain
    TAU_R2G: Float  # rain freezing to graupel time scale (s)
    TAU_I2S: Float  # cloud ice to snow autoconversion time scale (s)
    TAU_L2R: Float  # cloud water to rain autoconversion time scale (s)
    TAU_V2L: Float  # water vapor to cloud water condensation time scale (s)
    TAU_L2V: Float  # cloud water to water vapor evaporation time scale (s)
    TAU_REVP: Float  # rain evaporation time scale (s)
    TAU_FREZ: Float  # cloud liquid freezing time scale (s)
    TAU_IMLT: Float  # cloud ice melting time scale (s)
    TAU_SMLT: Float  # snow melting time scale (s)
    TAU_GMLT: Float  # graupel melting time scale (s)
    TAU_WBF: Float  # Wegener Bergeron Findeisen time scale (s)
    CCN_O: Float  # ccn over ocean (1/cm^3)
    CCN_L: Float  # ccn over land (1/cm^3)
    RTHRESHU: Float  # unstable critical cloud drop radius (micro m)
    RTHRESHS: Float  # stable critical cloud drop radius (micro m)
    IN_CLOUD_LIQ: Bool  # use in-cloud liquid
    IN_CLOUD_ICE: Bool  # use in-cloud frozen
    CLD_MIN: Float  # minimum cloud fraction
    QI_LIM: Float  # cloud ice limiter (0: no, 1: full, >1: extra) to prevent large ice build up
    QL_MLT: Float  # maximum cloud water allowed from melted cloud ice (kg/kg)
    QS_MLT: Float  # maximum cloud water allowed from melted snow (kg/kg)
    QL0_MAX: Float  # maximum cloud water value (autoconverted to rain) (kg/kg)
    PSAUT_QI_CRT: Float  # cloud ice to snow autoconversion threshold (kg/m^3)
    PWBF_QI_CRT: Float  # WBF liquid to ice freezing threshold (kg/m^3)
    PGAUT_QS_CRT: Float  # snow to graupel autoconversion threshold (0.6e-3 in Purdue Lin scheme) (kg/m^3)
    C_PAUT: Float  # cloud water to rain autoconversion efficiency

    @classmethod
    def init_to_none(cls) -> "GFDLMPV3Config":
        """Create an all-None instance, meant to be fully populated afterward."""
        return cls(**{f.name: None for f in dataclasses.fields(cls)})
