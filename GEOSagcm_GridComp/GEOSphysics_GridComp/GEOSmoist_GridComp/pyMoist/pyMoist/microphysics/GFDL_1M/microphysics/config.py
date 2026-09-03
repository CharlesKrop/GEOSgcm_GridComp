import dataclasses
from ndsl.dsl.typing import Float, Int, Bool, Float64
from ndsl import ndsl_log, Quantity
import f90nml
import os
import pyMoist.microphysics.GFDL_1M.microphysics.constants as constants
from ndsl.dsl.gt4py import GlobalTable


@dataclasses.dataclass
class GFDLMPV3NamelistConfig:
    """Namelist for the GFDL MP V3 microphysics scheme."""

    # -----------------------------------------------------------------------
    # namelist parameters
    # values listed are fallback in case they are not specified in the namelist
    # -----------------------------------------------------------------------
    NTIMES: Int = Int(1)
    """cloud microphysics sub cycles"""

    NCONDS: Int = Int(1)
    """condensation sub cycles"""

    CFFLAG: Int = Int(1)
    """cloud fraction scheme
    1: GFDL cloud scheme
    2: Xu and Randall (1996)
    3: Park et al. (2016)
    4: Gultepe and Isaac (2007)
    """

    ICLOUD_F: Int = Int(0)
    """GFDL cloud scheme
    0: subgrid variability based scheme
    1: same as 0, but for old fvgfs implementation
    2: binary cloud scheme
    3: extension of 0
    """

    IRAIN_F: Int = Int(0)
    """cloud water to rain autoconversion scheme
    0: subgrid variability based scheme
    1: no subgrid varaibility
    """

    INFLAG: Int = Int(1)
    """ ice nucleation scheme
    1: Hong et al. (2004)
    2: Meyers et al. (1992)
    3: Meyers et al. (1992)
    4: Cooper (1986)
    5: Fletcher (1962)
    """

    IGFLAG: Int = Int(3)
    """ice generation scheme
    1: WSM6
    2: WSM6 with 0 at 0 C
    3: WSM6 with 0 at 0 C and fixed value at - 10 C
    4: combination of 1 and 3
    """

    IFFLAG: Int = Int(1)
    """ice fall scheme
    1: Deng and Mace (2008)
    2: Heymsfield and Donner (1990)
    3: Combination of Deng and Mace (2008) and Mishra et al (2014, JGR)
    """

    REWFLAG: Int = Int(1)
    """cloud water effective radius scheme
    1: Martin et al. (1994)
    2: Martin et al. (1994), GFDL revision
    4: effective radius
    """

    REIFLAG: Int = Int(5)
    """cloud ice effective radius scheme
    1: Heymsfield and Mcfarquhar (1996)
    2: Donner et al. (1997)
    3: Fu (2007)
    4: Kristjansson et al. (2000)
    5: Wyser (1998)
    6: Sun and Rikus (1999), Sun (2001)
    7: effective radius
    """

    RERFLAG: Int = Int(1)
    """rain effective radius scheme
    1: effective radius
    """

    RESFLAG: Int = Int(1)
    """snow effective radius scheme
    1: effective radius
    """

    REGFLAG: Int = Int(1)
    """graupel effective radius scheme
    1: effective radius
    """

    RADR_FLAG: Int = Int(1)
    """radar reflectivity for rain
    1: Mark Stoelinga (2005)
    2: Smith et al. (1975), Tong and Xue (2005)
    3: Marshall-Palmer formula (https://en.wikipedia.org/wiki/DBZ_(meteorology))
    """

    RADS_FLAG: Int = Int(1)
    """radar reflectivity for snow
    1: Mark Stoelinga (2005)
    2: Smith et al. (1975), Tong and Xue (2005)
    3: Marshall-Palmer formula (https://en.wikipedia.org/wiki/DBZ_(meteorology))
    """

    RADG_FLAG: Int = Int(1)
    """radar reflectivity for graupel
    1: Mark Stoelinga (2005)
    2: Smith et al. (1975), Tong and Xue (2005)
    3: Marshall-Palmer formula (https://en.wikipedia.org/wiki/DBZ_(meteorology))
    """

    SEDFLAG: Int = Int(1)
    """sedimentation scheme
    1: implicit scheme
    2: explicit scheme
    3: lagrangian scheme
    4: combined implicit and lagrangian scheme
    """

    VDIFFFLAG: Int = Int(2)
    """wind difference scheme in accretion
    1: Wisner et al. (1972)
    2: Mizuno (1990)
    3: Murakami (1990)
    """

    DO_SCALE_DEP: Bool = Bool(True)
    """impose scale dependence using sigma function"""

    DO_SEDI_UV: Bool = Bool(True)
    """transport of horizontal momentum in sedimentation"""
    DO_SEDI_W: Bool = Bool(False)
    """transport of vertical momentum in sedimentation"""

    # WMP: 01-Dec-2025
    # sedi_heat makes Tropical Cyclones too intense (even unphysical, may be a conversion bug for DTDT)
    DO_SEDI_HEAT: Bool = Bool(False)
    """transport of heat in sedimentation"""

    DO_SEDI_MELT_QI: Bool = Bool(False)
    """melt cloud ice, snow, and graupel during sedimentation"""
    DO_SEDI_MELT_QS: Bool = Bool(False)
    """melt cloud ice, snow, and graupel during sedimentation"""
    DO_SEDI_MELT_QG: Bool = Bool(False)
    """melt cloud ice, snow, and graupel during sedimentation"""

    DO_QA: Bool = Bool(False)
    """do inline cloud fraction"""
    RAD_SNOW: Bool = Bool(True)
    """include snow in cloud fraciton calculation"""
    RAD_GRAUPEL: Bool = Bool(True)
    """include graupel in cloud fraction calculation"""
    RAD_RAIN: Bool = Bool(True)
    """include rain in cloud fraction calculation"""
    DO_CLD_ADJ: Bool = Bool(True)
    """do cloud fraction adjustment"""

    DO_REF: Bool = Bool(False)
    """do radar calculations"""

    Z_SLOPE_LIQ: Bool = Bool(True)
    """use linear mono slope for autocconversions"""
    Z_SLOPE_ICE: Bool = Bool(True)
    """use linear mono slope for autocconversions"""

    USE_RHC_CEVAP: Bool = Bool(False)
    """cap of rh for cloud water evaporation"""
    USE_RHC_REVAP: Bool = Bool(True)
    """cap of rh for rain evaporation"""

    USE_ENHANCED_DRY_EVAP: Bool = Bool(True)
    """Alternative minimum evaporation formula"""

    CONST_VW: Bool = Bool(False)
    """if True, the constants are specified by v * _fac"""
    CONST_VI: Bool = Bool(False)
    """if True, the constants are specified by v * _fac"""
    CONST_VS: Bool = Bool(False)
    """if True, the constants are specified by v * _fac"""
    CONST_VG: Bool = Bool(False)
    """if True, the constants are specified by v * _fac"""
    CONST_VR: Bool = Bool(False)
    """if True, the constants are specified by v * _fac"""

    LIQ_ICE_COMBINE: Bool = Bool(False)
    """combine all liquid water, combine all solid water"""
    SNOW_GRAUPLE_COMBINE: Bool = Bool(True)
    """combine snow and graupel"""

    PROG_CCN: Bool = Bool(True)
    """do prognostic ccn (Yi Ming's method)"""
    PROG_CIN: Bool = Bool(False)
    """do prognostic cin"""

    FIX_NEGATIVE: Bool = Bool(True)
    """fix negative water species"""

    DO_EVAP_TIMESCALE: Bool = Bool(True)
    """whether to apply a timescale to evaporation"""
    DO_COND_TIMESCALE: Bool = Bool(True)
    """whether to apply a timescale to condensation"""

    DO_HAIL: Bool = Bool(False)
    """use hail parameters instead of graupel"""

    CONSV_CHECKER: Bool = Bool(False)
    """turn on energy and water conservation checker"""

    DO_WARM_RAIN_MP: Bool = Bool(False)
    """do warm rain cloud microphysics only"""

    DO_WBF: Bool = Bool(True)
    """do Wegener Bergeron Findeisen process"""

    DO_BIGG: Bool = Bool(False)
    """do Bigg process"""

    DO_PSD_WATER_FALL: Bool = Bool(False)
    """calculate cloud water terminal velocity based on PSD"""
    DO_PSD_ICE_FALL: Bool = Bool(False)
    """calculate cloud ice terminal velocity based on PSD"""

    DO_PSD_WATER_NUM: Bool = Bool(False)
    """calculate cloud water number concentration based on PSD"""
    DO_PSD_ICE_NUM: Bool = Bool(True)
    """calculate cloud ice number concentration based on PSD"""

    CP_HEATING: Bool = Bool(False)
    """update temperature based on constant pressure"""

    DELAY_COND_EVAP: Bool = Bool(True)
    """do condensation evaporation only at the last time step"""

    DO_SUBGRID_PROC: Bool = Bool(True)
    """do temperature sentive high vertical resolution processes"""

    FAST_FR_MLT: Bool = Bool(True)
    """do freezing and melting in fast microphysics"""
    FAST_DEP_SUB: Bool = Bool(True)
    """do deposition and sublimation in fast microphysics"""

    DO_MP_DIAG: Bool = Bool(False)
    """enable microphysical quantities diagnostic"""

    MP_TIME: Float = Float(150.0)
    """maximum microphysics time step (s)"""

    N0W_SIG: Float = Float(1.2)
    """intercept parameter (significant) of cloud water (Lin et al. 1983) (1/m^4) (Martin et al. 1994)"""
    N0I_SIG: Float = Float(1.2)
    """intercept parameter (significant) of cloud ice (Lin et al. 1983) (1/m^4) (McFarquhar et al. 2015)"""
    N0R_SIG: Float = Float(8.0)
    """intercept parameter (significant) of rain (Lin et al. 1983) (1/m^4) (Marshall and Palmer 1948)"""
    N0S_SIG: Float = Float(3.0)
    """intercept parameter (significant) of snow (Lin et al. 1983) (1/m^4) (Gunn and Marshall 1958)"""
    N0G_SIG: Float = Float(4.0)
    """intercept parameter (significant) of graupel (Rutledge and Hobbs 1984) (1/m^4) (Houze et al. 1979)"""
    N0H_SIG: Float = Float(4.0)
    """intercept parameter (significant) of hail (Lin et al. 1983) (1/m^4) (Federer and Waldvogel 1975)"""

    N0W_EXP: Float = Float(66)
    """intercept parameter (exponent) of cloud water (Lin et al. 1983) (1/m^4) (Martin et al. 1994)"""
    N0I_EXP: Float = Float(10)
    """intercept parameter (exponent) of cloud ice (Lin et al. 1983) (1/m^4) (McFarquhar et al. 2015)"""
    N0R_EXP: Float = Float(6)
    """intercept parameter (exponent) of rain (Lin et al. 1983) (1/m^4) (Marshall and Palmer 1948)"""
    N0S_EXP: Float = Float(6)
    """intercept parameter (exponent) of snow (Lin et al. 1983) (1/m^4) (Gunn and Marshall 1958)"""
    N0G_EXP: Float = Float(6)
    """intercept parameter (exponent) of graupel (Rutledge and Hobbs 1984) (1/m^4) (Houze et al. 1979)"""
    N0H_EXP: Float = Float(4)
    """intercept parameter (exponent) of hail (Lin et al. 1983) (1/m^4) (Federer and Waldvogel 1975)"""

    MUW: Float = Float(11.0)
    """shape parameter of cloud water in Gamma distribution (Martin et al. 1994)"""
    MUI: Float = Float(1.0)
    """shape parameter of cloud ice in Gamma distribution (McFarquhar et al. 2015)"""
    MUR: Float = Float(1.0)
    """shape parameter of rain in Gamma distribution (Marshall and Palmer 1948)"""
    MUS: Float = Float(1.0)
    """shape parameter of snow in Gamma distribution (Gunn and Marshall 1958)"""
    MUG: Float = Float(1.0)
    """shape parameter of graupel in Gamma distribution (Houze et al. 1979)"""
    MUH: Float = Float(1.0)
    """shape parameter of hail in Gamma distribution (Federer and Waldvogel 1975)"""

    ALINW: Float = Float(3.0e7)
    """"a" in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)"""
    ALINI: Float = Float(11.72)
    """"a" in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)"""
    ALINR: Float = Float(842.0)
    """"a" in Lin et al. (1983) for rain (Liu and Orville 1969)"""
    ALINS: Float = Float(4.8)
    """"a" in Lin et al. (1983) for snow (straka 2009)"""
    ALING: Float = Float(1.0)
    """"a" in Lin et al. (1983), similar to a, but for graupel (Pruppacher and Klett 2010)"""
    ALINH: Float = Float(1.0)
    """"a" in Lin et al. (1983), similar to a, but for hail (Pruppacher and Klett 2010)"""

    BLINW: Float = Float(2.0)
    """"b" in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)"""
    BLINI: Float = Float(0.41)
    """"b" in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)"""
    BLINR: Float = Float(0.8)
    """"b" in Lin et al. (1983) for rain (Liu and Orville 1969)"""
    BLINS: Float = Float(0.25)
    """"b" in Lin et al. (1983) for snow (straka 2009)"""
    BLING: Float = Float(0.5)
    """"b" in Lin et al. (1983), similar to b, but for graupel (Pruppacher and Klett 2010)"""
    BLINH: Float = Float(0.5)
    """"b" in Lin et al. (1983), similar to b, but for hail (Pruppacher and Klett 2010)"""

    TICE_MLT: Float = Float(273.16)
    """can set ice melting temperature to 268 based on observation (Kay et al. 2016) (K)"""

    T_MIN: Float = Float(178.0)
    """minimum temperature to freeze - dry all water vapor (K)"""
    T_SUB: Float = Float(184.0)
    """minimum temperature for sublimation of cloud ice (K)"""

    RH_INC: Float = Float(0.30)
    """rh increment for complete evaporation of cloud water and cloud ice"""
    RH_INR: Float = Float(0.30)
    """rh increment for minimum evaporation of rain"""

    # simple process timescales
    TAU_R2G: Float = Float(900.0)
    """rain freezing to graupel time scale (s)"""
    TAU_I2S: Float = Float(300.0)
    """cloud ice to snow autoconversion time scale (s)"""
    TAU_L2R: Float = Float(450.0)
    """cloud water to rain autoconversion time scale (s)"""
    # other timescales
    TAU_V2L: Float = Float(75.0)
    """water vapor to cloud water condensation time scale (s)"""
    TAU_L2V: Float = Float(150.0)
    """cloud water to water vapor evaporation time scale (s)"""
    TAU_REVP: Float = Float(600.0)
    """rain evaporation time scale (s)"""
    TAU_FREZ: Float = Float(600.0)
    """cloud liquid freezing time scale (s)"""
    TAU_IMLT: Float = Float(600.0)
    """cloud ice melting time scale (s)"""
    TAU_SMLT: Float = Float(900.0)
    """snow melting time scale (s)"""
    TAU_GMLT: Float = Float(1200.0)
    """graupel melting time scale (s)"""
    # subgridz timescales
    TAU_WBF: Float = Float(1200.0)
    """Wegener Bergeron Findeisen time scale (s)"""

    CCN_O: Float = Float(90.0)
    """ccn over ocean (1/cm^3)"""
    CCN_L: Float = Float(270.0)
    """ccn over land (1/cm^3)"""

    RTHRESHU: Float = Float(7.0e-6)
    """unstable critical cloud drop radius (micro m)"""
    RTHRESHS: Float = Float(10.0e-6)
    """stable critical cloud drop radius (micro m)"""

    IN_CLOUD_LIQ: Bool = True
    """use in-cloud liquid"""
    IN_CLOUD_ICE: Bool = True
    """use in-cloud frozen"""

    CLD_MIN: Bool = Float(0.05)
    """minimum cloud fraction"""

    QI_LIM: Bool = Float(1.0)
    """cloud ice limiter (0: no, 1: full, >1: extra) to prevent large ice build up"""

    QL_MLT: Bool = Float(2.0e-3)
    """maximum cloud water allowed from melted cloud ice (kg/kg)"""
    QS_MLT: Bool = Float(1.0e-6)
    """maximum cloud water allowed from melted snow (kg/kg)"""

    QL0_MAX: Bool = Float(2.0e-3)
    """maximum cloud water value (autoconverted to rain) (kg/kg)"""

    PSAUT_QI_CRT: Bool = Float(1.0e-4)
    """cloud ice to snow autoconversion threshold (kg/m^3)"""
    PWBF_QI_CRT: Bool = Float(0.8e-4)
    """WBF liquid to ice freezing threshold (kg/m^3)"""
    PGAUT_QS_CRT: Bool = Float(0.6e-3)
    """snow to graupel autoconversion threshold (0.6e-3 in Purdue Lin scheme) (kg/m^3)"""

    C_PAUT: Float = Float(0.5)
    """cloud water to rain autoconversion efficiency"""

    # -----------------------------------------------------------------------
    # collection efficiencies for accretion
    # -----------------------------------------------------------------------
    # --- Cloud Water (Liquid) 3D Accretion ---
    # When True, these coefficients act as Aerodynamic Stokes Efficiencies applied to the raw 3D geometric integral.
    DO_3D_ACC_CLIQ: Bool = True
    """perform the new 3d accretion for cloud water"""
    C_PSACW: Float = Float(0.05)
    """cloud water to snow (HEAVY aerodynamic reduction required)"""
    C_PGACW: Float = Float(0.80)
    """cloud water to graupel/hail (Punches through air)"""
    C_PRACW: Float = Float(1.00)
    """cloud water to rain"""
    # --- Cloud Ice (Frozen) 3D Accretion ---
    # When .true., these coefficients account for both Aerodynamics AND "Bounce" (Sticking Efficiency) applied to the raw 3D geometric integral.
    DO_3D_ACC_CICE: Bool = False
    """perform the new 3d accretion for cloud ice"""
    C_PSACI: Float = Float(0.05)
    """cloud ice to snow accretion (Aerodynamics + Low sticking)"""
    C_PGACI: Float = Float(0.01)
    """cloud ice to graupel accretion (Aerodynamics + Very low sticking)"""
    C_PRACI: Float = Float(1.00)
    """cloud ice to rain accretion (High sticking to liquid)"""
    # --- Standard Macro-Particle Accretion ---
    # Interactions between precipitation species (Unaffected by 3D cice/cliq flags)
    C_PGACS: Float = Float(0.03)
    """snow to graupel accretion efficiency"""
    C_PRACS: Float = Float(1.00)
    """snow to rain accretion efficiency"""
    C_PSACR: Float = Float(1.00)
    """rain to snow accretion efficiency"""
    C_PGACR: Float = Float(1.00)
    """rain to graupel accretion efficiency"""

    IS_FAC: Float = Float(0.2)
    """cloud ice sublimation temperature factor"""
    SS_FAC: Float = Float(0.2)
    """snow sublimation temperature factor"""
    GS_FAC: Float = Float(0.2)
    """graupel sublimation temperature factor"""

    RH_FAC_EVAP: Float = Float(10.0)
    """cloud water evaporation relative humidity factor"""
    RH_FAC_COND: Float = Float(10.0)
    """cloud water condensation relative humidity factor"""

    SED_FAC: Float = Float(1.0)
    """coefficient for sedimentation fall, scale from 1.0 (implicit) to 0.0 (lagrangian)"""

    DO_ICE_PRES_SCALING: Bool = True
    """optional pressure scaling to accelerate ice settling in the upper troposphere"""

    VW_FAC: Float = Float(1.0)
    VI_FAC: Float = Float(1.0)
    VS_FAC: Float = Float(1.0)
    VG_FAC: Float = Float(1.0)
    VR_FAC: Float = Float(1.0)
    VH_FAC: Float = Float(1.0)

    VW_MIN: Float = Float(0.0)
    """minimum fall speed for cloud water (m/s)"""
    VI_MIN: Float = Float(0.01)
    """minimum fall speed or constant fall speed"""
    VS_MIN: Float = Float(0.25)
    """minimum fall speed or constant fall speed"""
    VG_MIN: Float = Float(3.0)
    """minimum fall speed or constant fall speed"""
    VR_MIN: Float = Float(4.0)
    """minimum fall speed or constant fall speed"""
    VH_MIN: Float = Float(9.0)
    """minimum fall speed or constant fall speed"""

    VW_MAX: Float = Float(0.01)
    """max fall speed for cloud water (m/s)"""
    VI_MAX: Float = Float(1.0)
    """max fall speed for ice"""
    VS_MAX: Float = Float(1.5)
    """max fall speed for snow"""
    VG_MAX: Float = Float(9.0)
    """max fall speed for graupel"""
    VR_MAX: Float = Float(12.0)
    """max fall speed for rain"""
    VH_MAX: Float = Float(19.0)
    """max fall speed for hail"""

    XR_A: Float = Float(0.25)
    """p value in Xu and Randall (1996)"""
    XR_B: Float = Float(100.0)
    """alpha_0 value in Xu and Randall (1996)"""
    XR_C: Float = Float(0.49)
    """gamma value in Xu and Randall (1996)"""

    TE_ERR: Float = Float(1.0e-5)
    """64bit: 1.e-14, 32bit: 1.e-7; turn off to save computer time"""
    TW_ERR: Float = Float(1.0e-8)
    """64bit: 1.e-14, 32bit: 1.e-7; turn off to save computer time"""

    RH_THRES: Float = Float(0.75)
    """minimum relative humidity for cloud fraction"""
    RHC_CEVAP: Float = Float(0.85)
    """maximum relative humidity for cloud water evaporation"""
    RHC_REVAP: Float = Float(0.85)
    """maximum relative humidity for rain evaporation"""

    F_DQ_P: Float = Float(3.0)
    """cloud fraction adjustment for supersaturation"""
    F_DQ_M: Float = Float(1.0)
    """cloud fraction adjustment for undersaturation"""

    FI2S_FAC: Float = Float(1.00)
    """maximum sink of cloud ice to form snow: 0-1"""
    FI2G_FAC: Float = Float(1.00)
    """maximum sink of cloud ice to form graupel/hail: 0-1"""
    FS2G_FAC: Float = Float(0.75)
    """maximum sink of snow to form graupel: 0-1"""

    BETA: Float = Float(1.22)
    """defined in Heymsfield and Mcfarquhar (1996)"""

    REWMIN: Float = Float(5.0)
    """minimum effective radius for cloud water (micron)"""
    REWMAX: Float = Float(10.0)
    """maximum effective radius for cloud water (micron)"""
    REIMIN: Float = Float(10.0)
    """minimum effective radius for cloud ice (micron)"""
    REIMAX: Float = Float(150.0)
    """maximum effective radius for cloud ice (micron)"""
    RERMIN: Float = Float(10.0)
    """minimum effective radius for rain (micron)"""
    RERMAX: Float = Float(10000.0)
    """maximum effective radius for rain (micron)"""
    RESSMIN: Float = Float(150.0)
    """minimum effective radius for snow (micron)"""
    RESSMAX: Float = Float(10000.0)
    """maximum effective radius for snow (micron)"""
    REGMIN: Float = Float(150.0)
    """minimum effective radius for graupel (micron)"""
    REGMAX: Float = Float(10000.0)
    """maximum effective radius for graupel (micron)"""

    REWFAC: Float = Float(1.0)
    """this is a tuning parameter to compromise the inconsistency between
    GFDL MP's PSD and cloud water radiative property's PSD assumption.
    after the cloud water radiative property's PSD is rebuilt,
    this parameter should be 1.0."""
    REIFAC: Float = Float(1.0)
    """this is a tuning parameter to compromise the inconsistency between
    GFDL MP's PSD and cloud ice radiative property's PSD assumption.
    after the cloud ice radiative property's PSD is rebuilt,
    this parameter should be 1.0."""

    @classmethod
    def init_from_default(cls) -> "GFDLMPV3NamelistConfig":
        """Create an instance initialized with default values."""
        return cls()

    @classmethod
    def init_from_namelist(cls, nml_path: str) -> "GFDLMPV3NamelistConfig":
        """Instantiate config from the gfdl_mp_nml group of a Fortran namelist file.

        For fields present in gfdl_mp_nml, cast and use that value.
        Otherwise, fall back to the default constant in constants.py.
        """
        if not os.path.isfile(nml_path):
            ndsl_log.error(f"[GFDL1M Microphysics: {nml_path} does not exist")
            raise FileNotFoundError(f"{nml_path} does not exist")

        try:
            full_nml = f90nml.read(nml_path)
        except Exception as e:
            ndsl_log.error(
                f"[GFDL1M Microphysics]: namelist exists at {nml_path} but read failed, bailing out",
                exc_info=e,
            )
            raise e

        mp_nml = full_nml.get("gfdl_mp_nml", {})
        ndsl_log.info(f"[GFDL1M Microphysics]: full microphysics namelist:\n{mp_nml}")

        # Validate that all namelist keys exist in the configuration
        valid_keys = {f.name.lower() for f in dataclasses.fields(cls)}
        unknown_keys = set(mp_nml.keys()) - valid_keys

        if unknown_keys:
            msg = f"[GFDL1M Microphysics]: Unknown parameter(s) in namelist: " f"{', '.join(sorted(unknown_keys))}"
            ndsl_log.error(msg)
            raise KeyError(msg)

        # Construct the set of keyword arguments to initialize the dataclass instance
        kwargs = {}
        for field in dataclasses.fields(cls):
            name = field.name
            key = name.lower()

            if key in mp_nml:
                # Value came from the namelist file - cast to the declared field type
                value = mp_nml[key]
                kwargs[name] = field.type(value)
            else:
                # Not overridden in the namelist - fall back to constants.py
                if not hasattr(constants, "_" + name):
                    msg = f"[GFDL1M Microphysics]: '{name}' does not have a fallback value specified, " f"it must be included in the namelist"
                    ndsl_log.error(msg)
                    raise AttributeError(msg)
                kwargs[name] = getattr(constants, "_" + name)

        return cls(**kwargs)


# types for tables in GFDLMPV3CloudMPConfig
# types must be declared here for all possible tables sizes
GlobalTableL2 = GlobalTable[(Float, (2))]
GlobalTableL3 = GlobalTable[(Float, (3))]
GlobalTableL4 = GlobalTable[(Float, (4))]
GlobalTableL5 = GlobalTable[(Float, (5))]
GlobalTableL20 = GlobalTable[(Float, (20))]
GlobalTableL3x10 = GlobalTable[(Float, (3, 10))]


@dataclasses.dataclass
class GFDLMPV3CloudMPConfig:
    """Configuration for the GFDL MP V3 microphysics scheme."""

    # heat capacities and related terms
    C_AIR: Float
    C_VAP: Float
    D0_VAP: Float
    D1_ICE: Float
    D1_VAP: Float
    LV00: Float64
    LI00: Float64
    LI20: Float64
    C1_VAP: Float64
    C1_LIQ: Float64
    C1_ICE: Float64

    # all other configuration parameters
    T_WFR: Float
    CPAUT0: Float64
    NORMW: Float64
    NORMI: Float64
    NORMR: Float64
    NORMS: Float64
    NORMG: Float64
    NORMH: Float64
    EXPOW: Float64
    EXPOI: Float64
    EXPOR: Float64
    EXPOS: Float64
    EXPOG: Float64
    EXPOH: Float64
    PCAW: Float64
    PCAI: Float64
    PCAR: Float64
    PCAS: Float64
    PCAG: Float64
    PCAH: Float64
    PCBW: Float64
    PCBI: Float64
    PCBR: Float64
    PCBS: Float64
    PCBG: Float64
    PCBH: Float64
    EDAW: Float64
    EDAI: Float64
    EDAR: Float64
    EDAS: Float64
    EDAG: Float64
    EDAH: Float64
    EDBW: Float64
    EDBI: Float64
    EDBR: Float64
    EDBS: Float64
    EDBG: Float64
    EDBH: Float64
    OEAW: Float64
    OEAI: Float64
    OEAR: Float64
    OEAS: Float64
    OEAG: Float64
    OEAH: Float64
    OEBW: Float64
    OEBI: Float64
    OEBR: Float64
    OEBS: Float64
    OEBG: Float64
    OEBH: Float64
    RRAW: Float64
    RRAI: Float64
    RRAR: Float64
    RRAS: Float64
    RRAG: Float64
    RRAH: Float64
    RRBW: Float64
    RRBI: Float64
    RRBR: Float64
    RRBS: Float64
    RRBG: Float64
    RRBH: Float64
    TVAW: Float64
    TVAI: Float64
    TVAR: Float64
    TVAS: Float64
    TVAG: Float64
    TVAH: Float64
    TVBW: Float64
    TVBI: Float64
    TVBR: Float64
    TVBS: Float64
    TVBG: Float64
    TVBH: Float64
    CGACI: Float
    CGACR: Float
    CGACS: Float
    CGACW: Float
    CRACI: Float
    CRACS: Float
    CRACW: Float
    CSACI: Float
    CSACR: Float
    CSACW: Float

    # tables - these are initialized as quantities and filled
    # manually so that they can be passed into stencils as GlobalTables
    ACC: Quantity
    ACCO: Quantity
    CGFR: Quantity
    CGMLT: Quantity
    CGSUB: Quantity
    CREVP: Quantity
    CSMLT: Quantity
    CSSUB: Quantity

    @classmethod
    def init_to_none(cls) -> "GFDLMPV3NamelistConfig":
        """Create an all-None instance, meant to be fully populated afterward."""
        return cls(**{f.name: None for f in dataclasses.fields(cls)})
