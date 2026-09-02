"""This file contains all constants for GFDL Microphysics Core V3. These constants should not be used outside of, the microphysics core, nor should any external
constants be required beyond what is in this file. Any values starting with "_" (e.g. _NTIMES) should not be used directly. Use the value found in GFDLMPV3Config
instead, these values are merely fallbacks to fill that if they are not specified in the namelist (replicating fortran behavior and providing a single source of truth
for all constants)."""

from ndsl.dsl.typing import Float, Int, Float64
import math

# -----------------------------------------------------------------------
# physics constants
# -----------------------------------------------------------------------
GRAV = Float(9.80665)  # acceleration due to gravity (m/s^2), ref: IFS

RGRAV = Float(1.0) / GRAV  # inversion of gravity acceleration (s^2/m)

PI = Float(4.0) * math.atan(Float(1.0))  # ratio of circle circumference to diameter

BOLTZMANN = Float(1.38064852e-23)  # boltzmann constant (J/K)
AVOGADRO = Float(6.02214076e23)  # avogadro number (1/mol)
RUNIVER = AVOGADRO * BOLTZMANN  # 8.314459727525675, universal gas constant (J/K/mol)
MMD = Float(2.89644e-2)  # dry air molar mass (kg/mol), ref: IFS
MMV = Float(1.80153e-2)  # water vapor molar mass (kg/mol), ref: IFS

RDGAS = Float(287.05)  # gas constant for dry air (J/kg/K): ref: GFDL, GFS
RVGAS = Float(461.50)  # gas constant for water vapor (J/kg/K): ref: GFDL, GFS

ZVIR = RVGAS / RDGAS - 1.0  # 0.6077667316114637
EPS = RDGAS / RVGAS  # 0.6219934994582882
EPSM1 = RDGAS / RVGAS - 1.0  # -0.3780065005417118

TICE = Float(273.15)  # freezing temperature (K): ref: GFDL, GFS

SATURATION_TABLE_LENGTH = Int(2621)
SATURATION_TABLE_TMIN = TICE - Float(160.0)
DELT = Float64(0.1)
RDELT = Float64(1.0) / DELT

CP_AIR = Float(1004.6)  # heat capacity of dry air at constant pressure (J/kg/K): ref: GFDL, GFS
CV_AIR = CP_AIR - RDGAS  # 717.55, heat capacity of dry air at constant volume (J/kg/K): ref: GFDL, GFS
CP_VAP = Float(4.0) * RVGAS  # 1846.0885419672554, heat capacity of water vapor at constnat pressure (J/kg/K)
CV_VAP = Float(3.0) * RVGAS  # 1384.5664064754415, heat capacity of water vapor at constant volume (J/kg/K)

C_ICE = Float(2.106e3)  # heat capacity of ice at 0 deg C (J/kg/K), ref: IFS
C_LIQ = Float(4.218e3)  # heat capacity of water at 0 deg C (J/kg/K), ref: IFS

DC_VAP = CP_VAP - C_LIQ  # - 2371.9114580327446, isobaric heating / cooling (J/kg/K)
DC_ICE = C_LIQ - C_ICE  # 2112.0, isobaric heating / cooling (J/kg/K)
D2_ICE = CP_VAP - C_ICE  # - 259.9114580327446, isobaric heating / cooling (J/kg/K)

HLV = Float(2.5e6)  # latent heat of evaporation at 0 deg C (J/kg): ref: GFDL, GFS
HLF = Float(3.3358e5)  # latent heat of fusion at 0 deg C (J/kg): ref: GFDL, GFS

LATS = HLV + HLF
LAT2 = LATS * LATS

VISD = Float(1.717e-5)  # dynamics viscosity of air at 0 deg C and 1000 hPa (Mason, 1971) (kg/m/s)
VISK = Float(1.35e-5)  # kinematic viscosity of air at 0 deg C  and 1000 hPa (Mason, 1971) (m^2/s)
VDIFU = Float(2.25e-5)  # diffusivity of water vapor in air at 0 deg C  and 1000 hPa (Mason, 1971) (m^2/s)
TCOND = Float(2.40e-2)  # thermal conductivity of air at 0 deg C  and 1000 hPa (Mason, 1971) (J/m/s/K)

RHO0 = Float(1.0)  # reference air density (kg/m^3), ref: IFS
CDG = Float(3.15121)  # drag coefficient of graupel (Locatelli and Hobbs, 1974)
CDH = Float(0.5)  # drag coefficient of hail (Heymsfield and Wright, 2014)

LV0 = Float64(HLV - DC_VAP * TICE)  # 3148711.3338762247, evaporation latent heat coeff. at 0 deg K (J/kg)
LI0 = Float64(HLF - DC_ICE * TICE)  # - 242413.92000000004, fussion latent heat coeff. at 0 deg K (J/kg)
LI2 = Float64(LV0 + LI0)  # 2906297.413876225, sublimation latent heat coeff. at 0 deg K (J/kg)

E00 = Float64(611.21)  # saturation vapor pressure at 0 deg C (Pa), ref: IFS

# -----------------------------------------------------------------------
# predefined parameters
# -----------------------------------------------------------------------
QPMIN = Float(1.0e-15)  # min value for suspended rain/snow/liquid/ice precip
QVMIN = Float(1.0e-15)  # min value for water vapor (treated as zero)
QCMIN = Float(1.0e-15)  # min value for cloud condensates (kg/kg)
CFMIN = Float(1.0e-5)  # min value for cloud fraction (unitless)
QFMIN = Float(1.0e-15)  # min value for sedimentation (kg/kg)

DZ_MIN = Float(1.0e-2)  # used for correcting flipped height (m)

RHOW = Float(1.0e3)  # density of cloud water (kg/m^3)
RHOI = Float(9.17e2)  # density of cloud ice (kg/m^3)
RHOR = Float(1.0e3)  # density of rain (Lin et al. 1983) (kg/m^3)
RHOS = Float(1.0e2)  # density of snow (Lin et al. 1983) (kg/m^3)
RHOG = Float(4.0e2)  # density of graupel (Rutledge and Hobbs 1984) (kg/m^3)
RHOH = Float(9.17e2)  # density of hail (Lin et al. 1983) (kg/m^3)

DT_FR = Float(8.0)  # t_wfr - dt_fr: minimum temperature water can exist (Moore and Molinero 2011)

RC = (Float(4.0) / Float(3.0)) * PI * RHOR

ONE_R8 = Float64(1.0)  # constant 1

# -----------------------------------------------------------------------
# namelist parameters
# values listed are fallback in case they are not specified in the namelist
# -----------------------------------------------------------------------
_NTIMES = Int(1)  # cloud microphysics sub cycles

_NCONDS = Int(1)  # condensation sub cycles

_CFFLAG = Int(1)  # cloud fraction scheme
# 1: GFDL cloud scheme
# 2: Xu and Randall (1996)
# 3: Park et al. (2016)
# 4: Gultepe and Isaac (2007)

_ICLOUD_F = Int(0)  # GFDL cloud scheme
# 0: subgrid variability based scheme
# 1: same as 0, but for old fvgfs implementation
# 2: binary cloud scheme
# 3: extension of 0

_IRAIN_F = Int(0)  # cloud water to rain autoconversion scheme
# 0: subgrid variability based scheme
# 1: no subgrid varaibility

_INFLAG = Int(1)  # ice nucleation scheme
# 1: Hong et al. (2004)
# 2: Meyers et al. (1992)
# 3: Meyers et al. (1992)
# 4: Cooper (1986)
# 5: Fletcher (1962)

_IGFLAG = Int(3)  # ice generation scheme
# 1: WSM6
# 2: WSM6 with 0 at 0 C
# 3: WSM6 with 0 at 0 C and fixed value at - 10 C
# 4: combination of 1 and 3

_IFFLAG = Int(1)  # ice fall scheme
# 1: Deng and Mace (2008)
# 2: Heymsfield and Donner (1990)
# 3: Combination of Deng and Mace (2008) and Mishra et al (2014, JGR)

_REWFLAG = Int(1)  # cloud water effective radius scheme
# 1: Martin et al. (1994)
# 2: Martin et al. (1994), GFDL revision
# 4: effective radius

_REIFLAG = Int(5)  # cloud ice effective radius scheme
# 1: Heymsfield and Mcfarquhar (1996)
# 2: Donner et al. (1997)
# 3: Fu (2007)
# 4: Kristjansson et al. (2000)
# 5: Wyser (1998)
# 6: Sun and Rikus (1999), Sun (2001)
# 7: effective radius

_RERFLAG = Int(1)  # rain effective radius scheme
# 1: effective radius

_RESFLAG = Int(1)  # snow effective radius scheme
# 1: effective radius

_REGFLAG = Int(1)  # graupel effective radius scheme
# 1: effective radius

_RADR_FLAG = Int(1)  # radar reflectivity for rain
# 1: Mark Stoelinga (2005)
# 2: Smith et al. (1975), Tong and Xue (2005)
# 3: Marshall-Palmer formula (https://en.wikipedia.org/wiki/DBZ_(meteorology))

_RADS_FLAG = Int(1)  # radar reflectivity for snow
# 1: Mark Stoelinga (2005)
# 2: Smith et al. (1975), Tong and Xue (2005)
# 3: Marshall-Palmer formula (https://en.wikipedia.org/wiki/DBZ_(meteorology))

_RADG_FLAG = Int(1)  # radar reflectivity for graupel
# 1: Mark Stoelinga (2005)
# 2: Smith et al. (1975), Tong and Xue (2005)
# 3: Marshall-Palmer formula (https://en.wikipedia.org/wiki/DBZ_(meteorology))

_SEDFLAG = Int(1)  # sedimentation scheme
# 1: implicit scheme
# 2: explicit scheme
# 3: lagrangian scheme
# 4: combined implicit and lagrangian scheme

_VDIFFFLAG = Int(2)  # wind difference scheme in accretion
# 1: Wisner et al. (1972)
# 2: Mizuno (1990)
# 3: Murakami (1990)

_DO_SCALE_DEP = True  # impose scale dependence using sigma function

_DO_SEDI_UV = True  # transport of horizontal momentum in sedimentation
_DO_SEDI_W = False  # transport of vertical momentum in sedimentation

# WMP: 01-Dec-2025
# sedi_heat makes Tropical Cyclones too intense (even unphysical, may be a conversion bug for DTDT)
_DO_SEDI_HEAT = False  # transport of heat in sedimentation

_DO_SEDI_MELT_QI = False  # melt cloud ice, snow, and graupel during sedimentation
_DO_SEDI_MELT_QS = False  # melt cloud ice, snow, and graupel during sedimentation
_DO_SEDI_MELT_QG = False  # melt cloud ice, snow, and graupel during sedimentation

_DO_QA = False  # do inline cloud fraction
_RAD_SNOW = True  # include snow in cloud fraciton calculation
_RAD_GRAUPEL = True  # include graupel in cloud fraction calculation
_RAD_RAIN = True  # include rain in cloud fraction calculation
_DO_CLD_ADJ = True  # do cloud fraction adjustment

_DO_REF = False  # do radar calculations

_Z_SLOPE_LIQ = True  # use linear mono slope for autocconversions
_Z_SLOPE_ICE = True  # use linear mono slope for autocconversions

_USE_RHC_CEVAP = False  # cap of rh for cloud water evaporation
_USE_RHC_REVAP = True  # cap of rh for rain evaporation

_USE_ENHANCED_DRY_EVAP = True  # Alternative minimum evaporation formula

_CONST_VW = False  # if True, the constants are specified by v * _fac
_CONST_VI = False  # if True, the constants are specified by v * _fac
_CONST_VS = False  # if True, the constants are specified by v * _fac
_CONST_VG = False  # if True, the constants are specified by v * _fac
_CONST_VR = False  # if True, the constants are specified by v * _fac

_LIQ_ICE_COMBINE = False  # combine all liquid water, combine all solid water
_SNOW_GRAUPLE_COMBINE = True  # combine snow and graupel

_PROG_CCN = True  # do prognostic ccn (Yi Ming's method)
_PROG_CIN = False  # do prognostic cin

_FIX_NEGATIVE = True  # fix negative water species

_DO_EVAP_TIMESCALE = True  # whether to apply a timescale to evaporation
_DO_COND_TIMESCALE = True  # whether to apply a timescale to condensation

_DO_HAIL = False  # use hail parameters instead of graupel

_CONSV_CHECKER = False  # turn on energy and water conservation checker

_DO_WARM_RAIN_MP = False  # do warm rain cloud microphysics only

_DO_WBF = True  # do Wegener Bergeron Findeisen process

_DO_BIGG = False  # do Bigg process

_DO_PSD_WATER_FALL = False  # calculate cloud water terminal velocity based on PSD
_DO_PSD_ICE_FALL = False  # calculate cloud ice terminal velocity based on PSD

_DO_PSD_WATER_NUM = False  # calculate cloud water number concentration based on PSD
_DO_PSD_ICE_NUM = True  # calculate cloud ice number concentration based on PSD

_CP_HEATING = False  # update temperature based on constant pressure

_DELAY_COND_EVAP = True  # do condensation evaporation only at the last time step

_DO_SUBGRID_PROC = True  # do temperature sentive high vertical resolution processes

_FAST_FR_MLT = True  # do freezing and melting in fast microphysics
_FAST_DEP_SUB = True  # do deposition and sublimation in fast microphysics

_DO_MP_DIAG = False  # enable microphysical quantities diagnostic

_MP_TIME = Float(150.0)  # maximum microphysics time step (s)

_N0W_SIG = Float(1.2)  # intercept parameter (significant) of cloud water (Lin et al. 1983) (1/m^4) (Martin et al. 1994)
_N0I_SIG = Float(1.2)  # intercept parameter (significant) of cloud ice (Lin et al. 1983) (1/m^4) (McFarquhar et al. 2015)
_N0R_SIG = Float(8.0)  # intercept parameter (significant) of rain (Lin et al. 1983) (1/m^4) (Marshall and Palmer 1948)
_N0S_SIG = Float(3.0)  # intercept parameter (significant) of snow (Lin et al. 1983) (1/m^4) (Gunn and Marshall 1958)
_N0G_SIG = Float(4.0)  # intercept parameter (significant) of graupel (Rutledge and Hobbs 1984) (1/m^4) (Houze et al. 1979)
_N0H_SIG = Float(4.0)  # intercept parameter (significant) of hail (Lin et al. 1983) (1/m^4) (Federer and Waldvogel 1975)

_N0W_EXP = Float(66)  # intercept parameter (exponent) of cloud water (Lin et al. 1983) (1/m^4) (Martin et al. 1994)
_N0I_EXP = Float(10)  # intercept parameter (exponent) of cloud ice (Lin et al. 1983) (1/m^4) (McFarquhar et al. 2015)
_N0R_EXP = Float(6)  # intercept parameter (exponent) of rain (Lin et al. 1983) (1/m^4) (Marshall and Palmer 1948)
_N0S_EXP = Float(6)  # intercept parameter (exponent) of snow (Lin et al. 1983) (1/m^4) (Gunn and Marshall 1958)
_N0G_EXP = Float(6)  # intercept parameter (exponent) of graupel (Rutledge and Hobbs 1984) (1/m^4) (Houze et al. 1979)
_N0H_EXP = Float(4)  # intercept parameter (exponent) of hail (Lin et al. 1983) (1/m^4) (Federer and Waldvogel 1975)

_MUW = Float(11.0)  # shape parameter of cloud water in Gamma distribution (Martin et al. 1994)
_MUI = Float(1.0)  # shape parameter of cloud ice in Gamma distribution (McFarquhar et al. 2015)
_MUR = Float(1.0)  # shape parameter of rain in Gamma distribution (Marshall and Palmer 1948)
_MUS = Float(1.0)  # shape parameter of snow in Gamma distribution (Gunn and Marshall 1958)
_MUG = Float(1.0)  # shape parameter of graupel in Gamma distribution (Houze et al. 1979)
_MUH = Float(1.0)  # shape parameter of hail in Gamma distribution (Federer and Waldvogel 1975)

_ALINW = Float(3.0e7)  # "a" in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)
_ALINI = Float(11.72)  # "a" in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)
_ALINR = Float(842.0)  # "a" in Lin et al. (1983) for rain (Liu and Orville 1969)
_ALINS = Float(4.8)  # "a" in Lin et al. (1983) for snow (straka 2009)
_ALING = Float(1.0)  # "a" in Lin et al. (1983), similar to a, but for graupel (Pruppacher and Klett 2010)
_ALINH = Float(1.0)  # "a" in Lin et al. (1983), similar to a, but for hail (Pruppacher and Klett 2010)

_BLINW = Float(2.0)  # "b" in Lin et al. (1983) for cloud water (Ikawa and Saito 1990)
_BLINI = Float(0.41)  # "b" in Lin et al. (1983) for cloud ice (Ikawa and Saita 1990)
_BLINR = Float(0.8)  # "b" in Lin et al. (1983) for rain (Liu and Orville 1969)
_BLINS = Float(0.25)  # "b" in Lin et al. (1983) for snow (straka 2009)
_BLING = Float(0.5)  # "b" in Lin et al. (1983), similar to b, but for graupel (Pruppacher and Klett 2010)
_BLINH = Float(0.5)  # "b" in Lin et al. (1983), similar to b, but for hail (Pruppacher and Klett 2010)

_TICE_MLT = Float(273.16)  # can set ice melting temperature to 268 based on observation (Kay et al. 2016) (K)

_T_MIN = Float(178.0)  # minimum temperature to freeze - dry all water vapor (K)
_T_SUB = Float(184.0)  # minimum temperature for sublimation of cloud ice (K)

_RH_INC = Float(0.30)  # rh increment for complete evaporation of cloud water and cloud ice
_RH_INR = Float(0.30)  # rh increment for minimum evaporation of rain

# simple process timescales
_TAU_R2G = Float(900.0)  # rain freezing to graupel time scale (s)
_TAU_I2S = Float(300.0)  # cloud ice to snow autoconversion time scale (s)
_TAU_L2R = Float(450.0)  # cloud water to rain autoconversion time scale (s)
# other timescales
_TAU_V2L = Float(75.0)  # water vapor to cloud water condensation time scale (s)
_TAU_L2V = Float(150.0)  # cloud water to water vapor evaporation time scale (s)
_TAU_REVP = Float(600.0)  # rain evaporation time scale (s)
_TAU_FREZ = Float(600.0)  # cloud liquid freezing time scale (s)
_TAU_IMLT = Float(600.0)  # cloud ice melting time scale (s)
_TAU_SMLT = Float(900.0)  # snow melting time scale (s)
_TAU_GMLT = Float(1200.0)  # graupel melting time scale (s)
# subgridz timescales
_TAU_WBF = Float(1200.0)  # Wegener Bergeron Findeisen time scale (s)

_CCN_O = Float(90.0)  # ccn over ocean (1/cm^3)
_CCN_L = Float(270.0)  # ccn over land (1/cm^3)

_RTHRESHU = Float(7.0e-6)  # unstable critical cloud drop radius (micro m)
_RTHRESHS = Float(10.0e-6)  # stable critical cloud drop radius (micro m)

_IN_CLOUD_LIQ = True  # use in-cloud liquid
_IN_CLOUD_ICE = True  # use in-cloud frozen

_CLD_MIN = Float(0.05)  # minimum cloud fraction

_QI_LIM = Float(1.0)  # cloud ice limiter (0: no, 1: full, >1: extra) to prevent large ice build up

_QL_MLT = Float(2.0e-3)  # maximum cloud water allowed from melted cloud ice (kg/kg)
_QS_MLT = Float(1.0e-6)  # maximum cloud water allowed from melted snow (kg/kg)

_QL0_MAX = Float(2.0e-3)  # maximum cloud water value (autoconverted to rain) (kg/kg)

_PSAUT_QI_CRT = Float(1.0e-4)  # cloud ice to snow autoconversion threshold (kg/m^3)
_PWBF_QI_CRT = Float(0.8e-4)  # WBF liquid to ice freezing threshold (kg/m^3)
_PGAUT_QS_CRT = Float(0.6e-3)  # snow to graupel autoconversion threshold (0.6e-3 in Purdue Lin scheme) (kg/m^3)

_C_PAUT = Float(0.5)  # cloud water to rain autoconversion efficiency

# -----------------------------------------------------------------------
# collection efficiencies for accretion
# -----------------------------------------------------------------------
# --- Cloud Water (Liquid) 3D Accretion ---
# When True, these coefficients act as Aerodynamic Stokes Efficiencies applied to the raw 3D geometric integral.
DO_3D_ACC_CLIQ = True  # perform the new 3d accretion for cloud water
C_PSACW = Float(0.05)  # cloud water to snow (HEAVY aerodynamic reduction required)
C_PGACW = Float(0.80)  # cloud water to graupel/hail (Punches through air)
C_PRACW = Float(1.00)  # cloud water to rain
# --- Cloud Ice (Frozen) 3D Accretion ---
# When .true., these coefficients account for both Aerodynamics AND "Bounce" (Sticking Efficiency) applied to the raw 3D geometric integral.
DO_3D_ACC_CICE = False  # perform the new 3d accretion for cloud ice
C_PSACI = Float(0.05)  # cloud ice to snow accretion (Aerodynamics + Low sticking)
C_PGACI = Float(0.01)  # cloud ice to graupel accretion (Aerodynamics + Very low sticking)
C_PRACI = Float(1.00)  # cloud ice to rain accretion (High sticking to liquid)
# --- Standard Macro-Particle Accretion ---
# Interactions between precipitation species (Unaffected by 3D cice/cliq flags)
C_PGACS = Float(0.03)  # snow to graupel accretion efficiency
C_PRACS = Float(1.00)  # snow to rain accretion efficiency
C_PSACR = Float(1.00)  # rain to snow accretion efficiency
C_PGACR = Float(1.00)  # rain to graupel accretion efficiency

IS_FAC = Float(0.2)  # cloud ice sublimation temperature factor
SS_FAC = Float(0.2)  # snow sublimation temperature factor
GS_FAC = Float(0.2)  # graupel sublimation temperature factor

RH_FAC_EVAP = Float(10.0)  # cloud water evaporation relative humidity factor
RH_FAC_COND = Float(10.0)  # cloud water condensation relative humidity factor

SED_FAC = Float(1.0)  # coefficient for sedimentation fall, scale from 1.0 (implicit) to 0.0 (lagrangian)

DO_ICE_PRES_SCALING = True  # optional pressure scaling to accelerate ice settling in the upper troposphere

VW_FAC = Float(1.0)
VI_FAC = Float(1.0)
VS_FAC = Float(1.0)
VG_FAC = Float(1.0)
VR_FAC = Float(1.0)
VH_FAC = Float(1.0)

VW_MIN = Float(0.0)  # minimum fall speed for cloud water (m/s)
VI_MIN = Float(0.01)  # minimum fall speed or constant fall speed
VS_MIN = Float(0.25)  # minimum fall speed or constant fall speed
VG_MIN = Float(3.0)  # minimum fall speed or constant fall speed
VR_MIN = Float(4.0)  # minimum fall speed or constant fall speed
VH_MIN = Float(9.0)  # minimum fall speed or constant fall speed

VW_MAX = Float(0.01)  # max fall speed for cloud water (m/s)
VI_MAX = Float(1.0)  # max fall speed for ice
VS_MAX = Float(1.5)  # max fall speed for snow
VG_MAX = Float(9.0)  # max fall speed for graupel
VR_MAX = Float(12.0)  # max fall speed for rain
VH_MAX = Float(19.0)  # max fall speed for hail

XR_A = Float(0.25)  # p value in Xu and Randall (1996)
XR_B = Float(100.0)  # alpha_0 value in Xu and Randall (1996)
XR_C = Float(0.49)  # gamma value in Xu and Randall (1996)

TE_ERR = Float(1.0e-5)  # 64bit: 1.e-14, 32bit: 1.e-7; turn off to save computer time
TW_ERR = Float(1.0e-8)  # 64bit: 1.e-14, 32bit: 1.e-7; turn off to save computer time

RH_THRES = Float(0.75)  # minimum relative humidity for cloud fraction
RHC_CEVAP = Float(0.85)  # maximum relative humidity for cloud water evaporation
RHC_REVAP = Float(0.85)  # maximum relative humidity for rain evaporation

F_DQ_P = Float(3.0)  # cloud fraction adjustment for supersaturation
F_DQ_M = Float(1.0)  # cloud fraction adjustment for undersaturation

FI2S_FAC = Float(1.00)  # maximum sink of cloud ice to form snow: 0-1
FI2G_FAC = Float(1.00)  # maximum sink of cloud ice to form graupel/hail: 0-1
FS2G_FAC = Float(0.75)  # maximum sink of snow to form graupel: 0-1

BETA = Float(1.22)  # defined in Heymsfield and Mcfarquhar (1996)

REWMIN = Float(5.0)  # minimum effective radius for cloud water (micron)
REWMAX = Float(10.0)  # maximum effective radius for cloud water (micron)
REIMIN = Float(10.0)  # minimum effective radius for cloud ice (micron)
REIMAX = Float(150.0)  # maximum effective radius for cloud ice (micron)
RERMIN = Float(10.0)  # minimum effective radius for rain (micron)
RERMAX = Float(10000.0)  # maximum effective radius for rain (micron)
RESSMIN = Float(150.0)  # minimum effective radius for snow (micron)
RESSMAX = Float(10000.0)  # maximum effective radius for snow (micron)
REGMIN = Float(150.0)  # minimum effective radius for graupel (micron)
REGMAX = Float(10000.0)  # maximum effective radius for graupel (micron)

REWFAC = Float(1.0)  # this is a tuning parameter to compromise the inconsistency between
# GFDL MP's PSD and cloud water radiative property's PSD assumption.
# after the cloud water radiative property's PSD is rebuilt,
# this parameter should be 1.0.
REIFAC = Float(1.0)  # this is a tuning parameter to compromise the inconsistency between
# GFDL MP's PSD and cloud ice radiative property's PSD assumption.
# after the cloud ice radiative property's PSD is rebuilt,
# this parameter should be 1.0.
