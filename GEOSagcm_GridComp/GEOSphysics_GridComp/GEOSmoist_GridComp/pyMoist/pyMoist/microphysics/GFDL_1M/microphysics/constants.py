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
