"""Constants used for shared pyMoist stencils/functions."""

from ndsl.dsl.typing import Bool, Float, Int

from pyMoist.constants import MAPL_AIRMW, MAPL_ALHF, MAPL_ALHL, MAPL_ALHS, MAPL_CP, MAPL_GRAV, MAPL_H2OMW, MAPL_PI

# surface type constants
SRF_TYPE_OCEAN = Int(0)
SRF_TYPE_LAND = Int(1)
SRF_TYPE_SNOW = Int(2)
SRF_TYPE_ICE = Int(3)
SRF_TYPE_LANDICE = Int(4)

# Ice fraction polynomial options
RAW_MODIS_POLYNOMIAL = Int(1)
JASON_ICE_POLYNOMIAL = Int(2)
V12_ICE_POLYNOMIAL = Int(3)
ICE_FRACTION_POLYNOMIAL = Int(3)

# Ice fraction glaciation shift options
GLAC_SHIFT_CONV = 7.5  # Will hold tropical liquid very high
GLAC_SHIFT_LAND = -1.0
GLAC_SHIFT_LANDICE = 3.0
GLAC_SHIFT_OCEAN = 4.0
GLAC_SHIFT_SEAICE = 2.0
GLAC_SHIFT_SNOW = 1.0

# Jason ice_fraction constants
# In anvil/convective clouds
JAT_ICE_ALL = Float(245.16)
JAT_ICE_MAX = Float(261.16)
JAT_ICE_PWR = Float(2.0)
# Over Land Ice SRF_TYPE == 4
JLIT_ICE_ALL = Float(236.16)
JLIT_ICE_MAX = Float(261.16)
JLIT_ICE_PWR = Float(5.0)
# Over Ice SRF_TYPE == 3
JIT_ICE_ALL = Float(236.16)
JIT_ICE_MAX = Float(261.16)
JIT_ICE_PWR = Float(5.0)
# Over Snow SRF_TYPE = 2
JST_ICE_ALL = Float(236.16)
JST_ICE_MAX = Float(261.16)
JST_ICE_PWR = Float(5.0)
# Over Land     SRF_TYPE = 1
JLT_ICE_ALL = Float(239.16)
JLT_ICE_MAX = Float(261.16)
JLT_ICE_PWR = Float(2.0)
# Over Oceans   SRF_TYPE = 0
JOT_ICE_ALL = Float(238.16)
JOT_ICE_MAX = Float(263.16)
JOT_ICE_PWR = Float(4.0)
# end of ice fraction constants

USE_BERGERON = False
USE_AEROSOL_NN = True
USE_NCLOUD_CLIM = False

WSUB_OPTION = Int(-1)
PDFSHAPE = Int(1)

EPSILON = MAPL_H2OMW / MAPL_AIRMW
K_COND = Float(2.4e-2)  # J m**-1 s**-1 K**-1
DIFFU = Float(2.2e-5)  # m**2 s**-1
TAUFRZ = Float(600.0)  # timescale for freezing
TAUMLT = Float(300.0)  # timescale for melting
CFMIN = Float(1.0e-5)  # minimum cloud fraction
QCMIN = Float(1.0e-8)  # minimum condensate (ql & qi) values
QPMIN = Float(1.0e-15)  # minimum precipitate (qr, qs, qg) values
DQCMAX = Float(1.0e-4)

T_HOM = Float(-38.0)  # Homogeneous freezing at -38 C
T_WIDTH = Float(10.0)  # Homogeneous freezing with -38 : -28 C

R_AIR = Float(3.47e-3)  # m3 Pa kg-1K-1

# particle radius
# Jason
ABETA = Float(0.07)
R13BBETA = Float(1.0) / Float(3.0) - Float(0.14)
BX = Float(100.0) * (Float(3.0) / (Float(4.0) * MAPL_PI)) ** (Float(1.0) / Float(3.0))
# Liquid  based on DOI 10.1088/1748-9326/3/4/045021
RHO_W = Float(1000.0)  # Density of liquid water in kg/m^3
RHO_S = Float(100.0)
RHO_G = Float(500.0)
RHO_I = Float(890.0)
LDISS = Float(0.07)  # tunable dispersion effect
LK = Float(0.75)  # tunable shape effect (0.5:1)
LBE = Float(1.0) / Float(3.0) - Float(0.14)
LBX = Float(LDISS * Float(1.0e3)) * (Float(3.0) / (Float(4.0) * MAPL_PI * LK * RHO_W * 1.0e-3)) ** (Float(1.0) / Float(3.0))
# LDRADIUS eqs are in cgs units

# combined constants
CPBGRAV = MAPL_CP / MAPL_GRAV
GRAVBCP = MAPL_GRAV / MAPL_CP
ALHLBCP = MAPL_ALHL / MAPL_CP
ALHFBCP = MAPL_ALHF / MAPL_CP
ALHSBCP = MAPL_ALHS / MAPL_CP

# base grid length for sigma calculation
SIGMA_DX = Float(750.0)
SIGMA_EXP = Float(2.0)

# control for order of plumes
SH_MD_DP = False

# Radar parameters
DBZ_VAR_INTERCP = Int(2)  # use variable intercept parameters: 1 - on, 2 - snow boost, 3 - hail instead of graupel
DBZ_LIQUID_SKIN = Int(1)  # use liquid skin on snow(1) and graupel/hail(2) in warm environments
REFL10CM_ALLOW_WET_GRAUPEL = Bool(False)
REFL10CM_ALLOW_WET_SNOW = Bool(True)
LIQUID_SKIN_SNOW = Bool(False)
LIQUID_SKIN_GRAUPEL = Bool(False)
LIQUID_SKIN_HAIL = Bool(True)
W_START = Float(6.0)
W_FULL = Float(12.0)

# Thompson radar constants
IIWARM = Bool(False)
# ..Rho_not used in fallspeed relations (rho_not/rho)**.5 adjustment.
RHO_NOT = Float(101325.0) / (Float(287.05) * Float(298.0))
# Mass power law relations:  mass = am*D**bm
#  Snow from Field et al. (2005), others assume spherical form.
AM_R = MAPL_PI * RHO_W / Float(6.0)
BM_R = Float(3.0)
AM_S = Float(0.069)
BM_S = Float(2.0)
BM_S_2 = Float(2.0 ** Int(2))
BM_S_3 = Float(2.0 ** Int(3))
AM_G = MAPL_PI * RHO_G / Float(6.0)
BM_G = Float(3.0)
AM_I = MAPL_PI * RHO_I / Float(6.0)
BM_I = Float(3.0)
AM_S_R001 = (Float(0.176) / Float(0.93)) * (Float(6.0) / MAPL_PI) * (Float(6.0) / MAPL_PI) * (AM_S / Float(900.0)) ** Int(2)
AM_G_R001 = (Float(0.176) / Float(0.93)) * (Float(6.0) / MAPL_PI) * (Float(6.0) / MAPL_PI) * (AM_G / Float(900.0)) ** Int(2)
# Fallspeed power laws relations:  v = (av*D**bv)*exp(-fv*D)
#  Rain from Ferrier (1994), ice, snow, and graupel from
#  Thompson et al (2008). Coefficient fv is zero for graupel/ice.
AV_R = Float(4854.0)
BV_R = Float(1.0)
FV_R = Float(195.0)
AV_S = Float(40.0)
BV_S = Float(0.55)
FV_S = Float(100.0)
AV_G = Float(442.0)
BV_G = Float(0.89)
BV_I = Float(1.0)
AV_C = Float(0.316946e8)
BV_C = Float(2.0)
# Generalized gamma distributions for rain, graupel and cloud ice.
# N(D) = N_0 * D**mu * exp(-lamda*D);  mu=0 is exponential.
MU_R = Float(0.0)
MU_G = Float(0.0)
MU_I = Float(0.0)
# Sum of two gamma distrib for snow (Field et al. 2005).
# N(D) = M2**4/M3**3 * [Kap0*exp(-M2*Lam0*D/M3)
# + Kap1*(M2/M3)**mu_s * D**mu_s * exp(-M2*Lam1*D/M3)]
# M2 and M3 are the (bm_s)th and (bm_s+1)th moments respectively
# calculated as function of ice water content and temperature.
MU_S = Float(0.6357)
KAP0 = Float(490.6)
KAP1 = Float(17.46)
LAM0 = Float(20.78)
LAM1 = Float(3.29)
# Y-intercept parameter for graupel is not constant and depends on
#  mixing ratio.  Also, when mu_g is non-zero, these become equiv
#  y-intercept for an exponential distrib and proper values are
#  computed based on same mixing ratio and total number concentration.
GONV_MIN = Float(1.0e2)
GONV_MAX = Float(1.0e6)
# For snow moments conversions (from Field et al. 2005)
SA = [5.065339, -0.062659, -3.032362, 0.029469, -0.000285, 0.31255, 0.000204, 0.003199, 0.0, -0.015952]
SB = [0.476221, -0.015896, 0.165977, 0.007468, -0.000141, 0.060366, 0.000079, 0.000594, 0.0, -0.003577]

SA3_BM_S = SA[2] * BM_S
SA4_BM_S = SA[3] * BM_S
SA6_BM_S = SA[5] * BM_S * BM_S
SA7_BM_S = SA[6] * BM_S
SA8_BM_S = SA[7] * BM_S * BM_S
SA10_BM_S = SA[9] * BM_S * BM_S * BM_S

SB3_BM_S = SB[2] * BM_S
SB4_BM_S = SB[3] * BM_S
SB6_BM_S = SB[5] * BM_S * BM_S
SB7_BM_S = SB[6] * BM_S
SB8_BM_S = SB[7] * BM_S * BM_S
SB10_BM_S = SB[9] * BM_S * BM_S * BM_S

# option for cloud liq/ice radii
LIQ_RADII_PARAM = Int(1)
ICE_RADII_PARAM = Int(1)

# defined to determine CNV_FRACTION
CNV_FRACTION_MIN = Float(500.0)
CNV_FRACTION_MAX = Float(1500.0)
CNV_FRACTION_EXP = Float(1.0)
