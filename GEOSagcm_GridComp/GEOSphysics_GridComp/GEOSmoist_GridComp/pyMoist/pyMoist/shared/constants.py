from ndsl.dsl.typing import Float, Int, Bool
from pyMoist.constants import MAPL_AIRMW, MAPL_ALHF, MAPL_ALHL, MAPL_ALHS, MAPL_H2OMW, MAPL_CP, MAPL_GRAV, MAPL_PI

# surface type constants
SRF_TYPE_OCEAN = Int(0)
SRF_TYPE_LAND = Int(1)
SRF_TYPE_SNOW = Int(2)
SRF_TYPE_ICE = Int(3)
SRF_TYPE_LANDICE = Int(4)

RAW_MODIS_POLYNOMIAL = Int(1)
JASON_ICE_POLYNOMIAL = Int(2)
V12_ICE_POLYNOMIAL = Int(3)
ICE_FRACTION_POLYNOMIAL = Int(3)

# --------------------------------------------------
# ice_fraction constants
# --------------------------------------------------
# FINAL REVISED SURFACE-DEPENDENT CLOUD PHASE CONSTANTS (Bias-Corrected)
# 1. Anvil / Convective Clouds (Deep updrafts, clean high-altitude cores)
# Observations: High updraft velocity dynamically preserves liquid down to deep
# temperatures. Freezing drops off exponentially close to homogeneous limit.
AT_ICE_ALL = Float(233.16)  # Strict homogeneous limit (-40C)
AT_ICE_MAX = Float(268.16)  # Latent heat maintains liquid until -5C
AT_ICE_PWR = Float(4.5)  # Asymmetric S-curve to shield liquid peak
# 2. Land Ice (Antarctica / Greenland)
# Bias Fix: Widens mixed-phase window and raises PWR to fix the severe polar
# downward LW deficit (-25 W/m²) and clear lower troposphere cold pools.
LIT_ICE_ALL = Float(234.16)  # Deep absolute freeze floor lowered to -39C
LIT_ICE_MAX = Float(268.15)  # Delays plateau glaciation onset to -5C
LIT_ICE_PWR = Float(4.2)  # Highly emissive summer liquid water shield
# 3. Sea Ice (Arctic / Southern Ocean Pack Ice)
# Bias Fix: Expands liquid window to restore thin supercooled liquid cloud tops.
# Eliminates the MAM positive SW surface heating and matches vertical ERA5 QL mass.
IT_ICE_ALL = Float(235.16)  # Lowers homogeneous floor to -38C
IT_ICE_MAX = Float(271.15)  # Maintains warm liquid threshold near -2C
IT_ICE_PWR = Float(4.5)  # High exponent shifts excess QI mass back to QL
# 4. Snow Surface (High-latitude winter land)
# Bias Fix: Shuts down spring continental shortwave overestimation and boundary
# layer cold biases across snow-covered Siberia and northern boreal zones.
ST_ICE_ALL = Float(236.16)  # Total freeze-out pushed down to -37C
ST_ICE_MAX = Float(268.15)  # Delays land ice crystal production to -5C
ST_ICE_PWR = Float(4.0)  # Stronger power curve guards spring liquid path
# 5. Land (Ice-free, ice-nucleating aerosol rich)
# Observations: Mineral and biological dust act as potent heterogeneous INPs.
# Mixed-phase clouds glaciate rapidly and uniformly throughout the -10C to -25C zone.
LT_ICE_ALL = Float(241.16)  # Dust forces total glaciation early at -32C
LT_ICE_MAX = Float(266.16)  # Active INPs seed ice starting at -7C
LT_ICE_PWR = Float(1.5)  # Near-linear transition curve clears liquid pooling
# 6. Oceans (Open water, mid-to-high latitude marine boundary layers)
# Bias Fix: Synchronized with Sea Ice limits to maintain high open-water marine
# cloud optical depths, mitigating mid-latitude high-altitude liquid biases.
OT_ICE_ALL = Float(235.16)  # Drops to 100% ice near -38C
OT_ICE_MAX = Float(271.15)  # Highly liquid-dominated near 0C to -2C
OT_ICE_PWR = Float(4.5)  # High power protects high marine LWP peak

# Jason constants
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
AM_S_R001 = (Float(0.176) / Float(0.93)) * (Float(6.0) / MAPL_PI) * (Float(6.0) / MAPL_PI) * (am_s / Float(900.0)) ** Int(2)
AM_G_R001 = (Float(0.176) / Float(0.93)) * (Float(6.0) / MAPL_PI) * (Float(6.0) / MAPL_PI) * (am_g / Float(900.0)) ** Int(2)
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
