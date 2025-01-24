"""Constants file for the GF2020 convection parameterization scheme"""

from gt4py.cartesian.gtscript import i32, f32

WRTGRADS = False
NREC = i32(0)
NTIMES = i32(0)
INT_TIME = f32(0)

NMP = i32(2)
LSMP = i32(1)
CNMP = i32(2)

USE_MEMORY = i32(-1)  # -1/0/1/2 .../10    !-

CONVECTION_TRACER = i32(0)  # 0/1:  turn ON/OFF the "convection" tracer

CLEV_GRID = i32(
    1
)  # 0/1/2: interpolation method to define environ state at the cloud levels (at face layer), default = 0
# clev_grid = 0 default method
# clev_grid = 1 interpolation method based on Tiedtke (1989)
# clev_grid = 2 for GATE soundings only

USE_REBCB = i32(1)  # 0/1: turn ON/OFF rainfall evap below cloud base, default = 0

VERT_DISCR = i32(1)  # 0/1: 1=new vert discretization, default = 0

SATUR_CALC = i32(1)  # 0/1: 1=new saturation specific humidity calculation, default = 0

SGS_W_TIMESCALE = i32(0)  # 0/1: vertical velocity for tau_ecmwf, default = 0

LIGHTNING_DIAG = i32(0)  # 0/1: do LIGHTNING_DIAGgnostics based on Lopez (2016, MWR)

APPLY_SUB_MP = i32(
    0
)  # 0/1: subsidence transport applied the to grid-scale/anvil ice/liq mix
#      ratio and cloud fraction

USE_WETBULB = i32(0)  # 0/1

OVERSHOOT = f32(0)  # 0, 1

MIN_ENTR_RATE = f32(1.0) / f32(40000.0)  # minimum allowed entrainment rate [m-1]

AUTOCONV = i32(
    1
)  # 1, 3 or 4 autoconversion formulation: (1) Kessler, (3) Kessler with temp dependence, (4) Sundvisqt

USE_MOMENTUM_TRANSP = i32(1)  # 0/1:  turn ON/OFF conv transp of momentum
LAMBAU_DEEP = f32(
    0
)  # default= 2.0 lambda parameter for deep/congestus convection momentum transp
LAMBAU_SHDN = f32(
    2
)  # default= 2.0 lambda parameter for shallow/downdraft convection momentum transp

DOWNDRAFT = i32(1)  # 0/1:  turn ON/OFF downdrafts, default = 1

MAX_TQ_TEND = f32(100)  # max T,Q tendency allowed (100 K/day)

ZERO_DIFF = i32(
    0
)  # to get the closest solution of the stable version Dec 2019 for single-moment

USE_SMOOTH_TEND = i32(
    0
)  # 0 => OFF, > 0 produces smoother tendencies (e.g.: for 1=> makes average between k-1,k,k+1)


MOIST_TRIGGER = i32(0)  # relative humidity effects on the cap_max trigger function
FRAC_MODIS = i32(0)  # use fraction liq/ice content derived from MODIS/CALIPO sensors
ADV_TRIGGER = i32(
    0
)  #  1=> Kain (2004),  2=> moisture adv trigger (Ma & Tan, 2009, Atmos Res)
EVAP_FIX = i32(1)  # fix total evap > column rainfall

OUTPUT_SOUND = i32(0)

TAU_OCEA_CP = f32(6) * f32(3600)
TAU_LAND_CP = f32(6) * f32(3600)

USE_CLOUD_DISSIPATION = f32(0)
USE_GUSTINESS = f32(0)
USE_RANDOM_NUM = f32(0)
DCAPE_THRESHOLD = f32(0)
BETA_SH = f32(2.2)
USE_LINEAR_SUBCL_MF = i32(1)
CAP_MAXS = f32(50)

# turn ON/OFF deep/shallow/mid plumes
ON = i32(1)
OFF = i32(0)

# General internal controls for the diverse options in GF
ENTRVERSION = i32(1)  # entr formulations

COUPL_MPHYSICS = True  # coupling with cloud microphysics (do not change  to false)

MELT_GLAC = True  # turn ON/OFF ice phase/melting

FEED_3DMODEL = True  # set "false" to not feedback the AGCM with the heating/drying/transport conv tendencies
USE_C1D = False  # turn ON/OFF the 'c1d' detrainment approach, don't change this.

FIRST_GUESS_W = False  # use it to calculate a 1st guess of the updraft vert velocity

LIQ_ICE_NUMBER_CONC = i32(1)

# rainfall evaporation(1) orig (2) mix orig+new (3) new
AEROEVAP = i32(1)

MAXENS = (i32(1),)  # 1  ensemble one on cap_max
MAXENS2 = (i32(1),)  #  1  ensemble two on precip efficiency
MAXENS3 = (i32(16),)  # 16 ensemble three done in cup_forcing_ens16 for G3d
ENSDIM = MAXENS * MAXENS2 * MAXENS3
ENS4 = i32(1)

# proportionality constant to estimate pressure
# gradient of updraft (Zhang and Wu, 2003, JAS)

PGCON = f32(0)
# numerical constraints
XMBMAXSHAL = f32(0.05)  # kg/m2/s
# mintracer   =  tiny(1.),&  ! kg/kg - tiny(x) #NOTE need a good solution to this 1/17/25
SMALLERQV = f32(1.0e-16)  # kg/kg
PI = f32(3.1415926536)

# miscelaneous other constants moved from within the convective scheme
AOT500 = f32(0.1)
