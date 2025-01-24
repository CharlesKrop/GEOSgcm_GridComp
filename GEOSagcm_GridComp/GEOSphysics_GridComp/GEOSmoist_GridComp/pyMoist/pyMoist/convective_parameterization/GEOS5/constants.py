"""GEOS5 specific constants"""

from gt4py.cartesian.gtscript import i32, f32
import pyMoist.constants as global_constants


# turn ON/OFF deep/shallow/mid plumes
ON = i32(1)
OFF = i32(0)

# General control for the diverse options in GF
ENTRNEW = True  # new entr formulation
COUPL_MPHYSICS = True  # coupling with cloud microphysics
# (do not change  to false)
MELT_GLAC = True  # turn ON/OFF ice phase/melting
DOWNDRAFT = True  # turn ON/OFF downdrafts
MOMENTUM = True  # turn ON/OFF conv transp of momentum

FEED_3DMODEL = True  # set "false" to not feedback the AGCM with the
# heating/drying/transport conv tendencies
USE_C1D = True  # turn ON/OFF the 'c1d' detrainment approach

VERT_DISCR = i32(1)

# autonversion formulation: (1) original , (2) MP_GT
CLOUDMP = i32(1)

# autonversion formulation: (1) Kessler, (2) Berry, (3) NOAA, (4) Sundvisqt
autoconv = i32(1)
# rainfall evaporation(1) orig (2) mix orig+new (3) new
aeroevap = i32(1)

MAXENS = i32(1)  # 1  ensemble one on cap_max
MAXENS2 = i32(1)  # 1  ensemble two on precip efficiency
MAXENS3 = i32(16)  # 16 ensemble three done in cup_forcing_ens16 for G3d
ENSDIM = MAXENS * MAXENS2 * MAXENS3
ENS4 = i32(1)


# physical constants
# NOTE we need to define a T_ice local to
# this code as it is *different* than the T_ice
# that comes from ConvPar_GF_Shared. Without
# doing this, the code is non-zero-diff

T_ICE_LOCAL = f32(250.16)  # K

# proportionality constant to estimate pressure
# gradient of updraft (Zhang and Wu, 2003, JAS)
PGCD = f32(1)
PGCON = f32(0)

# numerical constraints
XMBMAXSHAL = f32(0.05)  # kg/m2/s
# MINTRACER   =  tiny(1.),&  ! kg/kg - tiny(x) NOTE 1/17/2025 need a good solution to this
SMALLERQV = f32(1.0e-16)  # kg/kg
