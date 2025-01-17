"""GEOS5 Constants"""

from gt4py.cartesian.gtscript import i32, f32

# Shared GF Constants
#NOTE if GF2020 is implemented, these need to be moved to a shared file
# plume spectral size
INTEGER ,PARAMETER  :: maxiens=3 , deep=1 , shal=2 , mid=3
CHARACTER(LEN=10),PARAMETER,DIMENSION(maxiens)  :: cumulus_type = (/ &
                                                'deep      ' &
                                                ,'shallow   ' &
                                                ,'mid       ' &
                                                                /)
!------------------- namelist variables
!-- plume to be activated (1 true, 0 false): deep, shallow, congestus
INTEGER, DIMENSION(maxiens) :: icumulus_gf = (/1,0,1/)

!-- choice for the closures:
!--  deep   : 0 ensemble (all)          , 1 GR, 4 ll omega, 7 moist conv, 10 PB
!--  shallow: 0 ensemble (Wstar/BLQE)   , 1 Wstar, 4 heat-engine or 7 BLQE
!--  mid    : 0 ensemble (Wstar/BLQE/PB), 1 Wstar, 2 BLQE, 3 PB, 4 PB_BL
INTEGER, DIMENSION(maxiens) :: closure_choice = (/0,  7,  3/) ! deep, shallow, congestus

!-- gross entraiment rate: deep, shallow, congestus
REAL,       DIMENSION(maxiens) :: cum_entr_rate = (/&
                                        1.00e-4  & !deep
                                        ,2.00e-3  & !shallow
                                        ,9.00e-4  & !mid
                                                /)

INTEGER :: USE_TRACER_TRANSP = 1 != 0/1     - default 1

INTEGER :: USE_TRACER_SCAVEN = 1 != 0/1/2/3 - default 2

INTEGER :: USE_FLUX_FORM     = 1 != 1/2/3   - default 1

INTEGER :: USE_FCT           = 1 != 0/1     - default 1 (only for USE_FLUX_FORM     = 2)

INTEGER :: USE_TRACER_EVAP   = 1 != 0/1     - default 1 (only for USE_TRACER_SCAVEN > 0)

INTEGER :: USE_SCALE_DEP     = 1 != 0/1:  scale dependence flag, default = 1

INTEGER :: DICYCLE           = 1 != 0/1:  diurnal cycle closure, default = 1

REAL    :: ALP1              = 1 != 0/0.5/1: apply subsidence transport of LS/anvil cloud fraction using
                                !=          time implicit discretization

                                != boundary condition determination for the plumes
INTEGER :: BC_METH           = 0 ! 0: simple arithmetic mean around k22
                                ! 1: mass weighted mean around k22

REAL,   DIMENSION(maxiens) :: CUM_AVE_LAYER     =(/50.,   30.,   50. /)!= layer depth for average the properties
                                                                    != of source air parcels (mbar)
REAL    ::  AVE_LAYER         != layer depth for average the properties of source air parcels (mbar)

REAL    ::  TAU_DEEP         = 5400.  != deep      convective timescale
REAL    ::  TAU_MID          = 3600.  != congestus convective timescale

REAL    ::  C0_DEEP          = 2.e-3 != default= 3.e-3   conversion rate (cloud to rain, m-1) - for deep      plume
REAL    ::  C0_MID           = 2.e-3 != default= 2.e-3   conversion rate (cloud to rain, m-1) - for congestus plume
REAL    ::  C0_SHAL          = 0.    != default= 0.e-3   conversion rate (cloud to rain, m-1) - for shallow   plume
REAL    ::  QRC_CRIT         = 2.e-4 != default= 2.e-4   kg/kg
REAL    ::  QRC_CRIT_LND     = 3.e-4 != default= 2.e-4   kg/kg
REAL    ::  QRC_CRIT_OCN     = 3.e-4 != default= 2.e-4   kg/kg
REAL    ::  C1               = 0.0   != default= 1.e-3   conversion rate (cloud to rain, m-1) - for the 'C1d' detrainment approach

!- physical constants
REAL, PARAMETER ::  &
rgas    = 287.,    & ! J K-1 kg-1
cp      = 1004.,   & ! J K-1 kg-1
rv      = 461.,    & ! J K-1 kg-1
p00     = 1.e5,    & ! hPa
tcrit   = 258.,    & ! K
g       = MAPL_GRAV,&! m s-2
cpor    = cp/rgas, &
xlv     = 2.5e6,   & ! J kg-1
akmin   = 1.0,     & ! #
tkmin   = 1.e-5,   & ! m+2 s-2
ccnclean= 250.,    & ! # cm-3
T_0     = 273.16,  & ! K
T_ice   = 235.16,  & ! K
xlf     = 0.333e6, & ! latent heat of freezing (J K-1 kg-1)
max_qsat= 0.5,     & ! kg/kg
mx_buoy = cp*5. + xlv*2.e-3 ! temp exc=5 K, q deficit=2 g/kg (=> mx_buoy ~ 10 kJ/kg)

! Default autoconversion parameter for GEOS-Chem species [s-1]
REAL, PARAMETER       :: KC_DEFAULT_GCC = 5.e-3

LOGICAL :: CNV_2MOM = .FALSE.