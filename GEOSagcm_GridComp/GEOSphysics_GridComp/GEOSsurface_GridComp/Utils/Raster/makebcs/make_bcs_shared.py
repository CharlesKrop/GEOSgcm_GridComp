#!/usr/bin/env python3
#
# source install/bin/g5_modules

def get_script_head() :

   return  """#!/bin/csh -x

#SBATCH --output={EXPDIR}/{TMP_DIR}/logs/{GRIDNAME}/{GRIDNAME2}.log
#SBATCH --error={EXPDIR}/{TMP_DIR}/logs/{GRIDNAME}/{GRIDNAME2}.err
#SBATCH --account={account}
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --job-name={GRIDNAME2}.j
#SBATCH --constraint=sky|cas

cd {SCRATCH_DIR}

if( ! -d bin ) then
  /bin/ln -s {bin_dir}
endif

source bin/g5_modules
setenv MASKFILE {MASKFILE}
setenv MAKE_BCS_INPUT_DIR {MAKE_BCS_INPUT_DIR}
limit stacksize unlimited

if( ! -d geometry ) then
  mkdir -p geometry land/shared til rst data/MOM5 data/MOM6 clsm/plots
endif
"""

def get_change_til_file(grid_type):
  script = ""
  if grid_type == "Cubed-Sphere":

       script = """

cd geometry/{GRIDNAME}/
/bin/rm -f sedfile
if( {CUBED_SPHERE_OCEAN} == True ) then
cat > sedfile << EOF
s/{DATENAME}{IMO}x{POLENAME}{JMO}-Pfafstetter/OC{nc}x{nc6}-CF/g
s/CF{NC}x6C/PE{nc}x{nc6}-CF/g
EOF
sed -f sedfile       {GRIDNAME}{RS}.til > tile.file
/bin/mv -f tile.file {GRIDNAME}{RS}.til
/bin/rm -f sedfile
else
cat > sedfile << EOF
s/CF{NC}x6C/PE{nc}x{nc6}-CF/g
s/{DATENAME}{IMO}x{POLENAME}{JMO}-Pfafstetter/PE{imo}x{jmo}-{DATENAME}/g
EOF
sed -f sedfile       {GRIDNAME}{RS}.til > tile.file
/bin/mv -f tile.file {GRIDNAME}{RS}.til
/bin/rm -f sedfile
endif
cd ../../

"""

  if grid_type == "Lat-Lon" :

     script = """
cd geometry/{GRIDNAME}/
/bin/rm -f sedfile
cat > sedfile << EOF
s/DC{IM}xPC{JM}/PC{im}x{jm}-DC/g
s/{DATENAME}{IMO}x{POLENAME}{JMO}-Pfafstetter/PE{imo}x{jmo}-{DATENAME}/g
EOF
sed -f sedfile       {GRIDNAME}{RS}.til > tile.file
/bin/mv -f tile.file {GRIDNAME}{RS}.til
/bin/rm -f sedfile
cd ../../

"""
  return script

def get_script_mv(grid_type):

   mv_template = """

mkdir -p geometry/{GRIDNAME}
/bin/mv {GRIDNAME}.j geometry/{GRIDNAME}/.
/bin/cp til/{GRIDNAME}{RS}.til geometry/{GRIDNAME}/.
if( {TRIPOL_OCEAN} == True ) /bin/cp til/{GRIDNAME}{RS}.TRN geometry/{GRIDNAME}/.

/bin/mv rst til geometry/{GRIDNAME}/.

mkdir -p land/shared/

if(-f land/shared/CO2_MonthlyMean_DiurnalCycle.nc4) then
    echo "CO2 file exists."
else
    /bin/cp {MAKE_BCS_INPUT_DIR}/land/CO2/v1/CO2_MonthlyMean_DiurnalCycle.nc4  land/shared/CO2_MonthlyMean_DiurnalCycle.nc4
    echo "File does not exist. CO2 file copied successfully."
endif

mkdir -p land/{GRIDNAME}/clsm

/bin/mv clsm/irrig.dat   land/{GRIDNAME}/irrigation_{RC}.dat
/bin/mv clsm/vegdyn.data land/{GRIDNAME}/vegdyn_{RC}.dat
/bin/mv clsm/nirdf.dat   land/{GRIDNAME}/nirdf_{RC}.dat
/bin/mv clsm/visdf.dat   land/{GRIDNAME}/visdf_{RC}.dat
/bin/mv clsm/lai.dat     land/{GRIDNAME}/lai_clim_{RC}.data
/bin/mv clsm/green.dat   land/{GRIDNAME}/green_clim_{RC}.data
/bin/mv clsm/lnfm.dat    land/{GRIDNAME}/lnfm_clim_{RC}.data
/bin/mv clsm/ndvi.dat    land/{GRIDNAME}/ndvi_clim_{RC}.data

/bin/mv clsm/ar.new \\
        clsm/bf.dat \\
        clsm/ts.dat \\
        clsm/catchment.def \\
        clsm/cti_stats.dat \\
        clsm/tau_param.dat \\
        clsm/soil_param.dat \\
        clsm/mosaic_veg_typs_fracs \\
        clsm/soil_param.first \\
        clsm/bad_sat_param.tiles \\
        clsm/README \\
        clsm/lai.* \\
        clsm/AlbMap* \\
        clsm/g5fmt \\
        clsm/vegetation.hst2 \\
        clsm/pfaf_fractions.dat \\
        clsm/plots \\
        clsm/CLM_veg_typs_fracs \\
        clsm/Grid2Catch_TransferData.nc \\
        clsm/CLM_NDep_SoilAlb_T2m \\
        clsm/CLM4.5_abm_peatf_gdp_hdm_fc \\
        clsm/catch_params.nc4 \\
        clsm/catchcn_params.nc4 \\
        clsm/country_and_state_code.data \\
        land/{GRIDNAME}/clsm/

""" 
   mv_template = mv_template + get_change_til_file(grid_type)
   mv_template = mv_template + """

/bin/mv clsm/mkCatchParam.log ../logs/{GRIDNAME}/mkCatchParam.log
if( ! -d ../../geometry ) then
  mkdir -p ../../geometry ../../land ../../logs
  chmod 755 -R geometry land logs
endif
/bin/mv geometry/{GRIDNAME} ../../geometry/.
/bin/mv land/{GRIDNAME} ../../land/.
if( ! -d ../../land/shared ) then
    mkdir -p ../../land/shared
    chmod 755 -R ../../land/shared
endif
  /bin/mv land/shared/CO2_MonthlyMean_DiurnalCycle.nc4 ../../land/shared/.
cd ..
/bin/mv logs/{GRIDNAME} ../logs/.
cd ../
chmod 755 -R geometry land logs
/bin/rm -r {TMP_DIR}
"""

   return mv_template

