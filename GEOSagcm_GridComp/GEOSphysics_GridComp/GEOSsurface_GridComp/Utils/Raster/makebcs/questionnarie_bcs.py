#!/usr/bin/env python3
#
# source install/bin/g5_modules
#
# Newer GEOS code should load a module with GEOSpyD Python3 if not run:
#   module load python/GEOSpyD/Min4.10.3_py3.9
#

import os
import socket
import subprocess
import shlex
import ruamel.yaml
import shutil
import questionary
import glob
from datetime import datetime

def get_account():
   cmd = 'id -gn'
   p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
   (accounts, err) = p.communicate()
   p_status = p.wait()
   accounts = accounts.decode().split()
   return accounts[0]

def get_user():
   cmd = 'whoami'
   p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
   (user, err) = p.communicate()
   p_status = p.wait()
   user = user.decode().split()
   return user[0]

def get_configs_from_answers(answers):

   lbcsv    = answers['bcs_version']
   skipland = answers['skipland'] == 'Yes' 
   hostname = socket.gethostname()
   make_bcs_input_dir = ''
   if 'discover' in hostname or 'borgi' in hostname:
      make_bcs_input_dir = '/discover/nobackup/projects/gmao/bcs_shared/make_bcs_inputs/'
   else:
      make_bcs_input_dir = '/nobackup/gmao_SIteam/ModelData/make_bcs_inputs/'

   user   = get_user()
   expdir = '/discover/nobackup/'+user+'/BCS_PACKAGE/'+lbcsv+'/'
   now    = datetime.now()
   outdir = now.strftime("%Y%m%d%H%M%S")

   configs = []

   for grid_type in answers['grid_type']:
     print('Grid_type: ' + grid_type)
     for orslv in  answers['Ocean']:
       print('orslv: ' + orslv)
       for resolution in answers.get(grid_type,[]):
          print('resolution: ' +  resolution)
          
          NX = 8640 
          NY = 4320
          NT = 232000000
       
          maskfile = ''
       
          if orslv in['O1','T2','T3','T4','T1MOM6','T3MOM6','T4MOM6']:
             maskfile = make_bcs_input_dir+'shared/mask/GEOS5_10arcsec_mask_freshwater-lakes.nc'
             if lbcsv in ['F25', 'GM4', 'ICA']:
                maskfile = make_bcs_input_dir+'shared/mask/global.cat_id.catch.DL'
       
          if orslv in['O2','O3','CS']:
             maskfile = make_bcs_input_dir+'shared/mask/GEOS5_10arcsec_mask.nc'
             if lbcsv in ['F25', 'GM4', 'ICA']:
                maskfile = make_bcs_input_dir + 'shared/mask/global.cat_id.catch.GreatLakesCaspian_Updated.DL'
          if (maskfile == ''):
             print(" \!\!\!\! Invalid Ocean Resolution, stopping ")
             exit()
 
          if 'EASEv1' == grid_type or 'EASEv2' == grid_type:
             maskfile = 'shared/mask/GEOS5_10arcsec_mask.nc'
       
          if resolution in ['c768','c1440'] : 
             NX = 17280
             NY = 8640
          if resolution == 'c2800': 
             NX = 21600
             NY = 10800
          if resolution in ['c1536', 'c3072','c5760'] : 
             NX = 43200
             NY = 21600
       
          if 'GEOS5_10arcsec_mask' in maskfile:
             NX = 43200
             NY = 21600

          LATLON_OCEAN = False
          TRIPOL_OCEAN = False
          CUBED_SPHERE_OCEAN = False
          DATENAME = 'DE'
          POLENAME = 'PE'
          MOM_VERSION = 'UNDEF'
          if orslv in['O2','O3','O1']:
             LATLON_OCEAN = True
             DATENAME = 'DE'
             POLENAME = 'PE'
          if orslv in['T2','T3','T4']:
             TRIPOL_OCEAN = True
             MOM_VERSION = 'MOM5'
             DATENAME = 'TM'
             POLENAME = 'TM'
          if 'MOM6' in orslv:
             TRIPOL_OCEAN = True
             MOM_VERSION = 'MOM6'
             DATENAME = 'TM'
             POLENAME = 'TM'
          if  orslv == 'CS' :
             CUBED_SPHERE_OCEAN = True
       
          config = {}

          config['LATLON_OCEAN'] = LATLON_OCEAN
          config['TRIPOL_OCEAN'] = TRIPOL_OCEAN
          config['CUBED_SPHERE_OCEAN'] = CUBED_SPHERE_OCEAN
          config['DATENAME'] = DATENAME
          config['POLENAME'] = POLENAME
          config['MOM_VERSION'] =  MOM_VERSION

          config['skipland']  = skipland
          config['grid_type'] = grid_type
          config['lbcsv']     = lbcsv
          config['resolution']= resolution
          config['orslvs']    = orslv
          config ['im']  = answers['im'][resolution]
          config ['jm']  = answers['jm'][resolution]
          config ['imo'] = answers['im'][orslv]
          config ['jmo'] = answers['jm'][orslv]
          config ['NX']  = NX
          config ['NY']  = NY
          config ['NT']  = NT
          config ['MASKFILE']  = maskfile
          config ['expdir']    = expdir
          config ['outdir']    = outdir
          config ['inputdir']  = make_bcs_input_dir
          config ['NCPUS'] = 20


          configs = configs + [config]

   return configs

def ask_questions(default_grid="Cubed-Sphere"):
   
   user_name = get_user()
   questions = [
       {
            "type": "select",
            "name": "skipland",
            "message": "Skip land parameter files ?: \n \
   This option saves time when additional bcs are created that have \n \
   the exact same land parameters as an existing set of bcs because \n \
   the only difference between the two sets of bcs is the [non-tripolar] \n \
   ocean resolution. ",
            "choices": ["No", "Yes"],
            "default": "No",
        },

        {
            "type": "select",
            "name": "bcs_version",
            "message": "Select land BCS version: \n \
    BCs with 'archived*' produced by this code will differ from BCs in archived directories!!! \n \
    These differences are caused by compiler changes, code improvements and bug \n \
    fixes that were implemented since the archived BCs in the above-mentioned \n \
    directories were originally created.  The impact of these differences on \n \
    science is insignificant, and the parameter files produced by current \n \
    code are scientifically equivalent to the corresponding archived BCs. \n",
            "choices": [ \
                  "F25 : Fortuna-2_5   (archived*: n/a)", \
   "GM4 : Ganymed-4_0   (archived*: /discover/nobackup/projects/gmao/bcs_shared/legacy_bcs/Ganymed-4_0/)", \
   "ICA : Icarus        (archived*: /discover/nobackup/projects/gmao/bcs_shared/legacy_bcs/Icarus/)", \
   "NL3 : Icarus-NLv3   (archived*: /discover/nobackup/projects/gmao/bcs_shared/legacy_bcs/Icarus-NLv3/)", \
   "NL4 : NLv4 [SMAPL4] (archived*: /discover/nobackup/projects/gmao/smap/bcs_NLv4/NLv4/) \n\
          = NL3 + JPL veg height", \
   "NL5 : NLv5 [SMAPL4] (archived*: /discover/nobackup/projects/gmao/smap/SMAP_L4/L4_SM/bcs/CLSM_params/Icarus-NLv5_EASE/)\n \
         = NL3 + JPL veg height + PEATMAP", \
   "v06 : NL3 + JPL veg height + PEATMAP + MODIS snow alb", \
   "v07 : NL3 + PEATMAP", \
   "v08 : NL3 + MODIS snow alb", \
   "v09 : NL3 + PEATMAP + MODIS snow alb"],
            "default": "NL3 : Icarus-NLv3   (archived*: /discover/nobackup/projects/gmao/bcs_shared/legacy_bcs/Icarus-NLv3/)",
        },

       {
            "type": "checkbox",
            "name": "grid_type",
            "message": "Select grid types( select one or none of EASEv1 and EASEv2): \n ",
            "choices": ["Cubed-Sphere", "Lat-Lon", "EASEv2", "EASEv1"],
            "default": default_grid,
        },

       {
            "type": "checkbox",
            "name": "Lat-Lon",
            "message": "Select lat-lon resolution (multiple choices): \n ",
            "choices": ["b -- 2   deg $144x91$", "c -- 1   deg $288x181$", "d -- 1/2 deg $576x361$","e -- 1/4 deg $1152x721$"],
            "when": lambda x: "Lat-Lon" in x['grid_type'],
        },

       {
            "type": "checkbox",
            "name": "Cubed-Sphere",
            "message": "Select cubed-sphere resolution: \n ",
            "choices": [ \
                 "c12   -- 8    deg", \
                 "c24   -- 4    deg", \
                 "c48   -- 2    deg", \
                 "c90   -- 1    deg", \
                 "c180  -- 1/2  deg ( 56   km)", \
                 "c360  -- 1/4  deg ( 28   km)", \
                 "c720  -- 1/8  deg ( 14   km)", \
                 "c768  -- 1/10 deg ( 12   km)", \
                 "c1000 -- 1/10 deg ( 10   km)", \
                 "c1152 -- 1/10 deg (  8   km)", \
                 "c1440 -- 1/16 deg (  7   km)", \
                 "c1536 -- 1/16 deg (  7   km)", \
                 "c2880 -- 1/32 deg (  3   km)", \
                 "c3072 -- 1/32 deg (  3   km)", \
                 "c5760 -- 1/64 deg (  1.5 km)"],
            "when": lambda x:  "Cubed-Sphere" in x['grid_type'],
        },

       {
            "type": "checkbox",
            "name": "EASEv1",
            "message": "Select EASEv1 grid resolution: \n ",
            "choices": [ \
                 "M01  --  1km $34668x14688$", \
                 "M03  --  3km $11556x4896$", \
                 "M09  --  9km $3852x1632$", \
                 "M25  -- 25km $1383x586$", \
                 "M36  -- 36km $963x408$"],
            "when": lambda x: "EASEv1" in  x['grid_type'] ,
        },
       {
            "type": "checkbox",
            "name": "EASEv2",
            "message": "Select EASEv2 grid resolution: \n ",
            "choices": [ \
                 "M01  --  1km $34704x14616$", \
                 "M03  --  3km $11568x4872$", \
                 "M09  --  9km $3856x1624$", \
                 "M25  -- 25km $1388x584$", \
                 "M36  -- 36km $964x406$"],
            "when": lambda x: "EASEv2" in x['grid_type'],
        },

       {
            "type": "checkbox",
            "name": "Ocean",
            "message": "Select ocean resolution: \n ",
            "choices": [ \
                 "O1     --  Reynolds (Lon/Lat Data-Ocean:    $360x180$ )", \
                 "O2     --  Reynolds (Lon/Lat Data-Ocean:   $1440x720$ )", \
                 "O3     --  OSTIA    (Lon/Lat Data-Ocean:   $2880x1440$)", \
                 "T2     --  Tripolar (MOM5-Tripolar-Ocean:   $360x200$ )", \
                 "T3     --  Tripolar (MOM5-Tripolar-Ocean:   $720x410$ )", \
                 "T4     --  Tripolar (MOM5-Tripolar-Ocean:  $1440x1080$)", \
                 "T1MOM6 --  Tripolar (MOM6-Tripolar-Ocean:    $72x36$  )", \
                 "T3MOM6 --  Tripolar (MOM6-Tripolar-Ocean:   $580x458$ )", \
                 "T4MOM6 --  Tripolar (MOM6-Tripolar-Ocean:  $1440x1080$)", \
                 "CS     --  Cubed-Sphere Ocean  (Cubed-Sphere Data-Ocean)"],
            "when": lambda x:  "EASEv1" not in x['grid_type'] and "EASEv2" not in x['grid_type'],
        },
   ]
   answers = questionary.prompt(questions)
   if 'EASEv1' in answers['grid_type'] or 'EASEv2' in answers['grid_type'] : answers['Ocean'] = ['O1  $360x180$']
   answers['bcs_version'] = answers['bcs_version'].split()[0]

   path_q  = [
           {
            "type": "path",
            "name": "out_path",
            "message": "Enter desired BCS output directory (incl. full path) or press ENTER to use the default:",
            "default": "/discover/nobackup/"+user_name+"/BCS_PACKAGE/"+answers['bcs_version']+'/'
          },
   ]
   path_ = questionary.prompt(path_q)

   answers["out_path"] = path_["out_path"] 

   im = {}
   jm = {}

   for name_  in ['Lat-Lon', 'EASEv1', 'EASEv2', 'Ocean']: 
      long_res = answers.get(name_,[])
      res = []
      for x in long_res:
         short_res = x.split()[0]
         res = res + [short_res]
         imxjm = x.split('$')
         if len(imxjm) == 1 :
            #'CS' will be filled later on
            im[short_res] = 0
            jm[short_res] = 0
         else:
            strings = imxjm[1].split('x')
            im[short_res] = int(strings[0])
            jm[short_res] = int(strings[1])
      answers[name_] = res
 
   res = []
   for x in answers.get('Cubed-Sphere',[]):
      short_res = x.split()[0]
      res = res + [short_res]
      i = int(short_res[1:])
      im[short_res] = i
      jm[short_res] = 6*i
   answers['Cubed-Sphere'] = res 

   answers['im'] = im
   answers['jm'] = jm

   return answers

def print_config( config, indent = 0 ):
   print('\n')
   for k, v in config.items():
     if isinstance(v, dict):
        print("   " * indent, f"{k}:")
        print_config(v, indent+1)
     else:
        print("   " * indent, f"{k}: {v}")

if __name__ == "__main__":

   answers = ask_questions()
   for key, value in answers.items():
     if (key == 'im' or key) =='jm' :
        for key1, value1 in answers[key].items():
           print(key1, value1)

   configs = get_configs_from_answers(answers)
   print('\n print config file:\n')
   for config in configs:
      print_config(config)   
