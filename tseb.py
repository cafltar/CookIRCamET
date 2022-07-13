import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
import utils, aero, canopy, fluxes, solar

from pandas import read_csv, DataFrame
from constants import *

import logging
logging.basicConfig(level=logging.INFO)
home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Working')

#read from config file and input file
data = inputs()
resistance = aero.resistances(data)
radiation = solar.radiation(data)
partition = canopy.radiation(data)
flux = fluxes.rad_fluxes(data)

#calculate roughness params
resistance.roughness_lengths()
#calculate resistances
resistance.roughness_lengths()
resistance.rah_calc()
resistance.rx_calc()
resistance.rs_calc()
#calculate solar radiation
radiation.solar_angles()
radiation.solar_angles()
radiation.solar_angles()
radiation.solar_angles()

#calculate canopy transmissivity/reflectivity/soil albedo
partition.fdhc()
partition.
partition.
partition.
partition.
partition.

#Calculate radiation partitioning
fluxes.Rnlw()
fluxes.Rnsw()

#calculate aerodynamic resistances
resistance.roughness_lengths
resistance.rs
resistance.rx
resistance.rah

#calculate ground heat flux, sensible heat flux, latent heat flux
G = fluxes.G(Rns)
Hs
Hc
LEs
LEc
E
T

#write results to file
if  modeltype=='2SEB':
    out_df = DataFrame(columns=['datetime (UTC)','sun elevation (deg)','sun azimuth (deg)','LEc (W/m2)','LEs (W/m2)','Hc (W/m2)','Hs (W/m2)','H (W/m2)','G (W/m2)','Sns (W/m2)','Snc (W/m2)','Lns (W/m2)','Lnc (W/m2)','E (mm)', 'T (mm)'])
elif modeltype=='3SEB':
    out_df = DataFrame(columns=['datetime (UTC)','sun elevation (deg)','sun azimuth (deg)','LEc (W/m2)','LEs (W/m2)','LEr (W/m2)','Hc (W/m2)','Hs (W/m2)','Hr (W/m2)','G (W/m2)','Sns (W/m2)','Snr (W/m2)','Snc (W/m2)','Lns (W/m2)','Lnr (W/m2)','Lnc (W/m2)','E (mm)', 'T (mm)'])
else:
    raise(NameError('Invalid model'))
    
#write out_df to file 
outname = os.path.join(,'.csv')
out_df.to_csv(outname)
