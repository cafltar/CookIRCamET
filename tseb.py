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

#calculate roughness params
resistance.roughness_lengths()

#calculate resistances
resistance.rah_calc()
resistance.rx_calc()
resistance.rs_calc()
data.Tac = 

#calculate solar radiation
radiation.solarangles()
radiation.Rso()
radiation.Kbeam()

#calculate canopy transmissivity/reflectivity/soil albedo
partition = canopy.radiation(data, radiation)
partition.fdhc_calc()
partition.fsis_calc()
partition.mrf_plf_calc()
partition.taudir()
partition.taudiff()

#Calculate radiation partitioning
flux = fluxes.rad_fluxes(data,radiation,partition)
Lns = flux.Lns()
Lnr = flux.Lnr()
Lnc = flux.Lnc()
Sns = flux.Sns()
Snr = flux.Snr()
Snc = flux.Snc()
Rns = Sns + Lns
Rnr = Snr + Lnr
Rnc = Snc + Lnc 

#calculate ground heat flux, sensible heat flux, latent heat flux
G = fluxes.G(Rns+Rnr)
Hr =
Hs
Hc
LEr
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
