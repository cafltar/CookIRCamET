import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
import utils, aero, canopy, fluxes, solar
from pandas import read_csv, DataFrame

import logging
logging.basicConfig(level=logging.INFO)
home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Working')

#read from config file

#measurements:
modeltype = '2SEB'
#1SEB
#get Trad from mean of IR image
#2SEB
#get Tc, Ts
#3SEB
#get Tc, Ts, Tr

lat = #(dec deg)
lon = #(dec deg)
ele = #elevation (m)

zu = #measurement height of wind
zt = #measurement height of temperature

row = #row width
#read from file
#Ta, P, ea
#Tc, Ts, Tr, Trad
#Rn, Su, Sd, Lu, Ld
#time
#canopy height, fraction, width, LAI
#residue fraction

if modeltype=='1SEB':
    #out_df = DataFrame(columns=['datetime (UTC)','sun elevation (deg)','sun azimuth (deg)','LE (W/m2)','H (W/m2)','G (W/m2)','Rn (W/m2)','ET (mm)'])
elif modeltype=='2SEB':
    #out_df = DataFrame(columns=['datetime (UTC)','sun elevation (deg)','sun azimuth (deg)','LEc (W/m2)','LEs (W/m2)','Hc (W/m2)','Hs (W/m2)','H (W/m2)','G (W/m2)','Sns (W/m2)','Snc (W/m2)','Lns (W/m2)','Lnc (W/m2)','E (mm)', 'T (mm)'])
elif modeltype=='3SEB':
    #out_df = DataFrame(columns=['datetime (UTC)','sun elevation (deg)','sun azimuth (deg)','LEc (W/m2)','LEs (W/m2)','LEr (W/m2)','Hc (W/m2)','Hs (W/m2)','Hr (W/m2)','G (W/m2)','Sns (W/m2)','Snr (W/m2)','Snc (W/m2)','Lns (W/m2)','Lnr (W/m2)','Lnc (W/m2)','E (mm)', 'T (mm)'])
else:
    raise(NameError('Invalid model'))

#met_df = pd.read_csv()
#for index, r in df:
    #read in met vars   

    if modeltype=='1SEB':
        #calculate roughness lengths from canopy/bare soil/residue
        #calculate resistances
        #partition radiation
        #calculate fluxes
        #G,H,LE
        row =
    elif modeltype=='2SEB':
        #calculate roughness lengths from canopy/bare soil/residue
        #calculate resistances
        #partition radiation
        #calculate fluxes
        #G,               
        #Hs,Hc,
        #LEc,LEs
        row =
    elif modeltype=='3SEB':
        #calculate roughness lengths from canopy/bare soil/residue
        #calculate resistances
        #partition radiation
        #calculate fluxes
        #G,               
        #Hs,Hc,Hr
        #LEc,LEs,LEr
        row =
    else:
    
    #add row to out_df
    out_df = out_df.append(row)

#write out_df to file 
outname = os.path.join(,'.csv')
out_df.to_csv(outname)
