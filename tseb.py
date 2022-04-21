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

#measurements:

lat = #(dec deg)
lon = #(dec deg)
ele = #elevation (m)
Pz =101.3*(((293-0.0065*ele)/293)**5.26)#kPa
gamma = Pz*.000665#kPa/C
zu = #measurement height of wind
row = #row width
#read from file
#Ta, P, ea
#Tc, Ts
#Rn, Su, Sd, Lu, Ld
#time
#canopy height, fraction, width, LAI


#out_df = DataFrame(columns=['datetime (UTC)','sun elevation (deg)','sun azimuth (deg)','LEc (W/m2)','LEs (W/m2)','Hc (W/m2)','Hs (W/m2)','H (W/m2)','G (W/m2)','Sns (W/m2)','Snc (W/m2)','Lns (W/m2)','Lnc (W/m2)','E (mm)', 'T (mm)'])

#in_df = pd.read_csv()
#for index, r in df:
    #read in met vars

    #calculate roughness lengths from canopy/bare soil

    #calculate resistances

    #partition radiation

    #calculate fluxes
    #G

    #Hs,Hc,H

    #LEc,LEs,LE

    #2Temp

    #1Temp
    
    #PM
    
    #add row to out_df
    row = 
    out_df = out_df.append(row)

#write out_df to file 

outname = os.path.join(,'.csv')
out_df.to_csv(outname)
