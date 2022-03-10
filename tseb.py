import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
import utils
from scipy import constants
from scipy import special
from pysolar.solar import get_azimuth, get_altitude

from pandas import read_csv, read_excel, DataFrame

import logging
logging.basicConfig(level=logging.INFO)
home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Working')
pi = constants.pi
g = constants.g
boltz = constant.Stefan_Boltzmann_constant# W K-4 m-2
vonk = 0.4 #Von Karman
Tk = 273.16 #0 C
gammac = 0.000665 #psychometric constant
cp = 1005 #J/kg/K
rhoa = 1.205 #kg/m3

#measurements:
#Ta, P, RH
#Tc, Ts
#Rn
#time, lat, lon, z, altitude, aspect/slope?

#classes
class radiation:
    #Radiation model primary from Campbell and Norman, 1998
    #Inputs (if unavailable leave as None):
    #Ld - downwelling longwave (w/m2)
    #Sd - downwelling shortwave (w/m2)
    #Rn - net total (w/m2)
    #x - lad parameter
    #theta_s, phi_s solar elevation and azimuth (degrees)

    #Outputs:
    #Lc - longwave canopy (w/m2)
    #Sc - shortwave canopy (w/m2)
    #Ls - longwave soil (w/m2)
    #Ss - shortwave soil (w/m2)
    
    def __init__(self,Ld,Sd,Rn,theta_s,phi_s,x):
        self.Ld = Ld
        self.Sd = Sd
        self.Rn = Rn
        self.theta_s = np.pi/2-np.deg2rad(theta_s)#elevation->zenith, deg->rad
        self.phi_s = np.deg2rad(phi_s)
        self.x = x
        self.n = len(phi_s)#number of measurements
    def Kbe(self):
        return np.sqrt(self.x**2+np.tan(self.theta_s)**2)/(self.x+1.774*(self.x+1.182)**(-.733))
    def taudl(self):
        return None
    def dif_dir_frac(self):
        return None
    def rho_b(self):
        return None

class resistances:
    def __init__(self):
    def r_a(self):
        return None
    def r_c(self):
        return None
    def r_x(self):
        return None
    def r_s(self):
        return None
    
class meteo:
    def __init__():

class wind:
    def __init__():
