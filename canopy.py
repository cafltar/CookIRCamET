import sys
import numpy as np
import cv2
import time

from constants import *
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 

def soil_albedo():
    return

class canopy:
    def __init__(self,inputs_obj,thetas,psis,rhosoil):
        self.io = inputs_obj
        self.thetas = thetas#from running solar class methods
        self.psis = psis
        self.soilalbnir = soil_albedo()
        self.soilalbvis = soil_albedo()#from running soil albedo function

    def kb(self,theta):
        #canopy extinction coefficient - 
        #using procedure of Campbell and Norman (1998), Chapter 15 (CN98)
        return (np.sqrt(self.io.XE ** 2 + (np.tan(theta)) ** 2)) / (self.io.XE + 1.774 * (self.io.XE + 1.182) ** -0.733)
    def fcsolar_calc(self):
        self.fcsolar = 
    def mrf_calc(self):
        self.mrf_s =
        self.mrf_r =
    def plf_calc(self):
        self.plf_s =
        self.plf_r =
    def fdhc_calc(self):
        #the downward hemispherical view factor of the canopy is that which is sunlit+shaded
        self.fdhc =
    def taudir(self): 
        #def taudir to compute transmittance of DIRECT beam radiation through the canopy - C&N 98
        self.taudir_vis =
        self.taudir_nir =
    def taudiff(self): 
        #def taudiff to compute transmittance of DIFFUSE radiation through the canopy
        #by integrating taudir over all solar zenith and azimuth angles
        self.taudiff_vis =
        self.taudiff_nir =
    def rhocsdir(self): 
        #def rhocsdiff to compute reflectance of DIRECT radiation through the canopy - C&N 98
        self.rhocsdir_vis =
        self.rhocsdir_nir =
    def rhocsdiff(self):
        #def rhocsdiff to compute reflectance of DIFFUSE radiation through the canopy
        #by integrating all solar zenith and azimuth angles
        self.rhocsdiff_vis =
        self.rhocsdiff_nir =
