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
g = constants.g #m/s2
boltz = constant.Stefan_Boltzmann_constant# W/K4/m2
vonk = 0.4 #Von Karman
Tk = 273.16 #0 C
rho_a = 1.205 #kg/m3
C_solar = 1320 #W/m2
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# Psicrometric Constant kPa K-1
gamma_c = 0.0658
# gas constant for dry air, J/(kg*degK)
R_d = 287.04

#measurements:
#Ta, P, ea
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
    
    def __init__(self,Ld,Sd,Rn,theta_s,phi_s,x,lai,fc):
        self.Ld = Ld
        self.Sd = Sd
        self.Rn = Rn
        self.theta_s = np.pi/2-np.deg2rad(theta_s)#elevation->zenith, deg->rad
        self.phi_s = np.deg2rad(phi_s)
        self.x = x
        self.n = len(phi_s)#number of measurements
        self.lai_e = lai#effective lai
        self.lai_l = lai/fc#local lai
    def Kbe(self,angle):
        return np.sqrt(self.x**2+np.tan(angle)**2)/(self.x+1.774*(self.x+1.182)**(-.733))
    def taudl(self):
        #diffuse transmissivity
        taud = 0
        dangle = pi/40
        for angle in range(0, pi/2, dangle):
            akd = self.Kbe(angle)
            taub = np.exp(-akd * self.lai)
            taud += taub * np.cos(angle) * np.sin(angle) * dangle
            return 2.0 * taud
        
    def weiss_frac(self,p):
        #weiss 1985 - solar radiation fractions (shortwave)
        f_nir = .4
        f_vis = .6
        w = C_solar*10**(-1.195+.4459*np.log10(np.cos(self.theta_s))-.0345*np.log10(np.cos(self.theta_s))**2)
        S_vis = C_solar*f_vis
        S_nir = C_solar*f_nir
        S_dir_vis = (S_vis*np.exp(-.185*(p/1313.25)/np.cos(self.theta_s)))*np.cos(self.theta_s)
        S_dir_nir = (S_nir*np.exp(-.06*(p/1313.25)/np.cos(self.theta_s))-w)*np.cos(self.theta_s)
        S_dif_vis = 0.4*(S_vis*np.cos(self.theta_s)-S_dir_vis)
        S_dif_nir = 0.6*(S_nir*np.cos(self.theta_s)-S_dir_nir-w)

        f_nir = (S_dir_vis+S_dif_vis)/(S_dir_vis+S_dif_vis+S_dir_nir+S_dif_nir)
        f_vis = (S_dir_nir+S_dif_nir)/(S_dir_vis+S_dif_vis+S_dir_nir+S_dif_nir)
        
        S_vis = self.Sd*f_vis
        S_nir = self.Sd*f_nir 

        self.f_dir_vis = 
        self.f_dir_nir =
        self.f_dif_vis =
        self.f_dif_nir =
        
        return None
    def dif_dir_frac(self):
        return None
    def rho_b(self):
        return None
    def clump(self):
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
    def __init__(self,Ta,ea,es=None,p=None,z=0):
        self.Ta = Ta#Tair, K, met tower is C
        self.p = p#pressure, mb met, tower is kpa
        self.ea = ea#vapour pressure, mb met, tower is kpa
        self.es = es#sat vapour pressure, mb met, tower is kpa              
        if es is None:
            self.calc_es()#sat vapour pressure, mb met, tower is kpa
        if p is None:
            self.calc_pressure(z)
        self.calc_c_p()
        self.calc_lambda()
        self.calc_psicr()
        self.calc_rho()
        self.calc_rho_w()
        self.calc_delta_vapor_pressure()
        
    def calc_c_p(self):
        #c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1).
        q = epsilon * self.ea / (self.p + (epsilon - 1.0) * self.ea)
        # then the heat capacity of (moist) air
        self.c_p = (1.0 - q) * c_pd + q * c_pv
        
    def calc_lambda(self):
        #Lambda Latent heat of vaporisation (J kg-1).
        self.Lambda = 1e6 * (2.501 - (2.361e-3 * (self.Ta - 273.15))) 

    def calc_pressure(self,z):
        #z height above sea level (m).
        self.p = 1013.25 * (1.0 - 2.225577e-5 * z)**5.25588

    def calc_psicr(self):
        #psicr Psychrometric constant (mb C-1).

        self.psicr = self.c_p * self.p / (epsilon * self.Lambda)  

    def calc_rho(self):
        #density of air (kg m-3).
        # p is multiplied by 100 to convert from mb to Pascals
        self.rho = ((self.p * 100.0) / (R_d * self.Ta)) * (1.0 - (1.0 - epsilon) * self.ea / self.p)
        
    def calc_rho_w(self):
       #density of air-free water ata pressure of 101.325kPa
        #density of water (kg m-3)
    
        t = self.Ta - 273.15  # Temperature in Celsius
        self.rho_w = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
                 - 46.170461e-6 * t**3 + 105.56302e-9 * t**4
                 - 280.54253e-12 * t**5) / (1 + 16.897850e-3 * t)

    def calc_es(self):
        #es : saturation water vapour pressure (mb).
        T_C = self.Ta - 273.15
        self.es = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    
    def calc_delta_vapor_pressure(self):
        #slope of saturation water vapour pressure.
        #s slope of the saturation water vapour pressure (kPa K-1)

        T_C = self.Ta - 273.15
        self.s = (2503 * np.exp(17.27 * T_C / (T_C + 237.3))) / ((T_C + 237.3)**2)

    def calc_mixing_ratio(self):
        #ea  water vapor pressure at reference height (mb).
        #p   total air pressure (dry air + water vapour) at reference height (mb).
        #r 
        r = epsilon * ea / (p - ea)
        return r


    def calc_lapse_rate_moist(self):
        #moist-adiabatic lapse rate (K/m)
        r = self.calc_mixing_ratio()
        Gamma_w = ((g * (R_d * self.Ta**2 + self.Lambda * r * self.Ta)
                    / (self.c_p * R_d * self.Ta**2 + self.Lambda**2 * r * epsilon)))
        return Gamma_w
    
class wind:
    def __init__(self):
    def u_star(self):
        return None
    def l_mo(self):
        return None
    def u_s(self):
        return None
    def u_c(self):
        return None
    def psi_h(self):
        return None
    def psi_h(self):
        return None

#For initial estimates of ET.
def penman_monteith():
    return LEi
def priestley_taylor():
    return LEi
