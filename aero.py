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

def P_from_z(z):
    #z elevation msl
    return 101.3*((293-0.0065*z)/293)**5.26#kPa ASCE 70 eqn 3

def ea(P , Ta , RH):
    #def ea to compute actual vapor pressure of the air (kPa)
    #P = Barometric pressure (kPa)
    #Ta = Air temperature, usually around 2 m height (C)
    #RH = Relative humidity
    #Variable definitions internal to this function
    #es     #Saturated vapor pressure of the air (kPa)
    #RH     #Relative humidity (%)
    #Twet   #Wet bulb temperature (C)
    #Tdew   #Dew point temperature (C)
    #apsy   #Psychrometer coefficient
    # gammapsy   #Psychrometer constant

    es = esa(Ta)
    return es * RH / 100

def esa(T):#saturation vapor pressure kPa, T is temp in deg C
    return  0.61078 * np.exp((17.269 * T) / (237.3 + T))

def latent(T):#latent heat of vaporization kJ/kg at T deg C
    return 2501 - 0002.361 * T

def slope(T):#kPa/C - slope of vp curve, T in C
    return 4098 * (0.6108 * np.exp((17.269 * T) / (T + 237.3))) / ((T + 237.3) ** 2)

def rho_a(T,e,P):#density of moist air in kg/m3 - T in C, e vapor pressure kPa, P baro pressure kPa
    return P / (0.287 * (T + 273.16)) * (1 - 0.378 * e/ P)

def gamm_psy(P,T):#psychrometric constant kPa/C
    return ((1.013 * 10 ** -6) * P) / (0.622 * latent(T)) 

def Twet(Ta , ea, P): 
    #def Twet to calculate wet bulb temperature of the air.

    #Ta = Air temperature, usually around 2 m height (C)
    #ea = Actual vapor pressure of the air (kPa)

    #Compute humidity and psychrometric parameters
    #L      #Latent heat of vaporization (MJ/kg)
    #P  #Standard barometric pressure for Bushland at 1170 m above MSL (kPa)
    #delta  #Slope of the saturation vapor pressure-temperature curve (kPa/C)
    #gamma  #Psychrometric constant (kPa/C)
    #es     #Saturated vapor pressure of the air (kPa)

    L = latent(Ta)
    delta = slope(Ta) #FAO 56, p.37, eq.13
    gamma = gamma_psy(P,L)   #FAO 56, p.32, eq.8
    es = esa(Ta)
    return Ta - ((es - ea) / (gamma + delta))

class resistances:
    def __init__(self, inputs_obj):
        self.io = inputs_obj
        #Variables internal to this function:
        #d      #Zero plane displacement (m)
        #zom    #Roughness length for momentum transfer (m)
        #zoh    #Roughness length for heat diffusion (m)
        #n     #Iteration number
        #uf     #Friction velocity (m s-1)
        #PsiM   #Momentum stability correction (dimensionless)
        #PsiH   #Sensible heat stability correction (dimensionless)
        #rah    #Bulk Aerdynamic resistance (s m-1)
        #dT     #Air-surface temperature difference (Ta-Ts) (C)
        #Toh    #Toh is the aerodynamic surface temperature (C)
        #L      #Monin-Obukov length (m)
        #X      #Used to compute psih and psim in unstable conditions
        #u      #Wind speed adjusted over crop ( m s-1)
        
        self.mask_veg = self.io.fveg>0 and self.io.hr<=self.io.hc
        self.mask_bare = self.io.fveg<=0 and self.io.fres<=0
        self.mask_res = self.io.fveg<=0 and self.io.fres>0
        self.mask_tall_res = self.io.fveg>0 and self.io.hr>self.io.hc

        self.d, self.zom, self.zoh = self.roughness_lengths()
        
    def roughness_lengths(self):
        d = np.zeros(self.Ta.shape)
        zom = np.zeros(self.Ta.shape)
        zoh = np.zeros(self.Ta.shape)
        
        d[self.mask_veg] = dhc * self.hc[self.mask_veg]
        zom[self.mask_veg] = zomhc * self.hc[self.mask_veg]
        zoh[self.mask_veg] = zohzomc[self.mask_veg] * self.zom[self.mask_veg]
        
        d[self.mask_tall_res] = dhr * self.hr[self.mask_tall_res]
        zom[self.mask_tall_res] = zomhr * self.hr[self.mask_tall_res]
        zoh[self.mask_tall_res] = zohzomr * self.zom[self.mask_tall_res]

        d[self.mask_res] = dhr * self.hr[self.mask_res]
        zom[self.mask_res] = zomhr * self.hr[self.mask_res]
        zoh[self.mask_res] = zohzomr * self.zom[self.mask_res]

        d[self.mask_bare] = dhs * self.hs[self.mask_bare]
        zom[self.mask_bare] = zomhs * self.hs[self.mask_bare]
        zoh[self.mask_bare] = zohzoms * self.zom[self.mask_bare]

        return d, zom, zoh

    def rahmost(self): 
        #def to compute aerodynamic resistance (s/m) using Monin-Obukov (1954)
        #Similarity Theory (MOST), where correction coefficients for unstable and stable conditions
        #are given by Paulson (1970) and Webb (1970)

        PsiM = 0
        PsiH = 0
        uf = vonk * u / ((np.log((zu - d) / zom)) - PsiM)
        rahmost = ((np.log((zu - d) / (zoh))) - PsiH) / (vonk * uf)
        dT = Ta - Tr

        if np.abs(dT) < 0.01: dT = 0.01

        Toh = Ta - dT
        n = 1

        while n < nmax and np.abs(rah - rahmost) > tol:
            rah = rahmost
            L = rah * (uf ** 3) * (Toh + Tk) / (g * vonk * dT)

            if L > 0:  #Stable conditions

                PsiH = -5 * (zu - d) / L
                PsiM = -5 * (zu - d) / L

            else:    #Unstable conditions

                X = (1 - 16 * (zu - d) / L) ** 0.25
                PsiH = 2 * np.log((1 + (X ** 2)) / 2)
                PsiM = 2 * np.log((1 + X) / 2) + np.log((1 + (X ** 2)) / 2) - 2 * np.atan(X) + pi / 2

            uf = vonk * u / ((np.log((zu - d) / zom)) - PsiM)
            rahmost = ((np.log((zu - d) / (zoh))) - PsiH) / (vonk * uf)
            n = n + 1

        if n == nmax:
            uf = vonk * u / ((np.log((zu - d) / zom)))
            rahmost = ((np.log((zu - d) / (zoh)))) / (vonk * uf)

    def rx(self): 
        #def rx to compute resistance of heat transport between canopy and canopy
        #displacement height; taken from Norman et al. (1995) Appendix A

        #Variables internal to this function
        #Uc     #Wind speed at top of canopy (m s-1)
        #A      #Empirical factor
        #Udz    #Wind speed at momentum height d + zom (m s-1)
        #PsiM   #Stability correction for momentum at top of canopy,
        #                    assumed negligible due to roughness effects in the sublayer

        PsiM = 0

        uc = u * (np.log((hc - d) / zom)) / ((np.log((zu - d) / zom)) - PsiM)
        A = 0.28 * (LAIL ** (2 / 3)) * (hc ** (1 / 3)) * (s ** (-1 / 3))
        udz = uc * np.exp(-A * (1 - ((d + zom) / hc)))
        if udz < 0.1: udz = 0.1
        rx = (Cx / LAIL) * ((s / udz) ** 0.5)
        return rsh

    def rsh(self): 
        #def rsh to compute resistance of heat transport between soil and canopy displacement height
        #Taken from Norman et al. (1995) Appendix B AND Kustas and Norman (1999)

        #Variables internal to this function
        #Uc     #Wind speed at top of canopy (m s-1)
        #A      #Empirical factor
        #Us     #Wind speed just above the soil surface (m s-1)
        #PsiM   #Stability correction for momentum at top of canopy,
        #                    assumed negligible due to roughness effects in the sublayer

        PsiM = 0
        uc = u * (np.log((hc - d) / zom)) / ((np.log((zu - d) / zom)) - PsiM)
        A = 0.28 * (LAIL ** (2 / 3)) * (hc ** (1 / 3)) * (s ** (-1 / 3))
        us = uc * np.exp(-A * (1 - (0.05 / hc)))
        rsh = 1 / (c * ((np.abs(Ts - Tc)) ** (1 / 3)) + b * us)

        return rsh

    def rrh_stand(self):
        #standing residue thermal resistance per Aiken 2003
        return rrsh
    
    def rrh_flat(self):
        #flat residue thermal resistance per Lagos 2009
        return rrfh
