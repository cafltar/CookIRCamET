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

def roughness_lengths(h,fveg):
    if fveg>0:
        #Calculate d, zom, zoh
        d = dhc * h
        zom = zomhc * h
        zoh = zohzom * zom
    else:
        d = dhs * h
        zom = zomhs * h
        zoh = zohzom * zom
        
    return d, zom, zoh
    

def rahmost(h , LAI , uz, Ta , Tr , nmax , tol, fveg): 
    #def to compute aerodynamic resistance (s/m) using Monin-Obukov (1954)
    #Similarity Theory (MOST), where correction coefficients for unstable and stable conditions
    #are given by Paulson (1970) and Webb (1970)

    #zu = height of wind speed measurement (m)
    #h = canopy height (m)/soil roughness
    #fveg = veg fraction
    #uz = Wind speed measured at height z (m/s)
    #Ta = Air temperature (C)
    #Tr = Radiometric surface temperature (C)
    #nmax = Maximum number of interations
    #t = Tolerance of |rahMOST(n) - rahMOST(n+1)| where n is the nth interation

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

    d, zom, zoh = roughnesslengths(h,fveg)
    
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

def rx(hc, uz , s , LAI , row , wc, fveg): 
    #def rx to compute resistance of heat transport between canopy and canopy
    #displacement height; taken from Norman et al. (1995) Appendix A

    #uz = Wind speed measured over crop or reference at height zu (m/s)
    #hc = canopy height (m)
    #dhc = d/hc, where d is zero plane displacement (m)
    #zomhc = zom/hc, where zom is roughness length for momentum transfer (m)
    #zohzom = zoh/zom, where zoh is the scalar roughness length for heat diffusion
    #z = Height above ground of wind speed measurment (m)
    #s = Effective leaf diameter (=4*Leaf area / leaf perimeter) (m)
    #LAI = Leaf area index (m2 m-2)
    #row = Crop row spacing (m)
    #wc = Vegetation row width (m)
    #d      #Zero plane displacement (m)
    #zom    #Roughness length for momentum transfer (m)
    #zoh    #Roughness length for heat diffusion (m)
    #Uz     #Wind speed over crop surface (m s-1)

    d, zom, zoh = roughnesslengths(h,fveg)

    #Convert field LAI to local LAI
    #LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc

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
    
def rsh(h , zu , s , LAI , row , wc, Ts , Tc, fveg ) 
    #def rsh to compute resistance of heat transport between soil and canopy displacement height
    #Taken from Norman et al. (1995) Appendix B AND Kustas and Norman (1999)

    #uref = Wind speed measured over reference surface at height z (m/s)
    #hcref = Height of reference surface where wind speed was measured(m)
    #hc = Canopy height (m)
    #dhc = d/hc, where d is zero plane displacement (m)
    #zomhc = zom/hc, where zom is roughness length for momentum transfer (m)

    #zohzom = zoh/zom, where zoh is the scalar roughness length for heat diffusion
    #z = Height above ground of wind speed measurment (m)
    #s = Effective leaf diameter (=4*Leaf area / leaf perimeter) (m)
    #LAI = Leaf area index (m2 m-2)
    #row = Crop row spacing (m)
    #wc = Vegetation row width (m)

    #Ts = Soil temperature (C)
    #Tc = Canopy temperature (C)

    #d      #Zero plane displacement (m)
    #zom    #Roughness length for momentum transfer (m)
    #zoh    #Roughness length for heat diffusion (m)
    #Uz     #Wind speed over crop surface (m s-1)

    #Calculate d, zom, zoh
    d, zom, zoh = roughnesslengths(h,fveg)

    #Convert field LAI to local LAI
    #LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc

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
