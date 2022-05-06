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

def ea(P , Ta , HP , eaOPT):
    #def ea to compute actual vapor pressure of the air (kPa)
    #P = Barometric pressure (kPa)
    #Ta = Air temperature, usually around 2 m height (C)
    #HP = Humidity parameter, depends on eaOPT
    #eaOPT = Option to specify which humidity paramater is used
    #           to compute actual vapor pressure of the air (ea, kPa)
    #           eaOPT = 1: RH (%) is used
    #           eaOPT = 2: Twet (%) is used
    #           eaOPT = 3: Tdew (%) is used
    
    #Variable definitions internal to this function
    #es     #Saturated vapor pressure of the air (kPa)
    #RH     #Relative humidity (%)
    #Twet   #Wet bulb temperature (C)
    #Tdew   #Dew point temperature (C)
    #apsy   #Psychrometer coefficient
    # gammapsy   #Psychrometer constant

    if eaOPT = 1:
        RH = HP
        es = esa(Ta)
        ea = es * RH / 100
    else:
        if eaOPT = 2:
            Tw = HP
            apsy = 0.000662 #For aspirated psychrometers, FAO 56 p. 38
            gammapsy = P * apsy
            es = esa(Tw)
            ea = es - gammapsy * (Ta - Tw)
        else:
            Tdew = HP
            ea = esa(Tdew)
    return ea

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

#PAUL D COLAIZZI
def u_from_uref(hc,uz):
    #Calculate d, zom, zoh
    d = dhc * hc
    zom = zomhc * hc
    zoh = sc * zom

    if uz < 1: uz = 1

    #Adjust wind speed measured over reference surface to wind speed over crop surface with
    #hc difference from hcref
    if !np.isnan(hcref):
        u = uz * (np.log((10 - 0.67 * hcref) / (0.123 * hcref))) * (np.log((zu - d) / zom)) / (np.log((zu - 0.67 * hcref) / (0.123 * hcref))) / (np.log((10 - d) / zom))
    else:
        u = uz

    return u, d, zom, zoh
    

def rahmost(zu , hc , LAI , uz , hcref, Ta , Tr , nmax , tol): 
    #def to compute aerodynamic resistance (s/m) using Monin-Obukov (1954)
    #Similarity Theory (MOST), where correction coefficients for unstable and stable conditions
    #are given by Paulson (1970) and Webb (1970)

    #zu = height of wind speed measurement (m)
    #hc = canopy height (m)
    #dhc = d/hc, where d is zero plane displacement (m)
    #zomhc = zom/hc, where zom is roughness length for momentum transfer (m)
    #sc = zoh/zom, where zoh is the scalar roughness length for heat diffusion
    #LAI = Leaf area index (used in Perria aerodynamic model, m2 m-2)
    #uz = Wind speed measured at height z (m/s)
    #hcref = Height of reference surface where wind speed was measured(m) if not used then np.nan
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

    u, d, zom, zoh = u_from_uref(hc,uz)
    
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

def rx(zu , hc , dhc , zomhc , sc , uz , s , LAI , row , wc , hcref ) 
    #def rx to compute resistance of heat transport between canopy and canopy
    #displacement height; taken from Norman et al. (1995) Appendix A

    #uz = Wind speed measured over crop or reference at height zu (m/s)
    #hc = canopy height (m)
    #dhc = d/hc, where d is zero plane displacement (m)
    #zomhc = zom/hc, where zom is roughness length for momentum transfer (m)
    #sc = zoh/zom, where zoh is the scalar roughness length for heat diffusion
    #z = Height above ground of wind speed measurment (m)
    #s = Effective leaf diameter (=4*Leaf area / leaf perimeter) (m)
    #LAI = Leaf area index (m2 m-2)
    #row = Crop row spacing (m)
    #wc = Vegetation row width (m)
    #hcref = height of reference surface (m)

    #d      #Zero plane displacement (m)
    #zom    #Roughness length for momentum transfer (m)
    #zoh    #Roughness length for heat diffusion (m)
    #Uz     #Wind speed over crop surface (m s-1)

    u, d, zom, zoh = u_from_uref(hc,uz)

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
    
def rsh(uz , hcref , hc , zu , s , LAI , row , wc, Ts , Tc ) 
    #def rsh to compute resistance of heat transport between soil and canopy displacement height
    #Taken from Norman et al. (1995) Appendix B AND Kustas and Norman (1999)

    #uref = Wind speed measured over reference surface at height z (m/s)
    #hcref = Height of reference surface where wind speed was measured(m)
    #hc = Canopy height (m)
    #dhc = d/hc, where d is zero plane displacement (m)
    #zomhc = zom/hc, where zom is roughness length for momentum transfer (m)

    #sc = zoh/zom, where zoh is the scalar roughness length for heat diffusion
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
    u, d, zom, zoh = u_from_uref(hc,uz)

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

def rahRi(z , hc , dhc , zomhc , sc , nuRi , rhoa , uref , hcref, Ta , Tc ):
    #def to calculate aerodynamic resistance (s/m) using Richardson number for stability
    #correction (Kimball et al., 2015, Agron J 107(1): 129-141 and references therein).
    
    #z = height of wind speed measurement (m)
    #hc = canopy height (m)
    #dhc = d/hc, where d is zero plane displacement (m)
    #zomhc = zom/hc, where zom is roughness length for momentum transfer (m)
    #sc = zoh/zom, where zoh is the scalar roughness length for heat diffusion
    #nuRi = empirical constant used in ASHRAE equation for low (u < 1) wind speeds
    #rhoa = density of moist air (kg m-3)
    #uref = Wind speed measured over reference surface at height z (m/s)
    #hcref = Height of reference surface where wind speed was measured(m)
    #Ta = Air temperature (C)
    #Tc = Radiometric canopy temperature (C)
    #i = Maximum number of interations
    #t = Tolerance of |rahRi(n) - rahRi(n+1)| where n is the nth interation

    #Variables internal to this function:
    #d      #Zero plane displacement (m)
    #zom    #Roughness length for momentum transfer (m)
    #zoh    #Roughness length for heat diffusion (m)
    #u      #Wind speed adjusted over crop ( m s-1)
    #K      #Used to calculate PsiRi
    #Ri     #Richardson number
    #psiRi  #Used to determine stability condition

    #Calculate d, zom, zoh
    u, d, zom, zoh = u_from_uref(hc,uz)

    #Used ASHRAE (1972, p. 40) equation to calculate rah under low wind speed,
    #cited in Kimball et al., 2015, Agron J 107(1): 129-141
    #if u < 1:
    #    #dTca 
    #    dTca = np.abs(Tc - Ta)
    #    if dTca < 0.1: dTca = 0.1
    #    rahRi = rhoa * 1013 / (nuRi * (dTca ** (1 / 3)))
    if u < 1: u = 1

    K = 75 * (vonk ** 2) * (((zu - d + zom) / zom) ** (1 / 2)) / ((np.log((zu - d + zom) / zom)) ** 2)
    Ri = g * (Ta - Tc) * (zu - d) / ((Ta + Tk) * (u ** 2))
    #rahRi = Ri

    #Determine stability after Mahrt and Ek (1984)
    if Tc < Ta:     #Stable condition, Ri is positive
        psiRi = (1 + 15 * Ri) * ((1 + 5 * Ri) ** (1 / 2))
    else:    #Unstable condition, Ri is negative
        psiRi = 1 / (1 - 15 * Ri / (1 + K * ((-Ri) ** (1 / 2))))


    rahRi = psiRi * (1 / u) * ((1 / vonk) * np.log((zu - d + zom) / zom)) ** 2
    return rahRi
