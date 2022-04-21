import sys
import numpy as np
import cv2
import time
import utils
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 

def latent(T):#latent heat of vaporization kJ/kg at T deg C
    return 2501 - 0002.361 * T
def slope(T):#kPa/K - slope of vp curve, T in C
    return 4098 * (0.6108 * np.exp((17.27 * T) / (T + 237.3))) / ((T + 237.3) ** 2)
def rho_a(T,e,P):#density of moist air in kg/m3 - T in C, e vapor pressure kPa, P baro pressure kPa
    return P / (0.287 * (T + 273.16)) * (1 - 0.378 * e/ P)
def gamm_psy():#psychrometric constant
    return ((1.013 * 10 ** -6) * P) / (0.622 * latent(T))
    
    
def PMTc(solzen , BP , Ta , HP , eaOPT ,
              z , hc , dhc , zomhc , sc , LAI ,
              uref , hcref , Tr , i , t ,
              Rn , GRnday , GRnnight , rcday , rcnight ) 
#def PMTc to compute baseline (i.e., non water stressed) canopy temperature for a crop
#completely covering the soil using the PENMAN-MONTEITH equation.

#Variables internal to this function:
#G  #Soil heat flux (W m-2)
#rc  #Bulk canopy resistance (s m-1)
#gamma  #Psychromertic constant (kPa C-1)
#delta  #Slope of the saturation vapor pressure - temperature relation (kPa C-1)
#es     #Saturation vapor pressure of air temperature (kPa)
#ea1    #Actual vapor pressure at air temperature (kPa)
#rhoa   #Density of mosit air (kg m-3)
#ra    #Aerodynamic resistance (s m-1)
#gammastar  #=gamma*(1+rc/ra) (kPa C-1)

if solzen < 90:
#if Rn > 0:
G = Rn * GRnday
rc = rcday
Else
G = Rn * GRnnight
rc = rcnight


gamma = 0.000665 * BP
delta = slope(Ta)
es = 0.61078 * Exp((17.269 * Ta) / (237.3 + Ta))
ea1 = ea(BP, Ta, HP, eaOPT)
rhoa = BP / (0.287 * (Ta + 273.16)) * (1 - 0.378 * ea1 / BP)
ra = rahRi(z, hc, dhc, zomhc, sc, 5, rhoa, uref, hcref, Ta, Tr)
gammastar = gamma * (1 + rc / ra)

PMTc = Ta + ra * gammastar * (Rn - G) / (rhoa * 1013 * (delta + gammastar)) -
(es - ea1) / (delta + gammastar)



def Tsinitial(Ta , Tr , fr ,
ea , Tc ) 
#def Tsinitial to compute soil temperature Ts (C) in first iteration
#of two source model, based on initial canopy temperature estimated
#from Priestley-Taylor equation, directional brightness temperature
#of the surface, and IRT-vegetation view factor. Tsinitial is constrained by
#the estimated air wet bulb temperature.

#Ta = Air temperature, usually around 2 m height (C)
#Tr = Radiometric directional surface temperature (C)
#fr = Directional radiometer - vegetation view factor
#ea = Actual vapor pressure of the air (kPa)
#Tc = Canopy temperature estimated by Penman-Monteith or Priestley-Taylor equation (C)

#Variable definitions internal to this function
##A 
##Tdew   #Dew point temperature (C)
#Twet   #Wet bulb temperature (C)

Tsinitial = (Abs((Tr + 273.16) ^ 4 - fr * (Tc + 273.16) ^ 4) / (1 - fr)) ^ (1 / 4) - 273.16
#A = Log(ea / 0.61078)
#Tdew = 237.3 * A / (17.269 - A)
#Twet = Ta - 0.5 * (Ta - Tdew) + 5 #Approximation for Twet using "1/2 rule"

#Calculate humidity and psychrometric parameters
#L      #Latent heat of vaporization (MJ/kg)
#P  #Standard barometric pressure for Bushland at 1170 m above MSL (kPa)
#delta  #Slope of the saturation vapor pressure-temperature curve (kPa/C)
#gamma  #Psychrometric constant (kPa/C)
#es     #Saturated vapor pressure of the air (kPa)

L = 2.501 - 0.002361 * Ta
P = 88.21   #Standard barometric pressure for Bushland at 1170 m above MSL (kPa)
delta = 4098 * (0.6108 * Exp((17.27 * Ta) / (Ta + 237.3))) / ((Ta + 237.3) ^ 2) #FAO 56, p.37, eq.13
gamma = ((1.013 * 10 ^ -3) * P) / (0.622 * L)   #FAO 56, p.32, eq.8
es = 0.61078 * Exp((17.269 * Ta) / (237.3 + Ta))
Twet = Ta - ((es - ea) / (gamma + delta))

if Tsinitial < Twet: Tsinitial = Twet
    


def Twet(Ta , ea ) 
#def Twet to calculate wet bulb temperature of the air.

#Ta = Air temperature, usually around 2 m height (C)
#ea = Actual vapor pressure of the air (kPa)

#Compute humidity and psychrometric parameters
#L      #Latent heat of vaporization (MJ/kg)
#P  #Standard barometric pressure for Bushland at 1170 m above MSL (kPa)
#delta  #Slope of the saturation vapor pressure-temperature curve (kPa/C)
#gamma  #Psychrometric constant (kPa/C)
#es     #Saturated vapor pressure of the air (kPa)

L = 2.501 - 0.002361 * Ta
P = 88.21   #Standard barometric pressure for Bushland at 1170 m above MSL (kPa)
delta = 4098 * (0.6108 * Exp((17.27 * Ta) / (Ta + 237.3))) / ((Ta + 237.3) ^ 2) #FAO 56, p.37, eq.13
gamma = ((1.013 * 10 ^ -3) * P) / (0.622 * L)   #FAO 56, p.32, eq.8
es = 0.61078 * Exp((17.269 * Ta) / (237.3 + Ta))
Twet = Ta - ((es - ea) / (gamma + delta))



#PAUL D COLAIZZI
def rahmost(z , hc , dhc , zomhc , sc ,
                 LAI , uref , hcref, Ta , Tr , i , t ) 
#def to compute aerodynamic resistance (s/m) using Monin-Obukov (1954)
#Similarity Theory (MOST), where correction coefficients for unstable and stable conditions
#are given by Paulson (1970) and Webb (1970)

#z = height of wind speed measurement (m)
#hc = canopy height (m)
#dhc = d/hc, where d is zero plane displacement (m)
#zomhc = zom/hc, where zom is roughness length for momentum transfer (m)
#sc = zoh/zom, where zoh is the scalar roughness length for heat diffusion
#LAI = Leaf area index (used in Perria aerodynamic model, m2 m-2)
#uref = Wind speed measured over reference surface at height z (m/s)
#hcref = Height of reference surface where wind speed was measured(m)
#Ta = Air temperature (C)
#Tr = Radiometric surface temperature (C)
#i = Maximum number of interations
#t = Tolerance of |rahMOST(n) - rahMOST(n+1)| where n is the nth interation

#Variables internal to this function:
#d      #Zero plane displacement (m)
#zom    #Roughness length for momentum transfer (m)
#zoh    #Roughness length for heat diffusion (m)
#n     #Iteration number
#uf     #Friction velocity (m s-1)
#PsiM   #Momentum stability correction (dimensionless)
#Psih   #Sensible heat stability correction (dimensionless)
#rah    #Bulk Aerdynamic resistance (s m-1)
#dT     #Air-surface temperature difference (Ta-Ts) (C)
#Toh    #Toh is the aerodynamic surface temperature (C)
#L      #Monin-Obukov length (m)
#X      #Used to compute psih and psim in unstable conditions
#u      #Wind speed adjusted over crop ( m s-1)

#Calculate d, zom, zoh
d = dhc * hc
zom = zomhc * hc
zoh = sc * zom


#Adjust wind speed measured over reference surface to wind speed over crop surface with
#hc difference from hcref

if uref < 1: uref = 1

u = uref * (Log((10 - 0.67 * hcref) / (0.123 * hcref))) * (Log((z - d) / zom)) /
(Log((z - 0.67 * hcref) / (0.123 * hcref))) / (Log((10 - d) / zom))

PsiM = 0
Psih = 0
uf = 0.41 * u / ((Log((z - d) / zom)) - PsiM)
rahmost = ((Log((z - d) / (zoh))) - Psih) / (0.41 * uf)
dT = Ta - Tr
if Abs(dT) < 0.01: dT = 0.01
Toh = Ta - dT
n = 1

10 Do While n < i
rah = rahmost
L = rah * (uf ^ 3) * (Toh + 273.16) / (9.81 * 0.41 * dT)

if L > 0:  #Stable conditions

Psih = -5 * (z - d) / L
PsiM = -5 * (z - d) / L

Else    #Unstable conditions

X = (1 - 16 * (z - d) / L) ^ 0.25
Psih = 2 * Log((1 + (X ^ 2)) / 2)
PsiM = 2 * Log((1 + X) / 2) + Log((1 + (X ^ 2)) / 2) - 2 * Atn(X) + 3.1416 / 2



uf = 0.41 * u / ((Log((z - d) / zom)) - PsiM)
rahmost = ((Log((z - d) / (zoh))) - Psih) / (0.41 * uf)
n = n + 1

if n = i:
uf = 0.41 * u / ((Log((z - d) / zom)))
rahmost = ((Log((z - d) / (zoh)))) / (0.41 * uf)
GoTo 20
Else


if Abs(rah - rahmost) < t:
GoTo 20
Else
GoTo 10


Loop

20 

#def rx to compute resistance of heat transport between canopy and canopy
#displacement height; taken from Norman et al. (1995) Appendix A
def rx(uref , hc , dhc , zomhc , sc ,
            z , s , LAI , row , wc , Cx , hcref ) 

#uref = Wind speed measured over reference surface at height z (m/s)
#hc = canopy height (m)
#dhc = d/hc, where d is zero plane displacement (m)
#zomhc = zom/hc, where zom is roughness length for momentum transfer (m)
#sc = zoh/zom, where zoh is the scalar roughness length for heat diffusion
#z = Height above ground of wind speed measurment (m)
#s = Effective leaf diameter (=4*Leaf area / leaf perimeter) (m)
#LAI = Leaf area index (m2 m-2)
#row = Crop row spacing (m)
#wc = Vegetation row width (m)
#Cx = Empirical constant = 90 s m-1
#hcref = height of reference surface (m)

#d      #Zero plane displacement (m)
#zom    #Roughness length for momentum transfer (m)
#zoh    #Roughness length for heat diffusion (m)
#Uz     #Wind speed over crop surface (m s-1)

#Calculate d, zom, zoh
d = dhc * hc
zom = zomhc * hc
zoh = sc * zom

#Convert field LAI to local LAI
#LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
LAIL = LAI * row / wc

#Adjust wind speed measured over reference surface to wind speed over crop surface with
#hc difference from hcref
Uz = uref * (Log((10 - 0.67 * hcref) / (0.123 * hcref))) * (Log((z - d) / zom)) /
(Log((z - 0.67 * hcref) / (0.123 * hcref))) / (Log((10 - d) / zom))

#Variables internal to this function
#Uc     #Wind speed at top of canopy (m s-1)
#A      #Empirical factor
#Udz    #Wind speed at momentum height d + zom (m s-1)
#PsiM   #Stability correction for momentum at top of canopy,
#                    assumed negligible due to roughness effects in the sublayer

PsiM = 0
Uc = Uz * (Log((hc - d) / zom)) / ((Log((z - d) / zom)) - PsiM)
A = 0.28 * (LAIL ^ (2 / 3)) * (hc ^ (1 / 3)) * (s ^ (-1 / 3))
Udz = Uc * Exp(-A * (1 - ((d + zom) / hc)))
if Udz < 0.1: Udz = 0.1
rx = (Cx / LAIL) * ((s / Udz) ^ 0.5)



#def rsh to compute resistance of heat transport between soil and canopy displacement height
#Taken from Norman et al. (1995) Appendix B AND Kustas and Norman (1999)
def rsh(uref , hcref , hc , dhc , zomhc ,
sc , z , s , LAI , row , wc ,
b , c , Ts , Tc ) 

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
#b = Empirical constant = 0.012 or b = 0.012[1 + cos(psiw)],
#       where psiw is wind direction, ccw from north
#c = Empirical constant = 0.0025

#Ts = Soil temperature (C)
#Tc = Canopy temperature (C)

#d      #Zero plane displacement (m)
#zom    #Roughness length for momentum transfer (m)
#zoh    #Roughness length for heat diffusion (m)
#Uz     #Wind speed over crop surface (m s-1)

#Calculate d, zom, zoh
d = dhc * hc
zom = zomhc * hc
zoh = sc * zom

#Convert field LAI to local LAI
#LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
LAIL = LAI * row / wc

#Adjust wind speed measured over reference surface to wind speed over crop surface with
#hc difference from hcref
Uz = uref * (Log((10 - 0.67 * hcref) / (0.123 * hcref))) * (Log((z - d) / zom)) /
(Log((z - 0.67 * hcref) / (0.123 * hcref))) / (Log((10 - d) / zom))

#Variables internal to this function
#Uc     #Wind speed at top of canopy (m s-1)
#A      #Empirical factor
#Us     #Wind speed just above the soil surface (m s-1)
#PsiM   #Stability correction for momentum at top of canopy,
#                    assumed negligible due to roughness effects in the sublayer

PsiM = 0
Uc = Uz * (Log((hc - d) / zom)) / ((Log((z - d) / zom)) - PsiM)
A = 0.28 * (LAIL ^ (2 / 3)) * (hc ^ (1 / 3)) * (s ^ (-1 / 3))
Us = Uc * Exp(-A * (1 - (0.05 / hc)))
rsh = 1 / (c * ((Abs(Ts - Tc)) ^ (1 / 3)) + b * Us)



def ecini(LEc , gamma , rx , ra ,
               rho , LEs , ea ) 
#def to calculate ecini, the vapor pressure at the canopy surface using the initial
#canopy temperature that was calculated by the inverted ASCE-PM

ecini = ea + (LEs * gamma * ra) / (rho * 1013) +
(1 / ra + 1 / rx) * (LEc * gamma * rx * ra) / (rho * 1013)



def eacini(gamma , rx , ra ,
                rho , LEs , ea , ec ) 
#def to calculate eacini, the vapor pressure in the canopy boundary layer using
#the initial canopy temperature that was calculated by the inverted ASCE-PM

eacini = (ea / ra + ec / rx + LEs * gamma / rho / 1013) / (1 / ra + 1 / rx)



def rahRi(z , hc , dhc , zomhc , sc ,
               nuRi , rhoa , uref , hcref, Ta , Tc ) 
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
d = dhc * hc
zom = zomhc * hc
zoh = sc * zom


#Adjust wind speed measured over reference surface to wind speed over crop surface with
#hc difference from hcref

u = uref * (Log((10 - 0.67 * hcref) / (0.123 * hcref))) * (Log((z - d) / zom)) /
(Log((z - 0.67 * hcref) / (0.123 * hcref))) / (Log((10 - d) / zom))

#Used ASHRAE (1972, p. 40) equation to calculate rah under low wind speed,
#cited in Kimball et al., 2015, Agron J 107(1): 129-141
#if u < 1:
#    #dTca 
#    dTca = Abs(Tc - Ta)
#    if dTca < 0.1: dTca = 0.1
#    rahRi = rhoa * 1013 / (nuRi * (dTca ^ (1 / 3)))
#    GoTo 20
#

if u < 1: u = 1

K = 75 * (0.41 ^ 2) * (((z - d + zom) / zom) ^ (1 / 2)) / ((Log((z - d + zom) / zom)) ^ 2)
Ri = 9.81 * (Ta - Tc) * (z - d) / ((Ta + 273.16) * (u ^ 2))
#rahRi = Ri

#Determine stability after Mahrt and Ek (1984)
if Tc < Ta:     #Stable condition, Ri is positive
    psiRi = (1 + 15 * Ri) * ((1 + 5 * Ri) ^ (1 / 2))
    Else    #Unstable condition, Ri is negative
    psiRi = 1 / (1 - 15 * Ri / (1 + K * ((-Ri) ^ (1 / 2))))


rahRi = psiRi * (1 / u) * ((1 / 0.41) * Log((z - d + zom) / zom)) ^ 2

20 
def Tsimax(solzen , BP , Ta , HP , eaOPT ,
z , hc , dhc , zomhc , sc , LAI ,
uref , hcref , Tr , i , t ,
Rn , GRnday , GRnnight , rcday , rcnight ,
s , row , wc , b , c , Tc ) 

#def Tsimax to calculate the maximum initial soil temperature using the
    #FAO 56 (Rn - G) for available energy.

#Variables internal to this function:
    #G  #Soil heat flux (W m-2)
#rc  #Bulk canopy resistance (s m-1)
    #ea1    #Actual vapor pressure at air temperature (kPa)
#rhoa   #Density of mosit air (kg m-3)
    #ra    #Aerodynamic resistance (s m-1)
#rsh1   #Soil heat flux resistance (s m-1)

    if solzen < 90:
    #if Rn > 0:
G = Rn * GRnday
rc = rcday
Else
G = Rn * GRnnight
rc = rcnight


ea1 = ea(BP, Ta, HP, eaOPT)
rhoa = BP / (0.287 * (Ta + 273.16)) * (1 - 0.378 * ea1 / BP)
ra = rahRi(z, hc, dhc, zomhc, sc, 5, rhoa, uref, hcref, Ta, Tc)
#rsh1 = rsh(uref, hcref, hc, dhc, zomhc, sc, z, s, LAI, row, wc, b, c, Tr, Tc)
    Tsimax = Ta + (ra) * (Rn - G) / (rhoa * 1013)

    

    
