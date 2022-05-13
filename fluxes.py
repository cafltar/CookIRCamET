import sys
import numpy as np
import cv2
import time
from constants import *
import canopy, aero
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 

def G(R):

    #R (soil or residue) 24 hour time series
    #G (soil or residue) 24 hour time series
    #from Colaizzi 2016    
    Rmax = np.max(R)
    Rmin = np.min(R)
    return (R-Rmin)/(Rmax-Rmin)*(aG*Rmax+Rmin)-Rmin
    #if R < 0:
    #    return R * GRnday
    #else:
    #    return R * GRnnight
    
def LE(H,Rn,G):
    return Rn-G-H
def LEc(Hc,Rnc):
    return Rnc-Hc
def LEs(Rns,Hs,Gs):
    return Rns-Hs-Gs
def LEs(Rnr,Hr,Gr):
    return Rnr-Hr-Gr

def H(Tac,Ta,ra):
    return aero.rho_a(Tac)*(Tac-Ta)/ra*c_p
def Hc(Tac,Tc,rx):
    return aero.rho_a(Tac)*(Tc-Tac)/rx*c_p
def Hs(Tac,Ts,ra,fs):
    return aero.rho_a(Tac)*(Ts-Tac)/rs*c_p
def Hr(Tac,Tr,ra,fs):
    return aero.rho_a(Tac)*(Tr-Tac)/rs*c_p

def Lsky(P , Ta , RH): 
    #def Lsky to calculate hemispherical downwelling longwave irradiance from the sky (W m-2).
    #P = Barometric pressure (kPa)
    #Ta = Air temperature, usually around 2 m height (C)
    #RH = RH

    #Variable definitions internal to this function
    #ea1     #Actual vapor pressure of the air (kPa)
    # emisatm    #Atmospheric emissivity
    # boltz     #Stephan-Boltzmann constant (W m-2 K-4)

    ea1 = aero.ea(P, Ta, RH)

    emisatm = 0.7 + (0.000595) * ea1 * np.exp(1500 / (Ta + 273.16))
    #Idso et al (1981) WRR 17(2): 295-304, eq. 17 (hemispherical view factor, full longwave spectrum)

    #emisatm = 1.24 * ((10 * ea1 / (Ta + 273.1)) ** (1 / 7))
    #Brutsaert (1975) WRR 11: 742-744.

    return emisatm * boltz * ((Ta + Tk) ** 4)   

def Lns(P , Ta , RH , Ts , Tc , LAI , wc , row , fdhc, fres, fsoil): 
    #def Lnsoil to compute the net longwave radiation to the SOIL using the
    #fdhc VIEW FACTOR model (W m-2).

    #P = Barometric pressure (kPa)
    #Ta = Air temperature, usually around 2 m height (C)
    #RH
    #Ts = Directional brightness temperature of the soil (C)
    #Tc = Directional brightness temperature of the canopy (C)
    #LAI = Leaf area index (m2 m-2)
    #wc = Canopy width (m)
    #row = Row spacing (m)
    #emisSoil = Longwave emittance of the soil
    #emisveg = Longwave emittance of the canopy
    #kappair = Longwave extinction coefficient of the canopy
    #fdhc = downward hemispherical canopy view factor

    #Variable definitions internal to this function
    # ea1     #Actual vapor pressure of the air (kPa)
    # emisatm    #Atmospheric emissivity
    # boltz     #Stephan-Boltzmann constant (W m-2 K-4)
    # Lsky   #Incoming longwave radiation from sky (W m-2)
    # Lc     #Longwave radiation from canopy (W m-2)
    # Ls     #Longwave radiation from soil (W m-2)
    # lwexp  #np.exponential extinction of longwave radiation

    #Convert field LAI to local LAI
    # LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc
    Lc = emisveg * boltz * ((Tc + Tk) ** 4)
    Ls = emisSoil * boltz * ((Ts + Tk) ** 4)
    lwexp = np.exp(-kappIr * LAIL)
    
    return emisSoil * Lsky(P , Ta , RH ) * (1 - fdhc + fdhc * lwexp) + emisSoil * Lc * fdhc * (1 - lwexp) - Ls

    #LnsoilV = emisSoil * Lsky * (1 - wc / row + wc / row * lwexp) + 
    #emisSoil * Lc * wc / row * (1 - lwexp) - Ls

def Lnc(P , Ta , RH , Ts , Tc , LAI , wc , row  , fdhc ) :
    #def Lncanopy to compute the net longwave radiation to the CANOPY using the
    #VERTICAL VIEW FACTOR radiation model (W m-2).

    #P = Barometric pressure (kPa)
    #Ta = Air temperature, usually around 2 m height (C)
    #RH = Relative Humidity %
    #Ts = Directional brightness temperature of the soil (C)
    #Tc = Directional brightness temperature of the canopy (C)
    #LAI = Leaf area index (m2 m-2)
    #wc = Canopy width (m)
    #row = Row spacing (m)
    #emisSoil = Longwave emittance of the soil
    #emisveg = Longwave emittance of the canopy
    #kappIr = Longwave extinction coefficient of the canopy
    #fdhc = downward hemispherical canopy view factor

    #Variable definitions internal to this function
    # ea1     #Actual vapor pressure of the air (kPa)
    # emisatm    #Atmospheric emissivity
    # boltz     #Stephan-Boltzmann constant (W m-2 K-4)
    # Lsky   #Incoming longwave radiation from sky (W m-2)
    # Lc     #Longwave radiation from canopy (W m-2)
    # Ls     #Longwave radiation from soil (W m-2)
    # lwexp  #np.exponential extinction of longwave radiation

    #Convert field LAI to local LAI
    # LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
    LAIL = LAI * row / wc
    Lc = emisveg * boltz * ((Tc + Tk) ** 4)
    Ls = emisSoil * boltz * ((Ts + Tk) ** 4)
    lwexp = np.exp(-kappIr * LAIL)

    return (emisveg * Lsky(P , Ta , RH ) + emisveg * Ls - (1 + emisSoil) * Lc) * fdhc * (1 - lwexp)


def Sns(Rs , KbVis , KbNir , fsc , fdhc , thetas , psis , hc , wc , row , Pr , Vr , LAI): 

    #def Sns to calculate net shortwave radiation to the soil using
    #Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model

    #Rs = Global shortwave irradiance (W m-2)
    #KbVis = Fraction of direct beam shortwave irradiance in the visible spectra (no units)
    #KbNir = Fraction of direct beam shortwave irradiance in the near infrared spectra (no units)
    #fsc = Direct beam solar - canopy planar view factor (no units)
    #fdhc = Downward hemispherical view factor of canopy of a row crop
    #       (e.g., canopy viewed by an inverted radiometer)

    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #row = crop row spacing (m)

    #Pr = Horizontal, perpendicular distance from radiometer to row center (m)
    #Vr = Vertical distance of radiometer from soil surface (m)
    #LAI = Leaf area index, field (m2 m-2)
    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)

    #         taudirVis      #Shortwave direct beam canopy transmittance in the
    #visible spectra (no units)
    #         taudiffVis      #Shortwave diffuse canopy transmittance in the
    #visible spectra (no units)
    #         taudirNir      #Shortwave direct beam canopy transmittance in the
    #near infrared spectra (no units)
    #         taudiffNir      #Shortwave diffuse canopy transmittance in the
    #near infrared spectra (no units)

    #         TVis   #Transmitted visible irradiance (W m-2)
    # TNir   #Transmitted near infrared irradiance (W m-2)

    if Rs = 0: return 0

    wc = min(0.99*row,wc)

    taudirVis = canopy.taudir(thetas, psis, hc, wc, row, LAI)
    taudiffVis = canopy.taudiff(hc, wc, row, LAI)
    taudirNir = canopy.taudir(thetas, psis, hc, wc, row, LAI)
    taudiffNir = canopy.taudiff(hc, wc, row, LAI)

    TVis = Rs * PISI * ((fsc * taudirVis + 1 - fsc) * KbVis + (fdhc * taudiffVis + 1 - fdhc) * (1 - KbVis))

    TNir = Rs * (1 - PISI) * ((fsc * taudirNir + 1 - fsc) * KbNir + (fdhc * taudiffNir + 1 - fdhc) * (1 - KbNir))

    return TVis * (1 - rhosVis) + TNir * (1 - rhosNir)

def Snc(Rs , KbVis , KbNir , fsc , fdhc , thetas , psis , hc , wc , row , Pr , Vr , LAI): 

    #def Snc to calculate net shortwave radiation to the canopy using
    #Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model
    #Rs = Global shortwave irradiance (W m-2)
    #KbVis = Fraction of direct beam shortwave irradiance in the visible spectra (no units)
    #KbNir = Fraction of direct beam shortwave irradiance in the near infrared spectra (no units)
    #fsc = Direct beam solar - canopy planar view factor (no units)
    #fdhc = Downward hemispherical view factor of canopy of a row crop
    #       (e.g., canopy viewed by an inverted radiometer)

    #Thetas = Solar zenith angle (rad)
    #Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
    #and 90 degrees for perpendicular orientation (rad)
    #hc = Canopy height (m)
    #wc = canopy width (m)
    #row = crop row spacing (m)

    #Pr = Horizontal, perpendicular distance from radiometer to row center (m)
    #Vr = Vertical distance of radiometer from soil surface (m)
    #LAI = Leaf area index, field (m2 m-2)
    #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
    #ZetaVis = Leaf shortwave absorption in the visible (PAR) spectra (no units)

    #ZetaNir = Leaf shortwave absorption in the near infrared spectra (no units)
    #rhosVis = Soil reflectance in the visible (PAR) spectra (no units)
    #rhosNir = Soil reflectance in the near infrared spectra (no units)
    #PISI = Fraction of visible shortwave irriadiance in global irradiance;
    #~0.457 at Bushland, which agrees with Meek et al. (1984) for other Western US locations
    
    #taudirVis      #Shortwave direct beam canopy transmittance in the
    #visible spectra (no units)
    #taudiffVis      #Shortwave diffuse canopy transmittance in the
    #visible spectra (no units)
    #taudirNir      #Shortwave direct beam canopy transmittance in the
    #near infrared spectra (no units)
    #taudiffNir      #Shortwave diffuse canopy transmittance in the
    #near infrared spectra (no units)

    #alphadirVis      #Shortwave direct beam canopy reflectance in the
    #visible spectra (no units)
    #             alphadiffVis      #Shortwave diffuse canopy reflectance in the
    #visible spectra (no units)
    #             alphadirNir      #Shortwave direct beam canopy reflectance in the
    #near infrared spectra (no units)
    #alphadiffNir      #Shortwave diffuse canopy reflectance in the
    #near infrared spectra (no units)

    #SncdirVis  #Net shortwave direct beam radiation to the canopy in the
    #visible spectra (W m-2)
    #SncdiffVis  #Net shortwave diffuse radiation to the canopy in the
    #visible spectra (W m-2)
    #SncdirNir  #Net shortwave direct beam radiation to the canopy in the
    #near infrared spectra (W m-2)
    #SncdiffNir  #Net shortwave diffuse radiation to the canopy in the
    #near infrared spectra (W m-2)

    if Rs = 0: return  0
    wc = min(wc,0.99 * row)

    taudirVis = canopy.taudir(thetas, psis, hc, wc, row, LAI)
    taudiffVis = canopy.taudiff(hc, wc, row, LAI)
    taudirNir = canopy.taudir(thetas, psis, hc, wc, row, LAI)
    taudiffNir = canopy.taudiff(hc, wc, row, LAI)

    alphadirVis = canopy.rhocsdir(thetas, psis, hc, wc, row, LAI)
    alphadiffVis = canopy.rhocsdiff(hc, wc, row, LAI)
    alphadirNir = canopy.rhocsdir(thetas, psis, hc, wc, row, LAI)
    alphadiffNir = canopy.rhocsdiff(hc, wc, row, LAI)

    SncdirVis = Rs * PISI * KbVis * fsc * (1 - taudirVis) * (1 - alphadirVis)
    SncdiffVis = Rs * PISI * (1 - KbVis) * fdhc * (1 - taudiffVis) * (1 - alphadiffVis)
    SncdirNir = Rs * (1 - PISI) * KbNir * fsc * (1 - taudirNir) * (1 - alphadirNir)
    SncdiffNir = Rs * (1 - PISI) * (1 - KbNir) * fdhc * (1 - taudiffNir) * (1 - alphadiffNir)

    return SncdirVis + SncdiffVis + SncdirNir + SncdiffNir


                

       
