import sys
import numpy as np
import cv2
import time
import utils, canopy
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 

def G():

def LEc():

def LEs():

def H():

def Hc():

def Hs():


def Lsky(BP , Ta , HP , eaOPT ) 
#def Lsky to calculate hemispherical downwelling longwave irradiance from the sky (W m-2).

#BP = Barometric pressure (kPa)
#Ta = Air temperature, usually around 2 m height (C)
#HP = Humidity parameter, depends on eaOPT
#eaOPT = Option to specify which humidity paramater is used
#           to compute actual vapor pressure of the air (ea, kPa)
#           eaOPT = 1: RH (%) is used
#           eaOPT = 2: Twet (%) is used
#           eaOPT = 3: Tdew (%) is used

#Variable definitions internal to this function
 ea1     #Actual vapor pressure of the air (kPa)
 emisatm    #Atmospheric emissivity
 SB     #Stephan-Boltzmann constant (W m-2 K-4)

ea1 = ea(BP, Ta, HP, eaOPT)

emisatm = 0.7 + (0.000595) * ea1 * Exp(1500 / (Ta + 273.16))
#Idso et al (1981) WRR 17(2): 295-304, eq. 17 (hemispherical view factor, full longwave spectrum)

#emisatm = 1.24 * ((10 * ea1 / (Ta + 273.1)) ^ (1 / 7))
#Brutsaert (1975) WRR 11: 742-744.

SB = 0.0000000567

Lsky = emisatm * SB * ((Ta + 273.16) ^ 4)



def Lnsoil(BP , Ta , HP , Ts , 
                Tc , LAI , wc , row , eaOPT , 
                emissoil , emisveg , LWext , fdhc ) 
#def Lnsoil to compute the net longwave radiation to the SOIL using the
#fdhc VIEW FACTOR model (W m-2).

#BP = Barometric pressure (kPa)
#Ta = Air temperature, usually around 2 m height (C)
#HP = Humidity parameter, depends on eaOPT
#Ts = Directional brightness temperature of the soil (C)
#Tc = Directional brightness temperature of the canopy (C)
#LAI = Leaf area index (m2 m-2)
#wc = Canopy width (m)
#row = Row spacing (m)
#eaOPT = Option to specify which humidity paramater is used
#           to compute actual vapor pressure of the air (ea, kPa)
#           eaOPT = 1: RH (%) is used
#           eaOPT = 2: Twet (%) is used
#           eaOPT = 3: Tdew (%) is used
#emissoil = Longwave emittance of the soil
#emisveg = Longwave emittance of the canopy
#LWext = Longwave extinction coefficient of the canopy
#fdhc = downward hemispherical canopy view factor

#Variable definitions internal to this function
 ea1     #Actual vapor pressure of the air (kPa)
 emisatm    #Atmospheric emissivity
 SB     #Stephan-Boltzmann constant (W m-2 K-4)
 Lsky   #Incoming longwave radiation from sky (W m-2)
 Lc     #Longwave radiation from canopy (W m-2)
 Ls     #Longwave radiation from soil (W m-2)
 LWEXP  #Exponential extinction of longwave radiation

#Convert field LAI to local LAI
 LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
LAIL = LAI * row / wc

ea1 = ea(BP, Ta, HP, eaOPT)
emisatm = 0.7 + (0.000595) * ea1 * Exp(1500 / (Ta + 273.16))
#Idso et al (1981) WRR 17(2): 295-304, eq. 17 (hemispherical view factor, full longwave spectrum)

#emisatm = 1.24 * ((10 * ea1 / (Ta + 273.1)) ^ (1 / 7))
#Brutsaert (1975) WRR 11: 742-744.

SB = 0.0000000567
Lsky = emisatm * SB * ((Ta + 273.16) ^ 4)
Lc = emisveg * SB * ((Tc + 273.16) ^ 4)
Ls = emissoil * SB * ((Ts + 273.16) ^ 4)
LWEXP = Exp(-LWext * LAIL)

Lnsoil = emissoil * Lsky * (1 - fdhc + fdhc * LWEXP) + 
emissoil * Lc * fdhc * (1 - LWEXP) - Ls

#LnsoilV = emissoil * Lsky * (1 - wc / row + wc / row * LWEXP) + 
#emissoil * Lc * wc / row * (1 - LWEXP) - Ls



def Lncanopy(BP , Ta , HP , Ts , 
                  Tc , LAI , wc , row , eaOPT , 
                  emissoil , emisveg , LWext , fdhc ) 
#def Lncanopy to compute the net longwave radiation to the CANOPY using the
#VERTICAL VIEW FACTOR radiation model (W m-2).

#BP = Barometric pressure (kPa)
#Ta = Air temperature, usually around 2 m height (C)
#HP = Humidity parameter, depends on eaOPT
#Ts = Directional brightness temperature of the soil (C)
#Tc = Directional brightness temperature of the canopy (C)
#LAI = Leaf area index (m2 m-2)
#wc = Canopy width (m)
#row = Row spacing (m)
#eaOPT = Option to specify which humidity paramater is used
#           to compute actual vapor pressure of the air (ea, kPa)
#           eaOPT = 1: RH (%) is used
#           eaOPT = 2: Twet (%) is used
#           eaOPT = 3: Tdew (%) is used
#emissoil = Longwave emittance of the soil
#emisveg = Longwave emittance of the canopy
#LWext = Longwave extinction coefficient of the canopy
#fdhc = downward hemispherical canopy view factor

#Variable definitions internal to this function
 ea1     #Actual vapor pressure of the air (kPa)
 emisatm    #Atmospheric emissivity
 SB     #Stephan-Boltzmann constant (W m-2 K-4)
 Lsky   #Incoming longwave radiation from sky (W m-2)
 Lc     #Longwave radiation from canopy (W m-2)
 Ls     #Longwave radiation from soil (W m-2)
 LWEXP  #Exponential extinction of longwave radiation

#Convert field LAI to local LAI
 LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
LAIL = LAI * row / wc

ea1 = ea(BP, Ta, HP, eaOPT)
emisatm = 0.7 + (0.000595) * ea1 * Exp(1500 / (Ta + 273.16))
#Idso et al (1981) WRR 17(2): 295-304, eq. 17 (hemispherical view factor, full longwave spectrum)

#emisatm = 1.24 * ((10 * ea1 / (Ta + 273.1)) ^ (1 / 7))
#Brutsaert (1975) WRR 11: 742-744.

SB = 0.0000000567
Lsky = emisatm * SB * ((Ta + 273.16) ^ 4)
Lc = emisveg * SB * ((Tc + 273.16) ^ 4)
Ls = emissoil * SB * ((Ts + 273.16) ^ 4)
LWEXP = Exp(-LWext * LAIL)

Lncanopy = (emisveg * Lsky + emisveg * Ls - (1 + emissoil) * Lc) * fdhc * (1 - LWEXP)

def Sns(Rs , KbVIS , KbNIR , fsc , fdhc , 
             thetas , psis , hc , wc , row , 
             Pr , Vr , LAI , XE , ZetaVIS , 
             ZetaNIR , rhosVIS , rhosNIR , PISI ) 

#def Sns to calculate net shortwave radiation to the soil using
#Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model

#Rs = Global shortwave irradiance (W m-2)
#KbVIS = Fraction of direct beam shortwave irradiance in the visible spectra (no units)
#KbNIR = Fraction of direct beam shortwave irradiance in the near infrared spectra (no units)
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
        #ZetaVIS = Leaf shortwave absorption in the visible (PAR) spectra (no units)

#ZetaNIR = Leaf shortwave absorption in the near infrared spectra (no units)
        #rhosVIS = Soil reflectance in the visible (PAR) spectra (no units)
#rhosNIR = Soil reflectance in the near infrared spectra (no units)
        #PISI = Fraction of visible shortwave irriadiance in global irradiance;
#~0.457 at Bushland, which agrees with Meek et al. (1984) for other Western US locations

         taudirVIS      #Shortwave direct beam canopy transmittance in the
#visible spectra (no units)
         taudiffVIS      #Shortwave diffuse canopy transmittance in the
#visible spectra (no units)
         taudirNIR      #Shortwave direct beam canopy transmittance in the
#near infrared spectra (no units)
         taudiffNIR      #Shortwave diffuse canopy transmittance in the
#near infrared spectra (no units)

         TVIS   #Transmitted visible irradiance (W m-2)
 TNIR   #Transmitted near infrared irradiance (W m-2)

        If Rs = 0 Then
        Sns = 0
        GoTo 100
        End If

        If wc >= row Then wc = 0.99 * row

        taudirVIS = taudir(thetas, psis, hc, wc, row, LAI, XE, ZetaVIS, rhosVIS)
        taudiffVIS = taudiff(hc, wc, row, LAI, XE, ZetaVIS, rhosVIS)
        taudirNIR = taudir(thetas, psis, hc, wc, row, LAI, XE, ZetaNIR, rhosNIR)
        taudiffNIR = taudiff(hc, wc, row, LAI, XE, ZetaNIR, rhosNIR)

        TVIS = Rs * PISI * ((fsc * taudirVIS + 1 - fsc) * KbVIS + 
                            (fdhc * taudiffVIS + 1 - fdhc) * (1 - KbVIS))

        TNIR = Rs * (1 - PISI) * ((fsc * taudirNIR + 1 - fsc) * KbNIR + 
                                  (fdhc * taudiffNIR + 1 - fdhc) * (1 - KbNIR))

        Sns = TVIS * (1 - rhosVIS) + TNIR * (1 - rhosNIR)

        100 

        def Snc(Rs , KbVIS , KbNIR , fsc , fdhc , 
                     thetas , psis , hc , wc , row , 
                     Pr , Vr , LAI , XE , ZetaVIS , 
                     ZetaNIR , rhosVIS , rhosNIR , PISI ) 

        #def Snc to calculate net shortwave radiation to the canopy using
#Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model

        #Rs = Global shortwave irradiance (W m-2)
#KbVIS = Fraction of direct beam shortwave irradiance in the visible spectra (no units)
        #KbNIR = Fraction of direct beam shortwave irradiance in the near infrared spectra (no units)
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
                #ZetaVIS = Leaf shortwave absorption in the visible (PAR) spectra (no units)

#ZetaNIR = Leaf shortwave absorption in the near infrared spectra (no units)
                #rhosVIS = Soil reflectance in the visible (PAR) spectra (no units)
#rhosNIR = Soil reflectance in the near infrared spectra (no units)
                #PISI = Fraction of visible shortwave irriadiance in global irradiance;
#~0.457 at Bushland, which agrees with Meek et al. (1984) for other Western US locations

                 taudirVIS      #Shortwave direct beam canopy transmittance in the
#visible spectra (no units)
                 taudiffVIS      #Shortwave diffuse canopy transmittance in the
#visible spectra (no units)
                 taudirNIR      #Shortwave direct beam canopy transmittance in the
#near infrared spectra (no units)
                 taudiffNIR      #Shortwave diffuse canopy transmittance in the
#near infrared spectra (no units)

                 alphadirVIS      #Shortwave direct beam canopy reflectance in the
#visible spectra (no units)
                 alphadiffVIS      #Shortwave diffuse canopy reflectance in the
#visible spectra (no units)
                 alphadirNIR      #Shortwave direct beam canopy reflectance in the
#near infrared spectra (no units)
                 alphadiffNIR      #Shortwave diffuse canopy reflectance in the
#near infrared spectra (no units)

                 SncdirVIS  #Net shortwave direct beam radiation to the canopy in the
#visible spectra (W m-2)
                 SncdiffVIS  #Net shortwave diffuse radiation to the canopy in the
#visible spectra (W m-2)
                 SncdirNIR  #Net shortwave direct beam radiation to the canopy in the
#near infrared spectra (W m-2)
                 SncdiffNIR  #Net shortwave diffuse radiation to the canopy in the
#near infrared spectra (W m-2)

                If Rs = 0 Then
                Snc = 0
                GoTo 100
                End If

                If wc >= row Then wc = 0.99 * row

                taudirVIS = taudir(thetas, psis, hc, wc, row, LAI, XE, ZetaVIS, rhosVIS)
                taudiffVIS = taudiff(hc, wc, row, LAI, XE, ZetaVIS, rhosVIS)
                taudirNIR = taudir(thetas, psis, hc, wc, row, LAI, XE, ZetaNIR, rhosNIR)
                taudiffNIR = taudiff(hc, wc, row, LAI, XE, ZetaNIR, rhosNIR)

                alphadirVIS = rhocsdir(thetas, psis, hc, wc, row, LAI, XE, ZetaVIS, rhosVIS)
                alphadiffVIS = rhocsdiff(hc, wc, row, LAI, XE, ZetaVIS, rhosVIS)
                alphadirNIR = rhocsdir(thetas, psis, hc, wc, row, LAI, XE, ZetaNIR, rhosNIR)
                alphadiffNIR = rhocsdiff(hc, wc, row, LAI, XE, ZetaNIR, rhosNIR)

                SncdirVIS = Rs * PISI * KbVIS * fsc * (1 - taudirVIS) * (1 - alphadirVIS)
                SncdiffVIS = Rs * PISI * (1 - KbVIS) * fdhc * (1 - taudiffVIS) * (1 - alphadiffVIS)
                SncdirNIR = Rs * (1 - PISI) * KbNIR * fsc * (1 - taudirNIR) * (1 - alphadirNIR)
                SncdiffNIR = Rs * (1 - PISI) * (1 - KbNIR) * fdhc * (1 - taudiffNIR) * (1 - alphadiffNIR)

                Snc = SncdirVIS + SncdiffVIS + SncdirNIR + SncdiffNIR

                100 
                #def taudir to compute transmittance of DIRECT beam radiation through the canopy
#using procedure of Campbell and Norman (1998), Chapter 15 (CN98)
                def taudir(thetas , psis , hc , wc , row , 
                                LAI , XE , Zeta , rhosoil ) 

                #Thetas = Solar zenith angle (rad)
#Psis = Solar azimuth angle from row orientation, where Psis = 0 degrees for parallel
                #   and 90 degrees for perpendicular orientation (rad)
#hc = Canopy height (m)
                #wc = canopy width (m)
#LAI = Leaf area index, field (m2 m-2)
                #XE = Ratio of horizontal to vertical projected leaves (for spherical LADF, XE = 1)
#Zeta = Leaf absorptivity (usually 0.85 for VIS, 0.15 for NIR)
                #rhosoil = Soil reflectance (for Pullman clay loam, ~0.15 for VIS, ~0.25 for NIR)

 PLFi   #Path length fraction of continuous ellipse canopy
                 MRFi   #Multiple row function of continuous ellipse canopy
 Kdir   #Extinction coefficient for direct beam radiation
                 rhohor      #Reflectivity for horizontal leafs (VIS or NIR)
 rhodir      #Adjusted canopy reflectivity as a function of Kdir and rhohor (VIS or NIR)

                PLFi = PLF(thetas, psis, hc, wc, row)
                MRFi = MRF(thetas, psis, hc, wc, row)
                Kdir = ((XE ^ 2 + (Tan(thetas)) ^ 2) ^ 0.5) / (XE + 1.774 * (XE + 1.182) ^ -0.733) #CN98, 15.4, p. 251
rhohor = (1 - ((Zeta) ^ 0.5)) / (1 + ((Zeta) ^ 0.5))    #CN98, 15.7, p. 255
                rhodir = 2 * Kdir * rhohor / (Kdir + 1)     #CN98, 15.8, p. 257

#Convert field LAI to local LAI
                 LAIL   #Local LAI (i.e., within vegeation row) (m2 m-2)
LAIL = LAI * row / wc

taudir = (((rhodir ^ 2) - 1) * Exp(-(Zeta ^ 0.5) * Kdir * LAIL * PLFi * MRFi)) / 
(((rhodir * rhosoil) - 1) + rhodir * (rhodir - rhosoil) * 
Exp(-2 * (Zeta ^ 0.5) * Kdir * LAIL * PLFi * MRFi))    #CN98, 15.11, p. 257

                

       
