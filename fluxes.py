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

def Rnminmax(year_doy,Rn):
    days = np.unique(year_doy)
    Rnmax = np.zeros(Rn.shape)
    Rnmin = np.zeros(Rn.shape)    
    for i in range(days.shape[0]):
        mask = year_doy[:,0]==days[i,0] & year_doy[:,1]==days[i,1]
        Rmax_day = np.max(Rn[mask])
        Rmin_day = np.min(Rn[mask])
        Rnmax[mask]=Rmax_day
        Rnmin[mask]=Rmin_day
    return Rnmax, Rnmin

def G(R):
    #R (soil or residue) 24 hour time series
    #G (soil or residue) 24 hour time series
    #from Colaizzi 2016    
    Rmax, Rmin = Rnminmax(time, R)
    return (R-Rmin)/(Rmax-Rmin)*(aG*Rmax+Rmin)-Rmin
    
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
def Hr(Tac,Tr,rr,fs):
    return aero.rho_a(Tac)*(Tr-Tac)/rr*c_p

class rad_fluxes:
    def __init__(self,inputs_obj,thetas,psis,taudiff_vis,taudiff_nir,taudir_vis,taudir_nir):
        self.io = inputs_obj
        
    def Lsky(self): 
        #def Lsky to calculate hemispherical downwelling longwave irradiance from the sky (W m-2).
        # emisatm    #Atmospheric emissivity

        ea1 = aero.ea(P, Ta, RH)

        emisatm = 0.7 + (0.000595) * ea1 * np.exp(1500 / (Ta + 273.16))
        #Idso et al (1981) WRR 17(2): 295-304, eq. 17 (hemispherical view factor, full longwave spectrum)

        #emisatm = 1.24 * ((10 * ea1 / (Ta + 273.1)) ** (1 / 7))
        #Brutsaert (1975) WRR 11: 742-744.

        return emisatm * boltz * ((Ta + Tk) ** 4)   

    def Lns(P , Ta , RH , Ts , Tc , LAI , wc , row , fdhc, fres, fsoil): 
        #def Lnsoil to compute the net longwave radiation to the SOIL using the
        LAIL = LAI * row / wc
        Lc = emisveg * boltz * ((Tc + Tk) ** 4)
        Ls = emisSoil * boltz * ((Ts + Tk) ** 4)
        lwexp = np.exp(-kappIr * LAIL)

        return emisSoil * Lsky(P , Ta , RH ) * (1 - fdhc + fdhc * lwexp) + emisSoil * Lc * fdhc * (1 - lwexp) - Ls

    def Lnc(P , Ta , RH , Ts , Tc , LAI , wc , row  , fdhc ) :
        #def Lncanopy to compute the net longwave radiation to the CANOPY using the
        LAIL = LAI * row / wc
        Lc = emisveg * boltz * ((Tc + Tk) ** 4)
        Ls = emisSoil * boltz * ((Ts + Tk) ** 4)
        lwexp = np.exp(-kappIr * LAIL)

        return (emisveg * Lsky(P , Ta , RH ) + emisveg * Ls - (1 + emisSoil) * Lc) * fdhc * (1 - lwexp)


    def Sns(Rs , KbVis , KbNir , fsc , fdhc , thetas , psis , hc , wc , row , Pr , Vr , LAI): 

        #def Sns to calculate net shortwave radiation to the soil using
        #Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model

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


                

       
