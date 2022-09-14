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
    def __init__(self,inputs_obj,solar_obj,canopy_obj):
        self.io = inputs_obj
        self.sol = solar_obj
        self.can = canopy_obj
        
    def Lsky(self): 
        #def Lsky to calculate hemispherical downwelling longwave irradiance from the sky (W m-2).
        # emisatm    #Atmospheric emissivity
        emisatm = 0.7 + (0.000595) * self.io.ea * np.exp(1500 / (self.io.Ta + Tk))
        #Idso et al (1981) WRR 17(2): 295-304, eq. 17 (hemispherical view factor, full longwave spectrum)

        #emisatm = 1.24 * ((10 * ea1 / (Ta + 273.1)) ** (1 / 7))
        #Brutsaert (1975) WRR 11: 742-744.

        return emisatm * boltz * ((self.io.Ta + Tk) ** 4)   

    def Lns(self): 
        #def Lnsoil to compute the net longwave radiation to the SOIL using the
        self.Lc = self.io.emis_veg * boltz * (self.io.f_veg_sun*(self.io.T_veg_sun + Tk) ** 4+self.io.f_veg_shade*(self.io.T_veg_shade + Tk) ** 4)/self.io.f_veg
        self.Ls = self.io.emis_soil * boltz * (self.io.f_soil_sun*(self.io.T_soil_sun + Tk) ** 4+self.io.f_soil_shade*(self.io.T_soil_shade + Tk) ** 4)/self.io.f_soil*self.f_soil_rel
        self.Lr = self.io.emis_res * boltz * ((self.io.f_res_sun*(self.io.T_res_sun + Tk) ** 4+self.io.f_res_shade*(self.io.T_res_shade + Tk) ** 4)//self.io.f_res*(1-self.f_soil_rel)
        
        self.lwexp = np.exp(-self.io.kappa_ir * self.io.LAIL)

        return (self.io.emis_soil * self.Lsky() * (1 - self.can.fdhc + self.can.fdhc * self.lwexp) + self.io.emis_soil * self.Lc * self.can.fdhc * (1 - self.lwexp))*self.f_soil_rel - self.Ls

    def Lnr(self): 
        return (self.io.emis_res * self.Lsky() * (1 - self.can.fdhc + self.can.fdhc * self.lwexp) + self.io.emis_res * self.Lc * self.can.fdhc * (1 - self.lwexp))*(1-self.f_soil_rel) - self.Lr
                                              
    def Lnc(self) :
        return (self.io.emis_veg * self.Lsky() + self.io.emis_veg * (self.Ls+self.Lr) - (1 + self.io.emis_soil) * self.Lc) * self.can.fdhc * (1 - self.lwexp)

    def Sns(self): 
        #def Sns to calculate net shortwave radiation to the soil using
        #Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model
        self.TVis = self.io.Rs * self.io.pisi * ((self.can.fsis * self.can.taudir_vis + 1 - self.can.fsc) * self.sol.KbVis + (self.can.fdhc * self.can.taudiff_vis + 1 - self.can.fdhc) * (1 - self.sol.KbVis))
        self.TNir =  self.io.Rs * (1 - self.io.pisi) * ((self.can.fsis * self.can.taudir_nir + 1 - self.can.fsis) * self.sol.KbNir + (self.can.fdhc * self.can.taudiff_nir + 1 - self.can.fdhc) * (1 - self.sol.KbNir))

        return (TVis * (1 - self.io.soil_alb_vis) + TNir * (1 - self.io.soil_alb_nir))*(Rs>0) * self.io.f_soil_rel

    def Snr(self): 
        #def Snr to calculate net shortwave radiation to the res using
        #Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model - modified to account for res/soil mix
        self.TVis = self.io.Rs * self.io.pisi * ((self.can.fsis * self.can.taudir_vis + 1 - self.can.fsc) * self.sol.KbVis + (self.can.fdhc * self.can.taudiff_vis + 1 - self.can.fdhc) * (1 - self.sol.KbVis))
        self.TNir =  self.io.Rs * (1 - self.io.pisi) * ((self.can.fsis * self.can.taudir_nir + 1 - self.can.fsis) * self.sol.KbNir + (self.can.fdhc * self.can.taudiff_nir + 1 - self.can.fdhc) * (1 - self.sol.KbNir))

        return (TVis * (1 - self.io.res_alb_vis) + TNir * (1 - self.io.res_alb_nir))*(Rs>0) * (1-self.io.f_soil_rel)

    def Snc(self): 
        #def Snc to calculate net shortwave radiation to the canopy using
        #Campbell and Norman (1998) radiative transfer model and elliptical hedgerow geometric model

        SncdirVis = self.io.Rs * self.io.pisi * self.sol.KbVis * self.can.fsis * (1 - self.can.taudir_vis) * (1 - self.can.rhocsdir_vis)
        SncdiffVis = self.io.Rs * self.io.pisi * (1 - self.sol.KbVis) * self.can.fdhc * (1 - self.can.taudiff_vis) * (1 - self.can.rhocsdiff_vis)
        SncdirNir = self.io.Rs * (1 - self.io.pisi) * self.sol.KbNir * self.can.fsis * (1 - self.can.taudir_nir) * (1 - self.can.rhocsdir_nir)
        SncdiffNir =  self.io.Rs * (1 - self.io.pisi) * (1 - self.sol.KbNir) * self.can.fdhc * (1 - self.can.taudiff_nir) * (1 - self.can.rhocsdiff_nir)

        return SncdirVis + SncdiffVis + SncdirNir + SncdiffNir


                

       
