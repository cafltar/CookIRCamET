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
        
        self.mask_bare = (self.io.f_veg<=0) & (self.io.f_res<=0)
        self.mask_flat_res = (self.io.f_veg<=0) & (self.io.f_res>0) & (~self.io.stubble)
        self.mask_stand_res = (self.io.f_veg<=0) & (self.io.f_res>0) & (self.io.stubble)
        self.mask_veg_no_res = (self.io.f_veg>0) & (self.io.f_res<=0)
        self.mask_veg_flat_res = (self.io.f_veg>0) & (self.io.f_res>0) & (~self.io.stubble) 
        self.mask_veg_tall_stand_res = (self.io.f_veg>0) & (self.io.f_res>0) & (self.io.stubble) & (self.io.hr>self.io.hc)
        self.mask_veg_short_stand_res = (self.io.f_veg>0) & (self.io.f_res>0) & (self.io.stubble) & (self.io.hr<=self.io.hc)
                
    def roughness_lengths(self):
        #canopy
        self.d_c=self.io.dhc * self.hc
        self.zom_c=self.io.zomhc * self.hc
        self.zoh_c=self.io.zohzomc * self.zom
        #surface
        self.d_bs=self.io.dhbs * self.hbs
        self.zom_bs=self.io.zomhbs * self.hbs
        self.zoh_bs=self.io.zohzombs * self.zom_bs
        self.d_fr=self.io.dhfr * self.hfr
        self.zom_fr=self.io.zomhfr * self.hfr
        self.zoh_fr=self.io.zohzomfr * self.zom_fr
        self.d_ss=self.io.dhss * self.hss
        self.zom_ss=self.io.zomhss * self.hss
        self.zoh_ss=self.io.zohzomss * self.zom_ss

        self.d_s[self.io.stubble]=self.d_ss[self.io.stubble]
        self.zom_s[self.io.stubble]=
        self.zoh_s[self.io.stubble]=
        
        self.d_s[(~self.io.stubble) & (self.io.f_res>0)]=
        self.zom_s[(~self.io.stubble) & (self.io.f_res>0)]=
        self.zoh_s[(~self.io.stubble) & (self.io.f_res>0)]=

        self.d_s[(~self.io.stubble) & (~self.io.f_res>0)]=
        self.zom_s[(~self.io.stubble) & (~self.io.f_res>0)]=
        self.zoh_s[(~self.io.stubble) & (~self.io.f_res>0)]=

        self.d[(~self.io.f_veg>0)]=self.d_s[(~self.io.f_veg>0)]
        self.zom[(~self.io.f_veg>0)]=
        self.zoh[(~self.io.f_veg>0)]=

        self.d[(self.io.f_veg>0)]=self.d_c[(self.io.f_veg>0)]
        self.zom[(self.io.f_veg>0)]=
        self.zoh[(self.io.f_veg>0)]=

        
    def rah_calc(self): 
        #def to compute aerodynamic resistance (s/m) using Monin-Obukov (1954)
        #Similarity Theory (MOST), where correction coefficients for unstable and stable conditions
        #are given by Paulson (1970) and Webb (1970)

        PsiM = np.zeros(self.io.T_rad.shape)
        PsiH = np.zeros(self.io.T_rad.shape)
        uf = vonk * self.io.u / ((np.log((self.io.zu - self.d) / self.zom)) - PsiM)
        rahmost = ((np.log((self.io.zu - self.d) / (self.zoh))) - PsiH) / (vonk * uf)
        dT = self.io.Ta - self.io.T_rad

        if np.abs(dT) < 0.01: dT = 0.01

        Toh = self.io.Ta - dT
        n = 1

        while n < nmax and np.max(np.abs(rah - rahmost)) > tol:
            rah = rahmost
            L = rah * (uf ** 3) * (Toh + Tk) / (g * vonk * dT)

            #Stable conditions
            PsiH[L>0] = -5 * (self.io.zu - self.d) / L
            PsiM[L>0] = -5 * (self.io.zu - self.d) / L

            #Unstable conditions
            X = (1 - 16 * (self.io.zu - self.d) / L) ** 0.25
            PsiH[L<=0] = 2 * np.log((1 + (X ** 2)) / 2)
            PsiM[L<=0] = 2 * np.log((1 + X) / 2) + np.log((1 + (X ** 2)) / 2) - 2 * np.atan(X) + pi / 2

            uf = vonk * self.io.u / ((np.log((self.io.zu - self.d) / self.zom)) - PsiM)
            rahmost = ((np.log((self.io.zu - self.d) / (self.zoh))) - PsiH) / (vonk * uf)
            n = n + 1

        if n == nmax:
            uf = vonk * self.io.u / ((np.log((self.io.zu - self.d) / self.zom)))
            rahmost = ((np.log((self.io.zu - self.d) / (self.zoh)))) / (vonk * uf)
        self.rah = rahmost

    def rx_calc(self): 
        #def rx to compute resistance of heat transport between canopy and canopy
        #displacement height; taken from Norman et al. (1995) Appendix A

        #Variables internal to this function
        #Uc     #Wind speed at top of canopy (m s-1)
        #A      #Empirical factor
        #Udz    #Wind speed at momentum height d + zom (m s-1)

        PsiM = np.zeros(self.io.T_rad.shape)

        uc = self.io.u * (np.log((self.io.hc - self.d) / self.zom)) / ((np.log((self.io.zu - self.d) / self.zom)) - PsiM)
        A = 0.28 * (self.io.LAIL ** (2 / 3)) * (self.io.hc ** (1 / 3)) * (self.io.s ** (-1 / 3))
        udz = uc * np.exp(-A * (1 - ((self.d + self.zom) / self.io.hc)))
        if udz < 0.1: udz = 0.1
        self.rx = (self.io.Cx / self.io.LAIL) * ((self.io.s / udz) ** 0.5)
        

    def rsh_calc(self): 
        #def rsh to compute resistance of heat transport between soil/residue and canopy displacement height
        if False:
            #Taken from Norman et al. (1995) Appendix B AND Kustas and Norman (1999)

            #Variables internal to this function
            #Uc     #Wind speed at top of canopy (m s-1)
            #A      #Empirical factor
            #Us     #Wind speed just above the soil surface (m s-1)
            #PsiM   #Stability correction for momentum at top of canopy,
            #                    assumed negligible due to roughness effects in the sublayer

            PsiM = np.zeros(self.io.T_rad.shape)

            uc = self.io.u * (np.log((self.io.hc - self.d) / self.zom)) / ((np.log((self.io.zu - self.d) / self.zom)) - PsiM)
            A = 0.28 * (self.io.LAIL ** (2 / 3)) * (self.io.hc ** (1 / 3)) * (self.io.s ** (-1 / 3))
            us = uc * np.exp(-A * (1 - (0.05 / self.io.hc)))
            self.rsh = 1 / (self.io.c * ((np.abs(self.io.T_s - self.io.T_c)) ** (1 / 3)) + self.io.b * us)
        else:
            #SW model (1990)
            alpha = 2.5
            u_s = vonk*self.io.u/np.log((self.io.zu-self.d)/self.zom)#friction velocity
            Kh = vonk*u_s*(self.io.hc-self.d)#eddy diffusion coefficient at the top of the canopy
            self.rsh = self.hc*np.exp(alpha)/(alpha*Kh)*(np.exp(-alpha*self.zom_s/self.hc)-np.exp(-alpha*(self.d+self.zom)/self.hc))
