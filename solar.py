import sys
import numpy as np
import cv2
import time
import utils, aero, canopy
from time import sleep
from datetime import datetime
import os
import numpy as np
import logging 
from constants import *

class radiation:
    def __init__(self,inputs_obj):
        #pass in inputs obj to reduce excessive arguments
        self.io = inputs_obj
        
    def solarangles(self):
        #DOY = Julian Day of Year
        #t = Standard clock time of midpoint of period without decimals (hr)
        #       e.g., 12:45 am = 45; 3:15 pm = 1515
        #lon = longitude of weather station (dec deg)
        #lat = latitude of weather station (minute component)
        #lz = longitude of center of time zone (degrees West of Greenwich)
        #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
        #       respectively; lz = 0 (Greenwich and lz = 330 (Cairo).

        #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
        self.longr = pi / 180 * self.io.lon  #(rad)
        self.latr = pi / 180 * self.io.lat #(rad)
        self.lzr = pi / 180 * self.io.lz  #(rad)

        #Convert raw time to decimal time
        Tdec = np.floor(self.io.t / 100) + (self.io.t-floor(self.io.t/100)*100) / 60 - self.io.tstep/2

        #Compute solar noon
        # b  #Used to compute equation of time (radians)
        # Teq    #Seasonal correction for solar time, or equation of time (hour)
        # Tsn  #Solar noon, no daylight savings time correction (decimal hours)
        b = 2 * pi * (self.io.doy - 81) / 364     #FAO 56, p.48, eq.33
        Teq = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)    #(hr), FAO 56, p.48, eq.32
        Tsn = 12 + (12 / pi * (self.longr - self.lzr) - Teq)    #(hr), Evett (2001), p.A-138, eq.5.8

        #Find solar time angle in radians
        self.H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7

        H1x = self.H - pi * self.io.tstep / 24 # (rad), ASCE 70 2nd ed., p. 67, eq. (4-16)
        H2x = self.H + pi * self.io.tstep / 24 # (rad), ASCE 70 2nd ed., p. 67, eq. (4-17)
        #Compute solar declination
        self.d = 0.4093 * np.sin(2 * pi * self.io.doy / 365 - 1.39) #(rad), FAO 56, p.46, eq.24

        self.Hs = np.acos(-(np.tan(self.latr)) .* (np.tan(self.d))) # (rad), ASCE 70 2nd ed., p. 67, eq. (4-14) sunset hour angle
        self.solarzenith = np.min(90,180/pi*(np.acos(np.sin(self.latr) * np.sin(self.d) + np.cos(self.latr) * np.cos(self.d) * np.cos(self.H))))#FAO 56, p.226, eq.(3-15) but acos for zenith

        # Calculate X and Y components of solar azimuth
        # http://www.usc.edu/dept/architecture/mbs/tools/vrsolar/Help/solar_concepts.html#Azimuth
        # (accessed 12/05/2007)
        Xazm = np.sin(self.H) * np.cos(self.d);
        Yazm = np.cos(self.latr) * np.sin(self.d) - np.sin(self.latr) * np.cos(self.H) * np.cos(self.d);
        SATdec0 = 180/pi*np.atan(Xazm / Yazm);
        #Specify solar azimuth as CWN
        SATdec1 = (Tdec < Tsn and SATdec0 < 0)*(-SATdec0);
        SATdec2 = (Tdec < Tsn and SATdec0 > 0)*(180-SATdec0);
        SATdec3 = (Tdec > Tsn and SATdec0 < 0)*(180-SATdec0);
        SATdec4 = (Tdec > Tsn and SATdec0 > 0)*(360-SATdec0);
        self.solarazimuth = SATdec1 + SATdec2 + SATdec3 + SATdec4;

        # Set integration limits for Ra equation when sunrise or sunset occur
        # within computation period, or force Ra to zero when sun is below horizon
        #  ASCE 70 2nd ed., p. 68, eq. (4-19)
        self.H1 = min((max(-self.Hs,H1x)),self.Hs);
        self.H2 = min((max(-self.Hs,H2x)),self.Hs);
        self.H1 = min(self.H1,self.H2);
   
    #def Rso to compute clear sky radiation (W/m2)
    def Rso(self): 
        #Rso = Clear sky solar radiation (W/m2)
        #Gsc = Solar constant, usually 1367 W/m2
        #RsoOpt = Option to specify Rso calcultaion method:
        #   1 = Simple model based only on elevation
        #   2 = Beer#s law (accounts for P, turbidity, and path length)
        #   3 = Accounts for P, atm moisture, turbidity, and path length)
        #Kt = Turbidity coefficient (Kt = 1 recommended, ASCE-EWRI, 1-24-02, p.D7) (unitless)

        # H  #Solar time angle at midpoint of hourly or shorter period (rad)
        # H1  #Solar time angle at beginning of measurement period (rad)
        # H2  #Solar time angle at ending of measurement period (rad)

        # d  #Solar declination (rad)
        # X      #Arc cosine function not available in VB
        # Hs  #Solar time angle at sunset (-Hs = sunrise) (rad)

        # DR     #Inverse relative earth-sun distance (dimensionless)
        # ra     #Top of Atmosphere (TOA) solar radiation (W/m2)

        #Variable definitions internal to compuing Rso
        # P     #Barometric pressure (kPa)
        # sinPhi  #np.sine of Phi, where Phi is the angle of the sun above the horizon
        # ea     #Actual vapor pressure (kPa)
        # es     #Saturated vapor pressure of the air (kPa)
        # W  #Precipitable water in the atmosphere (mm)
        # Kb     #The clearness index for direct beam radiation (unitless)
        # Kd     #The corresponding index for diffuse beam radiation (unitless)

        # RH     #Relative humidity (%)
        # Calculate extraterrestrial radiation (Ra)
        dr = 1 + 0.033 * np.cos(2 * pi * self.io.doy / 365) # Inverse relative Earth-Sun distance
        # (no units),  ASCE 70 2nd ed., p. 67, eq. (4-13)
        Ra = 24 / (2 * pi * self.io.tstep) * Gsc * dr * ((self.H2 - self.H1) * np.sin(self.latr) * np.sin(self.d) + np.cos(self.latr) * np.cos(self.d) * (np.sin(self.H2) - np.sin(self.H1)))
        # (MJ m-2 t1-1),  ASCE 70 2nd ed., p. 66, eq. (4-11b)

        # ***********************************************************************
        # Calculate Rso, theoretical clear sky global solar irradiance

        # Make array for each RsoOpt
        self.Rso = np.zeros(Ra.shape)

        # RsoOpt = 1 (Simple)
        self.Rso[self.io.RsoOpt==1] = Ra[self.io.RsoOpt==1] * (0.75 + (2e-5) * self.io.ele)
        # (MJ/m2/t1),  ASCE 70 2nd ed., p. 63, eq. (4-3), (4-4)

        # RsoOpt = 2 (Beer's Law)
        # Compute sinphi, the sine of sun angle above horizon (path length)
        sinphix = np.sin(self.latr) * np.sin(self.d) + np.cos(self.latr) * np.cos(self.d) * np.cos(self.H)   
        sinphi = max(0.1,sinphix)
        #  ASCE 70 2nd ed., p. 65, eq. (4-10), minimum  = 0.10
        self.Rso[self.io.RsoOpt==2] = Ra[self.io.RsoOpt==2] * np.exp((-0.0018 * self.io.P) / (self.io.Kt * sinphi))    
        # (MJ/m2/t1), FAO 56, p.226, eq.(3-14)

        # RsoOpt = 3 (Atm moisture and turbidity)
        # Calculate beam transmittance (Kb)
        W = 0.14 * self.io.ea * self.io.P + 2.1 #  ASCE 70 2nd ed., p. 64, eq. (4-7)
        self.Kb = 0.98 * np.exp(((-0.00146 * self.io.P) / (self.io.Kt * sinphi)) - 0.075 * ((W / sinphi) ** 0.4)) 
        # ASCE 70 2nd ed., p. 67, eq. (4-6)
        # Calculate diffuse transmittance (Kd) as function of Kb
        Kbx = (Kb > 0.15)
        self.Kd = Kbx * (0.35 - 0.36 * Kb) + (1 - Kbx) * (0.18 + 0.82 * Kb)
        #  ASCE 70 2nd ed., p. 65, eq. (4-8)
        self.Rso[self.io.RsoOpt==3] = Ra[self.io.RsoOpt==3] * (Kb + Kd) # (MJ/m2/t1),
        #  ASCE 70 2nd ed., p. 64, eq. (4-5)
    
    def Kbeam(self):
        # ***********************************************************************
        # Calculate direct beam weighing factors for visible and near-infrared
        # bands, Kbvis and Kbnir, respectively, daytime only
        # Colaizzi et al., 2012, Agron J, 104(2): 225-240, eq. (A1) and (A2)
        Kbday = (self.io.Rs > 0) and (self.Rso > 0)
        self.KbVis = Kbday * max(0,min(((self.Kb / (self.Kb + self.Kd)) * self.io.KbVisConst * (self.io.Rs / Rso) ** self.io.KbVisExp),1))
        self.KbNir = Kbday * max(0,min(((self.Kb / (self.Kb + self.Kd)) * self.io.KbNirConst * (self.io.Rs / Rso) ** self.io.KbNirExp),1))
                              

                              
