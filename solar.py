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

def zenith(doy, tdec, lon, lat, lz):
    #DOY = Julian Day of Year
    #t = Standard clock time of midpoint of period without decimals (hr)
    #       e.g., 12:45 am = 45; 3:15 pm = 1515
    #lon = longitude of weather station (dec deg)
    #lat = latitude of weather station (minute component)
    #lz = longitude of center of time zone (degrees West of Greenwich)
    #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
    #       respectively; lz = 0 (Greenwich and lz = 330 (Cairo).

    #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
    longr = pi / 180 * lon  #(rad)
    latr = pi / 180 * lat #(rad)
    lzr = pi / 180 * lz  #(rad)

    #Convert raw time to decimal time
    #Tdec = Int(t / 100) + (Right(t, 2)) / 60
    
    #Compute solar noon
    # b  #Used to compute equation of time (radians)
    # Teq    #Seasonal correction for solar time, or equation of time (hour)
    # Tsn  #Solar noon, no daylight savings time correction (decimal hours)
    b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
    Teq = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)    #(hr), FAO 56, p.48, eq.32
    Tsn = 12 + (12 / pi * (longr - lzr) - Teq)    #(hr), Evett (2001), p.A-138, eq.5.8

    #Find solar time angle in radians
    H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7
    
    #Compute solar declination
    d = 0.409 * np.sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24

    #Compute sinPl, the sine of sun angle above horizon (path length)
    sinte = np.sin(latr) * np.sin(d) + np.cos(latr) * np.cos(d) * np.cos(H)     #FAO 56, p.226, eq.(3-15), minimum  = 0.10

    #Constrain sinPl >= 0
    if sinte >= 0:
        sinPhi = sinte
    else:
        sinPhi = 0

    #Compute Phi
    Xtemp = 1 - ((sinPhi) ** 2)  #FAO 56, p.47, eq.27

    if Xtemp < 0:
        X = 0.00001
    else:
        X = Xtemp


    Solarzenith = 90 - 180 / pi * (np.atan((sinPhi) / (X ** 0.5))) #(deg), FAO 56, p.47, eq.26
    return Solarzenith
                              

def Solarazimuth(DOY , Tdec , lon , lat, lz ):

    #DOY = Julian Day of Year
    #t = Standard clock time of midpoint of period without decimals (hr)
    #       e.g., 12:45 am = 45; 3:15 pm = 1515
    #lz = longitude of center of time zone (degrees West of Greenwich)
    #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
    #       respectively; lz = 0 (Greenwich and lz = 330 (Cairo).
    
    #Variable definitions internal to compuing Solarazimuth
    # pi     #pi
    # longr   #longitude of weather station (radians)
    # latr   #latitude of weather station (radians)
    # lzr    #longitude of center of time zone (radians West of Greenwich)
    ## Tdec   #Raw time converted to decimal time (hr)
    # H  #Solar time angle at midpoint of hourly or shorter period (rad)
    # d  #Solar declination (rad)
    # X      #Arc cosine function not available in VB
    #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
    
    longr = pi / 180 * lon  #(rad)
    latr = pi / 180 * lat #(rad)
    lzr = pi / 180 * lz  #(rad)

    #Convert raw time to decimal time
    #Tdec = np.int(t / 100) + (Right(t, 2)) / 60

    #Compute solar noon
    # b  #Used to compute equation of time (radians)
    # Teq    #Seasonal correction for solar time, or equation of time (hour)
    # Tsn  #Solar noon, no daylight savings time correction (decimal hours)
    b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
    Teq = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)    #(hr), FAO 56, p.48, eq.32
    Tsn = 12 + (12 / pi * (longr - lzr) - Teq)    #(hr), Evett (2001), p.A-138, eq.5.8
    
    #Find solar time angle in radians
    H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7
    
    #Compute solar declination
    d = 0.409 * np.sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24
    #Compute X and Y components of sun azimuth
    #http://www.usc.edu/dept/architecture/mbs/tools/vrsolar/Help/solar_concepts.html#Azimuth
    #(accessed 12/05/2007)
    # Xazm 
    # Yazm 
    # azm 
    Xazm = np.sin(H) * np.cos(d)
    Yazm = np.cos(latr) * np.sin(d) - np.sin(latr) * np.cos(H) * np.cos(d)
    azm = 180 / pi * np.atan(Xazm / Yazm)
    
    #Specify azimuth as South of East
    if Tdec < Tsn:
        if azm < 0:
            Solarazimuth = -90 - azm
        else:
            Solarazimuth = 90 - azm
    
    else:
        if azm < 0:
            Solarazimuth = 90 - azm
        else:
            Solarazimuth = 270 - azm
    return Solarazimuth

#def Rso to compute clear sky radiation (W/m2)
def Rso(DOY , t , Ta , HP , P, lon, lat, lz , ele , T1 , RsOpt , kt, eaOPT ): 
    #Rso = Clear sky solar radiation (W/m2)
    #DOY = Julian Day of Year
    #t = Standard clock time of midpoint of period without decimals (hr)
    #       e.g., 12:45 am = 45; 3:15 pm = 1515
    #Ta = Air temperature (C)
    #HP = Humidity parameter, depends on eaOPT
    #P = Measured barometric pressure (kPa)
    #lz = longitude of center of time zone (degrees West of Greenwich)
    #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
    #       respectively; lz = 0 (Greenwich and lz = 330 (Cairo).
    #ele = elevation above mean sea level (m)
    #t1 = length of timestep (hr)
    #Gsc = Solar constant, usually 1367 W/m2
    #RsoOpt = Option to specify Rso calcultaion method:
    #   1 = Simple model based only on elevation
    #   2 = Beer#s law (accounts for P, turbidity, and path length)
    #   3 = Accounts for P, atm moisture, turbidity, and path length)
    #Kt = Turbidity coefficient (Kt = 1 recommended, ASCE-EWRI, 1-24-02, p.D7) (unitless)
    #eaOPT = Option to specify whether dewpoint temperature (Tdew, C) or relative humidity (RH, %)
    #           is used to compute actual vapor pressure of the air (ea, kPa)
    #           eaOPT = 1: RH (%) is used
    #           eaOPT = 2: Twet (%) is used
    #           eaOPT = 3: Tdew (%) is used
    
    #Variable definitions internal to compuing Ra
    # pi     #pi
    # longd  #longitude of weather station (decimal degrees)
    # latr   #latitude of weather station (radians)
    # Tdec   #Raw time converted to decimal time (hr)

    # b  #Used to compute equation of time (radians)
    # Teq    #Seasonal correction for solar time, or equation of time (hour)
    # Tsn    #Solar noon (hour)
    # H  #Solar time angle at midpoint of hourly or shorter period (rad)
    # H1  #Solar time angle at beginning of measurement period (rad)
    # H2  #Solar time angle at ending of measurement period (rad)
    
    # d  #Solar declination (rad)
    # X      #Arc cosine function not available in VB
    # Hs  #Solar time angle at sunset (-Hs = sunrise) (rad)
    
    # DR     #Inverse relative earth-sun distance (dimensionless)
    # ra     #Top of Atmosphere (TOA) solar radiation (W/m2)
    
    #Variable definitions internal to compuing Rso
    # BP     #Barometric pressure (kPa)
    # sinPhi  #np.sine of Phi, where Phi is the angle of the sun above the horizon
    # ea     #Actual vapor pressure (kPa)
    # es     #Saturated vapor pressure of the air (kPa)
    # W  #Precipitable water in the atmosphere (mm)
    # Kb     #The clearness index for direct beam radiation (unitless)
    # Kd     #The corresponding index for diffuse beam radiation (unitless)
    
    # RH     #Relative humidity (%)
    # Twet   #Wet bulb temperature (C)
    # Tdew   #Dew point temperature (C)
    # apsy   #Psychrometer coefficient
    # gammapsy   #Psychrometer constant

    #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
    longd = lon   #(decimal degrees)
    latr = pi / 180 * lat  #(rad)

    #Convert raw time to decimal time; shift period endpoint to period midpoint by delaying 0.5*t1
    Tdec = np.int(t / 100) + ((t / 100) - np.int(t / 100)) * 100 / 60 #- 0.125 * t1

    #Find beginning and ending solar times of measurement period in radians
    b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
    Teq = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)    #(hr), FAO 56, p.48, eq.32
    Tsn = 12 + 4 / 60 * (longd - lz) - Teq #(hr), Evett (2001), p.A-138, eq.5.8
    H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7
    H1 = H - pi * T1 / 24 #(rad), FAO 56, p.48, eq.29
    H2 = H + pi * T1 / 24 #(rad), FAO 56, p.48, eq. 30

    #Compute sunset hour angle
    d = 0.409 * np.sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24
    X = 1 - ((np.tan(latr)) ** 2) * ((np.tan(d)) ** 2)  #FAO 56, p.47, eq.27
    if X <= 0: X = 0.00001
    Hs = pi / 2 - np.atan((-(np.tan(latr)) * (np.tan(d))) / (X ** 0.5)) #(rad), FAO 56, p.47, eq.26

    #Set integration limits for Ra equation when sunrise or sunset fall within computation period,
    #or force Ra zero when sun is below horizon
    if H1 < -Hs: H1 = -Hs #ASCE-EWRI, eq. 55, p. 39
    if H2 < -Hs: H2 = -Hs
    if H1 > Hs: H1 = Hs
    if H2 > Hs: H2 = Hs
    if H1 > H2: H1 = H2

    #Compute extraterrestrial radiation (Ra)
    DR = 1 + 0.033 * np.cos(2 * pi * DOY / 365) #(dimensionless), FAO 56, p.46, eq.23
    ra = 24 / (2 * pi * T1) * Gsc * DR * ((H2 - H1) * np.sin(latr) * np.sin(d) + np.cos(latr) * np.cos(d) * (np.sin(H2) - np.sin(H1))) #(MJ/m2/t1), FAO 56, p.47, eq.28
    
    #*******************************************************************************
    #Compute Rso

    #Rso Options (1, 2, or 3)
    
    #OPTION 1: SIMPlE
    if RsOpt = 1:
        Rso = (0.75 + (2 * 10 ** -5) * ele) * ra #(MJ/m2/t1), FAO 56, p.51, eq.37
        return Rso
    #Compute P and sinphi, which are required for Options 2 and 3
    #Determine P (either computed or measured)
    if np.isnan(P): P = 101.3 * (((293 - 0.0065 * ele) / 293) ** 5.26) #FAO 56, p.31, eq.7
    #Compute sinphi, the sine of sun angle above horizon (path length)
    sinPhi = np.sin(latr) * np.sin(d) + np.cos(latr) * np.cos(d) * np.cos(H)     #FAO 56, p.226, eq.(3-15), minimum  = 0.10
    if sinPhi < 0.1: sinPhi = 0.1
    if RsOpt = 2:
        Rso = ra * np.exp((-0.0018 * P) / (kt * sinPhi))    #(MJ/m2/t1), FAO 56, p.226, eq.(3-14)
        return Rso
    if RsOpt = 3:
        ea=aero.ea(P , Ta , HP , eaOPT)
        W = 0.14 * ea * P + 2.1 #FAO 56, p.227, eq.(3-19)
        Kb = 0.98 * np.exp(((-0.00146 * P) / (kt * sinPhi)) - 0.075 * ((W / sinPhi) ** 0.4)) #FAO 56, p.227, eq.(3-18), slightly modified in ASCE-EWSI, 1-24-02, p.D-7, eq.(D.2)
        if Kb >= 0.15:   #FAO 56, p.227, eq.(3-20), slightly modified in ASCE-EWSI, 1-24-02, p.D-8, eq.(D.4)
            Kd = 0.35 - 0.36 * Kb
        else:
            Kd = 0.18 + 0.82 * Kb
                                  
        Rso = (Kb + Kd) * ra    #(MJ/m2/t1), FAO 56, p.227, eq.(3-17)
        Rso = max(0,Rso)
        return Rso
#def Rsob to compute clear sky BEAM radiation (W/m2)
def Rsob(DOY , t , Ta , HP , P , lon, lat , lz , ele , T1 , RsOpt , kt , eaOPT ): 
    #Rsob = Clear sky BEAM solar radiation (W/m2)
    #DOY = Julian Day of Year
    #t = Standard clock time of midpoint of period without decimals (hr)
    #       e.g., 12:45 am = 45; 3:15 pm = 1515
    #Ta = Air temperature (C)
    #HP = Humidity parameter, depends on eaOPT
    #Pmea = Measured barometric pressure (kPa)
    #lz = longitude of center of time zone (degrees West of Greenwich)
    #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
    #       respectively; lz = 0 (Greenwich and lz = 330 (Cairo).
    #ele = elevation above mean sea level (m)
    #t1 = length of timestep (hr)
    #Gsc = Solar constant, usually 1367 W/m2
    #RsoOpt = Option to specify Rso calcultaion method:
    #   1 = Simple model based only on elevation
    #   2 = Beer#s law (accounts for P, turbidity, and path length)
    #   3 = Accounts for P, atm moisture, turbidity, and path length)
    #Kt = Turbidity coefficient (Kt = 1 recommended, ASCE-EWRI, 1-24-02, p.D7) (unitless)
    #eaOPT = Option to specify whether dewpoint temperature (Tdew, C) or relative humidity (RH, %)
    #           is used to compute actual vapor pressure of the air (ea, kPa)
    #           eaOPT = 1: RH (%) is used
    #           eaOPT = 2: Twet (%) is used
    #           eaOPT = 3: Tdew (%) is used

    #Variable definitions internal to compuing Ra
    # pi     #pi
    # longd  #longitude of weather station (decimal degrees)
    # latr   #latitude of weather station (radians)
    # Tdec   #Raw time converted to decimal time (hr)
    
    # b  #Used to compute equation of time (radians)
    # Teq    #Seasonal correction for solar time, or equation of time (hour)
    # Tsn    #Solar noon (hour)
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
    # Twet   #Wet bulb temperature (C)
    # Tdew   #Dew point temperature (C)
    # apsy   #Psychrometer coefficient
    # gammapsy   #Psychrometer constant
    
    #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
    longd = lon   #(decimal degrees)
    latr = pi / 180 * lat  #(rad)
    #Convert raw time to decimal time; shift period endpoint to period midpoint by delaying 0.5*t1
    Tdec = np.int(t / 100) + ((t / 100) - np.int(t / 100)) * 100 / 60 #- 0.5 * t1

    #Find beginning and ending solar times of measurement period in radians
    b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
    Teq = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)    #(hr), FAO 56, p.48, eq.32
    Tsn = 12 + 4 / 60 * (longd - lz) - Teq #(hr), Evett (2001), p.A-138, eq.5.8
    H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7
    H1 = H - pi * T1 / 24 #(rad), FAO 56, p.48, eq.29
    H2 = H + pi * T1 / 24 #(rad), FAO 56, p.48, eq. 30

    #Compute sunset hour angle
    d = 0.409 * np.sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24
    X = 1 - ((np.tan(latr)) ** 2) * ((np.tan(d)) ** 2)  #FAO 56, p.47, eq.27
    if X <= 0: X = 0.00001
    Hs = pi / 2 - np.atan((-(np.tan(latr)) * (np.tan(d))) / (X ** 0.5)) #(rad), FAO 56, p.47, eq.26

    #Set integration limits for Ra equation when sunrise or sunset fall within computation period,
    #or force Ra zero when sun is below horizon
    if H1 < -Hs: H1 = -Hs #ASCE-EWRI, eq. 55, p. 39
    if H2 < -Hs: H2 = -Hs
    if H1 > Hs: H1 = Hs
    if H2 > Hs: H2 = Hs
    if H1 > H2: H1 = H2

    #Compute extraterrestrial radiation (Ra)
    DR = 1 + 0.033 * np.cos(2 * pi * DOY / 365) #(dimensionless), FAO 56, p.46, eq.23
    ra = 24 / (2 * pi * T1) * Gsc * DR * ((H2 - H1) * np.sin(latr) * np.sin(d) + np.cos(latr) * np.cos(d) * (np.sin(H2) - np.sin(H1))) #(MJ/m2/t1), FAO 56, p.47, eq.28

    #*******************************************************************************
    #Compute Rso
    
    #Rso Options (1, 2, or 3)

    #OPTION 1: SIMPlE
    if RsOpt = 1:
        Rsob = (0.75 + (2 * 10 ** -5) * ele) * ra #(MJ/m2/t1), FAO 56, p.51, eq.37
        return Rsob
    #Compute P and sinphi, which are required for Options 2 and 3
    #Determine P (either computed or measured)
    if np.isnan(P):P = 101.3 * (((293 - 0.0065 * ele) / 293) ** 5.26) #FAO 56, p.31, eq.7
    #Compute sinphi, the sine of sun angle above horizon (path length)
    sinPhi = np.sin(latr) * np.sin(d) + np.cos(latr) * np.cos(d) * np.cos(H)     #FAO 56, p.226, eq.(3-15), minimum  = 0.10
    if sinPhi < 0.1: sinPhi = 0.1

    #OPTION 2: BEER#S lAW
    if RsOpt = 2:
        Rsob = ra * np.exp((-0.0018 * P) / (kt * sinPhi))    #(MJ/m2/t1), FAO 56, p.226, eq.(3-14)
        return Rsob
    #OPTION 3: Atm moisture and turbidity
    if RsOpt = 3:
        ea=aero.ea(P , Ta , HP , eaOPT)    
        W = 0.14 * ea * P + 2.1 #FAO 56, p.227, eq.(3-19)
        Kb = 0.98 * np.exp(((-0.00146 * P) / (kt * sinPhi)) - 0.075 * ((W / sinPhi) ** 0.4)) #FAO 56, p.227, eq.(3-18), slightly modified in ASCE-EWSI, 1-24-02, p.D-7, eq.(D.2)

        #    if Kb >= 0.15:   #FAO 56, p.227, eq.(3-20), slightly modified in ASCE-EWSI, 1-24-02, p.D-8, eq.(D.4)
        #        Kd = 0.35 - 0.36 * Kb
        #    Else
        #        Kd = 0.18 + 0.82 * Kb
        #    
        #    Rso = (Kb + Kd) * Ra    #(MJ/m2/t1), FAO 56, p.227, eq.(3-17)

        Rsob = Kb * ra  #(MJ/m2/t1)
        Rsob = max(0,Rsob)
        return Rsob                              

def Kbeam(Rs , KbConst , KbExp , DOY , t , Ta , P , lon , lat , lz , ele , T1 , RsOpt , kt , eaOPT ):
    if Rs < 1: return 0

    # Rsobeam    #Theoretical clear sky direct beam irradiance (W m-2)
    # Rsoglobal  #Theoretical clear sky global irradiance (W m-2)
    
    Rsobeam = Rsob(DOY , t , Ta , HP , P , lon, lat , lz , ele , T1 , RsOpt , kt , eaOPT)
    Rsoglobal = Rso(DOY , t , Ta , HP , P , lon, lat , lz , ele , T1 , RsOpt , kt , eaOPT )
    if Rsoglobal < 1: return 0

    return (Rsobeam / Rsoglobal) * KbConst * (Rs / Rsoglobal) ** KbExp                  
                              

                              

                              
