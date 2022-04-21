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

def zenith(doy, tdec, lon, lat, Lz)
#DOY = Julian Day of Year
#t = Standard clock time of midpoint of period without decimals (hr)
#       e.g., 12:45 am = 45; 3:15 pm = 1515
#LONGdeg = Longitude of weather station (degree component)
#LONGmin = Longitude of weather station (minute component)
#LATdeg = Latitude of weather station (degree component)
#LATmin = Latitude of weather station (minute component)
#Lz = Longitude of center of time zone (degrees West of Greenwich)
#       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
#       respectively; Lz = 0 (Greenwich and Lz = 330 (Cairo).

#Variable definitions internal to compuing Solarzenith

#Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
                              pi = 3.14159265358979
                              Longr = pi / 180 * (LONGdeg + LONGmin / 60)  #(rad)
Latr = pi / 180 * (LATdeg + LATmin / 60)  #(rad)
                              Lzr = pi / 180 * Lz  #(rad)

#Convert raw time to decimal time
                              #Tdec = Int(t / 100) + (Right(t, 2)) / 60

#Compute solar noon
                              # b As Double #Used to compute equation of time (radians)
# Teq As Double   #Seasonal correction for solar time, or equation of time (hour)
                              # Tsn As Double #Solar noon, no daylight savings time correction (decimal hours)
b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
                              Teq = 0.1645 * Sin(2 * b) - 0.1255 * Cos(b) - 0.025 * Sin(b)    #(hr), FAO 56, p.48, eq.32
Tsn = 12 + (12 / pi * (Longr - Lzr) - Teq)    #(hr), Evett (2001), p.A-138, eq.5.8

                              #Find solar time angle in radians
H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7

                              #Compute solar declination
d = 0.409 * Sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24

                              #Compute sinPL, the sine of sun angle above horizon (path length)
sinte = Sin(Latr) * Sin(d) + Cos(Latr) * Cos(d) * Cos(H)     #FAO 56, p.226, eq.(3-15), minimum  = 0.10

                              #Constrain sinPL >= 0
if sinte >= 0 Then
SinPhi = sinte
Else
SinPhi = 0


#Compute Phi
                              Xtemp = 1 - ((SinPhi) ^ 2)  #FAO 56, p.47, eq.27

if Xtemp < 0 Then
X = 0.00001
Else
X = Xtemp


Solarzenith = 90 - 180 / pi * (Atn((SinPhi) / (X ^ 0.5))) #(deg), FAO 56, p.47, eq.26

                              

                              def Solarazimuth(DOY As Integer, Tdec As Double, LONGdeg As Integer, LONGmin As Integer, _
                                                    LATdeg As Integer, LATmin As Integer, Lz As Double) As Double

                              #DOY = Julian Day of Year
#t = Standard clock time of midpoint of period without decimals (hr)
                              #       e.g., 12:45 am = 45; 3:15 pm = 1515
#LONGdeg = Longitude of weather station (degree component)
                              #LONGmin = Longitude of weather station (minute component)
#LATdeg = Latitude of weather station (degree component)
                              #LATmin = Latitude of weather station (minute component)
#Lz = Longitude of center of time zone (degrees West of Greenwich)
                              #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
#       respectively; Lz = 0 (Greenwich and Lz = 330 (Cairo).

                              #Variable definitions internal to compuing Solarazimuth
# pi As Double    #pi
                              # Longr As Double  #Longitude of weather station (radians)
# Latr As Double  #Latitude of weather station (radians)
                              # Lzr As Double   #Longitude of center of time zone (radians West of Greenwich)
## Tdec As Double  #Raw time converted to decimal time (hr)
# H As Double #Solar time angle at midpoint of hourly or shorter period (rad)
                              # d As Double #Solar declination (rad)
# X As Double     #Arc cosine function not available in VB

                              #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
pi = 3.14159265358979
Longr = pi / 180 * (LONGdeg + LONGmin / 60)  #(rad)
                              Latr = pi / 180 * (LATdeg + LATmin / 60)  #(rad)
Lzr = pi / 180 * Lz  #(rad)

                              #Convert raw time to decimal time
#Tdec = Int(t / 100) + (Right(t, 2)) / 60

                              #Compute solar noon
# b As Double #Used to compute equation of time (radians)
                              # Teq As Double   #Seasonal correction for solar time, or equation of time (hour)
# Tsn As Double #Solar noon, no daylight savings time correction (decimal hours)
                              b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
Teq = 0.1645 * Sin(2 * b) - 0.1255 * Cos(b) - 0.025 * Sin(b)    #(hr), FAO 56, p.48, eq.32
                              Tsn = 12 + (12 / pi * (Longr - Lzr) - Teq)    #(hr), Evett (2001), p.A-138, eq.5.8

#Find solar time angle in radians
                              H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7

#Compute solar declination
                              d = 0.409 * Sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24

#Compute X and Y components of sun azimuth
                              #http://www.usc.edu/dept/architecture/mbs/tools/vrsolar/Help/solar_concepts.html#Azimuth
#(accessed 12/05/2007)
                              # Xazm As Double
                              # Yazm As Double
                              # Azm As Double
                              Xazm = Sin(H) * Cos(d)
                              Yazm = Cos(Latr) * Sin(d) - Sin(Latr) * Cos(H) * Cos(d)
                              Azm = 180 / pi * Atn(Xazm / Yazm)

                              #Specify azimuth as South of East
if Tdec < Tsn Then
    if Azm < 0 Then
        Solarazimuth = -90 - Azm
    Else
        Solarazimuth = 90 - Azm
    
Else
    if Azm < 0 Then
        Solarazimuth = 90 - Azm
    Else
        Solarazimuth = 270 - Azm
    



#def Rso to compute clear sky radiation (W/m2)
                              def Rso(DOY As Integer, t As Integer, Ta As Double, HP As Double, Pmea As Double, _
                                           LONGdeg As Double, LONGmin As Double, LATdeg As Double, LATmin As Double, Lz As Double, _
                                           Ele As Double, T1 As Double, Gsc As Double, RsOpt As Integer, kt As Double, _
                                           Popt As Integer, eaOPT As Integer) As Double

                              #Rso = Clear sky solar radiation (W/m2)
#DOY = Julian Day of Year
                              #t = Standard clock time of midpoint of period without decimals (hr)
#       e.g., 12:45 am = 45; 3:15 pm = 1515
                              #Ta = Air temperature (C)
#HP = Humidity parameter, depends on eaOPT
                              #Pmea = Measured barometric pressure (kPa)
#Popt = Option to specify whether barometric pressure (P) is computed (from elevation) or measured
                              #   1 = P is computed based on elevation only (FAO 56, p.31, eq.7)
#   2 = P is measured
                              #LONGdeg = Longitude of weather station (degree component)
#LONGmin = Longitude of weather station (minute component)
                              #LATdeg = Latitude of weather station (degree component)
#LATmin = Latitude of weather station (minute component)
                              #Lz = Longitude of center of time zone (degrees West of Greenwich)
#       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
                              #       respectively; Lz = 0 (Greenwich and Lz = 330 (Cairo).
#Ele = Elevation above mean sea level (m)
                              #t1 = Length of timestep (hr)
#Gsc = Solar constant, usually 1367 W/m2
                              #RsoOpt = Option to specify Rso calcultaion method:
#   1 = Simple model based only on elevation
                              #   2 = Beer#s Law (accounts for P, turbidity, and path length)
                              #   3 = Accounts for P, atm moisture, turbidity, and path length)
#Kt = Turbidity coefficient (Kt = 1 recommended, ASCE-EWRI, 1-24-02, p.D7) (unitless)
                              #eaOPT = Option to specify whether dewpoint temperature (Tdew, C) or relative humidity (RH, %)
#           is used to compute actual vapor pressure of the air (ea, kPa)
                              #           eaOPT = 1: RH (%) is used
#           eaOPT = 2: Twet (%) is used
                              #           eaOPT = 3: Tdew (%) is used

#Variable definitions internal to compuing Ra
                              # pi As Double    #pi
# Longd As Double #Longitude of weather station (decimal degrees)
                              # Latr As Double  #Latitude of weather station (radians)
# Tdec As Double  #Raw time converted to decimal time (hr)

                              # b As Double #Used to compute equation of time (radians)
# Teq As Double   #Seasonal correction for solar time, or equation of time (hour)
                              # Tsn As Double   #Solar noon (hour)
# H As Double #Solar time angle at midpoint of hourly or shorter period (rad)
                              # H1 As Double #Solar time angle at beginning of measurement period (rad)
# H2 As Double #Solar time angle at ending of measurement period (rad)

                              # d As Double #Solar declination (rad)
# X As Double     #Arc cosine function not available in VB
                              # Hs As Double #Solar time angle at sunset (-Hs = sunrise) (rad)

# DR As Double    #Inverse relative earth-sun distance (dimensionless)
                              # ra As Double    #Top of Atmosphere (TOA) solar radiation (W/m2)

#Variable definitions internal to compuing Rso
                              # BP As Double    #Barometric pressure (kPa)
# SinPhi As Double #Sine of Phi, where Phi is the angle of the sun above the horizon
                              # ea As Double    #Actual vapor pressure (kPa)
# es As Double    #Saturated vapor pressure of the air (kPa)
                              # W As Double #Precipitable water in the atmosphere (mm)
# Kb As Double    #The clearness index for direct beam radiation (unitless)
                              # Kd As Double    #The corresponding index for diffuse beam radiation (unitless)

# RH As Double    #Relative humidity (%)
                              # Twet As Double  #Wet bulb temperature (C)
# Tdew As Double  #Dew point temperature (C)
                              # apsy As Double  #Psychrometer coefficient
# gammapsy As Double  #Psychrometer constant

                              #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
pi = 3.14159265358979
Longd = (LONGdeg + LONGmin / 60)   #(decimal degrees)
                              Latr = pi / 180 * (LATdeg + LATmin / 60)  #(rad)

#Convert raw time to decimal time; shift period endpoint to period midpoint by delaying 0.5*t1
                              Tdec = Int(t / 100) + ((t / 100) - Int(t / 100)) * 100 / 60 #- 0.125 * t1

#Find beginning and ending solar times of measurement period in radians
                              b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
Teq = 0.1645 * Sin(2 * b) - 0.1255 * Cos(b) - 0.025 * Sin(b)    #(hr), FAO 56, p.48, eq.32
                              Tsn = 12 + 4 / 60 * (Longd - Lz) - Teq #(hr), Evett (2001), p.A-138, eq.5.8
H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7
                              H1 = H - pi * T1 / 24 #(rad), FAO 56, p.48, eq.29
H2 = H + pi * T1 / 24 #(rad), FAO 56, p.48, eq. 30

                              #Compute sunset hour angle
d = 0.409 * Sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24
                              X = 1 - ((Tan(Latr)) ^ 2) * ((Tan(d)) ^ 2)  #FAO 56, p.47, eq.27
if X <= 0 Then X = 0.00001
Hs = pi / 2 - Atn((-(Tan(Latr)) * (Tan(d))) / (X ^ 0.5)) #(rad), FAO 56, p.47, eq.26

                              #Set integration limits for Ra equation when sunrise or sunset fall within computation period,
#or force Ra zero when sun is below horizon
                              if H1 < -Hs Then H1 = -Hs #ASCE-EWRI, eq. 55, p. 39
if H2 < -Hs Then H2 = -Hs
if H1 > Hs Then H1 = Hs
if H2 > Hs Then H2 = Hs
if H1 > H2 Then H1 = H2

#Compute extraterrestrial radiation (Ra)
                              DR = 1 + 0.033 * Cos(2 * pi * DOY / 365) #(dimensionless), FAO 56, p.46, eq.23
ra = 24 / (2 * pi * T1) * Gsc * DR * ((H2 - H1) * Sin(Latr) * Sin(d) + Cos(Latr) * Cos(d) * (Sin(H2) - Sin(H1))) #(MJ/m2/t1), FAO 56, p.47, eq.28

                              #*******************************************************************************
#Compute Rso

                              #Rso Options (1, 2, or 3)

#OPTION 1: SIMPLE
                              if RsOpt = 1 Then
                              Rso = (0.75 + (2 * 10 ^ -5) * Ele) * ra #(MJ/m2/t1), FAO 56, p.51, eq.37
GoTo 10

Else    #Compute P and sinphi, which are required for Options 2 and 3

                              #Determine P (either computed or measured)
if Popt = 1 Then
BP = 101.3 * (((293 - 0.0065 * Ele) / 293) ^ 5.26) #FAO 56, p.31, eq.7
                              Else
                              BP = Pmea
                              

                              #Compute sinphi, the sine of sun angle above horizon (path length)
SinPhi = Sin(Latr) * Sin(d) + Cos(Latr) * Cos(d) * Cos(H)     #FAO 56, p.226, eq.(3-15), minimum  = 0.10
                              if SinPhi < 0.1 Then SinPhi = 0.1

                              #OPTION 2: BEER#S LAW
                              if RsOpt = 2 Then
                                  Rso = ra * Exp((-0.0018 * BP) / (kt * SinPhi))    #(MJ/m2/t1), FAO 56, p.226, eq.(3-14)
GoTo 10
Else

#OPTION 3: Atm moisture and turbidity
                              if RsOpt = 3 Then
                                  if eaOPT = 1 Then
                                  RH = HP
                                  es = 0.61078 * Exp((17.269 * Ta) / (237.3 + Ta))
                                  ea = es * RH / 100
                                  Else
                                  if eaOPT = 2 Then
                                      Twet = HP
                                      apsy = 0.000662 #For aspirated psychrometers, FAO 56 p. 38
        gammapsy = BP * apsy
        es = 0.61078 * Exp((17.269 * Twet) / (237.3 + Twet))
        ea = es - gammapsy * (Ta - Twet)
    Else
        Tdew = HP
        ea = 0.61078 * Exp((17.269 * Tdew) / (237.3 + Tdew))
    
    
    W = 0.14 * ea * BP + 2.1 #FAO 56, p.227, eq.(3-19)
                                  Kb = 0.98 * Exp(((-0.00146 * BP) / (kt * SinPhi)) - 0.075 * ((W / SinPhi) ^ 0.4)) #FAO 56, p.227, eq.(3-18), slightly modified in ASCE-EWSI, 1-24-02, p.D-7, eq.(D.2)
    if Kb >= 0.15 Then   #FAO 56, p.227, eq.(3-20), slightly modified in ASCE-EWSI, 1-24-02, p.D-8, eq.(D.4)
                                      Kd = 0.35 - 0.36 * Kb
                                  Else
                                      Kd = 0.18 + 0.82 * Kb
                                  
                                  Rso = (Kb + Kd) * ra    #(MJ/m2/t1), FAO 56, p.227, eq.(3-17)





10 if Rso < 0 Then Rso = 0



#def Rsob to compute clear sky BEAM radiation (W/m2)
                              def Rsob(DOY As Integer, t As Integer, Ta As Double, HP As Double, Pmea As Double, _
                                            LONGdeg As Double, LONGmin As Double, LATdeg As Double, LATmin As Double, Lz As Double, _
                                            Ele As Double, T1 As Double, Gsc As Double, RsOpt As Integer, kt As Double, _
                                            Popt As Integer, eaOPT As Integer) As Double

                              #Rsob = Clear sky BEAM solar radiation (W/m2)
#DOY = Julian Day of Year
                              #t = Standard clock time of midpoint of period without decimals (hr)
#       e.g., 12:45 am = 45; 3:15 pm = 1515
                              #Ta = Air temperature (C)
#HP = Humidity parameter, depends on eaOPT
                              #Pmea = Measured barometric pressure (kPa)
#Popt = Option to specify whether barometric pressure (P) is computed (from elevation) or measured
                              #   1 = P is computed based on elevation only (FAO 56, p.31, eq.7)
#   2 = P is measured
                              #LONGdeg = Longitude of weather station (degree component)
#LONGmin = Longitude of weather station (minute component)
                              #LATdeg = Latitude of weather station (degree component)
#LATmin = Latitude of weather station (minute component)
                              #Lz = Longitude of center of time zone (degrees West of Greenwich)
#       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
                              #       respectively; Lz = 0 (Greenwich and Lz = 330 (Cairo).
#Ele = Elevation above mean sea level (m)
                              #t1 = Length of timestep (hr)
#Gsc = Solar constant, usually 1367 W/m2
                              #RsoOpt = Option to specify Rso calcultaion method:
#   1 = Simple model based only on elevation
                              #   2 = Beer#s Law (accounts for P, turbidity, and path length)
                              #   3 = Accounts for P, atm moisture, turbidity, and path length)
#Kt = Turbidity coefficient (Kt = 1 recommended, ASCE-EWRI, 1-24-02, p.D7) (unitless)
                              #eaOPT = Option to specify whether dewpoint temperature (Tdew, C) or relative humidity (RH, %)
#           is used to compute actual vapor pressure of the air (ea, kPa)
                              #           eaOPT = 1: RH (%) is used
#           eaOPT = 2: Twet (%) is used
                              #           eaOPT = 3: Tdew (%) is used

#Variable definitions internal to compuing Ra
                              # pi As Double    #pi
# Longd As Double #Longitude of weather station (decimal degrees)
                              # Latr As Double  #Latitude of weather station (radians)
# Tdec As Double  #Raw time converted to decimal time (hr)

                              # b As Double #Used to compute equation of time (radians)
# Teq As Double   #Seasonal correction for solar time, or equation of time (hour)
                              # Tsn As Double   #Solar noon (hour)
# H As Double #Solar time angle at midpoint of hourly or shorter period (rad)
                              # H1 As Double #Solar time angle at beginning of measurement period (rad)
# H2 As Double #Solar time angle at ending of measurement period (rad)

                              # d As Double #Solar declination (rad)
# X As Double     #Arc cosine function not available in VB
                              # Hs As Double #Solar time angle at sunset (-Hs = sunrise) (rad)

# DR As Double    #Inverse relative earth-sun distance (dimensionless)
                              # ra As Double    #Top of Atmosphere (TOA) solar radiation (W/m2)

#Variable definitions internal to compuing Rso
                              # BP As Double    #Barometric pressure (kPa)
# SinPhi As Double #Sine of Phi, where Phi is the angle of the sun above the horizon
                              # ea As Double    #Actual vapor pressure (kPa)
# es As Double    #Saturated vapor pressure of the air (kPa)
                              # W As Double #Precipitable water in the atmosphere (mm)
# Kb As Double    #The clearness index for direct beam radiation (unitless)
                              # Kd As Double    #The corresponding index for diffuse beam radiation (unitless)

# RH As Double    #Relative humidity (%)
                              # Twet As Double  #Wet bulb temperature (C)
# Tdew As Double  #Dew point temperature (C)
                              # apsy As Double  #Psychrometer coefficient
# gammapsy As Double  #Psychrometer constant

                              #Convert longitudes and latitude from degrees-minutes to decimal degrees and radians, respectively
pi = 3.14159265358979
Longd = (LONGdeg + LONGmin / 60)   #(decimal degrees)
                              Latr = pi / 180 * (LATdeg + LATmin / 60)  #(rad)

#Convert raw time to decimal time; shift period endpoint to period midpoint by delaying 0.5*t1
                              Tdec = Int(t / 100) + ((t / 100) - Int(t / 100)) * 100 / 60 #- 0.5 * t1

#Find beginning and ending solar times of measurement period in radians
                              b = 2 * pi * (DOY - 81) / 364     #FAO 56, p.48, eq.33
Teq = 0.1645 * Sin(2 * b) - 0.1255 * Cos(b) - 0.025 * Sin(b)    #(hr), FAO 56, p.48, eq.32
                              Tsn = 12 + 4 / 60 * (Longd - Lz) - Teq #(hr), Evett (2001), p.A-138, eq.5.8
H = 2 * pi / 24 * (Tdec - Tsn) #(rad), Evett (2001), p.A-138, eq.5.7
                              H1 = H - pi * T1 / 24 #(rad), FAO 56, p.48, eq.29
H2 = H + pi * T1 / 24 #(rad), FAO 56, p.48, eq. 30

                              #Compute sunset hour angle
d = 0.409 * Sin(2 * pi * DOY / 365 - 1.39) #(rad), FAO 56, p.46, eq.24
                              X = 1 - ((Tan(Latr)) ^ 2) * ((Tan(d)) ^ 2)  #FAO 56, p.47, eq.27
if X <= 0 Then X = 0.00001
Hs = pi / 2 - Atn((-(Tan(Latr)) * (Tan(d))) / (X ^ 0.5)) #(rad), FAO 56, p.47, eq.26

                              #Set integration limits for Ra equation when sunrise or sunset fall within computation period,
#or force Ra zero when sun is below horizon
                              if H1 < -Hs Then H1 = -Hs #ASCE-EWRI, eq. 55, p. 39
if H2 < -Hs Then H2 = -Hs
if H1 > Hs Then H1 = Hs
if H2 > Hs Then H2 = Hs
if H1 > H2 Then H1 = H2

#Compute extraterrestrial radiation (Ra)
                              DR = 1 + 0.033 * Cos(2 * pi * DOY / 365) #(dimensionless), FAO 56, p.46, eq.23
ra = 24 / (2 * pi * T1) * Gsc * DR * ((H2 - H1) * Sin(Latr) * Sin(d) + Cos(Latr) * Cos(d) * (Sin(H2) - Sin(H1))) #(MJ/m2/t1), FAO 56, p.47, eq.28

                              #*******************************************************************************
#Compute Rso

                              #Rso Options (1, 2, or 3)

#OPTION 1: SIMPLE
                              if RsOpt = 1 Then
                              Rsob = (0.75 + (2 * 10 ^ -5) * Ele) * ra #(MJ/m2/t1), FAO 56, p.51, eq.37
GoTo 10

Else    #Compute BP and sinphi, which are required for Options 2 and 3

                              #Determine P (either computed or measured)
if Popt = 1 Then
BP = 101.3 * (((293 - 0.0065 * Ele) / 293) ^ 5.26) #FAO 56, p.31, eq.7
                              Else
                              BP = Pmea
                              

                              #Compute sinphi, the sine of sun angle above horizon (path length)
SinPhi = Sin(Latr) * Sin(d) + Cos(Latr) * Cos(d) * Cos(H)     #FAO 56, p.226, eq.(3-15), minimum  = 0.10
                              if SinPhi < 0.1 Then SinPhi = 0.1

                              #OPTION 2: BEER#S LAW
                              if RsOpt = 2 Then
                                  Rsob = ra * Exp((-0.0018 * BP) / (kt * SinPhi))    #(MJ/m2/t1), FAO 56, p.226, eq.(3-14)
GoTo 10
Else

#OPTION 3: Atm moisture and turbidity
                              if RsOpt = 3 Then
                                  if eaOPT = 1 Then
                                  RH = HP
                                  es = 0.61078 * Exp((17.269 * Ta) / (237.3 + Ta))
                                  ea = es * RH / 100
                                  Else
                                  if eaOPT = 2 Then
                                      Twet = HP
                                      apsy = 0.000662 #For aspirated psychrometers, FAO 56 p. 38
        gammapsy = BP * apsy
        es = 0.61078 * Exp((17.269 * Twet) / (237.3 + Twet))
        ea = es - gammapsy * (Ta - Twet)
    Else
        Tdew = HP
        ea = 0.61078 * Exp((17.269 * Tdew) / (237.3 + Tdew))
    
    
    W = 0.14 * ea * BP + 2.1 #FAO 56, p.227, eq.(3-19)
                                  Kb = 0.98 * Exp(((-0.00146 * BP) / (kt * SinPhi)) - 0.075 * ((W / SinPhi) ^ 0.4)) #FAO 56, p.227, eq.(3-18), slightly modified in ASCE-EWSI, 1-24-02, p.D-7, eq.(D.2)

#    if Kb >= 0.15 Then   #FAO 56, p.227, eq.(3-20), slightly modified in ASCE-EWSI, 1-24-02, p.D-8, eq.(D.4)
#        Kd = 0.35 - 0.36 * Kb
                              #    Else
#        Kd = 0.18 + 0.82 * Kb
                              #    
#    Rso = (Kb + Kd) * Ra    #(MJ/m2/t1), FAO 56, p.227, eq.(3-17)

Rsob = Kb * ra  #(MJ/m2/t1)

                              
                              
                              

                              10 if Rsob < 0 Then Rsob = 0

                              

                              def Kbeam(Rs As Double, A As Double, b As Double, DOY As Integer, t As Integer, _
                                             Ta As Double, HP As Double, Pmea As Double, LONGdeg As Double, LONGmin As Double, _
                                             LATdeg As Double, LATmin As Double, Lz As Double, Ele As Double, T1 As Double, _
                                             Gsc As Double, RsOpt As Integer, kt As Double, Popt As Integer, eaOPT As Integer) As Double

                              #def Kbeam to calculate the fraction of direct beam irradiance in the visible
#(VIS) or near infrared (NIR) spectra, where VIS or NIR is determined by two
                              #empirical constants that were calibrated using unshadowed and shadowband irradiance
#measurements in the global and VIS spectra at the Bushland Weather Pen.

                              #Rs = Global shortwave irradiance (W m-2)
#A = Empirical constant used to calculate the fraction of direct beam irriadiance
                              #       in the visible spectra (no units)
#B = Empirical constant used to calculate the fraction of direct beam irriadiance
                              #       in the near infrared spectra (no units)
#C = Empirical constant used to calculate the fraction of direct beam irriadiance
                              #       in the near infrared spectra (no units)
#D = Empirical constant used to calculate the fraction of direct beam irriadiance
                              #       in the visible spectra (no units)
#DOY = Julian Day of Year
                              #t = Standard clock time of midpoint of period without decimals (hr)
#       e.g., 12:45 am = 45; 3:15 pm = 1515

                              #Ta = Air temperature (C)
#HP = Humidity parameter, depends on eaOPT
                              #Pmea = Measured barometric pressure (kPa)

#LONGdeg = Longitude of weather station (degree component)
                              #LONGmin = Longitude of weather station (minute component)

#LATdeg = Latitude of weather station (degree component)
                              #LATmin = Latitude of weather station (minute component)
#Lz = Longitude of center of time zone (degrees West of Greenwich)
                              #       75, 90, 105, 120 degrees for Eastern, Central, Mountain, and Pacific,
#       respectively; Lz = 0 (Greenwich and Lz = 330 (Cairo).
                              #Ele = Elevation above mean sea level (m)
#t1 = Length of timestep (hr)

                              #Gsc = Solar constant, usually 1367 W/m2
#RsoOpt = Option to specify Rso calcultaion method:
                              #   1 = Simple model based only on elevation
#   2 = Beer#s Law (accounts for P, turbidity, and path length)
#   3 = Accounts for P, atm moisture, turbidity, and path length)
                              #Kt = Turbidity coefficient (Kt = 1 recommended, ASCE-EWRI, 1-24-02, p.D7) (unitless)
#Popt = Option to specify whether barometric pressure (P) is computed (from elevation) or measured
                              #   1 = P is computed based on elevation only (FAO 56, p.31, eq.7)
#   2 = P is measured
                              #eaOPT = Option to specify whether dewpoint temperature (Tdew, C) or relative humidity (RH, %)
#           is used to compute actual vapor pressure of the air (ea, kPa)
                              #           eaOPT = 1: RH (%) is used
#           eaOPT = 2: Twet (%) is used
                              #           eaOPT = 3: Tdew (%) is used

if Rs < 1 Then
Kbeam = 0
GoTo 100


# Rsobeam As Double   #Theoretical clear sky direct beam irradiance (W m-2)
                              # Rsoglobal As Double #Theoretical clear sky global irradiance (W m-2)

Rsobeam = Rsob(DOY, t, Ta, HP, Pmea, LONGdeg, LONGmin, LATdeg, LATmin, _
Lz, Ele, T1, Gsc, RsOpt, kt, Popt, eaOPT)

Rsoglobal = Rso(DOY, t, Ta, HP, Pmea, LONGdeg, LONGmin, LATdeg, LATmin, _
Lz, Ele, T1, Gsc, RsOpt, kt, Popt, eaOPT)

if Rsoglobal < 1 Then
Kbeam = 0
GoTo 100


Kbeam = (Rsobeam / Rsoglobal) * A * (Rs / Rsoglobal) ^ b

100                            
                              

                              

                              
