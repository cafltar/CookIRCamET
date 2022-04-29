import sys
import numpy as np
import cv2
import time
from time import sleep
from datetime import datetime
import os
import numpy as np
from scipy import constants
from scipy import special

from pylepton.Lepton3 import Lepton3
from picamera import PiCamera
import logging 

pi = constants.pi
g = constants.g #m/s2
boltz = constants.Stefan_Boltzmann# W/K4/m2
vonk = 0.4 #Von Karman
Tk = 273.16 #0 C
rho_a = 1.205 #kg/m3
Gsc = 1367 #W/m2
lam = 2450 #kJ/kg
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# heat capacity of mixed air at constant pressure (J kg-1 K-1)
c_p = 1013

Cx = 90#Empirical constant  s m-1

b = 0.012#Empirical constant = 0.012 or b = 0.012[1 + cos(psiw)],
#       where psiw is wind direction, ccw from north
c = 0.0025#Empirical constant = 0.0025
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# gas constant for dry air, J/(kg*degK)
R_d = 287.04

#Monin-Obukhov parameters
dhc = .67#displacement height/canopy height C&N98, K95 say 0.65
zomhc = 0.125#momemtun roughness over canopy height K95
sc = 1#heat roughness over momentum roughness C&N say .2, K95 say exp(-2)

#Measurement heights (speed and temp)
zu = 2
zt = 2

#reference height if taken on separate height
hcref = np.nan

#bare soil roughness
#zombs = .4e-3 #m
#zoh_bs =

#soil heat flux
#GRnday = 0.1
#Grnnight = 0.5

#canopy stomatal resistance s/m
rcday = 50
rcnight = 200

#soil albedo vis/nir/wet/dry

#leave absorbance/emittance/reflectance

#leaf angle param


def gpscapture(GPS,ts):
    logging.info('GPS start %f',time.monotonic())
    gpsstring = None
    timestring = None    
    # Make sure to call gps.update() every loop iteration and at least twice
    # as fast as data comes from the GPS unit.
    # This returns a bool that's true if it parsed new data (you can ignore it
    # though if you don't care and instead look at the has_fix property).
    m=0
    while gpsstring is None and m<10:
        GPS.update()
        time.sleep(ts)
        if GPS.has_fix and GPS.update():
            gpsstring="{0:3.6f}_{1:3.6f}".format(GPS.longitude,GPS.latitude)
            timestring="{}{}{}{:02}{:02}{:02}".format(GPS.timestamp_utc.tm_year,  # Grab parts of the time from the
                                                      GPS.timestamp_utc.tm_mon,  # struct_time object that holds
                                                      GPS.timestamp_utc.tm_mday,  # the fix time.  Note you might
                                                      GPS.timestamp_utc.tm_hour,  # not get all data like year, day,
                                                      GPS.timestamp_utc.tm_min,  # month!
                                                      GPS.timestamp_utc.tm_sec)
        m = m+1
    logging.info('GPS end %f',time.monotonic())
    return gpsstring, timestring

def ircapture():
    device = "/dev/spidev0.0"
    t = 1
    n = 0
    m = 0
    with Lepton3(device) as l:
        while t>0.048 and m<10:
            a,_,t = l.capture()
            sleep(.185)
            #if n>10:
            m = m+1
            #  GPIO.output(13, 0)
            #  sleep(1)
            #  GPIO.output(13, 1)
            #  sleep(5)
            #  n = 0
            #n = n+1
            
    #cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
    #np.right_shift(a, 8, a)
    return a

def bgrcapture(ry,rx):
  with PiCamera(resolution = (ry,rx)) as camera:
    image = np.empty((ry*rx*3,),dtype=np.uint8)
    #should also fix shutter_speed, analog_gain,digital_gain, exposure_mode,awb_mode,awb_gains
    camera.start_preview()
    print(camera.exposure_speed,camera.iso,camera.awb_mode,camera.awb_gains)
    # Camera warm-up time
    sleep(2)
    camera.capture(image,'bgr')

    return image.reshape((rx,ry,3))


