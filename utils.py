import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
from scipy import constants
from pylepton.Lepton3 import Lepton3
from picamera import PiCamera
from scipy import special
from pysolar.solar import get_azimuth, get_altitude
import logging

pi = constants.pi
boltz = constant.Stefan_Boltzmann_constant# W K-4 m-2
vonk = 0.4 #Von Karman
Tk = 273.16 #0 C
gammac = 0.000665 #psychometric constant
cp = 1005 #J/kg/K
rhoa = 1.205 #kg/m3

#measurements:
#Ta, P, RH
#Tc, Ts
#Rn
#time, lat, lon, z, altitude, aspect/slope?

#classes
class canopy():
    def __init__(self,alpha, fvsl_v_meas, fvsh_v_meas, fs_v_meas, theta_s,theta_v,phi_s,phi_v):
        self.alpha = alpha
        self.fvsl_v_meas = fvsl_v_meas
        self.fvsh_v_meas = fvsh_v_meas
        self.fs_v_meas = fs_v_meas
        self.theta_s = np.pi/2-np.deg2rad(theta_s)#elevation->zenith, deg->rad
        self.theta_v = np.pi/2-np.deg2rad(theta_v)
        self.phi_s = np.deg2rad(phi_s)
        self.phi_v = np.deg2rad(phi_v)
        self.n = len(phi_s)#number of measurements
    def Kb(self,x,theta,phi):
        return None
    def Kbe(self,x,theta,phi):
        return np.sqrt(x**2+np.tan(theta)**2)/(x+1.774*(x+1.182)**(-.733))
    def Kbt(self,x,theta,phi):
        return None
    def Lsl(self,x,theta,phi,L):
        return (1-np.exp(-self.Kbe(x,theta,phi)*L))/self.Kbe(x,theta,phi)
    def Lsh(self,x,theta,phi,L):
        return L-self.Lsl(x,theta,phi,L)
    def fsl(self,x,theta,phi,L):
        #sun angle
        return np.exp(-self.Kbe(x,theta,phi)*L)
    def fvsl_v(self,x,theta_s,phi_s,theta_v,phi_v,L):
        return self.fsl(x,theta_s,phi_s,L)*self.fv_v(x,theta_v,phi_v,L)
    def fvsh_v(self,x,theta_s,phi_s,theta_v,phi_v,L):
        return (1-self.fsl(x,theta_s,phi_s,L))*self.fv_v(x,theta_v,phi_v,L)
    def fs_v(self,x,theta,phi,L):
        #view angle
        return np.exp(-self.Kbe(x,theta,phi)*L)
    def fv_v(self,x,theta,phi,L):
        #view angle
        return 1-self.fs_v(x,theta,phi,L)

class air():
    def __init__():

class soil():
    def __init__():
    

def readGPS(GPS,ts):
    logging.debug('GPS start %f %s',time.monotonic(),time.asctime(GPS.timestamp_utc))
    lat = list()
    lng = list()
    gpstime = list()
    last_print = time.monotonic()
        
    # Make sure to call gps.update() every loop iteration and at least twice
    # as fast as data comes from the GPS unit.
    # This returns a bool that's true if it parsed new data (you can ignore it
    # though if you don't care and instead look at the has_fix property).
    m=0
    while m<10:
        GPS.update()
        if time.monotonic()-last_print>=ts:
            last_print=time.monotonic()
            if GPS.has_fix and GPS.update():
                gpsstring="{0:3.4f}_{1:3.4f}".format(GPS.longitude,GPS.latitude)
                timestring="Fix time: {}/{}/{} {:02}:{:02}:{:02}".format(GPS.timestamp_utc.tm_mon,  # Grab parts of the time from the
                                                     GPS.timestamp_utc.tm_mday,  # struct_time object that holds
                                                     GPS.timestamp_utc.tm_year,  # the fix time.  Note you might
                                                     GPS.timestamp_utc.tm_hour,  # not get all data like year, day,
                                                     GPS.timestamp_utc.tm_min,  # month!
                                                     GPS.timestamp_utc.tm_sec)
                logging.info(timestring)
                break
            else:
                gpsstring="{0:3.4f}_{1:3.4f}".format(999.9999,999.9999)
        m+=1
        time.sleep(ts)
        logging.debug('GPS end %f',time.monotonic())
    return gpsstring


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
            
    cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(a, 8, a)
    return np.uint8(a)

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


