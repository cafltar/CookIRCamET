import sys
import numpy as np
import cv2
import time
from time import sleep
from datetime import datetime
import os
import numpy as np

from pylepton.Lepton3 import Lepton3
from picamera import PiCamera
import logging 

def gpscapture(GPS,ts):
    logging.info('GPS start %f',time.monotonic())
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
                gpsstring="{0:3.6f}_{1:3.6f}".format(GPS.longitude,GPS.latitude)
                timestring="{}{}{}{:02}{:02}{:02}".format(GPS.timestamp_utc.tm_year,  # Grab parts of the time from the
                                                     GPS.timestamp_utc.tm_mon,  # struct_time object that holds
                                                     GPS.timestamp_utc.tm_mday,  # the fix time.  Note you might
                                                     GPS.timestamp_utc.tm_hour,  # not get all data like year, day,
                                                     GPS.timestamp_utc.tm_min,  # month!
                                                     GPS.timestamp_utc.tm_sec)
                logging.info(timestring)
                logging.info('GPS end %f',time.monotonic())
                return gpsstring, timestring
               
        m+=1
        time.sleep(ts)
    logging.debug('GPS end %f',time.monotonic())
    gpsstring="{0:3.6f}_{1:3.6f}".format(999.999999,999.999999)
    timestring="99999999999999"
    logging.info(timestring)
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


