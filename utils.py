import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
from pylepton.Lepton3 import Lepton3
from picamera import PiCamera
#import RPi.GPIO as GPIO
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(13, GPIO.OUT, initial=1)
import logging

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
