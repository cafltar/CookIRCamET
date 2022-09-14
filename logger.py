#!/usr/bin/env python3
#import asyncio
#from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import cv2
from utils import ry, rx, bgrcapture, ircapture, gpscapture
from time import sleep
from datetime import datetime, timezone
import os
import numpy as np
import serial
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=10)

#from pandas import read_csv, read_excel, DataFrame
#from sklearn.mixture import GaussianMixture
#from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request

from multiprocessing import Pool, Process

import logging

logging.basicConfig(level=logging.DEBUG)
gpsPath='/home/pi/adagps_mod'
logging.info(gpsPath)
sys.path.insert(0,gpsPath)

import adafruit_gps

home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Images')
web = os.path.join(home,'CookIRCamET','static')
lep = os.path.join(home,'LeptonModule','software','raspberrypi_capture')
#initialise app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#Initialize GPS
ts=1
gps = adafruit_gps.GPS(uart, debug=False)
# Turn on the basic GGA and RMC info (what you typically want)
gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
# Set update rate to timestep in ms
ms=str(int(1e3*ts))
gps.send_command(bytearray("PMTK220,"+ms,'utf-8'))

gps.update()

#for logger it's stationary
current_spot,current_time_fix = gpscapture(gps,ts)
#current_spot = None
if current_spot is None:
      current_spot = 'nofix'
      
def bgrpcapture():
      while True:
            r = bgrcapture(ry,rx)
            now = datetime.now(timezone.utc)
            current_time = now.strftime("%Y%m%d%H%M%S")
            cv2.imwrite(os.path.join(web,'foo.bmp'),r)
            fname = current_time+'_'+current_spot+'_bgr.png'
            logging.info(os.path.join(p,fname))
            cv2.imwrite(os.path.join(p,fname),r)
            sleep(app.config['waittime'])
      
def irpcapture():
      while True:
            r = ircapture()
            now = datetime.now(timezone.utc)
            current_time = now.strftime("%Y%m%d%H%M%S")
            fname = current_time+'_'+current_spot+'_ir.png'
            logging.info(os.path.join(p,fname))
            cv2.imwrite(os.path.join(p,fname),r)
            cv2.normalize(r, r, 0, 65535, cv2.NORM_MINMAX)
            np.right_shift(r, 8, r)
            cv2.imwrite(os.path.join(web,'bar.bmp'),np.uint8(r))
            sleep(app.config['waittime'])
def smap(f):
      return f()
            
@app.route('/' )
def index():
      
      now = datetime.now()
      current_time = now.strftime("%Y%d%m%H%M%S")

      return render_template('camera.html', time=current_time+' '+current_spot)

if __name__ == '__main__':     
      if len(sys.argv)>1:
            waittime = int(sys.argv[1])
      else:
            waittime = 1800

      app.config['waittime'] = waittime
      q1=Process(target=bgrpcapture)
      q2=Process(target=irpcapture)
      q1.start()
      q2.start()
      app.run(debug=False,host='0.0.0.0')
      q1.join()
      q2.join()

      
