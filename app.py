#!/usr/bin/env python3
#import asyncio
#from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import cv2
from utils import bgrcapture, ircapture, gpscapture
import serial
from time import sleep
from datetime import datetime, timezone
import os
import numpy as np
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=10)

from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request

from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.DEBUG)
gpsPath='/home/pi/adagps_mod'
logging.info(gpsPath)
sys.path.insert(0,gpsPath)

import adafruit_gps

home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Images')
web = os.path.join(home,'CookIRCamET','static')
ry,rx=960,768
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

#modified functions for Pool
def bgrpcapture():
      return bgrcapture(ry,rx)

def smap(f):
      return f()

def capture():
      now = datetime.now(timezone.utc)
      current_time = now.strftime("%Y%m%d%H%M%S")
      current_spot0,current_time_fix = gpscapture(gps,ts)
      #current_spot0 = None
      #current_time_fix = None

      if current_spot0:
            current_spot = current_spot0
      else:
            current_spot = 'nofix'
      if current_time_fix:
            #yes! gps fix
            current_time=current_time_fix
      pool = Pool(processes=2)
      res = pool.map(smap,[bgrpcapture,ircapture])

      sleep(5)

      pool.close()
      pool.terminate()
      pool.join()
      
      print(res[0].shape)
      try:
            for r in res:
                  if r.shape[2]==1:
                        #r = ircapture()
                        fname = current_time+'_'+current_spot+'_ir.png'
                        logging.info(fname)
                        cv2.imwrite(os.path.join(p,fname),r)
                        cv2.normalize(r, r, 0, 65535, cv2.NORM_MINMAX)
                        np.right_shift(r, 8, r)
                        cv2.imwrite(os.path.join(web,'bar.bmp'),np.uint8(r))
                  else:
                        #r = bgrcapture(ry,rx)
                        cv2.imwrite(os.path.join(web,'foo.bmp'),r)
                        fname = current_time+'_'+current_spot+'_bgr.png'
                        logging.info(fname)
                        cv2.imwrite(os.path.join(p,fname),r)
      except Exception as e:
            logging.info(e)
            sys.exit(1)
      return current_time,current_spot

@app.route('/' )
def index():
      current_time, current_spot = capture()
      
      return render_template('camera.html', time=current_time+' '+current_spot)

if __name__ == '__main__':      
      app.run(debug=False,host='0.0.0.0')
      
