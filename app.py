#!/usr/bin/env python3
#import asyncio
#from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import cv2
from utils import bgrcapture, ircapture, gpscapture
import serial
from time import sleep
from datetime import datetime
import os
import numpy as np
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=10)

from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request

from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO)
gpsPath='/home/pi/adagps_mod'
logging.info(gpsPath)
sys.path.insert(0,gpsPath)

import adafruit_gps

home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Images')
web = os.path.join(home,'CookIRCamET','static')
ry,rx=160,128#3840,2160
#initialise app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#Initialize GPS
ts=.1
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=10)
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
      now = datetime.now()
      current_time = now.strftime("%Y%d%m%H%M%S")
      current_spot = gpscapture(gps,ts)


      pool = Pool(processes=2)
      res = pool.map(smap,[bgrpcapture,ircapture,gpspcapture])

      sleep(5)

      pool.close()
      pool.terminate()
      pool.join()
      
      print(res[0].shape)
      try:
            for r in res:
                  if r.shape[2]==1:
                        #r = ircapture()
                        cv2.imwrite(os.path.join(web,'bar.bmp'),r)
                        fname = current_time+'_'+current_spot+'_ir.png'
                        logging.info(fname)
                        #cv2.imwrite(os.path.join(p,fname),r)
                  else:
                        #r = bgrcapture(ry,rx)
                        cv2.imwrite(os.path.join(web,'foo.bmp'),r)
                        fname = current_time+'_'+current_spot+'_bgr.png'
                        logging.info(fname)
                        #cv2.imwrite(os.path.join(p,fname),r)
      except Exception as e:
            logging.info(e)
            sys.exit(1)
      return current_time

@app.route('/' )
def index():
      now_str = capture()
      
      return render_template('camera.html', time=now_str)

if __name__ == '__main__':      
      app.run(debug=False,host='0.0.0.0')
      
