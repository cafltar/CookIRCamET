#!/usr/bin/env python3
#import asyncio
#from concurrent.futures import ThreadPoolExecutor
import sys
import numpy as np
import cv2
from utils import bgrcapture, ircapture
from time import sleep
from datetime import datetime
import os
import numpy as np
#from pandas import read_csv, read_excel, DataFrame
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request

from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO)

home = os.path.join("/home","pi")
p = os.path.join(home,'Images')
web = os.path.join(home,'CafSensorPi','static')
ry,rx=160,128#3840,2160
#initialise app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def bgrpcapture():
      return bgrcapture(ry,rx)

def smap(f):
      return f()

def capture():
      now = datetime.now()
      current_time = now.strftime("%Y%d%m%H%M%S")

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
                        cv2.imwrite(os.path.join(web,'bar.bmp'),r)
                        fname = current_time+'_ir.bmp'
                        logging.info(fname)
                        #cv2.imwrite(os.path.join(p,fname),r)
                  else:
                        #r = bgrcapture(ry,rx)
                        cv2.imwrite(os.path.join(web,'foo.bmp'),r)
                        fname = current_time+'_bgr.bmp'
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
      
