#!/usr/bin/env python3
import sys
import numpy as np
import cv2
from utils2 import *
from time import sleep
from datetime import datetime, timezone
import os
import numpy as np
from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request
from multiprocessing import Pool, Process

import logging

logging.basicConfig(level=logging.DEBUG)

#initialise app
#app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#
#@app.route('/' )
#def index():
      
#      now = datetime.now()
#      current_time = now.strftime("%Y%d%m%H%M%S")

#      return render_template('camera.html', time=current_time)

if __name__ == '__main__':     
      if len(sys.argv)>1:
            waittime = int(sys.argv[1])
      else:
            waittime = 60
      flir = ir_cam(waittime)
      hq = bgr_cam(ry,rx,exp_time,frame_time,brightness,contrast,waittime)

      try:
            while True:
                  r = hq.capture()
                  ir = flir.capture()
                  sleep(waittime)
      except:
            del hq, flir

      
      #app.config['waittime'] = waittime
      #q1=Process(target=hq.capture)
      #q2=Process(target=flir.capture)
      #q1.start()
      #q2.start()
      #app.run(debug=False,host='0.0.0.0')
      #q1.join()
      #q2.join()

      
