#!/usr/bin/env python3
import sys
from utils2 import *
from time import sleep
import os

from threading import Thread
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':     
      if len(sys.argv)>1:
            waittime = int(sys.argv[1])
            cam_name = str(sys.argv[2])
      else:
            waittime = 3600
            cam_name = "default"
      flir = ir_cam(waittime, cam_name)
      hq = bgr_cam(ry,rx,exp_time,frame_time,brightness,contrast,waittime,cam_name)

      try:
            while True:
                  r = hq.capture()
                  ir = flir.capture()
                  sleep(waittime)
            #t1 = Thread(target = hq.run())
            #t2 = Thread(target = flir.run())
            #t1.start()
            #t2.start()
      except:
            #t1.join()
            #t2.join()
            del hq, flir
      
      
      
