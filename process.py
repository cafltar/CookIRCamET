import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np

from pandas import read_csv, read_excel, DataFrame
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import logging
logging.basicConfig(level=logging.INFO)
home = os.path.join("/home","pi")
p = os.path.join(home,'CookIRCamET','Images')
p2 = os.path.join(home,'CookIRCamET','Working')

n_components = 2

files = os.listdir(p)

for f in files:
    when = datetime.strftime(f.split('_')[0],'%Y%m%d%H%M%S')
    lon = float(f.split('_')[1])
    lat = float(f.split('_')[2])
    if 'bgr' in f:
        bgr = cv2.imread(os.path.join(p,f))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        img_size = h.shape()
        h = h.ravel()
        v = v.ravel()
        hv = np.hstack((h,v))
        classes = GaussianMixture(n_components=2).fit_predict(hsv.reshape(hv))
        classes = classes.unravel(img_size)
        fseg = f.split('_bgr')[0]+'_seg.png'
        cv2.imwrite(os.path.join(p2,fseg),classes)
    if 'ir' in f:
        ir = cv2.imread(os.path.join(p,f))

        
