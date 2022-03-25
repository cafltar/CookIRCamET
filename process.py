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
    when = datetime.strptime(f.split('_')[0],'%Y%m%d%H%M%S')
    lon = f.split('_')[1]
    lat = lon
    if lon != 'nofix':
        lon = float(f.split('_')[1])
        lat = float(f.split('_')[2])
    else:
        lon = np.nan
        lat = np.nan
        
    if 'bgr' in f:
        bgr = cv2.imread(os.path.join(p,f))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        img_size = h.shape
        h = h.ravel()
        v = v.ravel()
        hv = np.vstack((h.T,v.T)).T
        classes = GaussianMixture(n_components=2).fit_predict(hv)
        classes = classes.reshape(img_size)
        fseg = f.split('_bgr')[0]+'_seg.png'
        cv2.imwrite(os.path.join(p2,fseg),classes)
        logging.info(os.path.join(p2,fseg))
    if 'ir' in f:
        ir = cv2.imread(os.path.join(p,f))

        
