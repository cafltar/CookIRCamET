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

files = os.listdir(p)

for f in files:
    when = datetime.strftime(f.split('_')[0])
    lon = f.split('_')[1]
    lat = f.split('_')[1]
    if 'bgr' in f:
        bgr = cv2.imread(os.path.join(p,f))
    if 'ir' in f:
        ir = cv2.imread(os.path.join(p,f))
    
