#!/usr/bin/env python3
import sys
from utils2 import *
from utils_segmentation import register_ir
from pysolar import solarn
from time import sleep
from pandas import read_csv, DataFrame
import os
import cv2
import glob
import pickle
from threading import Thread
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__': 
    
    seg_model = pickle.load(open(os.path.join('Data/model_pipeline_V3_pa_batch_20241030_reduced.pk.sav'), 'rb'))
    input_vars = pickle.load(open(os.path.join('Data/input_vars_V3_pa_batch_20241030_reduced.pk.sav'), 'rb'))
    ir_cal = pickle.load(open(os.path.join('Data/calibration_nsar_lf_v3.pk.sav'), 'rb'))#V3
    #should read from file
    latlon = read_csv(open(os.path.join('latlon.csv')))
    lat = latlon.lat.values[0]
    lon = latlon.lon.values[0]
    if len(sys.argv)>1:
        waittime = int(sys.argv[1])
    else:
        waittime = 3600
    flir = ir_cam(waittime,save=False)
    hq = bgr_cam(ry,rx,exp_time,frame_time,brightness,contrast,waittime,save=False)
    #capture and run affine or read from files
    r = hq.capture()
    ir = flir.capture()
    #check if warp file exists
    if len(glob.glob(os.path.join('Data/*warp_mat*.pk')))>0:
        warp_mat = pickle.load(open(glob.glob(os.path.join('Data/*warp_mat*.pk'))[0], 'rb'))#
    else:
        _,_,v = cv2.split(cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV))
        warp_mat, _ = register_ir(ir,v.reshape(r.shape[0:2]),r,warp_mat=None)
    img = image_model(input_vars,seg_model,ir_cal,lat, lon,warp_mat=warp_mat)
    #if not, calculate affine
    #initialize file
    with open('oputput_data.csv','wb') as file:
        file.write('times,elevation,azimuth,fssun,fsshd,frsun,frshd,fvsun,fvshd,fwsun,fwshd,Tssun,Tsshd,Trsun,Trshd,Tvsun,Tvshd,Twsun,Twshd')
        file.write('\n')
        try: 
            labels_noon = np.nan
            noon_delta_old = 90
            while True:
                utc = datetim.strftime(datetime.now(timezone.utc)
                theta = solar.get_altitude(lat,lon,utc)
                phi = solar.get_azimuth(lat,lon,utc)
                day = (if theta > 10)
                noon_delta_new = np.abs(90-theta)
        
                bgr = hq.capture()
                ir = flir.capture()
                if noon_delta_new<noon_delta_old:
                    labels_noon = img.get_labels(self, bgr)
                else:
                    _ = img.get_labels(self, bgr)            
                if ~day: 
                    #use midday labels at night
                    #everything shadow
                    labels_noon[labels_noon==0]=4
                    labels_noon[labels_noon==1]=5
                    labels_noon[labels_noon==2]=6
                    labels_noon[labels_noon==3]=7
                    img.labels = labels_noon
                    
                f, T = img.get_temps(ir, bgr)
                data = datetime.strftime(utc,'%Y%m%d%H%M%S') + ', {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}, {0:2f4}'.format(theta, phi, *f, *T)
                file.write(data)
                file.write('\n')
                sleep(waittime)
        except Exception as e:
            print(e)
            del hq, flir

      
      


      
      
