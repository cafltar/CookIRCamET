import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import os
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib as mpl
from pandas import read_csv, read_excel, DataFrame
from time import perf_counter_ns
from skimage.feature import local_binary_pattern as LBP
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier,SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils_segmentation import get_features, p3, p4, p0, p00, n_components, plots, cornfusion

import pickle
import logging
logging.basicConfig(level=logging.INFO)

train_fraction = 0.8
model='pa_batch'
threshold="0.1*median"
datestr = datetime.strftime(datetime.now(),'%Y%m%d')
combine_all = True
if combine_all:
    linmod = PassiveAggressiveClassifier()
else:
    pass
for hy,version in zip(['2023','2023','2024'],['V1','V2','V3']):
    fs_list = []
    print(version)
    p1 = os.path.join('/90daydata/nsaru','work','CookIRCamET','Images','CookHY'+hy,version,'TifPng','RGB')
    p2 = os.path.join('/90daydata/nsaru','work','CookIRCamET','Images','CookHY'+hy,version,'TifPng')
    p11 = os.path.join('/90daydata/nsaru','work','CookIRCamET','Images','CprlHY'+hy,version,'TifPng','RGB')
    p22 = os.path.join('/90daydata/nsaru','work','CookIRCamET','Images','CprlHY'+hy,version,'TifPng')
    
    p1_list = [p1]
    p2_list = [p2]
    
    p1_list.append(p11)
    p2_list.append(p22)

    for di,do in zip(p1_list,p2_list):
        fs=os.listdir(di)
        print(di)
        print(do)
        [fs_list.append(os.path.join(di,f)) for f in fs]
    shuffle(fs_list)
    n_tot = len(fs_list)
    n_train = int(train_fraction*n_tot)
    n_test = n_tot-n_train
    
    if not combine_all:
        linmod = PassiveAggressiveClassifier()
    else:
        pass

    for filepath in fs_list:
        if 'bgr' in filepath:
            print(filepath)
            f = filepath.split('/')[-1]
            bgr = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
            di = filepath.split(f)[0]
            do = di.replace('RGB/','')
            print(f,di,do)
            labels = False
            f_labels = os.path.join(do,'SunShade',f.split('_bgr')[0]+'_class2.tif')
            if (os.path.exists(f_labels)):
               labels1 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
               labels = True

            f_labels = os.path.join(do,'SoilResVegSnow',f.split('_bgr')[0]+'_class4.tif')
            print(f_labels)
            if (os.path.exists(f_labels)):
                labels2 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
                labels2[labels2==4]=3#flowers->veg
                labels = (True & labels)
                if labels: 
                    #8-class
                    labels3 = 4*labels1+labels2
                    if not os.path.exists(os.path.join(do,'Masks')): os.mkdir(os.path.join(do,'Masks'))
                    cv2.imwrite(os.path.join(do,'Masks',f.split('_bgr')[0]+'_class8.png'),labels3)
 
                start = perf_counter_ns() 
                feat, chan_labs = get_features(bgr)
                labels1 = labels1.ravel()        
                labels2 = labels2.ravel() 
                labels3 = labels3.ravel() 
                print('Get features: {0:3.6f}s'.format((perf_counter_ns()-start)/10**9))

                n_feat = feat.shape[1]
                labels3 = labels3[np.isnan(np.sum(feat,axis=1))==False]
                feat = feat[np.isnan(np.sum(feat,axis=1))==False,:]
                start = perf_counter_ns() 
                linmod.partial_fit(feat,labels3,classes=[0,1,2,3,4,5,6,7])
                print('Partial fit: {0:3.6f}s'.format((perf_counter_ns()-start)/10**9))   

    
    if not combine_all:
        selector = SelectFromModel(linmod,threshold=threshold)
        keep = selector.transform(np.array(chan_labs).reshape(1,-1))
        print(keep)
        
        keep = selector.transform(np.arange(0,len(chan_labs),1).reshape(1,-1))

        filename = os.path.join(p3,'input_vars_'+version+'_'+model+'_'+datestr+'_reduced.pk.sav')
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb'
            pickle.dump(keep, f)

    else:
        pass

if combine_all:
    selector = SelectFromModel(linmod,threshold=threshold)
    keep = selector.transform(np.array(chan_labs).reshape(1,-1))
    print(keep)
    
    keep = selector.transform(np.arange(0,len(chan_labs),1).reshape(1,-1))

    filename = os.path.join(p3,'input_vars_all_'+model+'_'+datestr+'_reduced.pk.sav')
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb'
        pickle.dump(keep, f)
else:
    pass

if combine_all:    
    linmod = PassiveAggressiveClassifier()
    ss = StandardScaler()
    pipeline = Pipeline(steps=[("scaler", ss), ("linmod", linmod)])
    parameters = {'linmod__C':[0.0001,0.0003,0.001,0.003,0.01]}
    search = GridSearchCV(pipeline,parameters,cv=5)
        
    feats_list = []
    labels_list = []
    
for hy,version in zip(['2023','2023','2024'],['V1','V2','V3']):
    if not combine_all:    
        linmod = PassiveAggressiveClassifier()
        ss = StandardScaler()
        pipeline = Pipeline(steps=[("scaler", ss), ("linmod", linmod)])
        parameters = {'linmod__C':[0.0001,0.0003,0.001,0.003,0.01]}
        search = GridSearchCV(pipeline,parameters,cv=5)
    
        feats_list = []
        labels_list = []
    
    for filepath in fs_list:
        if 'bgr' in filepath:
            print(filepath)
            f = filepath.split('/')[-1]
            bgr = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
            di = filepath.split(f)[0]
            do = di.replace('RGB/','')
            
            labels = False
            f_labels = os.path.join(do,'Masks',f.split('_bgr')[0]+'_class8.png')
            if (os.path.exists(f_labels)):
                labels3 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
                
                start = perf_counter_ns() 
                feat, chan_labs = get_features(bgr,keep)
                labels3 = labels3.ravel() 
                print('Get features: {0:3.6f}s'.format((perf_counter_ns()-start)/10**9))

                n_feat = feat.shape[1]
                labels3 = labels3[np.isnan(np.sum(feat,axis=1))==False]
                feat = feat[np.isnan(np.sum(feat,axis=1))==False,:]       
                feats_list.append(feat)       
                labels_list.append(labels3)

    if not combine_all:
        print(feats_list[0].shape,feats_list[-1].shape)
        print(n_feat)
        n_feat = feats_list[0].shape[1]
        print(keep.shape)
        print(len(feats_list))
        print(len(labels_list))
        labels_list = [l.reshape(-1,1) for l in labels_list]
        print(np.vstack(feats_list).shape)   
        print(np.vstack(labels_list).shape)
        feats = np.vstack(feats_list).reshape((-1,n_feat)).astype(np.float32)
        labels = np.vstack(labels_list).astype(np.int32).ravel()
        print(feats.shape)
        print(labels.shape)
        
        train_feats, test_feats, train_labels, test_labels = train_test_split(feats, labels, test_size=(1-train_fraction), random_state=42)
    
        start = perf_counter_ns() 
        search.fit(train_feats,train_labels)
        pipeline = search.best_estimator_
        print('Fit: {0:3.6f}s'.format((perf_counter_ns()-start)/10**9))   
        
        print(chan_labs)              
        filename = os.path.join(p3,'model_pipeline_'+version+'_'+model+'_'+datestr+'_reduced.pk.sav')
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb'
            pickle.dump(pipeline, f)
    
        start = perf_counter_ns()
        pred = pipeline.predict(test_feats)
    
        M,f,a = cornfusion(test_labels,pred,n_components)
    
        print('Prediction time: {0:3.6f}s Accuracy: {1:3.6f}'.format((perf_counter_ns()-start)/10**9,a)) 
        print(search.best_params_)
        plt.matshow(M)
        plt.ylabel("Predicted")
        plt.xlabel("Observed")
        plt.title(version+" Confusion Matrix")
        plt.savefig(os.path.join(p3,'m_'+version+'_'+model+'_'+datestr+'_reduced.png'),dpi=300)
        
        print(f,a)
        
        M_df = {}
        M_df['sun_soil'] = M[:,0]
        M_df['sun_res'] = M[:,1]
        M_df['sun_can'] = M[:,2]
        M_df['sun_snow'] = M[:,3]
        M_df['shade_soil'] = M[:,4]
        M_df['shade_res'] = M[:,5]
        M_df['shade_can'] = M[:,6]
        M_df['shade_snow'] = M[:,7]
        M_df = DataFrame(M_df)
        M_df.to_csv(os.path.join(p3,'M_'+version+'_'+model+'_'+datestr+'_reduced.csv'))

if combine_all:
    print(feats_list[0].shape,feats_list[-1].shape)
    print(n_feat)
    n_feat = feats_list[0].shape[1]
    print(keep.shape)
    print(len(feats_list))
    print(len(labels_list))
    labels_list = [l.reshape(-1,1) for l in labels_list]
    print(np.vstack(feats_list).shape)   
    print(np.vstack(labels_list).shape)
    feats = np.vstack(feats_list).reshape((-1,n_feat)).astype(np.float32)
    labels = np.vstack(labels_list).astype(np.int32).ravel()
    print(feats.shape)
    print(labels.shape)
    
    train_feats, test_feats, train_labels, test_labels = train_test_split(feats, labels, test_size=(1-train_fraction), random_state=42)

    start = perf_counter_ns() 
    search.fit(train_feats,train_labels)
    pipeline = search.best_estimator_
    print('Fit: {0:3.6f}s'.format((perf_counter_ns()-start)/10**9))   
    
    print(chan_labs)              
    filename = os.path.join(p3,'model_pipeline_all_'+model+'_'+datestr+'_reduced.pk.sav')
    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb'
        pickle.dump(pipeline, f)

    start = perf_counter_ns()
    pred = pipeline.predict(test_feats)

    M,f,a = cornfusion(test_labels,pred,n_components)

    print('Prediction time: {0:3.6f}s Accuracy: {1:3.6f}'.format((perf_counter_ns()-start)/10**9,a)) 
    print(search.best_params_)
    plt.matshow(M)
    plt.ylabel("Predicted")
    plt.xlabel("Observed")
    plt.title(version+" Confusion Matrix")
    plt.savefig(os.path.join(p3,'m_all_'+model+'_'+datestr+'_reduced.png'),dpi=300)
    
    print(f,a)
    
    M_df = {}
    M_df['sun_soil'] = M[:,0]
    M_df['sun_res'] = M[:,1]
    M_df['sun_can'] = M[:,2]
    M_df['sun_snow'] = M[:,3]
    M_df['shade_soil'] = M[:,4]
    M_df['shade_res'] = M[:,5]
    M_df['shade_can'] = M[:,6]
    M_df['shade_snow'] = M[:,7]
    M_df = DataFrame(M_df)
    M_df.to_csv(os.path.join(p3,'M_all_'+model+'_'+datestr+'_reduced.csv'))
