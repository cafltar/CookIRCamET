#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import cv2
from time import sleep
from datetime import datetime
import pysolar
import os
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib as mpl
from pandas import read_csv, read_excel, DataFrame

from skimage.feature import local_binary_pattern as LBP
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
import pickle
import logging
logging.basicConfig(level=logging.INFO)

p0 = os.path.join('../../','raw','CookIRCamET','Images','CookHY2023')
p00 = os.path.join('../../','raw','CookIRCamET','Images','CprlHY2023')
p3 = os.path.join('../../','work','CookIRCamET','Working')

n_components3 = 8

def localSD(mat, n):    
    mat=np.float32(mat)
    mu = cv2.blur(mat,(n,n))
    mdiff=mu-mat
    mat2=cv2.blur(np.float64(mdiff*mdiff),(n,n))
    sd = np.float32(cv2.sqrt(mat2))
    
    return sd

def cornfusion(obs,pred,nclass):
    M = np.zeros((nclass,nclass))
    for i in range(obs.shape[0]):
        o = obs[i]
        p = pred[i]
        M[o,p] = M[o,p]+1
    correct = sum(obs==pred)
    total = len(pred)
    M = M/np.sum(np.sum(M))
    recall = np.diag(M)/np.sum(M,axis=1)
    precis = np.diag(M)/np.sum(M,axis=0)

    f1=(recall*precis/(recall+precis)*2)
    f1_weighted=np.sum(f1*np.sum(M, axis=1))

    return M, f1_weighted, correct/total


# In[ ]:

for version in ['V1','V2']:
    print(version)
    p1 = os.path.join('../../','work','CookIRCamET','Images','CookHY2023',version,'TifPng','RGB')
    p2 = os.path.join('../../','work','CookIRCamET','Images','CookHY2023',version,'TifPng')
    p11 = os.path.join('../../','work','CookIRCamET','Images','CprlHY2023',version,'TifPng','RGB')
    p22 = os.path.join('../../','work','CookIRCamET','Images','CprlHY2023',version,'TifPng')

    f_imgs=[]
    imgs=[]
    n_img=0
    for di,do in zip([p1,p11],[p2,p22]):
        fs=os.listdir(di)
        print(di)
        print(do)
        shuffle(fs)
        for f in fs:
            if 'bgr' in f:

                f_imgs = np.append(f_imgs,f)
                print(f)
                bgr = cv2.imread(os.path.join(di,f),cv2.IMREAD_UNCHANGED)
                time_place = f.split('_bgr.')[0].split('_')

                labels = False
                f_labels = os.path.join(do,'SunShade',f.split('_bgr')[0]+'_class2.tif')
                if (os.path.exists(f_labels)):
                    labels1 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
                    labels = True

                f_labels = os.path.join(do,'SoilResVegSnow',f.split('_bgr')[0]+'_class4.tif')
                if (os.path.exists(f_labels)):
                    labels2 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
                    labels2[labels2==4]=3#flowers->veg
                    labels = (True & labels)
                    if labels: 
                        #8-class
                        labels3 = 4*labels1+labels2
                        if not os.path.exists(os.path.join(do,'Masks')): os.mkdir(os.path.join(do,'Masks'))
                        cv2.imwrite(os.path.join(do,'Masks',f.split('_bgr')[0]+'_class8.png'),labels3)
                    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    l,a,bb = cv2.split(lab)
                    h,s,v = cv2.split(hsv)
                    b,g,r = cv2.split(bgr)
                    chan_funs = []
                    img_size = b.shape

                    ddepth = cv2.CV_16S

                    for c in [b,g,r,h,s,v,l,a,bb]:
                        chan_funs.append(c)
                        chan_funs.append(cv2.GaussianBlur(c,(15,15),cv2.BORDER_DEFAULT))
                        chan_funs.append(cv2.GaussianBlur(c,(31,31),cv2.BORDER_DEFAULT))
                        chan_funs.append(localSD(c, 127))
                        chan_funs.append(localSD(c, 63))
                        chan_funs.append(localSD(c, 31))
                        chan_funs.append(LBP(c, 32, 4, method='ror'))
                        chan_funs.append(LBP(c, 24, 3, method='ror'))
                        chan_funs.append(LBP(c, 16, 2, method='ror'))
                        chan_funs.append(cv2.Laplacian(c,ddepth,ksize=3))
                        chan_funs.append(cv2.Laplacian(c,ddepth,ksize=7))
                        chan_funs.append(cv2.Laplacian(c,ddepth,ksize=15))
                    ravels = []
                    for cf in chan_funs:
                        ravels.append(cf.ravel().T)
                    feat = np.vstack(ravels).T
                    labels1 = labels1.ravel()        
                    labels2 = labels2.ravel() 
                    labels3 = labels3.ravel() 

                    if not np.any(np.isnan(feat)):
                        imgs.append({'bgr':bgr,'feats':feat,'labels1':labels1,'labels2':labels2,'labels3':labels3})
                        n_img=n_img+1

                    del chan_funs, ravels, h,s,v,r,g,b,l,a,bb

    n_feat = feat.shape[1]



    feats_raw = []
    labels3 = []
    for sample in imgs:
        feats_raw.append(sample['feats'])
        labels3.append(sample['labels3'])
    del imgs

    feats_raw = np.array(feats_raw).reshape((-1,n_feat)).astype(np.float32)
    labels3 = np.array(labels3).reshape((-1,1)).astype(np.int32).ravel()

    #initial scaling
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats_raw)

    train_feats, test_feats, train_labels, test_labels = train_test_split(feats, labels3, test_size=0.2, random_state=42)

    #feature selection
    clf_mlp3 = MLPClassifier(max_iter=100000,random_state=42,hidden_layer_sizes=[180,84,40],activation='relu')
    sfs = SequentialFeatureSelector(clf_mlp3, n_features_to_select='auto',tol=0.001)
    sfs.fit(train_feats, train_labels)
    print(sfs.get_support())
    feats = sfs.transform(feats_raw)
    n_feats = feats.shape[1]

    #scale reduced data
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    filename = os.path.join(p3,'scaler_mlp_'+version+'_paper.pk.sav')
    pickle.dump(scaler, open(filename, 'wb'))

    train_feats, test_feats, train_labels, test_labels = train_test_split(feats, labels3, test_size=0.2, random_state=42)

    #tune hyperparameters
    layers = []

    for layer1 in range(1,11):
        for layer2 in range(1,6):
            layer = (layer1*n_feat, int(np.sqrt(n_feat*n_components3*layer1*layer2)), layer2*n_components3)
            layers.append(layer)

    parameters = {'activation':('relu','logistic'),'hidden_layer_sizes':layers}

    mlp = MLPClassifier(max_iter=100000,random_state=42)
    clf_mlp3 = HalvingGridSearchCV(mlp, parameters,n_jobs=-1,cv=5)

    clf_mlp3.fit(train_feats, train_labels3)
    filename = os.path.join(p3,'finalized_model3_mlp_'+version+'_final.pk.sav')
    pickle.dump(model_mlp3, open(filename, 'wb'))

    model_mlp3 = clf_mlp3.best_estimator_
    pred_mlp3 = clf_mlp3.predict(test_feats)

    M_mlp3,f3,a3 = cornfusion(test_labels3,pred_mlp3,n_components3)

    plt.matshow(M_mlp3)
    plt.ylabel("Predicted")
    plt.xlabel("Observed")
    plt.title(version+" Confusion Matrix")
    plt.savefig(os.path.join(p3,'m_'+version+'_final.png'),dpi=300)

    print(f3,a3)

    M_mlp3_df = {}
    M_mlp3_df['sun_soil'] = M_mlp3[:,0]
    M_mlp3_df['sun_res'] = M_mlp3[:,1]
    M_mlp3_df['sun_can'] = M_mlp3[:,2]
    M_mlp3_df['sun_snow'] = M_mlp3[:,3]
    M_mlp3_df['shade_soil'] = M_mlp3[:,4]
    M_mlp3_df['shade_res'] = M_mlp3[:,5]
    M_mlp3_df['shade_can'] = M_mlp3[:,6]
    M_mlp3_df['shade_snow'] = M_mlp3[:,7]
    M_mlp3_df = DataFrame(M_mlp3_df)
    M_mlp3_df.to_csv(os.path.join(p3,'M3_mlp_'+version+'_final.csv'))

    p3_df = DataFrame(clf_mlp3.best_params_)
    p3_df.to_csv(os.path.join(p3,'params3_mlp_'+version+'_final.csv'))





