#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
import cv2
import cv2.ml
from time import sleep
from datetime import datetime
import pysolar
import os
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib as mpl
from pandas import read_csv, read_excel, DataFrame
from skimage.feature import hessian_matrix_det as Hessian
from skimage.feature import local_binary_pattern as LBP
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import logging
logging.basicConfig(level=logging.INFO)


p0 = os.path.join('../../','raw','CookIRCamET','Images','CookHY2023')
p1 = os.path.join('../../','work','CookIRCamET','Images','CookHY2023','V1','TifPng','RGB')
p2 = os.path.join('../../','work','CookIRCamET','Images','CookHY2023','V1','TifPng')
p00 = os.path.join('../../','raw','CookIRCamET','Images','CprlHY2023')
p11 = os.path.join('../../','work','CookIRCamET','Images','CprlHY2023','V1','TifPng','RGB')
p22 = os.path.join('../../','work','CookIRCamET','Images','CprlHY2023','V1','TifPng')
p3 = os.path.join('../../','work','CookIRCamET','Working')

n_components3 = 8

def localSD(mat, n):    
    mat=np.float32(mat)
    mu = cv2.blur(mat,(n,n))
    mdiff=mu-mat
    mat2=cv2.blur(np.float64(mdiff*mdiff),(n,n))
    sd = np.float32(cv2.sqrt(mat2))
    
    return sd


# In[ ]:


f_imgs=[]
imgs=[]
n_img=0
for di,do in zip([p1,p11],[p2,p22]):
    fs=os.listdir(di)
    shuffle(fs)
    for f in fs:
        if 'bgr' in f:

            f_imgs = np.append(f_imgs,f)
            bgr = cv2.imread(os.path.join(di,f),cv2.IMREAD_UNCHANGED)
            time_place = f.split('_bgr.')[0].split('_')
            if len(time_place)==3 or 'nofix' in f:#V1

                labels = False
                f_labels = os.path.join(do,'SunShade',f.split('_bgr')[0]+'_class2.tif')
                if (os.path.exists(f_labels)):
                    labels1 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
                    labels = True

                f_labels = os.path.join(do,'SoilResVegSnow',f.split('_bgr')[0]+'_class4.tif')
                if (os.path.exists(f_labels)):
                    labels2 = cv2.imread(f_labels,cv2.IMREAD_UNCHANGED)
                    labels = (True & labels)
                if labels: 
                     #8-class
                    labels3 = 4*labels1+labels2
                    if not os.path.exists(os.path.join(do,'Masks')): os.mkdir(os.path.join(do,'Masks'))
                    cv2.imwrite(os.path.join(do,'Masks',f.split('_bgr')[0]+'_class8.tif'),labels3)
                    print(f)
                    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    l,a,bb = cv2.split(lab)
                    h,s,v = cv2.split(hsv)

                    sd_l1 = localSD(l, 127)
                    sd_l2 = localSD(l, 63)
                    sd_l3 = localSD(l, 31)

                    lbp_l1 = LBP(l, 32, 4, method='ror')
                    lbp_l2 = LBP(l, 24, 3, method='ror')
                    lbp_l3 = LBP(l, 16, 2, method='ror')

                    sd_a1 = localSD(a, 127)
                    sd_a2 = localSD(a, 63)
                    sd_a3 = localSD(a, 31)

                    lbp_a1 = LBP(a, 32, 4, method='ror')
                    lbp_a2 = LBP(a, 24, 3, method='ror')
                    lbp_a3 = LBP(a, 16, 2, method='ror')

                    sd_b1 = localSD(bb, 127)
                    sd_b2 = localSD(bb, 63)
                    sd_b3 = localSD(bb, 31)

                    lbp_b1 = LBP(bb, 32, 4, method='ror')
                    lbp_b2 = LBP(bb, 24, 3, method='ror')
                    lbp_b3 = LBP(bb, 16, 2, method='ror')

                    sd_h1 = localSD(h, 127)
                    sd_h2 = localSD(h, 63)
                    sd_h3 = localSD(h, 31)

                    lbp_h1 = LBP(h, 32, 4, method='ror')
                    lbp_h2 = LBP(h, 24, 3, method='ror')
                    lbp_h3 = LBP(h, 16, 2, method='ror')

                    sd_s1 = localSD(s, 127)
                    sd_s2 = localSD(s, 63)
                    sd_s3 = localSD(s, 31)

                    lbp_s1 = LBP(s, 32, 4, method='ror')
                    lbp_s2 = LBP(s, 24, 3, method='ror')
                    lbp_s3 = LBP(s, 16, 2, method='ror')

                    sd_v1 = localSD(v, 127)
                    sd_v2 = localSD(v, 63)
                    sd_v3 = localSD(v, 31)

                    lbp_v1 = LBP(v, 32, 4, method='ror')
                    lbp_v2 = LBP(v, 24, 3, method='ror')
                    lbp_v3 = LBP(v, 16, 2, method='ror')

                    ddepth = cv2.CV_16S

                    lap_l1 = cv2.Laplacian(l,ddepth,ksize=3)
                    lap_l2 = cv2.Laplacian(l,ddepth,ksize=7)
                    lap_l3 = cv2.Laplacian(l,ddepth,ksize=15)

                    lap_a1 = cv2.Laplacian(a,ddepth,ksize=3)
                    lap_a2 = cv2.Laplacian(a,ddepth,ksize=7)
                    lap_a3 = cv2.Laplacian(a,ddepth,ksize=15)

                    lap_b1 = cv2.Laplacian(bb,ddepth,ksize=3)
                    lap_b2 = cv2.Laplacian(bb,ddepth,ksize=7)
                    lap_b3 = cv2.Laplacian(bb,ddepth,ksize=15)

                    lap_h1 = cv2.Laplacian(h,ddepth,ksize=3)
                    lap_h2 = cv2.Laplacian(h,ddepth,ksize=7)
                    lap_h3 = cv2.Laplacian(h,ddepth,ksize=15)

                    lap_s1 = cv2.Laplacian(s,ddepth,ksize=3)
                    lap_s2 = cv2.Laplacian(s,ddepth,ksize=7)
                    lap_s3 = cv2.Laplacian(s,ddepth,ksize=15)

                    lap_v1 = cv2.Laplacian(v,ddepth,ksize=3)
                    lap_v2 = cv2.Laplacian(v,ddepth,ksize=7)
                    lap_v3 = cv2.Laplacian(v,ddepth,ksize=15)

                    img_size = l.shape
                    bb = bb.ravel()
                    a = a.ravel()
                    l = l.ravel()
                    h = h.ravel()
                    s = s.ravel()
                    v = v.ravel()
                    sd_l1 = sd_l1.ravel()
                    sd_l2 = sd_l2.ravel()
                    sd_l3 = sd_l3.ravel()
                    lbp_l1 = lbp_l1.ravel()
                    lbp_l2 = lbp_l2.ravel()
                    lbp_l3 = lbp_l3.ravel()
                    lap_l1 = lap_l1.ravel()
                    lap_l2 = lap_l2.ravel()
                    lap_l3 = lap_l3.ravel()
                    sd_a1 = sd_a1.ravel()
                    sd_a2 = sd_a2.ravel()
                    sd_a3 = sd_a3.ravel()
                    lbp_a1 = lbp_a1.ravel()
                    lbp_a2 = lbp_a2.ravel()
                    lbp_a3 = lbp_a3.ravel()
                    lap_a1 = lap_a1.ravel()
                    lap_a2 = lap_a2.ravel()
                    lap_a3 = lap_a3.ravel()
                    sd_b1 = sd_b1.ravel()
                    sd_b2 = sd_b2.ravel()
                    sd_b3 = sd_b3.ravel()
                    lbp_b1 = lbp_b1.ravel()
                    lbp_b2 = lbp_b2.ravel()
                    lbp_b3 = lbp_b3.ravel()
                    lap_b1 = lap_b1.ravel()
                    lap_b2 = lap_b2.ravel()
                    lap_b3 = lap_b3.ravel()
                    sd_h1 = sd_h1.ravel()
                    sd_h2 = sd_h2.ravel()
                    sd_h3 = sd_h3.ravel()
                    lbp_h1 = lbp_h1.ravel()
                    lbp_h2 = lbp_h2.ravel()
                    lbp_h3 = lbp_h3.ravel()
                    lap_h1 = lap_h1.ravel()
                    lap_h2 = lap_h2.ravel()
                    lap_h3 = lap_h3.ravel()
                    sd_s1 = sd_s1.ravel()
                    sd_s2 = sd_s2.ravel()
                    sd_s3 = sd_s3.ravel()
                    lbp_s1 = lbp_s1.ravel()
                    lbp_s2 = lbp_s2.ravel()
                    lbp_s3 = lbp_s3.ravel()
                    lap_s1 = lap_s1.ravel()
                    lap_s2 = lap_s2.ravel()
                    lap_s3 = lap_s3.ravel()
                    sd_v1 = sd_v1.ravel()
                    sd_v2 = sd_v2.ravel()
                    sd_v3 = sd_v3.ravel()
                    lbp_v1 = lbp_v1.ravel()
                    lbp_v2 = lbp_v2.ravel()
                    lbp_v3 = lbp_v3.ravel()
                    lap_v1 = lap_v1.ravel()
                    lap_v2 = lap_v2.ravel()
                    lap_v3 = lap_v3.ravel()
                    feat = np.vstack((l.T,a.T,bb.T,h.T,s.T,v.T,
                                      sd_l1.T,sd_l2.T,sd_l3.T,
                                      lbp_l1.T,lbp_l2.T,lbp_l3.T,
                                      lap_l1.T,lap_l2.T,lap_l3.T,
                                      sd_a1.T,sd_a2.T,sd_a3.T,
                                      lbp_a1.T,lbp_a2.T,lbp_a3.T,
                                      lap_a1.T,lap_a2.T,lap_a3.T,
                                      sd_b1.T,sd_b2.T,sd_b3.T,
                                      lbp_b1.T,lbp_b2.T,lbp_b3.T,
                                      lap_b1.T,lap_b2.T,lap_b3.T,
                                      sd_h1.T,sd_h2.T,sd_h3.T,
                                      lbp_h1.T,lbp_h2.T,lbp_h3.T,
                                      lap_h1.T,lap_h2.T,lap_h3.T,
                                      sd_s1.T,sd_s2.T,sd_s3.T,
                                      lbp_s1.T,lbp_s2.T,lbp_s3.T,
                                      lap_s1.T,lap_s2.T,lap_s3.T,
                                      sd_v1.T,sd_v2.T,sd_v3.T,
                                      lbp_v1.T,lbp_v2.T,lbp_v3.T,
                                      lap_v1.T,lap_v2.T,lap_v3.T)).T
                    #labels = np.sum(np.vstack((soil.ravel().T, residue.ravel().T*2, shadow.ravel().T*3, vegetation.ravel().T*4)).T,axis=1)
                    labels1 = labels1.ravel()        
                    labels2 = labels2.ravel() 
                    labels3 = labels3.ravel() 

                    if not np.any(np.isnan(feat)):
                        imgs.append({'bgr':bgr,'feats':feat,'labels1':labels1,'labels2':labels2,'labels3':labels3})
                        n_img=n_img+1

                    del lab, hsv, img ,l , a, bb, h, s, v, sd_l1,sd_l2,sd_l3,lbp_l1,lbp_l2,lbp_l3,sd_a1,sd_a2,sd_a3,lbp_a1,lbp_a2,lbp_a3, sd_b1,sd_b2,sd_b3,lbp_b1,lbp_b2,lbp_b3, sd_h1,sd_h2,sd_h3,lbp_h1,lbp_h2,lbp_h3,sd_s1,sd_s2,sd_s3,lbp_s1,lbp_s2,lbp_s3, sd_v1,sd_v2,sd_v3,lbp_v1,lbp_v2,lbp_v3, lap_l1,lap_l2,lap_l3,lap_a1,lap_a2, lap_a3, lap_b1, lap_b2,lap_b3,lap_h1,lap_h2,lap_h3,lap_s1,lap_s2,lap_s3,lap_v1,lap_v2,lap_v3

n_feat = feat.shape[1]


# In[ ]:


feats = []
labels1 = []
labels2 = []
labels3 = []
for sample in imgs:
    feats.append(sample['feats'])
    labels1.append(sample['labels1'])
    labels2.append(sample['labels2'])
    labels3.append(sample['labels3'])

feats = np.array(feats).reshape((-1,n_feat)).astype(np.float32)
labels1 = np.array(labels1).reshape((-1,1)).astype(np.int32).ravel()
labels2 = np.array(labels2).reshape((-1,1)).astype(np.int32).ravel()
labels3 = np.array(labels3).reshape((-1,1)).astype(np.int32).ravel()
scaler = StandardScaler()
filename = os.path.join(p3,'scaler_mlp.pk.sav')
#with joblib.parallel_backend('dask', wait_for_workers_timeout=60):
feats = scaler.fit_transform(feats)
pickle.dump(scaler, open(filename, 'wb'))


train_feats3, test_feats3, train_labels3, test_labels3 = train_test_split(feats, labels3, test_size=0.2, random_state=42)

def cornfusion(obs,pred,nclass):
    M = np.zeros((nclass,nclass))
    for i in range(obs.shape[0]):
        o = obs[i]
        p = pred[i]
        M[o,p] = M[o,p]+1
    return M

train_feats = train_feats3#[:,mask3]
test_feats = test_feats3#[:,mask3]

layers = []

for layer1 in range(1,11):
    for layer2 in range(1,6):
        layer = (layer1*n_feat, int(np.sqrt(n_feat*n_components3*layer1*layer2)), layer2*n_components3)
        layers.append(layer)

parameters = {'activation':('relu','logistic'),'hidden_layer_sizes':layers}

mlp = MLPClassifier(max_iter=100000,random_state=42)
clf_mlp3 = HalvingGridSearchCV(mlp, parameters,n_jobs=-1,cv=5)

clf_mlp3.fit(train_feats, train_labels3)

model_mlp3 = clf_mlp3.best_estimator_
pred_mlp3 = clf_mlp3.predict(test_feats)

M_mlp3 = cornfusion(test_labels3,pred_mlp3,n_components3)
M_mlp3 = M_mlp3/np.sum(np.sum(M_mlp3))

plt.matshow(M_mlp3)
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.title("V1 Confusion Matrix")
plt.savefig(os.path.join(p3,'m_v1.png'),dpi=300)

recall3 = np.diag(M_mlp3)/np.sum(M_mlp3,axis=1)
precis3 = np.diag(M_mlp3)/np.sum(M_mlp3,axis=0)

f3=(recall3*precis3/(recall3+precis3)*2)
f3_weighted=np.sum(f3*np.sum(M_mlp3, axis=1))

print(f3)
print(f3_weighted)

filename = os.path.join(p3,'finalized_model3_mlp.pk.sav')
pickle.dump(model_mlp3, open(filename, 'wb'))

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
M_mlp3_df.to_csv(os.path.join(p3,'M3_mlp.csv'))

p3_df = DataFrame(clf_mlp3.best_params_)
p3_df.to_csv(os.path.join(p3,'params3_mlp.csv'))





