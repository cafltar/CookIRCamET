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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from utils_segmentation import *

import pickle
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    version = sys.argv[1]
    model = sys.argv[2]
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
                filepath = os.path.join(di,f)
                bgr = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
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

                    feat = get_features(bgr)
                    labels1 = labels1.ravel()        
                    labels2 = labels2.ravel() 
                    labels3 = labels3.ravel() 

                    if not np.any(np.isnan(feat)):
                        imgs.append({'bgr':bgr,'feats':feat,'labels1':labels1,'labels2':labels2,'labels3':labels3})
                        n_img=n_img+1

    n_feat = feat.shape[1]

    feats_raw = []
    labels3 = []
    for sample in imgs:
        feats_raw.append(sample['feats'])
        labels3.append(sample['labels3'])
    del imgs

    feats_raw = np.array(feats_raw).reshape((-1,n_feat)).astype(np.float32)
    labels3 = np.array(labels3).reshape((-1,1)).astype(np.int32).ravel()
    train_feats, test_feats, train_labels, test_labels = train_test_split(feats_raw, labels3, test_size=0.2, random_state=42)

    #Pipeline
    #initial scaling
    scaler = StandardScaler()
    #Best parameter (CV score=0.892):
    #{'clf__activation': 'logistic', 'clf__hidden_layer_sizes': (432, 117, 32), 'pca__n_components': 0.999}

    if version=='V1':
        #pca
        pca = PCA(svd_solver='full',n_components=0.999)
        clf = MLPClassifier(max_iter=1000,hidden_layer_sizes=[432,117,32],activation='logistic')
        pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("clf", clf)])
        pipeline.fit(train_feats,train_labels)
        filename = os.path.join(p3,'model_pipeline_'+version+'_'+model+'_final.pk.sav')
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb'
            pickle.dump(pipeline.best_estimator_, f)

        pred = pipeline.predict(test_feats)

    elif version=='V2':
        #pca
        pca = PCA(svd_solver='full')
        clf = MLPClassifier(max_iter=1000)
        if model == 'mlp':
            #mlp
            clf = MLPClassifier(max_iter=1000)
            #tune hyperparameters
            layers = []

            for layer1 in [2,4,8,16]:
                for layer2 in [2,4]:
                    layer = (layer1*n_feat//2, int(np.sqrt(n_feat//2*n_components*layer1*layer2)), layer2*n_components)
                    layers.append(layer)

            parameters = {'pca__n_components':(0.99,0.999),'clf__activation':('relu','logistic'),'clf__hidden_layer_sizes':layers}

        elif model == 'hgbc':
            clf = HistGradientBoostingClassifier(max_iter=1000)

            parameters = {'pca__n_components':(0.99,0.999),'clf__min_samples_leaf':(20,200,2000,20000),'clf__max_bins':(31,63,127,255)}

        pipeline = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("clf", clf)])

        search = HalvingGridSearchCV(pipeline, parameters,n_jobs=-1,cv=5)

        search.fit(train_feats, train_labels)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        filename = os.path.join(p3,'model_pipeline_'+version+'_'+model+'_final.pk.sav')
        with open(filename, 'wb') as f:  # Python 3: open(..., 'wb'
            pickle.dump(search.best_estimator_, f)
            
        pred = search.predict(test_feats)
        
    M,f,a = cornfusion(test_labels,pred,n_components)

    plt.matshow(M)
    plt.ylabel("Predicted")
    plt.xlabel("Observed")
    plt.title(version+" Confusion Matrix")
    plt.savefig(os.path.join(p,'m_'+version+'_'+model+'_final.png'),dpi=300)

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
    M_df.to_csv(os.path.join(p3,'M_'+version+'_'+model+'_final.csv'))

    p_df = DataFrame(search.best_params_)
    p_df.to_csv(os.path.join(p3,'params_'+version+'_'+model+'_final.csv'))
