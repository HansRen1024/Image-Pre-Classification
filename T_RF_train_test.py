#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:09:08 2018

@author: hans
"""

import cv2
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skimage import transform
import datetime

def hist_feature_gray(list_txt):
    root_path = 'image/'
    with open(list_txt, 'r') as f:
        line = f.readline()
        img = cv2.imread(root_path+line.split(' ')[0], 0)
        img=transform.resize(img, (227, 227,3))
        img = img.astype(np.uint8)
        feature = cv2.calcHist([img],[0],None,[256],[0,256]).reshape(1,-1)
        label = np.array([int(line.split(' ')[1].split('\n')[0])])
        line = f.readline()
        num = 2
        while line:
            img_path = os.path.join(root_path,line.split(' ')[0])
            if not os.path.isfile(img_path):
                line = f.readline()
                continue
            print("%d dealing with %s ..." %(num, line.split(' ')[0]))
            img = cv2.imread(img_path, 0)
            img=transform.resize(img, (227, 227,3))
            img = img.astype(np.uint8)
            hist_cv = cv2.calcHist([img],[0],None,[256],[0,256]).reshape(1,-1)
            feature = np.vstack((feature,hist_cv))
            label = np.hstack((label,np.array([int(line.split(' ')[1].split('\n')[0])])))
            num+=1
            line = f.readline()
    joblib.dump(feature, list_txt.split('.')[0]+filename,compress=5)
    joblib.dump(label, list_txt.split('.')[0]+'_label.pkl', compress=5)
    return feature, label

def hist_feature_rgb(list_txt):
    root_path = 'image/'
    with open(list_txt, 'r') as f:
        line = f.readline()
        img_path = os.path.join(root_path,line.split(' ')[0])
        img = cv2.imread(img_path)
        img=transform.resize(img, (227, 227,3))
        img = img.astype(np.uint8)
        b = img[:,:,0]*255
        g = img[:,:,1]*255
        r = img[:,:,2]*255
        feature_b = cv2.calcHist([b],[0],None,[256],[0,256]).reshape(1,-1)
        feature_g = cv2.calcHist([g],[0],None,[256],[0,256]).reshape(1,-1)
        feature_r = cv2.calcHist([r],[0],None,[256],[0,256]).reshape(1,-1)
        feature = np.hstack((feature_b,feature_g,feature_r))
        label = np.array([int(line.split(' ')[1].split('\n')[0])])
        line = f.readline()
        num = 2
        while line:
            img_path = os.path.join(root_path,line.split(' ')[0])
            if not os.path.isfile(img_path):
                line = f.readline()
                continue
            print("%d dealing with %s ..." %(num, line.split(' ')[0]))
            img = cv2.imread(img_path)
            img=transform.resize(img, (227, 227,3))
            img = img.astype(np.uint8)
            b = img[:,:,0]*255
            g = img[:,:,1]*255
            r = img[:,:,2]*255
            feature_b = cv2.calcHist([b],[0],None,[256],[0,256]).reshape(1,-1)
            feature_g = cv2.calcHist([g],[0],None,[256],[0,256]).reshape(1,-1)
            feature_r = cv2.calcHist([r],[0],None,[256],[0,256]).reshape(1,-1)
            hist_cv = np.hstack((feature_b,feature_g,feature_r))
            feature = np.vstack((feature,hist_cv))
            label = np.hstack((label,np.array([int(line.split(' ')[1].split('\n')[0])])))
            num+=1
            line = f.readline()
    joblib.dump(feature, list_txt.split('.')[0]+filename,compress=5)
    joblib.dump(label, list_txt.split('.')[0]+'_label.pkl', compress=5)
    return feature, label

def save_feature():
    t1 = datetime.datetime.now()
    if mode==0:
        X_train, y_train = hist_feature_gray(train_list)
        t2 = datetime.datetime.now()
        X_test, y_test = hist_feature_gray(test_list)
    elif mode==1:
        X_train, y_train = hist_feature_rgb(train_list)
        t2 = datetime.datetime.now()
        X_test, y_test = hist_feature_rgb(test_list)
    t3 = datetime.datetime.now()
    print("\ntime of extracting train features: %0.2f"%(t2-t1).total_seconds())
    print("time of extracting test features: %0.2f"%(t3-t2).total_seconds())

def train_model():
    t3 = datetime.datetime.now()
    X_train = joblib.load(train_list.split('.')[0]+filename)
    y_train = joblib.load(train_list.split('.')[0]+'_label.pkl')
    #train & save
    
#    from sklearn import tree
#    t = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
#    t = t.fit(X_train, y_train)
#    X_test = joblib.load(test_list.split('.')[0]+filename)
#    y_test = joblib.load(test_list.split('.')[0]+'_label.pkl')
#    t_pred = t.predict(X_test)
#    print t.score(X_test, y_test)
#    print t_pred[:100]
    
    # criterion: 分支的标准(gini/entropy), n_estimators: 树的数量, bootstrap: 是否随机有放回, n_jobs: 可并行运行的数量
    rf = RandomForestClassifier(n_estimators=25,criterion='entropy',bootstrap=True,n_jobs=4,random_state=80) # 随机森林
    rf = rf.fit(X_train, y_train)
    joblib.dump(rf, 'rf_model.pkl')
    t4 = datetime.datetime.now()
    print("time of training model: %0.2f"%(t4-t3).total_seconds())
    scores = cross_val_score(rf, X_train, y_train,scoring='accuracy' ,cv=3)
    print("Train Cross Avg. Score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    
def test_model():
    global rf_score
    t4 = datetime.datetime.now()
    #load data
    X_test = joblib.load(test_list.split('.')[0]+filename)
    y_test = joblib.load(test_list.split('.')[0]+'_label.pkl')
    #load model & test
    rf_model = joblib.load('rf_model.pkl')
    rf_score = rf_model.score(X_test, y_test)
    rf_pred = rf_model.predict(X_test)
    print rf_pred[:10]
    t5 = datetime.datetime.now()
    print("time of testing model: %0.2f"%(t5-t4).total_seconds())
    scores = cross_val_score(rf_model, X_test, y_test,scoring='accuracy' ,cv=10)
    print("Test Cross Avg. Score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


mode=1
if mode==0:
    filename = '_feature_gray.pkl'
elif mode==1:
    filename = '_feature_rgb.pkl'

train_list = "train_all.txt"
test_list = "test_all.txt"

#train_list = "train.txt"
#test_list = "test.txt"

if __name__ == '__main__':
    
#    save_feature()
    
    train_model()

    test_model()
    
    
