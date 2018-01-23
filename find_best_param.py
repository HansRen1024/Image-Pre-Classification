#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:24:32 2018

@author: hans
"""

from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

filename = '_feature_rgb.pkl'
train_list = "train_all.txt"
test_list = "test_all.txt"

def adaboost(n):
    clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='gini',max_depth=11, min_samples_split=400, \
                                                         min_samples_leaf=30,max_features=30,random_state=10), \
                             algorithm="SAMME", n_estimators=n, learning_rate=0.001,random_state=10)
    return clf

def findBestParam():
    X_train = joblib.load(train_list.split('.')[0]+filename)
    y_train = joblib.load(train_list.split('.')[0]+'_label.pkl')
    X_test = joblib.load(test_list.split('.')[0]+filename)
    y_test = joblib.load(test_list.split('.')[0]+'_label.pkl')
    best_test_score=0
    best_train_score=0
    best_param=0
    for n in range(1500,1501,10):
        clf = adaboost(n)
        clf = clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print ("--------------------------------\nCurrent train score: %.4f" %train_score)
        print ("Current test score: %.4f" %test_score)
        print ("Current param: %d" %n)
        if test_score > best_test_score:
            best_test_score = test_score
            best_train_score = train_score
            best_param = n
    print ("--------------------------------\nBest train score: %.4f" %best_train_score)
    print ("Best test score: %.4f" %best_test_score)
    print ("Best param: %d" %best_param)

if __name__ == '__main__':
    findBestParam()
