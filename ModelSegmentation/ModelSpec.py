#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:46 2017

For modifying model (training) specifications and tuning parameters in the experiments

@author: xiaomuliu
"""
import numpy as np
from sklearn import linear_model, ensemble
from sklearn.model_selection import StratifiedKFold

def get_model_params(model_name, rand_seed=1234):
    if model_name=='logit_l2':
        fixed_params = dict(penalty='l2', solver='liblinear', n_jobs=1)
        model = linear_model.LogisticRegression(**fixed_params)    
        tuning_params = dict(C=np.logspace(-2, 2, 5))
    elif model_name=='GBM':
        fixed_params = dict(learning_rate=0.05, subsample=0.9, max_features=0.4, min_samples_leaf=1, random_state=rand_seed)
        model = ensemble.GradientBoostingClassifier(**fixed_params)
        tuning_params = dict(n_estimators=np.arange(100,250,50),max_depth=[3,4,5],min_samples_split=np.arange(100,700,200))
    # SVM, adaboost, random forest, ...    
        
    return dict(model=model,tuning_params=tuning_params)    
    
def get_cv_obj(kfolds=5, rand_seed=1234):    
    return StratifiedKFold(n_splits=kfolds, shuffle=False, random_state=rand_seed)    