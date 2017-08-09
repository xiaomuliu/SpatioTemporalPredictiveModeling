#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:43:54 2017

@author: xiaomuliu
"""

import numpy as np
#from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
#NOTE: scikit learn v0.18 changed 'cross_validation' module and 'grid_search' module to 'model_selection'
from sklearn.model_selection import GridSearchCV 

#def _SeqData_scaler(X, scaling='standardize', mean=None, sd=None):
#    # scaling Structured sequential data for LSTM models
#    # input X is of shape (#samples, time steps, #features)
#    # output y is of shape (#samples,)
#    # scaling applied w.r.t the 3rd dimension
#    if scaling=='standardize':
#        if mean is None:
#            mean = np.apply_over_axes(np.mean, X, [0,1])
#        if sd is None:
#            sd = np.apply_over_axes(np.std, X, [0,1])        
#    elif scaling=='minmax':
#        if mean is None:
#            mean = np.apply_over_axes(np.min, X, [0,1])  
#        if sd is None:
#            sd = (np.apply_over_axes(np.max, X, [0,1])-np.apply_over_axes(np.min, X, [0,1])) 
#            sd[sd==0] = 1 # if min and max values are the same
#            
#    X_scaled = (X-mean) / sd     
#    return X_scaled, mean, sd

    
#def fit_pred(X_train, y_train, X_test, clf, clf_name, tune_param_dict, CV_obj, scaling='standardize'):
#    """
#    Tune model('clf')'s parameters in tune_param_dicts 
#    using training data and cross validation. Then fit the entire training data
#    using the optimal parameters and predict the test data
#    """  
#    if scaling is not None:
#        X_train, m, sd  = _SeqData_scaler(X_train,scaling)        
#        X_test, _, _ = _SeqData_scaler(X_test,scaling, m, sd)
#        
#    pipe = Pipeline(steps=[(clf_name, clf)])
#    estimator = GridSearchCV(pipe, tune_param_dict, cv=CV_obj, scoring='roc_auc', n_jobs=1)
#    estimator.fit(X_train, y_train); 
#    if hasattr(clf,'predict_proba'):       
#        y_scores = estimator.best_estimator_.predict_proba(X_test)
#    elif hasattr(clf,'decision_function'):
#        y_scores = estimator.best_estimator_.decision_function(X_test)
#    
#    if y_scores.ndim>1:
#        y_scores = y_scores[:,1]  # Some classifier return scores for both classes
#    
#    return dict(pred_score=y_scores, CV_result=estimator.cv_results_, best_param=estimator.best_params_)

def model_fit(X_train, y_train, clf, clf_name, tune_param_dict, CV_obj, scaling='standardize'):
    """
    Tune model('clf')'s parameters in tune_param_dicts 
    using training data and cross validation. Then fit the entire training data
    using the optimal parameters 
    """ 
    if scaling is not None:
        if scaling=='standardize':
            scaler = preprocessing.StandardScaler()
        elif scaling=='minmax':
            scaler = preprocessing.MinMaxScaler() 
    pipe = Pipeline(steps=[('standardize',scaler), (clf_name, clf)]) if scaling is not None else \
           Pipeline(steps=[(clf_name, clf)])
    pipe = Pipeline(steps=[(clf_name, clf)])
    estimator = GridSearchCV(pipe, tune_param_dict, cv=CV_obj, scoring='roc_auc', n_jobs=1)
    estimator.fit(X_train, y_train); 
    
    return dict(model=estimator.best_estimator_, CV_result=estimator.cv_results_, best_param=estimator.best_params_)    
    
def model_pred(X_test, model):
    """
    """ 
    if hasattr(model,'predict_proba'):       
        y_scores = model.predict_proba(X_test)
    elif hasattr(model,'decision_function'):
        y_scores = model.decision_function(X_test)
    
    if y_scores.ndim>1:
        y_scores = y_scores[:,1]  # Some classifier return scores for both classes
    
    return y_scores


                 
if __name__ == '__main__':   
    import re
    import time
    import cPickle as pickle
    from ModelSpec import get_model_params, get_cv_obj
    import sys
    sys.path.append('..') 
    import StructureData.LoadData as ld 
    from Misc.ComdArgParse import ParseArg    
        
    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    group_pkl = re.search('(?<=group=)([\w\./]+)',infiles).group(1)
    train_data = re.search('(?<=traindata=)([\w\./]+)',infiles).group(1)
    test_data = re.search('(?<=testdata=)([\w\./\s]+)(?=cluster)',infiles).group(1) # File names are seperated by whitespace
    #test_data_list = re.findall('[\w\./]+',test_data.group(0))
    test_data_list = np.array(test_data.rstrip().split('\n'))
    # sort test data list by chunk No
    chunkNo_list = np.zeros(len(test_data_list)).astype(int)
    for idx, fn in enumerate(test_data_list):
        chunkNo_list[idx] = int(re.search('(?<=chunk)(\d+)',fn).group(1))
        
    new_idx = np.argsort(chunkNo_list)
    test_data_list = test_data_list[new_idx]

    cluster_pkl = re.search('(?<=cluster=)([\w\./]+)',infiles).group(1)
    district_pkl = re.search('(?<=district=)([\w\./]+)',infiles).group(1)
    
    filePath_save = outpath if outpath is not None else './SharedData/ModelData/'
     
    # Assign parameters
    target_crime = re.search('(?<=targetcrime=)(\w+)',params).group(1)
    clf_name = re.search('(?<=model=)(\w+)',params).group(1)
    kfolds = int(re.search('(?<=kfolds=)(\d+)',params).group(1))
    r_seed = int(re.search('(?<=rseed=)(\d+)',params).group(1))
    train_region = re.search('(?<=trainregion=)([A-Za-z]+)',params).group(1)
    test_region = re.search('(?<=testregion=)([A-Za-z]+)',params).group(1)

    # If more than one cluster/district is provided (connected by '_'), the union of their regions will be assumed
    cluster_Nos = {'train':None,'test':None}
    district_Nos = {'train':None,'test':None}    
    if train_region != 'city':   
        train_region_num_str = re.search('(?<=trainregionNo=)([\d_]+)',params).group(1)       
        if train_region == 'cluster':
            cluster_Nos['train'] = map(int,train_region_num_str.split('_'))
        elif train_region == 'district':
            district_Nos['train'] = train_region_num_str.split('_')
    else:
        train_region_num_str = ''
        
    if test_region != 'city':
        test_region_num_str = re.search('(?<=testregionNo=)([\d_]+)',params).group(1)
        if test_region == 'cluster':
            cluster_Nos['test'] = map(int,test_region_num_str.split('_'))
        elif test_region == 'district':
            district_Nos['test'] = test_region_num_str.split('_')
    else:
        test_region_num_str = ''
                          
        
    CV_skf = get_cv_obj(kfolds,r_seed)
    scaling = 'minmax'      
                
    filename_dict = dict(group=group_pkl) 
    if cluster_pkl != 'NA':            
        filename_dict['cluster'] = cluster_pkl    
    if district_pkl != 'NA':
        filename_dict['district'] = district_pkl                    
  
    #----------------------------------------#
    mask_sample_region = {'train':train_region,'test':test_region}   
      
    start = time.time()
    
#    if train_region=='city' and test_region=='city':
#        filename_dict['feature'] = feature_balanced_h5
#        X_train, y_train, X_test, _, = ld.load_train_test_seq_data(filename_dict['feature'],target_crime,'Label',balanced=True)
#        sample_mask = dict(train=np.ones(len(X_train)).astype(bool),test=np.ones(len(X_test)).astype(bool))
#        
#        clf = get_model_params(clf_name,rand_seed=r_seed,X=X_train)['model']
#        tuning_params = get_model_params(clf_name,rand_seed=r_seed,X=X_train)['tuning_params']                 
#        tuning_param_dicts = dict([(clf_name+'__'+key, val) for key,val in tuning_params.items()])       
#    else:
#        filename_dict['feature'] = feature_h5  
#        loading_info = ld.load_paired_train_test_seq_data(filename_dict, mask_sample_region=mask_sample_region, \
#                                                      cluster_Nos=cluster_Nos, district_Nos=district_Nos, \
#                                                      balance=True, rand_seed=r_seed, load_city=False,\
#                                                      save_format='h5',crime_type=target_crime,target='Label')
#        sample_mask = loading_info['sample_mask']
#        X_train, y_train = loading_info['train_data_subset']
#        X_test, _ = loading_info['test_data_subset']
#
#        clf = get_model_params(clf_name,rand_seed=r_seed,X=X_train)['model']
#        tuning_params = get_model_params(clf_name,rand_seed=r_seed,X=X_train)['tuning_params']                 
#        tuning_param_dicts = dict([(clf_name+'__'+key, val) for key,val in tuning_params.items()])
 

    
    sample_mask = {}
    if train_region=='city':
        filename_dict['train'] = train_data
        X_train, y_train = ld.load_struct_data_h5(filename_dict['train'],target_crime,'Label',split='train')
        
        clf = get_model_params(clf_name,rand_seed=r_seed,X=X_train)['model']
        tuning_params = get_model_params(clf_name,rand_seed=r_seed,X=X_train)['tuning_params']                 
        tuning_param_dicts = dict([(clf_name+'__'+key, val) for key,val in tuning_params.items()])  
        # train models
        fitting = model_fit(X_train, y_train, clf, clf_name, tuning_param_dicts, CV_skf, scaling)
        model, cv_results, best_params = fitting['model'], fitting['CV_result'], fitting['best_param']
        print('CV results ('+train_region+train_region_num_str+'):')
        print(cv_results)
        print('Best_parameters ('+train_region+train_region_num_str+'):')
        print(best_params)    
    
        sample_mask['train']=np.ones(len(X_train)).astype(bool)
    if test_region=='city':
        # test models by test sample chunks
        pred_scores_stacked = []
        for fn in test_data_list:
            filename_dict['test'] = fn
            X_test, y_test = ld.load_struct_data_h5(filename_dict['test'],target_crime,'Label', split='test')
        
            pred_scores_stacked.append(model_pred(X_test,model))
                
        pred_scores_stacked = np.hstack(pred_scores_stacked)
        
        sample_mask['test']=np.ones(len(pred_scores_stacked)).astype(bool)     
        
        
    end = time.time()
    print('Elapsed time: %.1f' % (end - start))
    
    # save prediction scores
    score_save = filePath_save+'PredScore_'+train_region+train_region_num_str+'_'+test_region+test_region_num_str+'.csv'
    np.savetxt(score_save,pred_scores_stacked,delimiter=',') 
    mask_save = filePath_save+'SampleMask_'+train_region+train_region_num_str+'_'+test_region+test_region_num_str+'.pkl'
    with open(mask_save,'wb') as out_file:
        pickle.dump(sample_mask, out_file, pickle.HIGHEST_PROTOCOL)  