#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:12:20 2017
# Training/testing sub-regions

@author: xiaomuliu
"""
    
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
#NOTE: scikit learn v0.18 changed 'cross_validation' module and 'grid_search' module to 'model_selection'
from sklearn.model_selection import GridSearchCV 
import sys
sys.path.append('..') 
import StructureData.LoadData as ld 


def pair_train_test_sample(filenames, mask_sample_region={'train':None,'test':None}, cluster_Nos={'train':None,'test':None},
                           district_Nos={'train':None,'test':None},chunk_size={'train':None,'test':None}):
    """
    filenames: a dict with keys 'district'/'cluster' and 'group' with their filenames being the corresponding values
    """

    # load cluster info
    if 'cluster' in filenames.keys():
        cluster_label, _ = ld.load_clusters(filenames['cluster'])
        cluster_mask = {'train':np.in1d(cluster_label, cluster_Nos['train']) if cluster_Nos['train'] is not None else None,
                        'test':np.in1d(cluster_label, cluster_Nos['test']) if cluster_Nos['test'] is not None else None}
    # load district info
    if 'district' in filenames.keys():
        district_label = ld.load_districts(filenames['district'])
        district_mask = {'train':np.in1d(district_label,district_Nos['train']) if district_Nos['train'] is not None else None,
                         'test':np.in1d(district_label,district_Nos['test']) if district_Nos['test'] is not None else None}

    # load time interval info
    group_info = ld.load_train_test_group(filenames['group'])
    groups_train, groups_test = group_info['train_groups'], group_info['test_groups']  
    
    sample_mask = {'train':[],'test':[]}
    if mask_sample_region['train']=='cluster':
        M = cluster_mask['train']
    elif mask_sample_region['train']=='district':
        M = district_mask['train']
    for i in range(groups_train[0],groups_train[-1], chunk_size['train']):
        if i < groups_train[-1] and i+chunk_size['train'] > groups_train[-1]:
            # the end chunk may has smaller size
            sample_mask['train'].append(np.tile(M,groups_train[-1]+1-i))
        else:
            sample_mask['train'].append(np.tile(M,chunk_size['train']))
                                           

    if mask_sample_region['test']=='cluster':
        M = cluster_mask['test']
    elif mask_sample_region['test']=='district':
        M = district_mask['test']
    for i in range(groups_test[0],groups_test[-1], chunk_size['test']):
        if i < groups_test[-1] and i+chunk_size['test'] > groups_test[-1]:
            # the end chunk may has smaller size
            sample_mask['test'].append(np.tile(M,groups_test[-1]+1-i))
        else:
            sample_mask['test'].append(np.tile(M,chunk_size['test']))
                
    return sample_mask




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
    import cPickle as pickle
    from ModelSpec import get_model_params, get_cv_obj
    import sys
    sys.path.append('..') 
    from StructureData.StructPredModelData_daily import downsample
    from Misc.ComdArgParse import ParseArg    
    import time    
    
    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    group_pkl = re.search('(?<=group=)([\w\./]+)',infiles).group(1)
    train_data = re.search('(?<=traindata=)([\w\./\s]+)(?=testdata)',infiles).group(1)
    test_data = re.search('(?<=testdata=)([\w\./\s]+)(?=cluster)',infiles).group(1) # File names are seperated by whitespace
    
    # sort train/test data list by chunk No
    train_data_list = np.array(train_data.rstrip().split('\n'))
    chunkNo_list = np.zeros(len(train_data_list)).astype(int)
    for idx, fn in enumerate(train_data_list):
        chunkNo_list[idx] = int(re.search('(?<=chunk)(\d+)',fn).group(1))
        
    new_idx = np.argsort(chunkNo_list)
    train_data_list = train_data_list[new_idx]
    
    
    test_data_list = np.array(test_data.rstrip().split('\n'))
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
  
    chunk = re.search('(?<=chunksize=)(\d+) (\d+)', params)
    chunk_size = dict(train=int(chunk.group(1)), test=int(chunk.group(2))) 
                         
    clf = get_model_params(clf_name,rand_seed=r_seed)['model']
    tuning_params = get_model_params(clf_name,rand_seed=r_seed)['tuning_params']                 
    tuning_param_dicts = dict([(clf_name+'__'+key, val) for key,val in tuning_params.items()]) 
        
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
          
    
    sample_mask = pair_train_test_sample(filename_dict, mask_sample_region, cluster_Nos, district_Nos, chunk_size)
    
    if train_region=='city':
        X_train, y_train = ld.load_struct_data_h5(train_data_list[0], target_crime,'Label',split='train_city')    
        sample_mask['train'] = np.ones(len(X_train)).astype(bool)
        
    else:              
        X_train_stacked = []
        y_train_stacked = []

        for i,fn in enumerate(train_data_list):
            X_train, y_train = ld.load_struct_data_h5(fn, target_crime,'Label', split='train_chunk')
        
            
            X_train_stacked.append(X_train[sample_mask['train'][i],:])
            y_train_stacked.append(y_train[sample_mask['train'][i]])
        
        
        # convert to np arrays
        X_train_stacked = np.vstack(X_train_stacked)
        y_train_stacked = np.hstack(y_train_stacked)    
        # downsampling
        sampleInd = downsample(y_train_stacked,rand_state=r_seed,shuffle=True)
        X_train = X_train_stacked[sampleInd,:]
        y_train = y_train_stacked[sampleInd]
    

        sample_mask['train'] = np.hstack(sample_mask['train'])

    # train models
    fitting = model_fit(X_train, y_train, clf, clf_name, tuning_param_dicts, CV_skf, scaling)
    model, cv_results, best_params = fitting['model'], fitting['CV_result'], fitting['best_param']
    print('CV results ('+train_region+train_region_num_str+'):')
    print(cv_results)
    print('Best_parameters ('+train_region+train_region_num_str+'):')
    print(best_params)    
            
        
    # test models by test sample chunks  
    pred_scores_stacked = []
    for i,fn in enumerate(test_data_list):
        X_test, _ = ld.load_struct_data_h5(fn, target_crime,'Label', split='test')
        if test_region=='city':
            pred_scores_stacked.append(model_pred(X_test,model))
        else:
            pred_scores_stacked.append(model_pred(X_test[sample_mask['test'][i],:],model))
                
    pred_scores_stacked = np.hstack(pred_scores_stacked)

        
    if test_region=='city':      
        sample_mask['test'] = np.ones(len(pred_scores_stacked)).astype(bool)
    else:
        sample_mask['test'] = np.hstack(sample_mask['test'])
        
    end = time.time()
    print('Elapsed time: %.1f' % (end - start))    

        
    # save prediction scores
    score_save = filePath_save+'PredScore_'+train_region+train_region_num_str+'_'+test_region+test_region_num_str+'.csv'
    np.savetxt(score_save,pred_scores_stacked,delimiter=',') 
    mask_save = filePath_save+'SampleMask_'+train_region+train_region_num_str+'_'+test_region+test_region_num_str+'.pkl'
    with open(mask_save,'wb') as out_file:
        pickle.dump(sample_mask, out_file, pickle.HIGHEST_PROTOCOL)      