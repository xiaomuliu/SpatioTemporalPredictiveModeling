#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:21:10 2016

@author: xiaomuliu
"""
import numpy as np
import sys
if sys.version_info[0] > 2:
    import pickle # python 3.x deprecated cPickle 
else:
    import cPickle as pickle
from StructClassData import downsample

def load_crime_data(filenames):
    crime_data = []
    if isinstance(filenames, list):              
        for c in filenames:     
            with open(filenames,'rb') as input_file:
                crime_data.append(pickle.load(input_file))
    else:
        with open(filenames,'rb') as input_file:
            crime_data.append(pickle.load(input_file))

    return crime_data if len(crime_data)>1 else crime_data[0]     

def load_train_test_data(filename):
    with open(filename,'rb') as input_file:
        data_df = pickle.load(input_file)
        
    featurename_idx = data_df.columns.difference(['Label','SplitIndicator'])
    feature_names = featurename_idx.values.astype('str')
    X_train = data_df.ix[data_df['SplitIndicator']=='train',featurename_idx].values
    y_train = data_df.ix[data_df['SplitIndicator']=='train','Label'].values
    X_test = data_df.ix[data_df['SplitIndicator']=='test',featurename_idx].values
    y_test = data_df.ix[data_df['SplitIndicator']=='test','Label'].values

    return X_train, y_train, X_test, y_test, feature_names

def load_train_test_group(filename):
    with open(filename,'rb') as input_file:
        train_test_groups = pickle.load(input_file)
    return train_test_groups     
    
def load_baseline_data(filename, crimetype):
    with open(filename,'rb') as input_file:
        baseline = pickle.load(input_file)
    model_names = [crimetype+'_LT_den', crimetype+'_ST_den']
    
    return baseline.ix[:,np.in1d(baseline.columns, model_names)].values
                       
                       
def load_grid(filename):
    with open(filename,'rb') as input_file:
        grid_list = pickle.load(input_file)
    grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, mask_grdInCity_im = grid_list
    return grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, mask_grdInCity_im

def load_clusters(filename):
    with open(filename,'rb') as input_file:
        clusters = pickle.load(input_file)
    return clusters['label'], clusters['ranked_features']  

def load_cluster_mask(filename,target_cluster):
    with open(filename,'rb') as input_file:
        cluster_masks = pickle.load(input_file)
    return cluster_masks[target_cluster]                 
    
def load_subregion_train_test_data(filenames, target_cluster, balance=True, rand_seed=1234, load_city=True):
    """
    filenames: a dict with keys 'feature','cluster', and 'group' with their filenames being the corresponding values
    """
    # load raw data              
    X_train, y_train, X_test, y_test, feature_names = load_train_test_data(filenames['feature'])
    # load cluster info
    cluster_label, cluster_stat = load_clusters(filenames['cluster'])
    cluster_mask = np.array(cluster_label==target_cluster)
    # load time interval info
    group_info = load_train_test_group(filenames['group'])
    groups_train, groups_test = group_info['train_groups'], group_info['test_groups']  
    Ngroups_train, Ngroups_test = len(groups_train), len(groups_test)
    
    sample_mask = {'train': np.tile(cluster_mask,Ngroups_train),'test':np.tile(cluster_mask,Ngroups_test)}
    X_train_cluster, y_train_cluster = X_train[sample_mask['train'],:], y_train[sample_mask['train']]
    X_test_cluster, y_test_cluster = X_test[sample_mask['test'],:], y_test[sample_mask['test']]    
                        
    if balance:
        sampleInd_cluster = downsample(y_train_cluster,rand_state=rand_seed,shuffle=True)
        sampleInd = downsample(y_train,rand_state=rand_seed,shuffle=True)
        X_train_cluster = X_train_cluster[sampleInd_cluster,:]
        y_train_cluster = y_train_cluster[sampleInd_cluster]
        if load_city:
            X_train = X_train[sampleInd,:]  
            y_train = y_train[sampleInd]  
    
    return_dict = {'cluster_mask':cluster_mask, 'sample_mask':sample_mask, 
                   'cluster_stat':cluster_stat[target_cluster] if cluster_stat is not None else None,'feature_names':feature_names,
                   'groups':[groups_train, groups_test],
                   'train_data_subregion':[X_train_cluster, y_train_cluster],
                   'test_data_subregion':[X_test_cluster, y_test_cluster]}
    if load_city:
        return_dict.update({'train_data_city':[X_train, y_train], 'test_data_city':[X_test, y_test]})
        
    return return_dict    
    
def load_feature_ranks(filename, target_regions):
    with open(filename,'rb') as input_file:
        feature_ranking_df = pickle.load(input_file)
        
    feature_names = feature_ranking_df.index.values     
    if not isinstance(target_regions,list):
        target_regions=[target_regions]

    feature_ranks = [feature_ranking_df[region].values for region in target_regions]      
     
    return feature_names,feature_ranks     