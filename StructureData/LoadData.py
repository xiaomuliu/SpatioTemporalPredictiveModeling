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
from StructPredModelData_daily import downsample
import pandas as pd
import h5py

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

def load_struct_data_h5(filename,crime_type=None,target='Label',split='train'):       
    with h5py.File(filename, 'r') as hf:
        if split=='train_city':
            X = hf['feature_array/'+crime_type][:]
        else:  
            X = hf['feature_array'][:]
        if target=='Label':
            y = hf['label_array/'+crime_type][:]
        else:
            y = hf['target_array/'+crime_type][:]
        
    return X, y
    
def load_train_test_seq_data(filename,crime_type='',target='Label',balanced=False):
       
    with h5py.File(filename, 'r') as hf:
        if balanced:
            X_train = hf['train_data/'+crime_type+'/feature_array'][:]
        else:
            # training feature data for imbalanced data are the same for different crime types
            X_train = hf['train_data/feature_array'][:]  
        
        X_test = hf['test_data/feature_array'][:]
        if target=='Label':
            y_train = hf['train_data/'+crime_type+'/label_array'][:]
            y_test = hf['test_data/'+crime_type+'/label_array'][:]
        else:
            y_train = hf['train_data/'+crime_type+'/target_array'][:]
            y_test = hf['test_data/'+crime_type+'/target_array'][:]
        
    return X_train, y_train, X_test, y_test

    
def load_baseline_data(filename, crime_type):
    with open(filename,'rb') as input_file:
        baseline = pickle.load(input_file)
    model_names = [crime_type+'_LT_den', crime_type+'_ST_den']
    
    return baseline.ix[:,np.in1d(baseline.columns, model_names)].values
        
def load_train_test_group(filename):
    with open(filename,'rb') as input_file:
        train_test_groups = pickle.load(input_file)
    return train_test_groups     
                                                  
def load_grid(filename):
    with open(filename,'rb') as input_file:
        grid_list = pickle.load(input_file)
    grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, mask_grdInCity_im = grid_list
    return grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, mask_grdInCity_im

def load_clusters(filename):
    with open(filename,'rb') as input_file:
        clusters = pickle.load(input_file)
    return clusters['label'], clusters['ranked_features']  

def load_cluster_mask(filename,target_clusters):
    with open(filename,'rb') as input_file:
        cluster_masks = pickle.load(input_file)
    # If more than one target clusters are provided, the union of their masks will be returned
    if not hasattr(target_clusters,'__len__'):
        target_clusters = [target_clusters]
    for i,d in enumerate(target_clusters):
        if i==0:
            M = cluster_masks[d] # the initial mask
        else:
            M = np.logical_or(M,cluster_masks[d])
    return M                  
    
def load_districts(filename):
    with open(filename,'rb') as input_file:
        district_label = pickle.load(input_file)
    return district_label  

def load_district_mask(filename,target_districts):
    with open(filename,'rb') as input_file:
        district_masks = pickle.load(input_file)
    # If more than one target districts are provided, the union of their masks will be returned
    if not hasattr(target_districts,'__len__'):
        target_districts = [target_districts]
    for i,d in enumerate(target_districts):
        if i==0:
            M = district_masks[d] # the initial mask
        else:
            M = np.logical_or(M,district_masks[d])
    return M     
    

#def load_paired_train_test_data(filenames, mask_sample_region={'train':'city','test':'city'}, cluster_Nos={'train':None,'test':None},
#                                district_Nos={'train':None,'test':None},balance=True, rand_seed=1234,load_city=True,\
#                                save_format='pickle', crime_type=None,target='Label'):
#    """
#    filenames: a dict with keys 'feature','district'/'cluster' and 'group' with their filenames being the corresponding values
#    """
#    # load raw data            
#    X_train, y_train, X_test, y_test, feature_names = load_train_test_data(filenames['feature'],save_format,crime_type,target)
#    # load cluster info
#    if 'cluster' in filenames.keys():
#        cluster_label, _ = load_clusters(filenames['cluster'])
#        cluster_mask = {'train':np.in1d(cluster_label, cluster_Nos['train']) if cluster_Nos['train'] is not None else None,
#                        'test':np.in1d(cluster_label, cluster_Nos['test']) if cluster_Nos['test'] is not None else None}
#    # load district info
#    if 'district' in filenames.keys():
#        district_label = load_districts(filenames['district'])
#        district_mask = {'train':np.in1d(district_label,district_Nos['train']) if district_Nos['train'] is not None else None,
#                         'test':np.in1d(district_label,district_Nos['test']) if district_Nos['test'] is not None else None}
#
#    # load time interval info
#    group_info = load_train_test_group(filenames['group'])
#    groups_train, groups_test = group_info['train_groups'], group_info['test_groups']  
#    Ngroups_train, Ngroups_test = len(groups_train), len(groups_test)
#    
#    sample_mask = {}
#    if mask_sample_region['train']=='city':
#        sample_mask['train'] = np.ones(len(y_train)).astype(bool)
#    if mask_sample_region['test']=='city':
#        sample_mask['test'] = np.ones(len(y_test)).astype(bool)
#    if mask_sample_region['train']=='cluster':
#        sample_mask['train'] = np.tile(cluster_mask['train'],Ngroups_train)
#    if mask_sample_region['test']=='cluster':
#        sample_mask['test'] = np.tile(cluster_mask['test'],Ngroups_test)
#    if mask_sample_region['train']=='district':
#        sample_mask['train'] = np.tile(district_mask['train'],Ngroups_train)
#    if mask_sample_region['test']=='district':
#        sample_mask['test'] = np.tile(district_mask['test'],Ngroups_test)    
#                       
#    X_train_sub, y_train_sub = X_train[sample_mask['train'],:], y_train[sample_mask['train']]
#    X_test_sub, y_test_sub = X_test[sample_mask['test'],:], y_test[sample_mask['test']]    
#                        
#    if balance:
#        sampleInd_cluster = downsample(y_train_sub,rand_state=rand_seed,shuffle=True)
#        sampleInd = downsample(y_train,rand_state=rand_seed,shuffle=True)
#        X_train_sub = X_train_sub[sampleInd_cluster,:]
#        y_train_sub = y_train_sub[sampleInd_cluster]
#        if load_city:
#            X_train = X_train[sampleInd,:]  
#            y_train = y_train[sampleInd]  
#    
#    return_dict = {'sample_mask':sample_mask, 
#                   'groups':[groups_train, groups_test],
#                   'train_data_subset':[X_train_sub, y_train_sub],
#                   'test_data_subset':[X_test_sub, y_test_sub]}
#    if load_city:
#        return_dict.update({'train_data_city':[X_train, y_train], 'test_data_city':[X_test, y_test]})
#        
#    return return_dict    
#
#def load_paired_train_test_seq_data(filenames, mask_sample_region={'train':'city','test':'city'}, cluster_Nos={'train':None,'test':None},
#                                district_Nos={'train':None,'test':None},balance=True, rand_seed=1234,load_city=True,\
#                                save_format='pickle', crime_type=None,target='Label'):
#    """
#    filenames: a dict with keys 'feature','district'/'cluster' and 'group' with their filenames being the corresponding values
#    """
#    # load raw data            
#    X_train, y_train, X_test, y_test = load_train_test_seq_data(filenames['feature'],crime_type,target)
#    # load cluster info
#    if 'cluster' in filenames.keys():
#        cluster_label, _ = load_clusters(filenames['cluster'])
#        cluster_mask = {'train':np.in1d(cluster_label, cluster_Nos['train']) if cluster_Nos['train'] is not None else None,
#                        'test':np.in1d(cluster_label, cluster_Nos['test']) if cluster_Nos['test'] is not None else None}
#    # load district info
#    if 'district' in filenames.keys():
#        district_label = load_districts(filenames['district'])
#        district_mask = {'train':np.in1d(district_label,district_Nos['train']) if district_Nos['train'] is not None else None,
#                         'test':np.in1d(district_label,district_Nos['test']) if district_Nos['test'] is not None else None}
#
#    # load time interval info
#    group_info = load_train_test_group(filenames['group'])
#    groups_train, groups_test = group_info['train_groups'], group_info['test_groups']  
#    Ngroups_train, Ngroups_test = len(groups_train), len(groups_test)
#    
#    sample_mask = {}
#    if mask_sample_region['train']=='city':
#        sample_mask['train'] = np.ones(len(y_train)).astype(bool)
#    if mask_sample_region['test']=='city':
#        sample_mask['test'] = np.ones(len(y_test)).astype(bool)
#    if mask_sample_region['train']=='cluster':
#        sample_mask['train'] = np.tile(cluster_mask['train'],Ngroups_train)
#    if mask_sample_region['test']=='cluster':
#        sample_mask['test'] = np.tile(cluster_mask['test'],Ngroups_test)
#    if mask_sample_region['train']=='district':
#        sample_mask['train'] = np.tile(district_mask['train'],Ngroups_train)
#    if mask_sample_region['test']=='district':
#        sample_mask['test'] = np.tile(district_mask['test'],Ngroups_test)    
#                      
#    X_train_sub, y_train_sub = X_train[sample_mask['train'],:,:], y_train[sample_mask['train']]
#    X_test_sub, y_test_sub = X_test[sample_mask['test'],:,:], y_test[sample_mask['test']]    
#                        
#    if balance:
#        sampleInd_cluster = downsample(y_train_sub,rand_state=rand_seed,shuffle=True)
#        sampleInd = downsample(y_train,rand_state=rand_seed,shuffle=True)
#        X_train_sub = X_train_sub[sampleInd_cluster,:,:]
#        y_train_sub = y_train_sub[sampleInd_cluster]
#        if load_city:
#            X_train = X_train[sampleInd,:,:]  
#            y_train = y_train[sampleInd]  
#    
#    return_dict = {'sample_mask':sample_mask, 
#                   'groups':[groups_train, groups_test],
#                   'train_data_subset':[X_train_sub, y_train_sub],
#                   'test_data_subset':[X_test_sub, y_test_sub]}
#    if load_city:
#        return_dict.update({'train_data_city':[X_train, y_train], 'test_data_city':[X_test, y_test]})
#        
#    return return_dict    
 
    
def load_feature_ranks(filename, target_regions):
    with open(filename,'rb') as input_file:
        feature_ranking_df = pickle.load(input_file)
        
    feature_names = feature_ranking_df.index.values     
    if not isinstance(target_regions,list):
        target_regions=[target_regions]

    feature_ranks = [feature_ranking_df[region].values for region in target_regions]      
     
    return feature_names,feature_ranks     