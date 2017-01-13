#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
======================================
k-means clustering of crime count data
======================================
Created on Wed Oct  5 16:00:56 2016

@author: xiaomuliu
"""
import numpy as np
import cPickle as pickle
from sklearn import preprocessing, decomposition
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import VisualizeCluster as vc
import time
import matplotlib.pyplot as plt


def rank_kmeans_features(clusterObj,featureNames):
    """
    Return a list of length of number of clusters
    Each element of the returned list is a tuple which constains a 
    feature names and mean values ranked by corresponding center coordinates (means) 
    
    clusterObj: an instance of k-means estimator
    """
    k = clusterObj.n_clusters
    centers = clusterObj.cluster_centers_  # shape(n_clusters, n_features)
    
    descend_sort = lambda x: np.sort(x)[::-1]
    descend_argsort = lambda x: np.argsort(x)[::-1]
    sorted_means = np.apply_along_axis(descend_sort, axis=1, arr=centers)
    sorted_idx = np.apply_along_axis(descend_argsort, axis=1, arr=centers)

    ranked_features = [featureNames[sorted_idx[i,:]] for i in xrange(k)]     
    
    ranked_features_inCluster = [(ranked_features[i].tolist(),sorted_means[i].tolist()) for i in xrange(k)] 
    return ranked_features_inCluster

    
if __name__ == '__main__':
          
    # load features   
    feature_pkl = './FeatureData/grid_660/feature_dataframe.pkl'
    with open(feature_pkl,'rb') as input_file:
        data_df = pickle.load(input_file)
        
    featureNames = data_df.columns.values
    X = data_df.values

    # load grid info  
    grid_pkl = '../SharedData/SpatialData/grid_660.pkl'
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, mask_grdInCity_im = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
    
    # setup estimator objects
    r_seed = 1234
    Nclusters = np.arange(3,15,3) 
    
    zscore_scaler = preprocessing.StandardScaler()
    pca = decomposition.PCA(n_components=3)
    k_means = KMeans(random_state=r_seed, n_init=10) 
    
    # **************************** Experiment 1 **************************** *# 
    # Raw data + kmeans

    # K-means clustering is "isotropic" in all directions of space,
    # and therefore tends to produce more or less round (rather than elongated) clusters.
    # In this situation leaving variances unequal is equivalent to 
    # putting more weight on variables with smaller variance,
    # so clusters will tend to be separated along variables with greater variance.
              
    fig, axes = plt.subplots(2, 2, figsize=(12,9)) 
    start = time.time()
    for i, nc in enumerate(Nclusters):
        k_means.set_params(n_clusters=nc);
        #y_pred = k_means.fit_predict(X)
        y_pred = k_means.fit(X).labels_
        axis = axes[np.unravel_index(i,dims=(2,2))]        
        vc.plot_clusters(axis, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)
        vc.colorbar_index(ncolors=nc,ax=axis,shrink=0.6)
        
    end = time.time()
    print (end-start)

    nc = 6
    k_means.set_params(n_clusters=nc);
    y_pred = k_means.fit(X).labels_
    ranked_features = rank_kmeans_features(k_means,featureNames)
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)        
    vc.colorbar_index(ncolors=nc,ax=ax,shrink=0.6)  
    
    # **************************** Experiment 2 **************************** *# 
    # standardized data + kmeans     
    pipe1 = Pipeline(steps=[('standardize',zscore_scaler), ('kmeans', k_means)])
    fig, axes = plt.subplots(2, 2, figsize=(12,9)) 
    start = time.time()
    for i, nc in enumerate(Nclusters):
        pipe1.set_params(kmeans__n_clusters=nc)
        y_pred = pipe1.fit_predict(X)
        axis = axes[np.unravel_index(i,dims=(2,2))]        
        vc.plot_clusters(axis, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)
        vc.colorbar_index(ncolors=nc,ax=axis,shrink=0.6)
    
    end = time.time()
    print (end-start)
    
    nc = 6
    pipe1.set_params(kmeans__n_clusters=nc);
    y_pred = pipe1.fit_predict(X)
    ranked_features = rank_kmeans_features(k_means,featureNames)
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)          
    vc.colorbar_index(ncolors=nc,ax=ax,shrink=0.6)  
    print("Features ranked by cluster centers:")
    for i, fm in enumerate(ranked_features):
        print("Cluster " + str(i))
        print zip(fm[0],np.around(fm[1],2))
        
    
    # **************************** Experiment 3 **************************** *# 
    # PCA of standardized data + kmeans 
    pipe2 = Pipeline(steps=[('standardize',zscore_scaler), ('PCA',pca), ('kmeans', k_means)])
    fig, axes = plt.subplots(2, 2, figsize=(12,9)) 
    start = time.time()
    for i, nc in enumerate(Nclusters):
        pipe2.set_params(kmeans__n_clusters=nc)
        y_pred = pipe2.fit_predict(X)
        axis = axes[np.unravel_index(i,dims=(2,2))]        
        vc.plot_clusters(axis, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)
        vc.colorbar_index(ncolors=nc,ax=axis,shrink=0.6)
    
    end = time.time()
    print (end-start)