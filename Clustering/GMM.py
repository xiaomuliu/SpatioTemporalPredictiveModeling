#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
===========================================
Gaussian mixture models of crime count data
===========================================

Created on Thu Oct  6 15:02:44 2016

@author: xiaomuliu
"""

import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture # NOTE: scikit learn v0.18 renamed GMM to GaussianMixture
from sklearn.pipeline import Pipeline
import VisualizeCluster as vc
import itertools
import matplotlib.pyplot as plt


def rank_gmm_features(GMMobj,featureNames):
    """
    Return a list of length of number of clusters
    Each element of the returned list is a tuple which constains
    a. feature names
    b. mean values ranked by corresponding center coordinates (means) 
    c. covariance matrix with rows/columns ordered by ranked features
    """
    K = GMMobj.n_components
    centers = GMMobj.means_  # shape(n_components, n_features)
    Covs = GMMobj.covariances_
    P = len(featureNames)
    cov_type = GMMobj.get_params()['covariance_type']
    # shape
    #(n_components,)                        if 'spherical',
    #(n_features, n_features)               if 'tied',
    #(n_components, n_features)             if 'diag',
    #(n_components, n_features, n_features) if 'full'

    descend_sort = lambda x: np.sort(x)[::-1]
    descend_argsort = lambda x: np.argsort(x)[::-1]
    sorted_means = np.apply_along_axis(descend_sort, axis=1, arr=centers)
    sorted_idx = np.apply_along_axis(descend_argsort, axis=1, arr=centers)

    ranked_features = [featureNames[sorted_idx[i,:]] for i in xrange(K)]     
    
    # Make covariance matrix of shape (n_components, n_features, n_features)                 
    if cov_type=='spherical':
        Covs = np.array([var*np.eye(P) for var in Covs])
    elif cov_type=='tied':
        Covs = np.tile(Covs,(K,1)).reshape((K,P,P))
    elif cov_type=='diag':
        Covs = np.array([np.diag(var) for var in Covs])
    elif cov_type=='full':
        Covs = Covs    
       
    ranked_features_inCluster = [(ranked_features[i].tolist(),sorted_means[i].tolist(),Covs[i,sorted_idx[i,:],:][:,sorted_idx[i,:]]) 
                                 for i in xrange(K)]                             
    return ranked_features_inCluster

        
if __name__ == '__main__':
    import re
    import time
    import cPickle as pickle
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infiles = args['input']
    outfile = args['output']
    params = args['param']

    #convert string input to tuple       
    infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)
      
    grid_pkl, feature_pkl = infile_match.group(1), infile_match.group(2)

    # load grid info  
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
    
    # load features   
    with open(feature_pkl,'rb') as input_file:
        data_df = pickle.load(input_file)
        
    featureNames = data_df.columns.values
    X = data_df.values
    
    # setup estimator objects
    Ncomponents = int(params)
    r_seed = 1234
    max_iter = 100
    n_init = 1
    #'full': each component has its own general covariance matrix
    #'tied': all components share the same general covariance matrix
    #'diag': each component has its own diagonal covariance matrix
    #'spherical': each component has its own single variance
    cov_types = ('full', 'tied', 'diag', 'spherical')
    color_iter = itertools.cycle(['red', 'green', 'blue','orange'])
    
    zscore_scaler = preprocessing.StandardScaler()
    gmm = GaussianMixture(n_init=n_init, init_params='kmeans', max_iter=max_iter, random_state=r_seed)

    figpath = outfile if outfile is not None else './Figures/GMM/Ncomp_'+str(Ncomponents)+'/'    
    # **************************** Experiment 1 **************************** *# 
    # Raw data
    print '========== RAW DATA ============'                
    print 'BIC scores:'
    start = time.time()
    for cov_type in cov_types:
        gmm.set_params(n_components=Ncomponents,covariance_type=cov_type)
        gmm.fit(X)
        print cov_type, np.round(gmm.bic(X))
        
    end = time.time()
    print 'Elapsed time:', round(end-start,2) 
                                  
    # use 'full' covariance estimation
    gmm.set_params(covariance_type='full');
    y_pred = gmm.fit(X).predict(X)
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)        
    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6)  
    figname = 'GMM_segmentation_raw.png'    
    vc.save_figure(figpath+figname,fig)

    ranked_features = rank_gmm_features(gmm,featureNames)
    
    # **************************** Experiment 2 **************************** *# 
    # standardized data
    # The following part needs further investigation:
    # -----------------------------------------------------------------    
    # For GMM, standardizating feature may lead to worse results. 
    # Since standardization will make covariance to have diagonal of ones
    # For this problem, as all features are in the same unit (count), 
    # feature standarization is not necessary. And one can use raw data and 
    # let GMM to estimate means and covariances of each components
    # -----------------------------------------------------------------
    print '========== STANDARDIZED DATA =========='
    pipe1 = Pipeline(steps=[('standardize',zscore_scaler), ('GMM', gmm)])
    
    start = time.time()
    for cov_type in cov_types:
        pipe1.set_params(GMM__n_components=Ncomponents,GMM__covariance_type=cov_type)
        pipe1.fit(X)
        print cov_type, np.round(pipe1.named_steps['GMM'].bic(X))
    
    end = time.time()
    print 'Elapsed time:', round(end-start,2)            
                  

    pipe1.set_params(GMM__covariance_type='full');
    y_pred = pipe1.fit(X).predict(X)  
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_pred, grid_2d, flattened=True, mask=mask_grdInCity)          
    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6)  
    figname = 'GMM_segmentation_std.png'    
    vc.save_figure(figpath+figname,fig)
     
    gmm = pipe1.named_steps['GMM']
    ranked_features = rank_gmm_features(gmm,featureNames)
    print("Features ranked by cluster centers:")
    for i, fmc in enumerate(ranked_features):
        print("Cluster " + str(i))
        print zip(fmc[0],np.around(fmc[1],2),np.around(fmc[2],2))
            
