#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=====================================
Variational Bayesian Gaussian Mixture
=====================================
Created on Mon Oct 17 17:01:44 2016

@author: xiaomuliu
"""

import numpy as np
from sklearn import preprocessing
from sklearn.mixture import BayesianGaussianMixture 
from sklearn.mixture import GaussianMixture # NOTE: scikit learn v0.18 renamed GMM to GaussianMixture
from sklearn.pipeline import Pipeline
import VisualizeCluster as vc
import matplotlib.pyplot as plt

def plot_component_weights(ax, estimator, title):
    ax.set_title(title)
    ax.get_xaxis().set_tick_params(direction='out')
    ax.yaxis.grid(True, alpha=0.7)
    ax.bar(np.arange(len(estimator.weights_)), estimator.weights_, 
           width=0.9, color='blue')
    ax.set_xlim(-.6, estimator.n_components - .4)
    ax.set_ylim(0., 1.1)
    ax.set_ylabel('Weight of each component')    
    
        
if __name__ == '__main__':       
    import re
    import sys
    import time
    import cPickle as pickle
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infiles = args['input']
    outfile = args['output']
    params = args['param']
   
    infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)     
    grid_pkl, feature_pkl = infile_match.group(1), infile_match.group(2)
    
    # load grid info  
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, mask_grdInCity_im = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 

    # load features   
    with open(feature_pkl,'rb') as input_file:
        data_df = pickle.load(input_file)
        
    featureNames = data_df.columns.values
    X = data_df.values
    
    # setup estimator objects
    param_match = re.match('(\d+) ([-+]?\d*\.\d+|\d+)', params)
    Ncomponents = int(param_match.group(1))
    gamma = float(param_match.group(2)) #weight_concentration_prior
    
    r_seed = 1234
    max_iter = 1000
    n_init = 1
    
    zscore_scaler = preprocessing.StandardScaler()
    bgmm = BayesianGaussianMixture(n_components=Ncomponents, n_init=n_init, init_params='kmeans', 
                                   weight_concentration_prior_type="dirichlet_process", covariance_type='full',
                                   max_iter=max_iter, random_state=r_seed)
    pipe1 = Pipeline(steps=[('standardize',zscore_scaler), ('BGMM', bgmm)])

    figpath = outfile if outfile is not None else './Figures/VBGMM/Ncomp_'+str(Ncomponents)+'/'
    
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    fig.suptitle("Infinite mixture with a Dirichlet process prior")
    start = time.time()

    pipe1.set_params(BGMM__weight_concentration_prior=gamma);
    pipe1.fit(X)
    plot_component_weights(ax, pipe1.named_steps['BGMM'],
                           r"$\gamma_0=%0.1e$" % gamma)   #raw string literals
    figname = 'comp_weights_gamma_'+str(gamma)+'.png'    
    vc.save_figure(figpath+figname,fig)
                           
    end = time.time()
    print 'Elapsed time:', round(end-start,2) #13.1s
  
    y_pred1 = pipe1.fit(X).predict(X)  
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_pred1, grid_2d, flattened=True, mask=mask_grdInCity)          
    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6)  
    figname = 'VBGMM_segmentation_gamma_'+str(gamma)+'.png'    
    vc.save_figure(figpath+figname,fig)
    
#    # compare with EM Gaussian mixture model
#    gmm = GaussianMixture(n_components=Ncomponents, n_init=n_init, init_params='kmeans', covariance_type='full', max_iter=max_iter, random_state=r_seed)
#    pipe2 = Pipeline(steps=[('standardize',zscore_scaler), ('GMM', gmm)])
#    y_pred2 = pipe2.fit(X).predict(X)
#    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
#    vc.plot_clusters(ax, y_pred2, grid_2d, flattened=True, mask=mask_grdInCity)        
#    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6)  
#    figname = 'GMM_segmentation.png'    
#    vc.save_figure(figpath+figname,fig)