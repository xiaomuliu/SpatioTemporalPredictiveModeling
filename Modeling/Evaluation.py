#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:51:29 2017

# For thesis

@author: xiaomuliu
"""
import numpy as np
from scipy.integrate import simps 
from scipy.stats import ttest_rel 
import matplotlib.pyplot as plt
import os 
import itertools  
import sys
sys.path.append('..') 
from ImageProcessing.KernelSmoothing import bin_point_data_2d
from Grid.SetupGrid import flattened_to_geoIm

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

import matplotlib
params = {'legend.fontsize': SMALL_SIZE,
         'axes.labelsize': MEDIUM_SIZE,
         'axes.titlesize':MEDIUM_SIZE,
         'xtick.labelsize':SMALL_SIZE,
         'ytick.labelsize':SMALL_SIZE}
matplotlib.rcParams.update(params)

def EvalIdx_subgroup(timeIdx, scores, crime_data, group_seq, grid_2d, evalIdx='PAI', mask=None, areaPercent=np.linspace(0,0.05,101)):
    """
    Calculate Prediction Accuracy Index (PAI) for a certain group where PAI is defined as the number of crime counts 
    within a percent area ranked by scores.
    """

    if mask is None:
        mask = np.ones(len(grid_2d[0])*len(grid_2d[1])).astype('bool')
    
    group = group_seq[timeIdx]
    CrimePts = crime_data.ix[crime_data['GROUP']==group,['X_COORD','Y_COORD']].values
    
    cellsize = (np.abs(np.diff(grid_2d[0][:2])), np.abs(np.diff(grid_2d[1][:2])))
    binned_crime_pts = bin_point_data_2d(CrimePts, grid_2d, cellsize, stat='count', geoIm=False)   
    binned_crime_pts = binned_crime_pts.ravel(order='F')[mask] # flatten
    
    Ncells = np.nansum(mask)
    scores_subgroup = scores[timeIdx*Ncells:(timeIdx+1)*Ncells]
 
    N = len(areaPercent)
    evalIdx_vec = np.zeros(N)                         
    if evalIdx=='PAI':                               
        rankIdx = np.argsort(scores_subgroup)[::-1]  # rank scores (descending)                     
        counts_rankedby_scores = binned_crime_pts[rankIdx]
    
        for i in xrange(N):
            evalIdx_vec[i] = np.nansum(counts_rankedby_scores[:int(areaPercent[i]*Ncells)])  
        
        evalIdx_vec /= float(np.nansum(binned_crime_pts)) 
    elif evalIdx=='PEI':            
        rankIdx = np.argsort(scores_subgroup)[::-1]  # rank scores (descending)                  
        counts_rankedby_scores = binned_crime_pts[rankIdx]
        counts_ranked = np.sort(binned_crime_pts)[::-1]
    
        for i in xrange(N):
            evalIdx_vec[i] = np.nansum(counts_rankedby_scores[:int(areaPercent[i]*Ncells)]) / \
                      float(np.nansum(counts_ranked[:int(areaPercent[i]*Ncells)]))  
        evalIdx_vec[np.isnan(evalIdx_vec)] = 0 #replace zero-division results          
    elif evalIdx=='Dice':
        idx_rankedby_scores = np.argsort(scores_subgroup)[::-1]
        idx_rankedby_counts = np.argsort(binned_crime_pts)[::-1]        
    
        for i in xrange(N):
            Ntop = int(areaPercent[i]*Ncells)
            area_rankedby_scores_bool = np.zeros(Ncells).astype(bool)
            area_rankedby_counts_bool = np.zeros(Ncells).astype(bool)
            area_rankedby_scores_bool[idx_rankedby_scores[:Ntop]] = True
            area_rankedby_counts_bool[idx_rankedby_counts[:Ntop]] = True
            
            evalIdx_vec[i] = np.sum(np.logical_and(area_rankedby_scores_bool, area_rankedby_counts_bool)) / float(Ntop)
        evalIdx_vec[np.isnan(evalIdx_vec)] = 0 #replace zero-division results     
    return evalIdx_vec                 

def EvalIdx_subgroup_boot(timeIdx, scores, crime_data, group_seq, grid_2d, evalIdx='PAI', mask=None, 
                          areaPercent=np.linspace(0,0.05,101), Nboots=1, r_seed=1234):
    """
    Calculate bootstrapping prediciton evaluation metrics
    """

    if mask is None:
        mask = np.ones(len(grid_2d[0])*len(grid_2d[1])).astype('bool')
    
    group = group_seq[timeIdx]
    CrimePts = crime_data.ix[crime_data['GROUP']==group,['X_COORD','Y_COORD']].values
    
    cellsize = (np.abs(np.diff(grid_2d[0][:2])), np.abs(np.diff(grid_2d[1][:2])))
    binned_crime_pts = bin_point_data_2d(CrimePts, grid_2d, cellsize, stat='count', geoIm=False)   
    binned_crime_pts = binned_crime_pts.ravel(order='F')[mask] # flatten
    
    Ncells = np.nansum(mask)
    scores_subgroup = scores[timeIdx*Ncells:(timeIdx+1)*Ncells]
 
    # bootstrapping  
    np.random.seed(r_seed)
    # more likely to sample nonzero crime incident cells in order to avoid sampling all no crime cells
    sample_wgt = np.zeros(Ncells)
    sample_wgt[binned_crime_pts!=0]=10    
    sample_wgt[binned_crime_pts==0]=1              
    boot_sample_idx = np.random.choice(Ncells, size=(Ncells, Nboots), replace=True, p=sample_wgt/float(np.sum(sample_wgt))) 
    scores_subgroup_boot = scores_subgroup[boot_sample_idx]
    binned_crime_pts_boot = binned_crime_pts[boot_sample_idx]                      
                             
    Neval = len(areaPercent)
    evalIdx_vec = np.zeros((Neval,Nboots)) 
    for k in xrange(Nboots):                        
        if evalIdx=='PAI':                               
            rankIdx = np.argsort(scores_subgroup_boot[:,k])[::-1]  # rank scores (descending)                     
            counts_rankedby_scores = binned_crime_pts_boot[:,k][rankIdx]
        
            for i in xrange(Neval):
                evalIdx_vec[i,k] = np.nansum(counts_rankedby_scores[:int(areaPercent[i]*Ncells)])  
            
            evalIdx_vec[:,k] /= float(np.nansum(binned_crime_pts_boot[:,k])) 
        elif evalIdx=='PEI':            
            rankIdx = np.argsort(scores_subgroup_boot[:,k])[::-1]  # rank scores (descending)                  
            counts_rankedby_scores = binned_crime_pts_boot[:,k][rankIdx]
            counts_ranked = np.sort(binned_crime_pts_boot[:,k])[::-1]
        
            for i in xrange(Neval):
                evalIdx_vec[i,k] = np.nansum(counts_rankedby_scores[:int(areaPercent[i]*Ncells)]) / \
                          float(np.nansum(counts_ranked[:int(areaPercent[i]*Ncells)]))   
        elif evalIdx=='Dice':
            idx_rankedby_scores = np.argsort(scores_subgroup_boot[:,k])[::-1]
            idx_rankedby_counts = np.argsort(binned_crime_pts_boot[:,k])[::-1]        
        
            for i in xrange(Neval):
                Ntop = int(areaPercent[i]*Ncells)
                area_rankedby_scores_bool = np.zeros(Ncells).astype(bool)
                area_rankedby_counts_bool = np.zeros(Ncells).astype(bool)
                area_rankedby_scores_bool[idx_rankedby_scores[:Ntop]] = True
                area_rankedby_counts_bool[idx_rankedby_counts[:Ntop]] = True
                
                evalIdx_vec[i,k] = np.sum(np.logical_and(area_rankedby_scores_bool, area_rankedby_counts_bool)) / float(Ntop)

    return evalIdx_vec 


    
def EvalIdx_array(scores, crime_data, group_seq, grid_2d, evalIdx='PAI', mask=None, areaPercent=np.linspace(0,0.05,101), 
                  boot=False, Nboots=1, r_seed=1234):
    """
    Return a prediction index array of shape (n_groups, n_evaluations) where n_evaluations is equal to the number of 
    area percentages
    """
    Ngroups = len(np.unique(group_seq))
    Neval = len(areaPercent)
    evalIdx_array = np.zeros((Ngroups,Neval,Nboots)) if boot else np.zeros((Ngroups,Neval))

    for i in xrange(Ngroups):
        if boot:
            evalIdx_array[i:(i+1),:,:] = EvalIdx_subgroup_boot(i, scores, crime_data, group_seq, grid_2d, evalIdx, mask,
                                                             areaPercent, Nboots, r_seed)
        else:
            evalIdx_array[i:(i+1),:] = EvalIdx_subgroup(i, scores, crime_data, group_seq, grid_2d, evalIdx, mask, areaPercent)
                               
    return evalIdx_array 
    
def integrate_AUC(y,x):
    """
    Using Simpson's rule to integrate samples to get the area under curve.
    """    
    area = simps(y, x)
    return area    

def save_figure(fig_name,fig):
    if not os.path.exists(os.path.dirname(fig_name)):
        os.makedirs(os.path.dirname(fig_name))
    fig.savefig(fig_name)     
    
def plot_evalIdx(evalIdx_list, areaPercent, model_names, evalIdx='PAI', sd_err=False, cmap='rainbow', ls=['-'], colors=['red'],
                 show=True, save_fig=False, fig_name='PAI.png'):
    n_models = len(evalIdx_list)
    n_areas = len(areaPercent)
    means = np.zeros((n_areas,n_models))
    stds = np.zeros((n_areas,n_models))
    auc = np.zeros(n_models) 
    
    for i in xrange(n_models):
        #if evalIdx array is a bootstrapped array (dim=3) then calculate the mean/std over all groups and bootstrapping samples
        means[:,i] = np.nanmean(evalIdx_list[i],0) if evalIdx_list[i].ndim==2 else np.apply_over_axes(np.nanmean, evalIdx_list[i], [0,2]).squeeze()
        stds[:,i] = np.nanstd(evalIdx_list[i],0) if evalIdx_list[i].ndim==2 else np.apply_over_axes(np.nanstd, evalIdx_list[i], [0,2]).squeeze()
        auc[i] = integrate_AUC(means[:,i],areaPercent) 
    
    if colors is None:    
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in np.linspace(0, 1, n_models)]
                
    #ln_styles = itertools.cycle(ls) if len(ls) != len(colors) else ls
    
    cl_ln = list(itertools.product(colors,ls))

    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
#    for i, (cl, ln) in enumerate(zip(colors,ln_styles)):
#        plt.plot(areaPercent*100, means[:,i]*100, linestyle=ln, linewidth=3, color=cl, label=model_names[i]+': AUC=%.4f' % auc[i]) 
#        if sd_err:
#            plt.errorbar(areaPercent*100, means[:,i]*100, yerr=stds[:,i]*100, ecolor=cl)

    for i, (cl, ln) in enumerate(cl_ln):
        plt.plot(areaPercent*100, means[:,i]*100, linestyle=ln, linewidth=2.5, color=cl, label=model_names[i]+': %.4f' % auc[i]) 
        if sd_err:
            plt.errorbar(areaPercent*100, means[:,i]*100, yerr=stds[:,i]*100, ecolor=cl)        
            
    plt.xlim([100*areaPercent[0], 100*areaPercent[-1]])
    plt.ylim([99*np.min(means), 101*np.max(means)])
    plt.xlabel('Area %')
    
    ax.grid(color='gray', linestyle='--',alpha=0.7)
    
    y_labels = {'PAI':'Hit rate %','PEI':'Efficiency %','Dice':'Dice %'}
    plt.ylabel(y_labels[evalIdx])
    #plt.legend(loc="lower right") 
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height * 0.9])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),title="AUC")

    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig)   
       
        
def plot_auc_series(evalIdx_list, areaPercent, model_names, cmap='rainbow', mkr=['.'],
                    show=True, save_fig=False, fig_name='AUC.png', sd_err=False):
    """
    plot AUC vs group sequence
    """
    n_models = len(evalIdx_list)
    n_groups = evalIdx_list[0].shape[0]
    
    auc_series = [np.apply_along_axis(integrate_AUC, 1, evalIdx_vec, areaPercent) for evalIdx_vec in evalIdx_list]
           
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_models)]
    markers = itertools.cycle(mkr)         
              
    fig = plt.figure(figsize=(12,10))
    for i, (cl, mk) in enumerate(zip(colors,markers)):
        if auc_series[i].ndim==1:
            plt.plot(np.arange(n_groups), auc_series[i], marker=mk, markersize=8, color=cl, label=model_names[i])
        else:
            # bootstrapped auc
            plt.plot(np.arange(n_groups), np.mean(auc_series[i],1), marker=mk, markersize=8, color=cl, label=model_names[i])
            if sd_err:
                plt.errorbar(np.arange(n_groups), np.mean(auc_series[i],1), yerr=np.std(auc_series[i],1), ecolor=cl)

    plt.xlim([0.0, n_groups])
    plt.ylim([np.min(auc_series)*0.99, np.max(auc_series)*1.01])
    plt.xlabel('group (time unit)')
    plt.ylabel('AUC')
    plt.legend(loc="lower right") 
    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig) 
    
            
def ttest_pval(sample_list, log=False, infinitesimal=1e-20):
    """
    Calculate pairwise paired t-test p-values
    """
    pval_mat = np.zeros((len(sample_list),len(sample_list))) 
    for i,a in enumerate(sample_list):
        for j,b in enumerate(sample_list):
            if i==j:
                continue
            else:
                _,pval_mat[i,j] = ttest_rel(a,b)
    
#    if log:
#        pval_mat[pval_mat==0] = infinitesimal            
#        pval_mat = np.log10(pval_mat)    
#    return pval_mat    

    return np.log10(pval_mat) if log else pval_mat    
            
    
def auc_sig_test(evalIdx_list, areaPercent, model_names, log=False, plot=True,
                 show=True, save_fig=False, fig_name='Ttest_pval.png'):
    """
    Return a matrix of p-values of paired T-test of auc's
    """
    n_models = len(model_names)
    auc_series = [np.apply_along_axis(integrate_AUC, 1, evalIdx_vec, areaPercent) for evalIdx_vec in evalIdx_list]
    pval_mat = ttest_pval(auc_series,log=log)
    
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        fig.subplots_adjust(left=0.25,bottom=0.4)
        cax = ax.imshow(pval_mat, interpolation='nearest', origin='upper')
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(model_names,rotation='vertical')
        ax.set_yticklabels(model_names)
        title = 'Paired T-test (log) p-value matrix' if log else 'Paired T-test p-value matrix'
        ax.set_title(title)
        plt.colorbar(cax,ax=ax,shrink=0.8)
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)   
          
    return pval_mat

    
        
def visualize_pred(score_list, grid_2d, mask, titles, show=True, save_fig=False, fig_name='Vis_pred.png'):
    grd_x, grd_y = grid_2d 
      
    for scores, title in zip(score_list, titles):
        pred_im = flattened_to_geoIm(scores,len(grd_x),len(grd_y), mask)
        fig = plt.figure(figsize=(12,12))
        plt.imshow(pred_im, interpolation='nearest', origin='upper', cmap='jet')
        plt.colorbar()
        plt.title(title)
        #plt.show()
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)
 
        
def visualize_actual_pts(pt_data,shpfile,show=True, save_fig=False, fig_name='Vis_pts.png'):
    """
    pt_data must be in format of a ndarray (n_pts, 2) with first column 
    being x-coords and second column being y-coords
    """
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.set_aspect('equal')
    shpfile.plot(ax=ax, color='white')
    plt.scatter(pt_data[:,0], pt_data[:,1], s=30, c='r',marker='x')
    #plt.show()
    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig)           
        

def visualize_pred_actual_superimposed(pt_data,pred_shp,field,region_shp=None,cmap='YlOrRd',title=None,show=True,save_fig=False,fig_name='Vis_pred_pts.png'):        
    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.set_aspect('equal')
    if region_shp is not None:
        region_shp.plot(ax=ax, color='white')
    pred_shp.ix[~np.isnan(pred_shp[field]),:].plot(ax=ax, column=field, cmap=cmap)
    ax.scatter(pt_data[:,0], pt_data[:,1], s=30, c='r',marker='x')
    plt.title(title)
    #plt.show()
    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig)         