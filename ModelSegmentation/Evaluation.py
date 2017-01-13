#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:51:29 2017

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

def EvalIdx_subgroup(timeIdx, scores, crime_data, group_seq, grid_2d, evalIdx='PAI', mask=None, areaPercent=np.linspace(0,1,101)):
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

    return evalIdx_vec                 
    
 
def EvalIdx_array(scores, crime_data, group_seq, grid_2d, evalIdx='PAI', mask=None, areaPercent=np.linspace(0,1,101)):
    """
    Return a prediction index array of shape (n_groups, n_evaluations) where n_evaluations is equal to the number of 
    area percentages
    """
    Ngroups = len(np.unique(group_seq))
    Neval = len(areaPercent)
    evalIdx_array = np.zeros((Ngroups,Neval))

    for i in xrange(Ngroups):
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
    
def plot_evalIdx(evalIdx_list, areaPercent, model_names, evalIdx='PAI', sd_err=False, areaPct_ub=0.1, cmap='rainbow', ls=['-'],
                 show=True, save_fig=False, fig_name=['PAI.png','PAI_zoom.png']):
    n_models = len(evalIdx_list)
    n_areas = len(areaPercent)
    means = np.zeros((n_areas,n_models))
    stds = np.zeros((n_areas,n_models))
    auc = np.zeros(n_models) 
    
    for i in xrange(n_models):
        means[:,i] = np.nanmean(evalIdx_list[i],0)
        stds[:,i] = np.nanstd(evalIdx_list[i],0)
        auc[i] = integrate_AUC(means[:,i],areaPercent) 
       
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_models)]
    ln_styles = itertools.cycle(ls)
    
    fig1 = plt.figure(figsize=(12,12))
    for i, (cl, ln) in enumerate(zip(colors,ln_styles)):
        plt.plot(areaPercent*100, means[:,i]*100, linestyle=ln, linewidth=3, color=cl, label=model_names[i]+': AUC=%.4f' % auc[i]) 
        if sd_err:
            plt.errorbar(areaPercent*100, means[:,i]*100, yerr=stds[:,i]*100, ecolor=cl)

    plt.xlim([100*areaPercent[0], 100*areaPercent[-1]])
    plt.ylim([99*np.min(means), 101*np.max(means)])
    plt.xlabel('Top risky area %')
    plt.ylabel(evalIdx+' %')
    #plt.axis('equal')
    plt.legend(loc="lower right") 
    if save_fig:
        save_figure(fig_name[0],fig1)
    if not show:
        plt.close(fig1)   
       
    # zoom-in (top % area <= x%: upper bound)
    if areaPct_ub < 1:
        means_sub = means[areaPercent<=areaPct_ub,:]
        stds_sub = stds[areaPercent<=areaPct_ub,:]
        areaPercent_sub = areaPercent[areaPercent<=areaPct_ub]      
        auc_sub = [integrate_AUC(means_sub[:,i],areaPercent_sub) for i in xrange(n_models)] ####

        fig2 = plt.figure(figsize=(12,12))
        for i, (cl, ln) in enumerate(zip(colors,ln_styles)):
            plt.plot(areaPercent_sub*100, means_sub[:,i]*100, linestyle=ln, linewidth=3, color=cl, label=model_names[i]+': AUC=%.4f' % auc_sub[i])
            if sd_err:
                plt.errorbar(areaPercent_sub*100, means_sub[:,i]*100, yerr=stds_sub[:,i]*100, ecolor=cl) 


        plt.xlim([100*areaPercent[0], 100*areaPct_ub])
        plt.ylim([99*np.min(means_sub), 101*np.max(means_sub)])
        plt.xlabel('Top risky area %')
        plt.ylabel(evalIdx+' %')
        plt.legend(loc="lower right")
        if save_fig:
            save_figure(fig_name[1],fig2)
        if not show:
            plt.close(fig2) 
            

def plot_auc_series(evalIdx_list, areaPercent, model_names, areaPct_ub=0.1, cmap='rainbow', mkr=['.'],
                    show=True, save_fig=False, fig_name=['AUC.png','AUC_zoom.png']):
    """
    plot AUC vs group sequence
    """
    n_models = len(evalIdx_list)
    n_groups = evalIdx_list[0].shape[0]
    
    auc_series = [np.apply_along_axis(integrate_AUC, 1, evalIdx_vec, areaPercent) for evalIdx_vec in evalIdx_list]
           
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_models)]
    markers = itertools.cycle(mkr)         
              
    fig1 = plt.figure(figsize=(12,10))
    for i, (cl, mk) in enumerate(zip(colors,markers)):
        plt.plot(np.arange(n_groups), auc_series[i], marker=mk, markersize=8, color=cl, label=model_names[i])

    plt.xlim([0.0, n_groups])
    plt.ylim([np.min(auc_series)*0.99, np.max(auc_series)*1.01])
    plt.xlabel('group (time unit)')
    plt.ylabel('AUC')
    plt.legend(loc="lower right") 
    if save_fig:
        save_figure(fig_name[0],fig1)
    if not show:
        plt.close(fig1) 
    
    # sequence of auc of zoomed-in (top % area <= x% upper bound) areas
    if areaPct_ub < 1:
        areaPercent_sub = areaPercent[areaPercent<=areaPct_ub]      
        auc_series_sub = [np.apply_along_axis(integrate_AUC, 1, evalIdx_vec[:,areaPercent<=areaPct_ub], areaPercent_sub) for evalIdx_vec in evalIdx_list]

        fig2 = plt.figure(figsize=(12,10))
        for i, (cl, mk) in enumerate(zip(colors,markers)):
            plt.plot(np.arange(n_groups), auc_series_sub[i], marker=mk, markersize=8, color=cl, label=model_names[i])

        plt.xlim([0.0, n_groups])
        plt.ylim([np.min(auc_series_sub)*0.99, np.max(auc_series_sub)*1.01])
        plt.xlabel('group (time unit)')
        plt.ylabel('AUC')
        plt.legend(loc="lower right")
        if save_fig:
            save_figure(fig_name[1],fig2)
        if not show:
            plt.close(fig2) 

            
def ttest_pval(sample_list, log=False):
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
                
    return np.log10(pval_mat) if log else pval_mat        
            
    
def auc_sig_test(evalIdx_list, areaPercent, model_names, areaPct_ub=0.1, log=False, plot=True,
                 show=True, save_fig=False, fig_name=['Ttest_pval.png','Ttest_pval_zoom.png']):
    """
    Return a matrix of p-values of paired T-test of auc's
    """
    n_models = len(model_names)
    auc_series = [np.apply_along_axis(integrate_AUC, 1, evalIdx_vec, areaPercent) for evalIdx_vec in evalIdx_list]
    pval_mat = ttest_pval(auc_series,log=log)
    
    if plot:
        fig1, ax = plt.subplots(1, 1, figsize=(8,6))
        fig1.subplots_adjust(left=0.25,bottom=0.4)
        cax = ax.imshow(pval_mat, interpolation='nearest', origin='upper')
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(model_names,rotation='vertical')
        ax.set_yticklabels(model_names)
        title = 'Paired T-test (log) p-value matrix' if log else 'Paired T-test p-value matrix'
        ax.set_title(title)
        plt.colorbar(cax,ax=ax,shrink=0.8)
        if save_fig:
            save_figure(fig_name[0],fig1)
        if not show:
            plt.close(fig1)   
            
    if areaPct_ub < 1:
        areaPercent_sub = areaPercent[areaPercent<=areaPct_ub]      
        auc_series_sub = [np.apply_along_axis(integrate_AUC, 1, evalIdx_vec[:,areaPercent<=areaPct_ub], areaPercent_sub) for evalIdx_vec in evalIdx_list]
        pval_mat_sub = ttest_pval(auc_series_sub,log=log)
        
        if plot:            
            fig2, ax = plt.subplots(1, 1, figsize=(8,6))
            fig2.subplots_adjust(left=0.25,bottom=0.4)
            cax = ax.imshow(pval_mat_sub, interpolation='nearest', origin='upper')
            ax.set_xticks(np.arange(n_models))
            ax.set_yticks(np.arange(n_models))
            ax.set_xticklabels(model_names,rotation='vertical')
            ax.set_yticklabels(model_names)
            title = 'Paired T-test (log) p-value matrix' if log else 'Paired T-test p-value matrix'
            ax.set_title(title)
            plt.colorbar(cax,ax=ax,shrink=0.8)
            if save_fig:
                save_figure(fig_name[1],fig2)
            if not show:
                plt.close(fig2) 
          
    return pval_mat
        
def visualize_pred(score_list, grid_2d, mask, titles):
    grd_x, grd_y = grid_2d 
      
    for scores, title in zip(score_list, titles):
        pred_im = flattened_to_geoIm(scores,len(grd_x),len(grd_y), mask)
        plt.figure(figsize=(8,8))
        plt.imshow(pred_im, interpolation='nearest', origin='upper', cmap='jet')
        plt.colorbar()
        plt.title(title)
        plt.show()
    
def visualize_actual_pts(pt_data,shpfile):
    """
    pt_data must be in format of a ndarray (n_pts, 2) with first column 
    being x-coords and second column being y-coords
    """
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_aspect('equal')
    shpfile.plot(ax=ax, color='white')
    plt.scatter(pt_data[:,0], pt_data[:,1], s=8, c='r',marker='.')
    plt.show()            