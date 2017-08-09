#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
====================================
Feature selection
Reference:
    http://scikit-learn.org/stable/modules/feature_selection.html
====================================
Created on Mon Sep 19 14:59:19 2016

@author: xiaomuliu
"""
import numpy as np
from sklearn.pipeline import Pipeline
#NOTE: scikit learn v0.18 changed 'cross_validation' module and 'grid_search' module to 'model_selection'
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, mutual_info_classif, chi2, RFE
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import os

def save_figure(fig_name,fig):
    if not os.path.exists(os.path.dirname(fig_name)):
        os.makedirs(os.path.dirname(fig_name))
    fig.savefig(fig_name)

def univar_score(X, y, feature_names, K_best=None, percentile=None, criterion='f_score', rank=True, 
                 plot=True, plot_pval=False, save_fig=False, fig_name='univar_rank.png', show=True):
    """
    Select features according to a number/percentile of the highest scores
    Plot scores and/or p-values for each features
    K_best: K features to keep (if both K_best and percentile are provided, percentile will be suppressed.)
    percentile: Percent of features to keep 
    criterion should be one of the following: f_score (default), mutual_info, chi2 (needs X to be non-negative)
    NOTE: F-test captures only linear dependency. Mutual information can capture any kind of dependency between variables
    """
    
    score_funcs = dict(f_score=f_classif, mutual_info=mutual_info_classif, chi2=chi2)
    score_func = score_funcs[criterion]
    P = X.shape[1]    

    if K_best is not None:
        selector = SelectKBest(score_func, k=K_best)
        N_keep = K_best
    elif percentile is not None:
        selector = SelectPercentile(score_func, percentile=percentile)    
        N_keep = int(percentile/float(100)*P)
    selector.fit(X,y);

    # scores
    feature_scores = selector.scores_
    rank_idx = np.argsort(feature_scores)[::-1] if rank else np.ones(P)
    
    ranked_scores = feature_scores[rank_idx][:N_keep]
    ranked_features = feature_names[rank_idx][:N_keep]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        fig.subplots_adjust(bottom=0.4)
        ax.bar(np.arange(N_keep), ranked_scores, width=.3,
               label='Univariate scores', color='b')
        ax.set_xticks(np.arange(N_keep))
        ax.set_xticklabels(ranked_features,rotation='vertical',horizontalalignment='center')
        ax.set_xlim(0, N_keep+1)
        ax.set_ylabel('score')
        ax.set_title('Feature Scores')
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)

    # p-values (only supports f_score and chi2)
    if selector.pvalues_ is not None:  
        feature_pvals = np.log10(selector.pvalues_)
        feature_pvals /= feature_pvals.max()
        ranked_pvals = feature_pvals[rank_idx]
         
        if plot_pval:
            fig, ax = plt.subplots(1, 1, figsize=(12,11))
            fig.subplots_adjust(bottom=0.4)
            ax.bar(np.arange(N_keep), ranked_pvals, width=.3,
                    label=r'P-values of univariate scores ($Log(p_{value})$)', color='b')
            ax.set_xticklabels(ranked_features,rotation='vertical',horizontalalignment='center')
            ax.set_xticks(np.arange(N_keep))
            ax.set_xlim(0, N_keep+1)
            ax.set_xlabel('feature index')
            ax.set_ylabel('normalized log p-value')
            ax.set_title('P-values of Feature Scores')
        if not show:
            plt.close(fig)    
    
    return dict(ranked_features=ranked_features, ranks=rank_idx, scores=feature_scores, pvals=feature_pvals) 


def cv_scores(X, y, estimators, param_grids, CVobj):
    score_means = []
    score_stds = []            
    for estimator, param_grid in zip(estimators, param_grids):     
        search = GridSearchCV(estimator, param_grid, cv=CVobj, n_jobs=1)    
        search.fit(X, y);
        score_means.append(search.cv_results_['mean_test_score'])
        score_stds.append(search.cv_results_['std_test_score']) 
    return score_means, score_stds      
    
    
def plot_FS_cv(ax, score_means, score_stds, clf_names, xvals, xlab='number of selected features', cmap='rainbow'):    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, len(clf_names))]    
    for i, clf_name in enumerate(clf_names):    
        ax.errorbar(xvals, score_means[i], score_stds[i], color=colors[i], label=clf_name)
    
    ax.set_xlim(xvals[0]-0.01*(np.ptp(xvals)), xvals[-1]+0.01*(np.ptp(xvals)))
    ax.set_xlabel(xlab)
    ax.set_ylabel('accuracy')
    ax.legend(loc='lower right')


      
def univar_FS_cv(X, y, clfs, clf_names, CVobj, Ks=None, percentiles=None, criterion='f_score', 
                 plot=True, save_fig=False, fig_name='univar_CV.png', show=True):
    """
    Select a series of candidate number of features then do classification. 
    Plot the cross-validated accurarcy vs. number of selected features
    clfs: a list of classifier objects
    CVobj: cross validation object
    Ks: a list of numbers of features to keep (if both Ks and percentiles are provided, percentiles will be suppressed.)
    percentiles: a list of percents of features to keep 
    criterion should be one of the following: f_score (default), mutual_info, chi2
    """   
    score_funcs = dict(f_score=f_classif, mutual_info=mutual_info_classif, chi2=chi2)
    score_func = score_funcs[criterion]
    
    if Ks is not None:
        selector = SelectKBest(score_func)
        param_grid = dict(selector__k=Ks)
        n_keep = Ks
    elif percentiles is not None:
        selector = SelectPercentile(score_func)
        param_grid = dict(selector__percentile=percentiles)
        n_keep = np.round(percentiles/float(100)*X.shape[1])
                
    pipes = [Pipeline([('selector', selector), ('classifier', clf)]) for clf in clfs]   
    param_grids = [param_grid]*len(clfs)
        
    score_means, score_stds = cv_scores(X, y, pipes, param_grids, CVobj)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        plot_FS_cv(ax, score_means, score_stds, clf_names, n_keep)
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig) 
            
    return score_means, score_stds 
    

def RFE_rank(X, y, clfs, feature_names, K_best=None, percentile=None):
    """
    Features are ranked by recursively pruing the feature associated with the smallest absolute weight
    Return a list where each element is the ranked feature set corresponding to the estimator in 'clfs'
    """
    if K_best is not None:
        N_keep = K_best
    elif percentile is not None:    
        N_keep = int(percentile/float(100)*X.shape[1])
    else:
        N_keep = X.shape[1]
  
    ranked_features = []
    rank_idx = []  
    for clf in clfs:    
        rfe = RFE(clf, n_features_to_select=1, step=1)   
        rfe.fit(X,y);
        ranked_features.append(feature_names[np.argsort(rfe.ranking_)][:N_keep])
        rank_idx.append(rfe.ranking_)

    return dict(ranked_features=ranked_features, ranks=rank_idx) 
    
    
def RFE_FS_cv(X, y, clfs, clf_names, CVobj, rm_per_step=1, n_keep=None, 
              plot=True, save_fig=False, fig_name='RFE_CV.png', show=True):
    """
    Select a series of candidate number of features through RFE then do classification. 
    Plot the cross-validated accurarcy vs. number of selected features
    rm_per_step: the (integer) number of features to remove at each iteration.
    n_keep: recursively-pruning stops when 'n_keep' number of features to select is reached
    """    
    # NOTE: this task could be done using 
    # RFECV(estimator=clf, step=rm_per_step, cv=CVobj scoring='accuracy'
    # However, RFECV does not provide sufficient info of cross validation results   
    P = X.shape[1]
    if n_keep is None:
        n_keep=np.arange(np.ceil((P-1) / float(rm_per_step))) + 1 
    param_grid = {'n_features_to_select':n_keep}
  
    estimators = [RFE(estimator=clf, step=rm_per_step) for clf in clfs]
    param_grids = [param_grid]*len(clfs)
                  
    score_means, score_stds = cv_scores(X, y, estimators, param_grids, CVobj)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        plot_FS_cv(ax, score_means, score_stds, clf_names, n_keep)
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)
            
    return score_means, score_stds 

def plot_L1_path(X, y, clfs, clf_names, reg_params, save_fig=False, fig_name='L1_path.png', show=True):
    """
    clfs: a list of classifiers (only support logistic regression with L1 penalty (lasso) and linear SVM with L1 penalty)
    reg_params: regularization parameters for each classifier in 'clfs'
    """ 
    if not isinstance(clfs,list):
        clfs = [clfs]
    if not isinstance(clf_names,list):
        clf_names = [clf_names] 

    if len(clfs)==1:
        fig, axes = plt.subplots(1, 1, figsize=(10,8))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, len(clfs), figsize=(14,8))                 
    for clf, clf_name, ax in zip(clfs, clf_names, axes):        
        coefs = []
        for c in reg_params:
            clf.set_params(C=c)
            clf.fit(X, y);
            coefs.append(clf.coef_.ravel().copy())
        
        ax.plot(np.log10(reg_params), coefs)
        ax.set_xlabel('log(C)')
        ax.set_ylabel('Coefficients')
        ax.set_title(clf_name)
    if save_fig:
        save_figure(fig_name,fig)
    if not show:
        plt.close(fig)
        
def L1_score(X, y, clfs, clf_names, feature_names, K_best=None, percentile=None, criterion='max',
             rank=True, plot=True, save_fig=False, fig_name='L1_rank.png', show=True):
    """
    L1 stability feature selection
    # currently only support logistic regression as the base model
    criterion: determines how the scores are calculated over all the scores of different regularization parameters
    either 'max' or 'mean'
    """
    if not isinstance(clfs,list):
        clfs = [clfs]
    if not isinstance(clf_names,list):
        clf_names = [clf_names] 

    if K_best is not None:
        N_keep = K_best
    elif percentile is not None:    
        N_keep = int(percentile/float(100)*X.shape[1])
    else:
        N_keep = X.shape[1]
  
    ranked_features = []
    scores = [] 
    
    for clf in clfs:
        clf.fit(X,y);
        # NOTE:
        # clf.all_scores_ : is of shape = [n_features, n_reg_parameter]
        # Tclf.scores_ is the max of all_scores_ over all regularization parameters.
        if criterion=='max':
            feature_scores = clf.scores_ 
        elif criterion=='mean':
            feature_scores = np.mean(clf.all_scores_,1) 
        rank_idx = np.argsort(feature_scores)[::-1][:N_keep] if rank else np.ones(X.shape[1])        
        rank_idx = np.argsort(feature_scores)[::-1]
        scores.append(feature_scores[rank_idx][:N_keep])
        ranked_features.append(feature_names[rank_idx][:N_keep])
    
    if plot:
        if len(clfs)==1:
            fig, axes = plt.subplots(1, 1, figsize=(12,11))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, len(clfs), figsize=(16,11))
        fig.subplots_adjust(bottom=0.4)
        for clf_name, f_names, score, ax in zip(clf_names, ranked_features, scores, axes): 
            ax.bar(range(N_keep), score, width=.3, color='b', align='center')
            ax.set_xticks(range(N_keep))
            ax.set_xticklabels(f_names,rotation='vertical',horizontalalignment='center')
            ax.set_xlim(-1, N_keep+1)
            ax.set_ylabel('Feature Scores')
            ax.set_title(clf_name)
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)
            
    return dict(ranked_features=ranked_features, ranks=rank_idx, scores=scores)
    
        
def L1_FS_cv(X, y, clfs, clf_names, CVobj, reg_params, plot=True, save_fig=False, fig_name='L1_CV.png', show=True):
    """
    reg_params: a list of regularization parameters(C values) to be searched over
    """    
    param_grids = [{'C':reg_params}]*len(clfs)
           
    score_means, score_stds = cv_scores(X, y, clfs, param_grids, CVobj)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        plot_FS_cv(ax, score_means, score_stds, clf_names, xvals=np.log10(reg_params), xlab='log(C)')
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig) 
            
    return score_means, score_stds     

def tree_score(X, y, clfs, clf_names, feature_names, K_best=None, percentile=None, plot=True, 
               save_fig=False, fig_name='feature_imp.png', show=True):    
    """
    Use feature-importance attributes of tree-based ensemble methods to rank features
    Return ranked features and corresponding importance scores
    """
    if not isinstance(clfs,list):
        clfs = [clfs]
    if not isinstance(clf_names,list):
        clf_names = [clf_names] 

    if K_best is not None:
        N_keep = K_best
    elif percentile is not None:    
        N_keep = int(percentile/float(100)*X.shape[1])
    else:
        N_keep = X.shape[1]
  
    ranked_features = []
    rank_idx = []
    scores = [] 
    for clf in clfs:     
        clf.fit(X,y);
        importances = clf.feature_importances_
        ranking = np.argsort(importances)[::-1]
        rank_idx.append(ranking)
        scores.append(importances[ranking][:N_keep])
        ranked_features.append(feature_names[ranking][:N_keep])
    
    if plot:
        if len(clfs)==1:
            fig, axes = plt.subplots(1, 1, figsize=(12,11))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, len(clfs), figsize=(16,11))
        fig.subplots_adjust(bottom=0.4)
        for clf_name, f_names, score, ax in zip(clf_names, ranked_features, scores, axes): 
            ax.bar(range(N_keep), score, width=.3, color='b', align='center')
            ax.set_xticks(range(N_keep))
            ax.set_xticklabels(f_names,rotation='vertical',horizontalalignment='center')
            ax.set_xlim(-1, N_keep+1)
            ax.set_ylabel('Feature Importance')
            ax.set_title(clf_name)
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)
            
    return dict(ranked_features=ranked_features, ranks=rank_idx, scores=scores)


def SelectFromModel_Kbest(X, y, clf, K_best=1):
    """
    selecting-k-best version of 'scikit learn.feature selection.SelectFromModel' 
    """
    clf.fit(X, y);
    Kbest_idx = np.argsort(clf.feature_importances_)[::-1][:K_best]
    return X[:,Kbest_idx]
    
    
def tree_FS_cv(X, y, clfs, clf_names, CVobj, percentiles=np.linspace(10,100,10),
               plot=True, save_fig=False, fig_name='tree_CV.png', show=True):
    """
    NOTE: THIS FUNCTION DOES WORK FOR NOW!
    Although 'FunctionTransformer' can take 'y' argument, whe applying method 'fit_tranform', 
    the 'transform' function only takes 'X' argument. In the source code 
        https://github.com/scikit-learn/scikit-learn/blob/412996f/sklearn/base.py#L470
   
        def fit_transform(self, X, y=None, **fit_params):
            ...
            return self.fit(X, y, **fit_params).transform(X)
     
    However, our custormized FunctionTransformer 'SelectFromModel_Kbest' needs 'y' as well
    in the 'tranform'. So this class method needs to be modified.          
        
        
    Select a series of candidate number of features then do classification. 
    Plot the cross-validated accurarcy vs. number of selected features
    clfs: a list of classifier objects
    CVobj: cross validation object
    percentiles: a list of percents of features to keep 
    """
    n_keep = np.round(percentiles*0.01*X.shape[1])   
    pipes = [Pipeline([('selector', FunctionTransformer(SelectFromModel_Kbest, pass_y=True)),\
                       ('classifier', clf)]) for clf in clfs] 
    
    param_grids = []   
    for clf in clfs:
        param_grids.append(dict(selector__kw_args=[{'clf':clf,'K_best':k} for k in n_keep]))
        
    score_means, score_stds = cv_scores(X, y, pipes, param_grids, CVobj)
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        plot_FS_cv(ax, score_means, score_stds, clf_names, n_keep)
        if save_fig:
            save_figure(fig_name,fig)
        if not show:
            plt.close(fig)
            
    return score_means, score_stds     

  