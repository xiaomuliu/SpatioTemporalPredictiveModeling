#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
===================================
Perform multiple feature selection strategies
===================================
Created on Wed Oct 26 16:15:27 2016

@author: xiaomuliu
"""

import numpy as np
from sklearn import preprocessing, linear_model, ensemble
#NOTE: scikit learn v0.18 changed 'cross_validation' module and 'grid_search' module to 'model_selection'
from sklearn.model_selection import StratifiedKFold 
import FeatureSelection as fs


def bag_feature_selection(X, y, feature_names, CVobj, K_best=None, pct_best=None, percentiles=np.linspace(10,100,10), 
                          Cs_l1=np.logspace(-4, -1, 7), C_l2=0.01, rand_L1_params=None, RF_params=None, GBM_params=None,
                          scaling=None, rand_seed=1234, plot=True, save_fig=False, fig_names=None, show=True):
    """
    Bag four different types of feature selection process. 
    1. Univariate feature selection (f-score based ranking)
    2. Recursive feature elimination
    3. L1-based feature selection 
    4. Tree-based feature selection
    For 1,2,3, logistic regression and linear SVM are used as base models
    For 4, random forest and gradient tree boosting are used as base models
    For each process, a plot of (cross-validation) accuracy vs. number of selected features 
    (or regularization parameter for L1-based methods) are given 
    
    Input: 
    CVobj: cross validation object
    K_best: K best features to keep 
    pct_best: Percent of features to keep (If both K_best and pct_best are provided, pct_best will be suppressed.)
    percentiles: a list of percents of features to keep and search over
    Cs_l1: a list of C values (1/regularization parameters) to be searched over
    C_l2, RF_params, GBM_param: default parameter for base models (logistic, svm, forest)
    fig_names: a dict where keys and items are figure category and the corresponding filenames to save
        
    Return a dict containing ranked features from univariate feature selection,
    recursive feature elimination (a list where each element corresponds to a model), 
    and tree-based (a list where each element corresponds to a model)
    """
    
    p = X.shape[1]
    if K_best is not None:
        n_selected = K_best
    elif pct_best is not None:    
        n_selected = int(pct_best*0.01*p)
    else:
        n_selected = p
      
    if scaling is not None:
        if scaling=='standard':
            X = preprocessing.StandardScaler().fit_transform(X)
        elif scaling=='minmax':
            X = preprocessing.MinMaxScaler().fit_transform(X)    
        
        
    ranked_features = {}    
    feature_ranks = {}
    # Note: The combination of penalty='l2' and loss='hinge' are not supported when dual=False
#    clfs = [linear_model.LogisticRegression(C=C_l2, penalty='l2'), svm.LinearSVC(C=C_l2, penalty='l2', loss='squared_hinge')]
#    clf_names = ['logistic regression','linear SVM']
    clfs = [linear_model.LogisticRegression(C=C_l2, penalty='l2')]
    clf_names = ['logistic regression']

    # ******************* Univariate feature selection ******************* #    
    univar_scores= fs.univar_score(X, y, feature_names, K_best=n_selected, criterion='f_score',\
                                   plot=plot, save_fig=save_fig, fig_name=fig_names['univar_rank'],show=show)    
    ranked_features['univar'], feature_ranks['univar'] = univar_scores['ranked_features'], univar_scores['ranks']
    fs.univar_FS_cv(X, y, clfs, clf_names, CVobj, percentiles=percentiles, criterion='f_score',\
                    plot=plot, save_fig=save_fig, fig_name=fig_names['univar_CV'], show=show);  
      
    # ******************* Recursive feature elimination ******************* #
    RFE_scores = fs.RFE_rank(X, y, clfs, feature_names, K_best=n_selected) 
    ranked_features['RFE'], feature_ranks['RFE'] = RFE_scores['ranked_features'], RFE_scores['ranks']
#    # The following code takes very long time to run
#    n_keep = np.round(percentiles*0.01*p).astype(int)   
#    fs.RFE_FS_cv(X, y, clfs, clf_names, CVobj, rm_per_step=1, n_keep=n_keep,\
#                 plot=plot, save_fig=save_fig, fig_name=fig_names['RFE_CV'],show=show);
             
    #******************* L1-based feature selection *************************#    
    if rand_L1_params is None:
        rand_L1_params = dict(C=Cs_l1, scaling=0.5, sample_fraction=0.75, n_resampling=100, 
                              selection_threshold=0.25, random_state=rand_seed, n_jobs=1)
        
    rand_L1 = linear_model.RandomizedLogisticRegression(**rand_L1_params)
    L1_scores = fs.L1_score(X, y, [rand_L1], ['randomized sparse model'], feature_names, K_best=n_selected, criterion='mean',\
                            plot=plot, save_fig=save_fig, fig_name=fig_names['L1_rank'], show=show)
    ranked_features['L1'], feature_ranks['L1'] = L1_scores['ranked_features'], L1_scores['ranks']

    # NOTE: For L1 logistic regression,‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty.
    clfs[0].set_params(**{'penalty':'l1','solver':'liblinear'}) 
#    clfs[1].set_params(**{'penalty':'l1','dual':False})
            
    fs.plot_L1_path(X, y, clfs, clf_names, Cs_l1, save_fig=save_fig, fig_name=fig_names['L1_path'], show=show);    
    fs.L1_FS_cv(X, y, clfs, clf_names, CVobj, Cs_l1, \
                plot=plot, save_fig=save_fig, fig_name=fig_names['L1_CV'], show=show);    
          
    # **************** Tree-based feature selection ************************ #
    # For tree-based methods, raw features values could be used 
    if RF_params is None:
        RF_params = {'n_estimators': 1000,'max_features': 'auto','min_samples_split': 1, 
                     'bootstrap': True, 'oob_score': True, 'random_state': rand_seed, 'n_jobs': -1}
    if GBM_params is None:
        GBM_params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 1,
                      'max_features': 'auto', 'min_samples_leaf': 1, 'random_state': rand_seed}
                  
    RF = ensemble.RandomForestClassifier(**RF_params)              
    GBM = ensemble.GradientBoostingClassifier(**GBM_params)
    clfs = [RF, GBM]
    clf_names = ['Random Forest','Gradient Boosting Machine']

    tree_scores = fs.tree_score(X, y, clfs, clf_names, feature_names, K_best=n_selected, plot=plot,\
                                save_fig=save_fig, fig_name=fig_names['feature_imp'], show=show)
    ranked_features['tree'], feature_ranks['tree'] = tree_scores['ranked_features'], tree_scores['ranks']
#    # DO NOT DO THIS. See doctring of function fs.tree_FS_cv
#    fs.tree_FS_cv(X, y, clfs, clf_names, CVobj, percentiles, plot=plot,\
#                  save_fig=save_fig, fig_name=fig_names['tree_CV'], show-show)
    
    return ranked_features, feature_ranks
    
if __name__ == '__main__':   
    import collections
    import re
    import cPickle as pickle
    import pandas as pd
    import sys, os
    import time    
    sys.path.append('..') 
    import StructureData.LoadData as ld            
    from Misc.ComdArgParse import ParseArg  

    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    target_crime = re.search('(?<=targetcrime=)(\w+)', infiles).group(1)     
    feature_pkl = re.search('(?<=feature=)([\w\./]+)',infiles).group(1)
    
    filePath_save = outpath if outpath is not None else "./"
                
    kfolds = int(re.search('(?<=kfolds=)(\d+)',params).group(1))
    r_seed = int(re.search('(?<=rseed=)(\d+)',params).group(1))  
    CV_skf = StratifiedKFold(n_splits=kfolds, shuffle=False, random_state=r_seed)
    
    # test parameters
    rand_L1_params = dict(C=np.logspace(-2, 2, 9), scaling=0.5, sample_fraction=0.75, n_resampling=100, 
                          selection_threshold=0.25, random_state=r_seed, n_jobs=1)
    RF_params = dict(n_estimators=1000, max_features='auto', min_samples_split=1, 
                     bootstrap=True, oob_score=True, random_state=r_seed, n_jobs=-1)
    GBM_params = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.9,
                      max_features=0.5, min_samples_leaf=1, random_state=r_seed)                 
    params = dict(CVobj=CV_skf, pct_best=100, percentiles=np.linspace(10,100,10), Cs_l1=np.logspace(-2, 2, 9), 
                  C_l2=100, rand_L1_params=rand_L1_params, RF_params=RF_params, GBM_params=GBM_params, 
                  scaling='minmax',rand_seed=r_seed, plot=True, save_fig=True, show=False)    
        
    
    fig_categories = ['univar_rank','univar_CV','L1_path','L1_CV','L1_rank','feature_imp']
    fig_savefile = {}

    ranked_features = collections.defaultdict(dict) # a nested dict
    feature_ranks = collections.defaultdict(dict) # a nested dict
    
    # ************************ load data *****************************#
    X_train, y_train, _, _, feature_names = ld.load_train_test_data(feature_pkl,'h5',target_crime,'Label')

    
    # ****************** bagged feature selection ************************#
    for fig_cat in fig_categories:
        fig_savefile[fig_cat] = filePath_save+'Figures/FeatureSelection/'+target_crime+'/'+fig_cat+'.png'
    
    params.update(dict(X=X_train,y=y_train,feature_names=feature_names,fig_names=fig_savefile))
    start = time.time()    
    ranked_features, feature_ranks = bag_feature_selection(**params)
    end = time.time()
    print(end-start)

    # Save ranked features
             
    # From each crime type, arrange ranked features in a dataframe where each feature name is a row index, 
    # and columns are feature selection methods
    feature_ranking = dict(ranked_features=ranked_features, feature_names=feature_names, feature_ranks=feature_ranks)
    methods = ['univar (f-score)','RFE (logit)','rand-L1 (logit)','RF','GBM']
    categories = ['univar','RFE','L1','tree']

    colidx = methods
    rowidx = np.arange(1,len(feature_names)+1)
    # For each call type, 'feature ranks' and 'ranked features' are of type ndarray. 
    # If two or more methods belong to the same category, 'feature ranks' and 'ranked features' 
    # are stored in a list,respectively
    
    rank_list = []
    feature_list = []

    for cat in categories:
        r_l = feature_ranking['feature_ranks'][cat]
        f_l = feature_ranking['ranked_features'][cat]
        if isinstance(r_l,list):
            for r in r_l:
                rank_list.append(r+1) # Rank starts from '1'
        else:
            rank_list.append(r_l+1)
        if isinstance(f_l,list):
            for f in f_l:
                feature_list.append(f)
        else:
            feature_list.append(f_l)    
                
    feature_ranking_df1 = pd.DataFrame(np.array(rank_list).T.astype(int), index=feature_names, columns=colidx)
    feature_ranking_df2 = pd.DataFrame(np.array(feature_list).T, index=rowidx, columns=colidx) 
    feature_ranking_df2.index.name = 'rank'
    
             
    savefile_csv1 = filePath_save+'FeatureRank/'+target_crime+'_feature_ranking_dataframe1.csv'
    if not os.path.exists(os.path.dirname(savefile_csv1)):
        os.makedirs(os.path.dirname(savefile_csv1))
    feature_ranking_df1.to_csv(savefile_csv1) 
    savefile_csv2 = filePath_save+'FeatureRank/'+target_crime+'_feature_ranking_dataframe2.csv'
    feature_ranking_df2.to_csv(savefile_csv2) 
    
    savefile_df1 = filePath_save+'FeatureRank/'+target_crime+'_feature_ranking_dataframe1.pkl'
    with open(savefile_df1,'wb') as output:
        pickle.dump(feature_ranking_df1, output, pickle.HIGHEST_PROTOCOL)
        
    savefile_df2 = filePath_save+'FeatureRank/'+target_crime+'_feature_ranking_dataframe2.pkl'
    with open(savefile_df2,'wb') as output:
        pickle.dump(feature_ranking_df2, output, pickle.HIGHEST_PROTOCOL) 