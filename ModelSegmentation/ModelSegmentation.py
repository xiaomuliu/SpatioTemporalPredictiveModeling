#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:37:31 2016

@author: xiaomuliu
"""

import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
#NOTE: scikit learn v0.18 changed 'cross_validation' module and 'grid_search' module to 'model_selection'
from sklearn.model_selection import GridSearchCV 
import Evaluation as ev
    

def fit_pred(X_train, y_train, X_test, clf, clf_name, tune_param_dict, CV_obj, region, standardize=True):
    """
    Tune model('clf')'s parameters in tune_param_dicts 
    using training data and cross validation. Then fit the entire training data
    using the optimal parameters and predict the test data
    """  
    zscore_scaler = preprocessing.StandardScaler() 
    pipe = Pipeline(steps=[('standardize',zscore_scaler), (clf_name, clf)]) if standardize else \
           Pipeline(steps=[(clf_name, clf)])
    estimator = GridSearchCV(pipe, tune_param_dict, cv=CV_obj, n_jobs=1)
    estimator.fit(X_train, y_train); 
    if hasattr(clf,'predict_prob'):       
        y_scores = estimator.best_estimator_.predict_proba(X_test)
    elif hasattr(clf,'decision_function'):
        y_scores = estimator.best_estimator_.decision_function(X_test)
    
    if y_scores.ndim>1:
        y_scores = y_scores[:,1]  # Some classifier return scores for both classes
    
    return y_scores

   
def get_eval_lists(evalIdx_names, score_list, CrimeData, groups, grid_2d, mask, areaPercent):
    eval_dict = {} 
    for ev_idx in evalIdx_names: 
        evalIdx_list = [ev.EvalIdx_array(scores, CrimeData, groups, grid_2d, ev_idx, mask, areaPercent[ev_idx]) for scores in score_list]
        eval_dict[ev_idx] = evalIdx_list                
    
    return eval_dict
                 

def plot_eval_pred(eval_dict, evalIdx_names, model_name_list, areaPercent, areaPct_ub, fig_names):
    for ev_idx in evalIdx_names:
        ev.plot_evalIdx(eval_dict[ev_idx], areaPercent[ev_idx], model_name_list, ev_idx, sd_err=False, areaPct_ub=areaPct_ub[ev_idx], 
                    ls=['-','--'],show=False, save_fig=True, fig_name=fig_names[ev_idx]['EvalIdx']);    
        ev.plot_auc_series(eval_dict[ev_idx], areaPercent[ev_idx], model_name_list, areaPct_ub[ev_idx],
                           mkr=['.','x'],show=False, save_fig=True, fig_name=fig_names[ev_idx]['AUC']);
        ev.auc_sig_test(eval_dict[ev_idx], areaPercent[ev_idx], model_name_list, areaPct_ub[ev_idx], log=True,
                        plot=True, show=False, save_fig=True, fig_name=fig_names[ev_idx]['t_pval']);


                 
if __name__ == '__main__':   
    import re
    import time
    from ModelSpec import get_model_params, get_cv_obj
    import sys
    sys.path.append('..') 
    import StructureData.LoadData as ld 
    from Misc.ComdArgParse import ParseArg    

    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    grid_pkl = re.search('(?<=grid=)([\w\./]+)',infiles).group(1)
    crime_pkl = re.search('(?<=crime=)([\w\./]+)',infiles).group(1)
    group_pkl = re.search('(?<=group=)([\w\./]+)',infiles).group(1)
    feature_pkl = re.search('(?<=feature=)([\w\./]+)',infiles).group(1)
    feature_balanced_pkl = re.search('(?<=featureB=)([\w\./]+)',infiles).group(1)
    baseline_pkl = re.search('(?<=baseline=)([\w\./]+)',infiles).group(1)
    cluster_pkl = re.search('(?<=cluster=)([\w\./]+)',infiles).group(1)
    cluster_mask_pkl = re.search('(?<=mask=)([\w\./]+)',infiles).group(1)
    
    filePath_save = outpath if outpath is not None else './Evaluation/'
     
    
    # Assign parameters
    target_crime = re.search('(?<=targetcrime=)(\w+)',params).group(1)
    clf_name = re.search('(?<=model=)(\w+)',params).group(1)
    kfolds = int(re.search('(?<=kfolds=)(\w+)',params).group(1))
    r_seed = int(re.search('(?<=rseed=)(\w+)',params).group(1))    
    eval_str = re.search('(?<=eval=)[\w\s\.]+',params).group(0)    
    eval_tuplist = re.findall('([a-zA-Z]+ )((?:\d*\.\d+\s*|\d+\s*){4})',eval_str)  # e.g. [('PAI ','0 1 100 0.1'), ('PEI ','0.1 0.5 50 0.2')]

    evalIdx_names = []
    areaPct = {}
    areaPctUB = {} 
    for eval_tup in eval_tuplist:
        interval_params = list(map(float, eval_tup[1].split()))  # convert string to list of numbers
        intervals = np.linspace(interval_params[0], interval_params[1],interval_params[2])
        ev_idx = eval_tup[0].rstrip()
        evalIdx_names.append(ev_idx)
        areaPct[ev_idx] = intervals
        areaPctUB[ev_idx] = interval_params[3]
                           
    clf = get_model_params(clf_name,rand_seed=r_seed)['model']
    tuning_params = get_model_params(clf_name,rand_seed=r_seed)['tuning_params']                 
    tuning_param_dicts = dict([(clf_name+'__'+key, val) for key,val in tuning_params.items()]) 
        
    CV_skf = get_cv_obj(kfolds,r_seed)
                
    _, grd_x, grd_y, _, mask_grdInCity, _ = ld.load_grid(grid_pkl)   
    grid_2d = (grd_x,grd_y)
    
    groups_test = ld.load_train_test_group(group_pkl)['test_groups']
    filename_dict = dict(group=group_pkl)
             
    filename_dict['cluster'] = cluster_pkl    
    cluster_label, _ = ld.load_clusters(cluster_pkl)   
    target_clusters = np.unique(cluster_label)
    subregions = ['SubRegion'+str(c) for c in target_clusters]
    regions = ['City']+subregions        
        
    CrimeData = ld.load_crime_data(crime_pkl)           
    baseline_test = ld.load_baseline_data(baseline_pkl, target_crime)            

    start = time.time()
    # load city data 
    filename_dict['feature'] = feature_balanced_pkl
    X_train_city, y_train_city, X_test_city, _, _ = ld.load_train_test_data(filename_dict['feature'])
    
    # train and test city models
    pred_scores_city = fit_pred(X_train_city, y_train_city, X_test_city, clf, clf_name,\
                                tuning_param_dicts, CV_skf, region='City')
    
    # city model evaluation
    score_list = [pred_scores_city, baseline_test[:,0], baseline_test[:,1]]            
    eval_dict = get_eval_lists(evalIdx_names, score_list, CrimeData, groups_test, grid_2d, mask_grdInCity, areaPct)
    model_name_list = [clf_name, 'LT density', 'ST density']     
    
    fig_names = {}
    for ev_idx in evalIdx_names:
        fig_names[ev_idx] = {}
        for field in ['EvalIdx', 'AUC','t_pval']:
            fig_names[ev_idx][field] = [filePath_save+'City/'+ev_idx+'_'+field+z+'.png' for z in ['','_zoom']]


    plot_eval_pred(eval_dict, evalIdx_names, model_name_list, areaPct, areaPctUB, fig_names)

    end = time.time()
    print('Running time (city) %.1f' % (end - start))
    
    # iterate over subregions           
    for target_cluster in target_clusters: 
        start = time.time()
        # load sub-region data
        filename_dict['feature'] = feature_pkl
        subregion_info = ld.load_subregion_train_test_data(filename_dict, target_cluster, \
                                                           balance=True, rand_seed=r_seed, load_city=False)
        
        sample_mask = subregion_info['sample_mask']
        X_train_sub, y_train_sub = subregion_info['train_data_subregion']
        X_test_sub, _ = subregion_info['test_data_subregion']
        
        # train and test sub-region models
        pred_scores_sub = fit_pred(X_train_sub, y_train_sub, X_test_sub, clf, clf_name, tuning_param_dicts,
                                              CV_skf, region='SubRegion'+str(target_cluster))
        
        # subregion model evaluation
        mask_grdInCluster = ld.load_cluster_mask(cluster_mask_pkl,target_cluster)
        score_list = [pred_scores_sub, pred_scores_city[sample_mask['test']], \
                      baseline_test[sample_mask['test'],0], baseline_test[sample_mask['test'],1]]             
        eval_dict = get_eval_lists(evalIdx_names, score_list, CrimeData, groups_test, grid_2d, mask_grdInCluster, areaPct)
        model_name_list = [clf_name+'_subregion', clf_name+'_city','LT density', 'ST density']
        
        for ev_idx in evalIdx_names:
            for field in ['EvalIdx', 'AUC','t_pval']:
                fig_names[ev_idx][field] = [filePath_save+'SubRegion'+str(target_cluster)+'/'+ev_idx+'_'+field+z+'.png' for z in ['','_zoom']]
        
        plot_eval_pred(eval_dict, evalIdx_names, model_name_list, areaPct, areaPctUB, fig_names)
   
        end = time.time()
        print('Running time (sub-region '+str(target_cluster)+') %.1f' % (end - start))  