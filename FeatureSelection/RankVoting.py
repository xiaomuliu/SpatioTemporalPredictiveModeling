#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
======================================
ensemble feature selection
======================================
Created on Tue Nov  1 16:33:58 2016

@author: xiaomuliu
"""
import numpy as np
 
def select_by_avg_ranks(feature_ranks,feature_names,K_best=None,percentile=None,weights=None):
    """
    Select best (K_best/percentile) features according to the (weighted) average feature ranks
    """
    p = len(feature_names)
    if K_best is not None:
        N_keep = K_best
    elif percentile is not None:    
        N_keep = int(percentile/float(100)*p)
    else:
        N_keep = p
    
    if weights is None:    
        weights = np.ones(feature_ranks.shape[1])
    
    avg_ranks = np.average(feature_ranks,1,weights)   
    sel_idx = np.argsort(avg_ranks)[:N_keep]
    sel_features = feature_names[sel_idx]

    return sel_features, sel_idx, avg_ranks 

def select_by_proportion(feature_ranks,feature_names,K_best=None,percentile=None,prop_threshold=None,weights=None):
    """
    Select best (K_best/percentile) features according to the fraction of times 
    the features being ranked above the 'prop_threshold' over different methods 
    """
    p = len(feature_names)
    if K_best is not None:
        N_keep = K_best
    elif percentile is not None:    
        N_keep = int(percentile/float(100)*p)
    else:
        N_keep = p
    
    if prop_threshold is None:
        prop_threshold = N_keep
        
    if weights is None:    
        weights = np.ones(feature_ranks.shape[1])
    
    neg_avg_ranks = -np.average(feature_ranks,1,weights) #use to break the tie    
        
    N_methods = feature_ranks.shape[1]
    fraction = lambda x: np.sum(x<=prop_threshold)/float(N_methods)    
    frac = np.apply_along_axis(fraction, 1, feature_ranks)  
    
    to_sort = np.array(zip(frac,neg_avg_ranks),dtype=[('fraction',float),('-avg_rank',float)])
    sel_idx = np.argsort(to_sort,order=['fraction','-avg_rank'])[::-1][:N_keep]
    sel_features = feature_names[sel_idx]

    return sel_features, sel_idx, frac    

if __name__=='__main__':  
    import re
    import cPickle as pickle
    import sys 
    import collections
    import pandas as pd
    sys.path.append('..') 
    import StructureData.LoadData as ld            
    from Misc.ComdArgParse import ParseArg  
  
    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    target_crime = re.search('(?<=targetcrime=)(\w+)', infiles).group(1)     
    featureRank_pkl = re.search('(?<=featurerank=)([\w\./]+)',infiles).group(1)
    
    filePath_save = outpath if outpath is not None else "../SharedData/FeatureData/"

    prop_threshold = int(re.search('(?<=Nfeatures=)(\d+)', params).group(1)) 
     
    methods = ['univar (f-score)','RFE (logit)','rand-L1 (logit)','RF','GBM']    
    weights = np.ones(len(methods))
    Kbest = None
    
    colidx = ['avg rank','frac']
    
    rank_voting_avgrank = collections.defaultdict(dict) # a nested dict
    rank_voting_prop = collections.defaultdict(dict)

    feature_names, feature_ranks = ld.load_feature_ranks(featureRank_pkl)

    # select by average ranks
    sel_features, sel_idx, avg_ranks = select_by_avg_ranks(feature_ranks,feature_names,K_best=Kbest,weights=weights)
    rank_voting_avgrank = dict(selected_features=sel_features, selected_idx=sel_idx, \
                               feature_names=feature_names, feature_scores=avg_ranks)
    
    # select by proportion of times
    sel_features, sel_idx, frac = select_by_proportion(feature_ranks,feature_names,K_best=Kbest,\
                                                       prop_threshold=prop_threshold,weights=weights)
    rank_voting_prop = dict(selected_features=sel_features, selected_idx=sel_idx, \
                            feature_names=feature_names, feature_scores=frac)
        
    feature_ranking_df = pd.DataFrame(np.c_[avg_ranks,frac], index=feature_names, columns=colidx)

    savefile_df = filePath_save+target_crime+'_voted_feature_ranking_dataframe.pkl'
    with open(savefile_df,'wb') as output:
        pickle.dump(feature_ranking_df, output, pickle.HIGHEST_PROTOCOL) 
    savefile_csv = filePath_save+target_crime+'_voted_feature_ranking_dataframe.csv'
    feature_ranking_df.to_csv(savefile_csv)  
            
                    
    savefile_dict = filePath_save+target_crime+'_voted_feature_ranking_avg_rank.pkl'
    with open(savefile_dict,'wb') as output:
        pickle.dump(rank_voting_avgrank, output, pickle.HIGHEST_PROTOCOL)
        
    savefile_dict = filePath_save+target_crime+'_voted_feature_ranking_fracSel.pkl'
    with open(savefile_dict,'wb') as output:
        pickle.dump(rank_voting_prop, output, pickle.HIGHEST_PROTOCOL)    
        
        