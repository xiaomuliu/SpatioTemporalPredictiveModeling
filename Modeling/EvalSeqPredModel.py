#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:17:25 2017

@author: xiaomuliu
"""

import numpy as np
import Evaluation as ev
import sys
sys.path.append('..') 
import StructureData.LoadData as ld 

def get_eval_region_mask(region, region_num, nsamples=None, regionlabel_file=None, regionmask_file=None):
    # create evaluation region mask (regarding the city vector grid)
    if region=='city':
        region_mask = np.ones(nsamples).astype(bool)
        mask_grdInRegion = None
    elif region=='cluster':
        cluster_label, _ = ld.load_clusters(regionlabel_file) 
        region_mask = np.in1d(cluster_label,region_num)
        mask_grdInRegion = ld.load_cluster_mask(regionmask_file,region_num)
    elif region=='district':
        district_label = ld.load_districts(regionlabel_file) 
        region_mask = np.in1d(district_label,region_num)
        mask_grdInRegion = ld.load_district_mask(regionmask_file,region_num)
    return region_mask, mask_grdInRegion

def get_eval_lists(evalIdx_names, score_list, CrimeData, groups, grid_2d, mask, areaPercent, boot=False, Nboots=1, r_seed=1234):
    eval_dict = {} 
    for ev_idx in evalIdx_names: 
        evalIdx_list = [ev.EvalIdx_array(scores, CrimeData, groups, grid_2d, ev_idx, mask, areaPercent[ev_idx], boot, Nboots, r_seed) for scores in score_list]
        eval_dict[ev_idx] = evalIdx_list                
    
    return eval_dict
                 

def plot_eval_pred(eval_dict, evalIdx_names, model_name_list, areaPercent, fig_names):
    for ev_idx in evalIdx_names:
        ev.plot_evalIdx(eval_dict[ev_idx], areaPercent[ev_idx], model_name_list, ev_idx, sd_err=False,
                    ls=['-','--'], show=False, save_fig=True, fig_name=fig_names[ev_idx]['EvalIdx']);    
        ev.plot_auc_series(eval_dict[ev_idx], areaPercent[ev_idx], model_name_list, 
                           mkr=['.','x'], show=False, save_fig=True, fig_name=fig_names[ev_idx]['AUC']);
        ev.auc_sig_test(eval_dict[ev_idx], areaPercent[ev_idx], model_name_list, log=True,
                        plot=True, show=False, save_fig=True, fig_name=fig_names[ev_idx]['t_pval']);
                        
                        
if __name__ == '__main__':   
    import re
    import cPickle as pickle
    from Misc.ComdArgParse import ParseArg    

    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
     
    grid_pkl = re.search('(?<=grid=)([\w\./]+)',infiles).group(1)
    crime_pkl = re.search('(?<=crime=)([\w\./]+)',infiles).group(1)
    group_pkl = re.search('(?<=group=)([\w\./]+)',infiles).group(1)
    baseline_pkl = re.search('(?<=baseline=)([\w\./]+)',infiles).group(1)
    predscore_filelist = re.search('(?<=predscore=)([\w\./\s;]+(?=samplemask))',infiles).group(1).rstrip('; ').split('; ')
    samplemask_filelist = re.search('(?<=samplemask=)([\w\./\s;]+(?=cluster))',infiles).group(1).rstrip('; ').split('; ')
    cluster_pkl = re.search('(?<=cluster=)([\w\./]+)',infiles).group(1)
    cluster_mask_pkl = re.search('(?<=clustermask=)([\w\./]+)',infiles).group(1)
    district_pkl = re.search('(?<=district=)([\w\./]+)',infiles).group(1)
    district_mask_pkl = re.search('(?<=districtmask=)([\w\./]+)',infiles).group(1)
    
    filePath_save = outpath if outpath is not None else './Evaluation/'
         
    # Assign parameters
    target_crime = re.search('(?<=targetcrime=)(\w+)',params).group(1)
    eval_region = re.search('(?<=evalregion=)(\w+)',params).group(1)
    eval_region_num = re.search('(?<=evalregionNo=)(\w+)',params).group(1)
    model_name_list = re.search('(?<=model=)([\w\s:;]+(?=evalspec))',params).group(1).rstrip('; ').split(';')
       
    eval_str = re.search('(?<=evalspec=)[\w\s\.]+',params).group(0)    
    eval_tuplist = re.findall('([a-zA-Z]+ )((?:\d*\.\d+\s*|\d+\s*){3})',eval_str)  # e.g. [('PAI ','0 1 100'), ('PEI ','0.1 0.5 50')]

    evalIdx_names = []
    areaPct = {}
    for eval_tup in eval_tuplist:
        interval_params = list(map(float, eval_tup[1].split()))  # convert string to list of numbers
        intervals = np.linspace(interval_params[0], interval_params[1],interval_params[2])
        ev_idx = eval_tup[0].rstrip()
        evalIdx_names.append(ev_idx)
        areaPct[ev_idx] = intervals                        

    #------------------ Load data ----------------------------------------
    _, grd_x, grd_y, _, mask_grdInCity, _ = ld.load_grid(grid_pkl)   
    grid_2d = (grd_x,grd_y)
     
    groups_test = ld.load_train_test_group(group_pkl)['test_groups']            

    CrimeData = ld.load_crime_data(crime_pkl)           
    baseline_test = ld.load_baseline_data(baseline_pkl,target_crime)  
    
    score_list = []
    for p_file, m_file in zip(predscore_filelist, samplemask_filelist):
        with open(m_file,'rb') as input_file:
            samplemask = pickle.load(input_file)
        predscores = np.loadtxt(p_file,delimiter=',')
        predscores_city = np.tile(np.zeros(np.sum(mask_grdInCity)),len(groups_test))
        #assign prediction scores to the entire city
        predscores_city[samplemask['test']] = predscores
        score_list.append(predscores_city)   
        
    if eval_region=='city':
        nsamples = np.sum(mask_grdInCity)
        evl_region_num = None
        region_label_file = None
        region_mask_file = None
    elif eval_region=='cluster':
        nsamples = None
        eval_region_num = map(int,eval_region_num.split('_'))
        region_label_file = cluster_pkl
        region_mask_file = cluster_mask_pkl
    elif eval_region=='district':
        nsamples = None
        eval_region_num = eval_region_num.split('_')
        region_label_file = district_pkl
        region_mask_file = district_mask_pkl
        
    region_mask, mask_grdInRegion = get_eval_region_mask(eval_region, eval_region_num, nsamples, region_label_file, region_mask_file)
    if eval_region=='city':
        mask_grdInRegion = mask_grdInCity
    region_sample_mask = np.tile(region_mask,len(groups_test))
    #--------------------------------------------------------------------- 
    baseline_test = baseline_test[-len(region_sample_mask):,:] #sequential models have truncated output

    score_list = score_list + [baseline_test[:,0], baseline_test[:,1]]              
    score_list = [s[region_sample_mask] for s in score_list] 
    model_name_list = [m.strip() for m in model_name_list]            
    model_name_list = model_name_list+['LT density', 'ST density']
    eval_dict = get_eval_lists(evalIdx_names, score_list, CrimeData, groups_test, grid_2d, mask_grdInRegion, areaPct)
#    eval_dict = get_eval_lists(evalIdx_names, score_list, CrimeData, groups_test, grid_2d, mask_grdInRegion, areaPct,boot=True, Nboots=1000, r_seed=1234)
    
    fig_names = {}
    for ev_idx in evalIdx_names:
        fig_names[ev_idx] = {}
        for field in ['EvalIdx','AUC','t_pval']:
            fig_names[ev_idx][field] = filePath_save+ev_idx+'_'+field+'.png'
    
    plot_eval_pred(eval_dict, evalIdx_names, model_name_list, areaPct, fig_names)
    
