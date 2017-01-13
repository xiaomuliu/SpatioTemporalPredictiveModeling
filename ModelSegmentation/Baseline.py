#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
========================================
Baseline models
========================================
Created on Thu Sep 15 17:25:47 2016

@author: xiaomuliu
"""

import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import ImageProcessing.KernelSmoothing as ks

def stack_baseline_model_values(colNames, stacked_array, stacked_colNames,\
                                num_chunks, block_size, func, *args, **kwargs):
    """
    Assign time-dependent model value chunk to the corresponding places in the stacked array
    """
    for i in xrange(num_chunks):
        stacked_array[i*block_size:(i+1)*block_size,np.in1d(stacked_colNames,colNames)] = func(i,*args,**kwargs)
                               
    return stacked_array


def intensity_model_subgroup(timeIdx, crime_data, group_seq, period, grid_2d, filter_2d, mask=None, density=True):
    group = group_seq[timeIdx]
    CrimePts = crime_data.ix[(crime_data['GROUP']>=group-period[0]) & (crime_data['GROUP']<=group-period[1]),
                             ['X_COORD','Y_COORD']].values
    KS = ks.kernel_smooth_2d_conv(CrimePts, grid_2d, filter_2d, flatten=False)
    # flatten
    KS = KS.ravel(order='F')[mask,np.newaxis]
    if density==True:
        KS = KS/np.sum(KS)
    return KS    
    
    
def structure_baseline_model(crime_data, crime_types, groupSeq, grid2d, filter2d, period_LT, period_ST, mask=None, density=True):
    """
    Return a baseline model value matrix (n_samples, 2*n_crime_types) 
    where each crime type has a long-term density and a short-term density prediction values
    """
    if mask is None:
        mask = np.ones(len(grid2d[0])*len(grid2d[1])).astype('bool')
    Ngrids = np.nansum(mask)
    Ngroups = len(groupSeq)

    baseline_array = np.zeros((Ngrids*Ngroups,2*len(crime_types)))
    col_names = [[c+'_LT_den', c+'_ST_den'] for c in crime_types]
    col_names = list(chain.from_iterable(col_names)) #unpack list to merge to one list 
    for crimetype, crimedata in zip(crime_types, crime_data):
        baseline_array = stack_baseline_model_values(crimetype+'_LT_den', baseline_array, col_names, Ngroups, Ngrids, intensity_model_subgroup,
                                                  crimedata, groupSeq, period_LT, grid2d, filter2d, mask, density)  
        
        baseline_array = stack_baseline_model_values(crimetype+'_ST_den', baseline_array, col_names, Ngroups, Ngrids, intensity_model_subgroup,
                                                  crimedata, groupSeq, period_ST, grid2d, filter2d, mask, density) 
    
    return baseline_array


def plot_baseline(example_LT,example_ST):    
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(121)
    im = plt.imshow(example_LT, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('Long-term intensity model example')
    plt.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(example_ST, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('Short-term intensity model example')
    plt.colorbar(im)    
 
    
if __name__ == '__main__':
    import cPickle as pickle
#    from SetupGrid import flattened_to_geoIm 
    import pandas as pd
    import re
    import sys
    sys.path.append('..')   
    from Misc.ComdArgParse import ParseArg
    from StructureData.GroupData import get_groups
    from StructureData.LoadData import load_grid

    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
   
    infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)     
    grid_pkl, filePath_crime = infile_match.group(1), infile_match.group(2)
    
    filePath_save = outpath if outpath is not None else '../SharedData/ModelData/'

    param_match1 = re.search('(?<=group=)(\d+) (\d{4}-\d{2}-\d{2}) (\d{4}-\d{2}-\d{2})', params)
    param_match2 = re.search('(?<=traintest=)(\d{4}-\d{2}-\d{2}\s*){4}', params)
    param_match3 = re.search('(?<=ltst=)(\d*\.\d+\s*|\d+\s*){7}', params)   
    
    # Assign parameters
    group_size = int(param_match1.group(1)) 
    group_dateRange = (param_match1.group(2), param_match1.group(3))
    p2 = param_match2.group(0).split()
    dateRange_test = (p2[2],p2[3])
    p3 = param_match3.group(0).split()
    # sigma:gaussian kernel parameter; period_LT, period_ST: long-term & short-term range (unit group)
    sigma, period_LT, period_ST = (float(p3[0]),float(p3[1])),(int(p3[3]),int(p3[4])),(int(p3[5]),int(p3[6]))
    
        
    # load crime data
    CrimeTypes = ["Homicide","SexualAssault","Robbery","AggAssaultBattery","SimAssaultBattery", \
                  "Burglary","Larceny","MVT","MSO_Violent","All_Violent","Property"]
    
    CrimeData = []              
    for crimetype in CrimeTypes:     
        crime_pkl = filePath_crime + crimetype + '_08_14_grouped.pkl'
        with open(crime_pkl,'rb') as input_file:
            CrimeData.append(pickle.load(input_file))
            
    gauss_filter = ks.gaussian_filter_2d(bandwidth=sigma, window_size=(4*2*sigma[0]+1,4*2*sigma[0]+1))

    # test set
    groups_test = get_groups(group_size, group_dateRange[0], group_dateRange[1], dateRange_test[0], dateRange_test[1])
           
                 
    # load spatial feature data    
    _, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, _ = load_grid(grid_pkl)        
    grid_2d = (grd_x,grd_y)
    cellsize_2d = (grd_x[1]-grd_x[0],grd_y[1]-grd_y[0]) 
                     
    baseline_test = structure_baseline_model(CrimeData, CrimeTypes, groups_test, grid_2d, gauss_filter, \
                                             period_LT, period_ST, mask_grdInCity, density=True)                  
                      
#        # Plot one time slice example to verity
#        Ngrids = grd_vec_inCity.shape[0]     
#        example_LT = flattened_to_geoIm(baseline_test[(5*Ngrids):((5+1)*Ngrids),-4].squeeze(),len(grd_x),len(grd_y),mask_grdInCity) 
#        example_ST = flattened_to_geoIm(baseline_test[(5*Ngrids):((5+1)*Ngrids),-3].squeeze(),len(grd_x),len(grd_y),mask_grdInCity)
#          
#        plot_baseline(example_LT,example_ST)
            
    # Save objects                                            
    savefile_nparray = filePath_save+'baseline.pkl'
    with open(savefile_nparray,'wb') as output:
        pickle.dump(baseline_test, output, pickle.HIGHEST_PROTOCOL)

    model_names = [[c+'_LT_den', c+'_ST_den'] for c in CrimeTypes]
    model_names = list(chain.from_iterable(model_names)) #unpack list to merge to one list 
    baseline_df = pd.DataFrame(baseline_test,columns=model_names)
    savefile_df = filePath_save+'baseline_dataframe.pkl'
    with open(savefile_df,'wb') as output: 
        pickle.dump(baseline_df, output, pickle.HIGHEST_PROTOCOL)


                 