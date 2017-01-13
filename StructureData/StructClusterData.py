#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=======================================
Binned Crime Count Count Feature Array
=======================================
Created on Wed Oct  5 11:33:40 2016

@author: xiaomuliu
"""
import numpy as np
from datetime import datetime
import cPickle as pickle
import sys
sys.path.append('..')
from ImageProcessing.KernelSmoothing import bin_point_data_2d


def point_data_subset(data, date_range=None, coord_only=True):
    """
    Return a subset of crime point data given the specified date or year/month range
    date_range is a list-like object containing two string elements of which the first one 
    specifies the beginning date while the second one specifies the end date. 
    The date string should be in form of 'YYYY-MM-DD'
    """
    if date_range is not None:
        date_range = [datetime.strptime(d,'%Y-%m-%d').date() for d in date_range]          
        data_sub = data.ix[(data['DATEOCC']>=date_range[0]) & (data['DATEOCC']<=date_range[1]),:]
    else:
        data_sub = data
        
    if coord_only:
        data_sub = data_sub[['X_COORD','Y_COORD']]

    return data_sub

 
def flatten_binned_2darray(points, grid, cellsize=None):
    """
    Return 1D vector of binned point counts
    """
    if cellsize is None:
        # assuming a regular grid with equal cell size, the cellsize is of [size_x, size_y]
        cellsize = (np.abs(np.diff(grid[0][:2])), np.abs(np.diff(grid[1][:2])))
    
    binned_data = bin_point_data_2d(points, grid, cellsize, stat='count', geoIm=False)   
    flattened_data = binned_data.ravel(order='F')
              
    return flattened_data     
    
    
if __name__=='__main__':  
    import pandas as pd
    import re
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infiles = args['input']
    outfile = args['output']
    params = args['param']

    #convert string input to tuple
    param_match = re.match('(\d+) (\d+) (\d{4}-\d{2}-\d{2}) (\d{4}-\d{2}-\d{2})',params)
    cellsize = (int(param_match.group(1)), int(param_match.group(2)))
    LT_range = (param_match.group(3), param_match.group(4))
      
    default_grid_file = "../SharedData/SpatialData/grid_"+str(cellsize[0])+"_"+str(cellsize[1])+"/grid_"+str(cellsize[0])+".pkl"    
    default_crimedata_path = "../SharedData/CrimeData/" 
    
    infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles)
      
    grid_pkl = infile_match.group(1) if infiles is not None else default_grid_file
    crimedata_path = infile_match.group(2) if infiles is not None else default_crimedata_path
    
    CrimeTypes = {'Homicide':'Homicide',
                  'SexualAssault':'Criminal_Sexual_Assault',
                  'Robbery':'Robbery',
                  'AggAssaultBattery':'Aggravated_Assault_and_Aggravated_Battery',
                  'SimAssaultBattery':'Simple_Assault_and_Simple_Battery',
                  'Burglary':'Burglary',
                  'Larceny':'Larceny',
                  'MVT':'Motor_Vehicle_Theft',
                  'MSO_Violent':'More_Serious_Offenses_Violent_Crime',
                  'All_Violent':'Violent_Crime_of_All_Types',
                  'Property':'Property_Crime'}
    
    #NOTE: dict-type objects have no order              
    CrimeData_list = []
    CrimeType_list = []

    for crimetype_abbr, crimetype_str in CrimeTypes.items():        
        fileName_load = crimedata_path + crimetype_abbr + "_08_14.pkl"
        with open(fileName_load,'rb') as input_file:
            CrimeData_list.append(point_data_subset(pickle.load(input_file),LT_range))
            CrimeType_list.append(crimetype_str)

    # load grid info  
    with open(grid_pkl,'rb') as grid_file:
        grid_list = pickle.load(grid_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
    
    # bin crime data of each type
    hist_array = np.zeros((Ngrids,len(CrimeData_list)))
    for i, point_data in enumerate(CrimeData_list):
        hist_array[:,i] = flatten_binned_2darray(point_data.values, grid_2d, cellsize)[mask_grdInCity]

#    # verify
#    from SetupGrid import flattened_to_geoIm
#    import matplotlib.pyplot as plt
#    
#    hist_2d = flattened_to_geoIm(hist_array[:,0],len(grd_x),len(grd_y),mask=mask_grdInCity)
#
#    fig = plt.figure(figsize=(8, 6))
#    plt.imshow(hist_2d, interpolation='nearest', origin='upper', cmap='jet')
#    plt.title('Binned homicide count')
#    plt.colorbar()
    
    # *********************** Save objects ******************* #
    default_savePath = "../Clustering/FeatureData/grid_"+str(cellsize[0])+"_"+str(cellsize[1])+"/" 
    filePath_save = outfile if outfile is not None else default_savePath
       
    feature_names = ['Homicide','Criminal_Sexual_Assault','Robbery','Aggravated_Assault_and_Aggravated_Battery',\
                     'Simple_Assault_and_Simple_Battery','Burglary','Larceny','Motor_Vehicle_Theft'] 
    feature_idx = [CrimeType_list.index(name) for name in feature_names]                 
    feature_array = hist_array[:,feature_idx]
    feature_dict = {'FeatureArray':feature_array, 'FeatureName':feature_names}                           
    savefile_dict = filePath_save+'feature_dict.pkl'

    with open(savefile_dict,'wb') as output:
        pickle.dump(feature_dict, output, pickle.HIGHEST_PROTOCOL)
   
    feature_df = pd.DataFrame(feature_array,columns=feature_names)
    savefile_df = filePath_save+'feature_dataframe.pkl' 
    with open(savefile_df,'wb') as output: 
        pickle.dump(feature_df, output, pickle.HIGHEST_PROTOCOL)
             