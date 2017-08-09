#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:44:30 2017

@author: xiaomuliu
"""

import cPickle as pickle
import re
import numpy as np
import pandas as pd
import geopandas as gpd
#import matplotlib.pyplot as plt
from SetupGrid import get_cluster_mask, flattened_to_geoIm
import sys
sys.path.append('..')
from Misc.ComdArgParse import ParseArg
from FeatureExtraction.SpatialRelation import pt_poly_membership, pt_to_nearest_geoObj

args = ParseArg()
infiles = args['input']
outpath = args['output']
params = args['param']

infile_match = re.match('([\w\./]+) ([\w\./]+)',infiles) 
if infile_match.group(0) is not None:
    grid_pkl, district_shp = infile_match.group(1), infile_match.group(2)
else:                   
    grid_pkl = '../SharedData/SpatialData/grid.pkl'
    district_shp = '../SharedData/GISData/cpd_districts/cpd_districts.shp'

filePath_save = outpath if outpath is not None else '../SharedData/SpatialData/'

cell_match = re.match('(\d+) (\d+)',params)
cellsize = (int(cell_match.group(1)), int(cell_match.group(2)))
              
# load grid info 
with open(grid_pkl,'rb') as grid_file:
    grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, maskIm_grdInCity = pickle.load(grid_file)    
                                
    
district_shp ='../SharedData/GISData/cpd_districts/cpd_districts.shp' 

proj_str = "+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 " + \
              "+x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs +ellps=GRS80 +towgs84=0,0,0" 
    
shpfile = gpd.read_file(district_shp)
shpfile.plot()

district_nums = np.unique(shpfile['DISTRICT'].values)
district_nums = district_nums[district_nums!='031'] #remove district 031

district_label = np.zeros(len(grd_vec_inCity)).astype('str')
#district_mask = {}
for dist_num in district_nums:
    isIndistrict = pt_poly_membership(grd_vec_inCity,shpfile.ix[shpfile['DISTRICT']==dist_num],proj=proj_str)['indicator']
    isIndistrict = isIndistrict.flatten()
#    district_mask[dist_num] = get_cluster_mask(mask_grdInCity,isIndistrict,True)

#    #visualize     
#    m = flattened_to_geoIm(isIndistrict,len(grd_x),len(grd_y),mask=mask_grdInCity)
#    fig = plt.figure()
#    plt.imshow(m) 
 
    district_label[isIndistrict] = dist_num

# assign grid cells that do not fall in any districts to their nearest districts
out_cells = grd_vec_inCity[district_label=='0.0']
nearest_district_idx = pt_to_nearest_geoObj(out_cells,shpfile.ix[shpfile['DISTRICT']!='031',:])
nearest_district = shpfile.ix[nearest_district_idx,'DISTRICT'].values
district_label[district_label=='0.0'] = nearest_district

district_mask = {}
for dist_num in district_nums:   
    district_mask[dist_num] = get_cluster_mask(mask_grdInCity,district_label==dist_num,True)

   
savefile = filePath_save+'masks_district.pkl'   
with open(savefile,'wb') as outputfile:
    pickle.dump(district_mask, outputfile, pickle.HIGHEST_PROTOCOL) 
savefile = filePath_save+'label_district.pkl'   
with open(savefile,'wb') as outputfile:
    pickle.dump(district_label, outputfile, pickle.HIGHEST_PROTOCOL)    
    
district_label_df = pd.DataFrame(district_label)    
district_label_df.to_csv(filePath_save+'label_district.csv',header=False,index=False)    