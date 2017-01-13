#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:02:43 2017

@author: xiaomuliu
"""

import SetupGrid
import cPickle as pickle
import re
import numpy as np
import os.path
# Add package path to python path at runtime
import sys
sys.path.append('..')
# Or set environment variable
# PYTHONPATH=..
from Misc.ComdArgParse import ParseArg
from StructureData.LoadData import load_clusters

args = ParseArg()
infile = args['input']
outfile = args['output']
cell = args['param']

#convert string input to tuple
cell_match = re.match('(\d+) (\d+)',cell)
cellsize = (int(cell_match.group(1)), int(cell_match.group(2)))


default_path = "../SharedData/SpatialData/grid_"+str(cellsize[0])+"_"+str(cellsize[1])+"/"           
grid_pkl = os.path.dirname(infile)+"/grid.pkl" if infile is not None else default_path+"grid.pkl"
with open(grid_pkl,'rb') as grid_file:
    _, _, _, _, mask_grdInCity, _ = pickle.load(grid_file)

              
if infile is not None:
    cluster_pkl = infile
else:
    name_match = re.match('^clusters_[\w]+.pkl$',default_path)                     
    cluster_pkl = default_path+name_match.group(0) 
    
cluster_label, _ = load_clusters(cluster_pkl)
cluster_mask_list = [SetupGrid.get_cluster_mask(mask_grdInCity,cluster_label,target_label) for target_label in np.sort(np.unique(cluster_label))]            

                     
if outfile is not None:
    cluster_mask_pkl = outfile
else:
    name_match = re.match('^clusters_([\w]+)',cluster_pkl)                     
    model_name = name_match.group(1)
    cluster_mask_pkl = default_path+'masks_'+model_name+'.pkl'
     
with open(cluster_mask_pkl,'wb') as mask_file:
    pickle.dump(cluster_mask_list, mask_file, pickle.HIGHEST_PROTOCOL)    