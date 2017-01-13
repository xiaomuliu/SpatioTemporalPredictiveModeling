#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 17:11:50 2017

@author: xiaomuliu
"""
import SetupGrid
import geopandas as gpd
import cPickle as pickle
import re
# Add package path to python path at runtime
import sys
sys.path.append('..')
# Or set environment variable
# PYTHONPATH=..
from Misc.ComdArgParse import ParseArg


args = ParseArg()
infile = args['input']
outfile = args['output']
cell = args['param']

#convert string input to tuple
cell_match = re.match('([0-9]+) ([0-9]+)',cell)
cellsize = (int(cell_match.group(1)), int(cell_match.group(2)))

fileName_load = infile if infile is not None else "../SharedData/GISData/City_Boundary/City_Boundary.shp"
filePath_save = outfile if outfile is not None else "../SharedData/SpatialData/" 

city_shp = gpd.read_file(fileName_load)
                                                         
grd_vec, grd_x, grd_y = SetupGrid.vectorized_grid(city_shp,cellsize)

mask_grdInCity = SetupGrid.grid_within_bndy(grd_vec,city_shp)['mask']
grd_vec_inCity = grd_vec[mask_grdInCity,:]

maskIm_grdInCity = SetupGrid.grid_within_bndy([grd_x,grd_y],city_shp,im2d_mask=True,geoIm=True)['mask']
                   
savefile = filePath_save+'grid.pkl'
grid_list = [grd_vec, grd_x, grd_y, grd_vec_inCity, mask_grdInCity, maskIm_grdInCity]
with open(savefile,'wb') as output:
    pickle.dump(grid_list, output, pickle.HIGHEST_PROTOCOL) 