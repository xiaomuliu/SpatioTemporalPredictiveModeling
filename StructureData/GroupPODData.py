# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 17:58:40 2017

@author: xiaomuliu
"""
from GroupData import group_date_series
import cPickle as pickle
import re
import sys
sys.path.append('..')
from Misc.ComdArgParse import ParseArg

args = ParseArg()
inpath = args['input']
outpath = args['output']
params = args['param']

param_match = re.match('(\d+) (\d{4}-\d{2}-\d{2}) (\d{4}-\d{2}-\d{2})',params)
group_size = int(param_match.group(1))
startdate, enddate = param_match.group(2), param_match.group(3)

filePath_load = inpath if inpath is not None else "../SharedData/WeatherData/"
filePath_save = outpath if outpath is not None else "../SharedData/WeatherData/"

fileName_load = filePath_load +"PODs_05_14.pkl"
with open(fileName_load,'rb') as input_file:
    PODdata = pickle.load(input_file) 

PODdata_group_df = PODdata
date_cols = ['INSTALL_DATE1','REMOVE_DATE1','INSTALL_DATE2']
group_cols = ['INSTALL_GROUP1','REMOVE_GROUP1','INSTALL_GROUP2']
for date_col, group_col in zip(date_cols, group_cols):
    PODdata_group_df[group_col] = group_date_series(PODdata[date_col], group_size, startdate, enddate, index_order=True)['groups']

savefile = filePath_save + 'PODs_05_14_grouped.pkl'
with open(savefile,'wb') as output:
    pickle.dump(PODdata_group_df, output, pickle.HIGHEST_PROTOCOL)