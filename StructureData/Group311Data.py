# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 17:55:29 2017

@author: xiaomuliu
"""
from GroupData import group_temporal_data
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

filePath_load = inpath if inpath is not None else "../SharedData/311Data/"
filePath_save = outpath if outpath is not None else "../SharedData/311Data/"

CallTypes = ['Vacant_and_Abandoned_Buildings_Reported','Street_Lights_All_Out',\
             'Alley_Lights_Out','Street_Lights_One_Out']
for calltype in CallTypes:
    fileName_load = filePath_load + calltype + '_11_14.pkl' 
    with open(fileName_load,'rb') as input_file:
        CallsData = pickle.load(input_file)

    CallsData = group_temporal_data(CallsData, group_size, startdate, enddate)['data']    
    #CallsData = match_groups(CallsData, CrimeData)['data'] 

    fileName_save = filePath_save + calltype + '_11_14_grouped.pkl'
    with open(fileName_save,'wb') as output:
        pickle.dump(CallsData, output, pickle.HIGHEST_PROTOCOL)