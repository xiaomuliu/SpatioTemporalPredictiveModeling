# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 17:39:50 2017

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

filePath_load = inpath if inpath is not None else "../SharedData/CrimeData/"
filePath_save = outpath if outpath is not None else "../SharedData/CrimeData/"

CrimeTypes = ["Homicide","SexualAssault","Robbery","AggAssaultBattery","SimAssaultBattery", \
              "Burglary","Larceny","MVT","MSO_Violent","All_Violent","Property"]
for crimetype in CrimeTypes:     
    fileName_load = filePath_load + crimetype + '_08_14.pkl'
    with open(fileName_load,'rb') as input_file:
        CrimeData = pickle.load(input_file)

    CrimeData = group_temporal_data(CrimeData, group_size, startdate, enddate)['data']

    fileName_save = filePath_save + crimetype+ '_08_14_grouped.pkl'
    with open(fileName_save,'wb') as output:
        pickle.dump(CrimeData, output, pickle.HIGHEST_PROTOCOL)