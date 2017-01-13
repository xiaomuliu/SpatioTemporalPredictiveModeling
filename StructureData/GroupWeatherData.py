# -*- coding: utf-8 -*-
"""
Created on Sat Jan 07 17:51:08 2017

@author: xiaomuliu
"""
            
from GroupData import group_temporal_data
from datetime import datetime
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


pkl_file = open(filePath_load+'Weather_01_14.pkl', 'rb')
WeatherData = pickle.load(pkl_file)
pkl_file.close()

toDate = lambda x: datetime.strptime(x,'%Y-%m-%d').date()
WeatherData = WeatherData.ix[(WeatherData.DATE>=toDate(startdate))&(WeatherData.DATE<=toDate(enddate))]

WeatherData = group_temporal_data(WeatherData,group_size, startdate, enddate)['data']

savefile = filePath_save + 'Weather_08_14_grouped.pkl'
with open(savefile,'wb') as output:
    pickle.dump(WeatherData, output, pickle.HIGHEST_PROTOCOL)