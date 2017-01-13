#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=====================================
Importing and filtering weather data
=====================================
Created on Mon Sep 26 15:41:03 2016

@author: xiaomuliu
"""

import pandas as pd
from datetime import datetime, date
import cPickle as pickle


def filter_data_by_variable(weather_data, weather_var):        

    if 'All' in weather_var:
        weather_data_sub = weather_data
    else:
        # keep date column
        date_col = weather_data.Date
        if not isinstance(weather_var,list):
            weather_var = [weather_var]
        
        weather_data_sub = pd.concat([date_col, weather_data.ix[:,weather_data.columns.isin(weather_var)]],
                                                                axis=1, ignore_index=False)
        
    return weather_data_sub

    
def munge_weather_data(weather_data):  
    
    if not isinstance(weather_data['Date'].values[0], date):
        # convert occ_date to date class
        toDate = lambda x: datetime.strptime(x,'%Y-%m-%d').date()   
        weather_data.Date = weather_data.Date.apply(toDate)                       
                                                 
    # rename 'Date' to 'DATE' in order to match crime data
    weather_data.rename(columns={"Date": "DATE"},inplace=True)    
    
    # order the data by date
    weather_data = weather_data.sort_values(['DATE'],axis=0)
    weather_data.index = range(len(weather_data.index))  
        
    return weather_data

        
if __name__=='__main__':
    # Add package path to python path at runtime
    import sys
    sys.path.append('..')
    # Or set environment variable
    # PYTHONPATH=..
    
    from Misc.ComdArgParse import ParseArg
    
    files = ParseArg()
    infile = files['input']
    outfile = files['output']

    fileName_load = infile if infile is not None else "./WeatherData/WeatherData_Daily_2001-01-01_2014-12-31.csv"
    fileName_save = outfile if outfile is not None else "../SharedData/WeatherData/Weather_01_14.pkl" 
   
    WeatherData = pd.read_csv(fileName_load)
    
    WeatherVar = ['Tsfc_F_avg','Rh_PCT_avg','Psfc_MB_avg','CldCov_PCT_avg','Tapp_F_avg','Spd_MPH_avg','PcpPrevHr_IN']
    WeatherData_sub = filter_data_by_variable(WeatherData, WeatherVar)
    WeatherData_sub = munge_weather_data(WeatherData_sub)
                      
    with open(fileName_save,'wb') as output_file:
        pickle.dump(WeatherData_sub, output_file, pickle.HIGHEST_PROTOCOL)  # save as a .pkl file
    
