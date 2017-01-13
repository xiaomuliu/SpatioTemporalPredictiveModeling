#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
===================================
Data aggregation and grouping
===================================
Created on Fri Aug 26 16:44:10 2016

@author: xiaomuliu
"""
import numpy as np
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import Day

def DOW_code_to_abbr(x):
    """
    Convert 'day of week' integer code to abbreviation (Monday is 0 and Sunday is 6)
    """
    return {
            0:'MON',
            1:'TUE',
            2:'WED',
            3:'THU',
            4:'FRI',
            5:'SAT',
            6:'SUN'
    }.get(x,None) 

def group_temporal_data(data, groupsize, startdate=None,enddate=None):
    """
    Group incident data by group size according to the date attribute (with irregular step size) in data
    """    
    if 'DATEOCC' in data.columns.values:
        data.rename(columns={'DATEOCC':'DATE'},inplace=True);
    
    data['DATE'] = pd.to_datetime(data['DATE'])
        
    if startdate is None:
        startdate = data['DATE'].iloc[0]
    if enddate is None:
        enddate = data['DATE'].iloc[-1]    
    
    groups = np.zeros(len(data))
    groupsize = int(groupsize)
    # # generate weekly sequence on given day of week
    # DOWabbr = DOW_code_to_abbr(datetime.weekday(startdate))
    # group_seq = pd.date_range(startdate, enddate, freq='W-'+DOWabbr)
    group_seq = pd.date_range(startdate, enddate, freq=str(groupsize)+'D')
        
    # sort data by date so the following optimized code can be applied 
    data.sort_values(by='DATE',inplace=True); 
    
    group_No = 1
    for d in group_seq:
        # The element-wise for-loop is slow
        #days_inGroup = pd.date_range(d, d+(groupsize-1)*Day(), freq='D')
        #isInGroup = np.array([d in days_inGroup for d in data['DATE']])
        #isInGroup = np.array([(day>=days_inGroup[0]) & (day<=days_inGroup[-1]) for day in data['DATE']])
        # Assuming data is sorted by date, the below code is faster
        isInGroup = np.array((data['DATE']>=d) & (data['DATE']<=d+(groupsize-1)*Day()))
        groups[isInGroup] = group_No
        group_No = group_No+1
    
    groups = groups.astype(int)
    data['GROUP'] = groups    
    return {'groups':groups,'data':data}


def get_date_array(startdate, enddate):
    """
    Return an array of daily dates with fields: date, year, month, day
    """
    date_seq = pd.date_range(startdate, enddate, freq='D')
    year = [datetime.strftime(d,'%Y') for d in date_seq]
    month = [datetime.strftime(d,'%m') for d in date_seq]
    day = [datetime.strftime(d,'%d') for d in date_seq]
    date_array = pd.DataFrame(dict(DATE=date_seq,YEAR=year,MONTH=month,DAY=day))
    date_array.sort_values(by='DATE',inplace=True);        
    return date_array  
        

def get_groups(groupsize, group_startdate, group_enddate, query_startdate, query_enddate):
    """
    Given the date period and the group size, generate a group sequence. 
    Then find the corresponding groups of a specified query date period 
    dates must be a string of format '%Y-%m-%d'
    """    
    date_array = get_date_array(group_startdate, group_enddate)
    groupsize = int(groupsize)
    group_seq = pd.date_range(group_startdate, group_enddate, freq=str(groupsize)+'D') 
    
    groups = np.zeros(len(date_array))
    group_No = 1
    for d in group_seq:
        isInGroup = np.array((date_array['DATE']>=d) & (date_array['DATE']<=d+(groupsize-1)*Day()))
        groups[isInGroup] = group_No
        group_No = group_No+1
    groups = groups.astype(int)
    
    #query_startdate = datetime.strptime(query_startdate,'%Y-%m-%d')
    query_startdate = pd.to_datetime(query_startdate)
    query_enddate = pd.to_datetime(query_enddate)
    groups = groups[((date_array['DATE']>=query_startdate) & (date_array['DATE']<=query_enddate)).values]
    return np.unique(groups)
        

def group_date_series(date_series, groupsize, startdate=None,enddate=None,index_order=True):
    """
    Group date series by group size according to the dates
    """    
    
    # set errors='coerce' so that invalid datetime values (e.g. -1) will be converted to NaT
    date_series = pd.to_datetime(date_series,errors='coerce')
        
    if startdate is None:
        startdate = date_series.iloc[0]
    if enddate is None:
        enddate = date_series.iloc[-1]    
    
#    groups = np.zeros(len(date_series))
    groups = -1*np.ones(len(date_series))
    groupsize = int(groupsize)
    # # generate weekly sequence on given day of week
    group_seq = pd.date_range(startdate, enddate, freq=str(groupsize)+'D')
        
    # sort data by date so the following optimized code can be applied 
    date_series.sort_values(inplace=True); 
    
    group_No = 1
    for d in group_seq:
        isInGroup = np.array((date_series>=d) & (date_series<=d+(groupsize-1)*Day()))
        groups[isInGroup] = group_No
        group_No = group_No+1
    
    group_df = date_series.to_frame()
    group_df['GROUP']= groups.astype(int)   
    if index_order:
        # re-order the group series in the order of indices in date_series
        group_df.sort_index(inplace=True)
        groups = group_df['GROUP'].values
  
    return {'groups':groups,'data':group_df}


       
if __name__ == '__main__':
    import cPickle as pickle
    filePath = "/Users/xiaomuliu/CrimeProject/SpatioTemporalModeling/ModelSegmentation_py/"
    
    # Crime Data
    group_size = 7
    startdate = '2008-01-01'
    enddate = '2014-12-31'
    CrimeTypes = ["Homicide","SexualAssault","Robbery","AggAssaultBattery","SimAssaultBattery", \
                  "Burglary","Larceny","MVT","MSO_Violent","All_Violent","Property"]
    for crimetype in CrimeTypes:     
        fileName_load = filePath + 'CrimeData/' + crimetype + '_08_14.pkl'
        with open(fileName_load,'rb') as input_file:
            CrimeData = pickle.load(input_file)

        CrimeData = group_temporal_data(CrimeData, group_size, startdate, enddate)['data']

        fileName_save = filePath + 'CrimeData/' + crimetype+ '_08_14_grouped.pkl'
        with open(fileName_save,'wb') as output:
            pickle.dump(CrimeData, output, pickle.HIGHEST_PROTOCOL)
    
            
#     #NOTE: Technically, both weather and 311-calls data groups should be matched with crime data's groups,
#     #as there may not be any records in a certain group. However, here as the group size is 7 and every week
#     #there are crime records and weather data, it is not necessary to use function match_groups
#     #for the grouping process.          
            
    # Weather Data    
    pkl_file = open(filePath+'WeatherData/Weather_01_14.pkl', 'rb')
    WeatherData = pickle.load(pkl_file)
    pkl_file.close()
    
    toDate = lambda x: datetime.strptime(x,'%Y-%m-%d').date()
    WeatherData = WeatherData.ix[(WeatherData.DATE>=toDate(startdate))&(WeatherData.DATE<=toDate(enddate))]
    
    WeatherData = group_temporal_data(WeatherData,group_size, startdate, enddate)['data']

    savefile = filePath + 'WeatherData/Weather_08_14_grouped.pkl'
    with open(savefile,'wb') as output:
        pickle.dump(WeatherData, output, pickle.HIGHEST_PROTOCOL)
        
    # 311 Calls Data
    
#    # NOTE: Technically, 311-calls data groups should be matched to the data of each crime type individually.
#    # However since the group size is 7. All the grouped crime data have the same group sequence. Here only 
#    # one crime data (All_Violent) is used
#    fileName_load = filePath + 'CrimeData/All_Violent_08_14_grouped.pkl'
#    with open(fileName_load,'rb') as input_file:
#        CrimeData = pickle.load(input_file)

        
    CallTypes = ['Vacant_and_Abandoned_Buildings_Reported','Street_Lights_All_Out',\
                 'Alley_Lights_Out','Street_Lights_One_Out']
    for calltype in CallTypes:
        fileName_load = filePath + '311Data/'+calltype + '_11_14.pkl' 
        with open(fileName_load,'rb') as input_file:
            CallsData = pickle.load(input_file)

        CallsData = group_temporal_data(CallsData, group_size, startdate, enddate)['data']    
        #CallsData = match_groups(CallsData, CrimeData)['data'] 

        fileName_save = filePath + '311Data/' + calltype+ '_11_14_grouped.pkl'
        with open(fileName_save,'wb') as output:
            pickle.dump(CallsData, output, pickle.HIGHEST_PROTOCOL)
            
            
    # POD data
    fileName_load = filePath +"PODdata/PODs_05_14.pkl"
    with open(fileName_load,'rb') as input_file:
        PODdata = pickle.load(input_file) 

    PODdata_group_df = PODdata
    date_cols = ['INSTALL_DATE1','REMOVE_DATE1','INSTALL_DATE2']
    group_cols = ['INSTALL_GROUP1','REMOVE_GROUP1','INSTALL_GROUP2']
    for date_col, group_col in zip(date_cols, group_cols):
        PODdata_group_df[group_col] = group_date_series(PODdata[date_col], group_size, startdate, enddate, index_order=True)['groups']

    savefile = filePath + 'PODdata/PODs_05_14_grouped.pkl'
    with open(savefile,'wb') as output:
        pickle.dump(PODdata_group_df, output, pickle.HIGHEST_PROTOCOL)