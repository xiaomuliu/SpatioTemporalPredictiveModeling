#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:02:25 2016

@author: xiaomuliu
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cPickle as pickle
import re

def to_date(d):
    date_format = '%m/%d/%Y' if len(d)==10 else '%m/%d/%y'
    return datetime.strptime(d, date_format).date()

def munge_311_data(CallsData,category='light',startDate=None,endDate=None): 
    if category == 'light':
        KeepAttr = ["Creation_Date","Status","Completion_Date","X_Coordinate","Y_Coordinate"]
        colNames = ["CREATE_DATE","STATUS","COMPLETE_DATE","X_COORD","Y_COORD"]       
        #date_format = '%m/%d/%y'
        
    elif category == 'bldg':
        KeepAttr = ["DATE_SERVICE_REQUEST_WAS_RECEIVED","X_COORDINATE","Y_COORDINATE"]
        colNames = ["CREATE_DATE","X_COORD","Y_COORD"]
        #date_format = '%m/%d/%Y'
    else:
        raise ValueError("CallType must be either 'light' or 'bldg'.")
        
    # keep relevant attributes    
    CallsData.columns = np.array([re.sub(' ','_',name) for name in CallsData.columns.values])
    CallsData = CallsData[KeepAttr]
    CallsData.columns = colNames
    
    # remove duplicated requests
    if category=='light':
        CallsData = CallsData.ix[np.in1d(CallsData['STATUS'], ['Open','Completed']),:]
        CallsData.drop(['STATUS'],axis=1,inplace=True)
    
    # remove imcomplete date entries
    CallsData.dropna(axis=0, how='any', inplace=True);
        
    # convert DATE to date class
    # Note: Some records have date of format mm/dd/yyyy some of format mm/dd/yy
     
    #CallsData['CREATE_DATE'] = pd.to_datetime(CallsData['CREATE_DATE'],format=date_format)
    CallsData['CREATE_DATE'] = CallsData['CREATE_DATE'].apply(to_date)
    
    if category=='light':
        #CallsData['COMPLETE_DATE'] = pd.to_datetime(CallsData['COMPLETE_DATE'],format=date_format)
        CallsData['COMPLETE_DATE'] = CallsData['COMPLETE_DATE'].apply(to_date)
       
    startDate = CallsData['CREATE_DATE'][0] if startDate is None else to_date(startDate)
    endDate = CallsData['CREATE_DATE'][-1] if endDate is None else to_date(endDate)
        
    CallsData = CallsData.ix[np.logical_and(CallsData['CREATE_DATE'].values>=startDate, CallsData['CREATE_DATE'].values<=endDate),:]
                                                          
    # order the data by date
    CallsData = CallsData.sort_values(['CREATE_DATE'],axis=0)
    CallsData.index = range(len(CallsData.index))  
        
    return CallsData


def flatten_311_data(CallsData,startDate=None,endDate=None):    
    """
    For lights-out data, duplicate each record N times (in the unit of day) 
    where N is the length of fixing period. By doing this way, the 3D binned
    counts can be easily calculated as the uniformly-spaced time axis is set.  
    """
    DateArray = []
    XcoordArray = []
    YcoordArray = []
    for i in xrange(len(CallsData)):
        if CallsData.ix[i,'CREATE_DATE']==CallsData.ix[i,'COMPLETE_DATE']:
            DateRange = [CallsData.ix[i,'CREATE_DATE']]
        else:
            DateRange = pd.date_range(CallsData.ix[i,'CREATE_DATE'], CallsData.ix[i,'COMPLETE_DATE']-timedelta(days=1), freq='D')
            DateRange = [d.date() for d in DateRange]
            
        DateArray = np.r_[DateArray,DateRange]
        XcoordArray = np.r_[XcoordArray,np.repeat(CallsData.ix[i,'X_COORD'],len(DateRange))]
        YcoordArray = np.r_[YcoordArray,np.repeat(CallsData.ix[i,'Y_COORD'],len(DateRange))] 
                           
    FlattenedData = pd.DataFrame(dict(DATE=DateArray, X_COORD=XcoordArray, Y_COORD=YcoordArray))
    
    startDate = FlattenedData['DATE'][0] if startDate is None else to_date(startDate)
    endDate = FlattenedData['DATE'][-1] if endDate is None else to_date(endDate)
    FlattenedData = FlattenedData.ix[np.logical_and(FlattenedData['DATE'].values>=startDate, FlattenedData['DATE'].values<=endDate),:]
    
    FlattenedData = FlattenedData.sort_values(['DATE'],axis=0)
    FlattenedData.index = range(len(FlattenedData.index))     
    return FlattenedData 
           

    
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
    
    filePath_load = infile if infile is not None else "./OtherData/"
    filePath_save = outfile if outfile is not None else "../SharedData/311Data/"
    
    CallTypes = ['Vacant_and_Abandoned_Buildings_Reported','Street_Lights_All_Out',\
                 'Alley_Lights_Out','Street_Lights_One_Out']
    for calltype in CallTypes:
        fileName_load = filePath_load + '311_Service_Requests_'+calltype + ".csv"
        CallsData = pd.read_csv(fileName_load)
        category = 'light' if re.search('Lights', calltype) is not None else 'bldg'

        startDate = '01/01/2011'
        endDate = '12/31/2014' 
        CallsData = munge_311_data(CallsData,category,startDate,endDate)
                    
        if category=='light':
            CallsData = flatten_311_data(CallsData,startDate,endDate)
        elif category=='bldg':
            CallsData.rename(columns = {'CREATE_DATE':'DATE'},inplace=True);
            
                # save                     
        fileName_save = filePath_save + calltype + "_11_14.pkl"
        with open(fileName_save,'wb') as output:
            pickle.dump(CallsData, output, pickle.HIGHEST_PROTOCOL)    