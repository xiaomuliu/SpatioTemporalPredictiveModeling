#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 17:20:14 2016

@author: xiaomuliu
"""
import pandas as pd
from datetime import datetime, date
import cPickle as pickle

def munge_crime_data(CrimeData):  
    
    if not isinstance(CrimeData['DATEOCC'].values[0], date):
        # convert DATEOCC to date class
        toDate = lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date()   
        CrimeData.DATEOCC = CrimeData.DATEOCC.apply(toDate)                       
    
    # add attribute 'day of week'
    # NOTE: For date.weekday coding, Monday is 0 and Sunday is 6, 
    # while datetime.isoweekday() returns the day of the week as an integer, where Monday is 1 and Sunday is 7
    DOW = pd.Series(CrimeData.DATEOCC.apply(date.weekday),index=CrimeData.index,name='DOW')
    
    CrimeData.insert(CrimeData.columns.get_loc('DAY')+1,'DOW', DOW)
    CrimeData.drop(["AREA","FBI_CD","LOCATION"],axis=1,inplace=True)
    
    CrimeData["CURR_IUCR"] = CrimeData["CURR_IUCR"].astype('string')
    CrimeData["BEAT"] = CrimeData["BEAT"].astype('string')
    CrimeData["DISTRICT"] = CrimeData["DISTRICT"].astype('string')
    
    # convert categorical variables to 'category' data type
    cat_var = ["BEAT","DISTRICT","CURR_IUCR","DOW"]  
    CrimeData.loc[:,cat_var] = CrimeData.loc[:,cat_var].apply(lambda x: x.astype('category'))   
    
    # remove rows that have NA location info
    CrimeData = CrimeData[CrimeData['X_COORD'].notnull() & CrimeData['Y_COORD'].notnull()]
    CrimeData["X_COORD"] = CrimeData["X_COORD"].astype('int')
    CrimeData["Y_COORD"] = CrimeData["Y_COORD"].astype('int')
        
    # order the data by date
    CrimeData = CrimeData.sort_values(['DATEOCC'],axis=0)
    CrimeData.index = range(len(CrimeData.index))  
        
    return CrimeData
 
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
    
    filePath_load = infile if infile is not None else "./CrimeData/DWH/"
    filePath_save = outfile if outfile is not None else "../SharedData/CrimeData/"
    
#    CrimeTypes = ["Homicide","SexualAssault","Robbery","AggAssault","AggBattery","SimAssault","SimBattery", \
#             "Burglary","Larceny","MVT","UUW","Narcotics","MSO_Violent","All_Violent","Property"]
    CrimeTypes = ["AggAssault","AggBattery","SimAssault","SimBattery","UUW","Narcotics"]         
             
    for crimetype in CrimeTypes:
        fileName_load = filePath_load + crimetype + "_08_14.csv"
        CrimeData = pd.read_csv(fileName_load)
        CrimeData = munge_crime_data(CrimeData)
        
        fileName_save = filePath_save + crimetype + "_08_14.pkl"
        with open(fileName_save,'wb') as output:
            pickle.dump(CrimeData, output, pickle.HIGHEST_PROTOCOL)