# -*- coding: utf-8 -*-
"""
======================================
Functions of importing crime data
======================================

Created on Thu Aug 18 11:44:04 2016

@author: xiaomuliu
"""

def Crime2UCR(CrimeType):
    return {
        'Homicide' : ['0110','0130'],
        'CriminalSexualAssault' : ['0261','0262','0263','0264','0265','0266','0271','0272','0273','0274','0275','0281','0291','1753','1754'],
        'Robbery' : ['0312','0313','031A','031B','0320','0325','0326','0330','0331','0334','0337','033A','033B','0340'],
        'AggravatedAssault' : ['051A','051B','0520','0530','0550','0551','0552','0553','0555','0556','0557','0558'],
        'AggravatedBattery' : ['041A','041B','0420','0430','0450','0451','0452','0453','0461','0462','0479','0480','0480','0481',\
                               '0482','0483','0485','0488','0489','0490','0491','0492','0493','0495','0496','0497','0498','0510'],
        'Burglary' : ['0610','0620','0630','0650'],
        'Larceny' : ['0810','0820','0840','0841','0842','0843','0850','0860','0865','0870','0880','0890','0895'],
        'MotorVehicleTheft' : ['0910','0915','0917','0918','0920','0925','0927','0928','0930','0935','0937','0938'],
        'Arson' : ['1010','1020','1025','1090'],    
        'ViolentCrime' : ['0110','0130','0261','0262','0263','0264','0265','0266','0271','0272','0273','0274','0275','0281',\
                          '0291','1753','1754','0312','0313','031A','031B','0320','0325','0326','0330','0331','0334','0337',\
                          '033A','033B','0340','051A','051B','0520','0530','0550','0551','0552','0553','0555','0556','0557',\
                          '0558','041A','041B','0420','0430','0450','0451','0452','0453','0461','0462','0479','0480','0480',\
                          '0481','0482','0483','0485','0488','0489','0490','0491','0492','0493','0495','0496','0497','0498','0510'],
        'PropertyCrime' : ['0610','0620','0630','0650','0810','0820','0840','0841','0842','0843','0850','0860','0865','0870',
                           '0880','0890','0895','0910','0915','0917','0918','0920','0925','0927','0928','0930','0935','0937',
                           '0938','1010','1020','1025','1090']    
    }.get(CrimeType,'ViolentCrime') # Violent crime is default if CrimeType not found

import pandas as pd
from datetime import datetime,date

def FilterDataByCrimeType(CrimeData,CrimeType):
  
    CrimeData.columns = ["DATEOCC","CURR_IUCR","BEAT","DISTRICT","FBI_CD","X_COORD","Y_COORD","YEAR","LAT","LONG"]
    # remove year 2015 data
    CrimeData = CrimeData[CrimeData.YEAR<2015]
  
    # filter data by IUCR
    UCR = Crime2UCR(CrimeType)
    CrimeData_sub = CrimeData[CrimeData['CURR_IUCR'].isin(UCR)]
  
    # convert dateocc to date class
    # CrimeData_sub.DATEOCC = pd.to_datetime(CrimeData.DATEOCC,format="%m/%d/%Y %I:%M:%S %p")
    CrimeData_sub.DATEOCC = [datetime.strptime(datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p').strftime('%d-%B-%y'),'%d-%B-%y') for x in CrimeData_sub.DATEOCC]
    
    # add attribute 'Month','Day' and 'day of week'
    Month = pd.Series([datetime.strftime(x,'%m') for x in CrimeData_sub.DATEOCC],dtype=int,index=CrimeData_sub.index,name='MONTH')
    Day = pd.Series([datetime.strftime(x,'%d') for x in CrimeData_sub.DATEOCC],dtype=int,index=CrimeData_sub.index,name='DAY')
    
    pieces = [CrimeData_sub.loc[:,["DATEOCC","YEAR"]], Month, Day, CrimeData_sub.drop(["DATEOCC","YEAR"],axis=1)]
    CrimeData_sub = pd.concat(pieces,axis=1,ignore_index=False)
   
    CrimeData_sub.loc[:,["BEAT","DISTRICT","CURR_IUCR","FBI_CD"]] = CrimeData_sub.loc[:,["BEAT","DISTRICT","CURR_IUCR","FBI_CD"]].apply(lambda x: x.astype('category'))   
    
    # remove rows that have NA location info
    CrimeData_sub = CrimeData_sub[CrimeData_sub['X_COORD'].notnull() & CrimeData_sub['Y_COORD'].notnull()]
    CrimeData_sub["X_COORD"] = CrimeData_sub["X_COORD"].astype('int')
    CrimeData_sub["Y_COORD"] = CrimeData_sub["Y_COORD"].astype('int')
    
    CrimeData_sub['INC_CNT'] = 1
  
    return CrimeData_sub

    
def ImportCrimeData(fileName):
    CrimeData = pd.read_csv(fileName)
  
    # convert dateocc to date class
    # CrimeData.DATEOCC = [datetime.strftime(datetime.strptime(CrimeData.DATEOCC,'%Y-%m-%d'),'%Y-%m-%d') for x in CrimeData.DATEOCC]
    toDate = lambda x: datetime.strptime(x,'%Y-%m-%d').date()   
    CrimeData.DATEOCC = CrimeData.DATEOCC.apply(toDate)      
    
    # order the data by date
    CrimeData = CrimeData.sort_values(['DATEOCC'],axis=0)
    CrimeData.index = range(len(CrimeData.index))  
    # add attribute 'day of week' as an integer, where Monday is 0 and Sunday is 6
    # NOTE: datetime.isoweekday() returns the day of the week as an integer, where Monday is 1 and Sunday is 7
    DOW = pd.Series(CrimeData.DATEOCC.apply(date.weekday),index=CrimeData.index,name='DOW')
  
    # CrimeData.insert(CrimeData.columns.tolist().index('DAY'),'DOW',DOW)    
    CrimeData.insert(CrimeData.columns.get_loc('DAY'),'DOW',DOW)
    
    if "AREA" in CrimeData.columns:
        CrimeData["AREA"] = CrimeData["AREA"].astype('category')
        
    CrimeData.loc[:,["BEAT","DISTRICT","CURR_IUCR","FBI_CD","DOW"]] = CrimeData.loc[:,["BEAT","DISTRICT","CURR_IUCR","FBI_CD","DOW"]].apply(lambda x: x.astype('category'))
    
    return CrimeData