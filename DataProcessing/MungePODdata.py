#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:55:29 2016

@author: xiaomuliu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cPickle as pickle

def to_date(d):
    date_format = '%m/%d/%Y' if len(d)==10 else '%m/%d/%y'
    return datetime.strptime(d, date_format).date()

def munge_POD_data(PODdata,uptoDate=None): 
    
    PODdata = PODdata.ix[PODdata['INSTALL_DATE1']>0,:] # deal negative values
    PODdata['POD_ID'] = PODdata['POD_ID'].apply(lambda x: str(int(x))) 
            
    # convert INSTALL_DATE to date class
    # Note: installation dates are numbers indicating the incremental number of days since Jan 1st, 2005. 
    # For example installation date = 1 means it was installed on Jan 1st, 2005. 2 means Jan 2nd 2005, etc...
    init_date = datetime.strptime('01/01/2005', '%m/%d/%Y').date() 
    to_actual_date = lambda x: init_date+timedelta(days=x-1) if x!=-1 else -1 
    
    PODdata[['INSTALL_DATE1','REMOVE_DATE1','INSTALL_DATE2']] = \
            PODdata[['INSTALL_DATE1','REMOVE_DATE1','INSTALL_DATE2']].applymap(to_actual_date)
            
    uptoDate = (PODdata['INSTALL_DATE1'].values)[-1] if uptoDate is None else to_date(uptoDate)
        
    PODdata = PODdata.ix[PODdata['INSTALL_DATE1'].values<=uptoDate,:]

    # order the data by inital installation date
    PODdata = PODdata.sort_values(['INSTALL_DATE1'],axis=0)

    # For PODs being removed then reinstalled/relocated, differetiate their IDs by add suffixes _1, _2,...    
    pod_set, cnts = np.unique(PODdata['POD_ID'].values, return_counts=True)
    dup_pods, dup_cnts = pod_set[cnts>1], cnts[cnts>1]
    for pod, cnt in zip(dup_pods, dup_cnts):
        new_ids = [PODdata.ix[PODdata['POD_ID']==pod,'POD_ID'].iloc[i] +'_'+str(i) for i in xrange(cnt)]
        PODdata.ix[PODdata['POD_ID']==pod,'POD_ID'] = new_ids
                                                      
    # reset the indices
    PODdata.index = range(len(PODdata.index))  
        
    return PODdata


              
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
    
    fileName_load = infile if infile is not None else "./OtherData/PODs.csv"
    fileName_save = outfile if outfile is not None else "../SharedData/PODdata/PODs_05_14.pkl"
    
    PODdata = pd.read_csv(fileName_load)
   
    uptoDate = '12/31/2014' 
    PODdata = munge_POD_data(PODdata,uptoDate)
                     
    # save  
    with open(fileName_save,'wb') as output:
        pickle.dump(PODdata, output, pickle.HIGHEST_PROTOCOL)    