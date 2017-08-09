#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:27:51 2017

# Do not downsample training data

@author: xiaomuliu
"""


import numpy as np
import sys
if sys.version_info[0] > 2:
    import pickle # python 3.x deprecated cPickle 
else:
    import cPickle as pickle
import pandas as pd
import h5py
sys.path.append('..')
import ImageProcessing.KernelSmoothing as ks
from sklearn.preprocessing import OneHotEncoder


def stack_time_independent_features(feature_values, feature_names, stacked_feature_matrix, stacked_feature_names, num_chunks):
    """
    Assign time-independent features to the corresponding places in the stacked feature array where each time-independent 
    feature chunk is repeated num_chunks times 
    """
    # Vertically stack feature value arrays
    stacked_feature_matrix[:,np.in1d(stacked_feature_names,feature_names)] = np.tile(feature_values,reps=(num_chunks,1))
    return stacked_feature_matrix

def stack_time_dependent_features(feature_names, stacked_feature_matrix, stacked_feature_names,\
                                  num_chunks, block_size, func, *args, **kwargs):
    """
    Assign time-dependent features chunk to the corresponding places in the stacked feature array
    """
    for i in xrange(num_chunks):
        stacked_feature_matrix[i*block_size:(i+1)*block_size,np.in1d(stacked_feature_names,feature_names)] = \
                               func(i,*args,**kwargs)
                               
    return stacked_feature_matrix

def temporal_feature_subgroup(timeIdx, temporal_data, group_seq, num_grids, feature_names=['DOW']):
    group = group_seq[timeIdx]
    time_cols = [col for col in temporal_data.columns if col in feature_names]
    temporal_features = temporal_data.loc[group, time_cols]
    # one-hot encoding categorical variables
    K_values = {'Month':12, 'DOW':7}
    K = [K_values[f] for f in feature_names]
    encoder = OneHotEncoder(n_values=K)
    #temporal_features_code = encoder.fit_transform(temporal_features).toarray().reshape(-1,np.sum(K))
    temporal_features_code = encoder.fit_transform([temporal_features]).toarray()
    temporal_features_code = np.tile(temporal_features_code, reps=(num_grids,1))
    return temporal_features_code
    
def weather_feature_subgroup(timeIdx, weather_data, group_seq, num_grids):
    group = group_seq[timeIdx]
    weather_cols = [col for col in weather_data.columns if col not in ['DATE', 'GROUP']]
    weather_data_group_avg = weather_data.ix[weather_data['GROUP']==group, weather_cols].mean(axis=0).values
    # broadcast values to the shape of spatial grid
    weather_data_group_avg = np.tile(weather_data_group_avg, reps=(num_grids,1))
    return weather_data_group_avg  
    
def long_term_intensity_feature_subgroup(timeIdx, crime_data, group_seq, period, grid_2d, filter_2d, mask=None, density=True):
    group = group_seq[timeIdx]
    crimepts = crime_data.ix[(crime_data['GROUP']>=group-period[0]) & (crime_data['GROUP']<=group-period[1]),
                             ['X_COORD','Y_COORD']].values
    KS_LT = ks.kernel_smooth_2d_conv(crimepts, grid_2d, filter_2d, flatten=False)
    # flatten; np.newaxis will change shape of array from (**,) to (**,1); np.squeeze will do the opposite
    KS_LT = KS_LT.ravel(order='F')[mask,np.newaxis]
    if density==True:
        KS_LT = KS_LT/np.sum(KS_LT)
    return KS_LT
    
def short_term_intensity_feature_subgroup(timeIdx, crime_data, group_seq, period, grid_2d, filter_3d, mask=None, density=True):
    group = group_seq[timeIdx]
    crimepts = crime_data.ix[(crime_data['GROUP']>=group-period[0]) & (crime_data['GROUP']<=group-period[1]),
                             ['X_COORD','Y_COORD','GROUP']]
    #grd_t = np.unique(crimepts['GROUP'].values)
    grd_t = np.arange(group-period[0],group-period[1]+1)
    crimepts = crimepts.values
    grd_x, grd_y = grid_2d
    grid_3d = (grd_x,grd_y,grd_t)

    KS_ST = ks.kernel_smooth_separable_3d_conv(crimepts, grid_3d, filter_3d, flatten=False)  
    KS_ST = KS_ST[:,:,-1] # take out last time slice
    # flatten; np.newaxis will change shape of array from (**,) to (**,1); np.squeeze will do the opposite
    KS_ST = KS_ST.ravel(order='F')[mask,np.newaxis]
    if density==True:
        KS_ST = KS_ST/np.sum(KS_ST) 
    return KS_ST

# NOTE: long-term and short-term intensity features can be calucated for all time slice (chunk) at once rather than
# calculate them each for one chunk. However this will eat up memory. Therefore speed has to be sacrificed for space.                       

def count_consec_val(vec, val):
    """
    Count consecutive value 'val' in the vector 'vec'. For example, 
    vec=[1, 0, 0, 1, 0, 0, 0, 2, 0, 1, 1] and val=0
    The corresponding consecutive zeros are 
    [ 0, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0]
    """
    vec = np.array(vec)
    cnt = np.zeros(len(vec))
    start_idx = 0
    for i in xrange(len(vec)):
        cnt[i]=np.sum(vec[start_idx:i]==val)
        if vec[i]!=val:
            start_idx = i    
    return cnt  

 
def consec_presence_feature(data, grid_2d, cellsize_3d, group_seq, buffer_period=0, mask=None, num_chunks=None, presence=True,truncate=True):
    """
    Count the time groups (e.g. weeks) of a consecutive presence/absence. For example, 
    the 11 groups of crime count for a certain cell is as follows
    [1, 0, 0, 1, 0, 0, 0, 2, 0, 1, 1].
    The corresponding consecutive zeros are 
    [ 0, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0]
    """
    IncPts = data.ix[(data['GROUP']>=group_seq[0]-buffer_period) & (data['GROUP']<=group_seq[-1]),['X_COORD','Y_COORD','GROUP']]
    #grd_t = np.unique(IncPts['GROUP'].values)
    grd_t = np.arange(group_seq[0]-buffer_period,group_seq[-1]+1)
    IncPts = IncPts.values
    grd_x, grd_y = grid_2d
    grid_3d = (grd_x,grd_y,grd_t)

    if num_chunks is None:
        num_chunks = len(group_seq)
    if mask is None:
        mask = np.ones(len(grd_x)*len(grd_y)).astype('bool')
    
    binned_pts = ks.bin_point_data_3d(IncPts, grid_3d, cellsize_3d, stat='count', geoIm=False)
    if presence:
        # count consecutive presence  
        binned_pts = (binned_pts>0).astype(int)
    else:
        # count consecutive absense 
        binned_pts = (binned_pts==0).astype(int)
        
    consec_cnt = np.apply_along_axis(count_consec_val, 2, binned_pts, val=1)
    consec_cnt = consec_cnt[:,:,-len(group_seq):] #truncate buffer groups
    consec_cnt = consec_cnt.ravel(order='F')[np.tile(mask,reps=num_chunks),np.newaxis]

    if truncate:
       # Since consecutive numbers are unbounded, some will be affected by the buffer_period.
       consec_cnt[consec_cnt>buffer_period] = buffer_period        
    
    return consec_cnt  

    
def pod_proximity_feature(timeIdx, POD_dist, POD_time, group_seq):
    group = group_seq[timeIdx]
    # Find PODs that have already been installed before this temporal group
          
    # NOTE: Some PODs have been removed or relocate. 
    is_exist = ((POD_time['REMOVE_GROUP1']==-1) & (POD_time['INSTALL_GROUP1']<group)) | \
               ((POD_time['REMOVE_GROUP1']!=-1) & (POD_time['INSTALL_GROUP1']<group) & (POD_time['REMOVE_GROUP1']>group)) | \
               ((POD_time['INSTALL_GROUP2']!=-1) & (POD_time['INSTALL_GROUP2']<group))
    installed_PODs = POD_time.ix[is_exist,['POD_ID']].values.squeeze()
    POD_minDist = POD_dist.ix[:,installed_PODs].min(axis=1)     

    return POD_minDist.values[:,np.newaxis]

            
def assign_target(crime_data, grid_2d, cellsize_3d, mask=None, num_chunks=None, labeling=True, class_label=(0, 1)):
    """
    Return a target variable vector where each element correpondings to one example in 3D grid (x,y,t)
    For binary classification problem, the label class is determinded by number of crime incident in the grid cell (#crime>0: class 1; #crime=0: class 0)
    For regression problem, the count values will be used
    """
    crimepts = crime_data[['X_COORD','Y_COORD','GROUP']]
    #grd_t = np.unique(crimepts['GROUP'].values)
    g = crimepts['GROUP'].values
    grd_t = np.arange(np.min(g),np.max(g)+1)
    crimepts = crimepts.values
    grd_x, grd_y = grid_2d
    grid_3d = (grd_x,grd_y,grd_t)
    
    binned_pts = ks.bin_point_data_3d(crimepts, grid_3d, cellsize_3d, stat='count', geoIm=False)

    if num_chunks is None:
        num_chunks = len(np.unique(crime_data['GROUP']))
    if mask is None:
        mask = np.ones(len(grd_x)*len(grd_y)).astype('bool')
    
    target = binned_pts.ravel(order='F')[np.tile(mask,reps=num_chunks)]    
    if labeling:
        label = np.zeros(len(target))
        label[target==0] = class_label[0]
        label[target!=0] = class_label[1]
    else:
        label = None

    return target, label

   
def structure_feature_target_array(feature_crime_data, feature_crime_types, target_crime_data, target_crime_types, 
                                   groupSeq, varName, var_type_idx, grid2d, SpatialFeature, POD_data, filter2d, filter3d,
                                   period_LT, period_ST, cellsize_3d, temporal_data=None, temporal_var=['DOW'], 
                                   weather_data=None, calls_data=None, call_buffer_period=0, 
                                   crime_buffer_period=0, mask=None, density=True, labeling=True, class_label=(0,1)):
    """
    The whole process of (a) stacking features into an array of shape [num_chunks*chunksize, num_var] where each
    chunk corresponds to a time slice in which feature values are put in flattened grid order; (b) assiging label 
    for each example (3D cell) according to crime incident number for each crime type. 
    For (a), the steps include: (1) insert time-independent (spatial) features; (2) extract and insert long-term and
    short-term crime intensity features; (3) insert 311-calls features; and (4) insert weather feature.
    For (b), the returned label array is of shape (n_grids, n_crimetypes)
    """
    if mask is None:
        mask = np.ones(len(grid2d[0])*len(grid2d[1])).astype('bool')
    Ngrids = np.nansum(mask)
    Ngroups = len(groupSeq)
    
    varName_space = varName[var_type_idx['space']]
    varName_time = varName[var_type_idx['time']]
    varName_pod = varName[var_type_idx['POD']]
    varName_LTcrime = varName[var_type_idx['LT']]
    varName_STcrime = varName[var_type_idx['ST']]
    varName_weather = varName[var_type_idx['weather']]
    varName_311 = varName[var_type_idx['311']]
    varName_crime_pres = varName[var_type_idx['pres']]
    varName_crime_abs = varName[var_type_idx['abs']]

    featureArray = np.zeros((Ngrids*Ngroups,len(varName)))

    
    # spatial features
    # Replicate spatial features for all group chunks
    featureArray = stack_time_independent_features(SpatialFeature.values, varName_space, featureArray, varName, Ngroups)

    # temporal features
    featureArray = stack_time_dependent_features(varName_time, featureArray, varName, Ngroups, Ngrids,\
                                                 temporal_feature_subgroup, temporal_data, groupSeq, Ngrids, temporal_var)   
    
    # POD features
    featureArray = stack_time_dependent_features(varName_pod, featureArray, varName, Ngroups, Ngrids, \
                                                 pod_proximity_feature, POD_data['dist'], POD_data['time'], groupSeq)

    # long-term crime intensity features
    for var_LT, crimedata in zip(varName_LTcrime, feature_crime_data):        
        featureArray = stack_time_dependent_features(var_LT, featureArray, varName, Ngroups, Ngrids, long_term_intensity_feature_subgroup,
                                                     crimedata, groupSeq, period_LT, grid2d, filter2d, mask, density)

    # short-term crime intensity features
    for var_ST, crimedata in zip(varName_STcrime, feature_crime_data): 
        featureArray = stack_time_dependent_features(var_ST, featureArray, varName, Ngroups, Ngrids, short_term_intensity_feature_subgroup, 
                                                     crimedata, groupSeq, period_ST, grid2d, filter3d, mask, density) 
                                                   
    # weather features
    if weather_data is not None:
        varName_weather = [col for col in weather_data.columns if col not in ['DATE', 'GROUP']]
        featureArray = stack_time_dependent_features(varName_weather, featureArray, varName, Ngroups, Ngrids,\
                                                     weather_feature_subgroup, weather_data, groupSeq, Ngrids) 

    # 311 calls features
    if calls_data is not None:
        for calltype, calldata in zip(varName_311, calls_data):
            featureArray[:,np.in1d(varName,calltype)] = consec_presence_feature(calldata, grid2d, cellsize_3d, \
                         groupSeq, call_buffer_period, mask, Ngroups, presence=True, truncate=True)
          
    # crime present/absent duration features
    for var_pres, var_abs, crimedata in zip(varName_crime_pres, varName_crime_abs, feature_crime_data): 
        featureArray[:,np.in1d(varName,var_pres)] = consec_presence_feature(crimedata, grid2d, cellsize_3d, 
                     groupSeq, crime_buffer_period, mask, Ngroups, presence=True, truncate=True)              
        featureArray[:,np.in1d(varName,var_abs)] = consec_presence_feature(crimedata, grid2d, cellsize_3d, 
                     groupSeq, crime_buffer_period, mask, Ngroups, presence=False, truncate=True) 
                
    
    targetArray = np.zeros((Ngrids*Ngroups,len(target_crime_types)))
    labelArray = np.zeros((Ngrids*Ngroups,len(target_crime_types))).astype('int')
    for i, target_data in enumerate(target_crime_data):  
        crimepts = target_data.ix[(target_data['GROUP']>=groupSeq[0]) & (target_data['GROUP']<=groupSeq[-1]),['X_COORD','Y_COORD','GROUP']]
        targetArray[:,i], labelArray[:,i] = assign_target(crimepts, grid2d, cellsize_3d, mask, Ngroups, labeling, class_label)
               
    return {'feature':featureArray,'target':targetArray,'label':labelArray}


def downsample(label, size=None, rand_state=0, shuffle=False):
    """
    Balance 2-class data by downsampling.
    size, if specified, should be array-like where the first element indicates the size of 
    minority class after downsampling and the second element indicates the size of majority 
    class after downsampling. By default, it makes two classes have equal sizes
    
    Shuffle=False by default which means samples with the same label are contiguous. 
    Shuffling it first may be essential to get a meaningful cross-validation result.
    One can either shuffling the data directly or use an inbuilt option of some cross validation iterators, such as KFold,
    to shuffle the data indices before splitting them. sk.cross_validation.ShuffleSplit will also do the work.
    """ 
    classes = np.unique(label)
    minority = np.argmin(np.array([np.sum(label==classes[0]), np.sum(label==classes[1])]))
    
    if minority == 0:
        minor_label=classes[0]
    else:
        minor_label=classes[1]
        
    if size is None:
        size = (np.sum(label==minor_label),np.sum(label==minor_label))
     
    r = np.random.RandomState(rand_state)        
    #NOTE: x= np.where() returns a tuple of (np_arry,)
    sampleInd_minor = r.permutation(np.where(label==minor_label)[0])[:size[0]] 
    sampleInd_major = r.permutation(np.where(label!=minor_label)[0])[:size[1]]

    sampleInd = np.r_[sampleInd_minor,sampleInd_major]

    if shuffle==True:
        sampleInd = r.permutation(sampleInd)

    return sampleInd

    
def save_struct_data_h5(featureArray,targetArray,labelArray,targetCrimeTypes,filePath,split='train'):
    
    savefile_b = filePath+'/feature_target_'+split+'_dataframe.h5'
    with h5py.File(savefile_b, 'w') as hf:
        hf.create_dataset('feature_array', data=featureArray)
        for crimetype in targetCrimeTypes:                
            hf.create_dataset('target_array/'+crimetype, data=targetArray[crimetype])
            hf.create_dataset('label_array/'+crimetype, data=labelArray[crimetype])
            

if __name__ == '__main__':  
    import re
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    from LoadData import load_grid
    from GroupData import get_groups
    import time
    
    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
   
    infile_match = re.findall('([\w\./]+)',infiles)     
    grid_pkl, filePath_crime, filePath_weather, filePath_311, filePath_POD, filePath_spfeature = infile_match
    
    filePath_save = outpath if outpath is not None else '../SharedData/FeatureData/'
     
    param_match1 = re.search('(?<=group=)(\d+) (\d{4}-\d{2}-\d{2}) (\d{4}-\d{2}-\d{2})', params)
    param_match2 = re.search('(?<=traintest=)(\d{4}-\d{2}-\d{2}\s*){4}', params)
    param_match3 = re.search('(?<=ltst=)(\d*\.\d+\s*|\d+\s*){7}', params) 
    param_match4 = re.search('(?<=label=)([A-Za-z]+)',params)
    param_match5 = re.search('(?<=featurecrimetypes=)([A-Za-z_\s]+(?=targetcrimetypes))', params)
    param_match6 = re.search('(?<=targetcrimetypes=)([A-Za-z_\s]+(?=chunk))', params)
    param_match7 = re.search('(?<=chunk=)(\d+) (\d+)', params)
    
    # Assign parameters
    group_size = int(param_match1.group(1)) 
    group_dateRange = (param_match1.group(2), param_match1.group(3))
    p2 = param_match2.group(0).split()
    dateRange_train, dateRange_test = (p2[0],p2[1]), (p2[2],p2[3])
    p3 = param_match3.group(0).split()
    # sigma:gaussian kernel parameter; lam:exponential kernel parameter; period_LT, period_ST: long-term & short-term range (unit group)
    sigma, lam, period_LT, period_ST = (float(p3[0]),float(p3[1])),float(p3[2]),(int(p3[3]),int(p3[4])),(int(p3[5]),int(p3[6]))
    labeling = param_match4.group(1) in ('True','true','TRUE','T')
    chunksize_train, chunksize_test = int(param_match7.group(1)), int(param_match7.group(2)) 
    rand_seed = 1234
    
    
    
#    ###
#    # For quick debug
#    grid_pkl = '../SharedData/SpatialData/grid_500_500/grid.pkl'
#    filePath_crime = "../SharedData/CrimeData/"
#    filePath_weather="../SharedData/WeatherData/"
#    filePath_311="../SharedData/311data/"
#    filePath_POD="../SharedData/PODdata/"
#    filePath_spfeature="../SharedData/FeatureData/grid_500_500/"
#    filePath_save = "../SharedData/FeatureData/grid_500_500/Tr12_13_Te14/"
#    group_size=1
#    group_dateRange = ('2008-01-01', '2014-12-31')
#    dateRange_train, dateRange_test = ('2013-03-01', '2013-05-31'), ('2014-03-01', '2014-05-31')
#    sigma, lam, period_LT, period_ST = (1,1),0.071,(1461,28), (28,2)
#    labeling = True
#    rand_seed = 1234
#    chunksize_train = 61 
#    chunksize_test = 31
#    FeatureCrimeTypes = ['Homicide', 'SexualAssault', 'Robbery', 'AggAssault', 'AggBattery', 'SimAssault', 'SimBattery', 'Burglary', 'Larceny', 'MVT', 'UUW', 'Narcotics', 'MSO_Violent', 'All_Violent', 'Property']
#    TargetCrimeTypes = ['All_Violent','Property']
#    ###
#    
    
    
    # load crime data 
    FeatureCrimeTypes = re.findall('[A-Za-z_]+',param_match5.group(0))
    FeatureCrimeData = []              
    for crimetype in FeatureCrimeTypes:     
        crime_pkl = filePath_crime + crimetype + '_08_14_grouped.pkl'
        with open(crime_pkl,'rb') as input_file:
            FeatureCrimeData.append(pickle.load(input_file))
    
    TargetCrimeTypes = re.findall('[A-Za-z_]+',param_match6.group(0))
    TargetCrimeData = [] 
    for crimetype in TargetCrimeTypes:     
        crime_pkl = filePath_crime + crimetype + '_08_14_grouped.pkl'
        with open(crime_pkl,'rb') as input_file:
            TargetCrimeData.append(pickle.load(input_file))

    # save train-test example indicators
    groups_train = get_groups(group_size, group_dateRange[0], group_dateRange[1], dateRange_train[0], dateRange_train[1])
    groups_test = get_groups(group_size, group_dateRange[0], group_dateRange[1], dateRange_test[0], dateRange_test[1])

#    savefile_dict = filePath_save+'train_test_groups.pkl'
#    train_test_groups = {'train_groups':groups_train,'test_groups':groups_test}
#    with open(savefile_dict,'wb') as output:
#        pickle.dump(train_test_groups, output, pickle.HIGHEST_PROTOCOL)
#     
#    np.savetxt(filePath_save+'train_groups.csv',train_test_groups['train_groups'],fmt='%d',delimiter=',')    
#    np.savetxt(filePath_save+'test_groups.csv',train_test_groups['test_groups'],fmt='%d',delimiter=',')        
    
    # load weather data
    weather_pkl = filePath_weather+'Weather_08_14_grouped.pkl'
    with open(weather_pkl,'rb') as input_file:
        WeatherData = pickle.load(input_file)
    
    # load 311 calls data    
    CallTypes = ['Street_Lights_All_Out','Alley_Lights_Out','Street_Lights_One_Out']
    CallsData = []             
    for calltype in CallTypes:
        calls_pkl = filePath_311+ calltype + "_11_14_grouped.pkl" 
        with open(calls_pkl,'rb') as input_file:
            CallsData.append(pickle.load(input_file))   

    # load POD data    
    POD_data = {}         
    pod_time_pkl = filePath_POD+'PODs_05_14_grouped.pkl'
    with open(pod_time_pkl,'rb') as input_file:
        POD_data['time'] = pickle.load(input_file).ix[:,['POD_ID','INSTALL_GROUP1','REMOVE_GROUP1','INSTALL_GROUP2']]   

    temporal_var = ['DOW']
    varName_time = ['DOW'+str(d) for d in xrange(7)]
    temp = TargetCrimeData[-1].ix[:,temporal_var+['GROUP']] # use one crime data to get temporal data  
    for t_v in temporal_var:
        temp[t_v] = temp[t_v].astype(int) # convert categorical variables to int so that numerical functions can be applied
    temporal_data = temp.groupby('GROUP').mean() # this temporal_data is indexed by GROUP         
                    
    varName_pod = ['Dist2POD']
    varName_weather = ["Tsfc_F_avg","Rh_PCT_avg","Psfc_MB_avg","CldCov_PCT_avg","Tapp_F_avg","Spd_MPH_avg","PcpPrevHr_IN"]
    varName_STcrime = [c+'_short' for c in FeatureCrimeTypes]
    varName_LTcrime = [c+'_long' for c in FeatureCrimeTypes]
    varName_311 = ["Street_Lights_All_Out","Street_Lights_One_Out","Alley_Lights_Out"]
    varName_crime_pres = [c+'_consec_pres' for c in FeatureCrimeTypes]
    varName_crime_abs = [c+'_consec_abs' for c in FeatureCrimeTypes]

    
    # load spatial feature data            
    spfeature_pkl = filePath_spfeature+'SpFeature_dataframe.pkl'
    with open(spfeature_pkl,'rb') as input_file:
        SpatialFeature = pickle.load(input_file)
         
    if 'geometry' in SpatialFeature.columns:
        SpatialFeature.drop('geometry', axis=1, inplace=True) # remove 'geometry' column
    
    # load POD proximity data
    pod_dist_pkl = filePath_spfeature+'PODdist_dataframe.pkl'
    with open(pod_dist_pkl,'rb') as input_file:
        POD_data['dist'] = pickle.load(input_file)   
                                        
    # Set up parameters 
    _, grd_x, grd_y, _, mask_grdInCity, _ = load_grid(grid_pkl)
    grid_2d = (grd_x,grd_y)
    cellsize_2d = (grd_x[1]-grd_x[0],grd_y[1]-grd_y[0]) 
    cellsize_3d = cellsize_2d+(1,) # (size_x, size_y,size_t)
    
    gauss_filter = ks.gaussian_filter_2d(bandwidth=sigma, window_size=(4*2*sigma[0]+1,4*2*sigma[0]+1))
    gauss_exp_filter = ks.gaussian_exponential_filter_3d(bandwidth=(sigma[0],sigma[1],lam),\
                                                         window_size=(4*2*sigma[0]+1,4*2*sigma[1]+1,period_ST[0]-period_ST[1]))['space-time']               
    
    varName_space = SpatialFeature.columns.values.tolist()
    varName = np.array(varName_space + varName_time + varName_pod + varName_weather + varName_STcrime + varName_LTcrime +\
                       varName_311 + varName_crime_pres + varName_crime_abs)
        
    var_type_idx = {}
    keys = ['space','time','POD','weather','LT','ST','311','pres','abs']
    subnames = [varName_space, varName_time, varName_pod, varName_weather, varName_LTcrime, varName_STcrime,\
                varName_311, varName_crime_pres, varName_crime_abs]
    for key,subname in zip(keys,subnames):
        var_type_idx[key]=np.in1d(varName,subname)
                
    #------------------------------

    # Buffer periods are used to set up upper bound range of keeping track of consecutive abscent/present features
    buffer_period_crime = int(365/group_size)
    buffer_period_call = int(365/group_size)

    # Structure training samples and downsample(balance) samples by chunk so that they can be manageable in scale 
    
    target_train = {}
    label_train = {}
    for crimetype in TargetCrimeTypes:    
        target_train[crimetype] = []
        label_train[crimetype] = []

    chunk_No = 0
    for i in range(groups_train[0],groups_train[-1], chunksize_train):
        start = time.time()
        if i < groups_train[-1] and i+chunksize_train > groups_train[-1]:
            # the end chunk may have a smaller size
            groups_train_chunk = np.arange(i,groups_train[-1]+1)
        else:
            groups_train_chunk = np.arange(i,i+chunksize_train)
            
        structured_array_train = structure_feature_target_array(FeatureCrimeData, FeatureCrimeTypes, TargetCrimeData, TargetCrimeTypes,\
                                                                groups_train_chunk, varName, var_type_idx, grid_2d, \
                                                                SpatialFeature, POD_data, gauss_filter, gauss_exp_filter, \
                                                                period_LT, period_ST, cellsize_3d, temporal_data, temporal_var,\
                                                                WeatherData, CallsData,buffer_period_call, buffer_period_crime,\
                                                                mask_grdInCity, density=True, labeling=labeling, class_label=(0,1))    
      
        featureArray_train = structured_array_train['feature']
        targetArray_train = structured_array_train['target']
        labelArray_train = structured_array_train['label']
        
        for j, crimetype in enumerate(TargetCrimeTypes):
            target_train[crimetype] = targetArray_train[:,j]
            label_train[crimetype] = labelArray_train[:,j]
        
      
        # save training sample
        save_struct_data_h5(featureArray_train,target_train,label_train,TargetCrimeTypes,filePath_save,split='train_chunk'+str(chunk_No))
        chunk_No += 1
        
        end = time.time()
        print('Running time (training chunk) %.1f' % (end - start))    
        
    #------------------------------
    # structure test samples and save them by chunk so that each chunk can be read in memory
    target_test = {}
    label_test = {}
    for crimetype in TargetCrimeTypes:    
        target_test[crimetype] = []
        label_test[crimetype] = []
    
    chunk_No = 0 
    for i in range(groups_test[0],groups_test[-1], chunksize_test):
        start = time.time()
        if i < groups_test[-1] and i+chunksize_test > groups_test[-1]:
            # the end chunk may has smaller size
            groups_test_chunk = np.arange(i,groups_test[-1]+1)
        else:
            groups_test_chunk = np.arange(i,i+chunksize_test)    
        
        structured_array_test = structure_feature_target_array(FeatureCrimeData, FeatureCrimeTypes, TargetCrimeData, TargetCrimeTypes,\
                                                               groups_test_chunk, varName, var_type_idx, grid_2d, \
                                                               SpatialFeature, POD_data, gauss_filter, gauss_exp_filter, \
                                                               period_LT, period_ST, cellsize_3d, temporal_data, temporal_var, \
                                                               WeatherData, CallsData, buffer_period_call, buffer_period_crime, \
                                                               mask_grdInCity, density=True, labeling=labeling, class_label=(0,1))    
        featureArray_test = structured_array_test['feature']
        targetArray_test = structured_array_test['target']
        labelArray_test = structured_array_test['label']

        for j, crimetype in enumerate(TargetCrimeTypes):
            target_test[crimetype] = targetArray_test[:,j]
            label_test[crimetype] = labelArray_test[:,j]

        # save test sample                 
        save_struct_data_h5(featureArray_test,target_test,label_test,TargetCrimeTypes,filePath_save,split='test_chunk'+str(chunk_No))
        chunk_No += 1

        end = time.time()
        print('Running time (test chunk) %.1f' % (end - start))    
