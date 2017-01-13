#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
====================================
Structure data
Add New:
    1. RTM feature
    2. 311 calls
    3. crime consecutive presenet and absent duration
    4. time-varying POD poximity
====================================
Created on Mon Aug 29 16:08:52 2016

@author: xiaomuliu
"""
import numpy as np
import sys
if sys.version_info[0] > 2:
    import pickle # python 3.x deprecated cPickle 
else:
    import cPickle as pickle
import pandas as pd

sys.path.append('..')
import ImageProcessing.KernelSmoothing as ks


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

   
def weather_feature_subgroup(timeIdx, weather_data, group_seq, num_grids):
    group = group_seq[timeIdx]
    weather_cols = [col for col in weather_data.columns if col not in ['DATE', 'GROUP']]
    weather_data_group_avg = weather_data.ix[weather_data['GROUP']==group, weather_cols].mean(axis=0).values
    # broadcast values to the shape of spatial grid
    weather_data_group_avg = np.tile(weather_data_group_avg, reps=(num_grids,1))
    return weather_data_group_avg  

    
def long_term_intensity_feature_subgroup(timeIdx, crime_data, group_seq, period, grid_2d, filter_2d, mask=None, density=True):
    group = group_seq[timeIdx]
    CrimePts = crime_data.ix[(crime_data['GROUP']>=group-period[0]) & (crime_data['GROUP']<=group-period[1]),
                             ['X_COORD','Y_COORD']].values
    KS_LT = ks.kernel_smooth_2d_conv(CrimePts, grid_2d, filter_2d, flatten=False)
    # flatten; np.newaxis will change shape of array from (**,) to (**,1); np.squeeze will do the opposite
    KS_LT = KS_LT.ravel(order='F')[mask,np.newaxis]
    if density==True:
        KS_LT = KS_LT/np.sum(KS_LT)
    return KS_LT
    
def short_term_intensity_feature_subgroup(timeIdx, crime_data, group_seq, period, grid_2d, filter_3d, mask=None, density=True):
    group = group_seq[timeIdx]
    CrimePts = crime_data.ix[(crime_data['GROUP']>=group-period[0]) & (crime_data['GROUP']<=group-period[1]),
                             ['X_COORD','Y_COORD','GROUP']]
    grd_t = np.unique(CrimePts['GROUP'].values)
    CrimePts = CrimePts.values
    grd_x, grd_y = grid_2d
    grid_3d = (grd_x,grd_y,grd_t)

    KS_ST = ks.kernel_smooth_separable_3d_conv(CrimePts, grid_3d, filter_3d, flatten=False)  
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
    cnt = np.zeros(len(vec))
    start_idx = 0
    for i in xrange(len(vec)):
        cnt[i]=np.sum(vec[start_idx:i]==val)
        if vec[i]!=val:
            start_idx = i    
    return cnt  

 
def consec_presence_feature(data, grid_2d, cellsize_3d, group_seq, buffer_period=0, mask=None, num_chunks=None, presence=True):
    """
    Count the time groups (e.g. weeks) of a consecutive presence/absence. For example, 
    the 11 groups of crime count for a certain cell is as follows
    [1, 0, 0, 1, 0, 0, 0, 2, 0, 1, 1].
    The corresponding consecutive zeros are 
    [ 0, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0]
    """
    IncPts = data.ix[(data['GROUP']>=group_seq[0]-buffer_period) & (data['GROUP']<=group_seq[-1]),['X_COORD','Y_COORD','GROUP']]
    grd_t = np.unique(IncPts['GROUP'].values)
    IncPts = IncPts.values
    grd_x, grd_y = grid_2d
    grid_3d = (grd_x,grd_y,grd_t)
    
    binned_pts = ks.bin_point_data_3d(IncPts, grid_3d, cellsize_3d, stat='count', geoIm=False)
    if presence==False:
        # count consecutive absense  
        binned_pts = (binned_pts>0).astype(int)
    
    if num_chunks is None:
        num_chunks = len(group_seq)
    if mask is None:
        mask = np.ones(len(grd_x)*len(grd_y)).astype('bool')
        
    consec_absent_cnt = np.apply_along_axis(count_consec_val, 2, binned_pts, val=1)
    consec_absent_cnt = consec_absent_cnt[:,:,-len(group_seq):] #truncate buffer groups
    consec_absent_cnt = consec_absent_cnt.ravel(order='F')[np.tile(mask,reps=num_chunks),np.newaxis]
    return consec_absent_cnt  

    
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
    
    
def assign_label(crime_data, grid_2d, cellsize_3d, mask=None, num_chunks=None, class_label=(0, 1)):
    """
    Return a label vector where each element correpondings to one example in 3D grid (x,y,t)
    The label class is determinded by number of crime incident in the grid cell (#crime>0: class 1; #crime=0: class 0)
    """
    CrimePts = crime_data[['X_COORD','Y_COORD','GROUP']]
    grd_t = np.unique(CrimePts['GROUP'].values)
    CrimePts = CrimePts.values
    grd_x, grd_y = grid_2d
    grid_3d = (grd_x,grd_y,grd_t)
    
    binned_pts = ks.bin_point_data_3d(CrimePts, grid_3d, cellsize_3d, stat='count', geoIm=False)

    if num_chunks is None:
        num_chunks = len(np.unique(crime_data['GROUP']))
    if mask is None:
        mask = np.ones(len(grd_x)*len(grd_y)).astype('bool')
        
    label = np.array(binned_pts>0).astype('int')
    label = label.ravel(order='F')[np.tile(mask,reps=num_chunks)]
    label[label==0] = class_label[0]
    label[label==1] = class_label[1]
    return label


def structure_feature_label_array(crime_data, crime_types, groupSeq, varName, var_type_idx, grid2d, SpatialFeature, POD_data, filter2d, filter3d,
                                  period_LT, period_ST, cellsize_3d, weather_data=None, calls_data=None, call_buffer_period=0, 
                                  crime_buffer_period=0, mask=None, density=True, class_label=(0,1)):
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
    
    # POD features
    featureArray = stack_time_dependent_features(varName_pod, featureArray, varName, Ngroups, Ngrids, \
                                                 pod_proximity_feature, POD_data['dist'], POD_data['time'], groupSeq)
    
    # long-term crime intensity features
    for var_LT, crimedata in zip(varName_LTcrime, crime_data):        
        featureArray = stack_time_dependent_features(var_LT, featureArray, varName, Ngroups, Ngrids, long_term_intensity_feature_subgroup,
                                                     crimedata, groupSeq, period_LT, grid2d, filter2d, mask, density)
    # short-term crime intensity features
    for var_ST, crimedata in zip(varName_STcrime, crime_data): 
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
                         groupSeq, call_buffer_period, mask, Ngroups, presence=True)
    
    # crime present/absent duration features
    for var_pres, var_abs, crimedata in zip(varName_crime_pres, varName_crime_abs, crime_data): 
        featureArray[:,np.in1d(varName,var_pres)] = consec_presence_feature(crimedata, grid2d, cellsize_3d, 
                     groupSeq, crime_buffer_period, mask, Ngroups, presence=True)              
        featureArray[:,np.in1d(varName,var_abs)] = consec_presence_feature(crimedata, grid2d, cellsize_3d, 
                     groupSeq, crime_buffer_period, mask, Ngroups, presence=False) 
                       
    labelArray = np.zeros((Ngrids*Ngroups,len(crime_types))).astype('int')
    for i, target_data in enumerate(crime_data):    
        CrimePts = target_data.ix[(target_data['GROUP']>=groupSeq[0]) & (target_data['GROUP']<=groupSeq[-1]),['X_COORD','Y_COORD','GROUP']]
        labelArray[:,i] = assign_label(CrimePts, grid2d, cellsize_3d, mask, Ngroups, class_label)
    return {'feature':featureArray,'label':labelArray}



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

def save_struct_data(featureArray_train,featureArray_test,labelArray_train,labelArray_test,varName,CrimeTypes,filePath):
    # all crime types in one file
    feature_array = np.r_[featureArray_train, featureArray_test]
    label_array = np.r_[labelArray_train,labelArray_test]
    train_test_indicator = np.r_[np.repeat('train',len(labelArray_train)), np.repeat('test',len(labelArray_test))]                             
    feature_label_dict = {'FeatureArray':feature_array, 'Label':label_array,'SplitIndicator':train_test_indicator}                           
    savefile_dict = filePath+'/all_crimetype_feature_label_dict.pkl'
    with open(savefile_dict,'wb') as output:
        pickle.dump(feature_label_dict, output, pickle.HIGHEST_PROTOCOL)

    
    feature_df = pd.DataFrame(feature_array,columns=varName)
    label_df = pd.DataFrame(label_array,columns=['Label_'+c for c in CrimeTypes])
    split_ind_df = pd.DataFrame(train_test_indicator,columns=['SplitIndicator'])
    feature_label_df = pd.concat([feature_df,label_df,split_ind_df],axis=1)
    savefile_df = filePath+'/all_crimetype_feature_label_dataframe.pkl'
    with open(savefile_df,'wb') as output: 
        pickle.dump(feature_label_df, output, pickle.HIGHEST_PROTOCOL)
        
    # seperate file for each crime type
    for i, crimetype in enumerate(CrimeTypes): 
        label_array = np.r_[labelArray_train[:,i],labelArray_test[:,i]]
        train_test_indicator = np.r_[np.repeat('train',len(labelArray_train)), np.repeat('test',len(labelArray_test))]                             
        feature_label_dict = {'FeatureArray':feature_array, 'Label':label_array,'SplitIndicator':train_test_indicator}                           
        savefile_dict = filePath+'/'+crimetype+'_feature_label_dict.pkl'
        with open(savefile_dict,'wb') as output:
            pickle.dump(feature_label_dict, output, pickle.HIGHEST_PROTOCOL)
    
        
        feature_df = pd.DataFrame(feature_array,columns=varName)
        label_df = pd.DataFrame(label_array,columns=['Label'])
        split_ind_df = pd.DataFrame(train_test_indicator,columns=['SplitIndicator'])
        feature_label_df = pd.concat([feature_df,label_df,split_ind_df],axis=1)
        savefile_df = filePath+'/'+crimetype+'_feature_label_dataframe.pkl'
        with open(savefile_df,'wb') as output: 
            pickle.dump(feature_label_df, output, pickle.HIGHEST_PROTOCOL)
    
        
    # balanced data    
    for i, crimetype in enumerate(CrimeTypes):    
        feature_array_balanced = np.r_[featureArray_train_balanced[crimetype], featureArray_test]
        label_balanced = np.r_[label_train_balanced[crimetype], labelArray_test[:,i]]
        train_test_indicator_balanced = np.r_[np.repeat('train',len(label_train_balanced[crimetype])), np.repeat('test',len(labelArray_test))]                             
        feature_label_dict_balanced = {'FeatureArray':feature_array_balanced, 'Label':label_balanced,'SplitIndicator':train_test_indicator_balanced}                           
        savefile_dict = filePath+'/'+crimetype+'_balanced_feature_label_dict.pkl'
        with open(savefile_dict,'wb') as output:
            pickle.dump(feature_label_dict_balanced, output, pickle.HIGHEST_PROTOCOL)
    
            
        feature_df_balanced = pd.DataFrame(feature_array_balanced,columns=varName)
        label_df_balanced = pd.DataFrame(label_balanced,columns=['Label'])
        split_ind_df_balanced = pd.DataFrame(train_test_indicator_balanced,columns=['SplitIndicator'])
        feature_label_df_balanced = pd.concat([feature_df_balanced,label_df_balanced,split_ind_df_balanced],axis=1)
        savefile_df = filePath+'/'+crimetype+'_balanced_feature_label_dataframe.pkl'
        with open(savefile_df,'wb') as output: 
            pickle.dump(feature_label_df_balanced, output, pickle.HIGHEST_PROTOCOL)     
    
    
                 
if __name__ == '__main__':  
    import re
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    from LoadData import load_grid
    from GroupData import get_groups
    
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
    
    # Assign parameters
    group_size = int(param_match1.group(1)) 
    group_dateRange = (param_match1.group(2), param_match1.group(3))
    p2 = param_match2.group(0).split()
    dateRange_train, dateRange_test = (p2[0],p2[1]), (p2[2],p2[3])
    p3 = param_match3.group(0).split()
    # sigma:gaussian kernel parameter; lam:exponential kernel parameter; period_LT, period_ST: long-term & short-term range (unit group)
    sigma, lam, period_LT, period_ST = (float(p3[0]),float(p3[1])),float(p3[2]),(int(p3[3]),int(p3[4])),(int(p3[5]),int(p3[6]))
    buffer_period_crime = period_LT[0]
    buffer_period_call = period_LT[0]
    rand_seed = 1234
        
    # load crime data
    CrimeTypes = ["Homicide","SexualAssault","Robbery","AggAssaultBattery","SimAssaultBattery", \
                  "Burglary","Larceny","MVT","MSO_Violent","All_Violent","Property"]  
    CrimeData = []              
    for crimetype in CrimeTypes:     
        crime_pkl = filePath_crime + crimetype + '_08_14_grouped.pkl'
        with open(crime_pkl,'rb') as input_file:
            CrimeData.append(pickle.load(input_file))
    

    # save train-test example indicators
    groups_train = get_groups(group_size, group_dateRange[0], group_dateRange[1], dateRange_train[0], dateRange_train[1])
    groups_test = get_groups(group_size, group_dateRange[0], group_dateRange[1], dateRange_test[0], dateRange_test[1])

    savefile_dict = filePath_save+'train_test_groups.pkl'
    train_test_groups = {'train_groups':groups_train,'test_groups':groups_test}
    with open(savefile_dict,'wb') as output:
        pickle.dump(train_test_groups, output, pickle.HIGHEST_PROTOCOL)
            
    
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

    varName_pod = ['Dist2POD']
    varName_weather = ["Tsfc_F_avg","Rh_PCT_avg","Psfc_MB_avg","CldCov_PCT_avg","Tapp_F_avg","Spd_MPH_avg","PcpPrevHr_IN"]
    varName_STcrime = [c+'_short' for c in CrimeTypes]
    varName_LTcrime = [c+'_long' for c in CrimeTypes]
    varName_311 = ["Street_Lights_All_Out","Street_Lights_One_Out","Alley_Lights_Out"]
    varName_crime_pres = [c+'_consec_pres' for c in CrimeTypes]
    varName_crime_abs = [c+'_consec_abs' for c in CrimeTypes]

    
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
    varName = np.array(varName_space + varName_pod + varName_weather + varName_STcrime + varName_LTcrime +\
                       varName_311 + varName_crime_pres + varName_crime_abs)
        
    var_type_idx = {}
    keys = ['space','POD','weather','LT','ST','311','pres','abs']
    subnames = [varName_space, varName_pod, varName_weather, varName_LTcrime, varName_STcrime,\
                varName_311, varName_crime_pres, varName_crime_abs]
    for key,subname in zip(keys,subnames):
        var_type_idx[key]=np.in1d(varName,subname)
 
         
    # training set
    structured_array_train = structure_feature_label_array(CrimeData, CrimeTypes, groups_train, varName, var_type_idx, grid_2d, \
                                                           SpatialFeature, POD_data, gauss_filter, gauss_exp_filter, \
                                                           period_LT, period_ST, cellsize_3d, WeatherData, CallsData, \
                                                           buffer_period_call, buffer_period_crime, mask_grdInCity, \
                                                           density=True, class_label=(0,1))    
    featureArray_train = structured_array_train['feature']
    labelArray_train = structured_array_train['label']
    
    # test set
    structured_array_test = structure_feature_label_array(CrimeData, CrimeTypes, groups_test, varName, var_type_idx, grid_2d, \
                                                          SpatialFeature, POD_data, gauss_filter, gauss_exp_filter, \
                                                          period_LT, period_ST, cellsize_3d, WeatherData, CallsData, \
                                                          buffer_period_call, buffer_period_crime, mask_grdInCity, \
                                                          density=True, class_label=(0,1))    
    featureArray_test = structured_array_test['feature']
    labelArray_test = structured_array_test['label']

    # down-sample training set for each target crime type
    sampleInd = [downsample(label,rand_state=rand_seed,shuffle=True) for label in labelArray_train.T] #list where each element is a ndarray of indices of down-sampled samples
    featureArray_train_balanced = {}
    label_train_balanced = {}
    for i, crimetype in enumerate(CrimeTypes):
        featureArray_train_balanced[crimetype] = featureArray_train[sampleInd[i],:]
        label_train_balanced[crimetype] = labelArray_train[sampleInd[i],i]
    
    # Save objects         
    save_struct_data(featureArray_train,featureArray_test,labelArray_train,labelArray_test,varName,\
                     CrimeTypes,filePath=filePath_save)   
        
