#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
========================================
Spatial feature extraction

Add new: 
    1. RTM data (gas station, night club...)
    2. varying patch size when calculating geospatial attribute densities,
       providing a density for each patch size
    3. POD data (proximity is calcuated for each pair of grid cell and POD so that
       time varying proximity features can be derived later in the feature construction stage)
========================================

Created on Wed Nov 16 11:27:50 2016

@author: xiaomuliu
"""

import numpy as np
import geopandas as gpd
import pandas as pd
#import shapely.geometry as geom
from SpatialRelation import pt2geoObj_dist, pt_within_poly_stat, line_intersect_poly_stat, pt_poly_membership
from scipy.spatial import distance

def extract_proximity(grid,fields,shpfilenames):          
    minDist = np.zeros((len(grid),len(fields)))
    for i, filename in enumerate(shpfilenames):
        shpfile = gpd.read_file(filename)
        if 'Apartment_Complexes' in filename:
            # Apartment_Complexes has two entries with negative cocrdinates (bad values)
            keep = np.array([pt.coords[0]>(0,0) for pt in shpfile.geometry])
            shpfile = shpfile[keep]
        minDist[:,i] = pt2geoObj_dist(grid,shpfile)
    return minDist    

def extract_geo_attr_density(grid,fields,shpfilenames,patchsizes=None,proj_str=None):
    if patchsizes is None:
        patchsizes = [(grid[1,0]-grid[0,0],grid[1,1]-grid[0,1])]
    if proj_str is None:
        proj_str = "+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 " + \
              "+x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs +ellps=GRS80 +towgs84=0,0,0"  
        
    spAttrDen = np.zeros((len(grid),len(patchsizes)*len(fields)))   
    for i, filename in enumerate(shpfilenames):
        shpfile = gpd.read_file(filename)
        for j, patch_sz in enumerate(patchsizes): 
            if all(shpfile.geometry.geom_type == 'Point') or all(shpfile.geometry.geom_type =='Polygon'):
                spAttrDen[:,len(patchsizes)*i+j] = pt_within_poly_stat(grid,shpfile,patch_sz,stat_fun=np.nansum,proj=proj_str)
            elif all(shpfile.geometry.geom_type == 'LineString'):
                spAttrDen[:,len(patchsizes)*i+j] = line_intersect_poly_stat(grid,shpfile,patch_sz,stat_fun=np.nansum,proj=proj_str) 
    return spAttrDen            

def extract_bldg_attr(grid,fields,shpfilenames,patchsizes=None,proj_str=None):
    if patchsizes is None:
        patchsizes = [(grid[1,0]-grid[0,0],grid[1,1]-grid[0,1])]
    if proj_str is None:
        proj_str = "+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 " + \
              "+x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs +ellps=GRS80 +towgs84=0,0,0" 
        
    bldgAttr = np.zeros((len(grid),len(patchsizes)*len(fields))) 
    shpfile = gpd.read_file(shpfilenames)
    shpfile = shpfile[(shpfile['NON_STANDA'].isnull() | (shpfile['NON_STANDA']=='RSGARAGE')) & \
                      (shpfile['BLDG_STATU']=="ACTIVE") & (shpfile['X_COORD']!=0) & (shpfile['Y_COORD']!=0)]                
    for i, attr in enumerate(fields):
        if 'Bldg' in attr:
            if 'Story' in attr:
                shpfile_sub = shpfile.ix[shpfile['NON_STANDA']!='RSGARAGE',['STORIES','geometry']]
                field = 'STORIES'
            elif 'Unit' in attr:
                shpfile_sub = shpfile.ix[shpfile['NON_STANDA']!='RSGARAGE',['NO_OF_UNIT','geometry']]
                field = 'NO_OF_UNIT'
            else:
                shpfile_sub = shpfile.ix[shpfile['NON_STANDA']!='RSGARAGE',['geometry']]
                field = None
        elif 'Garage' in attr:
            shpfile_sub = shpfile.ix[shpfile['NON_STANDA']=='RSGARAGE',['geometry']]
            field = None
                  
        if 'Avg' in attr:
            statFun = np.nanmean
        elif 'Den' in attr:
            statFun = np.nansum
        
        for j, patch_sz in enumerate(patchsizes):    
            bldgAttr[:,len(patchsizes)*i+j] = pt_within_poly_stat(grid,shpfile_sub,patch_sz,field=field,stat_fun=statFun,proj=proj_str)     
    
    return bldgAttr
    
    
def extract_census(grid, census_filename, shpfilename, proj_str=None):
    if proj_str is None:
        proj_str = "+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 " + \
              "+x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs +ellps=GRS80 +towgs84=0,0,0" 
    
    CommCensus = pd.read_csv(census_filename)
    shpfile = gpd.read_file(shpfilename)
    
    # Dataframe read by pd.read_csv has already removed the '.' and '_' in field name strings
    # import re
    # CommCensus.columns = np.array([re.sub('\\.','_',name) for name in CommCensus.columns.values])
    # CommCensus.columns = np.array([re.sub('__','_',name) for name in CommCensus.columns.values])
    CommCensus.rename(columns={'Community Area Number':'AREA_NUMBE'},inplace=True)
    CommCensus["AREA_NUMBE"] = CommCensus["AREA_NUMBE"].astype('category')
    fields = CommCensus.columns.values[2:].tolist() # fields about census (excluding community area number and name)
    # field_census = [s.strip() for s in field_census] # remove white spaces on the sides of the strings
    
    shpfile.drop('AREA_NUM_1', axis=1, inplace=True)
    # NOTE: pandas dataframe category has an associated dtype. 
    # Convert AREA_NUMBE to int so that it matches AREA_NUMBE in census dataframe
    shpfile["AREA_NUMBE"] = shpfile["AREA_NUMBE"].astype('int').astype('category')
    # community_shp["AREA_NUMBE"].cat.set_categories=range(1,len(community_shp)+1)
    # Merge community demographic dataframe and community (shapefile) geo-dataframe
    shpfile = shpfile.merge(CommCensus,on="AREA_NUMBE")
    
    census = pt_poly_membership(grid,shpfile,field=fields,proj=proj_str)['stat']
    
    return census, fields

def extract_POD(grid, PODdata):
    POD_coords = PODdata.ix[:,['X_COORD','Y_COORD']].values
    POD_id = PODdata['POD_ID'].values
    dist_mat = distance.cdist(grid, POD_coords, 'euclidean')
    PODdist_df = pd.DataFrame(dist_mat,columns=POD_id)
    PODdist_df.index.name = 'GRID_IDX'
    PODdist_df.columns.name = 'POD_ID'
    return PODdist_df
    
if __name__=='__main__':
    import time
    import cPickle as pickle
    #from fiona.crs import from_string
    import re
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    from StructureData.LoadData import load_grid
    
    args = ParseArg()
    infiles = args['input']
    outpath = args['output']
    params = args['param']
   
    infile_match = re.findall('([\w\./]+)',infiles)     
    grid_pkl, filePath_GIS, filePath_census, filePath_POD = infile_match
    
    filePath_save = outpath if outpath is not None else '../SharedData/FeatureData/'
     
    param_match = re.findall('\d*\.\d+|\d+', params)
    patch_ratios = [float(p) for p in param_match]

                       
    # projection (PROJ.4) string
    proj_str = "+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 " + \
              "+x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs +ellps=GRS80 +towgs84=0,0,0"  
                 
    # Proximity fields and files
    RTM_proximity = ['Apartment_Complexes','ATMBanks','Bars','Gas_Stations','Grocery_Stores',\
           'HealthcareCenters_&_Gymnasiums','HomelessShelters','Laundromats','LiquorStores','NightClubs',\
           'Post_Offices','Recreation_Centers','Rental_Halls','Retail_Shops','Variety_Stores']
    RTM_proximity = ['RTM/Chicago_'+s for s in RTM_proximity]

    shpfilename_proximity = RTM_proximity + ['Major_Streets/Major_Streets','police_stations_poly/police_stations',\
                                         'School_Grounds/School_Grounds','Parks_Aug2012/Parks_Aug2012','Hospitals/Hospitals',\
                                         'Libraries/Libraries','CTA_RailLines/CTA_RailLines','CHA/cha_locations']                                   
    shpfilename_proximity = [filePath_GIS + s + '.shp' for s in shpfilename_proximity]
    field_proximity = ['Apt','ATM','Bar','GasStation','Grocery','Gym','Shelter','Laundromat',\
                       'Liquor','NightClub','PO','Recreation','Rental','Retail','VarietyShop',\
                       'Street','CPDstation','School','Park','Hospital','Library','CTArail',\
                       'CHA']
    field_proximity = ['Dist2'+f for f in field_proximity]  
    
    # Spatial Density fields and files
    RTM_density = ['Bars','LiquorStores']
    RTM_density = ['RTM/Chicago_'+s for s in RTM_density]

    shpfilename_density = RTM_density + ['Street_Center_Line/Transportation','CTA_BusStops/CTA_BusStops']
    shpfilename_density = [filePath_GIS + s + '.shp' for s in shpfilename_density]
    field_density = ['Bar','Liquor','Street','BusStop']
    field_density = [f+'Den' for f in field_density]

    # Building fields and files
    shpfilename_building = filePath_GIS + 'Buildings/Buildings.shp' 
    field_building = ['BldgDen','GarageDen','AvgBldgStory','AvgBldgUnit']

    # Demographics fields and files
    shpfilename_demographic = filePath_GIS + 'Community_bndy/CommAreas.shp'
    census_filename = filePath_census + 'Census_Data_socioeconomic_2008_2012.csv'
    CommCensus = pd.read_csv(census_filename)
    field_census = CommCensus.columns.values[2:].tolist()
    
    field_density_ext = [f+'_'+str(i) for f in field_density for i in xrange(len(patch_ratios))]
    field_building_ext = [f+'_'+str(i) for f in field_building for i in xrange(len(patch_ratios))]
    featureName = field_proximity + field_building_ext + field_density_ext + field_census
    featureName = [name.strip() for name in featureName] # remove white spaces on the sides of the strings

    # POD data               
    pod_filename = filePath_POD +"PODs_05_14_grouped.pkl"
    with open(pod_filename,'rb') as input_file:
        PODdata = pickle.load(input_file)               
        
    # load grid
    _, grd_x, grd_y, grd, _, _ = load_grid(grid_pkl)
    cellsize = (grd_x[1]-grd_x[0],grd_y[1]-grd_y[0]) 
    patchsizes = [(r*cellsize[0],r*cellsize[1]) for r in patch_ratios]

                  
    # ==================== Extract features ================================ #               
    # Proximity   
    start = time.time()
    minDist = extract_proximity(grd,field_proximity,shpfilename_proximity)
    end = time.time()
    print('Running time (proximity features) %.1f' % (end - start))

    # Spatial attribute density       
    start = time.time()     
    spAttrDen = extract_geo_attr_density(grd,field_density,shpfilename_density,patchsizes,proj_str)
    end = time.time()        
    print('Running time (spatial attribute density features) %.1f' % (end - start))
        
    # Building attributes   
    start = time.time() 
    bldgAttr = extract_bldg_attr(grd,field_building,shpfilename_building,patchsizes,proj_str)
    end = time.time()        
    print('Running time (building features) %.1f' % (end - start))
    
    # Demographics                      
    start = time.time()                  
    census, _ = extract_census(grd, census_filename, shpfilename_demographic, proj_str)
    end = time.time()        
    print('Running time (census features) %.1f' % (end - start))

    # Concatenate all features into a dataframe of shape (N_grids * N_features)
    featureArry = np.c_[minDist,bldgAttr,spAttrDen,census]
    spFeature_df = pd.DataFrame(featureArry,columns=featureName)
      
#    # convert pandas dataframe to geopandas geodataframe
#    grd_geometry = [geom.Point(xy) for xy in zip(grd[:,0].tolist(), grd[:,1].tolist())]
#    # convert PORJ4 string to dict
#    projCRS = from_string(proj_str)
#    spFeature_gdf = gpd.GeoDataFrame(spFeature_df, crs=projCRS, geometry=grd_geometry)
    
#    # Plot pure spatial feature maps
#    for feature_name in featureName:
#        spFeature_gdf.plot(column=feature_name,cmap='jet')
         
    # Computes distance between each pair of the two collections (a) grid coords (b) POD coords.        
    PODdist_df = extract_POD(grd, PODdata)
    
    # ================== save feature objects ============================== #      
    savefile_nparray = filePath_save+'SpFeature_nparray.pkl'
    with open(savefile_nparray,'wb') as output:
        pickle.dump(featureArry, output, pickle.HIGHEST_PROTOCOL)
    
    savefile_df = filePath_save+'SpFeature_dataframe.pkl'
    with open(savefile_df,'wb') as output:
        pickle.dump(spFeature_df, output, pickle.HIGHEST_PROTOCOL)
    
#    savefile_gdf = filePath_save+'SpFeature_geoDataframe.pkl'
#    with open(savefile_gdf,'wb') as output:
#        pickle.dump(spFeature_gdf, output, pickle.HIGHEST_PROTOCOL)
        
    savefile_pod = filePath_save+'PODdist_dataframe.pkl'
    with open(savefile_pod,'wb') as output:
        pickle.dump(PODdist_df, output, pickle.HIGHEST_PROTOCOL)    