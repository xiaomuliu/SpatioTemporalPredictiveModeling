# -*- coding: utf-8 -*-
"""
===================================
Spatial relation functions
===================================
Created on Tue Aug 23 15:29:26 2016

@author: xiaomuliu
"""

import shapely.geometry as geom
import numpy as np
import geopandas as gpd

def pts_to_gpdDf(pts,field_name=None,field_val=None):
    """
    Convert points in a numpy array to geopandas dataframe (with optional field)  
    """
    series_points = gpd.GeoSeries([geom.Point(x, y) for x, y in pts])
    df_points = gpd.GeoDataFrame(series_points)
    if field_name is not None and field_val is not None:  
        df_points[field_name] = field_val
        df_points.columns = ['Geometry', field_name]     
    else:
        df_points.columns = ['Geometry']
    
    return df_points 


def gpdObj_to_npObj(gpdObj):
    """
    Convert values in a geopandas dataframe/series to numpy array 
    """
    npObj = gpdObj.apply(lambda x: [np.asarray(x_i) for x_i in x]).values
    return npObj


def min_dist(point, geom_obj):
    return geom_obj.distance(point).min()


def pt2geoObj_dist(grid,shpfile):
    """
    Return the distances of grid points to their nearest spatial geometry objects (from shapefile)  
    """
    gpd_pts = gpd.GeoSeries(geom.Point(x,y) for x, y in grid)
    # ##############################################################
    # This code below has error, need to figure out the reasons.
    # gpd_minDist = gpd_pts.geometry.apply(min_dist, args=(shpfile))
    # minDist = gpdObj_to_npObj(gpd_minDist)
    # #############################################################
    minDist = np.zeros(len(grid))    
    for i in xrange(len(minDist)):
        minDist[i] = shpfile.distance(gpd_pts[i]).min()
    
    return minDist

    
# NOTE: 'pt_within_poly_stat' and 'line_intersect_poly_stat' can be written as a generic function where a polyon patch can be 
# constructed as a shapely geometry object then methods such as 'within()' and 'intersect()' will be invoked. However,
# for rectanguler patches, using logic opereation for 'within' case may be faster.    
    
def pt_within_poly_stat(grid,shpfile,patchsize,field=None,stat_fun=None,proj=None,*args,**kwargs):
    """
    Return the descriptive statistics of spatial geometry objects (points/polygons from shapefile) 
    within a patch centered at a grid point (e.g. count of points within a polygon patch).
    Polygon patch types only support rectangular for now.
    """
    stat = np.zeros(len(grid))
    if proj is None:
        proj = shpfile.crs
    if stat_fun is None:
        stat_fun = np.nansum    
        
    shpfile = shpfile.to_crs(proj)
    shpfile['ctr'] = shpfile.centroid
    # shpfile = shpfile.set_geometry('ctr')
    shpfile['ctr_x'] = shpfile['ctr'].apply(lambda x: int(x.x))
    shpfile['ctr_y'] = shpfile['ctr'].apply(lambda x: int(x.y))

    for i, grd_pt in enumerate(grid):
        minx, maxx = grd_pt[0]-np.floor(patchsize[0]/2), grd_pt[0]+np.ceil(patchsize[0]/2)
        miny ,maxy = grd_pt[1]-np.floor(patchsize[1]/2), grd_pt[1]+np.ceil(patchsize[1]/2) 
        # NOTE the diffference between 'and' and '&', as well as 'logically True' and 'bit-wise True'
        # (np array vector-based logic operations)
        isInPatch = (shpfile['ctr_x'].values >= minx) & (shpfile['ctr_x'].values <= maxx) &\
                    (shpfile['ctr_y'].values >= miny) & (shpfile['ctr_y'].values <= maxy)
        
        if field is None:
            stat[i] = stat_fun(isInPatch, *args, **kwargs)
        else:
            if all(isInPatch==False):
                stat[i] = 0
            else:
                stat[i] = stat_fun(shpfile[field][isInPatch].values,*args,**kwargs)
        
    return stat    

def line_intersect_poly_stat(grid,shpfile,patchsize,field=None,stat_fun=None,proj=None,*args,**kwargs):
    """
    Return the descriptive statistics of spatial line objects (from shapefile) 
    for a patch centered at a grid point (e.g. count of line segements within a polygon patch).
    Polygon patch types only support rectangular for now.
    """
    stat = np.zeros(len(grid))
    if proj is None:
        proj = shpfile.crs
    if stat_fun is None:
        stat_fun = np.nansum
        
    shpfile = shpfile.to_crs(proj)

    for i, grd_pt in enumerate(grid):
        minx, maxx = grd_pt[0]-np.floor(patchsize[0]/2), grd_pt[0]+np.ceil(patchsize[0]/2)
        miny ,maxy = grd_pt[1]-np.floor(patchsize[1]/2), grd_pt[1]+np.ceil(patchsize[1]/2)     
        # Rectangular polygons can be constructed using the shapely.geometry.box() function.
        # It makes a rectangular polygon from the provided bounding box values, with counter-clockwise order by default.
        # For constructing other polgyon, refer to shapely.geometry.polygon        
        patch_poly = geom.box(minx, miny, maxx, maxy, ccw=True)
        # intersects() contains any of these relations: contains(), crosses(), equals(), touches(), and within().        
        isIntersectPatch = [patch_poly.intersects(line) for line in shpfile.geometry]
        #gpd_isIntersectPatch = shpfile.geometry.apply(patch_poly.intersects)
        #isIntersectPatch = gpdObj_to_npObj(gpd_isIntersectPatch)
        if field is None:
            stat[i] = stat_fun(isIntersectPatch, *args, **kwargs)
        else:
            if all(isIntersectPatch==False):
                stat[i] = 0
            else:
                stat[i] = stat_fun(shpfile[field][isIntersectPatch].values,*args,**kwargs)
        
    return stat  

def pt_poly_membership(grid,shpfile,field=None,stat_fun=None,proj=None,*args,**kwargs):
    """
    Return the indicator array specifying which polygons a point from 'grid' falls into,
    if necessary, calculate descriptive statistics for each point where the sample values 
    are drawn from the 'shapefile' dataframe regarding the corresponding polygons.
    """
    stat = np.zeros((len(grid),len(field)))
    isInPoly = np.zeros((len(grid),len(shpfile.geometry)),dtype=bool)
    if proj is None:
        proj = shpfile.crs
    if stat_fun is None:
        stat_fun = lambda x: x
        
    shpfile = shpfile.to_crs(proj)    
    for i, grd_pt in enumerate(grid):
        isInPoly[i,:] = np.array([poly.contains(geom.Point(grd_pt)) for poly in shpfile.geometry])
        if all(isInPoly[i,:]==False):
            stat[i,:] = np.nan
        else:
            stat[i,:] = shpfile.ix[isInPoly[i,:],field].apply(stat_fun, axis=1, *args, **kwargs).values    

    return {'indicator':isInPoly,'stat':stat}
