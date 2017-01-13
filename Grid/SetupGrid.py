# -*- coding: utf-8 -*-
"""
==================================
Specify Spatial Grids
==================================
Created on Mon Aug 22 14:42:26 2016

@author: xiaomuliu
"""
      
import numpy as np
from shapely.geometry import Point  
          
def vectorized_grid(shpfile,cellsize,center=True):
    cellsize_x, cellsize_y = cellsize
    range_x = shpfile.bounds[['minx','maxx']].values[0].astype(int)
    range_y = shpfile.bounds[['miny','maxy']].values[0].astype(int)
    # grids cover a rectangular area
    grd_x = np.arange(range_x[0], range_x[1], step=cellsize_x, dtype=int)
    grd_y = np.arange(range_y[0], range_y[1], step=cellsize_y, dtype=int)
    
    if center:
        # assume grid coordinates are the center of cells (for later calulating points in polygon)
        # shift by half cell size
        grd_x = grd_x+cellsize_x/2
        grd_y = grd_y+cellsize_x/2
    
    grd_rect = np.meshgrid(grd_x,grd_y)    
    # reshape the mesh grid of a vector of points of form [(x1,y1),(x2,y1),(x3,y1),...,(x1,y2),(x2,y2),...,(xn,ym)]
    grd_vec = np.dstack(grd_rect).reshape(len(grd_x)*len(grd_y),2)
    # equivalently,
    # grd_vec = np.vstack([grd_rect[0].ravel(), grd_rect[1].ravel()]).T
    
    return grd_vec, grd_x, grd_y

    
def array_to_geoIm(array2d):
    """
    # ************************************************************ #
    # The input 2d  matrix are in the form of
    #         y_coord   181xxxx, ..., 195xxxx 
    # x_coord             
    # 109xxxx           val_00, ...,  val_0N
    # ...
    # 120xxxx           val_M0, ...,  val_MN
    #
    #
    # Return a re-arrange matrix with elements to be as the following
    #         x_coord   109xxxx, ..., 120xxxx 
    # y_coord             
    # 195xxxx           val_0N, ...,  val_MN
    # ...
    # 181xxxx           val_00, ...,  val_M0
    
    # ************************************************************ #
    """ 
    flipped_array = np.copy(np.fliplr(array2d).T) 
    return flipped_array


def grid_within_bndy(grid,shpfile,poly_index=0,im2d_mask=False,geoIm=False):
    """
    Returns grid cells within boundary which is defined by a polygon (indicated by poly_index) from the shpfile,
    as well as the corresponding mask.
    'grid' must be provided by a list with two vectors (or a 2d array) points coordinates
    (one for x coordinates, one for y coordinates)
    If im2d_mask=False, these coordinates can be of irregular grids and all grid coordinates must be given.
    If im2d_mask=True, the corrdinates can just be x-direction and y-direction coordinates.
    And the returned mask is in 2d (image) matrix form.
    X and y coordinates in grid are assumed to be in ascending order. 
    """
    if im2d_mask==True:
        grd_mesh = np.meshgrid(grid[0],grid[1])
        grid_vec = np.dstack(grd_mesh).reshape(len(grid[0])*len(grid[1]),2)
    else:
        grid_vec = grid
        
    mask_grdInPoly = [shpfile.geometry[poly_index].contains(Point(pt)) for pt in grid_vec]
    mask_grdInPoly = np.array(mask_grdInPoly)
    grd_inPoly = grid_vec[mask_grdInPoly,:]

    if im2d_mask==True:
        mask_grdInPoly = mask_grdInPoly.reshape((len(grid[0]),len(grid[1])),order='F') #column major 
        if geoIm==True:
            # please see the docstring of function 'array_to_geoIm' for the specifiction of x-y coordination layout
            mask_grdInPoly = array_to_geoIm(mask_grdInPoly)
            
    return {'mask':mask_grdInPoly, 'grid':grd_inPoly}


def flattened_to_geoIm(flattened_array,nx,ny,mask=None):
    """
    Convert flattened grids to image (in geospatial order) 
    """
    if mask is None:
        mask = np.ones(nx*ny).astype('bool')
        
    geo_array = np.ones(nx*ny)*np.nan
    geo_array[mask] = flattened_array
    geo_im = geo_array.reshape((nx,ny),order='F')
    geo_im = array_to_geoIm(geo_im)

    return geo_im

    
def flattened_to_grdIm(flattened_array,nx,ny,mask=None):
    """
    Convert flattened grids to matrix (in mesh grid order) 
    """
    if mask is None:
        mask = np.ones(nx*ny).astype('bool')
        
    grd_array = np.ones(nx*ny)*np.nan
    grd_array[mask] = flattened_array
    grd_im = grd_array.reshape((nx,ny),order='F')

    return grd_im

def get_cluster_mask(mask_grdInCity,cluster_label, target_cluster):
    """
    Given the cluster label for the grids in the city, return the target cluster's
    mask for the flattened rectangular grid.
    """
    cluster_mask = np.array(cluster_label==target_cluster)
    mask_grdInCluster = np.zeros(mask_grdInCity.shape).astype('bool')
    idx_inCity = np.where(mask_grdInCity==True)[0]
    idx_inCluster = idx_inCity[cluster_mask] 
    mask_grdInCluster[idx_inCluster] = True  
    return mask_grdInCluster
    
