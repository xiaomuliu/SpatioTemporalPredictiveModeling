#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=================================================
Kernel Smoothing and Kernel Density Estimation
=================================================

Created on Wed Sep  7 15:41:28 2016

@author: xiaomuliu
"""
import numpy as np
from scipy.stats import binned_statistic_2d, binned_statistic_dd  
from sklearn.neighbors import KernelDensity 
from scipy.ndimage.filters import convolve, gaussian_filter
import sys
sys.path.append('..') 
from Grid.SetupGrid import array_to_geoIm

def bin_point_data_2d(points, grid, cellsize, stat='count', bins=None, geoIm=True, mask=None):
    """
    Bin two-dimensional point data with coordinate information to grid cells
    points: an array of coordinates of points
    grid: a tuple or a list of coordinates of grid cell center
    bins: a tuple or a list of the bin edges in each dimension (x_edge, y_edge = bins).
          Either grid or bins must be specified.
    geoIm: If true, the returned array will be arranged by geo-image index
    """
    pt_x = points[:,0]
    pt_y = points[:,1]
    grd_x, grd_y = grid
    if bins is None:
        # Since grd_x grd_y correspond to the center of each grid cell, we add half cell size to each side of the grid coordinate to
        # get the edge values
        xedges = np.r_[grd_x-cellsize[0]/2, grd_x[-1]+cellsize[0]/2]
        yedges = np.r_[grd_y-cellsize[1]/2, grd_y[-1]+cellsize[1]/2]
        bins = (xedges, yedges)
        
    
    # #counts (histogram)
    # Hist2d, _, _ = np.histogram2d(pt_x, pt_y, bins=(xedges, yedges))
    # Hist2d_vec = Hist2d.reshape(-1,order='F') # Pay attention to 'C order' and 'Fortran order'
    # Hist2d_masked_vec = Hist2d_vec[mask]
    
    # scipy.stats.binned_statistic_2d gives a generalization of a histogram2d function allowing 
    # the computation of the sum, mean, median, or other statistic of the values within each bin
    Bin2d_stat, _, _, _ = binned_statistic_2d(pt_x, pt_y, values=None, statistic=stat, bins=bins)
    
    if geoIm==True:
        Bin2d_stat = array_to_geoIm(Bin2d_stat)
    
    if mask is not None:
        Bin2d_stat[mask==False]=np.nan

    return Bin2d_stat
        

def bin_point_data_3d(points, grid, cellsize, stat='count', bins=None, geoIm=True, mask=None):
    """
    Bin three-dimensional point data with coordinate information to grid cells
    points: an array of coordinates of points
    grid: a tuple or a list of coordinates of grid cell center
    bins: a tuple or a list of the bin edges in each dimension.
          Either grid or bins must be specified.
    geoIm: If true, the returned array will be arranged by geo-image index for the first 2D space
    """

    grd_x, grd_y, grd_t = grid
    if bins is None:
        # Since grd_x grd_y correspond to the center of each grid cell, we add half cell size to each side of the grid coordinate to
        # get the edge values
        xedges = np.r_[grd_x-cellsize[0]/2, grd_x[-1]+cellsize[0]/2]
        yedges = np.r_[grd_y-cellsize[1]/2, grd_y[-1]+cellsize[1]/2]
        tedges = np.r_[grd_t, grd_t[-1]+cellsize[2]]
        bins = (xedges, yedges, tedges)
        
    Bin3d_stat, _, _ = binned_statistic_dd(sample=points, values=None, statistic=stat, bins=bins)
    
    if geoIm==True:
        for i in Bin3d_stat.shape[2]:
            Bin3d_stat[:,:,i] = array_to_geoIm(Bin3d_stat[:,:,i])
        
    if mask is not None:
        for i in Bin3d_stat.shape[2]:
            Bin3d_stat[:,:,i][mask==False]=np.nan

    return Bin3d_stat

  

def KDE_2d(points, kernel='gaussian', bw=1, grid=None, image=True, log=False):
    """
    Return kernel density estimated values 
    grid: an array of which each column gives one dimesion's grids. 
    The returned array is a stacked KDE values (ngrid * 1). 
    image: If true, then return reshape the output as images of which shape is determined by grid
    log: If true, return log density values
    """
    kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(points)
    
    if grid is None:
        grid = points
        
    grd_x, grd_y = grid
    grid_mesh = np.meshgrid(grd_x,grd_y)
    #grid_mesh_stacked = np.dstack(grid_mesh).reshape(len(grd_x)*len(grd_y),2)
    grid_mesh_stacked = np.vstack([grid_mesh[0].ravel(),grid_mesh[1].ravel()]).T
        
    density = kde.score_samples(grid_mesh_stacked) # score_samples returns log density values
    if log==False:
        density = np.exp(density)
    
    if image==True:
        density = density.reshape((len(grd_x),len(grd_y)),order='F')
    
    return density


# Theratically these two processes are not equivalent. 
# Process 1: Do KDE along each dimension which gives normalized density values. Then (outer) multiply these values. 
# Process 2: Do KDE based on all dimesions
# Then reason is the density normalization.
                    
def KDE_separable_3d(points, kernel={'space':'gaussian','time':'exponential'}, bw={'space':1,'time':1}, 
                     grid=None, image=True, log=False):
    """
    points: points' coordinates (x,y,t)
    grid: an array of which each column gives one dimesion's grids.
    The returned array is a stacked KDE values (ngrid * 1). If image==true,
    then return reshape the output as time series of images (3D array: 2D image * 1D time)
    of which shape is determined by grid. 
    
    Theratically the following two processes are not equivalent. 
    Process 1: Do KDE along each dimension which gives normalized density values. Then (outer) multiply these values. 
    Process 2: Do KDE based on all dimesions
    Then reason is the density normalization.
    """
    if grid is None:
        grid = points

    grd_x, grd_y, grd_t = grid
    # grid_mesh_stacked is of form [(x1,y1,t1),(x2,y1,t1),...,(x1,y2,t1),(x2,y2,t1),...,(x1,y1,t2),(x2,y1,t2),...,(xm,yn,tq)]
    # i.e. x-axis coordinates alternate fastest then y-axis then t-axis.
    #grid_mesh = np.meshgrid(grd_y,grd_t,grd_x)
    #grid_mesh_stacked = np.vstack([grid_mesh[2].ravel(), grid_mesh[0].ravel(), grid_mesh[1].ravel()]).T
    

    # Fisrt do 1D-KDE along t-axis
    # density_t dimension: nt by 1
    kde_t = KernelDensity(kernel=kernel['time'], bandwidth=bw['time']).fit(points[:,2][:,np.newaxis]) 
    density_t = kde_t.score_samples(grd_t[:,np.newaxis])
    
    # Then do 2D-KDE along xy-plane 
    # density_s dimension: (nx*ny) by 1       
    density_s = KDE_2d(points[:,:2], kernel=kernel['space'], bw=bw['space'], grid=(grd_x, grd_y),image=False)
    
    density_st = np.outer(density_s,density_t)
    
    if log==False:
        density_st = np.exp(density_st) 
        
    if image==True:
        density_st = density_st.reshape((len(grd_x),len(grd_y), len(grd_t)), order='F')
        
    return density_st    

def gaussian_filter_2d(bandwidth=(1,1), window_size=None):
    """
    Return a spatio gaussian filter with weights calculated by
    K(x,y) = 1/(2*pi*sigma_x*sigma_y)*exp(x^2/(2*sigma_x^2)+y^2/(2*sigma_y^2))
    bandwidth: (bw_x, bw_y)
    """
    if window_size is None:
        # truncated at 4 standard deviation for gaussian filter
        size_x, size_y = 9*bandwidth[0], 9*bandwidth[1]
    else:
        size_x, size_y = window_size
    
    h_x, h_y = bandwidth
    if (size_x % 2) == 0:
        start_x = -(size_x/2) 
        end_x = size_x/2
    else:
        start_x = -((size_x-1)/2)
        end_x = (size_x-1)/2
    if (size_y % 2) == 0:
        start_y = -(size_y/2) 
        end_y = size_y/2
    else:
        start_y = -((size_y-1)/2)
        end_y = (size_y-1)/2
        
    grd_x = np.linspace(start_x, end_x, size_x)
    grd_y = np.linspace(start_y, end_y, size_y)   

    filter_x = 1/float(np.sqrt(2*np.pi)*h_x)*np.exp(-0.5*grd_x**2/float(h_x**2))
    filter_y = 1/float(np.sqrt(2*np.pi)*h_y)*np.exp(-0.5*grd_y**2/float(h_y**2))    
    filter_s = np.outer(filter_x,filter_y)
    filter_s = filter_s/float(np.sum(filter_s)) # normalize gaussian filter
    return filter_s   
    
    
def kernel_smooth_2d_conv(points, grid, filter_2d, flatten=False):
    """
    Return kernel smoothed 2D array
    points: points' coordinates (x,y,t)
    grid: (x,y), an array of which each column gives one dimesion's grids. 
    flatten: If true, the returned values will be arranged in stacked form (vectorized images)
    
    Note:     
    This function utilizes scipy.ndimage convolve which only works with real valued data. 
    scipy.signal.convolve2D (and fftconvolve) are more generic, but much slower
    """
    # assuming a regular grid with equal cell size, the cellsize is of [size_x, size_y]
    cellsize = (np.abs(np.diff(grid[0][:2])), np.abs(np.diff(grid[1][:2])))
    binned_data = bin_point_data_2d(points, grid, cellsize, stat='count', geoIm=False)
   
    smoothed_data = convolve(binned_data, weights=filter_2d, mode='constant', cval=0.0) 
    
    if flatten==True:
        smoothed_data = smoothed_data.ravel(order='F')
              
    return smoothed_data 
    
def gaussian_kernel_smooth_2d(points, grid, sigma=1,truncate=4.0, flatten=False):
    """
    Return gaussian kernel smoothed 2D array
    points: points' coordinates (x,y,t)
    grid: (x,y), an array of which each column gives one dimesion's grids.
    truncate: truncate the filter at this many standard deviations
    Returned array of same shape as input.
    The returned array is an image of kernel smoothed values. If flatten==true,
    then return reshape the output as (ngrid * 1) 
    Note: 
    kernel_smooth_2d_conv is a generic function for other kernels
    """
    # assuming a regular grid with equal cell size, the cellsize is of [size_x, size_y]
    cellsize = (np.abs(np.diff(grid[0][:2])), np.abs(np.diff(grid[1][:2])))
    binned_data = bin_point_data_2d(points, grid, cellsize, stat='count', geoIm=False)
   
    smoothed_data = gaussian_filter(binned_data, sigma=sigma, mode='constant', cval=0.0, truncate=truncate) 
    
    if flatten==True:
        smoothed_data = smoothed_data.ravel(order='F')
          
    return smoothed_data     
    
      
def gaussian_exponential_filter_3d(bandwidth=(1,1,0.1), window_size=None):
    """
    Return a spatio-temporal filter with weights calculated by
    K(x,y,t) = 1/(2*pi*sigma_x*sigma_y)*exp(x^2/(2*sigma_x^2)+y^2/(2*sigma_y^2)) * exp(-lambda*t)
    bandwidth: (bw_x, bw_y, bw_t)
    """
    if window_size is None:
        # truncated at 4 standard deviation for gaussian filter and 2/rate for exponential filter
        size_x, size_y = 9*bandwidth[0], 9*bandwidth[1]
        size_t = 2/bandwidth[2]
    else:
        size_x, size_y, size_t = window_size
    
    h_x, h_y, h_t = bandwidth
    if (size_x % 2) == 0:
        start_x = -(size_x/2) 
        end_x = size_x/2
    else:
        start_x = -((size_x-1)/2)
        end_x = (size_x-1)/2
    if (size_y % 2) == 0:
        start_y = -(size_y/2) 
        end_y = size_y/2
    else:
        start_y = -((size_y-1)/2)
        end_y = (size_y-1)/2
        
    grd_x = np.linspace(start_x, end_x, size_x)
    grd_y = np.linspace(start_y, end_y, size_y)
    grd_t = np.linspace(0,size_t-1,size_t)
    
    filter_x = 1/float(np.sqrt(2*np.pi)*h_x)*np.exp(-0.5*grd_x**2/float(h_x**2))
    filter_y = 1/float(np.sqrt(2*np.pi)*h_y)*np.exp(-0.5*grd_y**2/float(h_y**2))    
    filter_s = np.outer(filter_x,filter_y)
    filter_s = filter_s/float(np.sum(filter_s)) # normalize gaussian filter
    filter_t = np.exp(-h_t*grd_t)
    filter_st = np.outer(filter_s,filter_t).reshape(filter_s.shape+filter_t.shape) # concatenate tuple A and tuple B: A+B
    return {'space':filter_s, 'time':filter_t, 'space-time':filter_st}

def gaussian_constant_filter_3d(bandwidth=(1,1), window_size=None, constant=1):
    """
    Return a spatio-temporal filter with weights calculated by
    K(x,y,t) = 1/(2*pi*sigma_x*sigma_y)*exp(x^2/(2*sigma_x^2)+y^2/(2*sigma_y^2)) * constant
    bandwidth: (bw_x, bw_y, bw_t)
    """
    if window_size is None:
        # truncated at 4 standard deviation for gaussian filter and 2/rate for exponential filter
        size_x, size_y = 9*bandwidth[0], 9*bandwidth[1]
        size_t = 10
    else:
        size_x, size_y, size_t = window_size
    
    h_x, h_y = bandwidth
    if (size_x % 2) == 0:
        start_x = -(size_x/2) 
        end_x = size_x/2
    else:
        start_x = -((size_x-1)/2)
        end_x = (size_x-1)/2
    if (size_y % 2) == 0:
        start_y = -(size_y/2) 
        end_y = size_y/2
    else:
        start_y = -((size_y-1)/2)
        end_y = (size_y-1)/2
        
    grd_x = np.linspace(start_x, end_x, size_x)
    grd_y = np.linspace(start_y, end_y, size_y)
    grd_t = np.linspace(0,size_t-1,size_t)
    
    filter_x = 1/float(np.sqrt(2*np.pi)*h_x)*np.exp(-grd_x**2/float(2*h_x**2))
    filter_y = 1/float(np.sqrt(2*np.pi)*h_y)*np.exp(-grd_y**2/float(2*h_y**2))  
    filter_s = np.outer(filter_x,filter_y)
    filter_s = filter_s/float(np.sum(filter_s)) # normalize gaussian filter
    filter_t = np.ones(len(grd_t))*constant
    filter_st = np.outer(filter_s,filter_t).reshape(filter_s.shape+filter_t.shape) # concatenate tuple A and tuple B: A+B
    return {'space':filter_s, 'time':filter_t, 'space-time':filter_st}


def kernel_smooth_separable_3d_conv(points, grid, filter_3d, flatten=False):
    """
    Return kernel smoothed 3D array
    points: points' coordinates (x,y,t)
    grid: (x,y,t), an array of which each column gives one dimesion's grids.
    filter_3d: filter weights values 
    The returned array is an image of kernel smoothed values. If flatten==true,
    then return reshape the output as (ngrid * 1) 
    
    NOTE:
    (a) This function utilizes scipy.ndimage convolve which only works with real valued data. 
    scipy.signal.convolve2D (and convolve, fftconvolve) are more generic, but much slower
    
    (b) Regarding convolved signal, using a 1D case to illustrate:
    a is of length m; b is of length n; assuming n is an even number
    sp.ndimage.filters.convolve1d(a,b,mode='constant') will return a length m signal
    where the first element is obtained by placing filter centered at a[0]. i.e.
    a[0]*b[n/2]+a[1]*b[n/2-1]+...+a[n/2]*b[0]. 
    like-wise for the tail of convolved signal.
    One can change the 'origin' argument in convolve function to shift filter.
    """
    # assuming a regular grid with equal cell size, the cellsize is of [size_x, size_y, size_t]
    # cellsize = np.r_[np.squeeze(np.abs(np.diff(grid[:2,:,0],axis=0))), np.abs(np.diff(grid[0,0,:2],axis=0))]
    cellsize = (np.abs(np.diff(grid[0][:2])), np.abs(np.diff(grid[1][:2])), np.abs(np.diff(grid[2][:2])))
    binned_data = bin_point_data_3d(points, grid, cellsize, stat='count', geoIm=False)
   
    smoothed_data = convolve(binned_data, weights=filter_3d, mode='constant', cval=0.0, origin=(0,0,-filter_3d.shape[2]//2)) 
    
    if flatten==True:
        smoothed_data = smoothed_data.ravel(order='F')          
    return smoothed_data 
        

def normalize_to_density(values):
    return values/float(np.nansum(values))    
    
           
if __name__=='__main__':
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    import cPickle as pickle
    import geopandas as gpd
    from Grid.SetupGrid import vectorized_grid, grid_within_bndy 
    import time
   
    # Plot kernels
    filters =  gaussian_exponential_filter_3d(bandwidth=(2,2,0.1),window_size=(11,11,20),)
    filter_s = filters['space']
    filter_t = filters['time']

    X = np.linspace(-5, 5, 11)
    Y = np.linspace(-5, 5, 11)
    Xgrd, Ygrd = np.meshgrid(X,Y)
    T = np.linspace(0,19,20)
    fig = plt.figure()
    #ax1= fig.add_subplot(1, 2, 1, projection='3d')
    #ax1.plot_surface(Xgrd,Ygrd,filter_s,cmap='jet')
    #ax1.set_zlim((0,0.005))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.contour(Xgrd,Ygrd,filter_s,cmap='jet')
    ax1.set_title('Gaussian filter')  
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(T,filter_t,'ko--')
    ax2.set_title('exponential filter')
    plt.show()
    
    
    # Apply to crime data
    filePath = "../SharedData/CrimeData/"
    pkl_file = open(filePath+'BURGLARY_08_14_grouped.pkl', 'rb')
    CrimeData = pickle.load(pkl_file)
    pkl_file.close() 
    
  
    # Setup grid
    path_GIS = "../SharedData/GISData/"
    shpfile_city = path_GIS + "City_Boundary/City_Boundary.shp"
    city_shp = gpd.read_file(shpfile_city)
        
    cellsize = (500,500)
    grd_vec, grd_x, grd_y = vectorized_grid(city_shp,cellsize)
    mask_grdInCity = grid_within_bndy(grd_vec,city_shp)['mask']
    grd_vec_inCity = grd_vec[mask_grdInCity,:]
    
    # mask in image form
    mask_grdInCity_im = grid_within_bndy([grd_x,grd_y],city_shp,im2d_mask=True,geoIm=True)['mask']


    # Compare two kernel filtering methods (KDE vs. convolution) using a subset of data
    CrimePts = CrimeData.ix[:,['X_COORD','Y_COORD','GROUP']][:10000]
    grd_t = np.unique(CrimePts['GROUP'].values)
    CrimePts = CrimePts.values
    grid_2d = (grd_x,grd_y)
    grid_3d = (grd_x,grd_y,grd_t)
    
    
    # 2D cases
    start = time.time()
    KS2d = gaussian_kernel_smooth_2d(CrimePts[:,:-1], grid_2d, sigma=1,truncate=4.0) # note: the bandwidth is of unit in pixel
    end = time.time()
    print('2D convolution: %.4f s' % (end-start))
        
    start = time.time()
    KDE2d = KDE_2d(CrimePts[:,:-1], kernel='gaussian', bw=500, grid=grid_2d) # note: the bandwidth is of unit in coordinate
    end = time.time()
    print('2D KDE: %.4f s' % (end-start))
    
    # 3D cases
    start = time.time()
    filter_3d = gaussian_exponential_filter_3d(bandwidth=(1,1,0.1),window_size=(9,9,10))['space-time']
    KS3d = kernel_smooth_separable_3d_conv(CrimePts, grid_3d, filter_3d) 
    end = time.time()
    print('3D convolution: %.4f s' % (end-start))
        
    start = time.time()
    KDE3d = KDE_separable_3d(CrimePts, kernel={'space':'gaussian','time':'exponential'}, bw={'space':500,'time':0.1}, grid=grid_3d)
    end = time.time()
    print('3D KDE: %.4f s' % (end-start))
    
     
    # plot smoothed/density images
    # 2D
    KS2d_flip = array_to_geoIm(KS2d)
    KS2d_flip[mask_grdInCity_im==False] = np.nan
    KDE2d_flip = array_to_geoIm(KDE2d)
    KDE2d_flip[mask_grdInCity_im==False] = np.nan

    # verify flattened image pixel values equal to the 2d form image values
    KS2d_flatten = gaussian_kernel_smooth_2d(CrimePts[:,:-1], grid_2d, sigma=1,truncate=4.0, flatten=True)
    KS2d_flatten[mask_grdInCity==False] = np.nan
    KS2d_flatten = KS2d_flatten.reshape((len(grd_x),len(grd_y)),order='F')
    KS2d_flatten_flip = array_to_geoIm(KS2d_flatten)
    #np.all(KS2d_flatten_flip==KS2d_flip)  #nan!=nan
    #neq = np.where(KS2d_flatten_flip!=KS2d_flip)
    #np.all(np.isnan(KS2d_flip[neq]))
    #np.all(np.isnan(KS2d_flatten_flip[neq]))
    print (np.allclose(KS2d_flatten_flip, KS2d_flip, rtol=1e-05, atol=1e-08, equal_nan=True))
    
    
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(121)
    im = plt.imshow(KS2d_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D kernel smoothed image of binned burglary count')
    plt.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(KDE2d_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D KDE of burglary point data')
    plt.colorbar(im)
    
    # normalize (kernel smoothed) intensiy images to density images
    # also re-normalize KDE density images after masking
    KS2d_flip_den = normalize_to_density(KS2d_flip)
    KDE2d_flip_den = normalize_to_density(KDE2d_flip)
    
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(121)
    im = plt.imshow(KS2d_flip_den, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D density image of binned burglary count')
    plt.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(KDE2d_flip_den, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D density of burglary point data')
    plt.colorbar(im)
    
    
    # compare smoothing by ndi.convolution and ndi.gaussian_filter
    filter_2d = gaussian_filter_2d(bandwidth=(1,1), window_size=(9,9))
    KS2d_conv = kernel_smooth_2d_conv(CrimePts[:,:-1], grid_2d, filter_2d)
    KS2d_conv_flip = array_to_geoIm(KS2d_conv)
    KS2d_conv_flip[mask_grdInCity_im==False] = np.nan
    print (np.allclose(KS2d_conv_flip, KS2d_flip, rtol=1e-05, atol=1e-08, equal_nan=True))
    
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(121)
    im = plt.imshow(KS2d_conv_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D kernel smoothed image by ndi.convolve')
    plt.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(KS2d_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D kernel smoothed image by ndi.gaussian_filter')
    plt.colorbar(im)
    
    
    
    # For 3D smoothing, take one time slice to show
    KS3d_flip = array_to_geoIm(KS3d[:,:,20])
    KS3d_flip[mask_grdInCity_im==False] = np.nan
    KDE3d_flip = array_to_geoIm(KDE3d[:,:,20])
    KDE3d_flip[mask_grdInCity_im==False] = np.nan
    
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(121)
    im = plt.imshow(KS3d_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('3D kernel smoothed image of binned burglary count')
    plt.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(KDE3d_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('3D KDE of burglary point data')
    plt.colorbar(im)
    
    
    # Verify 3D smoothing with constant filter on t-axis by comparing with 2D kernel smoothing
    # When filtering along t-axis with an constant filter, then 2D case is just a cross-sectional example of 3D smoothing
    # Pick slice #21 (index 20), since window_size in t is 10, 2d points should be gathered from group 12 to 21 
    # to be used in 2d smoothing
    
    
    filter_3d = gaussian_constant_filter_3d(bandwidth=(1,1),window_size=(9,9,10))['space-time']
    KS3d_const = kernel_smooth_separable_3d_conv(CrimePts, grid_3d, filter_3d) 
    
    KS3d_const_flip = array_to_geoIm(KS3d_const[:,:,20])
    KS3d_const_flip[mask_grdInCity_im==False] = np.nan 
    
    CrimePts_sub = CrimeData.ix[(CrimeData['GROUP']>=12) & (CrimeData['GROUP']<=21),['X_COORD','Y_COORD','GROUP']].values
    # convert standard deviation
    KS2d_sub = gaussian_kernel_smooth_2d(CrimePts_sub[:,:-1], grid_2d, sigma=1,truncate=4)
    KS2d_sub_flip = array_to_geoIm(KS2d_sub)
    KS2d_sub_flip[mask_grdInCity_im==False] = np.nan
       
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(121)
    im = plt.imshow(KS3d_const_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('3D smoothing with constant temporal filter')
    plt.colorbar(im)
    ax = fig.add_subplot(122)
    im = plt.imshow(KS2d_sub_flip, interpolation='nearest', origin='upper', cmap='jet')
    ax.set_title('2D smoothing')
    plt.colorbar(im)
    
    print (np.allclose(KS2d_sub_flip, KS3d_const_flip, rtol=1e-05, atol=1e-08, equal_nan=True))