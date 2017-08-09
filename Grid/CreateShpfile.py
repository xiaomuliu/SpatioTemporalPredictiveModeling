#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:29:55 2017

reference: 
    https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    http://gis.stackexchange.com/a/52708/8104

@author: xiaomuliu
"""

from osgeo import ogr
from math import ceil

def create_poly_from_coords(outLayer, featureDefn, coords, data=None):  
    # Create ring        
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    poly = poly.ExportToWkt()
                
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(poly)
    
#    # Field needs to be created with the data type defined. This functionality is not available for now
#    if data is not None:
#        # 'data' is a dict with keys as field names and values and corresponding values
#        for k,v in data.items():
#            outLayer.CreateField(ogr.FieldDefn(k, ogr.OFTInteger))
#            outFeature.SetField(k, v)
            
    outLayer.CreateFeature(outFeature)        
    outFeature = None        
    return outLayer

def create_regular_grid_poly(outLayer, featureDefn, xmin, xmax, ymin, ymax, gridHeight, gridWidth, data=None):
    # get rows and columns
    rows = ceil((ymax-ymin)/float(gridHeight))
    cols = ceil((xmax-xmin)/float(gridWidth))

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin+gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight
    
   # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
#            # Field needs to be created with the data type defined. This functionality is not available for now
#            if data is not None:
#                # 'data' is a list of dicts of which length equal to the number of grid cells 
#                # In the list each dict element has keys as field names and values and corresponding values
#                for k,v in data[countcols+countrows-2].items():
#                    outLayer.CreateField(ogr.FieldDefn(k, ogr.OFTInteger))
#                    outFeature.SetField(k, v)
            outLayer.CreateFeature(outFeature)
            outFeature = None
            
            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth
        
    return outLayer           

def create_rect_poly_from_ctr(outLayer, featureDefn, centers, gridHeight, gridWidth, data=None):    
    # Create ring 
    for (i,coord) in enumerate(centers):
        x_c,y_c = coord
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_c-0.5*gridWidth, y_c+0.5*gridHeight)
        ring.AddPoint(x_c+0.5*gridWidth, y_c+0.5*gridHeight)
        ring.AddPoint(x_c+0.5*gridWidth, y_c-0.5*gridHeight)
        ring.AddPoint(x_c-0.5*gridWidth, y_c-0.5*gridHeight)
        ring.AddPoint(x_c-0.5*gridWidth, y_c+0.5*gridHeight)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # add new geom to layer
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(poly)
#        # Field needs to be created with the data type defined. This functionality is not available for now
#        if data is not None:
#            # 'data' is a list of dicts of which length equal to the number of grid cells 
#            # In the list each dict element has keys as field names and values and corresponding values
#            for k,v in data[i].items():
#                outLayer.CreateField(ogr.FieldDefn(k, ogr.OFTInteger))
#                outFeature.SetField(k, v)
        outLayer.CreateFeature(outFeature)
        outFeature = None       
    return outLayer
    
        
def create_poly_shapefile(outputGridfn, poly_fun, *args, **kwargs):
    """
    create a polygon shapefile using the function poly_fun    
    """
    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    
    outLayer = poly_fun(outLayer, featureDefn, *args, **kwargs)

    # Save and close DataSources
    outDataSource = None  

    
if __name__ == "__main__": 
    import geopandas as gpd
    import numpy as np
    out_file1 = './shape/test1.shp'
    create_poly_shapefile(out_file1, create_regular_grid_poly, xmin=100, xmax=200, ymin=50, ymax=80, gridHeight=5, gridWidth=10)  
    shpfile = gpd.read_file(out_file1)

    # add fields
    shpfile2 = shpfile
    shpfile2['new_field']=np.arange(0,60,1)
    proj_str = "+proj=tmerc +lat_0=36.66666666666666 +lon_0=-88.33333333333333 +k=0.9999749999999999 " + \
              "+x_0=300000 +y_0=0 +datum=NAD83 +units=us-ft +no_defs +ellps=GRS80 +towgs84=0,0,0"
    shpfile2.crs = proj_str
    out_file2 = './shape/test2.shp'
    shpfile2.to_file(driver='ESRI Shapefile',filename=out_file2)
    
    out_file3 = './shape/test3.shp'
    x_ctr, y_ctr = np.arange(10,20,1), np.arange(30,20,-1)
    centers = np.c_[x_ctr,y_ctr]
    create_poly_shapefile(out_file3, create_rect_poly_from_ctr, centers, gridHeight=1, gridWidth=0.5)  
    shpfile3 = gpd.read_file(out_file3)
    shpfile3.crs = proj_str
    shpfile3.to_file(driver='ESRI Shapefile',filename=out_file3)