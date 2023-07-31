#author: alex sun
#date: 09042022
#purpose: compare csr5d to glofas
#====================================================================
import pandas as pd
import pickle as pkl
import numpy as np
import glob
import os,sys
import xarray as xr
from datetime import date

import matplotlib.pyplot as plt

rootDir = '/home/suna/work/grace/data'    

#currently this is for conus only
def getExtents():
    extents={
    "min_lat": 24,
    "max_lat": 50,
    "min_lon": -125,
    "max_lon": -66,
    } 
    return extents

def loadGLOFAS4CONUS(root =None, reLoad=False, convertToNC=False):
    """Subset GLOFAS daily files for CONUS
    Param
    ------
    rootDir, root directory of all glofas netcdf files

    Returns
    -------
    ds, combined dataset from 1980 to 2020
    """
    if root is None:
        root = rootDir

    #hard code the variable name for river discharge
    varname = 'dis24'
    
    extents = getExtents()
    outncfile = os.path.join(root, 'glofas_conus.nc')
    if reLoad:
        if convertToNC:
            for iyear in range(1980,2021):
                gribfile = os.path.join(root, 'glofas/river_discharge{0:4d}.grib'.format(iyear))
                ds = xr.open_dataset(gribfile, engine='cfgrib')
                #save as nc file
                ds.to_netcdf(os.path.join(root,'glofas/river_discharge{0:4d}.nc'.format(iyear)))

        bigDS=[]        
        for iyear in range(2000,2021):
            ncfile = os.path.join(root, 'glofas/river_discharge{0:4d}.grib'.format(iyear))
            print ('processing ', ncfile)

            ds = xr.open_dataset(ncfile, engine='cfgrib')
            da = ds[varname]
            #make sure the lat and lon are stored in asending order
            da = da.sortby(da.latitude)
            #subsetting 
            da = da.sel(latitude=slice(extents['min_lat'],extents['max_lat']),
                        longitude=slice(extents['min_lon'],extents['max_lon']))        
            bigDS.append(da)
            da = None

        combined = xr.concat(bigDS, dim='time')
        combined.to_netcdf(outncfile)            
    #load up the saved netcdf file    
    ds = xr.open_dataset(outncfile)
    return ds

def extractBasinOutletQ(loc, ds=None, startDate='2002-01-01', endDate='2020-12-31'):
    """Load GloFAS output for given location
    Param
    -----
    loc, (lat, lon) tuple
    """
    lat,lon = loc
    if ds is None:
        ds = loadGLOFAS4CONUS(rootDir, reLoad=False)

    da = ds['dis24']    
    if 'longitude' in da.dims:
        da = da.rename({'longitude':'lon', 'latitude':'lat'})
    
    #Extract Q 
    lat_toler = 0.02    
    lon_toler = 0.02
    daQ = da.sel(lon=slice(lon-lon_toler,lon+lon_toler), lat=slice(lat-lat_toler, lat+lat_toler))
    while len(daQ['lat'])==0: 
        print ('not finding cell, increasing lat toler')
        lat_toler+=0.01
        daQ = da.sel(lon=slice(lon-lon_toler,lon+lon_toler), lat=slice(lat-lat_toler, lat+lat_toler))        
    while len(daQ['lon'])==0:
        print ('not finding cell, increasing lon toler')
        lon_toler+=0.01
        daQ = da.sel(lon=slice(lon-lon_toler,lon+lon_toler), lat=slice(lat-lat_toler, lat+lat_toler))        
    daQ = daQ.sel(time=slice(startDate,endDate))    

    return daQ


