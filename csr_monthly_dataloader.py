#author: alex sun
#purpose: prepare csr monthly data
#date: 07312023, revisit for monthly data interpolation
#===============================================================
from typing import Tuple
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os, glob, sys
import pandas as pd
import time
from datetime import datetime
import xscale.signal.fitting as xfit
from myutils import getGraceRoot,getRegionExtent
from dataloader_global import loadClimatedatasets

def loadMask():
    """Load CSR land mask
    """
    maskfile = os.path.join(getGraceRoot(), 'data/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
    with xr.open_dataset(maskfile) as ds:
        mask = ds.to_array().squeeze() 
        #shift from 0, 360 to -179,180
        mask.coords['lon'] = (mask.coords['lon'] + 180) % 360 - 180
        mask = mask.sortby(mask.lon)
    return mask

def filterData(da, removeSeason=True, removeTrend=True):
    """Filter CSR data. For flow, this should filter out mean+trend+seasonal

    da, unfiltered DataArray
    removeSeason, if True, remove seasonal and return interannual, seasonal, trend; 
                  otherwise, only return trend, and detrended
    Returns
    -------
    interannual, seasonal, trend: interannual, seasonal, and linear trend
    """
    #Detrend
    da.coords['time'] = da.coords['time'].data.astype(np.int64)
    da = da.chunk('auto')
    if removeTrend:
        trend = xfit.trend(da,dim='time',type='linear')
        da = da-trend

    #remove seasonal    
    if removeSeason:
        modes = xfit.sinfit(da, dim='time', periods=[182.625,365.25])
        #modes = xfit.sinfit(da, dim='time', periods=[365.25])
        #modes = modes.chunk(chunks={'lat': 90, 'lon': 90, 'time': 1})
        seasonal = xfit.sinval(modes=modes, coord=da.time)
        interannual = da - seasonal 
        return interannual,seasonal,trend
    else:
        return da, trend

def getDataSet(dsfile):
    """Import CSR monthly dataset and do subsetting
    Params
    ------
    dsfile, name of the dataset file
    
    Returns
    ------
    da, dataarray holding tws
    csr_dates, days elapsed since 2002/1/1 in actual date format
    """
    ds = xr.open_dataset(dsfile, chunks='auto')
    da= ds['lwe_thickness']
    #shift from 0, 360 to -179,180
    da.coords['lon'] = (da.coords['lon'] + 180) % 360 - 180
    da = da.sortby(da.lon)

    #CSR da['time'] has days since 2002-01-01
    #form the actual grace dates
    csr_dates = pd.TimedeltaIndex(da['time'].values,unit='day')+datetime.strptime('2002-01-01', '%Y-%m-%d')

    return da,csr_dates

def getTWSDataArrays(region,reLoad=False,maskOcean=True, deseason=False, deTrend=True):    
    """Load TWS data array for a specific region
       Params:
       ------
       region: rect extent of the region to be subset [for global this should be ignored]
       reLoad, true to reload the dataarray
       maskOcean, mask the ocean out

       Returns:
       -------
       daInterannual, tws dataarray for the input region
       mask, mask
    """
    mask = loadMask()
    if maskOcean:
        #output netcdf name for land
        ncpath = os.path.join(getGraceRoot(), 'data/globalcsrmonthly_interannual.nc')
    else:
        #land + ocean
        ncpath = os.path.join(getGraceRoot(), 'data/globalcsrmonthly_interannual_all.nc')

    if reLoad:
        print ('reloading CSR monthly data...')
        datadir=os.path.join(getGraceRoot(), 'data/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc')
                                                   
        daCSR,csr_dates =getDataSet(datadir)        
        if deseason:
            daInterannual,_,_ = filterData(daCSR, removeSeason=deseason, removeTrend=deTrend)
        else:
            daInterannual,_ = filterData(daCSR, removeSeason=deseason, removeTrend=deTrend)

        bigarr = daInterannual.values
        if maskOcean:
            #zero out non-land pixels
            bigarr = np.einsum('kij,ij->kij',bigarr,mask)
        #make new da
        daInterannual = xr.DataArray(bigarr,name="lwe_thickness",coords=daInterannual.coords,dims=daInterannual.dims)
        #switch back to CSR dates
        daInterannual.coords['time'] =  csr_dates
        print ('writing to netcdf...')
        daInterannual.to_netcdf(ncpath)
    else:            
        daInterannual = xr.open_dataset(ncpath)['lwe_thickness']
    
    lat0,lon0,lat1,lon1 = region
    daInterannual = daInterannual.sel(lon = slice(lon0,lon1), lat= slice(lat0, lat1), time=slice('2002/04/01', '2022/06/30'))
    mask = mask.sel(lon = slice(lon0,lon1), lat= slice(lat0, lat1))    
    
    return daInterannual,mask

def convertMonthlyTo5d(cfg):
    """
    Take CSR monthly and interpolate into 5-day intervals
    """
    region = getRegionExtent(regionName='global')
    #
    da, mask = getTWSDataArrays(region=region, reLoad=cfg.monthly.reload, maskOcean=True, deseason=cfg.monthly.remove_season)
    print ('here', da.time)
    #do interpolation    
    da_5d = da.resample(time="5D").interpolate("linear")
    #slice the data
    da_5d = da_5d.sel(time=slice(cfg.data.start_date, cfg.data.end_date))
    return da_5d,mask

def loadFake5ddatasets(cfg, region:Tuple, coarsen:bool =True, mask_ocean=True, startYear=2002, endYear=2020):    
    """Load CSR5d data
    Params
    ------
    coarsen, true to coarsen the grid (currently default to 1x1)
    crs5d_version, version of csr5d data
    """    
    print ('before...')
    daFakeCSR5d, mask = convertMonthlyTo5d(cfg)
    if coarsen:
        print ('coarsening to 1 degree from 0.25 degree !!!!')
        #landmask = mask.coarsen(lat=4,lon=4,boundary='exact').mean()
        landmask = xr.open_dataset(os.path.join(getGraceRoot(), 'data/mylandmask.nc'))['LSM'].squeeze()
        daFakeCSR5d = daFakeCSR5d.coarsen(lat=4,lon=4,boundary='exact').mean()
    else:
        landmask = xr.open_dataset(os.path.join(getGraceRoot(), 'data/mylandmask025deg.nc'))['LSM'].squeeze()
        
    #04042022, for global using all cells
    if mask_ocean:
        mask = landmask
    else:    
        #use all cells    
        arr = np.zeros(landmask.values.shape)+1
        mask = xr.DataArray(arr, name='mask', dims= landmask.dims, coords=landmask.coords )
    print (daFakeCSR5d.shape, mask.shape)
    return daFakeCSR5d,mask,landmask

def getTWSDataArraysRaw(region,reLoad=False,maskOcean=True, deseason=False):    
    """Load TWS data array for a specific region w/o any filtering
       Params:
       ------
       region: rect extent of the region to be subset [for global this should be ignored]
       reLoad, true to reload the dataarray
       maskOcean, mask the ocean out

       Returns:
       -------
       da, tws dataarray for the input region
    """
    mask = loadMask()

    print ('reloading CSR monthly data...')
    datadir=os.path.join(getGraceRoot(), 'data/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc')
                                                
    daCSR,csr_dates =getDataSet(datadir)        

    bigarr = daCSR.values
    if maskOcean:
        #zero out non-land pixels
        bigarr = np.einsum('kij,ij->kij',bigarr,mask)
    #make new da
    da = xr.DataArray(bigarr,name="lwe_thickness",coords=daCSR.coords,dims=daCSR.dims)
    
    return da, csr_dates

def doRemoveSWE(daCSR, region, reLoad=False, startYear=2002, endYear=2020):
    daSWE = loadClimatedatasets(region, vartype='swe', 
                        daCSR5d= daCSR, 
                        aggregate5d=True, 
                        reLoad=reLoad, 
                        precipSource='era5',
                        startYear= startYear,
                        endYear= endYear,
                        fake5d=True
                        )
    #remove mean to form anomalies
    daSWE_mean = daSWE.sel(time=slice("2004/01/01", "2009/12/31")).mean(dim="time", skipna=True)
    daSWE = daSWE -daSWE_mean
    #Note: need to replace nan in swe, otherwise, detrend would not work
    daSWE = daSWE.fillna(0)
    daCSR.values = daCSR.values - daSWE.values
    return daCSR

def loadFake5ddatasetsNoSWE(cfg,  region:Tuple, reLoad:bool = False, coarsen:bool =True, 
                            mask_ocean=True, startYear=2002, endYear=2020):    
    """Load CSR5d data with SWE removal [08/05/2023]
    cfg: config file
    region: coordinates of the region
    """
    from csr_monthly_dataloader import getTWSDataArraysRaw

    if reLoad:
        #time of daCSR5d0 is in days elapsed [data starts from 4/18/2002]
        #monthly raw data
        daCSR0, oldtimes = getTWSDataArraysRaw(region, maskOcean=mask_ocean)
        daCSR0['time'] = oldtimes #restore datetime
        if coarsen:
            #this is needed for the climate data masking
            print ('coarsening to 1 degree from 0.25 degree !!!!')
            landmask = xr.open_dataset(os.path.join(getGraceRoot(), 'data/mylandmask.nc'))['LSM'].squeeze()
            daCSR = daCSR0.coarsen(lat=4,lon=4,boundary='exact').mean()
        else:
            landmask = xr.open_dataset(os.path.join(getGraceRoot(), 'data/mylandmask025deg.nc'))['LSM'].squeeze()
            daCSR = daCSR0

        #04042022, for global using all cells
        if mask_ocean:
            mask = landmask
        else:    
            #use all cells    
            arr = np.zeros(landmask.values.shape)+1
            mask = xr.DataArray(arr, name='mask', dims= landmask.dims, coords=landmask.coords )

        #Convert to 1d through linear interpolation
        da_fake5d = daCSR.resample(time="1D").interpolate("linear")
        #asun10/30/2023, load csr5d data to get the exact dates
        daCSR5d = xr.open_dataset(os.path.join(getGraceRoot(), 'data/globalcsr5d_notrend_swe_v1.nc'))['lwe_thickness']
        csr5d_dates = daCSR5d['time']
        da_fake5d = da_fake5d.sel(time=csr5d_dates)

        #remove SWE
        oldtimes = da_fake5d.time.values #record the datetimes
        da_fake5d = doRemoveSWE(da_fake5d, region, reLoad=True, startYear=2002, endYear=2020)
        #remove linear trend
        #first convert dates to days elapsed since 2002/04/01
        days_elapsed = (pd.to_datetime(oldtimes) - datetime.strptime('2002-04-01', '%Y-%m-%d')).days
        da_fake5d['time'] = days_elapsed
        #now do detrend
        daInterannual,_   = filterData(da_fake5d, removeSeason=False)    
        #restore time to datetime 
        daInterannual['time'] = oldtimes
        #do masking            
        bigarr = daInterannual.values
        if mask_ocean:
            bigarr = np.einsum('ijk,jk->ijk', bigarr, mask)
        da = xr.DataArray(bigarr,name="lwe_thickness",coords=daInterannual.coords,dims=daInterannual.dims)
        #upsampling to 0.25x0.25 
        #[10/30/2023, comment this out when coarsen is false ]
        da = da.interp(coords={'lat':daCSR0.lat, 'lon':daCSR0.lon, 'time':da.time}, method='nearest')
        #slice time period
        da = da.sel(time=slice(f'{startYear}/01/01', f'{endYear}/12/31'))
        print ('monthly faked5d data final shape', da.shape)
        #land only
        if mask_ocean:
            da.to_netcdf(os.path.join(getGraceRoot(), f'data/globalcsrfake5d_notrend_swe.nc'))
    else:
        if mask_ocean:
            da = xr.open_dataset(os.path.join(getGraceRoot(), f'data/globalcsrfake5d_notrend_swe.nc'))['lwe_thickness']
    return da

if __name__ == '__main__':
    from myutils import load_config, getRegionExtent
    region = getRegionExtent(regionName='global')
    config = load_config('config.yaml')

    #convertMonthlyTo5d(config)
    loadFake5ddatasetsNoSWE(config, region=region, reLoad=True, coarsen=True, startYear=2002, endYear=2019)