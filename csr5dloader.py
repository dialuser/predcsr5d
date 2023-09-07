#author: Alex Sun
#date: 3/25/2022
#purpose: load csr5d dataset
#rev date: 07/15, reviewed and cleaned up for new effort
#rev date: 11/10, revise for new Himanshu data.
#==========================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os, glob
import pandas as pd
import time
import sys
from datetime import datetime,timedelta
import xscale.signal.fitting as xfit
from scipy import signal

import sklearn.preprocessing as skp
from myutils import from_numpy_to_var as toTensor
import torch
from torch.utils.data import TensorDataset
from myutils import getGraceRoot
from tqdm import tqdm
import rioxarray 
import geopandas as gpd

rootpath = '/home/suna/work/grace/data' #for lambda machine use

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

def createMyLandMask(removeGreenland=False):
    """ Create a mask raster that masks out the Greenland and Hudson Bay
    """
    xds = rioxarray.open_rasterio(os.path.join(getGraceRoot(), 'data/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc'))
    xds.rio.write_crs("epsg:4326", inplace=True)
    #make sure the lat/lon is correct
    xds.coords['x'] = (xds.coords['x'] + 180.0) % 360 - 180.0
    xds = xds.sortby('x')       
    xds = xds.sortby('y')       

    if removeGreenland:
        #greenland shp
        gdf = gpd.read_file(os.path.join("/home/suna/work/deepsphere-weather", "data/GRL_adm0.shp"))
        gdf = gdf.to_crs('EPSG:4326')
        #this has the same dimensions as xds
        greenland = xds.rio.clip(gdf.geometry, gdf.crs, drop=False, invert=False, all_touched = True)
    
    #hudson bay shp
    gdf = gpd.read_file(os.path.join(getGraceRoot(), "data/iho.shp"))
    gdf = gdf.to_crs('EPSG:4326')
    #this has the same dimensions as xds
    hudson = xds.rio.clip(gdf.geometry, gdf.crs, drop=False, invert=False,all_touched = True)

    newmask = np.zeros(xds.shape, dtype=int)
    newmask[xds.values>=1.0] = 1
    if removeGreenland:
        newmask[greenland.values<=1.1] = 0   
    newmask[hudson.values<=1.1] = 0

    da = xr.DataArray(newmask, name='LSM', dims= xds.dims, coords=xds.coords)
    da = da.rename({'x':'lon', 'y':'lat'})
    #save a 0.25-deg mask
    da.to_netcdf(os.path.join(getGraceRoot(), 'data/mylandmask025deg.nc'))
    fig,ax = plt.subplots(1,1)
    da.plot(ax=ax)
    plt.savefig('testlandmask025.png')
    plt.close()
    #resample to 1 deg
    da = da.coarsen(lat=4,lon=4,boundary='exact').mean()
    da.to_netcdf(os.path.join(getGraceRoot(), 'data/mylandmask.nc'))

    fig,ax = plt.subplots(1,1)
    da.plot(ax=ax)
    plt.savefig('testlandmask.png')
    plt.close()
    

def getDataSet(dsfile, startDate='2002-04-01', endDate='2022-06-30'):
    """Import CSR 5d dataset and do temporal subsetting
    Params
    ------
    dsfile, name of the dataset file    
    startDate, endDate, start and end dates of Himanshu dataset
    """
    ds = xr.open_dataset(dsfile, chunks='auto')
    ds = ds.rename({
            '__xarray_dataarray_variable__': 'lwe_thickness',
        })    
    da= ds['lwe_thickness']        
    da = da.sel(time=slice(startDate,endDate))    
    oldtimes = da['time']
    #convert dates to days elapsed since 2002/04/01
    csr5d_days = (pd.to_datetime(da['time'].values) - datetime.strptime(startDate, '%Y-%m-%d')).days
    
    da.coords['time']=csr5d_days
    #form a mapper between time axes
    timedict = dict(zip(oldtimes.data,csr5d_days.tolist()))
    #reorder to time, lat, lon
    da = da.transpose('time','lat','lon')
    return da,oldtimes,timedict

def genNetCDF(extents=None, version=1):        
        #see https://towardsdatascience.com/basic-data-structures-of-xarray-80bab8094efa
        """Combine Himanshu raw text files into a netCDF file
           Note: V1 data was generated on Lambda         
        """
        if version == 0:
            foldername = os.path.join(rootpath, 'CSR_5day_mascons/*.xyz')
        elif version == 1:
            foldername = os.path.join(rootpath, 'CSR_5day_mascons_V1/*.xyz')

        files =sorted(glob.glob(foldername))
        bigDS = []
        startime = time.time()
        nCol = 360*4
        nRow = 180*4
        counter=0
        
        for item in tqdm(files):
            df = pd.read_csv(item, header=None,sep=" ",engine="python")
            df.columns=['lon','lat','ewh']
            df = df.pivot(index="lat", columns="lon")
            #drop the first level
            df = df.droplevel(0, axis=1)
            da = xr.DataArray(data=df)
            #shift from 0, 360 to -179,180
            da.coords['lon'] = (da.coords['lon'] + 180) % 360 - 180
            da = da.sortby(da.lon)            
            #subsetting
            if not extents is None:
                da = da.sel(lat=slice(extents['min_lat'],extents['max_lat']),
                        lon=slice(extents['min_lon'],extents['max_lon']))
            df = None
            filebase = os.path.basename(item)
            dataDay = pd.to_datetime(filebase[:10],format='%Y-%m-%d')
            da = da.expand_dims(dim={'time':[dataDay]},axis=2)
            bigDS.append(da)
            da = None
        combined = xr.concat(bigDS, dim='time')
        combined.to_netcdf(os.path.join(rootpath, 'globalcsr5d_raw_v{0}.nc'.format(version))) #note this file is moved to grace/data
        print ('time elapsed ', time.time()-startime)

def filterData(da, oldtimes, removeSeason=True, removeTrend=True, filterMethod="XFIT", returnDetrended=False):
    """Filter CSR data
    @todo: fit trends on training data only

    da, unfiltered DataArray
    removeSeason, if True, remove seasonal and return interannual, seasonal, trend; 
                  otherwise, only return trend, and detrended
    Returns
    -------
    interannual, seasonal, trend: interannual, seasonal, and linear trend
    """
    da.coords['time'] = da.coords['time'].data.astype(np.int64)
    da = da.chunk('auto')

    if removeTrend:
        trend = xfit.trend(da,dim='time',type='linear')    
        da = da-trend

    #remove seasonal
    if removeSeason:
        if filterMethod == 'XFIT': 
            modes = xfit.sinfit(da, dim='time', periods=[182.625,365.25])
            seasonal = xfit.sinval(modes=modes, coord=da.time)
            interannual = da - seasonal 
            #====asun 01152023, remove monthly mean
            interannual.coords['time'] = oldtimes #need real dates
            interannual = interannual.groupby("time.month") - interannual.groupby("time.month").mean("time")
            if returnDetrended:
                return interannual,da, seasonal,trend
            else:
                return interannual,seasonal,trend
        elif filterMethod == "BANDPASS":
            #remove any seasonal signal above monthly
            #from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JC018302
            #e extract the ∼5-day signal in various fields (GRACE, Pa, DEBOT dynamic residuals) using a fifth-order 
            # Butterworth band-pass filter with cut-off frequencies placed at 0.175 and 0.233 cpd (cycles per day) 
            # corresponding to periods of 5.7 and 4.3 days,
            # dt = 5.0 => treat as normalized rate 1
                
            fs = 1
            nyquist = 0.5*fs
            low_cutoff = 1.0/12
            print('cutoff= ', 1/low_cutoff*nyquist*5,' days')                    
            lo_band = low_cutoff/nyquist
            b, a = signal.butter(5, lo_band, btype='highpass', fs=fs)
            #assume time axis is the first axis
            arr = signal.filtfilt(b, a, arr, axis=0)
            if returnDetrended:
                return interannual, da, None, trend
            else:
                return interannual,seasonal,trend
        else:
            raise ValueError("filter method not supported")
    else:
        return da, trend

def getTWSDataArrays(region, reLoad=False, maskOcean=True, filtering=True, removeSeason=False, 
                     csr5d_version=1, useImputed=False, filterMethod='XFIT'):    
    """load mask"""
    #as0715: this should work for both global and land-only applications
    #        in the former case, mask is 1 everywhere
    mask = loadMask()
    
    if reLoad:     
        if not useImputed:
            print ("loading original CSR data")
            #set up nc file names
            rawncfile = os.path.join(getGraceRoot(),  f'data/globalcsr5d_raw_v{csr5d_version}.nc')
            #load data with basic cleanup
            daCSR,oldtimes,timedict = getDataSet(rawncfile)
        else:
            print ("loading imputed CSR data")
            #set up nc file names
            rawncfile = os.path.join(getGraceRoot(),  f'data/globalcsr5d_imputed_v{csr5d_version}.nc')
            #load imputed data (time is already in days)
            daCSR = xr.open_dataset(rawncfile)['tws']
            oldtimes = daCSR['time']
            #convert dates to days elapsed since 2002/04/01
            csr5d_days = (pd.to_datetime(daCSR['time'].values) - datetime.strptime('2002-04-01', '%Y-%m-%d')).days
            daCSR.coords['time']=csr5d_days
        
        if filtering:
            print ('Do filtering....')
            if removeSeason:
                print ('remove seasonal')
                daInterannual,_,_ = filterData(daCSR, oldtimes, removeSeason=removeSeason, filterMethod=filterMethod)            
            else:
                daInterannual,_   = filterData(daCSR, oldtimes, removeSeason=removeSeason)            

            bigarr = daInterannual.values
        else:
            print ('Bypass filtering')
            bigarr = daCSR.values

        if maskOcean:
            bigarr = np.einsum('ijk,jk->ijk', bigarr, mask)
            
        da = xr.DataArray(bigarr,name="lwe_thickness",coords=daCSR.coords,dims=daCSR.dims)
        da.coords['time'] = oldtimes
        if maskOcean:
            if removeSeason:
                da.to_netcdf(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_deseason_v{csr5d_version}.nc'))
            else:
                da.to_netcdf(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_v{csr5d_version}.nc'))
        else:
            if removeSeason:
                da.to_netcdf(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_all_deseason_v{csr5d_version}.nc'))
            else:
                da.to_netcdf(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_all_v{csr5d_version}.nc'))
    
    if maskOcean:
        #land only
        if removeSeason:
            da = xr.open_dataset(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_deseason_v{csr5d_version}.nc'))['lwe_thickness']
        else:
            da = xr.open_dataset(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_v{csr5d_version}.nc'))['lwe_thickness']
    else:
        #land and ocean
        if removeSeason:
            da = xr.open_dataset(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_all_deseason_v{csr5d_version}.nc'))['lwe_thickness']
        else:
            da = xr.open_dataset(os.path.join(getGraceRoot(), f'data/globalcsr5d_notrend_all_v{csr5d_version}.nc'))['lwe_thickness']
    
    lat0,lon0,lat1,lon1 = region    
    da = da.sel(lon = slice(lon0,lon1), lat= slice(lat0, lat1))    
    mask = mask.sel(lon = slice(lon0,lon1), lat= slice(lat0, lat1))

    return da,mask

def formDataSets(da,mask,toOneDegree=False, genMode='random'):
    """form datasets (not used in this project????)
    Params
    ------
    da, dataarray of interannual
    mask, global land mask
    """
    assert(da.shape[1:]==mask.shape)
    if toOneDegree:
        #downsample to 1x1 mask
        maskcoarse = mask.coarsen(lat=4,lon=4,boundary='exact').mean()
    
    mask = mask.values
    #zero out non-land pixels
    bigarr = da.values
    bigarr = np.einsum('kij,ij->kij',bigarr,mask)
    
    da = xr.DataArray(bigarr,coords=da.coords,dims=da.dims)
    if toOneDegree:
        #downsample to 1x1 degree    
        da = da.coarsen(lat=4,lon=4,boundary='exact').mean()
    
    if genMode=='random':
        bigarr = da.values
        #split training/testing       
        nData = bigarr.shape[0]
        nTrain = int(0.9*nData)
        nTest = nData - nTrain
        randarr = np.random.permutation(range(nData))
        trainarr = randarr[:nTrain]
        testarr = randarr[nTrain:]
        Xtrain = bigarr[trainarr,:,:]
        Xtest = bigarr[testarr,:,:]

    #
    #rescaling
    #
    ny,nx=Xtrain.shape[1:]
    Xtrain = Xtrain.reshape(Xtrain.shape[0],ny*nx)
    Xtest = Xtest.reshape(Xtest.shape[0],ny*nx)
    XMIN,XMAX=(0.0,1.0)
    scaler = skp.MinMaxScaler(feature_range=(XMIN, XMAX),copy=True)

    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    Xtrain  = Xtrain.reshape(Xtrain.shape[0],ny,nx)
    Xtest = Xtest.reshape(Xtest.shape[0],ny,nx)
    #*** form train dataset
    Xin = toTensor(Xtrain,dtype=torch.float32)
    Xin = torch.unsqueeze(Xin, dim=1)
    print ('Xtrain shape', Xin.shape)
    #not used label
    yin = toTensor(trainarr, dtype=torch.int32)
    trainDataset = TensorDataset(Xin,yin)
    #*** form test dataset
    Xin = toTensor(Xtest,dtype=torch.float32)
    Xin = torch.unsqueeze(Xin, dim=1)
    print ('Xtest shape', Xin.shape)
    #label, not used
    yin = toTensor(testarr, dtype=torch.int32)
    testDataset = TensorDataset(Xin,yin)
    #all dataset
    bigarr = bigarr.reshape(bigarr.shape[0],ny*nx)
    bigarr = scaler.transform(bigarr)
    bigarr = bigarr.reshape(bigarr.shape[0],ny,nx)
    Xin = toTensor(bigarr,dtype=torch.float32)
    Xin = torch.unsqueeze(Xin, dim=1)
    print ('Xall shape', Xin.shape)
    yin = toTensor(np.arange(bigarr.shape[0]), dtype=torch.int32)
    allDataset = TensorDataset(Xin,yin)
    if toOneDegree:
        return trainDataset,testDataset,allDataset,scaler,maskcoarse
    else:
        return trainDataset,testDataset,allDataset,scaler

def loadTWS_raw(region, maskOcean=True, csr5d_version=1):    
    """load mask"""
    #as0715: this should work for both global and land-only applications
    #        in the former case, mask is 1 everywhere
    mask = loadMask()
    
    print ("loading original CSR data")
    #set up nc file names
    rawncfile = os.path.join(getGraceRoot(),  f'data/globalcsr5d_raw_v{csr5d_version}.nc')
    #load data with basic cleanup
    daCSR,oldtimes,timedict = getDataSet(rawncfile)
    
    print ('Bypass filtering')
    bigarr = daCSR.values

    if maskOcean:
        bigarr = np.einsum('ijk,jk->ijk', bigarr, mask)

    da = xr.DataArray(bigarr,name="lwe_thickness",coords=daCSR.coords,dims=daCSR.dims)
    lat0,lon0,lat1,lon1 = region    
    da = da.sel(lon = slice(lon0,lon1), lat= slice(lat0, lat1))    
    mask = mask.sel(lon = slice(lon0,lon1), lat= slice(lat0, lat1))

    return da,oldtimes
    
def main():
    #genNetCDF(version=1)
    #createMyLandMask()
    from myutils import getRegionExtent
    region = getRegionExtent(regionName="global")
    getTWSDataArray_SWE(region=region, maskOcean=True, coarsen=True)

    return
    #llcrnrlon=-108.985-1.0; llcrnrlat=24.498131-1.0    
    llcrnrlon =-109; llcrnrlat=24 #the bound of downloaded ERA5 data is 24 to 50
    blocksize = 64
    cellsize  = 0.25
    lon0 = llcrnrlon; lon1 = lon0+(blocksize)*cellsize
    lat0 = llcrnrlat; lat1 = lat0+(blocksize)*cellsize
    region = (lat0,lon0,lat1,lon1)
    daTWS, mask = getTWSDataArrays(region, reLoad=False)
    seed = 3252022
    np.random.seed(seed)
    print (daTWS.coords)
    fig,axes = plt.subplots(2,2)
    axs = (axes[0,0], axes[0,1],axes[1,0],axes[1,1])
    for item, ax in zip([0, 10, 25, 39, 51], axs):
        daTWS.isel(time=item).plot(ax=ax)
    plt.savefig('testbasintws.png')
    plt.close()
    print (daTWS.shape)
if __name__ == '__main__':
    main()