#author: Alex Sun
#date: 03262022
#purpose: load 5d data
#date: 04042022
#      CSR5d has unit in cm for global
#date: 04102022, add GPM and ERA5 water vapor fluxes
#date: 07152022, revise for new effort
#date: 08042022, add a new datagenerator class
#date: 09020222, revisit and add documentation
#      check code quality, check accuracy
#date: 06192023, switch to version 1
###################################################################################################
import abc
from typing import Tuple
import numpy as np
from sklearn.utils import shuffle
import xarray as xr
import matplotlib.pyplot as plt
import os, glob,sys
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import datetime 
from dateutil.relativedelta import relativedelta
import random 
import geopandas as gpd
import rioxarray 

#custom package
from csr5dloader import getTWSDataArrays, filterData
from climate5dloader import getPrecipData, getWaterVaporData,getMERRA2SoilMoistureData,getGLEAMData,getAirTempData,getSSTData
from myutils import scaleDA,reScale,toTensor,getGraceRoot,getGridCellArea

def detrendDA(da):
    """detrend da
    """
    #convert dates to time elapsed
    startDate = '2002-04-01'
    csr5d_days = (pd.to_datetime(da['time'].values) - datetime.datetime.strptime(startDate, '%Y-%m-%d')).days
    
    #save old dates
    oldtimes = da.coords['time']
    da.coords['time']=csr5d_days

    da, datrend = filterData(da, removeSeason=False)
    da.coords['time'] = oldtimes
    return da

def load5ddatasets(region:Tuple, coarsen:bool =True, mask_ocean=True, reLoad=False, 
                   removeSeason=False, csr5d_version=1,
                   startYear=2002,endYear=2020):    
    """Load CSR5d data
    Params
    ------
    coarsen, true to coarsen the grid (currently default to 1x1)
    crs5d_version, version of csr5d data
    """    
    daCSR5d, mask = getTWSDataArrays(region, 
                    reLoad=reLoad, 
                    maskOcean=mask_ocean, 
                    removeSeason=removeSeason, 
                    csr5d_version=csr5d_version) #this has already been detrended in csr5dloader.py
    daCSR5d = daCSR5d.sel(time=slice(f'{startYear}/01/01', f'{endYear}/12/31'))
    if coarsen:
        print ('coarsening to 1 degree from 0.25 degree !!!!')
        #landmask = mask.coarsen(lat=4,lon=4,boundary='exact').mean()
        landmask = xr.open_dataset(os.path.join(getGraceRoot(), 'data/mylandmask.nc'))['LSM'].squeeze()
        daCSR5d = daCSR5d.coarsen(lat=4,lon=4,boundary='exact').mean()
    else:
        landmask = xr.open_dataset(os.path.join(getGraceRoot(), 'data/mylandmask025deg.nc'))['LSM'].squeeze()
        
    #04042022, for global using all cells
    if mask_ocean:
        mask = landmask
    else:    
        #use all cells    
        arr = np.zeros(landmask.values.shape)+1
        mask = xr.DataArray(arr, name='mask', dims= landmask.dims, coords=landmask.coords )
    print (daCSR5d.shape, mask.shape)
    return daCSR5d,mask,landmask

def getCSR5dLikeArray(da, daCSR5d, **kwargs):
    """ Make arrays that have the same length as the CSR5d
    """
    #get CSR5d dates
    timeaxis = daCSR5d['time'].values
    kwargs.setdefault("method", "rx5d")
    kwargs.setdefault("aggmethod", 'sum')
    kwargs.setdefault("name", 'variable')

    method = kwargs['method']
    if method=='rx5d':
        #maximum consequtive 5-day
        #get the 5-day total amount for precip, centered at each CSR5d date
        if kwargs['aggmethod'] == 'sum':
            daR5d = da.rolling(time=5, center=True,min_periods=1).sum()
        #get the rolling mean on daily data (this is suitable for state types), centered at each CSR5d date
        elif kwargs['aggmethod'] == 'average':
            daR5d = da.rolling(time=5, center=True,min_periods=1).mean()        
        #do not do anything
        elif kwargs['aggmethod'] == 'max':
            daR5d = da
        else:
            raise NotImplementedError()            
        halfwin = 2 #[day]
        #form CSR5d like time series
        bigarr = []
        for aday in timeaxis:
            aday = pd.to_datetime(aday,format='%Y-%m-%d').date()            
            #for a 5-day window centered at aday
            offsetDateStart = datetime.date.strftime(aday - relativedelta(days=halfwin),'%Y-%m-%d')
            offsetDateEnd   = datetime.date.strftime(aday + relativedelta(days=halfwin), '%Y-%m-%d')                        
            tmpda = daR5d.sel(time = slice(offsetDateStart,offsetDateEnd))
            if not tmpda.time.size==0:
                #take the max of the 5-day statistic (for mean, sum, and max)
                tmpda = tmpda.max(dim='time',skipna=True)
                bigarr.append(tmpda.values)
            else:
                bigarr.append(np.zeros((daR5d.shape[1:3]))+np.NaN)

        bigarr = np.stack(bigarr,axis=0)

        daNew = xr.DataArray(bigarr, dims=daCSR5d.dims, coords=daCSR5d.coords, name=kwargs['name'])
    else:
        raise Exception("method not implemented")
    return daNew

def loadClimatedatasets(region:Tuple, vartype:str, daCSR5d, aggregate5d=False, reLoad:bool=True,
                     precipSource='ERA5',saving=True,startYear=2002, endYear=2020):
    """
    #get precip data and aggregate according to CSR5d time intervals
    #!!! don't use the simple resample, use getCSR5dLikeArray() instead
    asun06192023: add extra parameters startYear and endYear
    Params
    ------
    reLoad: if reLoad is True, re-generate the csr5d like data arrays; else, load from disk
    """
    if vartype == 'precip':
        assert (precipSource in ['GPM', 'ERA5', 'REGEN','GPCP'])
        dirmap = {'GPM':'gpm', 'ERA5': 'era5', 'REGEN': 'regen', 'GPCP':'gpcp'}
        if reLoad: 
            #regenerate 5-day data, for precipitation we want the accumulative 5 day value
            daPrecip = getPrecipData(region, source=precipSource, aggregateTo5d=False, 
                        startYear=startYear, endYear=endYear)
            if aggregate5d:
                kwargs = {"method": "rx5d", "aggmethod": 'sum', 'name':'tp'}       
                daPrecip = getCSR5dLikeArray(daPrecip, daCSR5d, **kwargs)
                if saving:
                    daPrecip.to_netcdf(os.path.join(getGraceRoot(), f'data/{dirmap[precipSource]}/{dirmap[precipSource]}_P5d.nc'))
            else:                                
                daPrecip['time'] =daPrecip.indexes['time'].to_datetimeindex()
                if saving:
                    daPrecip.to_netcdf(os.path.join(getGraceRoot(), f'data/{dirmap[precipSource]}/{dirmap[precipSource]}_P1d.nc'))
        else:
            if aggregate5d:
                daPrecip = xr.open_dataset(os.path.join(getGraceRoot(), f'data/{dirmap[precipSource]}/{dirmap[precipSource]}_P5d.nc'))
            else:
                daPrecip = xr.open_dataset(os.path.join(getGraceRoot(), f'data/{dirmap[precipSource]}/{dirmap[precipSource]}_P1d.nc'))
                daPrecip = daPrecip.rename({'precipitationCal':'tp'})
            #need to convert to dataarray
            daPrecip = daPrecip['tp']
        return daPrecip        
    elif vartype == 'airtemp':
        if reLoad:
            daT =  getAirTempData(region, aggregateTo5d= False)

            if aggregate5d:
                #for air temperature we want 5-day average values
                kwargs = {"method": "rx5d", "aggmethod": 'average', 'name': 't2m'}
                daT = getCSR5dLikeArray(daT, daCSR5d, **kwargs)
                if saving:
                    daT.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/t2m_5d.nc'))
            else:
                if saving:
                    daT.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/t2m_1d.nc'))                
        else:
            if aggregate5d:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/t2m_5d.nc'))
            else:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/t2m_1d.nc'))
            daT = ds['t2m']
        return daT  
    elif vartype == 'sst':
        if reLoad:
            daSST =  getSSTData(region, aggregateTo5d= False)

            if aggregate5d:
                #for air temperature we want 5-day average values
                kwargs = {"method": "rx5d", "aggmethod": 'average', 'name': 't2m'}
                daSST = getCSR5dLikeArray(daSST, daCSR5d, **kwargs)
                if saving:
                    daSST.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/sst_5d.nc'))
            else:
                if saving:
                    daSST.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/sst_1d.nc'))                
        else:
            if aggregate5d:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/sst_5d.nc'))
            else:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/sst_1d.nc'))
            daSST = ds['sst']
        return daSST  

    elif vartype == 'watervapor':
        if reLoad:
            #for water vapor, we want 5-day average values
            kwargs = {"method": "rx5d", "aggmethod": 'average'}
            daU, daV = getWaterVaporData(region)
            #calculate magnitudes on daily data
            mag = np.sqrt(daU.values*daU.values + daV.values*daV.values)
            daM = xr.DataArray(mag, coords=daU.coords, dims=daU.dims, name='mag')
            if aggregate5d:
                kwargs['name'] = 'U'
                daU = getCSR5dLikeArray(daU, daCSR5d, **kwargs)
                kwargs['name'] = 'V'
                daV = getCSR5dLikeArray(daV, daCSR5d, **kwargs)
                kwargs['name'] = 'mag'
                daM = getCSR5dLikeArray(daM, daCSR5d, **kwargs)
                daU.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_u5d.nc'))
                daV.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_v5d.nc'))
                daM.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_m5d.nc'))                
            else:
                daM.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_m1d.nc'))
                daU.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_u1d.nc'))
                daV.to_netcdf(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_v1d.nc'))        
        else:
            if aggregate5d:
                daU = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_u5d.nc'))
                daV = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_v5d.nc'))
                daM = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_m5d.nc'))
            else:
                daU = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_u1d.nc'))
                daV = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_v1d.nc'))
                daM = xr.open_dataset(os.path.join(getGraceRoot(), 'data/era5/watervapor/wv_m1d.nc'))
                daU = daU.rename({'p88.162':'U'})
                daV = daV.rename({'p89.162':'V'})
            #convert to data array
            daU = daU['U']; daV=daV['V']; daM = daM['mag']
        return daU, daV, daM
    elif vartype == 'sm':
        if reLoad:
            daSM = getMERRA2SoilMoistureData(region) #the unit is [m3/m3]    

            if aggregate5d:
                #for soil moisture we want 5-day average values
                kwargs = {"method": "rx5d", "aggmethod": 'average'}
                kwargs['name'] = 'RZMC'        
                daSM = getCSR5dLikeArray(daSM, daCSR5d, **kwargs)
                daSM.to_netcdf(os.path.join(getGraceRoot(), 'data/merra2/merr2sm_5d.nc'))
            else:
                daSM.to_netcdf(os.path.join(getGraceRoot(), 'data/merra2/merr2sm_1d.nc'))
        else:
            if aggregate5d:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/merra2/merr2sm_5d.nc'))
            else:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/merra2/merr2sm_1d.nc'))
            daSM = ds['RZMC']
        return daSM
    elif vartype == 'gleam':
        if reLoad:
            daGleam = getGLEAMData(region) #the unit is [m3/m3]    

            if aggregate5d:
                #for ET we want 5-day accumulative values
                kwargs = {"method": "rx5d", "aggmethod": 'average'}
                kwargs['name'] = 'E'        
                daGleam = getCSR5dLikeArray(daGleam, daCSR5d, **kwargs)
                daGleam.to_netcdf(os.path.join(getGraceRoot(), 'data/gleam_5d.nc'))
            else:
                daGleam.to_netcdf(os.path.join(getGraceRoot(), 'data/gleam_1d.nc'))
        else:
            if aggregate5d:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/gleam_5d.nc'))
            else:
                ds = xr.open_dataset(os.path.join(getGraceRoot(), 'data/gleam_1d.nc'))
            daGleam = ds.E
        return daGleam
        
def selectConsecData(da, lookbackLen=4, lookforwardLen=1):
    #find data with consequtive dates
    dT = 5 #days for csr5d
    prevDate = None    
    goodInd = []
    allDates = da.coords['time'].values
    for i in range(lookbackLen, len(allDates)-lookbackLen):
        currDate = pd.to_datetime(allDates[i],format='%Y-%m-%d').date()            
        prevDate = pd.to_datetime(allDates[i-lookbackLen],format='%Y-%m-%d').date()            
        futuDate = pd.to_datetime(allDates[i+lookforwardLen],format='%Y-%m-%d').date()            
        deltaBT = (currDate - prevDate).days
        deltaFT = (futuDate - currDate).days
        if deltaBT == lookbackLen*dT and deltaFT == lookforwardLen*dT:
            goodInd.append(i)
    print ('number of good dates', len(goodInd))
    return goodInd

class TWSDataNorm():
    def __init__(self, region, seq_len=4, target_seq_len=1,                 
                 rescale_method='norm', trainvalRatio=(0.7,0.15), mask_ocean=False):
        """
        seq_len, lookback period
        target_seq_len, forecast period
        mask_ocean, True to remove ocean area
        rescale_method, method for normalization
        trainvalRatio, ratio between train and validation
        """
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len

        #load TWS data
        daCSR5d,mask,landmask = load5ddatasets(region, coarsen=True, mask_ocean=mask_ocean, reLoad=False, removeSeason=False)
        self.weights = getGridCellArea(daCSR5d,weighttype='latbased')
        
        #Find indices satisfy the consec requirement
        goodInd = selectConsecData(daCSR5d, lookbackLen=seq_len, lookforwardLen=target_seq_len)
        
        #Split the dataset using good indices
        nData = len(goodInd)
        nTrain = int(trainvalRatio[0]*nData)
        nVal = int(trainvalRatio[1]*nData)
        trainInd = goodInd[:nTrain]
        valInd = goodInd[nTrain:nTrain+nVal]
        testInd = goodInd[nTrain+nVal:]
        
        #these dataarrays include all data between start and end date so we can prepare the datasets appropriately
        daCSR5dTrain = daCSR5d.isel(time=slice(0, trainInd[-1]+target_seq_len))
        daCSR5dVal   = daCSR5d.isel(time=slice(trainInd[-1]+target_seq_len,  valInd[-1]+target_seq_len))
        daCSR5dTest  = daCSR5d.isel(time=slice(valInd[-1]+target_seq_len, len(daCSR5d.coords['time'])))
        assert( (len(daCSR5dTrain['time']) + len(daCSR5dVal['time']) + len(daCSR5dTest['time'])) == len(daCSR5d['time']))
        
        
        self.mask = mask.values #convert da to numpy
        
        #Do normalization
        kwargs={}
        if rescale_method == 'deseason':            
            trainMat, monthlyMean = scaleDA(daCSR5dTrain, mask, rescale_method, op='train', **kwargs)
            kwargs['monthlymean'] = monthlyMean
            valMat, _ = scaleDA(daCSR5dVal, mask, rescale_method, op='test', **kwargs)
            testMat, _ = scaleDA(daCSR5dTest, mask, rescale_method, op='test', **kwargs)
        elif rescale_method == 'minimax':   
            trainMat, arrmin,arrmax = scaleDA(daCSR5dTrain, mask, rescale_method, op='train', **kwargs)
            kwargs['min'] = arrmin
            kwargs['max'] = arrmax
            valMat, _, _ = scaleDA(daCSR5dVal, mask, rescale_method, op='test', **kwargs)
            testMat, _, _ = scaleDA(daCSR5dTest, mask, rescale_method, op='test', **kwargs)
        elif rescale_method == 'norm':   
            #this will raise a runtime warning because all-zero value arrays
            trainMat, mu, stdvar = scaleDA(daCSR5dTrain, mask, rescale_method, op='train', **kwargs)            
            kwargs['mu'] = mu
            kwargs['stdvar'] = stdvar
            valMat, _, _ = scaleDA(daCSR5dVal, mask, rescale_method, op='test', **kwargs)
            testMat, _, _ = scaleDA(daCSR5dTest, mask, rescale_method, op='test', **kwargs)
        elif rescale_method == 'global':
            #do single global mean/std
            trainMat, mu, stdvar = scaleDA(daCSR5dTrain, mask, rescale_method, op='train', **kwargs)            
            kwargs['mu'] = mu
            kwargs['stdvar'] = stdvar
            valMat, _, _ = scaleDA(daCSR5dVal, mask, rescale_method, op='test', **kwargs)
            testMat, _, _ = scaleDA(daCSR5dTest, mask, rescale_method, op='test', **kwargs)
        else:
            raise ValueError("Invalid scaling method")
        
        allMat = np.concatenate([trainMat,valMat,testMat], axis=0)
                
        self.Xtrain, self.Ytrain = self.formData(trainInd, allMat)
        self.Xval, self.Yval = self.formData(valInd, allMat)
        self.Xtest,self.Ytest = self.formData(testInd, allMat)

        self.goodInd = goodInd
        self.trainvalRatio = trainvalRatio
        self.split = {'train':trainInd, 'val':valInd, 'test':testInd, 'all':goodInd}
        self.daCSR5d = daCSR5d
        self.landmask = landmask
        self.kwargs = kwargs

    def formData(self, indarr, arr):
        #assemble X,Y using valid indices
        #this is set as a prediction problem, using previous tws to predict future tws
        X = []; Y=[]
        for item in indarr:    
            X.append(arr[item-self.seq_len:item])
            Y.append(arr[item:item+self.target_seq_len])
        return X, Y
    
    def rescaleAndGenDataArray(self, arr, split='test',rescale_method='norm'):
        """This method inverse transform data in the good indices only
        """
        rarr = reScale(arr, self.mask, method=rescale_method, **self.kwargs)
        #form DA        
        #first form dates according to target_seq_len
        allInd = []
        for ind in self.split[split]:
            for i in range(1, self.target_seq_len+1):
                allInd.append(ind+i)
        print ('len all ind', len(allInd))
        daTemp = self.daCSR5d.isel(time=allInd)
        da = xr.DataArray(rarr, name='lwe_thickness', dims= daTemp.dims, coords=daTemp.coords )
        return da
        
class TWSDataSet(torch.utils.data.Dataset):
    def __init__(self, X, Y, mask=None, seq_len=4, target_seq_len=1):
        assert(len(X) == len(Y))
        self.X = X
        self.Y = Y
        self.mask= mask
        
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        
    def __getitem__(self, index):
        input = toTensor(self.X[index])
        target= toTensor(self.Y[index])
        mask  = toTensor(self.mask)
        return input,target,mask

    def __len__(self):
        return len(self.X)

class ClimateDataNorm():
    def __init__(self, region, seq_len, target_seq_len, dA, goodInd, trainvalRatio, mask, 
                rescale_method='norm'):
        """
        seq_len, lookback period
        target_seeq_len, forecast period
        goodInd and trainvalRatio: these need to be passed from twsnorm!
        mask, this needs to be passed down from twsnorm
        """
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        self.mask = mask

        #Split the dataset using good indices
        nData = len(goodInd)
        nTrain = int(trainvalRatio[0]*nData)
        nVal = int(trainvalRatio[1]*nData)
        trainInd = goodInd[:nTrain]
        valInd = goodInd[nTrain:nTrain+nVal]
        testInd = goodInd[nTrain+nVal:]
        
        #these dataarrays include all data between start and end date so we can prepare the datasets appropriately
        daTrain = dA.isel(time=slice(0, trainInd[-1]+target_seq_len))
        daVal   = dA.isel(time=slice(trainInd[-1]+target_seq_len,  valInd[-1]+target_seq_len))
        daTest  = dA.isel(time=slice(valInd[-1]+target_seq_len, len(dA.coords['time'])))
        assert( (len(daTrain['time']) + len(daVal['time']) + len(daTest['time'])) == len(dA['time']))
        
        #Do normalization
        kwargs={}
        if rescale_method == 'minimax':   
            trainMat, arrmin,arrmax = scaleDA(daTrain, mask, rescale_method, op='train', **kwargs)
            kwargs['min'] = arrmin
            kwargs['max'] = arrmax
            valMat, _, _ = scaleDA(daVal, mask, rescale_method, op='test', **kwargs)
            testMat, _, _ = scaleDA(daTest, mask, rescale_method, op='test', **kwargs)
        
        elif rescale_method == 'norm':   
            #this will raise a runtime warning because all-zero value arrays
            trainMat, mu, stdvar = scaleDA(daTrain, mask, rescale_method, op='train', **kwargs)            
            kwargs['mu'] = mu
            kwargs['stdvar'] = stdvar
            valMat, _, _ = scaleDA(daVal, mask, rescale_method, op='test', **kwargs)
            testMat, _, _ = scaleDA(daTest, mask, rescale_method, op='test', **kwargs)
        
        allMat = np.concatenate([trainMat,valMat,testMat], axis=0)
        
        #We only need predictors
        self.Xtrain = self.formData(trainInd, allMat)
        self.Xval   = self.formData(valInd, allMat)
        self.Xtest  = self.formData(testInd, allMat)
        self.goodInd = goodInd
        
        self.split = {'train':trainInd, 'val':valInd, 'test':testInd, 'all':goodInd}
        self.kwargs = kwargs

    def formData(self, indarr, arr):
        #assemble X using valid indices
        X = [arr[item-self.seq_len:item] for item in indarr]
        return X
    
class ClimateDataSet():
    """class for combining TWS and climate forcings
    """
    def __init__(self, X, Y, X_clim, mask=None, seq_len=4, target_seq_len=1):
        assert(len(X) == len(Y))
        
        self.X = X
        self.Y = Y        
        self.X_clim = X_clim
        nClimateVar = len(X_clim)
        self.mask= mask
        if not mask is None:            
            for i in range(len(X)):
                #dimensions are X & X_clim:[seq_len, lat, lon], Y:[target_seq_len, lat, lon]
                self.X[i] = np.einsum('kij,ij->kij', self.X[i], self.mask)
                self.Y[i] = np.einsum('kij,ij->kij', self.Y[i], self.mask)
                for ivar in range(nClimateVar):
                    tempvar = np.einsum('kij,ij->kij', self.X_clim[ivar][i], self.mask)
                    #remove the nan values
                    tempvar[np.isnan(tempvar)]=0.0
                    self.X_clim[ivar][i] = tempvar
                #do debugging
                """
                fig,ax = plt.subplots(3,1)
                ax[0].imshow(X[i][0,:,:])
                ax[1].imshow(X_clim[0][i][0,:,:])
                ax[2].imshow(X_clim[1][i][0,:,:])
                plt.savefig('testinput.png')
                plt.close()                
                print (np.min(X[i]), np.min(self.X_clim[0][i]), np.min(self.X_clim[1][i]))
                """
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        
    def __getitem__(self, index):
        input = toTensor(self.X[index])
        target= toTensor(self.Y[index])
        mask  = toTensor(self.mask)
        P = toTensor(self.X_clim[0][index])
        ET = toTensor(self.X_clim[1][index])
        return input,target,P,ET,mask

    def __len__(self):
        return len(self.X)

class ARDataSet(torch.utils.data.Dataset):
    def __init__(self, X, Y, mask=None, seq_len=4, target_seq_len=1):
        assert(len(X) == len(Y))
        self.X = X
        self.Y = Y
        self.mask= mask
        
        self.seq_len = seq_len
        self.target_seq_len = target_seq_len
        
    def __getitem__(self, index):
        input = toTensor(self.X[index])
        target= toTensor(self.Y[index])
        mask  = toTensor(self.mask)
        return input,target,mask

    def __len__(self):
        return len(self.X)

class DataGenerator(metaclass=abc.ABCMeta):
    """A general class for preparing data pairs"""
    def __init__(self, region, varlist, context_len, target_len, mask_ocean=True):
        """
        Params:
        ------
        region, global 
        varlist, list of variables, 'TWS', 'P', 'T', 'cycle'
        context_len,
        target_len,
        mask_ocean
        """

        self.context_len = context_len
        self.target_len = target_len
        self.varlist = varlist

        assert ('TWS' in varlist)
        xds,mask,landmask = load5ddatasets(region, coarsen=True, mask_ocean=mask_ocean, 
            reLoad=False, removeSeason=False)
        
        self.mask = mask.values
        self.landmask = landmask
        #===========loading data =============#
        dataDict = {'TWS':xds}
        for item in varlist:
            if item == 'P':
                #Precip
                xds_Precip = loadClimatedatasets(region, vartype='precip', daCSR5d= xds, aggregate5d=True, reLoad=False, precipSource='ERA5')
                print ('Precip', xds_Precip.shape)
                dataDict['P'] = xds_Precip
            elif item == 'T':
                #air temp
                xds_T      = loadClimatedatasets(region, vartype='airtemp', daCSR5d= xds, aggregate5d=True,reLoad=False)
                print ('airtemp', xds_T.shape)
                dataDict['T'] = xds_T
            elif item == 'SST':
                # SST 
                """
                xds_SST    = loadClimatedatasets(region, vartype='sst', daCSR5d= xds, aggregate5d=True,reLoad=False)
                print ('SST', xds_SST.shape)
                dataDict['SST'] = xds_SST
                """
                pass
            elif item == 'cycle':
                # monthly sin/cos cycles
                data_months = pd.to_datetime(xds['time'].values).month.values
                #change to (nT, H,W) numpy arrays
                monthsin =  np.sin((data_months-1)*(2.*np.pi/12))[:, np.newaxis,np.newaxis]
                monthcos =  np.cos((data_months-1)*(2.*np.pi/12))[:, np.newaxis,np.newaxis]
                monthsin =  np.tile(monthsin, (180,360) )
                monthcos =  np.tile(monthcos, (180,360) )

                dataDict['SIN'] = xr.DataArray(monthsin, coords=xds.coords,dims=xds.dims)
                dataDict['COS'] = xr.DataArray(monthcos, coords=xds.coords,dims=xds.dims)
                print ('sin cycle', monthsin.shape, 'cos cycle', monthcos.shape)
        self.dataDict = dataDict


    def scaleData(self, da, varname, scalerType, scalerInfo, op, mask):
        """Scale data array 
        Params
        ------
        da,
        varname, 
        scalerType,
        scalerInfo,
        op, 'train' or 'test'
        mask, this can be land mask or use mask = ones for the whole dataset
        """                         
        if op=='train':
            if scalerType=='norm':
                #cellwise normalization
                arr = da.values
                #taking temporal mean and std
                mu = np.mean(arr, axis=0)
                sigma = np.std(arr, axis=0)
                print ('99 percentile of std', np.percentile(sigma.flatten(), 99))
                scalerInfo[varname] = {'type': scalerType, 'mu': mu, 'sigma': sigma}
                for i in range(arr.shape[0]):
                    arr[i][mask==1] = (arr[i][mask==1]-mu[mask==1])/sigma[mask==1]
            elif scalerType == 'globalnorm':
                #normalize with global mean/std
                arr = da.values
                #taking temporal mean and std
                mu = np.nanmean(arr)
                sigma = np.nanstd(arr)
                scalerInfo[varname] = {'type': scalerType, 'mu': mu, 'sigma': sigma}
                arr = (arr - mu) / sigma
            elif scalerType=='lognorm':
                arr = np.log(da.values+1e-3)-np.log(1e-3)
                #taking temporal mean and std
                mu = np.nanmean(arr)
                sigma = np.nanmean(arr) 
                scalerInfo[varname]= {'type': scalerType, 'mu':mu, 'sigma':sigma}
                arr = (arr - mu) / sigma
            elif scalerType == 'minmax':
                arr = da.values
                arrmin = np.nanmin(arr,axis=0)
                arrmax = np.nanmax(arr,axis=0)
                scalerInfo[varname] = {'type': scalerType, 'min': arrmin, 'max': arrmax}
                for i in range(arr.shape[0]):
                    arr[i][mask==1] = (arr[i][mask==1]-arrmin[mask==1])/(arrmax[mask==1]-arrmin[mask==1])
            else:
                raise ValueError("Invalid scalertype")
            return arr, scalerInfo    
        else:
            if scalerType=='norm':
                #cellwise normalization
                arr = da.values
                #taking temporal mean and std
                mu,sigma = scalerInfo[varname]['mu'], scalerInfo[varname]['sigma']
                for i in range(arr.shape[0]):
                    arr[i][mask==1] = (arr[i][mask==1]-mu[mask==1])/sigma[mask==1]
            elif scalerType=='lognorm':
                arr = np.log(da.values+1e-3)-np.log(1e-3)
                #taking temporal mean and std
                mu, sigma = scalerInfo[varname]['mu'], scalerInfo[varname]['sigma']
                arr = (arr - mu) / sigma

            elif scalerType == 'globalnorm':
                #normalize with global mean/std
                arr = da.values
                #taking temporal mean and std
                mu, sigma = scalerInfo[varname]['mu'], scalerInfo[varname]['sigma']
                arr = (arr - mu) / sigma
            elif scalerType == 'minmax':
                arr = da.values
                arrmax,arrmin = scalerInfo[varname]['max'],scalerInfo[varname]['min']
                scalerInfo[varname] = {'type':scalerType, 'min':arrmin, 'max':arrmax}
                for i in range(arr.shape[0]):
                    arr[i][mask==1] = (arr[i][mask==1]-arrmin[mask==1])/(arrmax[mask==1]-arrmin[mask==1])
            return arr, scalerInfo

    @abc.abstractmethod
    def formData(self):
        """
        """

    def scaleBack(self, arr, varName, split):
        """This method inverse transform data in the good indices only
        """
        mask = self.mask
        scalerType = self.scalerInfo[varName]['type']
        if scalerType=='norm':
            #cellwise normalization
            #taking temporal mean and std
            mu,sigma = self.scalerInfo['TWS']['mu'], self.scalerInfo['TWS']['sigma']
            for i in range(arr.shape[0]):
                arr[i][mask==1] = (arr[i][mask==1]*sigma[mask==1])+mu[mask==1]
        elif scalerType == 'globalnorm':
            #normalize with global mean/std
            mu, sigma = self.scalerInfo['TWS']['mu'], self.scalerInfo['TWS']['sigma']
            arr = arr * sigma +  mu
        elif scalerType == 'minmax':
            arrmax,arrmin = self.scalerInfo['TWS']['max'], self.scalerInfo['TWS']['min']
            for i in range(arr.shape[0]):
                arr[i][mask==1] = arr[i][mask==1]*(arrmax[mask==1]-arrmin[mask==1])+arrmin[mask==1]

        #form DA        
        #first form all dates according to target_seq_len
        allInd = []
        da = self.dataDict['TWS']
        for ind in self.scalerInfo['split'][split]:
            for i in range(1, self.target_len +1):
                allInd.append(ind+i)
        daTemp = da.isel(time=allInd)
        da = xr.DataArray(arr, name='lwe_thickness', dims= daTemp.dims, coords=daTemp.coords )
        return da

class ARGenerator(DataGenerator):
    """class for autoregression
    """
    def __init__(self, region, varlist, context_len, target_len, batch_size):

        DataGenerator.__init__(self, region, varlist, context_len, target_len)
        self.batch_size = batch_size

    def formData(self, randomTrainVal=True):
        def __findGoodConSecDays(tt, lookbackLen=5, lookforwardLen=1):
            #tt should be in days elapsed
            #find data with consequtive dates in TWS
            #note: this will get sequenceLen+1 gaps
            dT = 5 #days for csr5d
            prevDate = None    
            consecInd = []
            
            for i in range(lookbackLen, len(tt)-lookforwardLen):
                currDate = tt[i]
                prevDate = tt[i-lookbackLen]
                futuDate = tt[i+lookforwardLen]            
                deltaBT = (currDate - prevDate)
                deltaFT = (futuDate - currDate)
                if deltaBT == lookbackLen*dT and deltaFT == lookforwardLen*dT:
                    consecInd.append(i)

            print ('number of good dates', len(consecInd))
            return consecInd
        
        startDate = '2002-04-01'
        daTWS = self.dataDict['TWS']
        #the time axes must be the same for all data
        tt = (pd.to_datetime(daTWS['time'].values) - datetime.datetime.strptime(startDate, '%Y-%m-%d')).days
        #select consec dates
        consecInd = __findGoodConSecDays(tt, self.context_len, self.target_len)
        #input shape
        #[batch, seq, inputDim, H, W]
        self.__inputDim = len(self.dataDict.keys())
        H,W = daTWS.shape[1:]  #daTWS in (time, lat, lon)

        nData = len(consecInd)
        nTrain = int(0.7*nData)
        nVal = int(0.15*nData)
        #get train ind
        indTrain = consecInd[:nTrain]
        indVal = consecInd[nTrain:nTrain+nVal]
        indTest = consecInd[nTrain+nVal:]

        #do normalization on inputs/outputs
        scalerTypes={'TWS':'globalnorm', 'P': "lognorm", "T": "norm", "SIN":"minmax", "COS":"minmax"}
        scalerInfo = {}
        matDict = {}
        for item in self.dataDict.keys():
            da = self.dataDict[item]
            daTrain = da.isel(time=slice(0, indTrain[-1]+self.target_len))
            trainMat,scalerInfo = self.scaleData(daTrain, varname=item, scalerType= scalerTypes[item], op="train", 
                                      scalerInfo=scalerInfo, mask=self.mask)
            daVal     = da.isel(time=slice(indTrain[-1]+self.target_len,  indVal[-1]+self.target_len))
            valMat, _ = self.scaleData(daVal, varname=item, scalerType= scalerTypes[item], op="test", 
                                      scalerInfo=scalerInfo, mask=self.mask)

            daTest  = da.isel(time=slice(indVal[-1]+self.target_len, len(tt)))
            testMat, _ = self.scaleData(daTest, varname=item, scalerType= scalerTypes[item], op="test", 
                                      scalerInfo=scalerInfo, mask=self.mask)
            #make sure the combined datasets covers all data
            assert( (len(daTrain['time']) + len(daVal['time']) + len(daTest['time'])) == len(daTWS['time']))                    
            matDict[item] = np.concatenate([trainMat,valMat,testMat], axis=0)

            if item == 'TWS':
                scalerInfo['split']={'train':indTrain,'val':indTest,'test':indTest}

        self.scalerInfo = scalerInfo
        #form data in such a way that uses past TWS to predict future TWS
        X = np.zeros((nData, self.context_len, self.__inputDim, H, W)) #predictors
        Y = np.zeros((nData, self.target_len, 1, H, W)) #predictand
        
        predictors = ['TWS', 'P', 'T', 'SIN', 'COS']

        for ix, id in enumerate(consecInd):
            for ik, key in enumerate(predictors):
                X[ix,:, ik, :, :] = matDict[key][id-self.context_len:id]
            Y[ix,:, 0, :, :]  = matDict['TWS'][id:id+self.target_len]
        Xtrain = toTensor(X[:nTrain,...])
        Ytrain = toTensor(Y[:nTrain,...])
        Xval   = toTensor(X[nTrain:nTrain+nVal,...])
        Yval   = toTensor(Y[nTrain:nTrain+nVal,...])
        Xtest  = toTensor(X[nTrain+nVal:,...])
        Ytest  = toTensor(Y[nTrain+nVal:,...])

        mask   = toTensor(self.mask)
 
        trainDataset = ARDataSet(Xtrain, Ytrain, mask=self.mask, seq_len=self.context_len, 
                                target_seq_len= self.target_len)
        valDataset = ARDataSet(Xval, Yval, mask=self.mask, seq_len=self.context_len, 
                                target_seq_len= self.target_len)
        testDataset = ARDataSet(Xtest, Ytest, mask=self.mask, seq_len=self.context_len, 
                                target_seq_len= self.target_len)
        
        trainLoader = DataLoader(trainDataset,
                                batch_size=self.batch_size,
                                shuffle = True,
                                num_workers=4,
                                pin_memory=True
                                )
        valLoader = DataLoader(valDataset,
                                batch_size=self.batch_size,
                                shuffle = False,
                                num_workers=4,
                                pin_memory=True
                                )

        testLoader = DataLoader(testDataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True
                                )

        return trainLoader,valLoader, testLoader

class SNPGenerator(DataGenerator):
    """class for sequential NP
    [P, T, SIN, COS] -> [TWS]
    """
    def __init__(self, **kw_args):
        DataGenerator.__init__(self, **kw_args)

    def formData(self):
        def __findGoodConSecDays(tt, sequenceLen=5):
            #find data with consequtive dates in TWS
            #note: this will get sequenceLen+1 gaps
            #because I'm not using TWS, so there's no 
            #need to future data here
            dT = 5 #days for csr5d
            prevDate = None    
            consecInd = []
            
            for i in range(sequenceLen, len(tt)-sequenceLen):
                currDate = tt[i]
                prevDate = tt[i-sequenceLen]
                if (currDate - prevDate) == sequenceLen*dT:
                    consecInd.append(i)
            print ('number of good dates', len(consecInd))
            return consecInd

        def __formDataLoader(loaderType):
            task = {
                    'x_context': [],
                    'y_context': [],
                    'x_target': [],
                    'y_target': [],
                    }

            #split for training/valication
            if loaderType == 'train':
                rng = range(0, nTrain, 2)
            else:
                rng = range(nTrain, nData, 2) 
            for i in rng:
                x_context = X[i]
                y_context = Y[i]
                x_target = X[i+1]
                y_target = Y[i+1]
                task['x_context'].append(x_context)
                task['y_context'].append(y_context)
                task['x_target'].append(x_target)
                task['y_target'].append(y_target)

            # Stack batch and convert to PyTorch.
            task = {k: torch.tensor(np.stack(v, axis=0), dtype=torch.float32) for k, v in task.items()}
            print (task['x_context'].shape, task['x_target'].shape, task['y_context'].shape,task['y_target'].shape)
            
            return DataLoader(TensorDataset(task['x_context'], task['y_context'], task['x_target'], task['y_target']),
                            shuffle=(loaderType=='train'), batch_size=self.batch_size, num_workers=4)

        daTWS = self.dataDict['TWS']
        tt = list(daTWS.index.values)
    
        consecInd = __findGoodConSecDays(tt, sequenceLen=self.context_length)
        nData = len(consecInd)
        nTrain = int(0.85*nData)

        #do normalization on inputs/outputs
        scalerTypes={'TWS':'globalnorm', 'P': "lognorm", "T": "norm", "SIN":"minmax", "COS":"minmax"}
        scalerInfo = {}
        matDict = {}

        #do data normalization
        for item in self.dataDict.keys():
            da = self.dataDict[item]
            daTrain = da.isel(time=slice(0, consecInd[nTrain]))
            trainMat,scalerInfo = self.scaleData(daTrain, varname=item, scalerType= scalerTypes[item], op="train", 
                                      scalerInfo=scalerInfo, mask=self.mask)

            daTest  = da.isel(time=slice(consecInd[nTrain], len(tt)))
            testMat, _ = self.scaleData(daTest, varname=item, scalerType= scalerTypes[item], op="test", 
                                      scaelerInfo=scalerInfo, mask=self.mask)
            matDict[item] = np.concatenate([trainMat, testMat], axis=0)

        predictors = ['P', 'T', 'SIN', 'COS']
        self.__inputDim = len(predictors)
        H,W = daTWS.shape[1:]

        #form data in such a way learns sequence to sequence
        X = np.zeros((nData, self.context_len, self.__inputDim, H, W)) #precitors
        Y = np.zeros((nData, self.context_len, 1, H, W)) #predictand
        
        predictors = ['P', 'T', 'SIN', 'COS']
        for ix, id in enumerate(consecInd):
            for ik,key in enumerate(predictors):
                X[ix, :, ik, :, :] = matDict[key][id-self.context_len:id]
            Y[ix,:, 0, :, :]  = matDict['TWS'][id:id-self.context_len:id]                                                    

        #now divide into training/val and testing
        trainLoader = __formDataLoader(loaderType="train")
        testLoader  = __formDataLoader(loaderType="test")

        return trainLoader,testLoader

def main():
    llcrnrlon =-109; llcrnrlat=24 #the bound of downloaded ERA5 data is 24 to 50
    blocksize = 64
    cellsize  = 0.25
    lon0 = llcrnrlon; lon1 = lon0+(blocksize)*cellsize
    lat0 = llcrnrlat; lat1 = lat0+(blocksize)*cellsize    
    region = (lat0,lon0,lat1,lon1)

    #daCSR5d = load5ddatasets(region)
    #selectConsecData(daCSR5d)
    twsNorm = TWSDataNorm(region = region, seq_len=4, target_seq_len=1, rescale_method='norm')
    
    trainDataset = TWSDataSet(twsNorm.Xtrain, twsNorm.Ytrain, twsNorm.mask, seq_len=4, target_seq_len=1)
    valDataset = TWSDataSet(twsNorm.Xval, twsNorm.Yval, twsNorm.mask, seq_len=4, target_seq_len=1)
    testData = TWSDataSet(twsNorm.Xtest, twsNorm.Ytest, twsNorm.mask, seq_len=4, target_seq_len=1)
    print (len(trainDataset) + len(valDataset) + len(testData))
    print (len(twsNorm.goodInd))
if __name__ == '__main__':
    main()

