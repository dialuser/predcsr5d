#author: alex sun
#date: 06282023
#purpose: load tiff
"""
# BangladeshInundationHistory
 
The *BangladeshInundationHistory* dataset contains the model output published in the paper ["**Giezendanner et al.** , *Inferring the past: a combined CNN--LSTM deep learning framework to fuse satellites for historical inundation mapping*, CVPR 23 Earthvision workshop"](https://arxiv.org/abs/2305.00640).
 
The data to train and infer these maps will be made available on Radian Earth (link will be updated here), the code is available on [GitHub](https://github.com/GieziJo/cvpr23-earthvision-CNN-LSTM-Inundation).
 
The dataset is structured as follow:
- `CVPR23FractionalInundationHistory`:
 
    Contains 985 `.tiff` files covering most of Bangladesh every 8 days, at 500 meters resolution, from 2001 to 2022.
    The file is named `<timestamp>.tiff` where timestamp is unix time.
    This timestamp can be converted to datetime in python as follow:
    ```python
    import datetime
    datetime.datetime.fromtimestamp(timeStamp / 1000.0, tz=datetime.timezone.utc)
    ```
"""
import xarray as xr
import rioxarray 
from glob import glob
from natsort import natsorted
import os, sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hydrostats as Hydrostats

os.environ['USE_PYGEOS'] = '0'

def combineTiff2NC(ncoutfile):
    root_dir = '/home/suna/work/bangladeshflood'
    allFiles = natsorted(glob(os.path.join(root_dir, '*.tiff')))
    bigdata=[]
    for ix, item in enumerate(allFiles):
        filename =  os.path.basename(item)
        timeStamp = int(filename[:-5])
        timeStamp = datetime.datetime.fromtimestamp(timeStamp / 1000.0, tz=datetime.timezone.utc)
        da = rioxarray.open_rasterio(item,chunks=True)
        if ix == 0:
            #asun: 04202023, fixed the issue w/ Zhi's new domain simulations
            #we simply fix the coordinates
            lat = da.coords['y'].values
            lon = da.coords['x'].values

        da = xr.DataArray(da.values, dims=('time', 'lat', 'lon'), 
                coords={'time': [pd.to_datetime(timeStamp)], 
                        'lat': lat,
                        'lon': lon
                },
                name='fracinundation',
                attrs = da.attrs
        )        

        bigdata.append(da)
    bigdata = xr.concat(bigdata, dim='time')
    print (bigdata.coords)
    bigdata.to_netcdf(ncoutfile)    

def aggregateBangledash(ncfile):
    assert(os.path.exists(ncfile))

    da = xr.open_dataset(ncfile)['fracinundation']
    nT = len(da.time)
    fracArea = np.zeros(nT)
    for it in range(nT):
        arr = da.isel(time=it).values
        arr = arr[arr>=0]
        fracArea[it] = np.sum(arr)/len(arr)
    #form data series
    fracArea = pd.Series(fracArea, index=pd.to_datetime(da.time.values))
    
    return fracArea

def formUniformTWS(tws_avg):
    timestamps = tws_avg.time.values
    daterng = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='5D')    
    #get key/value pairs to map from daterng to timestamps
    K = []
    V = []
    for iy, t1 in enumerate(timestamps):
        for ix, t2 in enumerate(daterng):
            if abs((t1-t2).days)<3.0:
                K.append(ix)
                V.append(iy)          
    K = np.array(K)
    V = np.array(V)
    #assign values to 5-day dates, missing values are indicated by -999
    arrOut = np.zeros(daterng.shape) + np.NaN

    arrOut[K] = tws_avg.values[V]
    
    tws_avg = pd.Series(arrOut, index= pd.to_datetime(daterng))

    return tws_avg

def compare2CSR5d(config, ncfile):
    from dataloader_global import load5ddatasets
    fracArea = aggregateBangledash(ncfile)
    da = xr.open_dataset(ncfile)['fracinundation']
    lat0,lat1 = da.coords['lat'].min(), da.coords['lat'].max()
    lon0,lon1 = da.coords['lon'].min(), da.coords['lon'].max()

    region = (lat0,lon0,lat1,lon1)
    xds,_, _ = load5ddatasets(region=region, 
        coarsen=False, 
        mask_ocean=config.data.mask_ocean, 
        removeSeason=config.data.remove_season, 
        reLoad=config.data.reload, 
        csr5d_version=config.data.csr5d_version,
        startYear=datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
        endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year
    )
    #aggregate 
    tws_avg = xds.mean(dim=['lat', 'lon'])
    tws_avg = formUniformTWS(tws_avg)

    #resample fracArea
    original_fracarea = fracArea.copy()
    fracArea = fracArea[(fracArea.index>=pd.to_datetime(tws_avg.index[0])) & (fracArea.index<=pd.to_datetime(tws_avg.index[-1])) ]
    fracArea = fracArea.resample('5D')
    fracArea = fracArea.interpolate(method='linear')

    fig, ax = plt.subplots(1,1, figsize=(12, 8))
    tws_avg.plot.line(ax=ax, color='k', legend=True, linewidth=1.5)
    ax.set_ylabel('EWH [cm]')
    ax2= ax.twinx()
    fracArea.plot(ax=ax2, kind='line', legend=True, color='blue', linewidth=1.5)
    ax2.set_ylim([0, 0.6])
    ax2.set_ylabel('Fraction Inundation Area', fontdict={'color':'blue'})
    #original_fracarea.plot(ax=ax2, kind='line', legend=True, color='skyblue')
    print (len(fracArea), len(tws_avg))
    tws_avg = tws_avg[:-2]
    rho = Hydrostats.pearson_r(fracArea, tws_avg)
    plt.legend()
    print ('corr', rho)
    plt.savefig('tws_avg.png')
    plt.close()

if __name__ == '__main__':    
    from myutils import load_config
    config = load_config('config.yaml')

    itask = 3
    ncfile = f'data/bangladesh_combined.nc'

    if itask ==1:
        combineTiff2NC(ncfile)
    elif itask == 2:
        aggregateBangledash(ncfile)
    elif itask == 3:
        compare2CSR5d(config, ncfile)