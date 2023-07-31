#author: Alex Sun
#purpose: load gnss
#============================================================================
import os
import xarray as xr
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def loadGNSS(reLoad=False):
    import cftime
    """load h5 GNSS dataset and convert it to netcdf format
    Adusumilli, S., Borsa, A. A., Fish, M. A., McMillan, H. K., & Silverii, F. (2019). 
    A decade of water storage changes across the contiguous United States from GPS and Satellite 
    Gravity. Geophysical Research Letters, 46, 13006-13015. https://doi.org/10.1029/2019GL085370
    """
    ncfile = os.path.join("/home/suna/work/grace/data/gnss", "gnss_water_storage_v1.nc")

    if reLoad:
        hdf5file = os.path.join("/home/suna/work/grace/data/gnss", "gnss_water_storage_v1.h5")

        ncf = Dataset(hdf5file, diskless=True, persist=False)
        xds = xr.open_dataset(xr.backends.NetCDF4DataStore(ncf))
        #reformat
        lat = xds['latitude'].values
        lon = xds['longitude'].values
        epoch = xds['time'].values
        epoch = xr.CFTimeIndex(epoch).to_datetimeindex()
        
        xds = xr.Dataset(data_vars={'cmwe':(("time", "lon", "lat"), np.array(xds['cmwe'].values)), 
                            'signal_to_noise_ratio':(("lon", "lat"), np.array(xds['signal_to_noise_ratio'].values)),
                            'uncertainty': (("lon", "lat"), np.array(xds['uncertainty'].values))
                        },
                        coords={'lat': lat, 'lon': lon, 'time': epoch},
        )
        xds = xds.transpose('time', 'lat', 'lon')

        #export to netcdf
        xds.to_netcdf(ncfile)
        #do testing
        plt.figure()
        xds['cmwe'].isel(time=0).plot()
        plt.savefig('test_gnss.png')
        plt.close()

    da = xr.open_dataset(ncfile)['cmwe']
    return da

def getCSR5d(config):
    from myutils import getRegionExtent    
    from dataloader_global import load5ddatasets
    region = getRegionExtent(regionName="global")
    
    #TWS
    daCSR5d,_, _ = load5ddatasets(region=region, 
        coarsen=False, 
        mask_ocean=config.data.mask_ocean, 
        removeSeason=config.data.remove_season, 
        reLoad=config.data.reload, 
        csr5d_version=config.data.csr5d_version,
        startYear=datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
        endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year
    )
    #find dates CSR5d and GNSS share
    daGNSS = loadGNSS()  
    gnss_dates = pd.to_datetime(daGNSS.time.values)
    csr_dates =  pd.to_datetime(daCSR5d.time.values)

    #for each gnss we want to find the closest csr date
    maxdist = 7.5 # days
    gnss2csr = {}
    
    for ix,d1 in enumerate(gnss_dates):
        dist = [np.abs(d1-d2).days for d2 in csr_dates]
        min_dist = np.min(dist)
        if min_dist < maxdist:
            #get index
            gnss2csr[ix] = np.argmin(dist)
        
    print ('# of gnss data', len(gnss_dates))
    print (len(gnss2csr.keys()))
    
    #do spatial subsetting
    for item in gnss2csr.keys():
        print (csr_dates[gnss2csr[item]], gnss_dates[item])
def main():
    from myutils import load_config
    config = load_config('config.yaml')

    getCSR5d(config)
    #loadGNSS()

if __name__ == "__main__":
    main()