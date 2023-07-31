#Author: Alex Sun
#date: 06182023
#purpose:
#==============================================================================
import os
import glob
import xarray as xr
from natsort import natsorted

from myutils import getGraceRoot

def concatGPCPDailyFiles():
    bigarr = []
    for iyear in range(2002, 2023):
        files = os.path.join(getGraceRoot(), f'data/gpcp/{iyear}/*.nc')
        dailyFiles = natsorted(glob.glob(files))
        print (f'Number of files in {iyear}, {len(dailyFiles)}')
        for item in dailyFiles:
            bigarr.append(xr.open_dataset(item)['precip'])

    da = xr.concat(bigarr, dim='time')
    da = xr.DataArray(da.values,name="gpcp",coords=da.coords,dims=da.dims)
    #GPCP lon is from 0 360
    #shift from 0, 360 to -179,180        
    da.coords['longitude'] = (da.coords['longitude'] + 180) % 360 - 180
    da = da.sortby(da.longitude)
    da = da.sortby(da.latitude)
    if 'longitude' in da.dims:
        da = da.rename({'longitude':'lon', 'latitude':'lat'})   
    da = da.transpose('time', 'lat', 'lon')
    da.to_netcdf(os.path.join(getGraceRoot(), 'data/gpcp/gpcp_combined.nc'))

def loadGPCP(rootfolder):
    ds = xr.open_dataset(os.path.join(rootfolder, 'gpcp_combined.nc'))
    return ds['gpcp']


if __name__ == '__main__':
    concatGPCPDailyFiles()