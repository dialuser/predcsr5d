#author: alex sun
#date: 07/05
#purpose: convert grib to nc
# this must be run on TACC under conda env xarr
#
import os,sys
import xarray as xr
from glob import glob
from grdc import getRegionBound

regiondict = {
    'north_america': 'na',
    'south_america': 'sa',
    'europe': 'eu',
    'africa': 'af',
    'south_pacific': 'au',
    'asia': 'as',
}

def convertGRIB(region, reGen=False):
    rootdir = '/home/suna/work/predcsr5d/data/glofas'
    varname = 'dis24'

    if reGen:
        allfiles = sorted(glob(os.path.join(rootdir, regiondict[region], '*.grib')))
        print (allfiles)
        outncfile = os.path.join(rootdir, f'combined_ncfiles/glofas_{regiondict[region]}.nc')
        bigNC = []
        for afile in allfiles:
            print ('processing ', afile)
            da = xr.open_dataset(afile, engine='cfgrib')[varname]
            lon0, lat0, lon1, lat1 =  getRegionBound(region, source='glofas')
            #clean up lat/lon per grib file
            da.coords['longitude'] = (da.coords['longitude'] + 180) % 360 - 180 
            da = da.sortby('latitude')
            da = da.sel(latitude=slice(lat0,lat1), longitude=slice(lon0, lon1))
            bigNC.append(da)
            da = None

        bigNC = xr.concat(bigNC, dim='time')
        print ('finished')
        #rename 
        bigNC = bigNC.rename({'longitude':'lon', 'latitude':'lat'})
        bigNC.to_netcdf(outncfile)  
        #bigNC.isel(time=100).to_netcdf(outncfile)
if __name__ == "__main__":
    region = 'europe'
    convertGRIB(region=region, reGen=True)