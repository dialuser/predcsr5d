#date: 06/17/2023
#author: alex sun
#purpose: aggregate history
#=========================================================================
import os
import xarray as xr
import glob 
import time 

def combineREGEN():
    #combine all yearly data into a single file
    fileroot='/home/suna/work/grace/data/REGEN'
    filepath= os.path.join(fileroot, 'REGEN_AllStns_V1-2019_*.nc')
    files =sorted(glob.glob(filepath))

    allDS = []

    starttime = time.time()
    for afile in files:
        print (afile)
        ds = xr.open_dataset(afile)
        allDS.append(ds)
    combined = xr.concat(allDS, dim='time')
    print (combined)
    combined.to_netcdf(os.path.join(fileroot, 'regen_combined_deg01.nc'))

    print ('time elapsed ', time.time()-starttime)        

def loadREGEN(fileroot):
    ds = xr.open_dataset(os.path.join(fileroot, 'regen_combined_deg01.nc'))
    return ds['p']


if __name__ == '__main__':   
    combineREGEN()