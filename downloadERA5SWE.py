#Author: Alex Sun
#Date: 0801
#Purpose: download ERA5 SWE data from CDS portal
#Requirements: 
#  - cdsapi
#=================================================================================================
import os
import cdsapi
import xarray as xr
from calendar import monthrange
from subprocess import call
import glob

#set download folder
fileroot = '/home/suna/work/grace/data/era5/swe'
dailyfileroot = '/home/suna/work/grace/data/era5/swe/daily'

def downloadERA5SWE(startYear=2002, endYear=2022):
    """Downloads nc files for each calendar day
    """
    os.makedirs(fileroot, exist_ok=True)
    os.makedirs(dailyfileroot, exist_ok=True)
    c = cdsapi.Client()
    for iyear in range(startYear, endYear+1):
        yeartag = '{:4d}'.format(iyear)
        for imon in range(1,13):
            monthtag='{:02d}'.format(imon)
            daysInMonth = monthrange(iyear, imon)[1]    
            for iday in range(1,daysInMonth+1):
                daytag = '{:02d}'.format(iday)
                ncfile = os.path.join(fileroot, 'swe{0:4d}_{1:02d}_{2:02d}.nc'.format(iyear,imon,iday))
                print (ncfile)
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': [
                            'snow_depth',
                        ],
                        'year': yeartag,
                        'month': monthtag,
                        'day': daytag,
                        'time': [
                            '00:00', '01:00', '02:00',
                            '03:00', '04:00', '05:00',
                            '06:00', '07:00', '08:00',
                            '09:00', '10:00', '11:00',
                            '12:00', '13:00', '14:00',
                            '15:00', '16:00', '17:00',
                            '18:00', '19:00', '20:00',
                            '21:00', '22:00', '23:00',
                        ],
                    },
                    ncfile)
                #do postprocessing
                ds = xr.open_dataset(ncfile)
                ds = ds.resample(time='1d').mean()
                ds.to_netcdf(os.path.join(dailyfileroot, 'swe{0:4d}_{1:02d}_{2:02d}.nc'.format(iyear,imon,iday)))
                #remove the hourly file
                cmd = 'rm {0}'.format(ncfile)
                call(cmd, shell=True)

def concatDailyFiles():
    """concatening both variables
    There should be 6940 files from 2002/01/01 to 2021/12/31
    """
    import time
    filepath= os.path.join(fileroot, 'daily/*.nc')
    files =sorted(glob.glob(filepath))
    
    allDS = []
    starttime = time.time()
    for afile in files:
        ds = xr.open_dataset(afile)
        ds.coords['longitude'] = ds.coords['longitude']+0.125
        ds.coords['latitude']  = ds.coords['latitude'] +0.125
        ds = ds.isel(latitude=slice(0, 720))
        #make 1 deg
        #ds = ds.coarsen(latitude=4,longitude=4,boundary='exact').mean()        
        allDS.append(ds)
    combined = xr.concat(allDS, dim='time')
    combined.coords['longitude'] = (combined.coords['longitude'] + 180) % 360 - 180
    combined = combined.sortby(combined.longitude)
    combined = combined.sortby(combined.latitude)
    combined.to_netcdf('era5_swe_2002_2022_daily_025.nc')
    print ('time elapsed ', time.time()-starttime)        

if __name__ == '__main__':  

    downloadERA5SWE(2002, 2022)
    concatDailyFiles()
