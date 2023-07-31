#author: Alex Sun
#date: 06122023
#calculate basin flood plain area
import xarray as xr
import rasterio
import rioxarray 
from grdc import getBasinMask, getCatalog, getStationMeta
import matplotlib.pyplot as plt
import numpy as np

def loadFloodPlain(aoi, continent, plotting=False):
    rasterfile = f'/home/suna/work/predcsr5d/data/GFPLAIN250mTIFF/{continent}.TIF'

    # with rasterio.open(rasterfile) as dataset:
    #     data0 = dataset.read()
    #     print (data0.shape) 
    #     print (dataset.nodata)
    #     #rasterstats.zonal_stats(aoi, dataset, stats=['nodata'])
    #     #ws_count = dataset \
    #     #        .where(data.Snow_Albedo_Daily_Tile == 150)

    xds = xr.open_dataset(rasterfile)['band_data']
    xds.rio.write_crs("epsg:4326", inplace=True)
    xds = xds.sortby('y')     

    proj_crs = 'EPSG:5070'
    with rioxarray.set_options(export_grid_mapping=False):
        bbox=aoi.bounds
        #convert to [lon0,lat0,lon1,lat1]
        bbox=bbox.values.tolist()[0]
        basinxds = xds.sel(y=slice(bbox[1],bbox[3]),x=slice(bbox[0],bbox[2]))        

        clipped = basinxds.rio.clip(aoi.geometry, aoi.crs, from_disk=True, drop=True, invert=False)                      
        
        #now reproject to NAD83 / Conus Albers, unit: m
        clipped = clipped.rio.reproject(proj_crs)
        
        #mag of TWS motion
        clipped = clipped.where(clipped.notnull())        
        xcoords = clipped.coords['x'].values
        ycoords = clipped.coords['y'].values
        cellarea = np.abs((xcoords[1]-xcoords[0]) * (ycoords[1]-ycoords[0])) # in [m^2]
        
        arr = clipped.values.squeeze()
        aoi_c = aoi.copy()
        aoi_c = aoi_c.to_crs(proj_crs )
        basin_area = aoi_c['geometry'].area.values[0] # [m^2]
        flood_area = len(np.where(arr<=1)[0])*cellarea
        fraction_fplain =  flood_area/basin_area
        print (fraction_fplain)
        #this takes a long time
        if plotting:
            plt.figure()
            clipped.plot()
            plt.savefig('firstplot.png')
            plt.close()
        return fraction_fplain
def main():
    continent='NA'
    region = 'north_america'
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)
    try:
        for ix, row in dfStation.iterrows():
            stationID = row['grdc_no']
            basin_gdf = getBasinMask(stationID=stationID, region=region, gdf=gdf)
            print (basin_gdf)    
            loadFloodPlain(aoi=basin_gdf, continent=continent)
            print ("\n")
    except Exception:
        pass

if __name__ == '__main__':
    main()
