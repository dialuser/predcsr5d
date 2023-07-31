#author: alex sun
#date: 06112023
#purpose: get grfr data
#date: 06152023, get list from yuan
#=======================================================================================
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hydrostats as HydroStats
import geopandas as gpd

import glofas
from grdc import readStationSeries,getStationMeta,getCatalog,\
    getGloFASStations,extractBasinOutletQ,loadGLOFAS4CONUS


def getRegionBound(region):
    if region=='north_america':
        #return (-180, 10, -50, 70)
        #return (-169, 25, -20, 55)
        return (-125,25, -67, 55)
    elif region == 'south_america':
        return (-90, -50,  -30,  0)
    elif region == 'africa':
        return (-20, -35, 55, 40    )
    elif region == 'europe':
        return (0, 40, 40, 80)
    elif region == 'south_pacific':
        return (110, -10, 160, -45)

class GRFR():
    def __init__(self) -> None:
        self.grfr_root = '/home/suna/work/grfr'
        self.grdr_catalog = pd.read_csv(os.path.join(self.grfr_root, 'join_MERIT_GRDC_area_control_07.csv'))
        self.basinDict = self.__getMapper()
        self.region = 'north_america'

    
        self.dfStation = getStationMeta(getCatalog(self.region))
        #define date range
        self.daterng = pd.date_range(start='2002/01/01', end='2019/12/31', freq='1D')

        self.getGRFR()
    def getGRFR(self,startYear=1979,endYear=2019):
        """Load GRFR discharge (currently for conus, i.e., region 7)
            Dimensions:  (time: 14975, rivid: 346238)
            Coordinates:
            * time     (time) datetime64[ns] 1979-01-01 1979-01-02 ... 2019-12-31
            * rivid    (rivid) int32 71000001 71000002 71000003 ... 78028479 78028480
            Data variables:
            Qout     (time, rivid) float32 ...    
        """
        comIDset = list(self.basinDict.values())
        nc_path = os.path.join(self.grfr_root, 'output_pfaf_07_1979-2019.nc')
        ds = xr.open_dataset(nc_path)
        da = ds.Qout

        self.grfr_da = da.sel(rivid=comIDset, time=slice(f'{startYear}-01-01', f'{endYear}-12-31'))

    def compareGRDCToGRFR(self):
        basinDict={}
        stationIDset = list(self.basinDict.keys())
        for stationID in stationIDset:        
            dfGRDC = readStationSeries(stationID=stationID, region=self.region)
            dfGRDC = dfGRDC[dfGRDC.index.year>=2002].copy(deep=True)
            #this step introduces NaN!!!
            dfGRDC = dfGRDC.reindex(self.daterng)

            #convert da to df
            grfrQ = self.grfr_da.sel(rivid = self.basinDict[stationID]) #m3/s
            grfrQ = grfrQ.to_dataframe()['Qout']
            grfrQ = grfrQ[grfrQ.index.year>=2002].copy(deep=True)

            riv = self.dfStation[self.dfStation['grdc_no']==stationID]['river_x'].values[0]

            kge= HydroStats.kge_2012(
                            grfrQ.values,
                            dfGRDC.values.squeeze())
            print (stationID, riv, kge)
            basinDict[stationID] = kge
            fig,ax = plt.subplots(1,1)
            dfGRDC.plot(ax=ax, label='GRDC')
            grfrQ.plot(ax=ax, label='GRFR')
            plt.legend()
            ax.set_title(f'Station {stationID},river={riv}, kge={kge:4.3f}')
            plt.savefig(f'{stationID}.png')
            plt.close()

    def __getMapper(self):
        """create mapper between grdc stationid and comid
        """
        station_set = self.grdr_catalog['stationid'].values
        comid_set = self.grdr_catalog['COMID'].values
        basinDict = dict(zip(station_set, comid_set))
        return basinDict

    def compareGLOFASToGRFR(self):
        #define date range
        daterng = pd.date_range(start='2002/01/01', end='2020/12/31', freq='1D')

        gdf = getCatalog(self.region)
    
        dfStation = getGloFASStations(gdf)
        dsGloFAS = loadGLOFAS4CONUS()
        extents = glofas.getExtents()

        valid_stationSet = list(self.basinDict.keys()) 
        for ix, row in dfStation.iterrows():
            stationID = row['grdc_no']
            if stationID in valid_stationSet:
                riverName = row['river_x']
                lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])

                if lat>extents['min_lat'] and lat<extents['max_lat']:
                    if lon>extents['min_lon'] and lon<=extents['max_lon']:
                        daGloFAS = extractBasinOutletQ(loc=(lat,lon), ds=dsGloFAS)
                        daGloFAS = daGloFAS.squeeze() #this needs to be series 

                        dfGRDC = readStationSeries(stationID=stationID, region=self.region)
                        dfGRDC = dfGRDC[dfGRDC.index.year>=2002].copy(deep=True)

                        #this step introduces NaN!!!
                        dfGRDC = dfGRDC.reindex(daterng)
                        #convert da to df
                        dfGloFAS = pd.DataFrame(daGloFAS.values.squeeze(), index=daGloFAS.time, columns=['GloFAS'])

                        kge = HydroStats.nse(dfGloFAS['GloFAS'].to_numpy(), dfGRDC['Q'].to_numpy())
                        print (stationID, riverName, kge)                        

class GRDC():
    def __init__(self, region='north_america') -> None:
        self.region = region
        self.dfStation = getStationMeta(getCatalog(self.region))
        #get selected list (min area = 1.2e5 km2)
        self.validStationDF = pd.read_csv(os.path.join('data', 'GRDC_gage_list.csv'))
        print (self.validStationDF.shape)
        
        # self.validStationDF = pd.merge(
        #     self.dfStation,
        #     self.validStationDF,
        #     how="inner",
        #     on='grdc_no',
        #     sort=True,
        #     suffixes=("_x", "_y"),
        #     copy=True,
        #     indicator=False,
        #     validate=None,
        # )
        print (self.validStationDF)

    def plotValidStations(self):
        import cartopy.crs as ccrs
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.axes as maxes

        gdf = gpd.GeoDataFrame(
            self.validStationDF, 
            geometry=gpd.points_from_xy(self.validStationDF.lon, self.validStationDF.lat), 
            crs="EPSG:4326"
        )

        extents = getRegionBound(region=self.region)
        fig = plt.figure(figsize=(12,12))    
        crs_new = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=crs_new)
    
        gdf.plot(ax=ax, alpha=0.7, legend=True)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))    

        world.plot(ax=ax, alpha=0.5, facecolor="none", edgecolor='black')
        lon0,lat0,lon1,lat1 = extents
        ax.set_extent((lon0, lon1, lat0, lat1), crs_new)
        plt.savefig(f'grdc_{self.region}.png')
        plt.close()

def main():
    #do GRFR comparsion
    itask = 2
    if itask ==1:
        grfr = GRFR() 
        #grfr.compareGRDCToGRFR()
        grfr.compareGLOFASToGRFR()   
    elif itask==2:
        grdc = GRDC()
        grdc.plotValidStations()
if __name__ == '__main__':
    main()
