#author: alex sun
#date: 07/08/2023
#purpose: extract glofas flow series
#note: glofas region bounds are different from those used for grdc
#=========================================================================================
import pandas as pd
import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

from grdc import getStationMeta, getRegionBound, getGloFASStations, getBasinData,getCSR5dLikeArray,calculateMetrics
import xarray as xr
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from shapely.geometry import Point
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


regiondict = {
    'north_america': 'na',
    'south_america': 'sa',
    'europe': 'eu',
    'africa': 'af',
    'south_pacific': 'au',
    'asia': 'as',
}

def loadGLOFAS4Region(region):
    """Load netcdf files combined from GRIB 
    """
    rootdir = '/home/suna/work/predcsr5d/data/glofas'
    ncfile = os.path.join(rootdir, 'combined_ncfiles', f'glofas_{regiondict[region]}.nc')
    ds = xr.open_dataset(ncfile)

    return ds

def getCatalog(cfg, region):
    rootdir = f'/home/suna/work/predcsr5d/data/glofas'
    geosonfile = os.path.join(rootdir, f'{regiondict[region]}/stationbasins_{regiondict[region]}.geojson')

    gdf = gpd.read_file(geosonfile)
    gdf['grdc_no'] = pd.to_numeric(gdf['grdc_no'], downcast='integer')
    #filter stations that satisfy the min area requirements 
    cutoff_area = cfg.data.min_basin_area
    gdf = gdf[gdf['area_hys']>=cutoff_area]
    return gdf

def extractBasinOutletQ(loc, region, ds=None, startDate='2002-01-01', endDate='2020-12-31'):
    """Load GloFAS output for given location
    note: glofas cell is 0.05x0.05
    Param
    -----
    loc, (lat, lon) tuple
    """
    def getSingleSeries(da, lon, lat):
        daQ = da.sel(lon=slice(lon-lon_toler,lon+lon_toler), lat=slice(lat-lat_toler, lat+lat_toler))
        return daQ        

    lat,lon = loc
    if ds is None:
        ds = loadGLOFAS4Region(region)

    da = ds['dis24']    
    if 'longitude' in da.dims:
        da = da.rename({'longitude':'lon', 'latitude':'lat'})
    
    lon0, lat0, lon1, lat1 =  getRegionBound(region, source='glofas')
    nIter = 3 #if number of trials is greater than nIter than abandon the gage
    #test if the point is in the extent
    if  lat0<=lat<=lat1 and lon0<=lon<=lon1:
        #Extract Q [glofas cell size is 0.05] 
        lat_toler = 0.02   
        lon_toler = 0.02
        daQ = getSingleSeries(da, lon, lat)
        counter=0
        abandon = False
        while len(daQ['lat'])==0: 
            print ('not finding cell, increasing lat toler')
            lat_toler+=0.01
            daQ = getSingleSeries(da, lon, lat)
            counter+=1
            if counter>=nIter:
                abandon=True
                break

        if not abandon:
            counter=0        
            while len(daQ['lon'])==0:
                print ('not finding cell, increasing lon toler')
                lon_toler+=0.01
                daQ = getSingleSeries(da, lon, lat)       
                counter+=1
                if counter>=nIter:
                    abandon=True
                    break

        if not abandon:
            #handle the situation with multiple cells
            daQ = daQ.sel(time=slice(startDate,endDate))    
            arr = daQ.values
            ix = np.unravel_index(arr.argmax(), arr.shape) #this returns unflattened index tuple
            if len(daQ.lat)>1 and len(daQ.lon)<=1: #2D array
                daQ = daQ.isel(lat=ix[1])
            elif len(daQ.lon)>1 and len(daQ.lat)<=1: #2D array
                daQ = daQ.isel(lon=ix[1])
            elif len(daQ.lon)>1 and len(daQ.lat)>1: #3D array
                daQ = daQ.isel(lon=ix[2], lat=ix[1])
            daQ = daQ.squeeze()
            print ('=======max streamflow for ', daQ.max(dim='time').values, '===============')
            print ('=======min streamflow for ', daQ.min(dim='time').values, '===============')
            return daQ
        else:
            return None
    else:
        return None

def main(config, reGen=False):

    if reGen:
        allScores = {}
        xds = None
        for region in config.data.regions_glofas:
            gdf = getCatalog(config, region=region) #this includes all basin polygons

            dfStation = getGloFASStations(gdf)

            print (region, dfStation.shape)
            ds = loadGLOFAS4Region(region = region)

            for ix, row in dfStation.iterrows():
                stationID = row['grdc_no']
                #!!! change to use GloFAS lat/lon
                lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])
                daGloFAS = extractBasinOutletQ(loc=(lat,lon), ds=ds, region=region)
                if not daGloFAS is None:
                    if xds is None:
                        #only store xds on first call
                        basinTWS, basinP, xds, xds_Precip = getBasinData(config=config, 
                                stationID=stationID, 
                                lat=lat, lon=lon, gdf=gdf, 
                                region=region)
                    else:
                        basinTWS, basinP, _, _ = getBasinData(config=config, 
                                stationID=stationID, \
                                region=region, gdf=gdf, lat=lat, lon=lon, \
                                xds=xds, xds_Precip=xds_Precip)


                    kwargs = {
                        "method": "rx5d",
                        "aggmethod":'max',
                        "name":'Q'
                        }

                    daGloFAS = getCSR5dLikeArray(daGloFAS, basinTWS, **kwargs)
                    #06262023, normalize by drainage area
                    daGloFAS = daGloFAS/row['area_hys']

                    varDict={'TWS':basinTWS, 'P':basinP, 'Qs': daGloFAS}
                    
                    metricDict = calculateMetrics(config, stationID, varDict, onGloFAS=True)

                    print (f"{stationID},{lat},{lon},{row['river_x']},{row['area_hys']}") # 'MI', {metricDict['mi']}, 'CMI', {metricDict['cmi']}")
                    print (f"CMI {metricDict['cmi']}")

                    allScores[stationID] = metricDict
                else:
                    print (stationID, 'abandoned')
        pkl.dump(allScores, open(f'grdcresults/GLOFAS_all_{config.event.cutoff}.pkl', 'wb'))
    else:
        allScores = pkl.load(open(f'grdcresults/GLOFAS_all_{config.event.cutoff}.pkl', 'rb'))
    return allScores

def plotBivariate(cfg, use_percentile=False):
    """Reference: https://waterprogramming.wordpress.com/2022/09/08/bivariate-choropleth-maps/
    This plots global map on GloFAS gauges
    Note: it's important to keep the region order in config.yaml to get all plots
    otherwise, south_america and asia are not plotted right.
    
    Parameters

    """
    import seaborn as sns
    # Use geopandas for vector data and xarray for raster data
    import geopandas as gpd
    import rioxarray as rxr
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import ListedColormap
    from PIL import ImageColor
    from generativepy.color import Color
    from matplotlib.colors import rgb2hex
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection    

    #keep variable2 for legacy reasons    
    variable2='P_Q'
    dictkey = 'P'

    def plotRegion(ax ):

        #get GRDC station geopdf
        gdf = getCatalog(cfg, region=region) #this includes all basin polygons
        dfStation = getGloFASStations(gdf)

        dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
        dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
        print ('dfStation shape for region', region, dfStation.shape)
        counter=0
        ns = 0
        #note: need to make the last percentile inclusive, otherwise two points will be missing
        for i in range(len(cd)-1):
            for j in range(len(jd)-1):
                if i== len(cd)-2 and j==len(jd)-2:
                    groupDF = dfStation[(dfStation[variable2]>=cd[i]) & (dfStation['Q_TWS']>=jd[j]) ]
                elif i== len(cd)-2:
                    groupDF = dfStation[(dfStation[variable2]>=cd[i]) & (dfStation['Q_TWS']>=jd[j]) & (dfStation['Q_TWS']<jd[j+1])]
                elif j==len(jd)-2:
                    groupDF = dfStation[(dfStation[variable2]>=cd[i]) & (dfStation[variable2]<cd[i+1]) & (dfStation['Q_TWS']>=jd[j]) ]
                else:
                    groupDF = dfStation[(dfStation[variable2]>=cd[i]) & (dfStation[variable2]<cd[i+1]) & (dfStation['Q_TWS']>=jd[j]) & (dfStation['Q_TWS']<jd[j+1])]

                gdfCol = gpd.GeoDataFrame(groupDF[[variable2, 'Q_TWS']], geometry=gpd.points_from_xy(groupDF.long, groupDF.lat))
                gdfCol.plot(column=variable2, ax=ax, color=colorlist[counter], marker='o', markersize= 80, alpha=1.0,
                            edgecolor='#546E7A', legend = False)
                #for checking plots
                #for ix,row in gdfCol.iterrows():
                #    ax.annotate(text=f"{row[variable2]:4.2}, {row['Q_TWS']:4.2}", xy=row.geometry.centroid.coords[0], horizontalalignment='center')
                ns+=len(groupDF)
                counter+=1
        print ('total processed', ns)

    def hex_to_Color(hexcode):
        ### function to convert hex color to rgb to Color object (generativepy package)
        rgb = ImageColor.getcolor(hexcode, 'RGB')
        rgb = [v/256 for v in rgb]
        rgb = Color(*rgb)
        return rgb
    
    def genColorList(num_grps):
        ### get corner colors from https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/
        c00 = hex_to_Color('#D5F5E3') #light green
        c10 = hex_to_Color('#58D68D') #deep green 
        c01 = hex_to_Color('#F1948A') #light orange
        c11 = hex_to_Color('#E74C3C')
        ### now create square grid of colors, using color interpolation from generativepy package
        c00_to_c10 = []
        c01_to_c11 = []
        colorlist  = []
        for i in range(num_grps):
            c00_to_c10.append(c00.lerp(c10, 1/(num_grps-1) * i))
            c01_to_c11.append(c01.lerp(c11, 1/(num_grps-1) * i))
        
        for i in range(num_grps):
            for j in range(num_grps):
                colorlist.append(c00_to_c10[i].lerp(c01_to_c11[i], 1/(num_grps-1) * j))
        
        ### convert back to hex color
        colorlist = [rgb2hex([c.r, c.g, c.b]) for c in colorlist]
        return colorlist

    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    else:
        raise ValueError("invalid shape file")
    gdfHUC2 = gpd.read_file(shpfile)

    rootdir = '/home/suna/work/predcsr5d'

    # This can be converted into a `proj4` string/dict compatible with GeoPandas    
    fig,ax = plt.subplots(1,1, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    #DEFINE plot extent
    myextent = (-160,180,-50, 60)
    lon0,lon1,lat0,lat1 = myextent
    ax.set_extent(myextent,  crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', color='k', alpha=0.8)

    plotClimateMap = False
    if plotClimateMap:
        divider = make_axes_locatable(ax)        
        cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax)
        #==========plot climate regions=============
        koppenmap = rxr.open_rasterio(os.path.join(rootdir, 'data/koppenclimate/koppen5.tif'), masked=True)  
        print("The CRS for this data is:", koppenmap.rio.crs)
        print("The spatial extent is:", koppenmap.rio.bounds())
        
        koppenmap = koppenmap.sortby(koppenmap.y)
        koppenmap_region = koppenmap.sel(y=slice(lat0,lat1),x=slice(lon0,lon1))

        #
        #plot the Koppen climate map
        #note how colorbar is set up: level is 1 more than number of categories
        #
        cmap = sns.color_palette(["#85C1E9", "#FFD54F", "#DCEDC8", "#CFD8DC", "#E1BEE7"])
        im = koppenmap_region.plot(levels=[1, 2, 3, 4, 5, 6], colors=cmap, alpha=0.60, ax=ax, add_colorbar=False)
        print ('finished plotting climatemap')

    gdfHUC2.plot(ax=ax, facecolor="none", edgecolor="lightsteelblue", legend=False)       

    ax.set_title('')    
    ax.set_xlabel('')
    ax.set_ylabel('')

    #remove the lat/lon ticks and tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5) 

    #outputDict = pkl.load(open(f'grdcresults/GLOFAS_all_{cfg.event.cutoff}.pkl', 'rb'))            
    outputDict2 = pkl.load(open(f'grdcresults/glofas_all_events.pkl', 'rb'))
    validStations = list(outputDict2.keys())
    print(len(validStations))
    cmi=[]
    for stationID in validStations:        
        stationDict = outputDict2[stationID]                            
        cmi.append(stationDict[dictkey])       

    print ('no cmi', len(cmi))
    arr=[]
    for stationID in validStations:
        stationDict = outputDict2[stationID]            
        arr.append(stationDict['TWS'])  
    print ('no arr', len(arr))

    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), variable2:np.array(cmi), 'Q_TWS': arr})

    #should I load the GRDC numbers?
    if use_percentile:
        cd = np.percentile(cmi, [0, 25, 50, 75, 100])
        jd = np.percentile(arr, [0, 25, 50, 75, 100])
    else:
        cd = [0, 0.12, 0.2, 0.3, 0.5] #[0, 0.125, 0.25, 0.375, 0.5]
        jd = [0, 0.12, 0.2, 0.3, 0.5] #[0, 0.125, 0.25, 0.375, 0.5]

    print (np.percentile(cmi, [0, 25, 50, 75, 100]))
    print (np.percentile(arr, [0, 25, 50, 75, 100]))
    
    colorlist = genColorList(len(cd)-1)


    if plotClimateMap:
        #make color bar
        bounds = np.arange(0.5, 6.5)
        cb = plt.colorbar(im, cax=cax, ticks=bounds, orientation="horizontal", shrink=0.7)
        #cb.set_ticks([bounds[0]] + [(b0 + b1) / 2 for b0, b1 in zip(bounds[:-1], bounds[1:])] + [bounds[-1]])
        cb.set_label(label='')
        cb.ax.tick_params(labelsize=15,length=0)
        cb.ax.set_xticklabels(["", 'Tropical', 'Arid', 'Temperate', 'Cold', 'Polar'])

    for region in cfg.data.regions_glofas:
        plotRegion(ax)
        ### now create inset legend
        if region=='north_america':
            if use_percentile:
                percentile_bounds = [25, 50, 75, 100]
            else:
                percentile_bounds = [0.12, 0.2, 0.3, 0.5]

            ax = ax.inset_axes([0.0,0.12,0.33,0.33]) #x,y, W, H
            ax.set_aspect('equal', adjustable='box')
            count = 0
            xticks = [0]
            yticks = [0]
            for i,percentile_bound_p1 in enumerate(percentile_bounds):
                for j,percentile_bound_p2 in enumerate(percentile_bounds):
                    percentileboxes = [Rectangle((i,j), 1, 1)]
                    pc = PatchCollection(percentileboxes, facecolor=colorlist[count], alpha=0.85)
                    count += 1
                    ax.add_collection(pc)
                    if i == 0:
                        yticks.append(percentile_bound_p2)
                xticks.append(percentile_bound_p1)

            _=ax.set_xlim([0,len(percentile_bounds)])
            _=ax.set_ylim([0,len(percentile_bounds)])
            _=ax.set_xticks(list(range(len(percentile_bounds)+1)), xticks)
            _=ax.set_xlabel(variable2, fontsize=15)
            _=ax.set_yticks(list(range(len(percentile_bounds)+1)), yticks)
            _=ax.set_ylabel('TWS_Q', fontsize=15)
    
    plt.savefig(f'outputs/global_glofas_bivariate_{dictkey}.eps')
    plt.close()


if __name__ == '__main__':    
    from myutils import load_config
    config = load_config('config.yaml')

    reGen=True
    itask = 2
    if itask == 1:
        #generate global glofas metrics
        main(config, reGen=reGen)
    elif itask == 2:
        #figure 2A, plot bivariate plot for GLOFAS
        plotBivariate(config)

