#author: alex sun
#date: 7/1/2023
#purpose: load GSIM meta
import pandas as pd
import os
import numpy as np
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib.pyplot as plt

def getGSIMMeta():
    from shapely.geometry import Point
    cutoff = 120000 #km^2
    csvname = 'data/gsim/GSIM_metadata.csv'
    df = pd.read_csv(csvname, delimiter=',', usecols=['gsim.no', 'grdb.no', 'area', 'country', 'river', 'station', 'country', 'latitude', 'longitude', 'altitude'], na_values='NA')
    df = df[(df['area'].notna()) & (df['longitude'].notna()) & (df['latitude'].notna())]
    df = df[df['country'].isin(['US','CA'])]
    df = df.sort_values(by='area')
    df = df[df['area']>=cutoff]
    #convert to gdf
    df['coordinates'] = list(zip(df.longitude, df.latitude))
    df['coordinates'] = df['coordinates'].apply(Point)
    gdf = gpd.GeoDataFrame(df, geometry='coordinates')
    gdf = gdf.set_crs('epsg:4326')
    print (gdf.shape)    
    return gdf

def getRivID():
    def linestring_to_points(feature,line):
        return {feature:line.coords}

    grfr_root = '/home/suna/work/grfr'
    region = 7
    shpfile = os.path.join(grfr_root, f'riv_pfaf_{region}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp')
    gdf = gpd.read_file(shpfile)

    #gdf['points'] = gdf.apply(lambda l: linestring_to_points(l['COMID'], l['geometry']), axis=1)
    
    gdfPoints = getGSIMMeta()

    combined = gpd.sjoin_nearest(gdf, gdfPoints, distance_col='sdist', how='inner', max_distance=0.05)
    print (combined.shape)
    fig,ax = plt.subplots(2,1)
    gdfPoints.plot(ax=ax[0])
    combined.plot(ax=ax[1],column='COMID', color='b', marker='o') 
    plt.savefig('na_gsim.png')
    plt.close()   
if __name__ == '__main__':    
    from myutils import load_config

    getGSIMMeta()
    getRivID()