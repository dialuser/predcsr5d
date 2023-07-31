#author: Alex Sun
#date: 06182023
#date: 07202023, reviewed likelihood multiplication factor calculation
#date: 07232023, not using LMF, revised JRP calculation
#                added SI FPI implementation
#========================================================================================
import pandas as pd
import numpy as np
import xarray as xr
import sys,os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib.pyplot as plt
from  tqdm import tqdm
import pickle as pkl
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from datetime import datetime
import calendar

from grdc import getCatalog, getStationMeta,readStationSeries,getBasinData,getExtremeEvents
from grdc import getGloFASStations
from dataloader_global import load5ddatasets,loadClimatedatasets,getCSR5dLikeArray
import glofas_us
from glofas_us import loadGLOFAS4CONUS,extractBasinOutletQ

def getCompoundEvents(cfg, daQ, daP, daTWS, daFPI=None):
        #=============start extreme event analysis============
        def removeBadYears(da):
            da1 = da.sel(time=slice(cfg.data.start_date, '2016/12/31'))
            da2 = da.sel(time=slice('2019/01/01', cfg.data.end_date))
            da = xr.concat([da1,da2], dim='time')
            return da

        def calculateProb(ts1,ts2):
            nEvents = 0
            for ix, time1 in enumerate(ts1.index.tolist()):
                for iy, time2 in enumerate(ts2.index.tolist()):
                    dt = (time1-time2).days
                    if abs(dt)<=cfg.event.t_win:
                        nEvents+=1
            return nEvents

        def genTS(arrIn, K, V):
            #generate time series for 5-day uniform intervals
            arrOut = np.zeros(daterng.shape) + np.NaN
            arrOut[K] = arrIn[V]
            return arrOut

        eventMethod = cfg.event.event_method
        if eventMethod == 'POT':
            cutoff = cfg.event.cutoff
        else:
            cutoff = None
        
        daQ = removeBadYears(daQ)
        daP = removeBadYears(daP)
        daTWS = removeBadYears(daTWS)


        timestamps = daQ.time.values
        min_dist = cfg.event.t_win

        #as 07/11/2023, change Q, TWS, and P to uniform 5-day intervals
        daterng = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='5D')    
        #get key/value pairs to map from daterng to timestamps
        TWS = daTWS.values
        P = daP.values
        Q = daQ.values
        if not daFPI is None:
            FPI = daFPI.values
        K = []
        V = []
        for iy, t1 in enumerate(timestamps):
            for ix, t2 in enumerate(daterng):
                if abs((t1-t2).days)<3.0 and ~np.isnan(Q[iy]):
                    K.append(ix)
                    V.append(iy)  
                    break        
        K = np.array(K)
        V = np.array(V)
        #assign values to 5-day dates, missing values are indicated by NaN
        TWS = genTS(TWS, K, V)
        P = genTS(P, K, V)     
        Q = genTS(Q, K, V)
        if not daFPI is None:
            FPI = genTS(FPI, K, V)

        print ('Q len', Q.shape, len(daterng))
        _, tsQ   = getExtremeEvents(pd.Series(Q.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist, transform=False, returnExtremeSeries=True)        
        _,tsTWS  = getExtremeEvents(pd.Series(TWS.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist, transform=False, returnExtremeSeries=True)
        _, tsP   = getExtremeEvents(pd.Series(P.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist,transform=False,  returnExtremeSeries=True)
        _, tsFPI = getExtremeEvents(pd.Series(FPI.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist,transform=False,  returnExtremeSeries=True)
        
        
        #find co-occurrence
        #note tsQ and tsTWS only have extreme events in it
        #
        TWS_Q =  calculateProb(tsQ, tsTWS)
        P_Q   =  calculateProb(tsQ, tsP)
        #print ('Joint TWS_Q', TWS_Q, ' Joint P_Q ', P_Q)
        p_TWS_Q =  TWS_Q /len(tsQ)
        p_P_Q   =  P_Q /len(tsQ)
        SI_TWS = getSeasonalityIndex(tsTWS)
        SI_P = getSeasonalityIndex(tsP)
        SI_Q = getSeasonalityIndex(tsQ)
        SI_FPI = getSeasonalityIndex(tsFPI)
        return {'TWS':p_TWS_Q, 'P': p_P_Q, 
                'SI_TWS': SI_TWS, 'SI_P': SI_P, 
                'SI_Q': SI_Q,
                'SI_FPI': SI_FPI,
                'tsQ':tsQ, 'tsTWS':tsTWS, 'tsP':tsP}

def getSeasonalityIndex(ts):
    """Calcualte seasonality index
    cf: https://journals.ametsoc.org/view/journals/hydr/18/7/jhm-d-16-0207_1.xml
    """
    event_dates = ts.index.values
    theta=[]
    for item in event_dates:
        #get day of the year
        adate = pd.to_datetime(item)
        day_of_year = adate.timetuple().tm_yday
        year = adate.year
        days_in_year = 365 + calendar.isleap(year)
        theta.append(day_of_year*(np.pi*2/days_in_year))
    theta = np.array(theta)
    xbar = np.mean(np.cos(theta))
    ybar = np.mean(np.sin(theta))
    theta_bar = np.arctan(ybar/xbar)
    SI = np.sqrt(xbar*xbar+ybar*ybar)
    mean_date = theta_bar*np.pi*2/365
    return SI

def compound_events_analysis(cfg, region, plotStations=False):
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)
    reGen = cfg.regen
    if reGen:
        xds = None
        daterng = pd.date_range(start=cfg.data.start_date, end=cfg.data.end_date, freq='1D')
        threshold = cfg.data.cutoff_treshold #means tolerate 10% missing data during study period
        #print ("stationid,lat,lon,river,area,data_coverage")
        allEvents = {}
        for ix, row in dfStation.iterrows():
            stationID = row['grdc_no']
            riverName = row['river_x']
            lat,lon = float(row['lat']), float(row['long'])
            try:
                df = readStationSeries(stationID=stationID, region=region)
                df = df[df.index.year>=2002].copy(deep=True)
                
                #after this step, NaNs will exist when data is missing!!!
                df = df.reindex(daterng)

                #count number of valid values
                #only process gages have sufficient number of data
                #data_coverage = 1- df.isnull().sum().values/len(daterng)
                #if data_coverage>threshold and row['area_hys']>=cfg.data.min_basin_area:

                #count number of valid values in each year
                res = df.groupby(df.index.year).agg({'count'})/365.0
                resdf = res['Q']
                resdf = resdf[resdf['count']<cfg.data.year_threshold]

                #only process gages have sufficient number of data and big enough area
                if resdf.empty and row['area_hys']>=cfg.data.min_basin_area:
                    #drop NaN
                    df = df.dropna()
                    daQ = xr.DataArray(data=df['Q'].values/row['area_hys'], dims=['time'], coords={'time':df.index})

                    if xds is None:
                        #only store xds on first call
                        basinTWS, basinP, basinFPI, xds, xds_Precip, xdsC = getBasinData(config=cfg, stationID=stationID, 
                                                                        lat=lat, lon=lon, gdf=gdf, 
                                                                        region=region,
                                                                        returnFPI=True)
                    else:
                        basinTWS, basinP, basinFPI, _, _,_ = getBasinData(config=cfg, stationID=stationID, \
                                                            region=region, gdf=gdf, lat=lat, lon=lon, \
                                                            xds=xds, xds_Precip=xds_Precip,
                                                            returnFPI=True, xdsC=xdsC)

                    kwargs = {
                        "method": "rx5d",
                        "aggmethod":'max',
                        "name":'Q'
                        }
                    #convert Q to CSR5d intervals
                    #this steps may introduce NaN values
                    daQ = getCSR5dLikeArray(daQ, basinTWS, **kwargs)
                    resDict = getCompoundEvents(cfg, daQ, basinP, basinTWS, basinFPI)
                    
                    if plotStations:
                        #as0715, generate station event plots
                        fig = plt.figure(figsize=(8,6))

                        gs = fig.add_gridspec(2, 1,  height_ratios=(1, 4), 
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.15, hspace=0.0)            
                        ax0 = fig.add_subplot(gs[0,0])
                        ax1 = fig.add_subplot(gs[1,0])
                        ax2 = ax1.twinx()
                        basinP.plot(ax=ax0, color='gray')
                        basinTWS.plot.line(ax=ax1, color='green')
                        daQ.plot(ax = ax2, color='r')
                        
                        tsQ  = resDict['tsQ'] 
                        tsTWS= resDict['tsTWS']
                        tsP  = resDict['tsP']

                        tsQ.plot.line(linestyle='none', marker='o', ax=ax2, color='tab:red')
                        tsTWS.plot.line(linestyle='none', marker='o', ax=ax1, color='tab:green')
                        tsP.plot.line(linestyle='none', marker='o', ax=ax0, color='tab:gray')

                        ax0.set_title(f"{stationID}, {row['river_x']}")
                        ax1.set_title('')
                        ax2.set_title('')

                        ax2.yaxis.label.set_color('tab:red')
                        ax1.yaxis.label.set_color('tab:green')
                        ax2.yaxis.label.set_fontsize(15)
                        ax1.yaxis.label.set_fontsize(15)


                        plt.savefig(f'outputs/qplot_{stationID}.png')
                        plt.close()

                    #print (stationID, riverName, resDict)
                    print (stationID, riverName, resDict['SI_Q'])

                    allEvents[stationID] = resDict       
            except Exception as e: 
                raise Exception (e)
        pkl.dump(allEvents, open(f'grdcresults/{region}_all_events.pkl', 'wb'))             
    else:
        allEvents = pkl.load(open(f'grdcresults/{region}_all_events.pkl', 'rb'))   

def getUniformProb(nYear, period=365):
    def form_ts():
        arr = []
        lo = 0
        hi = 365
        for i in range(nYear):
            arr.append(rng.integers(low=lo, high=hi+1, size=1)[0])
            lo=hi
            hi+=365
        return arr
    rng = np.random.default_rng(seed=10)            
    nRep = 1000000 # 1000000
    cooccur = []
    for i in tqdm(range(nRep)):
        #form ts
        nEvent=0
        ts1 = sorted(form_ts())
        ts2 = sorted(form_ts())
        for time1 in ts1:
            for time2 in ts2:
                if abs(time1-time2)<10:
                    nEvent+=1
        cooccur.append(nEvent/nYear)
    print ('prob ', np.mean(np.array(cooccur)), np.std(np.array(cooccur)))
    #for 18 years, prob  0.05208983333333336 0.0524393641992973
    #for 16 years, this print out prob  0.051972375 0.055471181025460184

def glofas_compound_events_analysis(cfg, region):
    gdf = getCatalog(region, 'glofas')
    if 'area' in gdf.columns:
        gdf = gdf.rename(columns={'area':'area_hys'})    
    #0908 replace with GloFAS stations
    dfStation = getGloFASStations(gdf)
    reGen = cfg.regen
    if reGen:
        dsGloFAS = loadGLOFAS4CONUS()
        extents = glofas_us.getExtents()
        daterng = pd.date_range(start='2002/01/01', end='2020/12/31', freq='1D')
        xds = None
        allEvents = {}
        for ix, row in dfStation.iterrows():
            stationID = row['grdc_no']
            riverName = row['river_x']
            lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])

            try:
                #count number of valid values
                lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])
                print (lat, lon, row['area_hys'])

                if row['area_hys']>=cfg.data.min_basin_area and (lat>extents['min_lat'] and lat<extents['max_lat']):
                    if lon>extents['min_lon'] and lon<=extents['max_lon']:
                        stationID = row['grdc_no']
                        riverName = row['river_x']
                        print ('stationID', stationID, riverName)

                        #get glofas data between 2002 and 2020                
                        daGloFAS = extractBasinOutletQ(loc=(lat,lon), ds=dsGloFAS)
                        daGloFAS = daGloFAS.squeeze() #this needs to be series 
                        #========normalization==================
                        daGloFAS = daGloFAS/row['area_hys']

                        if xds is None:
                            #only store xds on first call
                            basinTWS, basinP, xds, xds_Precip = getBasinData(config=cfg, stationID=stationID, 
                                                                            lat=lat, lon=lon, gdf=gdf, 
                                                                            region=region)
                        else:
                            basinTWS, basinP, _, _ = getBasinData(config=cfg, stationID=stationID, \
                                                                region=region, gdf=gdf, lat=lat, lon=lon, \
                                                                xds=xds, xds_Precip=xds_Precip)

                        kwargs = {
                            "method": "rx5d",
                            "aggmethod":'max',
                            "name":'Q'
                            }
                        #convert Q to CSR5d intervals
                        #this steps may introduce NaN values
                        daGloFAS = getCSR5dLikeArray(daGloFAS, basinTWS, **kwargs)
                        resDict = getCompoundEvents(cfg, daGloFAS, basinP, basinTWS)
                        print (stationID, riverName, resDict)
                        allEvents[stationID] = resDict       
            except Exception as e: 
                raise Exception (e)
        pkl.dump(allEvents, open(f'grdcresults/glofas_{region}_all_events.pkl', 'wb'))             
    else:
        allEvents = pkl.load(open(f'grdcresults/glofas_{region}_all_events.pkl', 'rb'))   



def glofas_compound_events_analysis_global(cfg):
    from glofas_all import getCatalog as getGloFASCatalog,loadGLOFAS4Region,extractBasinOutletQ
    from grdc import getRegionBound
    
    reGen = cfg.regen
    if reGen:
        allEvents = {}
        for region in config.data.regions_glofas:
            print ('='*80)
            print (region.upper())

            gdf = getGloFASCatalog(config, region=region) #this includes all basin polygons

            if 'area' in gdf.columns:
                gdf = gdf.rename(columns={'area':'area_hys'})    

            #0908 replace with GloFAS stations
            dfStation = getGloFASStations(gdf)
            #load glofas nc file
            dsGloFAS = loadGLOFAS4Region(region = region)
            #get extent for each region
            lon0, lat0, lon1, lat1 =  getRegionBound(region, source='glofas')

            xds = None            
            for ix, row in dfStation.iterrows():
                stationID = row['grdc_no']
                riverName = row['river_x']
                try:
                    #count number of valid values
                    lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])
                    print (lat, lon, row['area_hys'])

                    if row['area_hys']>=cfg.data.min_basin_area and (lat0<=lat<lat1):
                        if lon0<=lon<lon1:
                            stationID = row['grdc_no']
                            riverName = row['river_x']
                            print ('stationID', stationID, riverName)

                            #get glofas data between 2002 and 2020                
                            daGloFAS = extractBasinOutletQ(loc=(lat,lon), ds=dsGloFAS, region=region)

                            #========normalization==================
                            daGloFAS = daGloFAS/row['area_hys']

                            if xds is None:
                                #only store xds on first call
                                basinTWS, basinP, xds, xds_Precip = getBasinData(config=cfg, stationID=stationID, 
                                                                                lat=lat, lon=lon, gdf=gdf, 
                                                                                region=region)
                            else:
                                basinTWS, basinP, _, _ = getBasinData(config=cfg, stationID=stationID, \
                                                                    region=region, gdf=gdf, lat=lat, lon=lon, \
                                                                    xds=xds, xds_Precip=xds_Precip)

                            kwargs = {
                                "method": "rx5d",
                                "aggmethod":'max',
                                "name":'Q'
                                }
                            #convert Q to CSR5d intervals
                            #this steps may introduce NaN values
                            daGloFAS = getCSR5dLikeArray(daGloFAS, basinTWS, **kwargs)
                            resDict = getCompoundEvents(cfg, daGloFAS, basinP, basinTWS)
                            print (stationID, riverName, resDict)
                            allEvents[stationID] = resDict       
                except Exception as e: 
                    raise Exception (e)
        pkl.dump(allEvents, open(f'grdcresults/glofas_all_events.pkl', 'wb'))             
    else:
        allEvents = pkl.load(open(f'grdcresults/glofas_all_events.pkl', 'rb'))   


def getCompoundEventCopula(cfg, daQ, daP, daTWS):
        #=============start extreme event analysis============
        def removeBadYears(da):
            da1 = da.sel(time=slice(cfg.data.start_date, '2016/12/31'))
            da2 = da.sel(time=slice('2019/01/01', cfg.data.end_date))
            da = xr.concat([da1,da2], dim='time')
            return da

        def calculateProb(ts1,ts2):
            nEvents = 0
            for ix, time1 in enumerate(ts1.index.tolist()):
                for iy, time2 in enumerate(ts2.index.tolist()):
                    dt = (time1-time2).days
                    if abs(dt)<=cfg.event.t_win:
                        nEvents+=1
            return nEvents

        def genTS(arrIn, K, V):
            #generate time series for 5-day uniform intervals
            arrOut = np.zeros(daterng.shape) + np.NaN
            arrOut[K] = arrIn[V]
            return arrOut

        eventMethod = cfg.copula.event_method
        if eventMethod == 'POT':
            cutoff = cfg.copula.threshold
        else:
            cutoff = None
        
        daQ = removeBadYears(daQ)
        daP = removeBadYears(daP)
        daTWS = removeBadYears(daTWS)
        timestamps = daQ.time.values
        min_dist = cfg.event.t_win

        #as 07/11/2023, change Q, TWS, and P to uniform 5-day intervals
        daterng = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='5D')    
        #get key/value pairs to map from daterng to timestamps
        TWS = daTWS.values
        P = daP.values
        Q = daQ.values
        K = []
        V = []
        for iy, t1 in enumerate(timestamps):
            for ix, t2 in enumerate(daterng):
                if abs((t1-t2).days)<3.0 and ~np.isnan(Q[iy]):
                    K.append(ix)
                    V.append(iy)  
                    break        
        K = np.array(K)
        V = np.array(V)
        #assign values to 5-day dates, missing values are indicated by NaN
        TWS = genTS(TWS, K, V)
        P = genTS(P, K, V)     
        Q = genTS(Q, K, V)

        _, tsQ   = getExtremeEvents(pd.Series(Q.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist, transform=False, returnExtremeSeries=True)
        _, tsTWS = getExtremeEvents(pd.Series(TWS.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist, transform=False, returnExtremeSeries=True)
        _, tsP   = getExtremeEvents(pd.Series(P.squeeze(), index=daterng), method=eventMethod, cutoff=cutoff, minDist=min_dist,transform=False,  returnExtremeSeries=True)

        #find co-occurrence
        #note tsQ and tsTWS only have extreme events in it
        #
        TWS_Q =  calculateProb(tsQ, tsTWS)
        P_Q   =  calculateProb(tsQ, tsP)
        print ('Joint TWS_Q', TWS_Q, ' Joint P_Q ', P_Q)
        p_TWS_Q =  TWS_Q /len(tsQ)
        p_P_Q   =  P_Q /len(tsQ)

        if eventMethod == 'POT':
            #for POT we want to fit copula on the whole time series, not extremes
            tempDF = pd.DataFrame(np.c_[TWS.squeeze(),Q.squeeze(),P.squeeze()], index=daterng)
            tempDF = tempDF.dropna().values
            return {'TWS':p_TWS_Q, 'P': p_P_Q, 
                    'esTWS': tempDF[:,0],
                    'esQ': tempDF[:,1], 
                    'esP': tempDF[:,2]
                    }
        else:
            return {'TWS':p_TWS_Q, 'P': p_P_Q, 'esTWS': tsTWS, 'esQ': tsQ, 'esP':tsP}

def copulaModel(cfg, region):
    """Calculate joint return period
    """
    import pyvinecopulib as pv
    #prepare data
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    xds = None
    daterng = pd.date_range(start=cfg.data.start_date, end=cfg.data.end_date, freq='1D')
    allEvents = {}

    for ix, row in dfStation.iterrows():
        stationID = row['grdc_no']
        lat,lon   = float(row['lat']), float(row['long'])
        try:
            df = readStationSeries(stationID=stationID, region=region)
            df = df[df.index.year>=2002].copy(deep=True)
            
            #after this step, NaNs will exist when data is missing!!!
            df = df.reindex(daterng)

            #count number of valid values
            #only process gages have sufficient number of data
            #data_coverage = 1 - df.isnull().sum().values/len(daterng)
            #if data_coverage>threshold and row['area_hys']>=cfg.data.min_basin_area:
            #count number of valid values in each year
            res = df.groupby(df.index.year).agg({'count'})/365.0
            resdf = res['Q']
            resdf = resdf[resdf['count']<cfg.data.year_threshold]

            #only process gages have sufficient number of data and big enough area
            if resdf.empty and row['area_hys']>=cfg.data.min_basin_area:
                print ("*******", stationID, "*******")
                #drop NaN
                df = df.dropna()
                daQ = xr.DataArray(data=df['Q'].values/row['area_hys'], dims=['time'], coords={'time':df.index})

                if xds is None:
                    #only store xds on first call
                    basinTWS, basinP, xds, xds_Precip = getBasinData(config=cfg, stationID=stationID, 
                                                                    lat=lat, lon=lon, gdf=gdf, 
                                                                    region=region)
                else:
                    basinTWS, basinP, _, _ = getBasinData(config=cfg, stationID=stationID, \
                                                        region=region, gdf=gdf, lat=lat, lon=lon, \
                                                        xds=xds, xds_Precip=xds_Precip)

                kwargs = {
                    "method": "rx5d",
                    "aggmethod":'max',
                    "name":'Q'
                    }
                #convert Q to CSR5d intervals
                #this steps may introduce NaN values
                daQ = getCSR5dLikeArray(daQ, basinTWS, **kwargs)
                Q = daQ.values
                TWS = basinTWS.values
                P = basinP.values
                ind = np.where(~np.isnan(Q))[0]
                Q = Q[ind]
                TWS = TWS[ind]
                P = P[ind]

                #Fit copula on annual maxima
                daQ = daQ.isel(time=ind)
                basinP = basinP.isel(time=ind)
                basinTWS = basinTWS.isel(time=ind)
                resDict = getCompoundEventCopula(cfg, daQ, basinP, basinTWS)
                
                Qa = resDict['esQ']
                TWSa = resDict['esTWS']
                Pa = resDict['esP']

                #
                #form data vector [there should be no nan in the data]
                controls = pv.FitControlsBicop(family_set=[pv.BicopFamily.student, 
                                                           pv.BicopFamily.gaussian, 
                                                           pv.BicopFamily.gumbel, 
                                                           pv.BicopFamily.clayton,
                                                           pv.BicopFamily.joe ],
                                                selection_criterion='aic')                                                                       
                if cfg.copula.event_method=='POT':
                    #convert from percentile to prob
                    thresh = cfg.copula.threshold/100.0
                else:
                    #margin defines prob of marginal variables
                    #so margin=0.2 means the event for the variable happens every 5 years  
                    thresh = 1.0-cfg.copula.margin
                
                #now fit bivariate copula using annual maxima
                data = np.c_[TWSa, Qa]
                #do TWS vs. Q                
                u = pv.to_pseudo_obs(data)

                cop2 = pv.Bicop(data=u, controls=controls)
                cuv = cop2.cdf([[thresh, thresh]])[0]

                JP_TWS =  1.0-2*thresh+cuv
                    
                print ('tws cuv', cuv)
                #do P vs. Q
                data = np.c_[Pa, Qa]
                u = pv.to_pseudo_obs(data)
                cop2 = pv.Bicop(data=u, controls=controls)
                cuv = cop2.cdf([[thresh, thresh]])[0]
                JP_P =  1.0-2*thresh+cuv
                if cfg.copula.event_method=='POT':
                    JP_TWS = JP_TWS*(365/5)
                    JP_P = JP_P*(365/5)
                print ('P cuv', cuv)
                #07/12/23 get co-occurrence probability from data 
                ##note this is obtained from running getUniformProb(), it is the random prob of yearly co-occurrence
                # during the 16-year study period
                p_random = 0.052 
                JProb_TWS_Q = np.max([resDict['TWS'], p_random])
                JProb_P_Q =   np.max([resDict['P'],   p_random])
                #
                #calculate joint return period using formula in 
                #Anaïs Couasnon 2020, natural hazard paper
                #Eq (4) 
                #Assuming independence between the two variables, these conditions would
                #return period for 5-year is = 1/(0.2x0.2*0.05) = 500
                #in the case of complete dependence, return period = 1/(0.2*1) = 5
                #RP_TWS = 1.0/(JProb_TWS_Q*JP_TWS)
                #RP_P = 1.0/(JProb_P_Q*JP_P)
                #asun 0723, switch to simple inverse of joint prob
                RP_TWS = 1.0/JP_TWS
                RP_P = 1.0/JP_P

                #likelihood multiplication factor (LMP)
                #from Ridder compound event paper [Eq 4 and 5]
                p_uni = cfg.copula.margin
                LMP_TWS = JProb_TWS_Q/(p_uni * p_uni)
                LMP_P = JProb_P_Q / (p_uni * p_uni)
                print ('joint occur prob TWS', JProb_TWS_Q, 'P ', JProb_P_Q)
                print ('copula based JRP: TWS ', RP_TWS, 'P ', RP_P)
                print ('LMP TWS', LMP_TWS, 'LMP P', LMP_P)
                allEvents[stationID] = {
                            'copula_tws': RP_TWS, 
                            'copula_p': RP_P, 
                            'LMP_TWS': LMP_TWS,
                            'LMP_P': LMP_P,
                }

        except Exception as e: 
            raise Exception (e)
    RT = int(1/cfg.copula.margin)
    pkl.dump(allEvents, open(f'grdcresults/copula_{region}_margin{RT}.pkl', 'wb'))

def plotJointReturn(cfg, ax, region, RT, var='TWS', cax=None):
    from matplotlib.colors import BoundaryNorm
    if region == 'north_america':
        if cfg.maps.global_basin == 'majorbasin':
            shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
        gdfMap = gpd.read_file(shpfile)
        #plot  boundaries
        #gdfMap.plot(ax=ax, facecolor="none", edgecolor="gray", legend=False)       

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)
    
    outputDict = pkl.load(open(f'grdcresults/copula_{region}_margin{RT}.pkl', 'rb'))        
    validStations = list(outputDict.keys())

    arr=[]
    
    for stationID in validStations:        
        stationDict = outputDict[stationID]
        if var=='TWS':             
            varname = 'TWS'               
            arr.append(stationDict['copula_tws'])
        else:
            varname = 'P'
            arr.append(stationDict['copula_p'])

    print ('len of stations', len(arr))
    print (var, np.nanmax(arr), np.nanmin(arr))
    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), varname:np.array(arr)})

    dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')

    cmap = ListedColormap((sns.color_palette("rocket", 8)).as_hex())
    gdfCol = gpd.GeoDataFrame(dfStation[varname], geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))

    vmax= int(1/(cfg.copula.margin*cfg.copula.margin*0.05))
    vmin = int(1/(cfg.copula.margin))

    if int(RT) == 1:
        bounds = [1,2,4,6,8,10,15,25]
        norm = BoundaryNorm(bounds, cmap.N)
    elif int(RT) == 5:
        #bounds = [5,10,20,40,60,100,200,500]
        bounds = [1,2,4,6,8,10,15,25]
        norm = BoundaryNorm(bounds, cmap.N)
    else:
        raise ValueError("Invalid RT")
    if cax is not None:
        gdfCol.plot(column=varname, ax=ax, cax=cax, cmap=cmap, norm=norm,marker='o', markersize=15, alpha=0.8,
            legend_kwds={'shrink': 0.5, 'extend':'max'}, 
            legend=True, vmin=vmin, vmax=vmax)                   
    else:
        gdfCol.plot(column=varname, ax=ax, cmap=cmap, marker='o', norm=norm, markersize= 15, alpha=0.8,
            legend=False, vmin=vmin, vmax=vmax)
    ax.text(0.01,0.03, f'{varname}', fontsize=14, transform=ax.transAxes)


if __name__ == '__main__':    
    from myutils import load_config
    config = load_config('config.yaml')
    
    itask = 4
    if itask ==0:
        #do bootstrapping
        getUniformProb(nYear=18)

    elif itask == 1:
        #do compound event analysis for grdc [for Figure 1]
        for region in config.data.regions:
            compound_events_analysis(cfg=config, region=region, plotStations=True)
    elif itask == 2:
        #do compound event analysis for glofas [this is not used anymore]
        #so far I only downloaded NA data
        glofas_compound_events_analysis(cfg=config, region="north_america")
    elif itask == 3:
        #do copula return period generation
        for region in config.data.regions:
            copulaModel(config, region)

    elif itask == 4:
        #this needs to be run after itask 3
        fig,axes = plt.subplots(1,2,figsize=(16,12),subplot_kw={'projection': ccrs.PlateCarree()})    
        
        ax0 = axes[0]
        ax1 = axes[1]

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax)

        divider2 = make_axes_locatable(ax1)
        cax2 = divider2.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax2)

        RT = int(1/config.copula.margin)
        for region in config.data.regions:
            im1 = plotJointReturn(config, region=region, ax=ax0, RT=RT, var='TWS',cax=None)
            im2 = plotJointReturn(config, region=region, ax=ax1, RT=RT, var='P', cax=cax2)
        
        ax0.set_extent([-180, 180, -60, 90])
        ax0.coastlines(resolution='110m', color='gray')
        ax1.set_extent([-180, 180, -60, 90])
        ax1.coastlines(resolution='110m', color='gray')

        ax0.set_xticklabels([])
        ax0.set_yticklabels([])        
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])        

        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f'outputs/copula_return_period{RT}.eps')

        plt.close()

    elif itask == 5:
        #figure 2
        glofas_compound_events_analysis_global(config)