#author: Alex Sun
#date: 3/25/2022
#purpose: collection of utility functions
#=================================================================
import enum
from torch.autograd import Variable
import torch
import numpy as np
import xarray as xr
import seaborn as sns
from sklearn.metrics import confusion_matrix

from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
import yaml
from attrdict import AttrDict
import rioxarray

def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def load_config(file):
    """Load configuration from YAML file.
    Parameters
    ----------
    file : str
        Path to configuration file.
    Returns
    -------
    conf_dict : attrdict.AttrDict
        AttrDict containing configurations.
    """
    # read configuration files
    with open(file, "r") as f:
        conf_dict = AttrDict(yaml.safe_load(f))
    return conf_dict


def toTensor(nparr):
    """utility function for tensor conversion
    """
    return torch.FloatTensor(nparr)

def getGraceRoot():
    #return r'/work/02248/alexsund/maverick2/grace'
    return r'/home/suna/work/grace'

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def area_grid(lat, lon):
    """from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def getGridCellArea(refDA,weighttype='latbased'):
    """
    """
    #print (refDA.lat)
    #two choices, area based or lat based
    if weighttype=='areabased':
        dA = area_grid(refDA.lat, refDA.lon)
        totalArea = dA.sum(dim=['lat','lon'])
        weights = dA/totalArea
    elif weighttype=='latbased':        
        weights = getLatWeights(refDA.lat)
    """
    plt.figure()
    dA.plot()
    plt.savefig('testarea.png')
    plt.close()    
    sys.exit()
    """
    return weights

def getLatWeights(lats):
    weights = np.cos(np.deg2rad(lats))
    
    return weights

def applyLandMask(obj, landmask):
    if type(landmask) == xr.DataArray:
        landmask = landmask.values
    landmask[landmask==0] = np.NaN
    if type(obj)== xr.DataArray:
        if len(obj.shape)==3:
            tmp = np.einsum('kij,ij->kij', obj.values, landmask)
        else:
            tmp = np.einsum('ij,ij->ij', obj.values, landmask)
        return xr.DataArray(tmp, dims = obj.dims, coords = obj.coords, name= obj.name)
    else:
        if len(obj.shape)==3:
            tmp = np.einsum('kij,ij->kij', obj, landmask)
        else:
            tmp = np.einsum('ij,ij->ij', obj, landmask)
        return tmp

def scaleDA(da, mask, method='norm',op='train',**kwargs):
    arr = da.values
    mm= mask
    print (arr.shape, mm.shape)
    if method=='norm':
        if op=='train':
            arr = da.values
            mu = np.mean(arr, axis=0)
            stdvar = np.std(arr, axis=0)
            print ('99 percentile of std', np.percentile(stdvar.flatten(), 99))
            """
            fig,ax=plt.subplots(2,1)
            ax[0].imshow(mu, origin='lower')
            im = ax[1].imshow(mu/stdvar, origin='lower')
            plt.colorbar(im, ax=ax[1])
            plt.savefig('teststatglobal.png')
            plt.close()
            print ('tws stat', np.max(mu), np.min(mu), np.max(stdvar[mm==1]), np.min(stdvar[mm==1]))
            sys.exit()
            """
        else:
            mu = kwargs['mu']
            stdvar = kwargs['stdvar']
        for i in range(arr.shape[0]):
            arr[i][mm==1] = (arr[i][mm==1]-mu[mm==1])/stdvar[mm==1]
        return arr, mu, stdvar
    elif method=='minimax':
        arr = da.values
        if op=='train':
            arrmin = np.nanmin(arr,axis=0)
            arrmax = np.nanmax(arr,axis=0)
        else:
            arrmin = kwargs['min']
            arrmax = kwargs['max']
        for i in range(arr.shape[0]):
            arr[i][mm==1] = (arr[i][mm==1]-arrmin[mm==1])/(arrmax[mm==1]-arrmin[mm==1])
        return arr, arrmin,arrmax
    elif method == 'global':        
        if op == 'train':
            arr = da.values
            #calculate stat on masked area only
            mu = np.mean(arr[:, mask==1])
            stdvar = np.std(arr[:, mask==1])
            print ('mu', mu, 'stdvar', stdvar)
        else:
            mu = kwargs['mu']
            stdvar = kwargs['stdvar']
        arr = (arr -mu) /stdvar

        return arr,mu,stdvar
    elif method== 'deseason':
        if op=='train':
            monthlyMean = da.groupby("time.month").mean("time")
        else:
            monthlyMean = kwargs['monthlymean']        
        da = da.groupby("time.month")-monthlyMean
        arr = da.values
        return arr, monthlyMean

def reScale(arr, mask, method='norm', **kwargs):
    if method=='norm':
        mu = kwargs['mu']
        stdvar = kwargs['stdvar']        
        arr = np.einsum('kij,ij->kij', arr,mask)
        
        for i in range(arr.shape[0]):
            arr[i][mask==1] = arr[i][mask==1]*stdvar[mask==1]+mu[mask==1]
        return arr
    elif method=='minimax':
        arrmin = kwargs['min']
        arrmax = kwargs['max']
        for i in range(arr.shape[0]):
            arr[i][mask==1] = arr[i][mask==1]*(arrmax[mask==1]-arrmin[mask==1])+arrmin[mask==1]
        return arr
    elif method=='global':
        mu = kwargs['mu']
        stdvar = kwargs['stdvar']        
        arr = np.einsum('kij,ij->kij', arr,mask)
        
        arr[:,mask==1] = arr[:,mask==1]*stdvar+mu
        
        return arr


def getShapeMask(filename, isProject, crs=None, statefp=None, countyfp=None):
    import geopandas
    gdf= geopandas.read_file(filename)
    if isProject:
        gdf = gdf.to_crs(crs)
    #subset the state mask
    if not statefp is None:
        gdf=gdf[gdf['STATEFP']==statefp]
    if not countyfp is None:
        gdf=gdf[gdf['COUNTYFP']==countyfp]
    return gdf


def getRegionExtent(regionName, blocksize=64, cellsize=0.25):
    if regionName=='cna':
        llcrnrlon =-109; llcrnrlat=24 #the bound of downloaded ERA5 data is 24 to 50

    elif regionName == 'nna':
        llcrnrlon =-109; llcrnrlat=32

    elif regionName == 'wna':
        #original, (28, -130, 60, -105)
        #regionExtents: (28.6,-124.75,48.25,-105)
        llcrnrlon =-125; llcrnrlat=32

    elif regionName == 'sna':
        llcrnrlon =-95; llcrnrlat= 30

    elif regionName == 'amazon':
        #regionExtents: (19, -79.7, -1.25, -60)
        llcrnrlon =-79; llcrnrlat=19
    elif regionName == 'india':
        llcrnrlon = 74; llcrnrlat= 9
    elif regionName == 'ceu':
        #central europe
        llcrnrlon = 9; llcrnrlat= 40
    elif regionName == 'global':
        llcrnrlon = -180; llcrnrlat= -90
    else:
        raise Exception('region not defined')
    if regionName != 'global':
        lon0 = llcrnrlon; lon1 = lon0+(blocksize)*cellsize
        lat0 = llcrnrlat; lat1 = lat0+(blocksize)*cellsize    
        region = (lat0,lon0,lat1,lon1)
    else:
        region = (-90,-180,90,180)        
    return region

def fillDates(daTWS, interval=5):
    """Find dates with missing data
    #as10302021, add missing data dates
    
    Returns
    -------
    Series with complete dates but nan as the missing values
    """
    if len(daTWS.shape)>1:
        ds = daTWS.isel(lat=0, lon=0).to_series() #converting to pd Series is probably not necessary
    else:
        ds = daTWS.to_series()
    #assuming the first date has data
    newdates = [ds.index[0]]    
    INTERVAL = interval
    for i in range(1, len(ds.index)):
        if (ds.index[i] - ds.index[i-1]).days>INTERVAL:
            tmp = ds.index[i-1]
            while (ds.index[i] - tmp).days>INTERVAL:
                tmp = tmp + timedelta(days=INTERVAL)
                newdates.append(tmp)
        newdates.append(ds.index[i])

    if len(daTWS.shape)>1:
        da = daTWS.reindex(time = pd.to_datetime(newdates), method=None)
        series = da.isel(lon=0, lat=0).to_series()
        validLabel = series.isna()
    else:
        ds = ds.reindex(index=pd.to_datetime(newdates), method=None)
        validLabel = ds.isna()
        da = xr.DataArray.from_series(ds)       
        da = da.rename({'index':'time'})

    return da, validLabel
    
def plot_index(df: pd.DataFrame, date_col: str, precip_col: str, save_file: str=None,
               index_type: str='SPI', bin_width: int=22, ax=None):
    #modified from standard Precipitation package

    pos_index = df.loc[df[precip_col] >= 0]
    neg_index = df.loc[df[precip_col] < 0]

    assert(not ax is None)
    ax.bar(pos_index[date_col], pos_index[precip_col], width=bin_width, align='center', color='b', alpha=0.5)
    ax.bar(neg_index[date_col], neg_index[precip_col], width=bin_width, align='center', color='r', alpha=0.5)
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel(index_type)

def eventcoinrate(da, db, columns, tau=10, delta=3, eventtypes=['precursor']):
    """
    da, target dataframe/dataarray
    db, source dataframe/dataarray (i.e., use db dates to compare to da dates)
    thresholds, given in quantile
    tau, min time window for da events (multiple events in tau will be counted as 1)
    delta, min time window for da-db event comparison (multiple db events in delta will be counted as one)
    eventtype, precursor (B is trigger of A), trigger (A is trigger of B)
    """
    def getEvents(alldates):
        #make sure annual maxima are not too close
        eventdates = []
        eventindices = []
        prevDate = None
        for ix, adate in enumerate(alldates):
            if (prevDate is None):
                eventdates.append(adate)
                eventindices.append(ix)
            else:
                #look at all existing event dates
                dT = (adate-prevDate).days
                if dT>delta:
                    eventdates.append(adate)
                    eventindices.append(ix)                    
            prevDate = adate            
        return eventdates, eventindices
        
    if type(da) == xr.DataArray:
        #simple conversion to series
        da = da.to_dataframe()
    if type(db) == xr.DataArray:
        db = db.to_dataframe()

    #get column names of data
    colA,colB = columns
    da = da.dropna()
    db = db.dropna()
   
    #get annual maxima time series
    daEventsAnnual = da.loc[da.groupby(da.index.year)[colA].idxmax()]
    dbEventsAnnual = db.loc[db.groupby(db.index.year)[colB].idxmax()]
    #filter events that are too close to each other
    datesA, indA = getEvents(daEventsAnnual.index.date)
    datesB, indB = getEvents(dbEventsAnnual.index.date)
    daEvents = daEventsAnnual.iloc[indA,:]
    dbEvents = dbEventsAnnual.iloc[indB,:]

    for eventtype in eventtypes:
        if eventtype == 'precursor':
            #series B is before A
            #do a double loop to find precursor rate
            #the fraction of A-type events that are preceded by at least one B-type event
            eventA = []; eventB=[]
            for ia,rowa in enumerate(datesA):
                for ib,rowb in enumerate(datesB):
                    deltaT =  (rowa-rowb).days
                    if deltaT>0 and deltaT<tau:
                        eventA.append(ia)
                        eventB.append(ib)
                        break         
            nEventA = len(daEvents)
            r_precursor = len(eventA)/nEventA
            daEvents = daEvents.iloc[eventA,:]
            dbEvents = dbEvents.iloc[eventB,:]
            return daEvents[colA], dbEvents[colB],r_precursor,nEventA,daEventsAnnual[colA],dbEventsAnnual[colB]
        elif eventtype == 'trigger':            
            #series A is trigger of series B
            #the fraction of B-type events that are followed by at least one A-type event
            eventA = []; eventB = []
            for ib,rowb in enumerate(datesB):
                for ia,rowa in enumerate(datesA):
                    deltaT =  (rowa-rowb).days
                    if deltaT>0 and deltaT<tau:
                        eventA.append(ia)
                        eventB.append(ib)
                        break
            nEventB = len(dbEvents)
            r_trigger = len(eventA)/nEventB
            daEvents = daEvents.iloc[eventA,:]
            dbEvents = dbEvents.iloc[eventB,:]
            return daEvents[colA], dbEvents[colB],r_trigger,nEventB

def getBasinDD(gdfTop):
    """get basin averaged drainage density
    """
    import geopandas
    gdf = geopandas.read_file(os.path.join(os.getcwd(), 'maps/drainage/watersheds.gdb'))
    gdf = gdf.to_crs('epsg:4326')    
    gdfTop['DD'] = np.zeros((gdfTop.shape[0],1))

    for indx,poly in gdfTop.iterrows(): 
        print ('basin', poly['NAME'])
        clipped = gdf.clip(poly.geometry)
        print ('number of rows', clipped.shape)
        #get area-weighted dd
        DD = 0
        sumArea = 0
        for ix,item in clipped.iterrows():            
            if not pd.isnull(item["areasqkm"]) and not pd.isnull(item["dd"]):
                DD+= float(item['dd'])*float(item['areasqkm'])
                sumArea += float(item['areasqkm'])
        if (sumArea>0):
            print ('total area',sumArea)
            print ('before', DD)
            DD = DD/sumArea
            print ('DD value is', DD)
            gdfTop.loc[indx, 'DD'] = DD
    gdfTop.to_file(os.path.join(os.getcwd(), 'maps/top60basin.shp'))
    return gdfTop


def getAshrafMask():
    """Load Ashraf 1x1 mask, this is a version that Antarctica and Greenland are removed
    """
    mask = np.loadtxt(os.path.join(os.getcwd(), '../grace/data/LandMask_OneDeg.txt'), delimiter=',')
    lat = np.arange(-89.5,90)
    lon = np.arange(-179.5,180)
    da = xr.DataArray(
        data=mask,
        coords=dict(
            lat = lat,
            lon= lon,
        ),
        attrs=dict(
            description="Ashraf mask",
            units="[]",
        ),
    )    
    #da.coords['lon'] = (da.coords['lon'] + 180.0) % 360 - 180.0
    da = da.sortby(da.lon)
    mask = da.values
    return mask

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)

def getSingleTWSIndex(daTWSA, returnClass=True,climatologyTWSDA=None):
    """Calculate climatology mean and std dev for series
    Assuming climatology at monthly level
    #classification from Zhao et al. 2017
    <-2.0 is an exceptional drought,
    -1.99 to -1.60 is an extreme drought
    -1.59 to -1.30 is a severe drought
    -1.29 to -0.80 is a moderate drought
    -0.79 to -0.50 is abnormally dry
    -0.49 to 0.49 is near normal, 
    0.50 to 0.79 is slightly wet
    0.80 to 1.29 is moderately wet
    1.30 to 1.59 is very wet
    1.60 to 1.99 is extremely wet, and
    >2.0 is exceptionally wet.
    """
    if climatologyTWSDA is None:
        mu = daTWSA.groupby('time.month').mean('time').values
        std_dev = daTWSA.groupby('time.month').std('time').values
    else:
        mu = climatologyTWSDA.groupby('time.month').mean('time').values
        std_dev = climatologyTWSDA.groupby('time.month').std('time').values

    tws_dsi = np.zeros(daTWSA.shape)
    for ix,item in enumerate(pd.to_datetime(daTWSA.time.values)):
        imon = item.date().month-1
        standarized_val = (daTWSA.values[ix]-mu[imon])/std_dev[imon]
        if returnClass is True:
            if standarized_val<=-2.0:
                tws_dsi[ix] = 0
            elif standarized_val>-2.0 and standarized_val<=-1.6:
                tws_dsi[ix] = 1
            elif standarized_val>-1.6 and standarized_val<=-1.3:
                tws_dsi[ix] = 2
            elif standarized_val>-1.3 and standarized_val<=-0.8:
                tws_dsi[ix] = 3
            elif standarized_val>-0.8 and standarized_val<=-0.5:
                tws_dsi[ix] = 4
            elif standarized_val>-0.5 and standarized_val<0.5:
                tws_dsi[ix] = 5
            elif standarized_val>=0.5 and standarized_val<0.8:
                tws_dsi[ix] = 6
            elif standarized_val>=0.8 and standarized_val<1.3:
                tws_dsi[ix] = 7
            elif standarized_val>=1.3 and standarized_val<1.6:
                tws_dsi[ix] = 8
            elif standarized_val>=1.6 and standarized_val<2.0:
                tws_dsi[ix] = 9            
            elif standarized_val>2.0:
                tws_dsi[ix] = 10              
        else:
            tws_dsi[ix] = standarized_val

    daTWS_DSI = xr.DataArray(tws_dsi, name='dsi', dims = daTWSA.dims, coords=daTWSA.coords)
    if returnClass:    
        categories = ['ED', 'XD', 'SD', 'MD', 'D', 'N', 'W','MW','SW','XW', 'EW']
        return daTWS_DSI, categories
    else:
        return daTWS_DSI

def make_confusion_matrix(trueDA,predDA, 
                          plotax,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          cmap='Blues',
                          title=None):
    '''
    from https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''
    cf = confusion_matrix(y_true=trueDA.values, y_pred=predDA.values)

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION    
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories,ax=plotax)

    if xyplotlabels:
        plotax.set_ylabel('True label')
        plotax.set_xlabel('Predicted label' + stats_text)
    else:
        plotax.set_xlabel(stats_text)
    
    if title:
        plotax.set_title(title)

def getFPIData(poly, twsDA, precipDA, crs, op='sum', weighting=False):
    """Generate FPI at grid level
    """
    #rename coords
    xdsTWS = twsDA.rename({'lat':'y', 'lon':'x'})      
    xdsTWS = xdsTWS.sortby('y')  
    xdsTWS.rio.write_crs("epsg:4326", inplace=True)

    xdsPrecip = precipDA.rename({'lat':'y', 'lon':'x'})      
    xdsPrecip = xdsPrecip.sortby('y')  
    xdsPrecip.rio.write_crs("epsg:4326", inplace=True)
    
    with rioxarray.set_options(export_grid_mapping=False):
        bbox=poly.bounds
        #convert to [lon0,lat0,lon1,lat1]
        bbox=bbox.values.tolist()[0]
        xdsTWS = xdsTWS.sel(y=slice(bbox[1],bbox[3]),x=slice(bbox[0],bbox[2]))
        xdsPrecip = xdsPrecip.sel(y=slice(bbox[1],bbox[3]),x=slice(bbox[0],bbox[2]))

        #note if the following step takes too long, re-generate the netcdf file
        clippedPrecip = xdsPrecip.rio.clip(poly.geometry, crs, from_disk=True, drop=True, invert=False)
        clippedPrecip = clippedPrecip.rename({'y':'lat', 'x':'lon'})      
        
        clippedTWS = xdsTWS.rio.clip(poly.geometry, crs, from_disk=True, drop=True, invert=False)
        clippedTWS = clippedTWS.rename({'y':'lat', 'x':'lon'})      
        
        """
        fig,ax=plt.subplots(4,1, figsize=(10,10))
        for ix,i in enumerate([100, 200, 300, 500]):
            clipped.isel(time=i).plot(ax=ax[ix])
        plt.savefig('testumag.png')
        plt.close()
        """
        nT = len(clippedTWS.time)
        fpi = np.zeros(clippedTWS.values.shape)
        
        maxTWS = clippedTWS.max(dim='time').values
        for i in range(1, nT):
            sdef = maxTWS - clippedTWS.isel(time=i-1).values
            fpi[i] = clippedPrecip.isel(time=i).values-sdef
        fpi = (fpi - np.nanmin(fpi))/(np.nanmax(fpi)-np.nanmin(fpi))
        fpiDA = xr.DataArray(fpi, dims=clippedTWS.dims, coords=clippedTWS.coords)

    #get weighted basin average
    basinAvgDA = calculateBasinAverage(fpiDA, op=op, weighting=weighting)
    return basinAvgDA

def calculateBasinAverage(da, op='sum', weighting=False):
    """Calculate weighted stats on a dataarray

    from pyproj import Geod
    from shapely.geometry import Polygon
    lat0,lon0,lat1,lon1 = region
    poly = Polygon([(lon0,lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
    # specify a named ellipsoid
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(poly)[0])
    print('# Geodesic area: {:.3f} m^2'.format(area))
    """
    if weighting:
        weights = np.cos(np.deg2rad(da.lat))
        weights.name = "weights"
        if op=='sum':
            da_weighted = da.weighted(weights).sum(dim=['lat', 'lon'])
        elif op=='mean':
            da_weighted = da.weighted(weights).mean(dim=['lat', 'lon'])
        return da_weighted
    else:
        if op == 'sum':
            return da.sum(dim=['lat', 'lon'])
        elif op == 'mean':
            return da.mean(dim=['lat', 'lon'])
        else:
            raise ValueError("Invalid option")

