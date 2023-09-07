#author: alex sun
#date: 0824
#GRDC Q unit m3/s
#CSR5d unit: cm
#find good grdc reference data
#Daily runoff data downloaded by using Download Station, selected such that drainage area>4e4 km2
#date: 0826, cleaned up for production. Use plotglobalmap and itask=2 for global
#rev date: 09072022, double check for grace monthly meeting
#rev date: 06092023, cleanup for manuscript
#rev date: 06232023, add plots for bivariate, joint occurrence, and CMI
#For bivariate, CMI vs. Joint Occurrence; for joint occurrence it's P and TWS; and CMI it's (TWS, Q|P)
#rev date: 07312023, consider remove season option, this is enabled by setting cfg.data.deseason to true
#rev date: 08022023, cleanup again
#rev date: decide on final runs: 
#          -- MAF on 5-day min dist for bivariate and compound event
#          -- MAF on 5-day min dist for fake5d bivariate and compound event 
#rev date: 08052023, implemented SWE removal  set cfg.data.removeSWE to True in config.yaml
#rev date: 08072023, fixed bug in doSWERemoval, should use anomaly by subtracting mean SWE
#=======================================================================================================
import pandas as pd
import os,sys
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import  numpy as np
import pickle as pkl
import cartopy.crs as ccrs
import hydrostats as HydroStats
import colorcet as cc
from datetime import datetime
from scipy.stats import kendalltau, gumbel_r
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from shapely.geometry import Point
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from generativepy.color import Color
from matplotlib.colors import rgb2hex
import rioxarray as rxr
import seaborn as sns
import statsmodels.api as SM

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.regressionCI import RegressionCI
from tigramite.independence_tests.gsquared import Gsquared
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.causal_effects import CausalEffects
from tigramite.models import LinearMediation

from myutils import getRegionExtent, getFPIData
from dataloader_global import load5ddatasets,loadClimatedatasets,getCSR5dLikeArray,loadTWSDataArray_SWE
import glofas_us
from glofas_us import extractBasinOutletQ, loadGLOFAS4CONUS
from myutils import bandpass_filter,removeClimatology


import warnings
warnings.filterwarnings("ignore")

#dictionary for plotting
regionNames = {
    'north_america': "North America", 
    "south_america": "South America", 
    "europe": "Europe",
    "africa": "Africa", 
    "south_pacific": "South-West Pacific",
    "global": '' }

hydroshedMaps={
    "north_america": 'hybas_na_lev04_v1c/hybas_na_lev04_v1c.shp',
    'south_america': 'hybas_sa_lev04_v1c/hybas_sa_lev04_v1c.shp',
    'europe': 'hybas_eu_lev04_v1c/hybas_eu_lev04_v1c.shp',
    'africa': 'hybas_af_lev04_v1c/hybas_af_lev04_v1c.shp',
    'south_pacific': "hybas_au_lev04_v1c/hybas_au_lev04_v1c.shp"
}
#number of neighbors for CMI
KNN = 6
#BASIN AREA
MIN_BASIN_AREA = 1.2e5

def getExtremeEvents(series, method='MAF', cutoff=90, transform=False, minDist=None, 
                     season=None, returnExtremeSeries=False):
    """   
    Params
    ------
    series, input dataframe
    method: MAF, mean annual flood; POT peak over threshold 
    cutoff: this is only used by POT
    minDist: minimum distance betweeen conseq events
    season: if None, restrict to season
    returnExtremeSeries: True to return time series
    """
    #transform variable to normal space [this operates on normal anomalies,]
    if transform:
        val = PowerTransformer().fit_transform(series.to_numpy()[:,np.newaxis]).squeeze()
        series = pd.Series(val, index=series.index)

    if method == 'MAF':
        #the following 2 lines extracted annual maxima w/o enforcing minDist
        #extremes = series.groupby(series.index.year).agg(['idxmax', 'max'])  
        #extremes = pd.Series(extremes['max'].values, index=extremes['idxmax'])
        #asun0711, enforce minimum distance between events
        nEvent = 5   #assuming five alternative maxima are enough
        extremes = series.groupby(series.index.year).nlargest(nEvent)
        extremes = extremes.reset_index(level=[0,1])  #flatten the multi-index      
        extremes = pd.Series(extremes[0].values, index=extremes['level_1'])
        
        #check the distance between events
        if not minDist is None:
            #remove close events
            nYears = len(series.index.year.unique())
            goodInd = [0]
            counter = 0
            for i in range(1, nYears):
                for j in range(nEvent):
                    #test each event in each year against the maxima from the prev year
                    if np.abs(extremes.index[i*nEvent+j] - extremes.index[goodInd[counter]]).days>minDist:                    
                        goodInd.append(i*nEvent+j)
                        counter+=1
                        break
                    elif j==nEvent-1:
                        #this should not be reached
                        raise ValueError('cannot find events ')            
            extremes = extremes.iloc[goodInd]      

    elif method == 'POT':
        #POT
        thresh = np.nanpercentile(series.to_numpy(), cutoff)
        #asun 08032023
        #I want to make sure  the largest events are selected        
        extremes = series.loc[series>=thresh]
        extremes = extremes.sort_values(ascending=False)

        if not minDist is None:
            #remove close events
            goodInd = [0]
            for i in range(1, extremes.shape[0]):
                flag=False
                for j in range(0, i):
                    if np.abs((extremes.index[i] - extremes.index[j]).days)<minDist:
                        flag=True
                if not flag:
                    goodInd.append(j)
            extremes = extremes.iloc[goodInd]                            

        if not season is None:
            #note season is zero-based
            if season == 'DJF':
                monrng = [11,0,1]
            elif season == 'MAM':
                monrng = [2,3,4]
            elif season == 'JJA':
                monrng = [5,6,7]
            elif season == 'SON':
                monrng = [8,9,10]
            elif season == 'MAMJJA':
                monrng = [2,3,4,5,6,7]                
            elif season == 'AMJJAS':
                monrng = [3,4,5,6,7,8]                
            else:
                raise ValueError("wrong season code")
            extremes = extremes[extremes.index.month.isin(monrng)]            
            print ('POT: num events extracted', len(extremes), '@ cutoff', cutoff, 'for season', season)
        else:
            print ('POT: num events extracted', len(extremes), '@ cutoff', cutoff)
        print ('*'*80)        
    else: 
        raise ValueError("invalid method")

    events = np.zeros((series.size),dtype=int)
    #drop bad data
    extremes = extremes.dropna()
    
    #for binary event series
    for ix, item in enumerate(series.index.tolist()):
        for time2 in extremes.index.tolist():
            if item==time2:
                events[ix] = 1
    #print a warning if the number annual maxima in events is not the same as the total number of years
    if len(np.where(events==1)[0]) != extremes.size:
        print ("warning:", len(np.where(events==1)[0]), extremes.size)
    print ('+++++++++++++++++++extremes ', extremes.size)
    if returnExtremeSeries:
        return events, extremes
    else:
        return events

def getBasinMask(stationID, region='north_america', gdf=None):    
    """ Get basin mask as a geopandas DF
    09062023: note 
    Param
    -----
    stationID, grdc_no
    region, one of the valid regions
    gdf, if not None, it contains the combined GDF for all stations a region
                      This is downloaded from GRDC as part of the station manual selection 
    Returns
    -------
    basin_gdf, basin mask is taken from the geojson file in each continent folder
               when downloading station catalog, check the "download watershed polygon" box
    """
    rootdir = '/home/suna/work/grace/data/grdc'
    if region == 'globalref':
        shpfile = os.path.join(rootdir, f'grdc_basins_smoothed_md_no_{stationID}.shp')
        basin_gdf = gpd.read_file(shpfile)
        return basin_gdf
    else:
        #suppress the SettingWithCopyWarning warning
        basin_gdf = gdf.loc[ gdf['grdc_no']==stationID].copy(deep=True)
        return basin_gdf

def getPredictor4Basin(gdf,lat,lon, xds):
    """Crop the global dataset for the specified basin
    Params
    ------
    gdf, mask of the basin
    xds, DataArray of the dataset to be masked

    Returns
    -------
    meanTS, the basin average time series (1D dataarray)
    """
    xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    xds.rio.write_crs("epsg:4326", inplace=True)
    #must rename coords for rioarray to work
    xds = xds.rename({'lat':'y', 'lon':'x'})      
    xds = xds.sortby('y')       
    try: 
        with rioxarray.set_options(export_grid_mapping=False):
            bbox=gdf.bounds
            #convert to [lon0,lat0,lon1,lat1]
            bbox=bbox.values.tolist()[0]
            basinxds = xds.sel(y=slice(bbox[1],bbox[3]),x=slice(bbox[0],bbox[2]))        
            #08/28/2022, add all_touched = True
            clipped = basinxds.rio.clip(gdf.geometry, gdf.crs, from_disk=True, drop=True, invert=False, all_touched = True)  
            meanTS =  clipped.where(clipped.notnull()).mean(dim=['y','x'])
    except:
        #get 3x3 region around station cell if the above process fails
        cellDA = xds.sel(y=slice(lat-1.5, lat+1.5), x=slice(lon-1.5, lon+1.5))
        print ('number of cells', cellDA.shape[1])
        meanTS = cellDA.where(cellDA.notnull()).mean(dim=['y','x'])
    return meanTS

def getBasinData(config, stationID, region, lat, lon, xds = None, 
                xds_Precip=None, gdf=None, returnFPI=False,xdsC = None, removeSWE=False):
    """Get basin-averaged time series
    Params
    ------
    stationID, grdc station id
    region, geo region
    lat,lon, location of the station
    xds, CSR.5d
    xds_Precip, precip data 
    08052023: add removeSWE option
    """
    gdf = getBasinMask(stationID=stationID, region=region, gdf=gdf)
    region = getRegionExtent(regionName="global")
    
    #TWS
    #07142023: use TWS 0.25 for TWS masking, but ERA5 1deg for climate masking
    #          ERA5 0.25 deg is too big to load into memory
    #          Do this in two steps. Step 1: turn coarsen in load5ddatasets to True, reload to True in both load5ddatasets and loadClimatedatasets
    #          Step 2: turn coarsen to False, reload to True in load5ddatasets, but False in loadClimatedatasets
    #          During regular run, the first should print (1022, 720, 1440) (720, 1440)
    if xds is None:
        if not removeSWE:
            print ('!!! Load original TWS')
            xds,_, _ = load5ddatasets(region=region, 
                coarsen=False, 
                mask_ocean=config.data.mask_ocean, 
                removeSeason=config.data.remove_season, 
                reLoad=config.data.reload, 
                csr5d_version=config.data.csr5d_version,
                startYear=datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
                endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year,
            )
        else:
            print ('!!!! Load TWS with  SWE removed')
            #note: run loadTWSDataArray_SWE in dataloader_global to pre-save the data  
            xds = loadTWSDataArray_SWE(
                region=region,
                coarsen=False,
                mask_ocean=config.data.mask_ocean, 
                removeSeason=config.data.remove_season, 
                reLoad=config.data.reload, 
                csr5d_version=config.data.csr5d_version,
                startYear=datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
                endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year,
            )

        if returnFPI:
            #aggregate TWS to 1deg to be used w/ 1 deg precip data
            xdsC = xds.coarsen(lat=4, lon=4).mean()

    basinTWS = getPredictor4Basin(gdf, lat, lon, xds)

    #Precip [use saved 5d data]
    if xds_Precip is None:
        xds_Precip = loadClimatedatasets(region, vartype='precip', daCSR5d= xds, \
                                        aggregate5d=True, 
                                        reLoad=config.data.reload_precip, 
                                        precipSource=config.data.precip_source,
                                        startYear= datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
                                        endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year)
    basinP = getPredictor4Basin(gdf, lat, lon, xds_Precip)

    #always use detrended twsDA to calculate fpi
    if returnFPI:
        basinFPI = getFPIData(gdf, xdsC, xds_Precip, gdf.crs, op='mean', weighting=True)

        return basinTWS, basinP, basinFPI, xds, xds_Precip, xdsC
    else:
        return basinTWS, basinP, xds, xds_Precip

def readStationSeries(stationID, region='north_america'):
    """ Parse GRDC runoff time series 
        Note: the null values are dropped at this stage 
    Params
    ------
    stationID: id of the GRDC station

    """
    rootdir = f'/home/suna/work/grace/data/grdc/{region}'
    
    stationfile = os.path.join(rootdir, f'{stationID}_Q_Day.Cmd.txt')
    #skip 37 header rows
    df = pd.read_csv(stationfile, encoding = 'unicode_escape', skiprows=37, index_col=0, delimiter=';', usecols=[0, 2], header=None)    
    df.columns=['Q']    
    df.index = pd.to_datetime(df.index)
    df.index.names = ['time']
    df['Q'] = pd.to_numeric(df['Q']) 

    #drop bad values
    df= df[df.Q>0]
    return df

def print_significant_links(N, var_names,
                                p_matrix,
                                val_matrix,
                                conf_matrix=None,
                                graph=None,
                                ambiguous_triples=None,
                                alpha_level=0.05):
        """Prints significant links.
        #asun: this is copied/modified from github pcmci.py
        Used for output of PCMCI and PCMCIplus. For the latter also information
        on ambiguous links and conflicts is returned.
        Note: this return sorted links for Q only

        Parameters
        ----------
        alpha_level : float, optional (default: 0.05)
            Significance level.
        p_matrix : array-like
            Must be of shape (N, N, tau_max + 1).
        val_matrix : array-like
            Must be of shape (N, N, tau_max + 1).
        conf_matrix : array-like, optional (default: None)
            Matrix of confidence intervals of shape (N, N, tau_max+1, 2).
        graph : array-like
            Must be of shape (N, N, tau_max + 1).
        ambiguous_triples : list
            List of ambiguous triples.
        """
        if graph is not None:
            sig_links = (graph != "")*(graph != "<--")
        else:
            sig_links = (p_matrix <= alpha_level)

        print("\n## Significant links at alpha = %s:" % alpha_level)
        for j in range(N):
            links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                     for p in zip(*np.where(sig_links[:, j, :]))}
            # Sort by value
            sorted_links = sorted(links, key=links.get, reverse=True)
            #return sorted_links for Q only [asun: 07042023]
            if j == 0:
                sorted_links_Q = sorted_links
            n_links = len(links)
            string = ("\n    Variable %s has %d "
                      "link(s):" % (var_names[j], n_links))
            for p in sorted_links:
                string += ("\n        (%s % d): pval = %.5f" %
                           (var_names[p[0]], p[1],
                            p_matrix[p[0], j, abs(p[1])]))
                string += " | val = % .3f" % (
                    val_matrix[p[0], j, abs(p[1])])
                if conf_matrix is not None:
                    string += " | conf = (%.3f, %.3f)" % (
                        conf_matrix[p[0], j, abs(p[1])][0],
                        conf_matrix[p[0], j, abs(p[1])][1])
                if graph is not None:
                    if p[1] == 0 and graph[j, p[0], 0] == "o-o":
                        string += " | unoriented link"
                    if graph[p[0], j, abs(p[1])] == "x-x":
                        string += " | unclear orientation due to conflict"
        
        return string, sorted_links_Q

def getCMI(cfg, stationID, df, river=None, saveDataFrame=False):
    """Get conditional MI 
    I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
    The conditional mutual information is a measure of how much uncertainty is shared by X and Y , but not by Z.
    Params
    ------
    stationID, id of the station
    df, dataframe containing all variables
    """
    plotting = cfg.data.plot_bivarate_scatter
    # CMI(X,Y|Z)
    # X = group 0; Y= group 1; Z = group 2
    # 0   1   2
    #[Q, TWS, P]
    var_names=df.columns
    data = df.values
    #form tigramite dataframe
    dataframe = pp.DataFrame(data, 
            datatime = {0:np.arange(data.shape[0])}, 
            var_names=var_names,
            missing_flag=-999.)

    #Calculate CMI    
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=RobustParCorr(), 
        verbosity=1)

    correlations = pcmci.get_lagged_dependencies(tau_min=cfg.causal.tau_min, tau_max=cfg.causal.tau_max, val_only=True)['val_matrix']
    
    if plotting:
        #plot the max lagged correlation for each station
        plt.figure()
        matrix_lags = np.argmax(np.abs(correlations), axis=2)
        tp.plot_densityplots(dataframe=dataframe, setup_args={'figsize':(15, 10)}, add_densityplot_args={'matrix_lags':matrix_lags});    
        #tp.plot_densityplots(dataframe=dataframe, add_densityplot_args={'matrix_lags':None})
        plt.savefig(f'outputs/test_density_plot{stationID}.png')
        plt.close()
        
        #plot the scatter plots for each station
        plt.figure()
        tp.plot_scatterplots(dataframe=dataframe, 
                setup_args={'figsize': (16,16), 'label_fontsize': 16},
                add_scatterplot_args={'matrix_lags':matrix_lags, 'color': 'blue', 'markersize':10, 'alpha':0.7})
        plt.savefig(f"outputs/test_scatterplot{stationID}.png")
        plt.close()

        #plot tigramite time series for each station [not used]
        plt.figure()
        tp.plot_timeseries(dataframe); 
        plt.savefig(f"outputs/test_timeseries{stationID}.png")
        plt.close()

    if saveDataFrame:
        #save dataframe for later use
        pkl.dump(dataframe, open(f'grdcdataframes/s_{stationID}.pkl', 'wb'))

    cmi_knn = CMIknn(
        significance='shuffle_test', 
        knn= cfg.data.knn, #if <1, this is fraction of all samples
        shuffle_neighbors= 10, 
        workers = 12,
        transform='rank',
        sig_samples=1000,
        verbosity=0)
    arr = data.T 

    # Subsample indices
    # x_indices = np.where(xyz == 0)[0]
    # y_indices = np.where(xyz == 1)[0]
    # z_indices = np.where(xyz == 2)[0]
    # index groups 0: X; 1: Y; 2: Z (all index groups may have multiple variables )

    #note: tigramite requires array to have X, Y, Z in rows and observations in columns
    xyz = np.array([0, 1, 2],dtype=int)
    cmi_knn_score = cmi_knn.get_dependence_measure(arr, xyz=xyz)
    #p_val = cmi_knn.get_shuffle_significance(arr, xyz=xyz, value=cmi_knn_score)
    
    # ==== Uncomment the following to do causal graph plot=======
    # if plotting:
    #     fig,ax=plt.subplots(1,1)
    #     matrix = tp.plot_lagfuncs(val_matrix=correlations, 
    #                                     setup_args={'var_names':var_names, 'x_base':5, 'y_base':.5})
    #     matrix.fig.suptitle(f'{river}, CMI:{cmi_knn_score:4.3f}', fontsize=10,color='red')
    #     plt.savefig(f'test_lag_plot_{stationID}.png')
    #     plt.close()

    #     if plotting:
    #         fig,ax = plt.subplots(2,1)
    #         tp.plot_timeseries(dataframe,  fig_axes=(fig,ax[0]))
    #         ax[0].set_title(f'GRDC No. {stationID}') 
            
    #         tp.plot_graph(
    #             val_matrix=results['val_matrix'],
    #             graph=results['graph'],
    #             var_names=var_names,
    #             link_colorbar_label='cross-MCI',
    #             node_colorbar_label='auto-MCI',
    #             vmin_edges=0.,
    #             vmax_edges = 0.3,
    #             edge_ticks=0.05,
    #             cmap_edges='OrRd',
    #             vmin_nodes=0,
    #             vmax_nodes=.5,
    #             node_ticks=.1,
    #             cmap_nodes='OrRd',
    #             fig_ax = ax[1],
    #             )
    #         plt.savefig(f'testcmi_{stationID}.png')
    #         plt.close()
    return cmi_knn_score

def doCausalAnalytics(cfg, df, binary=False):
    """Do causal analytics
    06202023
    see 
    https://github.com/jakobrunge/tigramite/blob/master/tutorials/case_studies/climate_case_study.ipynb
    """
    #assumption: target variable Q is always the first variable!!!
    # 0   1   2
    #[Q, TWS, P]
    var_names=df.columns
    data = df.values
    dataframe = pp.DataFrame(data, 
            datatime = df.index.values, 
            var_names=var_names,
            missing_flag=-999)

    tau_min = cfg.causal.tau_min
    tau_max = cfg.causal.tau_max
    
    #Formulate the set of potential links
    selected_links = {}
    # N is the number of variables (nodes)
    # link_assumptions[j] = {(i, -tau):"-?>" for i in range(self.N) for tau in range(1, tau_max+1)}
    #This initializes the graph with entries graph[i,j,tau] = link_type, i.e., link from i to j
    #@see documentation at
    #https://github.com/jakobrunge/tigramite/blob/master/tigramite/pcmci.py

    for i in range(len(var_names)):
        selected_links[i] = {}
    
    ivar=0    
    #TWS->Q, directed lagged links
    for jvar in [0,1]:
        for ilag in range(tau_min, tau_max+1):
            #exclude Q
            if cfg.causal.exclude_Q:
                if (jvar != 0):
                    selected_links[ivar][(jvar, -ilag)]='-?>'
            else:
                selected_links[ivar][(jvar, -ilag)]='-?>'
    #P->Q, directed lagged links
    ivar=0
    jvar=2
    for ilag in range(tau_min, tau_max+1):
        selected_links[ivar][(jvar, -ilag)]='-?>'
    #P->TWS directed lagged links
    ivar = 1 #TWS
    jvar = 2 #P
    for ilag in range(tau_min, tau_max+1):
        selected_links[ivar][(jvar, -ilag)]='-?>'
    #add self links for P and TWS
    for ivar in [1,2]:   
        for ilag in range(tau_min, tau_max+1):
            selected_links[ivar][(ivar, -ilag)]='-?>'

    # #add undirected links at time =0 
    # for ivar in [0,1]:
    #     for jvar in [1,2]:
    #         if ivar!=jvar:
    #             selected_links[ivar][(jvar, 0)]='o?o'

    pc_alpha = 0.1
    alpha_level = 0.05
    if binary:
        #figuring out the causal relationship between extreme events
        cmi_symb = CMIsymb(significance='shuffle_test', n_symbs=None)   #this is too slow
        gsquared = Gsquared(significance='analytic')     
        pcmci_cmi_symb = PCMCI( dataframe=dataframe, cond_ind_test=gsquared)

        results = pcmci_cmi_symb.run_pcmci(link_assumptions=selected_links, 
                                        tau_min = tau_min, 
                                        tau_max=tau_max, 
                                        pc_alpha=pc_alpha)
        
        #see ref at: https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_conditional_independence_tests.ipynb
        #CMI = G/(2*n_samples)
        outstr, sorted_links = print_significant_links(data.shape[1], var_names, 
                                p_matrix=results['p_matrix'],
                                val_matrix=results['val_matrix']/(2.*df.shape[0]),
                                alpha_level=alpha_level)
        #08222023, the following lines are the same between binary and continuous, can be combined!!!!
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=pcmci_cmi_symb.all_parents, tau_max=tau_max)
        med.fit_model_bootstrap(boot_blocklength=1, seed=28, boot_samples=200)
        
        ace = med.get_all_ace()
        ce_boots = med.get_bootstrap_of(function='get_all_ace', function_args={}, conf_lev=0.9)
        print ("average causal effect of TWS,P ", ace )

        #08212023, add printout of causal effect of TWSA and P at individual lags        
        target_var = 0 #Q
        input_vars = [1,2] #TWSA, P
        ce_dict={'TWSA':[], 'P':[]}
        for invar in input_vars:
            if invar == 1:
                ce_dict['TWSA'] = [med.get_ce(i=invar, tau=-tau, j=target_var) for tau in range(tau_min, tau_max+1)]            
            elif invar == 2:
                ce_dict['P']    = [med.get_ce(i=invar, tau=-tau, j=target_var) for tau in range(tau_min, tau_max+1)]            

    else:
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test= RobustParCorr() 
        )    
        results = pcmci.run_pcmci(
            link_assumptions= selected_links, 
            tau_min=tau_min,
            tau_max=tau_max, 
            pc_alpha=pc_alpha)
        
        #Extract sorted causal links for Q only
        outstr, sorted_links = print_significant_links(data.shape[1], var_names, 
                        p_matrix=results['p_matrix'],
                        val_matrix=results['val_matrix'],
                        alpha_level=alpha_level)
        
        print ("*****For Q only*****", sorted_links)

        #08222023, the following lines are the same between binary and continuous, can be combined!!!!        
        med = LinearMediation(dataframe=dataframe)                
        #here use the all_parents from the PCMCI causal discovery
        med.fit_model(all_parents=pcmci.all_parents, tau_max=tau_max)
        med.fit_model_bootstrap(boot_blocklength=1, seed=28, boot_samples=200)
        #at the lag of the maximum absolute causal effect
        ace = med.get_all_ace(lag_mode='absmax', exclude_i=True)
        ce_boots = med.get_bootstrap_of(function='get_all_ace', function_args={}, conf_lev=0.9)
        #08212023, add printout of causal effect of TWSA and P at individual lags
        target_var = 0 #Q
        input_vars = [1,2] #TWSA, P
        ce_dict={'TWSA':[], 'P':[]}
        for invar in input_vars:
            if invar == 1:
                ce_dict['TWSA'] = [med.get_ce(i=invar, tau=-tau, j=target_var) for tau in range(tau_min, tau_max+1)]            
            elif invar == 2:
                ce_dict['P']    = [med.get_ce(i=invar, tau=-tau, j=target_var) for tau in range(tau_min, tau_max+1)]            
        #
        print ("average causal effect of TWS,P ", ace)

    return outstr, sorted_links, results['p_matrix'],results['val_matrix'],ace,ce_boots,ce_dict

def fitScatterPlot(cfg, stationID, daTWS,daQ, river):
    #    
    tws = daTWS.values
    Q = daQ.values
    #08062023, linear model
    if cfg.data.log_transform:
        Q[Q<=0] = 1e-3
        Q = np.log(Q)
        #test if log linear relationship holds, lnQ = a+b*TWS
        dfTemp = pd.DataFrame(np.c_[tws,Q],columns=('TWS', 'Q'))
        dfTemp = dfTemp.dropna()        
        X = dfTemp['TWS'].values
        Y = dfTemp['Q'].values
        X = SM.add_constant(X)
        #=--test model----
        # X = np.arange(100)
        # epsil = np.random.normal(0,0.01)
        # Y = 10*X + 2.5 + epsil
        # X = SM.add_constant(X)
        mod = SM.OLS(Y,X)
        res = mod.fit(use_t=True)
        intercept, slope = res.params
        pval = res.f_pvalue
        print ("linear regression", intercept, slope, pval)


    timestamp =daTWS.time.values
    
    if cfg.event.season=="":
        seasoncode = None 
    else:
        seasoncode = cfg.event.season

    tws_events = getExtremeEvents(pd.Series(tws.squeeze(), index=timestamp), method=cfg.event.event_method, 
                                transform=False, minDist=cfg.event.t_win, cutoff=cfg.event.cutoff,
                                returnExtremeSeries=False, season=seasoncode)
    q_events  = getExtremeEvents(pd.Series(Q.squeeze(), index=timestamp), method=cfg.event.event_method,                         
                                transform=False, minDist=cfg.event.t_win, cutoff=cfg.event.cutoff,
                                returnExtremeSeries=False, season=seasoncode)
    #this does not work for seasonal
    if seasoncode is None:
        assert (len(tws_events) == len(timestamp))

    #define quandrants of scatter plot
    #note if there's missing record, the extreme will not show up on the plots
    groups={'I':[], 'II':[], 'III':[], "IV":[]}
    
    for count, (ix, iy) in enumerate(zip(tws_events, q_events)):
        if ix == 0 and iy == 0:
            groups['I'].append([tws[count], Q[count]])
        elif ix == 0 and iy==1 :
            groups['II'].append([tws[count], Q[count]])
        elif ix==1 and iy==0:
            groups['III'].append([tws[count], Q[count]])
        else:
            groups['IV'].append([tws[count], Q[count]])

    _, ax = plt.subplots(1,1,figsize=(6,6))
    for key in groups.keys():
        groups[key] = np.array(groups[key])
        if key == 'I':
            symbolcolor='gray'
            marker = 'o'
            alpha = 0.7
            markersize = 4
            label = 'I'
        elif key == 'II':
            symbolcolor='#0484bc'            
            marker = 'o'
            alpha = 1
            markersize = 6
            label = 'II (Q)'
        elif key == 'III':
            symbolcolor='#239B56'            
            marker = 'o'
            alpha = 1
            markersize = 6
            label = 'III (TWS)'
        else:
            symbolcolor='#af0f2b'
            marker = '*'
            alpha = 1
            markersize = 10
            label = 'IV (Q^TWS)'
        if not groups[key].size == 0:
            plt.plot(groups[key][:,0], groups[key][:,1], linestyle = 'None', color=symbolcolor, markersize=markersize, marker=marker, alpha=alpha, label=label)

    if cfg.data.log_transform and pval<0.05:
        fit_Q = intercept+slope*tws
        ax.plot(tws, fit_Q, color='k', linewidth=1.0)     
        ax.text(0.02, 0.02, f'pval={pval:4.2f}', fontsize=12, transform=ax.transAxes)
        ax.set_ylabel('log-Q, [m/(km^2 s]')
    else:
        ax.set_ylabel('Q, [m/(km^2 s]')

    ax.set_xlabel('TWS [cm]')
    ax.set_title(f'Station {stationID},  {river}')
    plt.legend(loc='best')
    if cfg.data.log_transform:
        logs = "log"
    else:
        logs = ""
    if not cfg.data.deseason:
        if cfg.event.event_method == 'MAF':
            plt.savefig(f'outputs/scatter_raw_{stationID}_{cfg.event.t_win}{cfg.event.season}{logs}.eps')
        else:
            plt.savefig(f'outputs/scatter_raw_{stationID}_{cfg.event.t_win}{cfg.event.season}_{cfg.event.event_method}{cfg.event.cutoff}{logs}.png')
    else:
        if cfg.event.event_method == "MAF":
            plt.savefig(f'outputs/noseason_scatter_raw_{stationID}_{cfg.event.t_win}{cfg.event.season}{logs}.png')
        else:
            plt.savefig(f'outputs/noseason_scatter_raw_{stationID}_{cfg.event.t_win}{cfg.event.season}_{cfg.event.event_method}{cfg.event.cutoff}{logs}.png')
    plt.close()
    if cfg.data.log_transform:
        return pval

def calculateMetrics(cfg, stationID, varDict, onGloFAS=False, river=None, binary=False):
    """Main route for calculating all metrics
    Intuitively, the MI between X and Y represents the reduction in the uncertainty of Y after 
    observing X (and vice versa). Notice that the MI is symmetric, i.e., I(X; Y ) = I(Y ; X).
    Note: 06252023: form uniform 5-day time series, use -999 as missing data label
    Params
    ------
    stationID
    varDict: dictionary of variables to consider, varDict has dataarrays
    onGloFAS: True
    Returns:
    -------
    mi, Estimated mutual information between each feature and the target.
    """
    from sklearn.feature_selection import mutual_info_regression
    import sklearn.preprocessing as skp

    def genTS(arrIn, K, V):
        #generate time series for 5-day uniform intervals
        arrOut = np.zeros(daterng.shape) - 999
        arrOut[K] = arrIn[V]
        return arrOut

    #06252023: TWS,P, and Q are created using valid TWS dates from raw CSR5d
    #          Q may have NaN values
    TWS=varDict['TWS'].values
    P  =varDict['P'].values
    #06252023: save the original CSR5d dates to timestamp for later use
    timestamps = pd.to_datetime(varDict['P'].time.values)
    pval = -1
    if onGloFAS:
        Q  =varDict['Qs'].values
    else:
        Q  =varDict['Q'].values
        #========================
        #08022023, add scatter plot on raw data
        #========================
        pval = fitScatterPlot(cfg, stationID, varDict['TWS'], varDict['Q'], river=river)

    #06252023: transform the variables
    #In addition to missing CSR5D dates, we also need to account for missing Q 
    #Find index of NaN Q values and remove the same indices from all variables
    #this assumes all numpy have the same temporal order
    ind = np.where(~np.isnan(Q))[0]
    orgArr = np.zeros(Q.shape, dtype=int)
    orgArr[ind] = 1
    TWS0 = TWS[ind]
    P0 = P[ind]
    Q0 = Q[ind]    
    #==============normalization================
    doNormalization=True
    if doNormalization:
        #convert to gaussian distributions
        Q0[Q0<=0] = 1e-4
        Q0 = skp.PowerTransformer(method='box-cox').fit_transform(Q0.reshape(-1,1)).squeeze()
        TWS0 = skp.PowerTransformer().fit_transform(TWS0.reshape(-1,1)).squeeze()
        P0 = skp.PowerTransformer().fit_transform(P0.reshape(-1,1)).squeeze()

    #get mutual information (MI) using Non-Null values
    dfMI = pd.DataFrame({'TWS': TWS0, 'P': P0, 'Q': Q0})    
    mi_scores = mutual_info_regression(dfMI[['TWS', 'P']], dfMI['Q'], discrete_features='auto', n_neighbors=KNN)
    mi_scores_twsonly = mutual_info_regression(dfMI[['TWS']], dfMI['Q'], discrete_features='auto', n_neighbors=KNN)

    #06252023: replace the original values with transformed values
    TWS[ind] = TWS0
    P[ind] = P0
    Q[ind] = Q0

    #create a new daterng containing uniform 5-day dates 
    daterng = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='5D')    
    #get key/value pairs to map from daterng to timestamps
    K = []
    V = []

    for iy, t1 in enumerate(timestamps):
        for ix, t2 in enumerate(daterng):
            if abs((t1-t2).days)<3.0 and ~np.isnan(Q[iy]):
                K.append(ix)
                V.append(iy)          
    K = np.array(K)
    V = np.array(V)
    #assign values to 5-day dates, missing values are indicated by -999
    TWS = genTS(TWS, K, V)
    P = genTS(P, K, V)     
    Q = genTS(Q, K, V)

    assert(len(TWS)==len(Q) and len(P)==len(Q) )

    dfa = pd.DataFrame({'Q':Q, 'TWS':TWS, 'P': P}, index=daterng)
    #=====get CMI ===========
    cmi_score = getCMI(cfg, stationID, dfa,  river=river, saveDataFrame=True)

    #extract binary events
    season = None
    method_name = 'POT'
    Q0 = Q[Q>-999]
    TWS0 = TWS[TWS>-999]
    P0 = P[P>-999]

    cutoff = cfg.event.cutoff #this must be percent
    #07/11, here timestamps[ind] are original CSR5d dates where Q is not NaN
    #for POT, no minDIST is required
    eventQ   = getExtremeEvents(pd.Series(Q0.squeeze(), index=timestamps[ind]), method=method_name, 
               cutoff=cutoff, transform=False, season=season)
    eventTWS = getExtremeEvents(pd.Series(TWS0.squeeze(), index=timestamps[ind]), method=method_name, 
               cutoff=cutoff, transform=False,season=season)
    eventP   = getExtremeEvents(pd.Series(P0.squeeze(), index=timestamps[ind]), method=method_name, 
               cutoff=cutoff, transform=False,season=season)

    #asun06242023, form arrays with missing data -999.
    eQa = np.zeros(Q.shape)-999
    eQa[Q>-999] = eventQ
    ePa = np.zeros(P.shape)-999
    ePa[P>-999] = eventP
    eTWSa = np.zeros(TWS.shape)-999
    eTWSa[TWS>-999] = eventTWS

    #=====get Causal Links ===========
    #do binary
    dfBinary = pd.DataFrame({'Q': eQa, 'TWS': eTWSa, 'P': ePa})       
    causal_str_bin, sorted_links_bin, p_mat_bin ,val_mat_bin,ace_bin,ce_boot_bin,ce_dict  = doCausalAnalytics(cfg, dfBinary, binary=True)
    
    #do everything real-valued
    causal_str, sorted_links, p_mat ,val_mat,ace, ce_boot, ce_dict = doCausalAnalytics(cfg, dfa)

    return {'mi':mi_scores, 'cmi':cmi_score, 'mi_tws': mi_scores_twsonly,
            'causal': causal_str,
            'sorted_link': sorted_links,
            'p_mat': p_mat,
            'val_mat': val_mat,
            'causal_bin': causal_str_bin,
            'sorted_link_bin': sorted_links_bin,
            'p_mat_bin': p_mat_bin,
            'val_mat_bin': val_mat_bin,
            'ace_bin': ace_bin,
            'ace': ace,
            'ce_boot_bin': ce_boot_bin,
            'ce_boot': ce_boot,
            'pval': pval,
            'ce_dict': ce_dict
    }

def getRegionBound(region, source='GRDC'):
    """These are used for plotting Figure 1
    #asun 07012023: make the submaps equal sizes
    #region, W,  H
    #NA     80   33
    #EU    40    33
    #AU    40    30
    AF     40    30
    SA     40    30
    """
    if source=='GRDC':
        #west, south, east, north
        if region=='north_america':
            #return (-130, 25, -50, 55)
            return (-130, 25, -50, 58)
        elif region == 'south_america':
            #return (-90, -55,  -5,  0)
            return (-75, -25, -35, 5)
        elif region == 'africa':
            #return (-20, -35, 55, 40 )
            #return (-20, -35, 55, 20)
            return (10, -35, 50, -5)
        elif region == 'europe':
            #return (0, 40, 40, 80)
            return (0, 30, 40, 63)
        elif region == 'south_pacific':
            #return (110, -45, 160, -10)
            #return (130, -40, 155, -15)
            return (115, -40, 155, -10)
        elif region == 'asia':
            return (60, 8, 145, 60)
    else:
        #west, south, east, north
        if region=='north_america':
            #return (-130, 25, -50, 55)
            return (-130, 25, -50, 58)
        elif region == 'south_america':
            #return (-90, -55,  -5,  0)
            return (-90, -55, -5, 10)
        elif region == 'africa':
            #return (-20, -35, 55, 40 )
            #return (-20, -35, 55, 20)
            return (-10, -35, 50, 20)
        elif region == 'europe':
            #return (0, 40, 40, 80)
            return (-10, 35, 60, 70)
        elif region == 'south_pacific':
            #return (110, -45, 160, -10)
            #return (130, -40, 155, -15)
            return (115, -40, 155, -10)
        elif region == 'asia':
            return (60, 8, 145, 60)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plotGlobalMap(allScores, region, gdfCatalog, dfStation, feature_column='TWS_MI', onGloFAS=False):
    """Plot global map [NOT USED!!!]
    Params
    ------
    allScores, dictionary of metrics
    region, region to be plotted
    gdfCatalog, catalog of stations with metadata
    dfStation, dataframe of station
    onGloFAS, plot related to glofas
    """

    #https://www.sc.eso.org/~bdias/pycoffee/codes/20160602/colorbar_demo.html
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.axes as maxes
    import matplotlib as mpl
    from shapely.geometry import Polygon

    gdfAll = []
    dfStation = gpd.GeoDataFrame(dfStation, geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
    
    for key,val in allScores.items():
        if key in dfStation['grdc_no'].to_list():            
            gdf = getBasinMask(stationID=key, region=region, gdf=gdfCatalog)
            if 'mi' in val.keys():
                gdf['TWS_MI'] = val['mi'][0]
                gdf['P_MI'] = val['mi'][1]
                gdf['T_MI'] = val['mi'][2]
                gdf['TWS_CMI'] = val['cmi']
                gdf['cmi_diff'] = val['cmi'] -  val['mi'][0]
            if 'NSE' in val.keys():
                gdf['NSE'] = val['NSE']
            if 'grdc_tws_tau' in val.keys():
                gdf['grdc_tws_tau'] = val['grdc_tws_tau']
            if 'grdc_glofas_tau' in val.keys():
                gdf['grdc_glofas_tau'] = val['grdc_glofas_tau']
            if 'grdc_precip_tau' in val.keys():
                gdf['grdc_precip_tau'] = val['grdc_precip_tau']
            if 'glofas_tws_tau' in val.keys():
                gdf['glofas_tws_tau'] = val['glofas_tws_tau']

            if 'grdc_tws_tau_ex' in val.keys():
                gdf['grdc_tws_tau_ex'] = val['grdc_tws_tau_ex']
            if 'grdc_glofas_tau_ex' in val.keys():
                gdf['grdc_glofas_tau_ex'] = val['grdc_glofas_tau_ex']
            if 'grdc_precip_tau_ex' in val.keys():
                gdf['grdc_precip_tau_ex'] = val['grdc_precip_tau_ex']
            if 'glofas_tws_tau_ex' in val.keys():
                gdf['glofas_tws_tau_ex'] = val['glofas_tws_tau_ex']

            gdfAll.append(gdf)
            
    gdfAll = pd.concat(gdfAll)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))    
    
    if region !='global':
        
        #lon0, lat0, lon1, lat1 = gdfAll.total_bounds
        lon0, lat0, lon1, lat1 = getRegionBound(region)
        #world = world.clip(Polygon([(lon0, lat0), (lon1,lat0), (lon1, lat1), (lon0, lat1), (lon0, lat0)]))
    else:
        lon0 = None
    crs_new = ccrs.PlateCarree()
    world = world.to_crs(crs_new)
    gdfAll = gdfAll.to_crs(crs_new)
    dfStation = dfStation.to_crs(crs_new)

    #10/17, note gdfAll is sorted by decreasing areas
    gdfAll = gdfAll.sort_values(by='area_hys') #, ascending=False)
    #
    # as 12/10/2022, filter by area_hys
    #     
    gdfAll = gdfAll[gdfAll['area_hys']>MIN_BASIN_AREA]

    fig = plt.figure(figsize=(12,12))    
    
    ax = fig.add_subplot(1, 1, 1, projection=crs_new)
    
    divider = make_axes_locatable(ax)
    if feature_column == 'NSE':
        #cmap = 'YlGnBu'
        cmap = discrete_cmap(10, base_cmap='YlGnBu')
    elif feature_column in ['grdc_tws_tau', 'grdc_glofas_tau', 'grdc_precip_tau', 'glofas_tws_tau',
                            'grdc_tws_tau_ex', 'grdc_glofas_tau_ex', 'grdc_precip_tau_ex', 'glofas_tws_tau_ex']:
        cmap = 'YlGnBu'
    else:
        #cmap = cc.cm.fire_r
        cmap = discrete_cmap(5, base_cmap='YlGnBu')

    cax = divider.append_axes("right", size="5%", axes_class=maxes.Axes, pad=0.05)

    gdfAll.plot(column=feature_column, ax=ax, alpha=0.7, legend=True, cax=cax, 
            vmin=0, vmax=1.0, cmap=cmap, 
            legend_kwds={'label': feature_column})

    world.plot(ax=ax, alpha=0.5, facecolor="none", edgecolor='black')
    if not lon0 is None:
        ax.set_extent((lon0, lon1, lat0, lat1), crs_new)
    # Plot lat/lon grid 
    gl = ax.gridlines(crs=crs_new, draw_labels=True,
                  linewidth=0.1, color='k', alpha=1, 
                  linestyle='--')
    #world_clipped.boundary.plot(ax=ax, color='gray', alpha=0.5)
    #dfStation.plot(ax=ax, facecolor="none", edgecolor="black")
    #ax.coastlines(resolution='110m', color='gray')
    #ax.set_aspect('equal')
    if onGloFAS:
        ax.set_title(f'GloFAS {regionNames[region]}')
    else:
        ax.set_title(f'GRDC {regionNames[region]}')

    if onGloFAS:
        plt.savefig(f'grdcresults/glofas_{region}_{feature_column}.png')
    else:
        plt.savefig(f'grdcresults/{region}_{feature_column}.png')
    plt.close()

def getStationMeta(dfRegion):
    grdcfile = '/home/suna/work/grace/data/grdc/GRDC_Stations.csv'
    df = pd.read_csv(grdcfile )
    #join with dfRegion to get missing columns
    dfJoin = pd.merge(
        df,
        dfRegion,
        how="inner",
        on='grdc_no',
        sort=True,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    )
    dfJoin = dfJoin.sort_values(by='area_hys', ascending=False)
    return dfJoin

def getGloFASStations(dfRegion):
    grdcfile = '/home/suna/work/grace/data/grdc/GRDC_Stations.csv'    
    df = pd.read_csv(grdcfile)    
    #join with dfRegion to get missing columns
    dfJoin = pd.merge(
        df,
        dfRegion,
        how="inner",
        on='grdc_no',
        sort=True,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    )
    glofasfile = '/home/suna/work/grace/data/grdc/Harrigan_et_al_Supplementary_Table_S1.csv'
    df2 = pd.read_csv(glofasfile)
    #rename Provider_ID to grdc_no
    df2 = df2.rename(columns={"Provider_ID": "grdc_no"})
    dfJoin = pd.merge(
        df2,
        dfJoin,
        how="inner",
        on='grdc_no',
        sort=True,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    )

    dfJoin = dfJoin.sort_values(by='area_hys', ascending=False)
    dfJoin['grdc_no'] = dfJoin['grdc_no'].astype('int32')
    print (dfJoin.shape)
    #print (dfJoin.columns)
    return dfJoin


def plotAllRegions(cfg, features=['TWS']):
    """plot attribute and basin shapes
    """
    globalGDF=[]
    allMIScores={}
    #get the maximum MI

    dfStations = []
    for region in cfg.data.regions:
        rootdir = '/home/suna/work/grace/data/grdc'
        geosonfile = os.path.join(rootdir, f'{region}/stationbasins.geojson')
        gdf = gpd.read_file(geosonfile)
        gdf['grdc_no'] = pd.to_numeric(gdf['grdc_no'], downcast='integer')
        globalGDF.append(gdf)
        miscores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores.pkl', 'rb'))
        allMIScores = allMIScores | miscores
        dfStations.append(getStationMeta(gdf))
    globalGDF = pd.concat(globalGDF, axis=0)
    dfStations= pd.concat(dfStations, axis=0)
    for feature in features:
        plotGlobalMap(allScores=allMIScores, gdfCatalog=globalGDF, region='global', 
                dfStation=dfStations, 
                feature_column=feature)

def getCatalog(region,source='grdc'):
    """Get catalog of GRDC stations
    """
    rootdir = f'/home/suna/work/grace/data/{source}'
    geosonfile = os.path.join(rootdir, f'{region}/stationbasins.geojson')

    gdf = gpd.read_file(geosonfile)
    gdf['grdc_no'] = pd.to_numeric(gdf['grdc_no'], downcast='integer')
    #as 09062023, the newer stationbasins.geojson files do not use area_hys anymore
    #so i rename the column to make the code work
    if 'area' in gdf.columns:
        gdf = gdf.rename(columns={'area':'area_hys'})    
    return gdf

def compareFlowRates(river, stationID, df, df2,nse):
    """Compare GRDC observed and Glofas simulated flow rates for a station
    """
    fig,ax = plt.subplots(1,1)
    df.plot(ax=ax, color='b', label='GLOFAS')
    df2.plot.line(ax=ax, color='r', label='GRDC')
    ax.set_title(f'{river}, NSE: {nse:3.2f}')
    plt.legend()
    plt.savefig(f'grdcresults/grdc_glofas/{stationID}_plot.png')
    plt.close()

def genGloFASMetrics(cfg, region, reGen=False):
    """Generate CMI metrics between GloFAS and CSR5d [not used]
    """
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    gdf = getCatalog(region, source='glofas')
    if 'area' in gdf.columns:
        gdf = gdf.rename(columns={'area':'area_hys'})

    #0908 replace with GloFAS stations
    dfStation = getGloFASStations(gdf)
    if reGen:
        dsGloFAS = loadGLOFAS4CONUS()
        extents = glofas_us.getExtents()
        daterng = pd.date_range(start='2002/01/01', end='2020/12/31', freq='1D')
        allScores = {}
        xds = None

        for ix, row in dfStation.iterrows():
            #0908, replace with GloFAS station data
            #lat,lon = float(row['lat']), float(row['long'])
            lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])
            print (lat, lon)

            if row['area_hys']>=cfg.data.min_basin_area and (lat>extents['min_lat'] and lat<extents['max_lat']):
                if lon>extents['min_lon'] and lon<=extents['max_lon']:
                    stationID = row['grdc_no']
                    riverName = row['river_x']
                    print ('stationID', stationID, riverName)

                    #get glofas data between 2002 and 2020                
                    daGloFAS = extractBasinOutletQ(loc=(lat,lon), ds=dsGloFAS)
                    daGloFAS = daGloFAS.squeeze() #this needs to be series 
                    try:
                        dfGRDC = readStationSeries(stationID=stationID, region=region)
                        dfGRDC = dfGRDC[dfGRDC.index.year>=2002].copy(deep=True)

                        #this step introduces NaN!!!
                        dfGRDC = dfGRDC.reindex(daterng)
                        #convert da to df
                        dfGloFAS = pd.DataFrame(daGloFAS.values.squeeze(), index=daGloFAS.time, columns=['GloFAS'])
                        print ('data len', len(dfGRDC), len(dfGloFAS))
                        nse = HydroStats.nse(dfGloFAS['GloFAS'].to_numpy(), dfGRDC['Q'].to_numpy())
                        print ('NSE=', nse)
                    except Exception as e:
                        nse = -99

                    if xds is None:
                        #only store xds on first call
                        basinTWS, basinP, xds, xds_Precip = getBasinData(config=cfg, stationID=stationID, 
                                                                        lat=lat, lon=lon, gdf=gdf, 
                                                                        region=region,
                                                                        removeSWE=cfg.data.removeSWE
                                                                        )
                    else:
                        basinTWS, basinP, _, _ = getBasinData(config=cfg, stationID=stationID, \
                                                            region=region, gdf=gdf, lat=lat, lon=lon, \
                                                            xds=xds, xds_Precip=xds_Precip,
                                                            removeSWE=cfg.data.removeSWE)


                    kwargs = {
                        "method": "rx5d",
                        "aggmethod":'max',
                        "name":'Q'
                        }

                    daGloFAS = getCSR5dLikeArray(daGloFAS, basinTWS, **kwargs)
                    #06262023, normalize by drainage area
                    daGloFAS = daGloFAS/row['area_hys']

                    varDict={'TWS':basinTWS, 'P':basinP, 'Qs': daGloFAS}
                    
                    metricDict = calculateMetrics(cfg, stationID, varDict, onGloFAS=True, river=riverName)
                    metricDict['NSE'] = nse
                    print (f"{stationID},{lat},{lon},{row['river_x']},{row['area_hys']}") # 'MI', {metricDict['mi']}, 'CMI', {metricDict['cmi']}")
                    print (f"CMI {metricDict['cmi']}")
                    allScores[stationID] = metricDict

        if  cfg.causal.exclude_Q:        
            if not cfg.data.deseason:
                pkl.dump(allScores, open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'wb'))
            else:
                pkl.dump(allScores, open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'wb'))
        else:
            if not cfg.data.deseason:
                pkl.dump(allScores, open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q.pkl', 'wb'))
            else:
                pkl.dump(allScores, open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q_noseason.pkl', 'wb'))
    else:
        if  cfg.causal.exclude_Q:                  
            if not cfg.data.deseason:
                allScores = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))
            else:
                allScores = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))
        else:
            if not cfg.data.deseason:
                allScores = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q.pkl', 'rb'))
            else:
                allScores = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q_noseason.pkl', 'rb'))

def compareExtremeEvents(region, reGen=False):
    """extract and compare extreme events
    """
    def getAnnualMaximaSeries(dsIn:pd.Series):
        return dsIn.groupby(dsIn.index.year).max()

    def extractEvents(ts, timeaxis, varname, **kwargs):
        """Extract events from the time series
        Params
        ------
        ts, numpy array
        timeaxis, array of dates

        Returns
        -------
        percentile
        extracted events
        """
        assert(type(ts)== np.ndarray)
        
        kwargs.setdefault('pcnt', None)
        if kwargs['method'] == "annual_maxima":
            #implementation of the annual maxima method
            RT = kwargs['RT']
            quantile = 1.0 - 1.0/RT
            annualmaxima = getAnnualMaximaSeries(pd.Series(ts,index=timeaxis))
            loc,scale = gumbel_r.fit(annualmaxima.values)
            pcntile = gumbel_r.ppf(quantile,loc,scale)
        
        elif kwargs['method'] == "pot":
            #implementation of the Peak of Threshold
            assert(not kwargs['pcnt'] is None)
            pcnt = kwargs['pcnt']
            if pcnt<1.0:
                print ("warning: convert pcnt to percentage")
                pcnt = pcnt*100.0
            pcntile = np.nanpercentile(ts,pcnt)
            print (varname, 'percentile', pcntile)

        events = np.where(ts>pcntile)[0]
        return pd.Series(ts[events], index = timeaxis[events]), pcntile

    gdf = getCatalog(region)
    #0908 replace glofas data
    #dfStation = getStationMeta(gdf)            
    dfStation = getGloFASStations(gdf)

    if reGen:
        allScores = {}
        dsGloFAS = loadGLOFAS4CONUS()
        extents = glofas_us.getExtents()
        daterng = pd.date_range(start='2002/01/01', end='2020/12/31', freq='1D')
        allScores = {}
        threshold = 0.9 #means 10% missing data during study period
        
        xds = None
        
        #iterate on GRDC stations
        for ix, row in dfStation.iterrows():     
            #0908, replace with GloFAS station data
            #lat,lon = float(row['lat']), float(row['long'])
            lat,lon = float(row['Latitude_GloFAS']), float(row['Longitude_GloFAS'])
            print (lat, lon)            

            #only process the station if it falls within the region extent
            if lat>extents['min_lat'] and lat<extents['max_lat']:
                if lon>extents['min_lon'] and lon<=extents['max_lon']:

                    stationID = row['grdc_no']
                    riverName = row['river_x']
                    print ('stationID', stationID)

                    #get glofas data between 2002 and 2020                
                    #@todo: double check 
                    daGloFAS = extractBasinOutletQ(loc=(lat,lon), ds=dsGloFAS)
                    daGloFAS = daGloFAS.squeeze() #this needs to be time series 
                    try:
                        dfGRDC = readStationSeries(stationID=stationID, region=region)
                        dfGRDC = dfGRDC[dfGRDC.index.year>=2002].copy(deep=True)

                        #this step introduces NaN!!!
                        dfGRDC = dfGRDC.reindex(daterng)
                        #convert da to df
                        dfGloFAS = pd.DataFrame(daGloFAS.values.squeeze(), index=daGloFAS.time, columns=['GloFAS'])
                        print ('data len', len(dfGRDC), len(dfGloFAS))
                        #first NaN removal 
                        ind = np.where(~np.isnan(dfGRDC.values))[0]
                        dfGRDC = dfGRDC.iloc[ind]
                        dfGloFAS = dfGloFAS.iloc[ind]
                        print ('data len after removing nan', len(dfGRDC), len(dfGloFAS))
                        
                        nse = HydroStats.nse(dfGloFAS['GloFAS'].to_numpy(), dfGRDC['Q'].to_numpy())
                        print ('NSE=', nse)                        
                        #count number of valid values
                        print (1- dfGRDC.isnull().sum().values/len(daterng))
                        if (1- dfGRDC.isnull().sum().values/len(daterng))>threshold:
                            #second NaN removal
                            dfGRDC = dfGRDC.dropna()
                            
                            daGRDC = xr.DataArray(data=dfGRDC['Q'].values, dims=['time'], coords={'time':dfGRDC.index})

                            if xds is None:
                                basinTWS, basinP, basinT,xds, xds_Precip, xds_T = getBasinData(stationID=stationID, lat=lat, lon=lon, gdf=gdf, region=region)
                            else:
                                basinTWS, basinP, basinT, _, _, _ = getBasinData(stationID=stationID, region=region, gdf=gdf, lat=lat, lon=lon, xds=xds, xds_Precip=xds_Precip, xds_T=xds_T)

                            kwargs = {
                                "method": "rx5d",
                                "aggmethod":'average',
                                "name":'Q'
                                }
                            #this steps may introduce NaN values, because of missing dates
                            daGRDC = getCSR5dLikeArray(daGRDC, basinTWS, **kwargs)
                            daGloFAS = getCSR5dLikeArray(daGloFAS, basinTWS, **kwargs)
                            #third NaN removal
                            print ('before', daGRDC.shape, daGloFAS.shape, basinP.shape, basinTWS.shape)
                            ind = np.where(~(np.isnan(daGRDC.values)))[0]
                            daGRDC  =daGRDC.isel(time=ind)
                            daGloFAS=daGloFAS.isel(time=ind)
                            basinP = basinP.isel(time=ind)
                            basinTWS = basinTWS.isel(time=ind)
                            print ('after', daGRDC.shape, daGloFAS.shape, basinP.shape, basinTWS.shape)
                            #calculate Kendall Tau on 5d extreme events
                            ukwargs={'method':'pot',
                                     'pcnt': 90.0
                            }
                            #use the CSR5d dates
                            timeaxis = basinTWS['time'].values
                            twsEvents, _      = extractEvents(basinTWS.values, timeaxis, 'tws', **ukwargs)
                            PrecipEvents, _   = extractEvents(basinP.values, timeaxis, 'precip', **ukwargs )
                            GRDC_QEvents, _   = extractEvents(daGRDC.values, timeaxis, 'grdc', **ukwargs)
                            GloFAS_QEvents, _ = extractEvents(daGloFAS.values, timeaxis, 'glofas',**ukwargs)
                            #AT this time, the number of events may be different
                            #because these are all sorted by time
                            #we can drop unmatched in others
                            nEvents = min([len(twsEvents), len(GRDC_QEvents), len(GloFAS_QEvents)])
                            PrecipEvents = PrecipEvents.iloc[:nEvents]
                            GloFAS_QEvents = GloFAS_QEvents.iloc[:nEvents]
                            twsEvents = twsEvents.iloc[:nEvents]
                            GRDC_QEvents = GRDC_QEvents.iloc[:nEvents]
                            #======tau on extreme events=================
                            #calculate tau between TWS and GRDC
                            tauGRDC_TWS_extreme, tws_pval = kendalltau(twsEvents.values, GRDC_QEvents.values)
                            #calculate tau between GRDC and GloFAS
                            tauGRDC_GloFAS_extreme, glofas_pval = kendalltau(GRDC_QEvents.values, GloFAS_QEvents.values)
                            #calculate tau between GRDC and P
                            tauGRDC_Precip_extreme, precip_pval = kendalltau(GRDC_QEvents.values, PrecipEvents.values)
                            #calculate tau between TWS and GloFAS
                            tauGlosFAS_TWS_extreme, twsg_pval = kendalltau(twsEvents.values, GloFAS_QEvents.values)

                            #======tau on everything =================
                            #calculate tau between TWS and GRDC
                            tauGRDC_TWS, tws_pval = kendalltau(basinTWS.values, daGRDC.values)
                            #calculate tau between GRDC and GloFAS
                            tauGRDC_GloFAS, glofas_pval = kendalltau(daGRDC.values, daGloFAS.values)
                            #calculate tau between GRDC and P
                            tauGRDC_Precip, precip_pval = kendalltau(daGRDC.values, basinP.values)
                            #calculate tau between TWS and GloFAS
                            tauGlosFAS_TWS, twsg_pval = kendalltau(basinTWS.values, daGloFAS.values)

                            print ('tauGRDC_TWS', tauGRDC_TWS, 'tauGRDC_GloFAS', tauGRDC_GloFAS, 'tauGRDC_Precip', tauGRDC_Precip, 'tauGlosFAS_TWS', tauGlosFAS_TWS)
                            allScores[stationID] = {
                                'grdc_tws_tau':    tauGRDC_TWS, 
                                'grdc_glofas_tau': tauGRDC_GloFAS,
                                'grdc_precip_tau': tauGRDC_Precip,
                                'glofas_tws_tau':  tauGlosFAS_TWS,
                                'grdc_tws_tau_ex':    tauGRDC_TWS_extreme, 
                                'grdc_glofas_tau_ex': tauGRDC_GloFAS_extreme,
                                'grdc_precip_tau_ex': tauGRDC_Precip_extreme,
                                'glofas_tws_tau_ex':  tauGlosFAS_TWS_extreme,
                            }
                    
                    except Exception as e:
                        raise Exception (e)
        pkl.dump(allScores, open(f"grdcresults/{region}_Tau_{ukwargs['pcnt']}.pkl", 'wb'))
    else:
        allScores = pkl.load(open(f"grdcresults/{region}_Tau_{ukwargs['pcnt']}.pkl", 'rb'))
    
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='grdc_tws_tau') 
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='grdc_glofas_tau') 
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='grdc_precip_tau') 
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='glofas_tws_tau') 

    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='grdc_tws_tau_ex') 
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='grdc_glofas_tau_ex') 
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='grdc_precip_tau_ex') 
    plotGlobalMap(allScores, region=region, gdfCatalog=gdf, dfStation=dfStation, feature_column='glofas_tws_tau_ex') 


def main(cfg, region):
    """revised 07142023
    note: for REGEN precip data, the end date should 12/31/2016
    for all others, the end date is 2021/12/31
    date: 07142023: revise data to make sure each year has more than 70% data available

    Params:
    cfg, configuration yaml
    region, the region to be analyzed
    """    
    gdf = getCatalog(region)
    print (gdf)
    dfStation = getStationMeta(gdf)
    reGen = cfg.regen
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    if reGen:
        allScores = {}
        xds = None
        daterng = pd.date_range(start=cfg.data.start_date, end=cfg.data.end_date, freq='1D')
        
        #print ("stationid,lat,lon,river,area,data_coverage")
        goodStations=0
        for ix, row in dfStation.iterrows():
            stationID = row['grdc_no']

            lat,lon = float(row['lat']), float(row['long'])
            try:
                df = readStationSeries(stationID=stationID, region=region)
                df = df[df.index.year>=2002].copy(deep=True)
                
                #make all time series have the same date length
                #after this step, NaNs will exist when data is missing!!!
                df = df.reindex(daterng)

                #count number of valid values in each year in fractions (not percent !!!!)
                res = df.groupby(df.index.year).agg({'count'})/365.0
                resdf = res['Q']
                #find completeness in each valid gages should have empty resdf
                resdf = resdf[resdf['count']<cfg.data.year_threshold]

                #only process gages have sufficient number of data and big enough area
                if resdf.empty and row['area_hys']>=cfg.data.min_basin_area:
                    #print (stationID, row['river_x'])
                    goodStations+=1
                    #drop NaN
                    df = df.dropna()
                    #06262023, normalize by drainage area
                    daQ = xr.DataArray(data=df['Q'].values/row['area_hys'], dims=['time'], coords={'time':df.index})
                    if xds is None:
                        #only store xds on first call
                        basinTWS, basinP, xds, xds_Precip = getBasinData(config=cfg, 
                                                            stationID=stationID, 
                                                            lat=lat, lon=lon, gdf=gdf, 
                                                            region=region,
                                                            removeSWE=cfg.data.removeSWE
                                                            )
                    else:
                        basinTWS, basinP, _, _ = getBasinData(config=cfg, 
                                                            stationID=stationID, 
                                                            region=region, gdf=gdf, lat=lat, lon=lon, 
                                                            xds=xds, xds_Precip=xds_Precip,
                                                            removeSWE=cfg.data.removeSWE)
                    #08042023, choose between max or sum
                    aggmethod = 'max'
                    if aggmethod =='sum':
                        kwargs = {
                            "method": "rx5d",
                            "aggmethod":'sum',
                            "name":'Q'
                            }
                        #convert Q to CSR5d intervals
                        #this steps may introduce NaN values
                        daQ = getCSR5dLikeArray(daQ, basinTWS, **kwargs)*86400
                    else:
                        kwargs = {
                            "method": "rx5d",
                            "aggmethod":'max',
                            "name":'Q'
                            }
                        #convert Q to CSR5d intervals
                        #this steps may introduce NaN values
                        daQ = getCSR5dLikeArray(daQ, basinTWS, **kwargs)

                    if cfg.data.deseason:
                        daQ      = removeClimatology(daQ, varname='Q', plotting=False)
                        basinTWS = removeClimatology(basinTWS, varname='TWS',plotting=False)
                        basinP   = removeClimatology(basinP, varname='P',plotting=False)
                    
                    varDict={'TWS':basinTWS, 'P':basinP, 'Q':daQ}

                    metricDict = calculateMetrics(cfg,stationID, varDict, river=row['river_x'])
                    print (f"{stationID},{lat},{lon},{row['river_x']},{row['area_hys']}") # 'MI', {metricDict['mi']}, 'CMI', {metricDict['cmi']}")
                    print (f"CMI {metricDict['cmi']}")
                    allScores[stationID] = metricDict
            except Exception as e: 
                raise Exception (e)

        #07312023, add test for no seasonality
        #08032023, deseason is not meaningful [I'll keep the code to avoid issues]
        #08032023, add event_method to pkl name [for MAF, the cutoff does not mean anything]
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:                
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'wb'))
            else:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'wb'))
        else:
            if not cfg.data.deseason:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q.pkl', 'wb'))
            else:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q_noseason.pkl', 'wb'))
    else:
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))
            else:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))
        else:
            if not cfg.data.deseason:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q.pkl', 'rb'))
            else:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q_noseason.pkl', 'rb'))

def plotBasinShapes(region):
    """Plot basin shapes
    """
    rootdir = '/home/suna/work/grace/data/grdc'
    geosonfile = os.path.join(rootdir, f'{region}/stationbasins.geojson')
    gdf = gpd.read_file(geosonfile)
    gdf['grdc_no'] = pd.to_numeric(gdf['grdc_no'], downcast='integer')
    basinAreas = gdf['area_hys'].to_numpy()
    print ('minimum area', np.min(basinAreas))
    print ('before filtering', gdf.shape)
    gdf = gdf[gdf['area_hys']>MIN_BASIN_AREA]
    print ('after filteringj', gdf.shape)

    allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores.pkl', 'rb'))

    fig = plt.figure(figsize=(12,12))    
    ax = fig.add_subplot(1, 1, 1, projection= ccrs.PlateCarree())
    selectedDF = gdf[gdf['grdc_no'].isin(allScores.keys())]
    #plot basin by basin
    for _, row in selectedDF.iterrows():
        ax.add_geometries([row['geometry']], crs=ccrs.PlateCarree(), \
                      facecolor="None", edgecolor='cornflowerblue', alpha=0.7)

    #selectedDF.plot(ax=ax, alpha=0.7, facecolor="none", edgecolor='cornflowerblue')
    print ('selected DF shaped', selectedDF.shape)
    lon0, lat0, lon1, lat1 = getRegionBound(region)
    ax.set_extent((lon0, lon1, lat0, lat1), ccrs.PlateCarree())

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))        
    world = world.to_crs(ccrs.PlateCarree())
    world.plot(ax=ax, alpha=0.5, facecolor="none", edgecolor='black')

    plt.title('CONUS GRDC Basins')
    plt.savefig(f"{region}_basinpolygons.png")
    plt.close()

def plotKoppenMap(cfg, region):
    """This is replaced by plotBivariate
    asun06202023: 
    """
    # Use geopandas for vector data and xarray for raster data
    import geopandas as gpd
    import rioxarray as rxr
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    #get global basin shape
    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    elif cfg.maps.global_basin == 'hydroshed':
        shpfile = os.path.join('maps/hydrosheds', hydroshedMaps[region])
    gdfBasinBound = gpd.read_file(shpfile)

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    rootdir = '/home/suna/work/predcsr5d'
    koppenmap = rxr.open_rasterio(os.path.join(rootdir, 'data/koppenclimate/koppen5.tif'), masked=True)
    
    print("The CRS for this data is:", koppenmap.rio.crs)
    print("The spatial extent is:", koppenmap.rio.bounds())

    lon0, lat0, lon1, lat1 = getRegionBound(region)
    koppenmap = koppenmap.sortby(koppenmap.y)
    koppenmap_na = koppenmap.sel(y=slice(lat0,lat1),x=slice(lon0,lon1))
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""
    
    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))
    validStations = list(outputDict.keys())

    cmi=[]
    for stationID in validStations:
        stationDict = outputDict[stationID]            
        cmi.append(stationDict['cmi'])       

    print ('no cmi', len(cmi))
    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), 'CMI':np.array(cmi)})
    dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))    
    cmap = sns.color_palette("deep", 5)    
    im = koppenmap_na.plot(levels=[1, 2, 3, 4, 5, 6], colors=cmap, alpha=0.60, ax=ax, add_colorbar=False)
    ax.set_title('')    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    cb = plt.colorbar(im, ticks=range(1,7), orientation="horizontal", pad=0.01,shrink=0.75)
    cb.set_label(label='')
    cb.ax.tick_params(labelsize=15)
    cb.ax.set_xticklabels(["", 'Tropical', 'Arid', 'Temperate', 'Cold', 'Polar'])
    
    gdfBasinBound.plot(ax=ax, facecolor="None", edgecolor="w", legend=False)       
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    cmap = ListedColormap(sns.color_palette(palette="rocket_r", n_colors=6).as_hex())    
    gdfCol = gpd.GeoDataFrame(dfStation['CMI'], geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
    gdfCol.plot(column='CMI', ax=ax, cax=cax, cmap=cmap, marker='o', markersize= 100, alpha=0.8,
                    legend_kwds={'shrink': 0.5, 'label': f"CMI", 'extend':'max'}, legend=True)

    plt.savefig(f'grdc_koppen_plot_{region}.png')
    plt.close()

def plotJRPMap(cfg, region, varname='TWS'):
    """
    asun06202023: plot joint probability of Q and varname
    """
    
    import seaborn as sns
    # Use geopandas for vector data and xarray for raster data
    import geopandas as gpd
    import rioxarray as rxr
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import ListedColormap

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    assert(varname in ['TWS', 'P'])

    #get global basin shape
    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    elif cfg.maps.global_basin == 'hydroshed':
        shpfile = os.path.join('maps/hydrosheds', hydroshedMaps[region])
    gdfBasinBound = gpd.read_file(shpfile)

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    rootdir = '/home/suna/work/predcsr5d'
    koppenmap = rxr.open_rasterio(os.path.join(rootdir, 'data/koppenclimate/koppen5.tif'), masked=True)
    
    print("The CRS for this data is:", koppenmap.rio.crs)
    print("The spatial extent is:", koppenmap.rio.bounds())

    lon0, lat0, lon1, lat1 = getRegionBound(region)
    koppenmap = koppenmap.sortby(koppenmap.y)
    koppenmap_na = koppenmap.sel(y=slice(lat0,lat1),x=slice(lon0,lon1))
    if not cfg.data.deseason:
        outputDict = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
    else:
        outputDict = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
    validStations = list(outputDict.keys())

    arr=[]
    for stationID in validStations:
        stationDict = outputDict[stationID]            
        arr.append(stationDict[varname])       

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), f'Q_{varname}':np.array(arr)})
    dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
    
    fig, ax = plt.subplots(1,1, figsize=(20,15))    
    #cmap = sns.color_palette("deep", 5)    
    cmap = sns.color_palette(["#93AACF", "#FFD54F", "#DCEDC8", "#CFD8DC", "#E1BEE7"])
    im = koppenmap_na.plot(levels=[1, 2, 3, 4, 5, 6], colors=cmap, alpha=0.60, ax=ax, add_colorbar=False)
    ax.set_title('')    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    cb = plt.colorbar(im, ticks=range(1,7), orientation="horizontal", pad=0.01, shrink=0.75)
    cb.set_label(label='')
    cb.ax.tick_params(labelsize=15)
    cb.ax.set_xticklabels(["", 'Tropical', 'Arid', 'Temperate', 'Cold', 'Polar'])
    
    gdfBasinBound.plot(ax=ax, facecolor="None", edgecolor="w", legend=False)       
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)

    cmap = ListedColormap(sns.color_palette(palette="rocket_r", n_colors=6).as_hex())    
    gdfCol = gpd.GeoDataFrame(dfStation[f'Q_{varname}'], geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
    gdfCol.plot(column=f'Q_{varname}', ax=ax, cax=cax, cmap=cmap, marker='o', markersize= 100, alpha=0.8,
                    legend_kwds={'shrink': 0.5, 'label': f"Q_{varname}", 'extend':'max'}, legend=True)

    plt.savefig(f'joint_{varname}_q_plot_{region}.png')
    plt.close()


def hex_to_Color(hexcode):
    from PIL import ImageColor

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

def genSingleColorList(num_grps):
    c01 = hex_to_Color('#F1948A') #light orange
    c11 = hex_to_Color('#E74C3C')
    colorlist  = []
    for i in range(num_grps):
        colorlist.append(c01.lerp(c11, 1/(num_grps-1) * i))
    colorlist = [rgb2hex([c.r, c.g, c.b]) for c in colorlist]
    return colorlist

def plotBivariate(cfg, region, ax, source='grdc', variable2='P_Q', use_percentile=False, ax_legend=None):
    """Reference: https://waterprogramming.wordpress.com/2022/09/08/bivariate-choropleth-maps/
    Parameters
    source, can be either 'grdc' or 'glofas'
    variable2: can be either CMI or P_Q
    """
    import seaborn as sns
    # Use geopandas for vector data and xarray for raster data
    import geopandas as gpd
    import rioxarray as rxr
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import ListedColormap

    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection    
    
    #08022023, I don't use CMI anymore
    assert(variable2 in ['CMI', 'P_Q'])
    if variable2 == 'CMI':
        dictkey = 'cmi'
    else:
        dictkey = 'P'

    #choose the global basin shapefile
    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    elif cfg.maps.global_basin == 'hydroshed':
        shpfile = os.path.join('maps/hydrosheds', hydroshedMaps[region])
    gdfBasinBound = gpd.read_file(shpfile)

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    rootdir = '/home/suna/work/predcsr5d'
    koppenmap = rxr.open_rasterio(os.path.join(rootdir, 'data/koppenclimate/koppen5.tif'), masked=True)  
    print("The CRS for this data is:", koppenmap.rio.crs)
    print("The spatial extent is:", koppenmap.rio.bounds())
    lon0, lat0, lon1, lat1 = getRegionBound(region)
    koppenmap = koppenmap.sortby(koppenmap.y)
    koppenmap_region = koppenmap.sel(y=slice(lat0,lat1),x=slice(lon0,lon1))

    #fig, ax = plt.subplots(1,1, figsize=figsizes(20,18))
    #
    #plot the Koppen climate map
    #note how colorbar is set up: level is 1 more than number of categories
    #

    cmap = sns.color_palette(["#85C1E9", "#FFD54F", "#DCEDC8", "#CFD8DC", "#E1BEE7"])
    im = koppenmap_region.plot(levels=[1, 2, 3, 4, 5, 6], colors=cmap, alpha=0.60, ax=ax, add_colorbar=False)
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

    if region=='north_america':
        bounds = np.arange(0.5, 6.5)
        cb = plt.colorbar(im, cax=ax_legend, ticks=bounds, orientation="horizontal", shrink=0.7)
        #cb.set_ticks([bounds[0]] + [(b0 + b1) / 2 for b0, b1 in zip(bounds[:-1], bounds[1:])] + [bounds[-1]])
        cb.set_label(label='')
        cb.ax.tick_params(labelsize=15,length=0)
        cb.ax.set_xticklabels(["", 'Tropical', 'Arid', 'Temperate', 'Cold', 'Polar'])

    #plot global basin boundaries
    gdfBasinBound.plot(ax=ax, facecolor="none", edgecolor="w", legend=False)       
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""
    
    #plot bivariates
    if source=='grdc':
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
            else:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
                
        else:
            if not cfg.data.deseason:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}_Q.pkl', 'rb'))
            else:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q_noseason.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}_Q.pkl', 'rb'))

        validStations = list(outputDict.keys())
            
    elif source=='glofas':
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                outputDict = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))                    
                outputDict2 = pkl.load(open(f'grdcresults/glofas_{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
            else:
                outputDict = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))                    
                outputDict2 = pkl.load(open(f'grdcresults/glofas_{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
        else:
            outputDict = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_Q_noseason.pkl', 'rb'))        
            outputDict2 = pkl.load(open(f'grdcresults/glofas_{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
        validStations = list(outputDict.keys())            
    else:
        raise ValueError()
    print (len(validStations))
    cmi=[]
    if variable2=='CMI':
        for stationID in validStations:        
            stationDict = outputDict[stationID]                            
            cmi.append(stationDict[dictkey])       
    else:
        for stationID in validStations:        
            stationDict = outputDict2[stationID]                            
            cmi.append(stationDict[dictkey])       

    print ('no cmi', len(cmi))
    arr=[]
    for stationID in validStations:
        stationDict = outputDict2[stationID]            
        arr.append(stationDict['TWS'])  
    print ('no arr', len(arr))

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), variable2:np.array(cmi), 'Q_TWS': arr})
    dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
    
    if region == 'north_america':
        if use_percentile:
            #[0802] this option is not used anymore
            cd = np.percentile(cmi, [0, 25, 50, 75, 100])
            jd = np.percentile(arr, [0, 25, 50, 75, 100])
        else:
            #print (np.percentile(cmi, [0, 25, 50, 75, 100]))
            print (np.percentile(cmi, [0, 45, 60, 80, 90, 100])) #=>[0.     0.125  0.1875 0.25   0.3125 0.4375]
            cd = [0, 0.12, 0.2, 0.3, 0.5]
            jd = [0, 0.12, 0.2, 0.3, 0.5]
        pkl.dump([cd,jd], open(f'grdcresults/{source}_{region}_cd_{variable2}.pkl', 'wb'))
    else:
        cd,jd = pkl.load(open(f'grdcresults/{source}_north_america_cd_{variable2}.pkl', 'rb'))
    print (cd, jd)
    colorlist = genColorList(len(cd)-1)
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
            gdfCol.plot(column=variable2, ax=ax, color=colorlist[counter], marker='o', markersize= 110, alpha=1.0,
                        edgecolor='#546E7A', legend = False)
            print (gdfCol.to_string())
            #for checking plots
            #for ix,row in gdfCol.iterrows():
            #    ax.annotate(text=f"{row[variable2]:4.2}, {row['Q_TWS']:4.2}", xy=row.geometry.centroid.coords[0], horizontalalignment='center')
            ns+=len(groupDF)
            counter+=1
    print ('total processed', ns)
    ### now create inset legend
    if region=='north_america':
        if use_percentile:
            percentile_bounds = [25, 50, 75, 100]
        else:
            percentile_bounds = [0.12, 0.2, 0.3, 0.5]

        ax = ax.inset_axes([0.7,0.12,0.35,0.35]) #x,y, W, H
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
        _=ax.set_ylabel('TWSA_Q', fontsize=15)
    

def plotLags(cfg, region):
    """plot lagged corrleation plot [this is not used]
    """
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    gdf = getCatalog(region)
    if not cfg.data.deseason:
        outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))    
    else:
        outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))    
    #Loop through the results
    validStations = list(outputDict.keys())
    dfMetrics = pd.DataFrame(np.zeros((len(validStations), 4)), index=validStations)
    dfMetrics.columns=['TWS_num', 'TWS_lag', 'P_num', 'P_lag']
    var_names = ['Q', 'TWS', 'P']
    N = len(var_names)
    alpha_level = 0.05

    for stationID in validStations:
        stationDict = outputDict[stationID]

        sorted_links = stationDict['sorted_link_bin']
        p_matrix = stationDict['p_mat_bin']
        val_matrix = stationDict['val_mat_bin']
        sig_links = (p_matrix <= alpha_level)
        
        #assuming the first variable is Q
        j=0
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                for p in zip(*np.where(sig_links[:, j, :]))}

        # Sort by value        
        sorted_links = sorted(links, key=links.get, reverse=True)            
        # record all lags
        allTWSLags=[]
        allPLags = []
        
        # record the type of lags        
        string = ""
        for p in sorted_links:
            string += ("\n        (%s % d): pval = %.5f" %
                        (var_names[p[0]], p[1],
                        p_matrix[p[0], j, abs(p[1])]))
            if var_names[p[0]] == 'TWS':
                allTWSLags.append(p[1])
            elif var_names[p[0]] == 'P':
                allPLags.append(p[1])
            string += " | val = % .3f" % (
                val_matrix[p[0], j, abs(p[1])])
        #print (string)
        if not allTWSLags == []:
            allTWSLags = np.array(allTWSLags)
            print (stationID, ', largest TWS lag', np.min(allTWSLags), 'number of TWS lags', len(allTWSLags))
            dfMetrics.loc[stationID, 'TWS_lag'] = abs(np.min(allTWSLags))
            dfMetrics.loc[stationID, 'TWS_num'] = len(allTWSLags)            
        else: 
            dfMetrics.loc[stationID, 'TWS_num'] = 0.5 

        if not allPLags == []:
            allPLags = np.array(allPLags)
            print (stationID, ', largest P lag', np.min(allPLags), 'number of P lags', len(allPLags))
            dfMetrics.loc[stationID, 'P_lag'] = abs(np.min(allPLags))
            dfMetrics.loc[stationID, 'P_num'] = len(allPLags)            
        else:
            dfMetrics.loc[stationID, 'P_num'] = 0.5 #this is needed so P_num won't be empty
    #conver to actual days
    dfMetrics['TWS_lag'] = dfMetrics['TWS_lag'] * 5 
    dfMetrics['P_lag'] = dfMetrics['P_lag'] * 5 
    print (dfMetrics['TWS_lag'].max(), dfMetrics['P_lag'].max())

def plotCausalEffectBivariate(cfg, region, pcnt=None, usepercentile=False):
    import seaborn as sns
    # Use geopandas for vector data and xarray for raster data
    import geopandas as gpd
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection    
    
    #get river basin shape
    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    elif cfg.maps.global_basin == 'hydroshed':
        shpfile = os.path.join('maps/hydrosheds', hydroshedMaps[region])
    gdfBasinBound = gpd.read_file(shpfile)

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    rootdir = '/home/suna/work/predcsr5d'
    koppenmap = rxr.open_rasterio(os.path.join(rootdir, 'data/koppenclimate/koppen5.tif'), masked=True)  
    print("The CRS for this data is:", koppenmap.rio.crs)
    print("The spatial extent is:", koppenmap.rio.bounds())
    lon0, lat0, lon1, lat1 = getRegionBound(region)
    koppenmap = koppenmap.sortby(koppenmap.y)
    koppenmap_region = koppenmap.sel(y=slice(lat0,lat1),x=slice(lon0,lon1))

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    fig, ax = plt.subplots(1,1, figsize=(20,15))    
    #
    #plot the Koppen climate map
    #note how colorbar is set up: level is 1 more than number of categories
    #
    plotClimateMap=False
    if plotClimateMap:
        cmap = sns.color_palette(["#93AACF", "#FFD54F", "#DCEDC8", "#CFD8DC", "#E1BEE7"])
        im = koppenmap_region.plot(levels=[1, 2, 3, 4, 5, 6], colors=cmap, alpha=0.60, ax=ax, add_colorbar=False)
        ax.set_title('')    
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if region=='north_america':
        bounds = np.arange(0.5, 6.5)
        cb = plt.colorbar(im, ticks=bounds, orientation="horizontal", pad=0.05, shrink=0.75)
        #cb.set_ticks([bounds[0]] + [(b0 + b1) / 2 for b0, b1 in zip(bounds[:-1], bounds[1:])] + [bounds[-1]])
        cb.set_label(label='')
        cb.ax.tick_params(labelsize=15,length=0)
        cb.ax.set_xticklabels(["", 'Tropical', 'Arid', 'Temperate', 'Cold', 'Polar'])

    #plot basin boundaries
    gdfBasinBound.plot(ax=ax, facecolor="None", edgecolor="w", legend=False)       

    #plot bivariates
    if pcnt is None:
        #this should be used for annual maximima
        if not cfg.data.deseason:
            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}.pkl', 'rb'))        
        else:
            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_noseason.pkl', 'rb'))        
    else: 
        if not cfg.data.deseason:
            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{pcnt}{swe}.pkl', 'rb'))        
        else:
            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{pcnt}{swe}_noseason.pkl', 'rb'))        

    validStations = list(outputDict.keys())
    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]

    TWS_ace=[]
    P_ace = []
    for stationID in validStations:
        stationDict = outputDict[stationID]
        if pcnt is None:
            ace = stationDict['ace']  
        else:
            ace = stationDict['ace_bin']  
        print (stationID, dfStation[dfStation['grdc_no']==stationID]['river_x'])
        print (ace)
        TWS_ace.append(ace[1])           
        P_ace.append(ace[2])

    print ('no ace', len(TWS_ace))
    TWS_ace = np.array(TWS_ace)
    P_ace = np.array(P_ace)

    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), 'TWS_ACE':TWS_ace, 'P_ACE': P_ace})
    dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
    
    if region == 'north_america':
        if usepercentile:
            cd = np.percentile(P_ace, [0, 25, 50, 75, 100])
            jd = np.percentile(TWS_ace, [0, 25, 50, 75, 100])
        else:
            cd = [0, 0.1, 0.2, 0.3, 0.4]
            jd = [0, 0.1, 0.2, 0.3, 0.4]

        print (cd)
        print (jd)
        pkl.dump([cd,jd], open(f'grdcresults/causaleffect_{region}_cd.pkl', 'wb'))
    else:
        cd,jd = pkl.load(open(f'grdcresults/causaleffect_north_america_cd.pkl', 'rb'))

    colorlist = genColorList(len(cd)-1)
    counter=0
    ns = 0
    #note: need to make the last percentile inclusive, otherwise two points will be missing
    for i in range(len(cd)-1):
        for j in range(len(jd)-1):
            print (cd[i], cd[i+1],jd[j],jd[j+1])
            if i== len(cd)-2 and j == len(jd)-2 :
                groupDF = dfStation[(dfStation['P_ACE']>=cd[i]) & (dfStation['TWS_ACE']>=jd[j])]
            elif i== len(cd)-2:                
                groupDF = dfStation[(dfStation['P_ACE']>=cd[i]) & (dfStation['TWS_ACE']>=jd[j]) & (dfStation['TWS_ACE']<jd[j+1])]                
            elif j==len(jd)-2:
                groupDF = dfStation[(dfStation['P_ACE']>=cd[i]) & (dfStation['P_ACE']<cd[i+1]) & (dfStation['TWS_ACE']>=jd[j]) ]
            else:
                groupDF = dfStation[(dfStation['P_ACE']>=cd[i]) & (dfStation['P_ACE']<cd[i+1]) & (dfStation['TWS_ACE']>=jd[j]) & (dfStation['TWS_ACE']<jd[j+1])]

            gdfCol = gpd.GeoDataFrame(groupDF[['P_ACE', 'TWS_ACE']], geometry=gpd.points_from_xy(groupDF.long, groupDF.lat))
            gdfCol.plot(column='TWS_ACE', ax=ax, color=colorlist[counter], marker='o', markersize= 110, alpha=1.0,
                        edgecolor='#546E7A', legend = False)
            #for checking plots
            for ix,row in gdfCol.iterrows():
                ax.annotate(text=f"{row['P_ACE']:4.2}, {row['TWS_ACE']:4.2}", xy=row.geometry.centroid.coords[0], horizontalalignment='center')

            ns+=len(groupDF)
            counter+=1
    print ('total processed', ns)
    #pd.set_option('display.max_rows', None)
    #print (dfStation.head(66))
    ### now create inset legend
    if region=='north_america':
        if usepercentile:
            percentile_bounds = [25, 50, 75, 100]
        else:
            percentile_bounds = [0.1, 0.2, 0.3, 0.4]
        ax = ax.inset_axes([0.7,0.1,0.35,0.35]) #x,y, W, H
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
        _=ax.set_xlabel('P', fontsize=15)
        _=ax.set_yticks(list(range(len(percentile_bounds)+1)), yticks)
        _=ax.set_ylabel('TWS', fontsize=15)
        
    if pcnt is None:
        plt.savefig(f'outputs/causaleffect_{region}_all.png')
    else:
        plt.savefig(f'outputs/causaleffect_{region}_{pcnt}.png')

    plt.close()

def plotCausalEffectLat(cfg, pcnt=None):
    """Plot ace for Figure 2
    Note: I used the annual maxima method, so pcnt is irrelevant.
    To use the binary, pcnt should be 90
    """
    import seaborn as sns
    from matplotlib.pyplot import cm
    fig = plt.figure(figsize=(8,8))    
    gs = fig.add_gridspec(2, 2,  height_ratios=(1, 4), 
                left=0.1, right=0.95, bottom=0.1, top=0.9,
                wspace=0.08, hspace=0.05)            
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    N = len(config.data.regions)
    colors = cm.tab10(np.linspace(0, 1, N))
    TWS_ace_global=[]
    P_ace_global=[]

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    for ix, region in enumerate(config.data.regions):
        #get GRDC station geopdf
        gdf = getCatalog(region)
        dfStation = getStationMeta(gdf)

        if pcnt is None:
            #use all data
            if cfg.causal.exclude_Q:
                if not cfg.data.deseason:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}.pkl', 'rb'))        
                else:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_noseason.pkl', 'rb'))        
            else:
                if not cfg.data.deseason:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_Q.pkl', 'rb'))        
                else:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_Q_noseason.pkl', 'rb'))        
        else: 
            #do it on extremes only
            if cfg.causal.exclude_Q:
                if not cfg.data.deseason:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{pcnt}{swe}.pkl', 'rb'))        
                else:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{pcnt}{swe}_noseason.pkl', 'rb'))        
            else:
                if not cfg.data.deseason:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{pcnt}{swe}_Q.pkl', 'rb'))        
                else:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{pcnt}{swe}_Q_noseason.pkl', 'rb'))        

        validStations = list(outputDict.keys())
        dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]

        TWS_ace=[]
        TWS_lb=[]
        TWS_ub=[]    
        P_ace = []
        P_lb = []
        P_ub = []

        for stationID in validStations:
            stationDict = outputDict[stationID]
            if pcnt is None:
                ace = stationDict['ace']
                boot =  stationDict['ce_boot']
            else:
                ace = stationDict['ace_bin']  
                boot = stationDict['ce_boot_bin']

            TWS_ace.append(ace[1])           
            P_ace.append(ace[2])
            #as0701, note this need to be defined as error not abs values
            TWS_lb.append(np.max([ace[1]-boot[0][1],0]))
            TWS_ub.append(np.max([boot[1][1]-ace[1],0]))
            P_lb.append(np.max([ace[2]-boot[0][2],0]))
            P_ub.append(np.max([boot[1][2]-ace[2],0]))
            TWS_ace_global.append(ace[1])
            P_ace_global.append(ace[2])

        print ('no ace', len(TWS_ace))
        TWS_ace = np.array(TWS_ace)
        P_ace = np.array(P_ace)
        TWS_lb = np.array(TWS_lb)
        TWS_ub = np.array(TWS_ub)
        P_ub = np.array(P_ub)
        P_lb = np.array(P_lb)

        dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), 
                            'TWS_ACE':TWS_ace, 'TWS_lb': TWS_lb, 'TWS_ub': TWS_ub,
                            'P_ACE': P_ace, 'P_lb':P_lb, 'P_ub': P_ub,
        })

        dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')

        lat = dfStation['lat'].values
        ax2.scatter(TWS_ace, lat, color=colors[ix], label=regionNames[region])
        asymmetric_error = np.array(list(zip(TWS_lb, TWS_ub))).T
        ax2.errorbar(TWS_ace, lat, xerr=asymmetric_error, yerr=None, linestyle='none', color= colors[ix])
        
        ax3.scatter(P_ace, lat, color=colors[ix], label=regionNames[region])
        asymmetric_error = np.array(list(zip(P_lb, P_ub))).T
        ax3.errorbar(P_ace, lat, xerr=asymmetric_error, yerr=None, linestyle='none', color=colors[ix])
        
        #0705, remove y labels because we are sharing y axis
        ax3.set_yticklabels([])
        for axis in ['top', 'bottom', 'left', 'right']:
                ax2.spines[axis].set_linewidth(1.5) 
                ax3.spines[axis].set_linewidth(1.5) 

        ax2.set_xlabel('TWS', fontdict={'fontsize': 15})
        ax2.xaxis.set_tick_params(labelsize=12)
        ax2.yaxis.set_tick_params(labelsize=12)
        ax3.xaxis.set_tick_params(labelsize=12)

        ax3.set_xlabel('P',   fontdict={'fontsize': 15})
        ax2.set_ylabel('Lat', fontdict={'fontsize': 15})

        ax3.legend(fontsize=12, loc='upper right')
        #08082023 make TWS and P limits the same
        ax2.set_xlim([0, 1.0])
        ax3.set_xlim([0, 1.0])

    ax0.hist(TWS_ace_global, bins=10, color='darkgray')
    ax1.hist(P_ace_global,   bins=10, color='darkgray')
    for axis in ['top', 'bottom', 'left', 'right']:
            ax0.spines[axis].set_linewidth(1.5) 
            ax1.spines[axis].set_linewidth(1.5) 

    plt.subplots_adjust(wspace=0.0)
    if pcnt is None:
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                plt.savefig(f'outputs/causaleffect_lat_all.eps')
            else:
                plt.savefig(f'outputs/causaleffect_lat_all_noseason.eps')
        else:
            if not cfg.data.deseason:
                plt.savefig(f'outputs/causaleffect_lat_all_Q.png')
            else:
                plt.savefig(f'outputs/causaleffect_lat_all_Q_noseason.eps')
    else:
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                plt.savefig(f'outputs/causaleffect_lat_{pcnt}.png')
            else:
                plt.savefig(f'outputs/causaleffect_lat_{pcnt}_noseason.png')
        else:
            if not cfg.data.deseason:
                plt.savefig(f'outputs/causaleffect_lat_{pcnt}_Q.png')
            else:
                plt.savefig(f'outputs/causaleffect_lat_{pcnt}_Q_noseason.png')

    plt.close()

def plotGRanD(cfg):
    """
    This generates the Mississippi zoom in plot
    
    """
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    from PIL import ImageColor
    from generativepy.color import Color
    from matplotlib.colors import rgb2hex
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection    
        
    region='north_america'
    shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    gdfBasins = gpd.read_file(shpfile)
    gdfSingle = gdfBasins[gdfBasins['NAME']==cfg.dam.basin]    
    
    #get Reservoir database
    gdfDam = gpd.read_file(os.path.join('data', 'GRanD_Version_1_3/GRanD_reservoirs_v1_3.shp'))
    gdfDam = gpd.clip(gdfDam, gdfSingle)

    #get HydroRiver
    gdfRiver = gpd.read_file(os.path.join('data', 'hydroriver/HydroRIVERS_v10_na.dbf'))
    gdfRiver = gdfRiver[gdfRiver['ORD_STRA']>=5]
    gdfRiver = gpd.clip(gdfRiver, gdfSingle)

    #get GRDC stations
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)
    dictkey = 'P'
    variable2 = 'P_Q'    

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    fig,ax = plt.subplots(1,1, figsize=(16,12))
    if not cfg.data.deseason:
        outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))                
        outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
    else:
        outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))                
        outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))

    validStations = list(outputDict.keys())
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

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), variable2:np.array(cmi), 
                            'Q_TWS': arr})

    dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
    print (dfStation.columns)
    dfStation['coordinates'] = list(zip(dfStation.long, dfStation.lat))
    dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
    dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
    dfStation = dfStation.set_crs('epsg:4326')
    dfStation = gpd.clip(dfStation, gdfSingle)

    #note: this must be consistent with the one used for global plots
    cd = [0, 0.12, 0.2, 0.3, 0.5]
    jd = [0, 0.12, 0.2, 0.3, 0.5]

    gdfSingle.plot(ax=ax,edgecolor='black', facecolor='none')
    gdfDam.plot(ax=ax, edgecolor='silver', facecolor='none')
    gdfRiver.plot(ax=ax, color='lightskyblue', alpha=0.7)

    colorlist = genColorList(len(cd)-1)
    counter=0
    ns = 0
    resDict={}
    FP = []
    JP_TWSQ = []
    JP_PQ = []
    #note: need to make the last percentile inclusive, otherwise two points will be missing
    stationList = [4120950,4122603,4127503,4123300,4127800]
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
            gdfCol = gpd.GeoDataFrame(groupDF[[variable2, 'Q_TWS', 'grdc_no']], geometry=gpd.points_from_xy(groupDF.long, groupDF.lat))
            gdfCol.plot(column=variable2, ax=ax, color=colorlist[counter], marker='o', 
                        markersize= 140, alpha=1.0,
                        edgecolor='#546E7A', legend = False)

            
            for x, y, label,q_tws,q_p in zip(gdfCol.geometry.x, gdfCol.geometry.y, gdfCol.grdc_no, gdfCol.Q_TWS, gdfCol.P_Q):
                if label in stationList:
                    ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
                fp = getGFPlainArea(stationID=label, region='north_america')
                resDict[label] = [fp, q_tws, q_p]
                FP.append(fp)
                JP_TWSQ.append(q_tws)
                JP_PQ.append(q_p)

            #for checking plots
            #for ix,row in gdfCol.iterrows():
            #    ax.annotate(text=f"{row[variable2]:4.2}, {row['Q_TWS']:4.2}", xy=row.geometry.centroid.coords[0], horizontalalignment='center')
            ns+=len(groupDF)
            counter+=1

    print ('total processed', ns)
    ### now create inset legend
    percentile_bounds = [0.12, 0.2, 0.3, 0.5]

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plotColorLegend = False
    if plotColorLegend:
        ax = ax.inset_axes([0.7,0.1,0.35,0.35]) #x,y, W, H
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

    #plot legend
    pmark = mpatches.Patch(facecolor='none',
                           edgecolor='silver',
                           linestyle='-',
                           alpha=0.8,
                           label='Reservoir')
    lmark = Line2D(
        [],
        [],
        color='lightskyblue',
        linewidth=3,
        alpha=0.4,
        label='River')
    handles, _ = ax.get_legend_handles_labels()

    ax.legend(
        handles=[
            *handles,
            lmark,
            pmark],
        loc='upper right',
        ncol=1,
        shadow=True,
        fontsize=15)

    #plot inset scatter plot
    plotFPScatter  = False
    if plotFPScatter:
        ax1 = ax.inset_axes([0.05,0.05,0.3,0.3]) #x,y, W, H
        #ax1.set_aspect('equal', adjustable='box')
        ax1.scatter(np.array(FP), np.array(JP_TWSQ), color='slategray')
        ax1.set_xlabel('FP Area', fontsize=15)
        ax1.set_ylabel('TWS_Q', fontsize=15)
    if not cfg.data.deseason:
        plt.savefig(f'outputs/GRanD_{cfg.dam.basin}.eps')
    else:
        plt.savefig(f'outputs/GRanD_{cfg.dam.basin}_noseason.eps')
    plt.close()

def getGFPlainArea(stationID, region):
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    xds = rioxarray.open_rasterio('data/GFPLAIN250mTIFF/NA.TIF')
    #get basin mask
    basinmask = getBasinMask(stationID=stationID, region=region, gdf=gdf)
    xds = xds.sortby('y')       
    try: 
        with rioxarray.set_options(export_grid_mapping=False):
            bbox=basinmask.bounds
            #convert to [lon0,lat0,lon1,lat1]
            bbox=bbox.values.tolist()[0]
            basinxds = xds.sel(y=slice(bbox[1],bbox[3]),x=slice(bbox[0],bbox[2]))        
            #08/28, add all_touched = True
            clipped = basinxds.rio.clip(basinmask.geometry, basinmask.crs, from_disk=True, drop=True, invert=False, all_touched = True)  
            totalFloodPlain =  clipped.where(clipped.notnull()).sum(dim=['y','x'])
            print (stationID, totalFloodPlain.values)
    except Exception:
        totalFloodPlain=np.NaN

    return totalFloodPlain

def printGRDCInfo(cfg):
    """Generate Latex table of GRDC stations
    """
    THRESHOLD = 0.12 #see comment in compound_events.py
    totalStations= 0
    goodStations = {'both': 0, 'p_only':0, 'tws_only':0}
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    with open('latextable.txt', 'w') as fid:
        fid.write("\\begin{longtable}[H] {p{3cm}p{3cm}p{2cm}p{2cm}p{2cm}} \n")
        fid.write("\caption{Selected GRDC stations. \label{tab:s1}} \\\\ \n")       
        fid.write("\\toprule \n")
        fid.write("Station ID & River & Drainage Area &  Co-occurring Extremes (TWS-Q | P-Q) \\\\ \n")
        fid.write("\\endhead \\\\ \n")
        fid.write("\\hline \\\\ \n")
        fid.write("\\endfoot \\\\ \n")
        fid.write("\\midrule \n ")

        for region in cfg.data.regions:
            gdf = getCatalog(region)
            dfStation = getStationMeta(gdf)

            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))        
            outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
    
            validStations = list(outputDict.keys())
            dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
            cmi=[]
            tws_arr=[]
            p_arr = []
            for stationID in validStations:
                stationDict = outputDict[stationID]            
                cmi.append(stationDict['cmi'])                
                stationDict2 = outputDict2[stationID]            
                tws_arr.append(stationDict2['TWS'])  
                p_arr.append(stationDict2['P'])
                #asun 08172023, this is used to support a sentence under compound event analysis
                if stationDict2['TWS']>THRESHOLD and stationDict2['P']>THRESHOLD:
                    goodStations['both'] += 1
                elif stationDict2['TWS']>THRESHOLD:
                    goodStations['tws_only'] += 1
                elif stationDict2['P']>THRESHOLD:
                    goodStations['p_only'] += 1

            print ('no arr', len(tws_arr))                   
            totalStations+=len(tws_arr)
            dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), 'CMI':np.array(cmi), 'Q_TWS': tws_arr, 'Q_P':p_arr})
            dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')

            for ix, row in dfStation.iterrows():
                #fid.write(f"{row['grdc_no']} & {row['river_x']} & {row['area_hys']:7.0f} &  {row['CMI']:5.3f} &  {row['Q_TWS']:4.2f} \\\\ \n")
                fid.write(f"{row['grdc_no']} & {row['river_x']} & {row['area_hys']:7.0f} &  {row['Q_TWS']:4.2f} | {row['Q_P']:4.2f} \\\\ \n")
                print (f"{row['grdc_no']}, {row['river_x']}, {row['area_hys']:7.0f}, {row['Q_TWS']:4.2f} | {row['Q_P']:4.2f} ")
        #fid.write("\\bottomrule \n")
        fid.write("\\end{longtable} \n") 
        print ("total stations used", totalStations)
        print ('Number of significant stations', goodStations)

def plotGRanD_MaxLag(cfg,region,binary=True):    
    """Plot Figure 1 Mississippi plot 
    mode: 'binary' or 'normal', if binary use event series, otherwise use normal time series
    """
    def plotSubplot(ax, columnName, markerColumnName, colorbar_label,subfig_title,plotBasemapLegend=False):
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)       
        fig.add_axes(cax)

        gdfSingle.plot(ax=ax,edgecolor='black', facecolor='none', zorder=1)
        gdfDam.plot(ax=ax, edgecolor='silver', facecolor='none', zorder=1)
        gdfRiver.plot(ax=ax, color='lightskyblue', alpha=0.5, zorder=1)

        gdfCol = gpd.GeoDataFrame(dfStation[columnName], 
                                geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
        markersizes = dfStation[markerColumnName].to_numpy()
        #get a unique list of markersizes
        uniquevals = np.unique(markersizes)

        #0705, because tau_min starts from 1 now, so we include all unique values
        uniquevals = uniquevals[0:]
        #0821, fix the size mismatch
        #markersizes = markersizes*20
        markersizes = markersizes*10

        #markersize: number of lags, vmax: max lag time
        gdfCol.plot(column=columnName, ax=ax, cax=cax, cmap=cmap, marker='o', markersize= markersizes, 
                    legend_kwds={'shrink': 0.5}, legend=True, 
                    vmin=0, vmax=1, zorder=2)
        cax.set_ylabel(colorbar_label, fontsize=14)
        sizeList = np.array(uniquevals, dtype=int)
        print ('size list', sizeList)        
        custom_markers = [
            Line2D([0], [0], marker="o", color='w', markerfacecolor='None', label=item, markeredgewidth=0.5, markeredgecolor="k", markersize=item) for ix,item in enumerate(sizeList)
        ]        
        legend2 = ax.legend(handles = custom_markers, loc='lower left', fontsize=11, frameon=True, title="# Lags")                
        ax.add_artist(legend2)       

        #remove the map tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.90,0.93, subfig_title, fontsize=16, transform=ax.transAxes)
        for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1.5) 
                
        if plotBasemapLegend:
            #plot legend
            pmark = mpatches.Patch(facecolor='none',
                                edgecolor='silver',
                                linestyle='-',
                                alpha=0.8,
                                label='Reservoir')
            lmark = Line2D(
                [],
                [],
                color='lightskyblue',
                linewidth=3,
                alpha=0.4,
                label='River')
            handles, _ = ax.get_legend_handles_labels()

            ax.legend(
                handles=[
                    *handles,
                    lmark,
                    pmark],
                loc='lower right',
                ncol=1,
                shadow=True,
                fontsize=11)

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""


    fig, axes = plt.subplots(2,1,figsize=(12, 12))
    ax = axes[0]

    nlags = 18
    region='north_america'

    shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    gdfBasins = gpd.read_file(shpfile)
    #get a single basin
    gdfSingle = gdfBasins[gdfBasins['NAME']==cfg.dam.basin]    
    
    #get Reservoir database
    gdfDam = gpd.read_file(os.path.join('data', 'GRanD_Version_1_3/GRanD_reservoirs_v1_3.shp'))
    gdfDam = gpd.clip(gdfDam, gdfSingle)

    #get HydroRiver
    gdfRiver = gpd.read_file(os.path.join('data', 'hydroriver/HydroRIVERS_v10_na.dbf'))
    gdfRiver = gdfRiver[gdfRiver['ORD_STRA']>=5]
    gdfRiver = gpd.clip(gdfRiver, gdfSingle)

    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))        
    validStations = list(outputDict.keys())
    outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]

    print (dfStation.columns)

    dfStation['coordinates'] = list(zip(dfStation.long, dfStation.lat))
    dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
    dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
    dfStation = dfStation.set_crs('epsg:4326')
    dfStation = gpd.clip(dfStation, gdfSingle)

    cmap = ListedColormap(sns.color_palette(palette="OrRd", n_colors=10).as_hex())    

    #DF for collecting metrics
    dfMetrics = pd.DataFrame(np.zeros((len(validStations), 6)), index=validStations)
    dfMetrics.columns=['rhoTWS', 'TWSLags', 'rhoP', 'PLags', 'SI_TWS', 'SI_P']

    #loop through all stations in the basin mask
    rhoTWS=[]
    rhoP = []
    for stationID in validStations:
        stationDict = outputDict[stationID]
        eventDict = outputDict2[stationID]
        if binary:
            sorted_links = stationDict['sorted_link_bin']
        else:
            sorted_links = stationDict['sorted_link']

        # record all lags
        allTWSLags=[]
        allPLags = []
        # record the type of lags
        #loop through the links, hard assumption 1==TWS, 2==P [this order must be the same as in PCMCI dataframe]
        for p in sorted_links:
            if p[0] == 1:
                allTWSLags.append(p[1])
            elif p[0] == 2: 
                allPLags.append(p[1])
        
        dataframe = pkl.load(open(f'grdcdataframes/s_{stationID}.pkl', 'rb'))
        pcmci = PCMCI(
            dataframe=dataframe, 
            cond_ind_test=RobustParCorr(), #parcorr,
            verbosity=1)
        
        #get max correlation
        correlations = pcmci.get_lagged_dependencies(tau_min=cfg.causal.tau_min, tau_max=cfg.causal.tau_max, val_only=True)['val_matrix']
        matrix_lags = np.argmax(np.abs(correlations), axis=2)
        rhoTWS.append(correlations[0, 1, matrix_lags[0,1]])
        rhoP.append(correlations[0, 2, matrix_lags[0,2]])

        if not allTWSLags == []:
            #hard assumption 1==TWS, 2==P [this order must be the same as in PCMCI dataframe]            
            allTWSLags = np.array(allTWSLags)
            dfMetrics.loc[stationID, 'rhoTWS'] = correlations[0, 1, matrix_lags[0,1]]  
            dfMetrics.loc[stationID, 'TWSLags'] = len(allTWSLags)     
            dfMetrics.loc[stationID, 'SITWS'] = eventDict['SI_TWS']       
        else:
            #we use some small value for empty lag 
            dfMetrics.loc[stationID, 'TWSLags'] = 0.5 

        if not allPLags == []:
            allPLags = np.array(allPLags)
            dfMetrics.loc[stationID, 'rhoP'] = correlations[0, 2, matrix_lags[0,2]]
            dfMetrics.loc[stationID, 'PLags'] = len(allPLags)            
            dfMetrics.loc[stationID, 'SIP'] = eventDict['SI_P']       
        else:
            dfMetrics.loc[stationID, 'PLags'] = 0.5 #this is needed so P_num won't be empty

    dfStation = dfStation.join(dfMetrics, on='grdc_no')
    
    #=========plot TWS subplot
    columnName = 'rhoTWS'
    markerColumnName = 'TWSLags'
    plotSubplot(axes[0], columnName=columnName, markerColumnName=markerColumnName, colorbar_label='Max corr',subfig_title='TWSA', plotBasemapLegend=True)
    
    #=========plot P subplot
    columnName = 'rhoP'
    markerColumnName = 'PLags'
    plotSubplot(axes[1], columnName=columnName, markerColumnName=markerColumnName, colorbar_label='Max corr', subfig_title='Precip')

    plt.subplots_adjust(wspace=0.07,hspace=0.05)
    
    plt.savefig("outputs/grand_lagplot.eps")
    plt.close()

    #plot scatter plot between SI_TWS and SI_P
    fig, ax = plt.subplots(1,1,figsize=(5,5))    
    dfStation.plot.scatter('SITWS', 'SIP', color='royalblue', ax=ax)
    #rho, pval = pearsonr(dfStation['SITWS'], dfStation['SIP'])
    #print (rho, pval)
    ax.set_xlabel('SI TWS', fontsize=13)
    ax.set_ylabel('SI P', fontsize=13)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.savefig('outputs/SI_scatterplot.png')
    plt.close()

def plotSI(cfg):
    """
    Plot SI diagrams
    """
    def plot_subfig(ax, col1, col2):
        """ plot a subfigure
        """
        im = ax.scatter(dfStation[col1], dfStation[col2], c=dfStation['climate'], 
                    cmap=cmap, marker='o', s=50)
        rho, pval = pearsonr(dfStation[col1], dfStation[col2])
        print (rho, pval)
        ax.axis('equal')
        if col2=='SITWS':
            ax.set_ylabel('TWSA', fontsize=15)
        elif col2=='SIP':
            ax.set_ylabel('P', fontsize=15)
        elif col2=='SIFPI':
            ax.set_ylabel('FPI', fontsize=15)

        ax.set_xlabel('Q', fontsize=15)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.02, 0.04, f'R={rho:3.2f}', fontsize=13, transform=ax.transAxes)

        return im

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    koppenmap = rxr.open_rasterio('data/koppenclimate/koppen5.tif', masked=True)  
    print("The CRS for this data is:", koppenmap.rio.crs)
    print("The spatial extent is:", koppenmap.rio.bounds())
    koppenmap = koppenmap.sortby(koppenmap.y)

    allMetrics=[]
    for region in config.data.regions:
        #get GRDC station geopdf
        gdf = getCatalog(region)
        dfStation = getStationMeta(gdf)

        if not cfg.data.deseason:
            outputDict = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
        else:
            outputDict = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
        validStations = list(outputDict.keys())

        dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
        dfStation['coordinates'] = list(zip(dfStation.long, dfStation.lat))
        dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
        dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
        dfStation = dfStation.set_crs('epsg:4326')

        #DF for collecting metrics
        dfMetrics = pd.DataFrame(np.zeros((len(validStations), 4)), index=validStations)
        dfMetrics.columns=['SITWS', 'SIP', 'SIQ', 'climate']

        #loop through all stations in the basin mask
        for stationID in validStations:
            row = dfStation[dfStation['grdc_no']==stationID]
            lat,lon = row['lat'].values[0], row['long'].values[0]
            climate = koppenmap.sel(x=lon, y=lat, method='nearest').values[0]
            eventDict = outputDict[stationID]

            dfMetrics.loc[stationID, 'SITWS'] = eventDict['SI_TWS']       
            dfMetrics.loc[stationID, 'SIP'] = eventDict['SI_P'] 
            dfMetrics.loc[stationID, 'SIQ'] = eventDict['SI_Q'] 
            #0723
            dfMetrics.loc[stationID, 'SIFPI'] = eventDict['SI_FPI'] 
            
            dfMetrics.loc[stationID, 'climate'] = climate

        dfStation = dfStation.join(dfMetrics, on='grdc_no')
        allMetrics.append(dfStation)

    dfStation = pd.concat(allMetrics)

    fig, axes = plt.subplots(1,2, figsize=(14, 6))
    cmap = ListedColormap(sns.color_palette(["#85C1E9", "#FFD54F", "#DCEDC8", "#CFD8DC"]).as_hex())
    im = plot_subfig(axes[0], col1='SIQ', col2='SITWS')
    im2 = plot_subfig(axes[1], col1='SIQ', col2='SIP')
    #0723
    #im3 = plot_subfig(axes[2], col1='SIQ', col2='SIFPI')


    plt.subplots_adjust(wspace=0.2)
    #spread between [1, 4]    
    bounds = np.arange(1.4, 4, 0.7)
    cb = fig.colorbar(im2, ax=axes.ravel().tolist(), ticks=bounds, orientation="vertical", pad=0.01)
    cb.ax.tick_params(labelsize=13)
    cb.ax.set_yticklabels(['Tropical', 'Arid', 'Temperate', 'Cold'])

    plt.savefig('outputs/all_SI_scatterplot.png')
    plt.close()

def compareGRDC_GloFAS(cfg):
    """Compare GRDC against GloFAS at global scale
    """
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""
    GRDC_TWS=[]
    GLOFAS_TWS = []

    if not cfg.data.deseason:
        glofas_outputDict2 = pkl.load(open(f'grdcresults/glofas_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
    else:
        glofas_outputDict2 = pkl.load(open(f'grdcresults/glofas_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
    validStations = list(glofas_outputDict2.keys())

    cd = [0, 0.12, 0.2, 0.3, 0.5]
    colorlist = genSingleColorList(num_grps=len(cd)-1)
    markercolors = []
    for region in cfg.data.regions:
        if not cfg.data.deseason:
            grdc_outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))            
        else:
            grdc_outputDict2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))            
        
        for stationID in validStations:        
            if stationID in grdc_outputDict2.keys():
                stationDict = grdc_outputDict2[stationID]                            
                GRDC_TWS.append(stationDict['TWS'])            
                if cd[0]<=stationDict['TWS']<cd[1]:
                    markercolors.append(colorlist[0])
                elif cd[1]<=stationDict['TWS']<cd[2]:
                    markercolors.append(colorlist[1])
                elif cd[2]<=stationDict['TWS']<cd[3]:
                    markercolors.append(colorlist[2])
                elif stationDict['TWS']>=cd[3]:
                    markercolors.append(colorlist[3])
                

                stationDict = glofas_outputDict2[stationID]                            
                GLOFAS_TWS.append(stationDict['TWS'])
                
    print ('number of common stations found', len(GLOFAS_TWS))

    fig, ax = plt.subplots(1,1,figsize=(5,5))    
    ax.scatter(np.array(GRDC_TWS), np.array(GLOFAS_TWS), c=markercolors, alpha=0.7)
    ax.set_xlabel('GRDC', fontsize=13)
    ax.set_ylabel('GLOFAS', fontsize=13)
    ax.set_xlim([0, 0.7])
    ax.set_ylim([0, 0.7])
    rho, pval = pearsonr(np.array(GRDC_TWS), np.array(GLOFAS_TWS))
    ax.text(0.8, 0.92, f'R={rho:3.2f}', fontsize=12, transform=ax.transAxes)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5) 
    if not cfg.data.deseason:
        plt.savefig('outputs/GRDC_GLOFAS_scatterplot.eps')
    else:
        plt.savefig('outputs/GRDC_GLOFAS_scatterplot_noseason.eps')
    plt.close()
    
    print (rho, pval)

def plot5d_vs_monthly(cfg):
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    #load 5d results
    arr5d=[]
    arrMon=[]
    for region in cfg.data.regions:
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                outputDict5d = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}.pkl', 'rb'))        
                outputDict5d2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
                outputDictmon = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_monthly.pkl', 'rb'))        
                outputDictmon2 = pkl.load(open(f'grdcresults/{region}_all_events_monthly.pkl', 'rb'))
            else:
                outputDict5d = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_{cfg.event.cutoff}{swe}_noseason.pkl', 'rb'))        
                outputDict5d2 = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
                outputDictmon = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_noseason_monthly.pkl', 'rb'))        
                outputDictmon2 = pkl.load(open(f'grdcresults/{region}_all_events_noseason_monthly.pkl', 'rb'))
        validStations = list(outputDict5d.keys())     
        for stationID in validStations:
            stationDict = outputDict5d2[stationID]            
            arr5d.append(stationDict['TWS'])  
            stationDict = outputDictmon2[stationID]            
            arrMon.append(stationDict['TWS'])  
    print ('no arr', len(arrMon))
    arrMon = np.array(arrMon)
    arr5d = np.array(arr5d)
    
    yy = sorted(np.unique(arr5d))
    xx = sorted(np.unique(arrMon))
    xdict = {item: i for i,item in enumerate(xx)}
    ydict = {item: i for i,item in enumerate(yy)}
    X,Y = np.meshgrid(xx,yy)
    C = np.zeros((len(yy),len(xx)))

    for itemx,itemy in zip(arrMon, arr5d):
        C[ydict[itemy],xdict[itemx]] += 1
    X = X.flatten()
    Y = Y.flatten()
    C = C.flatten()
    C = C*6    
    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, s=C, color='blue')

    plt.xlim([0, 0.6])
    plt.ylim([0, 0.6])
    plt.xlabel('CSR Monthly')
    plt.ylabel('CSR 5d')
    plt.title('Q-TWS joint occurrence probability')
    plt.savefig(f'compare5dToMonthly_{cfg.event.event_method}{swe}.png')
    plt.close()

def plotACEMap(cfg,axes,caxes,pcnt=None):
    """Plot average causal effect map for TWS and P together
    """
    
    import seaborn as sns
    # Use geopandas for vector data and xarray for raster data
    import geopandas as gpd
    import rioxarray as rxr
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import ListedColormap

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    #get global basin shape
    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
        gdfBasinBound = gpd.read_file(shpfile)
        #plot basin boundaries
        for ax in axes:
            gdfBasinBound.plot(ax=ax, facecolor="None", edgecolor="w", legend=False)       

    for ix, region in enumerate(config.data.regions):
        #get GRDC station geopdf
        gdf = getCatalog(region)
        dfStation = getStationMeta(gdf)

        if pcnt is None:
            #use all data
            if cfg.causal.exclude_Q:
                if not cfg.data.deseason:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}.pkl', 'rb'))        
                else:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_noseason.pkl', 'rb'))        
            else:
                if not cfg.data.deseason:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_Q.pkl', 'rb'))        
                else:
                    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_Q_noseason.pkl', 'rb'))        
        else:
            raise ValueError("Not implemented")

        validStations = list(outputDict.keys())
        dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]

        TWS_ace=[]
        P_ace = []
        for stationID in validStations:
            stationDict = outputDict[stationID]
            if pcnt is None:
                ace = stationDict['ace']  
            else:
                ace = stationDict['ace_bin']  
            TWS_ace.append(ace[1])           
            P_ace.append(ace[2])

        #this extracts both variables
        print ('no ace', len(TWS_ace))
        TWS_ace = np.array(TWS_ace)
        P_ace = np.array(P_ace)

        dfMetrics = pd.DataFrame({'grdc_no': np.array(validStations,dtype=np.int64), 'TWS_ACE':TWS_ace, 'P_ACE': P_ace})
        dfStation = dfStation.merge(dfMetrics, on='grdc_no', how='inner')
      
        cmap = ListedColormap(sns.color_palette(palette="rocket_r", n_colors=6).as_hex())    
        for varname,ax,cax in zip(['TWS', 'P'], axes,caxes):
            gdfCol = gpd.GeoDataFrame(dfStation[f'{varname}_ACE'], geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
            gdfCol.plot(column=f'{varname}_ACE', ax=ax, cax=cax, cmap=cmap, marker='o', markersize= 50, alpha=0.8, vmax=0.5, vmin=0.0,
                            legend_kwds={'shrink': 0.5, 'label': f"{varname}", "extend": "max"}, legend=True)

def plotBasinSI(cfg, region):
    """
    Plot SI diagrams
    """
    def plot_subfig(ax, col1, col2):
        """ plot a subfigure
        """
        im = ax.scatter(dfStation[col1], dfStation[col2], color='tab:blue', marker='o', s=50)
        rho, pval = pearsonr(dfStation[col1], dfStation[col2])
        print (rho, pval)
        ax.axis('equal')
        if col2=='SITWS_x':
            ax.set_ylabel('TWSA', fontsize=15)
        elif col2=='SIP_x':
            ax.set_ylabel('P', fontsize=15)

        ax.set_xlabel('Q', fontsize=15)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.02, 0.04, f'R={rho:3.2f}', fontsize=13, transform=ax.transAxes)

        return im

    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""


    allMetrics=[]
    
    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    gdfBasins = gpd.read_file(shpfile)
    gdfSingle = gdfBasins[gdfBasins['NAME']==cfg.dam.basin]    

    if not cfg.data.deseason:
        outputDict = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events{swe}.pkl', 'rb'))
    else:
        outputDict = pkl.load(open(f'grdcresults/{region}_{cfg.event.event_method}_all_events_noseason{swe}.pkl', 'rb'))
    validStations = list(outputDict.keys())

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfStation['coordinates'] = list(zip(dfStation.long, dfStation.lat))
    dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
    dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
    dfStation = dfStation.set_crs('epsg:4326')

    #DF for collecting metrics
    dfMetrics = pd.DataFrame(np.zeros((len(validStations), 3)), index=validStations)
    dfMetrics.columns=['SITWS', 'SIP', 'SIQ']

    #loop through all stations in the basin mask
    for stationID in validStations:
        row = dfStation[dfStation['grdc_no']==stationID]
        lat,lon = row['lat'].values[0], row['long'].values[0]
        eventDict = outputDict[stationID]

        dfMetrics.loc[stationID, 'SITWS'] = eventDict['SI_TWS']       
        dfMetrics.loc[stationID, 'SIP'] = eventDict['SI_P'] 
        dfMetrics.loc[stationID, 'SIQ'] = eventDict['SI_Q'] 
        
    dfStation = dfStation.join(dfMetrics, on='grdc_no')
    allMetrics.append(dfStation)

    allMetrics = pd.concat(allMetrics)
    dfStation = dfStation.merge(allMetrics, on='grdc_no', how='inner')
    print (dfStation.columns)
    dfStation['coordinates'] = list(zip(dfStation.long_x, dfStation.lat_x))
    dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
    dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
    dfStation = dfStation.set_crs('epsg:4326')
    dfStation = gpd.clip(dfStation, gdfSingle)

    fig, axes = plt.subplots(1,2, figsize=(14, 6))
    im = plot_subfig(axes[0], col1='SIQ_x', col2='SITWS_x')
    im2 = plot_subfig(axes[1], col1='SIQ_x', col2='SIP_x')

    plt.subplots_adjust(wspace=0.2)


    plt.savefig('outputs/basin_SI_scatterplot.png')
    plt.close()

def plotCausalEffectHeatMap(cfg, region, binary=True):
    """Plot heatmap of causal effect for precip and twsa
    """
    if cfg.data.removeSWE:
        swe='swe'
    else:
        swe=""

    if cfg.causal.exclude_Q:
        if not cfg.data.deseason:
            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}.pkl', 'rb'))        
        else:
            outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.event_method}_90{swe}_noseason.pkl', 'rb'))        

    validStations = list(outputDict.keys())

    #get GRDC station geopdf
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)

    shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    gdfBasins = gpd.read_file(shpfile)
    gdfSingle = gdfBasins[gdfBasins['NAME']==cfg.dam.basin]    

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]
    dfStation['coordinates'] = list(zip(dfStation.long, dfStation.lat))
    dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
    dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
    dfStation = dfStation.set_crs('epsg:4326')

    dfStation = gpd.clip(dfStation, gdfSingle)
    dfStation = dfStation.sort_values(by='lat',ascending=False)
    
    #form df for heatmap
    twsarr = []
    P_arr = []
    for ix, row in dfStation.iterrows():
        stationDict = outputDict[row['grdc_no']]
        res = stationDict['ce_dict']
        tws_ce = np.array(res['TWSA'])
        P_ce   = np.array(res['P'])
        #get CE for significant causal links only
        tws_vec = np.zeros(tws_ce.shape)+np.NaN
        P_vec = np.zeros(tws_ce.shape)+np.NaN
        if binary:
            sorted_links = stationDict['sorted_link_bin']
        else:
            sorted_links = stationDict['sorted_link']
        
        print (sorted_links)
        for p in sorted_links:
            #lag is negative, so negate it and reduce 1 for zero-based index
            lag_no = -p[1]-1
            if p[0] == 1:
                tws_vec[lag_no]=tws_ce[lag_no]
            elif p[0] == 2: 
                P_vec[lag_no]  =P_ce[lag_no]
        twsarr.append(tws_vec)
        P_arr.append(P_vec)

    twsDF = pd.DataFrame(np.stack(twsarr), index=dfStation['grdc_no'].values, columns=list(range(1,19)))
    P_DF = pd.DataFrame(np.stack(P_arr), index=dfStation['grdc_no'].values, columns=list(range(1,19)))
    
    fig,axes = plt.subplots(2,1,figsize=(12,12))
    g = sns.heatmap(twsDF, ax=axes[0], linewidth=0.5, cmap='flare', vmin=0, vmax=1.0,  cbar_kws={'label': 'TWSA CE'})
    g.set_facecolor('lightgray')
    axes[0].set_xlabel("")
    #axes[0].set_ylabel("TWSA", fontsize=14)
    g= sns.heatmap(P_DF, ax=axes[1], linewidth=0.5, cmap='flare', vmin=0, vmax=1.0, cbar_kws={'label': 'Precip CE'})
    g.set_facecolor('lightgray')

    axes[1].set_xlabel("5-day Lag",fontsize=14)
    #axes[1].set_ylabel("P", fontsize=14)
    plt.savefig('outputs/mississippi_causaleffect_heatmap.eps')
    plt.close()

if __name__ == '__main__':    
    #08022023 notes:
    #itask =1, regenerates all metrics for all global regions
    #itask =8, generates Figure 1A, 1B
    #itask =15, generates Figure 1C
    #itask =10, print the latex table [copy/past into latex]
    #itask =14, plot figure 2
    #itask =12, plot figure 3

    from myutils import load_config
    config = load_config('config.yaml')

    itask = 1
    if itask == 1:
        #turn reGen to True to reprocess all regions one by one to get MI, CMI, annual maxima
        for region in ['asia']: #config.data.regions:
            main(cfg=config, region=region)
    elif itask == 2:
        #plot all regions
        plotAllRegions(config, features=['TWS_MI', 'TWS_CMI', 'P_MI', 'T_MI'])
    elif itask == 3: 
        #turn reGen to True to reprocess all regions to get gloFAS related metrics
        genGloFASMetrics(cfg=config, region='north_america',reGen=True)
    elif itask == 4:
        #turn reGen to True to get extreme event tau
        compareExtremeEvents('north_america',reGen=False)
    elif itask == 5:
        plotBasinShapes(region='north_america')
    elif itask == 6:
        #plot single variate cmi
        plotKoppenMap(cfg=config, region="north_america")
    elif itask == 7:
        #plot joint probability between TWS and Q, or P and Q
        plotJRPMap(cfg=config, region="north_america", varname='P')        
    elif itask == 8: 
        #Figure 1 (grdc)  
        #Figure 1C is generated by using compareGRDC_GloFAS() [as0820, not used]
        #plot CMI vs. joint probability of TWS and P
        #(joint probability is generated by running compound_events.py)
        #use figsize (20, 12) to make maps appear in right sizes
        source = 'grdc'
        if source =='grdc':
            fig = plt.figure(figsize=(20, 12))    
            gs = fig.add_gridspec(3, 3,  height_ratios=(1, 1, 0.05),
                        left=0.1, right=0.95, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.02)       
            ax0 = fig.add_subplot(gs[0,:2])
            ax1 = fig.add_subplot(gs[0,2])
            ax2 = fig.add_subplot(gs[1,0])
            ax3 = fig.add_subplot(gs[1,1])
            ax4 = fig.add_subplot(gs[1,2])
            allax = [ax0,ax1,ax2,ax3,ax4]
            ax_legend = fig.add_subplot(gs[2,:])

            for region,theax in zip(config.data.regions, allax):
                plotBivariate(cfg=config, region=region, ax=theax, source='grdc', variable2='P_Q', ax_legend=ax_legend)
            plt.subplots_adjust(wspace=0.05,hspace=0.01,left=None, bottom=None, right=None, top=None)
            if not config.data.deseason:
                plt.savefig(f'outputs/grdc_bivariate_P_{config.event.event_method}.png')
            else:
                plt.savefig(f'outputs/grdc_bivariate_P_noseason_{config.event.event_method}.png')
            plt.close()
        else:
            fig,ax = plt.subplots(1,1, figsize=(12, 7.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad="2%")

            plotBivariate(cfg=config, region="north_america", ax=ax, source='glofas', variable2='P_Q', ax_legend=cax)
            if not config.data.deseason:
                plt.savefig(f"outputs/glofas_bivariate_P_{config.event.event_method}.png")
            else:
                plt.savefig(f"outputs/glofas_bivariate_P_noseason_{config.event.event_method}.png")
            plt.close()
        #plotBivariate(cfg=config, region="north_america", source='grdc', variable2='P_Q')
        #plotBivariate(cfg=config, region="north_america", source='glofas')    
    elif itask == 9: 
        #plot causal links
        plotLags(cfg=config, region="north_america")
    elif itask == 10:
        #Print latex table in SI A Table S1
        printGRDCInfo(cfg=config)
    elif itask == 11:
        plotCausalEffectBivariate(cfg=config, region="north_america", pcnt=None)
    elif itask == 12:
        #Figure 2 (the latitude comparison and histograms, note: I removed histograms in AI to save space
        plotCausalEffectLat(cfg=config, pcnt=None)
        #08222023, uncomment the following for binary event based ACE
        #plotCausalEffectLat(cfg=config, pcnt=90)
    elif itask == 13:
        #this is used to plot amplified MS basin gages
        #this is not used in the paper 
        plotGRanD(config)
    elif itask == 14:
        #Figure 3
        #generate figure using hydroriver, global dam database, correlation at max lag
        plotGRanD_MaxLag(config, region='north_america', binary=True)
    elif itask == 15:
        #Figure 1C (plot scatter plot)
        compareGRDC_GloFAS(config)
    elif itask == 16:
        #Support information, Figure S2
        plotSI(config)
    elif itask == 17:
        plot5d_vs_monthly(config)
    elif itask == 18:
        #this generates Figure 2 (the ACE map parts)
        fig,axes = plt.subplots(2,1, figsize=(10,8),subplot_kw={'projection': ccrs.PlateCarree()})    
        
        ax0 = axes[0]
        ax1 = axes[1]

        #create color axes
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax)

        divider2 = make_axes_locatable(ax1)
        cax2 = divider2.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax2)
        
        plotACEMap(config, axes=(ax0, ax1), caxes=(cax,cax2), pcnt=None)
        
        ax0.set_extent([-180, 180, -60, 90])
        ax0.coastlines(resolution='110m', color='gray')
        ax1.set_extent([-180, 180, -60, 90])
        ax1.coastlines(resolution='110m', color='gray')

        ax0.set_xticklabels([])
        ax0.set_yticklabels([])        
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])        

        plt.subplots_adjust(wspace=0.15)
        plt.savefig(f'outputs/global_ace_map.eps')

        plt.close()
    elif itask == 19:
        plotBasinSI(config, region="north_america")
    elif itask == 20:
        #Figure 3
        plotCausalEffectHeatMap(config, region="north_america", binary=True)        

    else:
        raise Exception("Invalid options")
