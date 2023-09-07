#author: alex sun
#date: 0824
#find good grdc reference data
#Daily runoff data downloaded by using Download Station, selected such that drainage area>4e4 km2
#date: 0826, cleaned up for production. Use plotglobalmap and itask=2 for global
#rev date: 09072022, double check for grace monthly meeting
#rev date: 06092023, cleanup for manuscript
#rev date: 06232023, add plots for bivariate, joint occurrence, and CMI
#For bivariate, CMI vs. Joint Occurrence; for joint occurrence it's P and TWS; and CMI it's (TWS, Q|P)
#rev date: 07312023, consider remove season option, this is enabled by setting cfg.data.deseason to true
#rev date: 08012023, change to use fake 5d data generated from monthly solutions
#rev date: 08022023, cleanup and add annotations
#rev date: 08052023, implemented SWE removal  cfg.data.removeSWE
#=======================================================================================================
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import  numpy as np
import pickle as pkl
import sys
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
from dataloader_global import load5ddatasets,loadClimatedatasets,getCSR5dLikeArray
import glofas_us
from glofas_us import extractBasinOutletQ, loadGLOFAS4CONUS
from myutils import bandpass_filter,removeClimatology
from csr_monthly_dataloader import convertMonthlyTo5d, loadFake5ddatasets,loadFake5ddatasetsNoSWE

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
#number of neighbors
KNN = 6
#BASIN AREA
MIN_BASIN_AREA = 1.2e5 #[km2]

def getExtremeEvents(series, method='MAF', cutoff=90, transform=False, minDist=None, 
                     season=None, returnExtremeSeries=False):
    """   
    method: MAF, mean annual flood; POT peak over threshold 
    """
    #from pyextremes import EVA
    #transform Q to normal space
    if transform:
        val = PowerTransformer().fit_transform(series.to_numpy()[:,np.newaxis]).squeeze()
        series = pd.Series(val, index=series.index)

    if method == 'MAF':
        #extremes = series.groupby(series.index.year).agg(['idxmax', 'max'])  
        #extremes = pd.Series(extremes['max'].values, index=extremes['idxmax'])
        #asun0711, enforce minimum distance between events
        nEvent = 5
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
                #loop through the n largest events in each year 
                #if the distance between two adjacent events is smaller than minDist, record the event index
                for j in range(nEvent):
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

        extremes = series.loc[series>=thresh]
        extremes = extremes.sort_values(ascending=False)
        if not minDist is None:
            #remove close events            
            goodInd = [0]
            #the following double loop ensures the higher magnitude events are always selected over lower-magnitude adjacent events
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
    #is there a better way?
    for ix, item in enumerate(series.index.tolist()):
        for iy, time2 in enumerate(extremes.index.tolist()):
            if (item==time2):
                events[ix] = 1
    if len(np.where(events==1)[0]) != extremes.size:
        print ("warning:", len(np.where(events==1)[0]), extremes.size)

    #check all events are accounted for
    #assert(len(np.where(events==1)[0]) == extremes.size)
    if returnExtremeSeries:
        return events, extremes
    else:
        return events

def getBasinMask(stationID, region='north_america', gdf=None):    
    """ Get basin mask as a geopandas DF
    Param
    -----
    stationID, grdc_no
    region, one of the valid regions
    gdf, if not None, it contains the combined GDF for all stations a region
                      This is downloaded from GRDC as part of the station manual selection 
    Returns
    -------
    basin_gdf, gdf of the basin

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
            #08/28, add all_touched = True
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
    """
    gdf = getBasinMask(stationID=stationID, region=region, gdf=gdf)
    region = getRegionExtent(regionName="global")
    
    #TWS
    #07142023: use TWS 0.25 for masking, but ERA5 1deg for masking
    #          ERA5 0.25 deg is too big to load into memory
    #          Do this in two steps. Step 1: turn coarsen in load5ddatasets to True, reload to True in both load5ddatasets and loadClimatedatasets=>1deg data
    #          Step 2: turn coarsen to False, reload to True in load5ddatasets, but False in loadClimatedatasets => 0.25 deg data
    if xds is None:
        if not removeSWE:
            print ('!!! Load original TWS')

            xds,_, _ = loadFake5ddatasets(cfg=config, 
                region=region, 
                coarsen=False, 
                mask_ocean=config.data.mask_ocean, 
                startYear=datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
                endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year
            )
        else:
            print ('!!! load TWS w/ SWE removal')
            xds = loadFake5ddatasetsNoSWE(cfg=config,
                region=region,
                coarsen=False,
                mask_ocean=config.data.mask_ocean,
                startYear=datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
                endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year
            )
        print ('loaded tws array shape', xds.shape)
    basinTWS = getPredictor4Basin(gdf, lat, lon, xds)

    #Precip [use saved 5d data]
    if xds_Precip is None:
        xds_Precip = loadClimatedatasets(region, vartype='precip', daCSR5d= xds, \
                                        aggregate5d=True, 
                                        reLoad=config.data.reload_precip, 
                                        precipSource=config.data.precip_source,
                                        startYear= datetime.strptime(config.data.start_date, '%Y/%m/%d').date().year,
                                        endYear=datetime.strptime(config.data.end_date, '%Y/%m/%d').date().year,
                                        fake5d=True)

    basinP = getPredictor4Basin(gdf, lat, lon, xds_Precip)

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
    The conditional mutual information is a measure of how much uncertainty is shared by X and
    Y , but not by Z.
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
    dataframe = pp.DataFrame(data, 
            datatime = {0:np.arange(data.shape[0])}, 
            var_names=var_names,
            missing_flag=-999.)

    #Calculate CMI    
    #parcorr = ParCorr(significance='analytic')
    pcmci = PCMCI(
        dataframe=dataframe, 
        cond_ind_test=RobustParCorr(), #parcorr,
        verbosity=1)

    correlations = pcmci.get_lagged_dependencies(tau_min=cfg.causal.tau_min, tau_max=cfg.causal.tau_max, val_only=True)['val_matrix']
    
    if plotting:
        plt.figure()
        matrix_lags = np.argmax(np.abs(correlations), axis=2)
        tp.plot_densityplots(dataframe=dataframe, setup_args={'figsize':(15, 10)}, add_densityplot_args={'matrix_lags':matrix_lags});    
        #tp.plot_densityplots(dataframe=dataframe, add_densityplot_args={'matrix_lags':None})
        plt.savefig(f'outputs/test_density_plot{stationID}.png')
        plt.close()

        plt.figure()
        tp.plot_scatterplots(dataframe=dataframe, 
                setup_args={'figsize': (16,16), 'label_fontsize': 16},
                add_scatterplot_args={'matrix_lags':matrix_lags, 'color': 'blue', 'markersize':10, 'alpha':0.7})
        plt.savefig(f"outputs/test_scatterplot{stationID}.png")
        plt.close()

        plt.figure()
        tp.plot_timeseries(dataframe); 
        plt.savefig(f"outputs/test_timeseries{stationID}.png")
        plt.close()

    if saveDataFrame:
        #save dataframe for later use
        pkl.dump(dataframe, open(f'grdcdataframes/s_{stationID}.pkl', 'wb'))

    cmi_knn = CMIknn(
        significance='shuffle_test', 
        knn=10, #if <1, this is fraction of all samples
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
    """
    do causal analytics
    06202023
    see 
    https://github.com/jakobrunge/tigramite/blob/master/tutorials/case_studies/climate_case_study.ipynb
    """
    #assumption: target variable Q is always the first!!!
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
    
    #formulate links
    selected_links = {}
    # link_assumptions[j] = {(i, -tau):"-?>" for i in range(self.N) for tau in range(1, tau_max+1)}
    for i in range(len(var_names)):
        selected_links[i] = {}
    ivar=0    
    for jvar in range(len(var_names)):
        for ilag in range(tau_min, tau_max+1):
            #exclude Q
            if cfg.causal.exclude_Q:
                if (jvar != 0):
                    selected_links[ivar][(jvar, -ilag)]='-?>'
            else:
                selected_links[ivar][(jvar, -ilag)]='-?>'

    pc_alpha = 0.1

    if binary:
        cmi_symb = CMIsymb(significance='shuffle_test', n_symbs=None)   #this is too slow
        gsquared = Gsquared(significance='analytic')     
        pcmci_cmi_symb = PCMCI( dataframe=dataframe, cond_ind_test=gsquared)

        results = pcmci_cmi_symb.run_pcmci(link_assumptions=selected_links, 
                                        tau_min = tau_min, tau_max=tau_max, 
                                        pc_alpha=pc_alpha)
        #see ref at: https://github.com/jakobrunge/tigramite/blob/master/tutorials/causal_discovery/tigramite_tutorial_conditional_independence_tests.ipynb
        #CMI = G/(2*n_samples)
        outstr, sorted_links = print_significant_links(data.shape[1], var_names, 
                                p_matrix=results['p_matrix'],
                                val_matrix=results['val_matrix']/(2.*df.shape[0]),
                                alpha_level=0.05)
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=pcmci_cmi_symb.all_parents, tau_max=tau_max)
        med.fit_model_bootstrap(boot_blocklength=1, seed=28, boot_samples=200)
        print (pcmci_cmi_symb.all_parents)
        ace = med.get_all_ace()
        ce_boots = med.get_bootstrap_of(function='get_all_ace', function_args={}, conf_lev=0.9)
        print ("average causal effect of TWS,P ", ace )
        print (ce_boots)
    else:
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test= RobustParCorr() #cmi_knn,
        )    
        results = pcmci.run_pcmci(
            link_assumptions= selected_links, 
            tau_min=tau_min,
            tau_max=tau_max, pc_alpha=pc_alpha)
        
        outstr, sorted_links = print_significant_links(data.shape[1], var_names, 
                        p_matrix=results['p_matrix'],
                        val_matrix=results['val_matrix'],
                        alpha_level=0.05)

        Y = [(0,0)]
        X = [(1,tau) for tau in range(-1,-tau_max,-1)]
        S = [(2,tau) for tau in range(-1, -tau_max, -1)]
        
        med = LinearMediation(dataframe=dataframe)
        med.fit_model(all_parents=pcmci.all_parents, tau_max=tau_max)
        med.fit_model_bootstrap(boot_blocklength=1, seed=28, boot_samples=200)
        ace = med.get_all_ace()
        ce_boots = med.get_bootstrap_of(function='get_all_ace', function_args={}, conf_lev=0.9)
        print ("average causal effect of TWS,P ", ace)
        print (ce_boots)
    return outstr, sorted_links, results['p_matrix'],results['val_matrix'],ace,ce_boots

def fitScatterPlot(cfg, stationID, daTWS,daQ, river):
    #    
    tws = daTWS.values
    Q = daQ.values
    timestamp =daTWS.time.values
    
    tws_events = getExtremeEvents(pd.Series(tws.squeeze(), index=timestamp), method='MAF', cutoff=90, transform=False, minDist=cfg.event.t_win, 
                                season=None, returnExtremeSeries=False)
    q_events  = getExtremeEvents(pd.Series(Q.squeeze(), index=timestamp), method='MAF', cutoff=90, transform=False, minDist=cfg.event.t_win, 
                                season=None, returnExtremeSeries=False)
    
    assert (len(tws_events) == len(timestamp))

    #define quandrants of scatter plot
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

    plt.figure(figsize=(6,6))
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
    
    plt.xlabel('TWS [cm]')
    plt.ylabel('Q [m/(km^2 s]')
    plt.title(f'Station {stationID}, {river}')
    plt.legend(loc='best')
    if not cfg.data.deseason:
        plt.savefig(f'outputs/fake5d_scatter_raw_{stationID}_{cfg.event.t_win}.png')
    else:
        plt.savefig(f'outputs/noseason_fake5d_scatter_raw_{stationID}_{cfg.event.t_win}.png')
    plt.close()

def calculateMetrics(cfg, stationID, varDict, onGloFAS=False, river=None, binary=False):
    """Calculate MI
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

    if onGloFAS:
        Q  =varDict['Qs'].values
    else:
        Q  =varDict['Q'].values
        #========================
        #08032023, add scatter plot on raw data
        #========================
        fitScatterPlot(cfg, stationID, varDict['TWS'], varDict['Q'], river=river)

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
        #convert to gaussian
        #Q = skp.PowerTransformer(method='box-cox').fit_transform(Q.reshape(-1,1)).squeeze()
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
    causal_str_bin, sorted_links_bin, p_mat_bin ,val_mat_bin,ace_bin,ce_boot_bin  = doCausalAnalytics(cfg, dfBinary, binary=True)
    
    #do everything real-valued
    causal_str, sorted_links, p_mat ,val_mat,ace, ce_boot = doCausalAnalytics(cfg, dfa)

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
            'ce_boot': ce_boot
    }

def getRegionBound(region, source='GRDC'):
    """
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
    """
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

def getCatalog(region,source='grdc'):
    rootdir = f'/home/suna/work/grace/data/{source}'
    geosonfile = os.path.join(rootdir, f'{region}/stationbasins.geojson')

    gdf = gpd.read_file(geosonfile)
    gdf['grdc_no'] = pd.to_numeric(gdf['grdc_no'], downcast='integer')
    return gdf

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
    dfStation = getStationMeta(gdf)
    reGen = cfg.regen
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
                        #08022023, get fake 5d data interpolated from monthly
                        #only store xds on first call
                        basinTWS, basinP, xds, xds_Precip = getBasinData(config=cfg, 
                                    stationID=stationID, 
                                    lat=lat, lon=lon, gdf=gdf, 
                                    region=region,
                                    removeSWE= cfg.data.removeSWE
                                    )
                    else:
                        basinTWS, basinP, _, _ = getBasinData(config=cfg, stationID=stationID, \
                                                            region=region, gdf=gdf, lat=lat, lon=lon, \
                                                            xds=xds, xds_Precip=xds_Precip,
                                                            removeSWE= cfg.data.removeSWE)
                    kwargs = {
                        "method": "rx5d",
                        "aggmethod":'max',
                        "name":'Q'
                        }
                    #convert Q to CSR5d intervals
                    #this steps may introduce NaN values
                    daQ = getCSR5dLikeArray(daQ, basinTWS, **kwargs)

                    if cfg.data.deseason:
                        daQ = removeClimatology(daQ, varname='Q', plotting=False)
                        basinTWS = removeClimatology(basinTWS, varname='TWS',plotting=False)
                        basinP = removeClimatology(basinP, varname='P',plotting=False)

                    varDict={'TWS':basinTWS, 'P':basinP, 'Q':daQ}
                    
                    metricDict = calculateMetrics(cfg,stationID, varDict, river=row['river_x'])
                    print (f"{stationID},{lat},{lon},{row['river_x']},{row['area_hys']}") # 'MI', {metricDict['mi']}, 'CMI', {metricDict['cmi']}")
                    print (f"CMI {metricDict['cmi']}")
                    allScores[stationID] = metricDict
            except Exception as e: 
                raise Exception (e)

        #07312023, add test for no seasonal
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_monthly.pkl', 'wb'))
            else:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_noseason_monthly.pkl', 'wb'))
        else:
            if not cfg.data.deseason:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_Q_monthly.pkl', 'wb'))
            else:
                pkl.dump(allScores, open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_Q_noseason_monthly.pkl', 'wb'))
    else:
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_monthly.pkl', 'rb'))
            else:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_noseason_monthly.pkl', 'rb'))
        else:
            if not cfg.data.deseason:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_Q_monthly.pkl', 'rb'))
            else:
                allScores = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_Q_noseason_monthly.pkl', 'rb'))


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
    source, can be either 'grdc' or 'gldas'
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
    
    assert(variable2 in ['CMI', 'P_Q'])
    if variable2 == 'CMI':
        dictkey = 'cmi'
    else:
        dictkey = 'P'

    
    if cfg.maps.global_basin == 'majorbasin':
        shpfile = os.path.join('maps', 'Major_Basins_of_the_World.shp')
    elif cfg.maps.global_basin == 'hydroshed':
        shpfile = os.path.join('maps/hydrosheds', hydroshedMaps[region])
    gdfHUC2 = gpd.read_file(shpfile)

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

    #plot HUC2 boundaries
    gdfHUC2.plot(ax=ax, facecolor="none", edgecolor="w", legend=False)       
    
    #plot bivariates
    if source=='grdc':
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_monthly.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_all_events_monthly.pkl', 'rb'))
            else:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_noseason_monthly.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_all_events_noseason_monthly.pkl', 'rb'))
                
        else:
            if not cfg.data.deseason:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_Q_monthly.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_all_events_monthly.pkl', 'rb'))
            else:
                outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}_Q_noseason_monthly.pkl', 'rb'))        
                outputDict2 = pkl.load(open(f'grdcresults/{region}_all_events_noseason_monthly.pkl', 'rb'))

        validStations = list(outputDict.keys())
            
    elif source=='glofas':
        if cfg.causal.exclude_Q:
            if not cfg.data.deseason:
                outputDict = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.cutoff}_monthly.pkl', 'rb'))                    
                outputDict2 = pkl.load(open(f'grdcresults/glofas_{region}_all_events_monthly.pkl', 'rb'))
            else:
                outputDict = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.cutoff}_noseason_monthly.pkl', 'rb'))                    
                outputDict2 = pkl.load(open(f'grdcresults/glofas_{region}_all_events_noseason_monthly.pkl', 'rb'))
        else:
            outputDict = pkl.load(open(f'grdcresults/{region}_GLOFAS_MIScores_{cfg.event.cutoff}_Q_noseason_monthly.pkl', 'rb'))        
            outputDict2 = pkl.load(open(f'grdcresults/glofas_{region}_all_events_noseason_monthly.pkl', 'rb'))
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
    
    #plt.savefig(f'outputs/{source}_bivariate_{region}_{dictkey}.png')
    #plt.close()


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

        
def plotGRanD_MaxLag(cfg,region):
    """Plot Figure 1 Mississippi plot 
    """
    def plotSubplot(ax, columnName, markerColumnName, colorbar_label,subfig_title,plotBasemapLegend=False):
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)       
        fig.add_axes(cax)

        gdfSingle.plot(ax=ax,edgecolor='black', facecolor='none')
        gdfDam.plot(ax=ax, edgecolor='silver', facecolor='none')
        gdfRiver.plot(ax=ax, color='lightskyblue', alpha=0.5)

        gdfCol = gpd.GeoDataFrame(dfStation[columnName], 
                                geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
        markersizes = dfStation[markerColumnName].to_numpy()
        #get a unique list of markersizes
        uniquevals = np.unique(markersizes)
        #0705, because tau_min starts from 1 now, so we include all unique values
        uniquevals = uniquevals[0:]
        markersizes = markersizes*10
        
        #markersize: number of lags, vmax: max lag time
        gdfCol.plot(column=columnName, ax=ax, cax=cax, cmap=cmap, marker='o', markersize= markersizes, 
                    legend_kwds={'shrink': 0.5}, legend=True, 
                    vmin=0, vmax=1)
        cax.set_ylabel(colorbar_label, fontsize=14)
        sizeList = np.array(uniquevals, dtype=int)
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
        ax.text(0.90,0.96, subfig_title, fontsize=16, transform=ax.transAxes)
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

    fig, axes = plt.subplots(2,1,figsize=(16, 16))
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

    outputDict = pkl.load(open(f'grdcresults/{region}_GRDC_MIScores_{cfg.event.cutoff}.pkl', 'rb'))        
    validStations = list(outputDict.keys())
    outputDict2 = pkl.load(open(f'grdcresults/{region}_all_events.pkl', 'rb'))

    dfStation = dfStation[dfStation['grdc_no'].isin(validStations)]

    print (dfStation.columns)

    dfStation['coordinates'] = list(zip(dfStation.long, dfStation.lat))
    dfStation['coordinates'] = dfStation['coordinates'].apply(Point)
    dfStation = gpd.GeoDataFrame(dfStation, geometry='coordinates')
    dfStation = dfStation.set_crs('epsg:4326')
    dfStation = gpd.clip(dfStation, gdfSingle)

    cmap = ListedColormap(sns.color_palette(palette="OrRd", n_colors=10).as_hex())    
    #cmap.colors[0] = '#BDC3C7'

    #var_names = ['Q', 'TWS', 'P']
    #alpha_level = 0.05

    #DF for collecting metrics
    dfMetrics = pd.DataFrame(np.zeros((len(validStations), 6)), index=validStations)
    dfMetrics.columns=['rhoTWS', 'TWSLags', 'rhoP', 'PLags', 'SI_TWS', 'SI_P']

    #loop through all stations in the basin mask
    rhoTWS=[]
    rhoP = []
    for stationID in validStations:
        stationDict = outputDict[stationID]
        eventDict = outputDict2[stationID]

        sorted_links = stationDict['sorted_link']
        print (sorted_links)
        
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
    columnName = 'rhoP'
    markerColumnName = 'PLags'

    #=========plot P subplot
    columnName = 'rhoP'
    markerColumnName = 'PLags'
    plotSubplot(axes[1], columnName=columnName, markerColumnName=markerColumnName, colorbar_label='Max corr', subfig_title='Precip')

    plt.subplots_adjust(wspace=0.07,hspace=0.05)
    
    plt.savefig("outputs/grand_lagplot.png")
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


if __name__ == '__main__':    
    from myutils import load_config
    config = load_config('config.yaml')

    itask = 8
    if itask == 1:
        #turn reGen to True to reprocess all regions one by one to get MI, CMI, annual maxima
        for region in config.data.regions:
            main(cfg=config, region=region)
    elif itask == 8: 
        #Figure 1A (grdc), Figure 1B (glofas), 
        #Figure 1C is generated by using compareGRDC_GloFAS()
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
                plt.savefig('outputs/grdc_bivariate_P_monthly.png')
            else:
                plt.savefig('outputs/grdc_bivariate_P_noseason_monthly.png')
            plt.close()
        else:
            fig,ax = plt.subplots(1,1, figsize=(12, 7.5))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad="2%")

            plotBivariate(cfg=config, region="north_america", ax=ax, source='glofas', variable2='P_Q', ax_legend=cax)
            if not config.data.deseason:
                plt.savefig("outputs/glofas_bivariate_P_monthly.png")
            else:
                plt.savefig("outputs/glofas_bivariate_P_noseason_monthly.png")
            plt.close()
    else:
        raise Exception("Invalid options")
