#author: Alex Sun
#date: 12/10/2022
#analyze preditor importance 
#date: 12/19/2022, revisit
#date: 01/05/2023, cleanup
#date: 01/07/2023, add causal lag discovery
#gdf blog, https://www.martinalarcon.org/2018-12-31-d-geopandas/
#===================================================================
import numpy as np
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
import sys,os
import pandas as pd
import geopandas as gpd

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,roc_auc_score,f1_score,confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.stats import kendalltau,pearsonr
import tqdm
import pickle as pkl
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from myutils import getGRACEDA, getBasinData,calculateBasinAverage,getFPI,getExtremeEvents,getFPIData,getUFPIData
from dataloader_global import loadPrecipDataSets, loadAirTempDataSets
from lstmclassifier import genDataset, lstmMain, torchMetrics
from copulagen import gridCopula, gridJRP, gridEventCount

#define minimum drainage area
BASINAREA = 9e4

def getRootDir():
    """Get root directory of grdc folder """
    return '/work/02248/alexsund/maverick2/grace/data/grdc'

def getCatalog(region):
    """ Get GRDC catalog for region
    """
    rootdir = getRootDir()
    geosonfile = os.path.join(rootdir, f'{region}/stationbasins.geojson')
    gdf = gpd.read_file(geosonfile)
    gdf['grdc_no'] = pd.to_numeric(gdf['grdc_no'], downcast='integer')
    return gdf

def getStationMeta(dfRegion):
    """Load station information
    """
    grdcfile =  os.path.join(getRootDir(), 'GRDC_Stations.csv')
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

def readStationSeries(stationID, region='north_america'):
    """ Parse GRDC runoff time series 
        Note: the null values are dropped at this stage 
    Params
    ------
    stationID: id of the GRDC station

    """
    rootdir = getRootDir()
    
    stationfile = os.path.join(rootdir, f'{region}/{stationID}_Q_Day.Cmd.txt')
    #skip 37 header rows
    df = pd.read_csv(stationfile, encoding = 'unicode_escape', skiprows=37, index_col=0, delimiter=';', usecols=[0, 2], header=None)    
    df.columns=['Q']    
    df.index = pd.to_datetime(df.index)
    df.index.names = ['time']
    #drop bad values
    df= df[df.Q>0]
    return df

def getCSR5dLikeArray(da, daCSR5d, **kwargs):
    """ Make arrays that have the same length as the CSR5d
    """
    #get CSR5d dates
    timeaxis = daCSR5d['time'].values
    kwargs.setdefault("method", "rx5d")
    kwargs.setdefault("aggmethod", 'sum')
    kwargs.setdefault("name", 'variable')

    method = kwargs['method']
    
    if method=='rx5d':
        #maximum consequtive 5-day
        #get the 5-day total amount for precip, centered at each CSR5d date
        if kwargs['aggmethod'] == 'sum':
            daR5d = da.rolling(time=5, center=True,min_periods=1).sum()
        #get the rolling mean on daily data (this is suitable for state types), centered at each CSR5d date
        elif kwargs['aggmethod'] == 'average':
            daR5d = da.rolling(time=5, center=True,min_periods=1).mean()        
        #do not do anything
        elif kwargs['aggmethod'] == 'max':
            daR5d = da
        else:
            raise NotImplementedError()            
        halfwin = 2 #[day]
        #form CSR5d like time series
        bigarr = []
        for aday in timeaxis:
            aday = pd.to_datetime(aday,format='%Y-%m-%d').date()            
            #for a 5-day window centered at aday
            offsetDateStart = datetime.date.strftime(aday - relativedelta(days=halfwin),'%Y-%m-%d')
            offsetDateEnd   = datetime.date.strftime(aday + relativedelta(days=halfwin), '%Y-%m-%d')                        
            tmpda = daR5d.sel(time = slice(offsetDateStart,offsetDateEnd))
            if not tmpda.time.size==0:
                #take the max of the 5-day statistic (for mean, sum, and max)
                if kwargs['aggmethod'] != 'max':
                    tmpda = tmpda.max(dim='time',skipna=True)            
                else:
                    #sun0105, use the middle-day value for streamflow
                    tmpda = tmpda.isel(time=2)
                bigarr.append(tmpda.values)
            else:
                bigarr.append(np.NaN)
        bigarr = np.stack(bigarr,axis=0)
        daNew = xr.DataArray(bigarr, dims=daCSR5d.dims, coords=daCSR5d.coords, name=kwargs['name'])
    else:
        raise Exception("method not implemented")
    return daNew

def featureExtract(inputDict):
    """
    tsDict, dictionary of dataframe [not used!!!]
    """
    from tsfresh.utilities.distribution import MultiprocessingDistributor
    from tsfresh.feature_extraction import extract_features,ComprehensiveFCParameters
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute,roll_time_series

    umag = inputDict['umag']
    tws = inputDict['tws']
    runoff = inputDict['Q']

    #form tsfresh df
    umag = umag.loc[:, ['umag']]
    tws = tws.loc[:,['tws']]    
    Xdf = pd.concat([umag, tws], axis=1)
    Xdf.reset_index(inplace=True)
    Xdf['id'] = 1

    runoff = runoff.loc[:,['Q']]
    eventQ = getExtremeEvents(runoff)

    print ('extracing features ...')
    Distributor = MultiprocessingDistributor(
                    n_workers=4,
                    disable_progressbar=False,
                    progressbar_title="Feature Extraction")
    pd.set_option('display.max_rows', 50)
    windowsize = 4
    df_rolled = roll_time_series(Xdf, column_id="id", column_sort="time", max_timeshift=windowsize, min_timeshift=windowsize)
    fc_parameters = {
        "mean": None,
        "mean_second_derivative_central": None,
        "skewness": None,
        "variation_coefficient": None,
        "sum_values" : None
    }

    extracted = extract_features(df_rolled,column_id="id", column_sort="time", 
                distributor=Distributor,
                default_fc_parameters=fc_parameters)    
    print (extracted.shape)
    eventQ = eventQ.iloc[windowsize:,:]
    
    features_filtered = select_features(extracted, eventQ )


def getBasinMask(stationID, region='globalref', gdf=None):    
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
    #suppress the SettingWithCopyWarning warning
    basin_gdf = gdf.loc[ gdf['grdc_no']==stationID].copy(deep=True)
    return basin_gdf

def getPrecipDA(regionExtents, testDA, useImpute=False, datasource='CSR',precipsource='ERA5',coarsen=False):
    """Extract precip data according to extent in testDA
       set reLoad to True to reload data

    testDA, dataarray including information on time
    coarsen, upsample to 1 degree
    precipsource, ERA5 or GPM
    datasource, CSR or ITSG
    """
    reLoad=False
    print (f'use {precipsource} precip data')
    precipDA = loadPrecipDataSets(regionExtents, daCSR5d=testDA, aggregate5d=True, aggregateMethod='CSR5d', 
                reLoad=reLoad, remove30yearnormal=False, useImpute=useImpute, datasource=datasource,precipsource=precipsource, coarsen=coarsen)    
    return precipDA

def getAirTempDA(regionExtents, testDA, useImpute=False, datasource='CSR', coarsen=False):
    """extract air temp data according to extent in testDA
    testDA, dataarray including information on time
    """
    reLoad=False
    airtempDA = loadAirTempDataSets(regionExtents, daCSR5d= testDA, aggregate5d=True, aggregateMethod='CSR5d', 
                reLoad=reLoad, useImpute=useImpute, datasource=datasource, coarsen=coarsen)    
    return airtempDA

def fitArray(X, y, modeltype):
    """Main routine for classification using xgboost or logisticregression
    args, parsed inputs
    X, input data
    y, binary event series

    """
    from xgboost import XGBClassifier
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    #perform scaling on X
    print ('modeltype  = ', modeltype)
    if modeltype == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs')
    elif modeltype == 'decisiontree':
        model = DecisionTreeClassifier()    
    elif modeltype == 'randomforest':
        model = RandomForestClassifier(n_estimators=1000)
    elif modeltype == "xgboost":
        
        fix_params = {'learning_rate': 0.002, 'objective': 'binary:logistic', 'subsample': 0.8, 'max_delta_step': 1, 'max_depth':3, 
                    'eval_metric': 'auc',  'n_jobs': 4, 'n_estimators':1000, 'scale_pos_weight':10, 'use_label_encoder':False}  
        model = XGBClassifier(learning_rate=fix_params['learning_rate'],
                              objective = fix_params['objective'],
                              max_depth = fix_params['max_depth'],
                              n_estimators = 1000,
                              max_delta_step = fix_params['max_delta_step'],
                              use_label_encoder=False,
                              scale_pos_weight = 10,
                              eval_metric = 'auc', 
                              subsample = 0.8, 
        )
        #csv = GridSearchCV(XGBClassifier(**fix_params), cv_params, scoring = 'f1', cv = 5, n_jobs=2)        
    else:
        raise ValueError("Invalid selection")

    y = y.squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_trains, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)    

    # learn relationship from training data
    if modeltype == 'xgboost':
        #uncomment to do grid search
        #csv.fit(X_trains, y_train)
        #print (csv.best_params_)        
        dtrain = xgb.DMatrix(X_trains, y_train)
        dval = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test, y_test)
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        #model = xgb.train(fix_params, dtrain, num_boost_round = 100, evals=evallist)
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X_trains, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
        # summarize performance
        print('Mean ROC AUC: %.5f' % np.mean(scores))
        model.fit(X_trains, y_train)
    else:
        model.fit(X_trains, y_train)

    #get precision
    if modeltype == 'logistic_regression':
        y_pred = model.predict(X_test)
        """
        yprob = model.predict_proba(X_test)
        y_pred = np.zeros((yprob.shape[0],1))
        y_pred[yprob[:,1]>0.018] = 1.0
        for ix, item in enumerate(y_test):
            print (item, yprob[ix], y_pred[ix])
        """
        # bootstrap predictions
        
        PODs=[]
        FARs=[]
        f1scores=[]

        n_iterations = 1000
        for i in range(n_iterations):
            X_bs, y_bs = resample(X_trains, y_train, replace=True)
            # make predictions
            yprob = model.predict_proba(X_bs)
            y_pred = np.zeros((yprob.shape[0],1))
            y_pred[yprob[:,1]>0.02] = 1.0

            CM = confusion_matrix(y_bs, y_pred)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            #false alarm rate
            FAR = FP/(TP+FP)
            POD = TP/(TP+FN)
            CSI = TP/(TP+FN+FP)

            # evaluate model
            f1scores.append(f1_score(y_bs, y_pred, average='weighted'))
            PODs.append(POD)
            FARs.append(FAR)
        
        print ('median f1', np.median(f1scores), ' POD', np.median(POD), 'FAR', np.median(FAR))
        
    else:
        y_pred = model.predict(X_test)        

    return    
    targets = ['no_flood', 'flood']
    print (classification_report(y_test, y_pred, target_names=targets))

    print ('='*80)
    print ('Feature importance')

    """
    if modeltype in ['logistic_regression']:
        #The positive scores indicate a feature that predicts class 1, 
        # whereas the negative scores indicate a feature that predicts class 0.
        nVar = int(len(model.coef_[0,:])/args.seq_len)
        for ii in range(nVar):
            if ii == 0:
                for jj in range(args.seq_len,0, -1):
                    print (f't-{jj}, ', end='')
                print ('') 
            for jj in range(args.seq_len):                    
                print ('%.2f,' % (model.coef_[0, (ii-1)*args.seq_len+jj]), end='')
            print ('')
    elif modeltype == 'randomforest': 
        pi = model.feature_importances_
        #pi = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
        #pi = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=0)
        #pi = pi.importances_mean
        #pi = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        nVar = int(len(pi)/args.seq_len)
        for ii in range(nVar):
            if ii == 0:
                for jj in range(args.seq_len,0, -1):
                    print (f't-{jj}, ', end='')
                print ('') 
            for jj in range(args.seq_len):                    
                print ('%.3f,' % (pi[(ii-1)*args.seq_len+jj]), end='')
            print ('')
    print ('='*80)
    """
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
            print(string)
        return string, sorted_links

def doCausalMetrics(x, y, varnames, binaryData=True, datatime=None):    
    """
    x and y should be binary events
    """
    from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    np.random.seed(20230107)
    cmi_symb = CMIsymb(significance='shuffle_test', n_symbs=None)
    if datatime is None:
        datatime = np.arange(data.shape[0])
    
    data = np.c_[x, y]
    dataframe = pp.DataFrame(data, 
            datatime = datatime,
            var_names=varnames)
    tau_max = 12
    pc_alpha = 0.1
    tau_min = 1
    #assumption: target variable Q is always the last!!!
    selected_links = {}
    for ivar in range(len(varnames)):
        selected_links[ivar] = []
        for jvar in range(len(varnames)):
            for ilag in range(tau_min, tau_max+1):
                #if  ivar != len(varnames)-1:
                #    selected_links[ivar].append((jvar, -ilag))
                #else:
                if (jvar != len(varnames)-1):
                    selected_links[ivar].append((jvar, -ilag))

    if binaryData:
        pcmci_cmi_symb = PCMCI( dataframe=dataframe, cond_ind_test=cmi_symb)

        results = pcmci_cmi_symb.run_pcmci(selected_links=selected_links, tau_min = tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
        outstr, sorted_links = print_significant_links(data.shape[1], varnames, p_matrix=results['p_matrix'],
                                    val_matrix=results['val_matrix'],
                                    alpha_level=0.05)
    else:
        parcorr = ParCorr(significance='analytic')
        cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
        )    

        results = pcmci.run_pcmci(selected_links= selected_links, tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
        outstr, sorted_links = print_significant_links(data.shape[1], varnames, p_matrix=results['p_matrix'],
                                    val_matrix=results['val_matrix'],
                                    alpha_level=0.05)

    return outstr, sorted_links, results['p_matrix'],results['val_matrix']

def assembleAllFeatureMatrix(goodInd, inputDict, trimtoITSG=False):
    umagDF  = inputDict['umag'][['umag']] 
    twsDF   = inputDict['tws'][['tws']]
    #asun0106, to compare to ITSG
    if trimtoITSG:
        twsDF = twsDF[twsDF.index<pd.to_datetime('2016-08-31')]

    precipDF= inputDict['precip'][['precip']] 
    fpiDF   = inputDict['fpi'][['fpi']]
    t2m     = inputDict['t2m'][['t2m']]
    
    #umagDF includes all data
    validInd = []
    for i in goodInd:
        if i<twsDF.shape[0]:
            validInd.append(i)
    
    runoffDF = inputDict['Q'].iloc[validInd,:]
    runoffDF = runoffDF.loc[:, ['Q']] 
    runoff = runoffDF.to_numpy()
    
    precipDF = precipDF.iloc[validInd,:]
    umagDF = umagDF.iloc[validInd,:]
    twsDF = twsDF.iloc[validInd,:]
    fpiDF = fpiDF.iloc[validInd, :]
    
    """
    fig,axes = plt.subplots(1,1, figsize=(12,12))
    twsDF.plot(ax=axes, color='r', label='tws')
    precipDF.plot(ax=axes, color='k', label='precip')
    fpiDF.plot(ax=axes, color='purple', label='FPI')

    ax = axes.twinx()
    umagDF.plot(ax=ax, color='g', label='umag')
    plt.savefig('test_all.png')
    plt.close()

    sys.exit()
    """
    #remove the null values
    df = pd.DataFrame(np.c_[twsDF.to_numpy(),umagDF.to_numpy(), precipDF.to_numpy(), fpiDF.to_numpy(), runoff], index=runoffDF.index)

    print ('before dropping na values', df.shape)
    #now remove all rows with nan
    df.dropna(how='any', inplace=True)
    print ('after dropping na values', df.shape)

    A  = df.to_numpy()
    return A, df

def getMetrics(goodInd, inputDict, cutoff=90, trimtoITSG=False, method_name='POT'):
    """Calculate extreme metrics
    inputDict, dictionary of input time series
    cutoff, threshold for POT only
    trimtoITSG, if true, limit data to 2016/08/31 to be consistent with ITSG
    """
    def getCorr(arr1, arr2, names, maxlags= 12):
        TYPE = 'pearson'
        corrs = []
        pvals  = []
        if TYPE == 'kendall':
            for ilag in range(maxlags):
                if ilag == 0:
                    cval, pval = kendalltau (arr1, arr2)
                else:
                    cval, pval = kendalltau(arr1[ilag:], arr2[:-ilag])
                corrs.append(cval)
                pvals.append(pval)
        else:
            for ilag in range(maxlags):
                if ilag == 0:
                    cval, pval = pearsonr(arr1, arr2)
                else:
                    cval, pval = pearsonr(arr1[ilag:], arr2[:-ilag])
                corrs.append(cval)
                pvals.append(pval)

        lag = np.argsort(np.abs(corrs))[-1]
        corr = corrs[lag]
        pval = pvals[lag]
        print ('corr', corr, 'p val', pval, 'lag', lag)
        return corr, pval, lag

    assert (method_name in ['POT', 'MAF'])
    A, df = assembleAllFeatureMatrix(goodInd, inputDict, trimtoITSG=True)    

    action = 'corr_only'
    if action == 'bootstrap':
        #bootstrapping
        nRepeat = 1000
        np.random.seed(0)
        TWSAll = np.zeros((nRepeat))
        UmagAll = np.zeros((nRepeat))
        PAll = np.zeros((nRepeat))
        FPIAll = np.zeros(nRepeat)
        for i in tqdm.tqdm(range(nRepeat)):
            ind = np.random.choice(np.arange(A.shape[0]), size=A.shape[0]) 
            arr = A[ind,:]   
            tws,umag, P, FPI, Q = (arr[:,0],arr[:,1],arr[:,2],arr[:,3], arr[:,4])


            eventQ = getExtremeEvents(pd.Series(Q.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False)
            eventTWS = getExtremeEvents(pd.Series(tws.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False)
            eventUmag = getExtremeEvents(pd.Series(umag.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False)
            eventP = getExtremeEvents(pd.Series(P.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False)
            eventFPI = getExtremeEvents(pd.Series(FPI.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False)
        
            TWSAll[i] = getCorr(eventQ, eventTWS, ('Q', 'TWS'))[0]
            UmagAll[i] = getCorr(eventQ, eventUmag, ('Q', 'U'))[0]
            PAll[i] = getCorr(eventQ, eventP, ('Q', 'P'))[0]
            FPIAll[i] = getCorr(eventQ, eventFPI, ('Q', 'FPI'))[0]

        conf_TWS = np.percentile(TWSAll, [2.5, 97.5])
        conf_Umag = np.percentile(UmagAll, [2.5, 97.5])
        conf_P = np.percentile(PAll, [2.5, 97.5])
        conf_FPI = np.percentile(FPIAll, [2.5, 97.5])
        print (np.mean(TWSAll), conf_TWS)
        print (np.mean(UmagAll), conf_Umag)
        print (np.mean(PAll), conf_P)
        print (np.mean(FPIAll), conf_FPI)
    else:
        #
        # Note all the f1score and classification reports are probably meaningless 
        # because we are comparing apples to oranges
        # Should only focus on causal analytics 
        #
        tws,umag, P, FPI, Q = (A[:,0],A[:,1],A[:,2],A[:,3], A[:,4])
        #try:    
        selectedSeason = None
        eventQ = getExtremeEvents(pd.Series(Q.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False, season=selectedSeason)
        eventTWS = getExtremeEvents(pd.Series(tws.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False,season=selectedSeason)
        eventUmag = getExtremeEvents(pd.Series(umag.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False,season=selectedSeason)
        eventP = getExtremeEvents(pd.Series(P.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False,season=selectedSeason)
        eventFPI = getExtremeEvents(pd.Series(FPI.squeeze(), index=df.index), method=method_name, cutoff=cutoff, transform=False,season=selectedSeason)
        
        #===========asun0108, causal effect===========
        #1 use all data
        causal_str, sorted_links, p_mat ,val_mat = doCausalMetrics(np.c_[P, FPI, tws], Q, 
                            varnames=['P', 'FPI', 'TWS', 'Q'], binaryData=False,
                            datatime=pd.to_datetime(df.index))

        #2 use extreme events only
        causal_str_bin, sorted_links_bin, p_mat_bin ,val_mat_bin = doCausalMetrics(np.c_[eventP, eventFPI, eventTWS ], eventQ, 
                            varnames=['P', 'FPI', 'TWS', 'Q'], 
                            datatime=pd.to_datetime(df.index))

        print ('TWS ', end='')
        getCorr(eventQ, eventTWS, ('Q', 'TWS'))
        print ('Umag ', end='')
        getCorr(eventQ, eventUmag, ('Q', 'U'))
        print ('Precip ', end='')
        getCorr(eventQ, eventP, ('Q', 'P'))
        print ('FPI ', end='')
        getCorr(eventQ, eventFPI, ('Q', 'FPI'))
        print ("")

        #do f1-score here?
        print ('Q vs. TWS')
        targets = ['no_flood', 'flood']
        labels = [0, 1]
        print (classification_report(eventQ, eventTWS, labels = labels, target_names=targets))
        print ('roc_auc_score', roc_auc_score(eventQ, eventTWS))
        print ('Q vs. Umag')
        print (classification_report(eventQ, eventUmag, target_names=targets))
        print ('roc_auc_score', roc_auc_score(eventQ, eventUmag))
        print ('Q vs. Precip')
        print (classification_report(eventQ, eventP, labels = labels, target_names=targets))
        print ('roc_auc_score', roc_auc_score(eventQ, eventP))
        print ('Q vs. FPI')
        print (classification_report(eventQ, eventFPI, labels = labels, target_names=targets))
        print ('roc_auc_score', roc_auc_score(eventQ, eventFPI))
        print ("*"*80)

        #=======
        print (f1_score(eventQ, eventTWS, average='weighted'))

        #convert to DF and plot
        """
        Q = PowerTransformer(method='box-cox').fit_transform(Q[:,np.newaxis])
        P = PowerTransformer().fit_transform(P[:,np.newaxis])
        tws = PowerTransformer().fit_transform(tws[:,np.newaxis])
        
        eventQdf = pd.Series(Q.squeeze(), index=df.index)
        eventPdf = pd.Series(P.squeeze(), index=df.index)
        eventFPIdf = pd.Series(FPI.squeeze(), index=df.index)
        eventTWSdf = pd.Series(tws.squeeze(), index = df.index)
        
        fig,axes = plt.subplots(1,1, figsize=(16,9))
        axes.plot(eventTWSdf, 'r', label='tws')
        axes.plot(eventPdf, 'k', label='precip')
        ax = axes.twinx()
        ax.plot(eventFPIdf, 'purple', label='FPI')
        axes.plot(eventQdf, 'g-x', label='Q')
        ax.legend()                        
        axes.legend()
        plt.savefig('test_all.png')
        plt.close()
        """
        #except: 
        #    print ('bad stations')
        CM = confusion_matrix(eventQ, eventFPI)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        #false alarm rate
        FAR = FP/(TP+FP)
        POD = TP/(TP+FN)
        CSI = TP/(TP+FN+FP)
        return {'auroc': roc_auc_score(eventQ, eventFPI), 
                'f1': f1_score(eventQ, eventFPI, average='weighted'), 
                'FAR': FAR, 'POD': POD, 'CSI':CSI, 
                'causal': causal_str,
                'sorted_link': sorted_links,
                'p_mat': p_mat,
                'val_mat': val_mat,
                'causal_bin': causal_str_bin,
                'sorted_link_bin': sorted_links_bin,
                'p_mat_bin': p_mat_bin,
                'val_mat_bin': val_mat_bin,
        }

def formMatrix(goodInd, inputDict, seq=3):
    """Form feature matrix 
    Params
    ------
    goodInd, this is from twsnorm class, indices with consecutive days
    inputDict, dictionary of the feature arrays
    seq, lookback length
    """
    #only take columns of interest
    umagDF  = inputDict['umag'][['umag']] 
    twsDF   = inputDict['tws'][['tws']]
    precipDF= inputDict['precip'][['precip']] 
    #extract FPI
    fpiDF = getFPI(precipDF, twsDF, nLags=3)    

    #only keep indices that are part of twsDF
    validInd = []
    for i in goodInd:
        if i<twsDF.shape[0]:
            validInd.append(i)

    runoffDF = inputDict['Q'].iloc[validInd,:]
    runoffDF = runoffDF.loc[:, ['Q']] 
    runoff = runoffDF.to_numpy()

    precipDF = precipDF.iloc[validInd,:]
    umagDF = umagDF.iloc[validInd,:]
    twsDF = twsDF.iloc[validInd,:]
    fpiDF = fpiDF.iloc[validInd, :]

    df = pd.DataFrame(np.c_[twsDF.to_numpy(),umagDF.to_numpy(), precipDF.to_numpy(), fpiDF.to_numpy(), runoff], index=runoffDF.index)
    print ('before dropping na values', df.shape)
    #now remove all rows with nan
    df.dropna(how='any', inplace=True)
    print ('after dropping na values', df.shape)
    A  = df.to_numpy()

    tws,umag, P, FPI, Q = (A[:,0],A[:,1],A[:,2],A[:,3], A[:,4])

    #Normalization [this needs to be done on the training data]
    Q = PowerTransformer(method='box-cox').fit_transform(Q[:,np.newaxis])
    P = PowerTransformer().fit_transform(P[:,np.newaxis])
    tws = PowerTransformer().fit_transform(tws[:,np.newaxis])
    UMAG = PowerTransformer().fit_transform(umag[:,np.newaxis])

    #Form feature array
    withUMAG = True
    if withUMAG:
        nPredictors  = 4
    else:
        nPredictors  = 3

    #!!!because we are now using whole array, chop off the seq
    for ix,item in enumerate(validInd):
        if item>=seq:
            break
    startInd = ix
    validInd = validInd[startInd:]
    runoff = runoff[startInd:,:]
    runoffDF = runoffDF.iloc[startInd:, :]
    #
    A = np.zeros((len(validInd), nPredictors*seq))
    for ix, i in enumerate(validInd):
        A[ix:ix+1, :seq] = tws[i-seq:i].T
        A[ix:ix+1, seq:2*seq] = P[i-seq:i].T
        A[ix:ix+1, 2*seq:3*seq] = fpiDF.iloc[i-seq:i,:].to_numpy().T
        if withUMAG:
            A[ix:ix+1, 3*seq:4*seq] = UMAG[i-seq:i].T

    #======================================
    #extract events on truncated flow
    eventQ = getExtremeEvents(pd.Series(runoff.squeeze(), index=runoffDF.index), method='POT', minDist=5, cutoff=85)    
    assert(A.shape[0] == eventQ.shape[0])    
    #++++++++++++++++++++++++++++++++++++++
    """
    #plot the series [this only works for seq=2]
    fig, axes = plt.subplots(1,1,figsize=(17,6))
    eventQDF = pd.DataFrame(eventQ, index=df.index)
    eventQDF.plot(ax=axes)
    plt.savefig('test_series.png')
    plt.close()
    """    

    #=====================================
    return A, eventQ

def prepareDataSets(args, twsNorm, regionExtents, useDeseason, classificationModel, useImpute, datasource='CSR',
                    precipsource='ERA5', method='POT', filterMethod='XFIT', task='genMetrics', RP_prob=0.02):
    """ Prepare datasets and do logistic regression; Predictors TWS, Umag, and precip
    The goal is to test which variables are more useful for predicting extreme flows
    
    Params
    ------
    twsNorm, tws normalization object
    regionExtents: extent of the region
    useDeseason, use deseasoned dataset
    classificationModel, not used
    useImpute, True to use impute CSR5d data
    datasource, 'CSR', 'ITSG' or 'NOAH'
    precipsource, currently only support 'ERA5' and 'GPM'
    method, 'POT', 'MAF'
    filterMethod, 'XFIT'
    task, can be 'genMetrics' or 'genJRP'
    """
    region = 'north_america'
    gdf = getCatalog(region)
    dfStation = getStationMeta(gdf)
    startDate = '2002/04/01'
    endDate = '2020/12/31'
    daterng = pd.date_range(start=startDate, end=endDate, freq='1D')
    missingdata_threshold = 0.8 # e.g., 0.9 #means tolerate 10% missing data during study period

    #load tws and gradient DA
    twsDA, umagDA = getGRACEDA(datasource=datasource, useImpute=useImpute)
    twsDA_deseason = twsNorm.daCSR_deseasoned.isel(time=twsNorm.goodInd)
    precipDA = getPrecipDA(regionExtents=regionExtents, testDA= twsNorm.daCSR5d, useImpute=useImpute, datasource=datasource, precipsource=precipsource, coarsen=True)
    airtempDA= getAirTempDA(regionExtents=regionExtents, testDA=twsNorm.daCSR5d, useImpute=useImpute, datasource=datasource,coarsen=True)
    print (twsDA.shape, precipDA.shape, airtempDA.shape, twsDA_deseason.shape)
    
    cutoff = 98

    if task == 'genMetrics':
        if useDeseason:
            fid = open(f'metrics_{datasource}_{precipsource}_cutoff{cutoff}_noseason_{method}_{filterMethod}.txt', 'w')
        else:
            fid = open(f'metrics_{datasource}_{precipsource}_cutoff{cutoff}_{method}.txt', 'w')
    
    #iterate through the stations
    outputDict={}
    for ix, row in dfStation.iterrows():        
        stationID = row['grdc_no']
        lat,lon = float(row['lat']), float(row['long'])
        try:
            df = readStationSeries(stationID=stationID, region=region)
            df = df[df.index.year>=2002].copy(deep=True)
            
            #convert to daily data, after this step, NaNs will exist when data is missing!!!
            df = df.reindex(daterng)

            #count number of valid values
            #only process gages have sufficient number of data
            if (1- df.isnull().sum().values/len(daterng))>=missingdata_threshold:

                daQ = xr.DataArray(data=df['Q'].values, dims=['time'], coords={'time':df.index})
                basinmaskDF = getBasinMask(stationID=stationID, region=region, gdf=gdf)
                basinArea = basinmaskDF['area_hys'].values[0] 
                if basinArea > BASINAREA:
                    print ('stationid', stationID, 'drainage area', f"{basinArea:.2f}", 'river ', basinmaskDF['river'].values[0])
                    
                    #asun 0125, normalize Q by drainage area
                    daQ = daQ / basinArea

                    umagAvg   = getBasinData(basinmaskDF, umagDA, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))
                    if useDeseason:
                        twsAvg    = getBasinData(basinmaskDF, twsDA_deseason, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))
                    else:
                        twsAvg    = getBasinData(basinmaskDF, twsDA, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))
                    precipAvg = getBasinData(basinmaskDF, precipDA, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))
                    t2mAvg    = getBasinData(basinmaskDF, airtempDA, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))
                    #always use detrended twsDA to calculate fpi
                    fpiAvg    = getFPIData(basinmaskDF, twsDA, precipDA, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))
                    
                    #aggregate to 5d
                    kwargs = {
                            "method": "rx5d",
                            "aggmethod":'max',
                            "name":'Q'
                    }      

                    daQ = getCSR5dLikeArray(daQ, twsAvg, **kwargs)
                    print (daQ.shape, twsAvg.shape)
                    assert(daQ.shape == twsAvg.shape == umagAvg.shape)
                    inputDict = {'tws': twsAvg.to_dataframe(name='tws'), 
                                'umag': umagAvg.to_dataframe(name='umag'), 
                                'precip': precipAvg.to_dataframe(name='precip'), 
                                'fpi': fpiAvg.to_dataframe(name='fpi'),
                                't2m': t2mAvg.to_dataframe(name='t2m'), 
                                'Q': daQ.to_dataframe(name='Q'),
                    }

                    """
                    #asun: 0114, uncomment to check the filtered signals
                    twsAvgOrg  = getBasinData(basinmaskDF, twsDA, gdf, op='mean', weighting=True).sel(time=slice(startDate,endDate))                
                    _, ax = plt.subplots(1,1,figsize=(12,8))
                    ax.plot(twsAvg, 'r')
                    ax.plot(twsAvgOrg, 'b--')
                    ax2 = ax.twinx()
                    ax2.plot(fpiAvg, 'g')
                    plt.savefig('test_bandpass.png')
                    plt.close()
                    sys.exit()
                    """

                    if task == 'genMetrics':
                        stationDict = getMetrics(twsNorm.consecInd, inputDict, cutoff=cutoff, trimtoITSG=True, method_name=method)
                        outputDict[stationID] = stationDict
                        fid.write(f"stationid {stationID} drainage area {basinArea:.2f}, river {basinmaskDF['river'].values[0]}\n")
                        fid.write(f"auroc {stationDict['auroc']:.3f}, f1 {stationDict['f1']:.3f}, POD {stationDict['POD']:.3f}, FAR {stationDict['FAR']:.3f}, 'CSI' {stationDict['CSI']:.3f} \n")                    
                        fid.write(f"All data {stationDict['causal']}\n")
                        fid.write(f"Events only {stationDict['causal_bin']}")
                        fid.write('\n')
                        fid.flush()
                    elif task == 'genJRP':
                        outputDict[stationID] = getJPR(twsNorm.consecInd, inputDict, trimtoITSG=True, cutoff= 1-np.array(RP_prob), threshMethod=method)

                    #save all results after each basin
                    if useDeseason:
                        if task == 'genMetrics':
                            pkl.dump(outputDict, open(f'metrics_{datasource}_{cutoff}_deseason_{method}_{filterMethod}.pkl', 'wb'))
                        elif task == 'genJRP':
                            pkl.dump(outputDict, open(f'metrics_{datasource}_{cutoff}_deseason_{method}_{filterMethod}_JRP.pkl', 'wb'))

                    else:
                        if task == 'genMetrics':
                            pkl.dump(outputDict, open(f'metrics_{datasource}_{cutoff}_{method}_{filterMethod}.pkl', 'wb'))
                        elif task == 'genJRP':
                            pkl.dump(outputDict, open(f'metrics_{datasource}_{cutoff}_{method}_{filterMethod}_JRP.pkl', 'wb'))

                #genDataset(twsNorm.goodInd, inputDict, seq = args.seq_len)                         
                #apply lags 
                #X,y = formMatrix(twsNorm.goodInd, inputDict, seq=args.seq_len)
                #fitArray(args, X,y,modeltype=classificationModel)
                #lstmMain(args, twsNorm.goodInd, inputDict, cutoff=90)
                
                #use causal links from metrics to generate the feature array
                #genPredictionModel(stationID, datasource, cutoff, inputDict, twsNorm.goodInd, alpha_level=0.05, 
                #        useDeseason=useDeseason, filterMethod=filterMethod, threshMethod=method)
        except Exception as e: 
            raise Exception (e)
    if task == 'genMetrics':
        fid.close()
    
def postprocessResults(datasource, var_names, useDeseason=False, alpha_level=0.05, plotPrecipLags=False, method='POT', filterMethod='XFIT'):
    """Post process Tigramite data
    Params
    -----
    datasource, can be "CSR", "NOAH" or "ITSG"
    var_names, This must be consistent with the one used to generate the dictionary ["FPI", "P", "Q"]
    useDeseason: de-seasoned TWS data
    alpha_level: alpha level used to filter significant links (must be consistent with the one used in analysis)
    plotPrecipLags, if True generate precip precursor (precip vs. Q), otherwise, only plot FPI vs. Q precursor plots
    method, 'POT' or 'MAF'
    filterMethod, 'XFIT'
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    import matplotlib.path as mpath
    import seaborn as sns
    
    def plotSubFigure(fig, ax, columnName, markerColumnName):
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        ax.set_extent([lonl, lonr, latl, latu],  crs=ccrs.PlateCarree())       
        nlags = 12
        
        fig.add_axes(cax)
        if columnName == 'FPI_lag':
            cmap = ListedColormap(sns.color_palette(palette="flare", n_colors=nlags).as_hex())    
        else:
            cmap = ListedColormap(sns.color_palette(palette="Blues", n_colors=nlags).as_hex())    
        cmap.colors[0] = '#BDC3C7'
        
        gdfHUC2.plot(ax=ax, facecolor="None", edgecolor="k", legend=False)
        gdfCol = gpd.GeoDataFrame(dfStation[columnName], geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
        markersizes = dfStation[markerColumnName].to_numpy()
        #get a unique list of markersizes
        uniquevals = np.unique(markersizes)
        uniquevals = uniquevals[1:]
        markersizes = markersizes*10
        
        #markersize: number of lags, vmax: max lag time
        gdfCol.plot(column=columnName, ax=ax, cax=cax, cmap=cmap, marker='o', markersize= markersizes, 
                    legend_kwds={'shrink': 0.5, 'label': f"{columnName} (days)"}, legend=True, 
                    vmin=0, vmax=nlags*5)
        
        #sns.scatterplot(
        #    y="lat", x="long", hue=columnName, edgecolor="k", linewidth=0.1, palette=cmap, data=dfStation, s=markersizes, ax=ax
        #)   
        
        # the size of circles are only approximate (here I use 2x instead of 10x for geopandas)
        
        sizeList = np.array(uniquevals, dtype=int)
        custom_markers = [
            Line2D([0], [0], marker="o", color='w', markerfacecolor='None', label=item, markeredgewidth=0.5, markeredgecolor="k", markersize=item) for ix,item in enumerate(sizeList)
        ]        
        legend2 = ax.legend(handles = custom_markers, loc='lower right', fontsize=13, frameon=True, title="# Lags")                
        ax.add_artist(legend2)
        
        #when using coastlines the tick labels disappear;
        #need to add ticker labels manually here    
        #ax.coastlines(resolution='110m', color='gray')
        #ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
        #ax.set_yticks([-60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        
        #lon_formatter = LongitudeFormatter(zero_direction_label=True)
        #lat_formatter = LatitudeFormatter()
        #ax.xaxis.set_major_formatter(lon_formatter)
        #ax.yaxis.set_major_formatter(lat_formatter)    

    region = 'north_america'
    gdf = getCatalog(region)
    #conus extent
    lonl = -124.7258
    latl = 24.498131
    lonr = -66.949895
    #latu = 49.384358
    latu = 52

    shpfile =  '/work/02248/alexsund/maverick2/csrflow/maps/HUC2.shp'
    gdfHUC2 = gpd.read_file(shpfile)

    figproj = ccrs.PlateCarree()
    # This can be converted into a `proj4` string/dict compatible with GeoPandas   
    if method == 'POT':
        if plotPrecipLags: 
            fig,axes = plt.subplots(3,2, figsize=(16,18), subplot_kw={'projection': figproj})
            
            titles=[['(a)', '(b)'],
                ['(c)', '(d)'], 
                ['(e)', '(f)']
            ]
        else:
            fig,axes = plt.subplots(3,1, figsize=(8,18), subplot_kw={'projection': figproj})
            
            titles=['(a)', '(b)', '(c)']
        cutoffs = [90, 95, 98]
    else:
        if plotPrecipLags:
            fig,axes = plt.subplots(1,2, figsize=(16,8), subplot_kw={'projection': figproj})
        else:
            fig,axes = plt.subplots(1,1, figsize=(12,8), subplot_kw={'projection': figproj})
        
        #note: this cutoff is a dummy value because it's not used in MAF calculations
        cutoffs = [98]

    for ix, cutoff in enumerate(cutoffs):
        if useDeseason:                                        
            outputDict = pkl.load(open(f'metrics_{datasource}_{cutoff}_deseason_{method}_{filterMethod}.pkl', 'rb'))
        else:
            outputDict = pkl.load(open(f'metrics_{datasource}_{cutoff}_{method}_{filterMethod}.pkl', 'rb'))

        #loop through the results
        validStations = list(outputDict.keys())
        dfMetrics = pd.DataFrame(np.zeros((len(validStations), 4)), index=validStations)
        dfMetrics.columns=['FPI_num', 'FPI_lag', 'P_num', 'P_lag']
        for stationID in validStations:
            stationDict = outputDict[stationID]
            sorted_links = stationDict['sorted_link_bin']
            p_matrix = stationDict['p_mat_bin']
            val_matrix = stationDict['val_mat_bin']

            N = len(var_names)
            string = ""
            sig_links = (p_matrix <= alpha_level)
            
            #assuming the last variable is Q
            j=N-1
            links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                        for p in zip(*np.where(sig_links[:, j, :]))}
            # Sort by value        
            sorted_links = sorted(links, key=links.get, reverse=True)            
            # record all lags
            allFPILags=[]
            allPLags = []
            # record the type of lags
            
            for p in sorted_links:
                string += ("\n        (%s % d): pval = %.5f" %
                            (var_names[p[0]], p[1],
                            p_matrix[p[0], j, abs(p[1])]))
                if var_names[p[0]] == 'FPI':
                    allFPILags.append(p[1])
                elif var_names[p[0]] == 'P':
                    allPLags.append(p[1])
                string += " | val = % .3f" % (
                    val_matrix[p[0], j, abs(p[1])])
            
            #print (string)
            if not allFPILags == []:
                allFPILags = np.array(allFPILags)
                print (stationID, ', largest FPI lag', np.min(allFPILags), 'number of FPI lags', len(allFPILags))
                dfMetrics.loc[stationID, 'FPI_lag'] = abs(np.min(allFPILags))
                dfMetrics.loc[stationID, 'FPI_num'] = len(allFPILags)            
            else: 
                dfMetrics.loc[stationID, 'FPI_num'] = 0.5 

            if not allPLags == []:
                allPLags = np.array(allPLags)
                print (stationID, ', largest P lag', np.min(allPLags), 'number of P lags', len(allPLags))
                dfMetrics.loc[stationID, 'P_lag'] = abs(np.min(allPLags))
                dfMetrics.loc[stationID, 'P_num'] = len(allPLags)            
            else:
                dfMetrics.loc[stationID, 'P_num'] = 0.5 #this is needed so P_num won't be empty
        #conver to actual days
        dfMetrics['FPI_lag'] = dfMetrics['FPI_lag'] * 5 
        dfMetrics['P_lag'] = dfMetrics['P_lag'] * 5 
        print (dfMetrics['FPI_lag'].max(), dfMetrics['P_lag'].max())

        dfStation = getStationMeta(gdf)
        selectedStations = list(outputDict.keys())
        dfStation = dfStation[dfStation['grdc_no'].isin(selectedStations)]

        dfStation = dfStation.join(dfMetrics, on='grdc_no')

        if plotPrecipLags:
            #plot precip as precursor
            ax = axes[ix,0] if method == 'POT' else  axes [0]
            plotSubFigure(fig, ax, columnName='FPI_lag', markerColumnName='FPI_num')
            if method == 'POT':
                ax.set_title(f'{titles[ix][0]} FPI, Threshold={cutoff}', fontsize=15)
            else:
                ax.set_title('FPI on MAF', fontsize=15)

            ax = axes[ix,1] if method == 'POT' else axes[1]
            plotSubFigure(fig, ax, columnName='P_lag', markerColumnName='P_num')
            if method == 'POT':
                ax.set_title(f'{titles[ix][1]} P, Threshold={cutoff}', fontsize=15)
            else:
                ax.set_title('Precip on MAF', fontsize=15)

        else:
            ax = axes[ix] if method == 'POT' else axes
            plotSubFigure(fig, ax, columnName='FPI_lag', markerColumnName='FPI_num')
            if method == 'POT':
                ax.set_title(f'{titles[ix]} FPI, Threshold={cutoff}', fontsize=15)
            if datasource == 'CSR':
                #plot precip precursor as separate plot!!!!                        
                fig0,ax0 = plt.subplots(1,1, figsize=(12,8), subplot_kw={'projection': figproj})            
                plotSubFigure(fig0, ax0, columnName='P_lag', markerColumnName='P_num')
                if method == 'POT':
                    ax0.set_title(f'P, Threshold={cutoff}', fontsize=15)                
                fig0.savefig(f"conus_P_{method}.png")
                plt.close(fig0)
    fig.tight_layout(h_pad=0.1, w_pad=0.2)
    fig.savefig(f"conus_{datasource}_{method}.png")
    plt.close(fig)

def genPredictionModel(stationID, datasource, cutoff, inputDict, goodInd, alpha_level=0.05, useDeseason=True, filterMethod='XFIT', threshMethod='POT'):

    if useDeseason:
        outputDict = pkl.load(open(f'metrics_{datasource}_{cutoff}_deseason_{threshMethod}_{filterMethod}.pkl', 'rb'))
    else:
        outputDict = pkl.load(open(f'metrics_{datasource}_{cutoff}_{threshMethod}_{filterMethod}.pkl', 'rb'))
    
    var_names=['P', 'FPI', 'Q']
    #loop through the results
    validStations = list(outputDict.keys())
    dfMetrics = pd.DataFrame(np.zeros((len(validStations), 4)), index=validStations)
    dfMetrics.columns=['FPI_num', 'FPI_lag', 'P_num', 'P_lag']

    stationDict = outputDict[stationID]
    sorted_links = stationDict['sorted_link_bin']
    p_matrix = stationDict['p_mat_bin']
    val_matrix = stationDict['val_mat_bin']

    N = len(var_names)
    j=N-1
    sig_links = (p_matrix <= alpha_level)
    links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                for p in zip(*np.where(sig_links[:, j, :]))}    
    # Sort by value        
    sorted_links = sorted(links, key=links.get, reverse=True)            
    
    allFPILags=[]
    allPLags = []
    addAntecedentQ = False
    # record the type of sig lags            
    for p in sorted_links:
        if var_names[p[0]] == 'FPI':
            allFPILags.append(p[1])
        elif var_names[p[0]] == 'P':
            allPLags.append(p[1])
    # generate the feature matrix
    A, df = assembleAllFeatureMatrix(goodInd, inputDict, trimtoITSG=True)
    
    fea_number = len(allFPILags)+len(allPLags)
    tws,umag, P, FPI, Q = (A[:,0],A[:,1],A[:,2],A[:,3], A[:,4])

    eventQ = getExtremeEvents(pd.Series(Q.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False)
    eventTWS = getExtremeEvents(pd.Series(tws.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False)
    eventUmag = getExtremeEvents(pd.Series(umag.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False)
    eventP = getExtremeEvents(pd.Series(P.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False)
    eventFPI = getExtremeEvents(pd.Series(FPI.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False)
    
    #assemble feature mat
    X = []
    Y = []
    nlags = 12
    print ('FPI predictors', allFPILags)
    print ('P predictors', allPLags)
    for item in allFPILags:
        X.append(FPI[nlags+item-1:item-1])
    
    for item in allPLags:
        X.append(P[nlags+item-1:item-1])
    if addAntecedentQ:
        X.append(Q[nlags-1:-1])    

    X = np.stack(X, axis=0).T
    X = StandardScaler().fit_transform(X)
    Y = eventQ[nlags:]
    
    fitArray(X, Y, modeltype='logistic_regression')
    sys.exit()

def getJPR(goodInd, inputDict, trimtoITSG=True, cutoff=None, threshMethod=""):
    A, df = assembleAllFeatureMatrix(goodInd, inputDict, trimtoITSG=True)    
    tws,umag, P, FPI, Q = (A[:,0],A[:,1],A[:,2],A[:,3], A[:,4])

    #get binary and actual event series
    eventQ, Q_ts     = getExtremeEvents(pd.Series(Q.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False, returnExtremeSeries=True)
    eventTWS, TWS_ts = getExtremeEvents(pd.Series(tws.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False, returnExtremeSeries=True)    
    eventP, P_ts     = getExtremeEvents(pd.Series(P.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False, returnExtremeSeries=True)
    eventFPI, FPI_ts = getExtremeEvents(pd.Series(FPI.squeeze(), index=df.index), method=threshMethod, cutoff=cutoff, transform=False, returnExtremeSeries=True)

    imethod = 1
    if imethod == 1:
        """use copula method
        """
        #count the number of events
        nEvent = len(np.where(eventQ==1)[0])
        #get the number of days in different years to account for partial years
        daysCount = df.groupby(df.index.year).agg('count')
        daysCount = np.sum(daysCount.values/365.0)
        #mean event interarrval in years
        mu = nEvent/daysCount
        #get JRP of FPI and Q
        dfInput = pd.DataFrame({'FPI':FPI_ts.values, 'Q':Q_ts.values})
        JRP_FPI_Q = gridCopula(dfInput, mu, threshold=cutoff)
        
        dfInput = pd.DataFrame({'tws':TWS_ts.values, 'Q':Q_ts.values})
        JRP_tws_Q = gridCopula(dfInput, mu, threshold=cutoff)

        dfInput = pd.DataFrame({'P':P_ts.values, 'Q':Q_ts.values})
        JRP_P_Q = gridCopula(dfInput, mu, threshold=cutoff)

        print ('FPI', 'tws', 'P')
        print (JRP_FPI_Q, JRP_tws_Q,  JRP_P_Q)
        
        #kendall's tau
        tauFPI_Q, pval_FPI_Q = kendalltau(FPI, Q)
        tauP_Q, pval_P_Q = kendalltau(P, Q)
        tauTWS_Q, pval_TWS_Q = kendalltau(tws, Q)

        return {'FPI': (JRP_FPI_Q, tauFPI_Q, pval_FPI_Q),
                'tws': (JRP_tws_Q, tauTWS_Q, pval_TWS_Q),
                'P':   (JRP_P_Q,   tauP_Q,   pval_P_Q),
        }
    else:
        #get JRP of FPI and Q
        dfInput = pd.DataFrame({'FPI': eventFPI,'Q': eventQ}, index=df.index) 
        JRP_FPI_Q  = gridJRP(dfInput)

        dfInput = pd.DataFrame({'tws':eventTWS, 'Q': eventQ}, index=df.index) 
        JRP_tws_Q = gridJRP(dfInput)

        dfInput = pd.DataFrame({'P':eventP,     'Q': eventQ}, index=df.index) 
        JRP_P_Q = gridJRP(dfInput)

        print ('FPI', 'tws', 'P')
        print (JRP_FPI_Q, JRP_tws_Q,  JRP_P_Q)

        return (JRP_FPI_Q, JRP_tws_Q, JRP_P_Q)

def postprocessJRPResults(RT_probs, datasource='CSR', varName='FPI', method='POT', filterMethod='XFIT', cutoff=98, plotvar='JRP'):
    """ plot JRP
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    import matplotlib.path as mpath
    import seaborn as sns
    
    def plotSubFigure(fig, ax, columnName, subtitle, vmin, vmax):
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="3%", pad=0.1, axes_class=plt.Axes)
        ax.set_extent([lonl, lonr, latl, latu],  crs=ccrs.PlateCarree())       
        
        nbins = 5
        fig.add_axes(cax)
        cmap = ListedColormap(sns.color_palette(palette="flare_r", n_colors=nbins).as_hex())    
        
        gdfHUC2.plot(ax=ax, facecolor="None", edgecolor="k", legend=False)

        gdfCol = gpd.GeoDataFrame(dfStation[columnName], geometry=gpd.points_from_xy(dfStation.long, dfStation.lat))
        
        #markersize: number of lags, vmax: max lag time
        gdfCol.plot(column=columnName, ax=ax, cax=cax, cmap=cmap, marker='o', markersize= 80, alpha=0.8,
                    legend_kwds={'shrink': 0.5, 'label': f"JRP {columnName}", 'extend':'max'}, legend=True, 
                    vmax=vmax
        )
        ax.set_title(subtitle, fontsize=15)
        
    outputDict = pkl.load(open(f'metrics_{datasource}_{cutoff}_deseason_{method}_{filterMethod}_JRP.pkl', 'rb'))

    region = 'north_america'
    gdf = getCatalog(region)
    #conus extent
    lonl = -124.7258
    latl = 24.498131
    lonr = -66.949895
    #latu = 49.384358
    latu = 52

    shpfile =  '/work/02248/alexsund/maverick2/csrflow/maps/HUC2.shp'
    gdfHUC2 = gpd.read_file(shpfile)

    figproj = ccrs.PlateCarree()
    # This can be converted into a `proj4` string/dict compatible with GeoPandas   
    fig,axes = plt.subplots(2,1, figsize=(16,18), subplot_kw={'projection': figproj})
    titles=['(a)', '(b)']
    plotCols = [2, 3] #these correspond to return period columns

    validStations = list(outputDict.keys())
    col_names = ['100_yr', '50_yr', '20_yr', '10_yr']
    if plotvar == 'JRP':
        dfMetrics = pd.DataFrame(np.zeros((len(validStations), 4)), index=validStations)
        dfMetrics.columns=col_names

    for ix, item in enumerate(plotCols):
        RP = 1.0/RT_probs[item]

        for stationID in validStations:
            stationDict = outputDict[stationID]            
            if plotvar == 'JRP':
                dfMetrics.loc[stationID]= stationDict[varName][0]         

        dfStation = getStationMeta(gdf)
        selectedStations = list(outputDict.keys())
        dfStation = dfStation[dfStation['grdc_no'].isin(selectedStations)]

        dfStation = dfStation.join(dfMetrics, on='grdc_no')

        plotSubFigure(fig, axes[ix], columnName=col_names[item], subtitle=f'{titles[ix]} Univariate return period {int(RP)} years', vmin=RP,vmax=RP*10)

    fig.tight_layout(h_pad=0.1, w_pad=0.2)
    fig.savefig(f"conus_{datasource}_{method}_JRP.png")
    plt.close(fig)
