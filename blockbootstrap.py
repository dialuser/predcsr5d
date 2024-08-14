#author: Alex Sun
#moving block bootstrap 
#date: 12/24/2023
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import pandas as pd
import matplotlib.pyplot as plt
from arch.bootstrap import MovingBlockBootstrap
import numpy as np

class BlockBS():
    def __init__(self, dictDF,nRZ=100) -> None:
        """
        dictDF, dictionary of dataframes
        """
        dfTWS = dictDF['TWS']
        dfQ   = dictDF['Q']
        dfP   = dictDF['P']
        resTWS= self.deseason(dfTWS, 'TWS')
        resQ  = self.deseason(dfQ, 'Q')
        resP  = self.deseason(dfP, 'P')

        self.bsResult = {}

        #now do block BS
        self.getBS2(resTWS=resTWS, resQ=resQ, resP=resP, nRZ=nRZ, isPlotting=False)

    def deseason(self,ts, varname, plotting=False):
        res=STL(ts, period=72, seasonal=73, robust=True, trend_deg=0, low_pass_deg=0 ).fit()
        
        if plotting:
            plt.figure()
            #plt.plot(res.observed, 'gray')
            #plt.plot(res.seasonal, 'r:', linewidth=1.5)
            plt.plot(res.trend, 'r:', linewidth=1.5)
            print (varname)
            print (res.seasonal)
            if not varname is None:
                plt.savefig(f'test_deseason{varname}.png')
            else:
                plt.savefig(f'test_deseason.png')

            plt.close()    

        return res
    
    def getBS(self, resTWS, resQ, resP, blocksize=73, nRZ=100, isPlotting=False):
        bs = MovingBlockBootstrap(blocksize, resTWS.resid, Q=resQ.resid, P=resP.resid, seed=12242023)

        counter=0
        for data in bs.bootstrap(nRZ):
            bs_x = data[0][0]
            bs_y = data[1]['Q']
            bs_z = data[1]['P']   

            bs_x = bs_x.values + resTWS.seasonal.values+resTWS.trend.values
            bs_y = bs_y.values + resQ.seasonal.values+resQ.trend.values
            bs_z = bs_z.values + resP.seasonal.values+resP.trend.values

            if isPlotting:
                print (bs_x)
                print (bs_y)
                print (bs_z)
                fig, axes = plt.subplots(3,1)
                axes[0].plot(bs_x)
                axes[1].plot(bs_y)
                axes[2].plot(bs_z)
                plt.savefig(f'bs_test_{counter}.png')
                plt.close()    
    
            self.bsResult[counter] = np.c_[bs_x, bs_y, bs_z]
            counter+=1

    def getBS2(self, resTWS, resQ, resP, blocksize=73, nRZ=100, isPlotting=False):
        bs = MovingBlockBootstrap(blocksize, resTWS.resid, seed=12242023)
        bsQ = MovingBlockBootstrap(blocksize, resQ.resid,  seed=12222023)
        bsP = MovingBlockBootstrap(blocksize, resP.resid,  seed=12212023)

        counter=0
        for data,dataQ,dataP in zip(bs.bootstrap(nRZ), bsQ.bootstrap(nRZ), bsP.bootstrap(nRZ)):
            bs_x = data[0][0]
            bs_y = dataQ[0][0]
            bs_z = dataP[0][0]   

            bs_x = bs_x.values + resTWS.seasonal.values+resTWS.trend.values
            bs_y = bs_y.values + resQ.seasonal.values+resQ.trend.values
            bs_z = bs_z.values + resP.seasonal.values+resP.trend.values

            if isPlotting:
                print (bs_x)
                print (bs_y)
                print (bs_z)
                fig, axes = plt.subplots(3,1)
                axes[0].plot(bs_x)
                axes[1].plot(bs_y)
                axes[2].plot(bs_z)
                plt.savefig(f'bs_test_{counter}.png')
                plt.close()    
    
            self.bsResult[counter] = np.c_[bs_x, bs_y, bs_z]
            counter+=1
