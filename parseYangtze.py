#author: alex sun
#date: 0906
#====================================================================
import pandas as pd
import calendar
import datetime
import numpy as np
import pickle as pkl
import os

gage_names = ['cuntan', 'wulong', 'yichang', 'zhicheng', 'shashi', 'jianli', 'lianhuatang', 'luoshan', 'hankou', 'huangshi', 'matouzhen', 'jiujiang', 'sanxia']

def parse_yangtze():
    rawfile = '/home/suna/work/predcsr5d/data/Yangtze-2000-2023.xlsx'
    #stations to extract

    gage_id = [60105400, 60803000, 60107300, 60107400, 60108300, 60110500, 60111200, 60111300, 60112200, 60112900, 60113205, 60113400, 
               60106980]
    alldates = []
    flowRates = []
    counter = 0
    with pd.ExcelFile(rawfile) as xls:
        for iyear in range(2000, 2022):
            for imon in range(1, 13):
                monthdays = calendar.monthrange(iyear,imon)[1]
                for iday in range(1, monthdays+1):
                    datestr = f"{iyear}-{imon:02d}-{iday:02d}"
                    df = pd.read_excel(xls, sheet_name=datestr, usecols=['STCD', 'Q'])
                    df = df.fillna(-999.000)
                    df = df.loc[df['STCD'].isin(gage_id)]
                    alldates.append(datestr)
                    flowRates.append(np.transpose(df[['Q']].values))
                    
                    counter+=1
    bigDF = pd.DataFrame(data=np.concatenate(flowRates), index=pd.to_datetime(alldates), columns=gage_names)
    #save it
    pkl.dump(bigDF, open('data/yangtzeQ.pkl', 'wb'))

def read_file_into_buffer(file_path):   
   with open(file_path, 'rb') as file:      
      file_contents = file.read().decode(errors='replace')
   return file_contents

def checkData():
    df = pkl.load(open('data/yangtzeQ.pkl', 'rb'))
    df = df[df.index.year>=2002].copy(deep=True)

    #now print out in the format of GRDC
    #for example
    #2004-01-01;--:--;  10000.000
    grdcdict = {'yichang': '2181600', 'hankou': '2181800'}
    for station in grdcdict.keys():
        file_content = read_file_into_buffer(os.path.join('/home/suna/work/grace/data/grdc/asia/original', f'{grdcdict[station]}_Q_Day.Cmd.txt'))
        adf = df[station]
        
        adf.index = adf.index.strftime('%Y-%m-%d')
        with open(f'data/{grdcdict[station]}_Q_Day.Cmd.txt', 'w') as fid:
            fid.write(file_content)
            fid.write('\n')
            adf = adf.astype('float')
            for indx, value in adf.items():
                fid.write(f'{indx};--:--; {value:8.3f}\n')
                

    #
if __name__ == '__main__':
    itask = 2
    if itask == 1:
        #parse the Excel file and save as pkl file
        parse_yangtze()
    elif itask == 2:
        #generate GRDB format
        checkData()
