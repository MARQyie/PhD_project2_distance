# Robustness checks: technological innovation 
''' This script perfomrs the robustness checks of technological innovation 
    for the paper ''
 '''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Data manipulation packages
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

# Parallel functionality pandas via Dask
import dask.dataframe as dd 
import dask 

# Other parallel packages
import multiprocessing as mp 
from joblib import Parallel, delayed 
num_cores = mp.cpu_count()

# Estimation package
from Code_docs.Help_functions.MD_panel_estimation import MultiDimensionalOLS, Transformation, Metrics 

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

#------------------------------------------------------------
# Load main data as a dask dataframe
columns = ['date','fips','msamd','cert','log_min_distance','ls',\
           'lti','ln_loanamout','ln_appincome','subprime',\
           'lien','owner','preapp', 'coapp','loan_originated',\
            'ln_ta','ln_emp','ln_num_branch','cb','automated','csm']
    
dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

#------------------------------------------------------------    
# Open and format data on internet usage
## Set path names and columns names
path_int = r'D:/RUG/Data/US_Census_Bureau/Internet_subscriptions/County/'
file_int = r'ACSDT1Y{}.B28002_data_with_overlays_2021-03-18T113757.csv'
usecols = ['GEO_ID', 'NAME', 'B28002_001E', 'B28002_002E', 'B28002_003E', 'B28002_004E',\
           'B28002_007E', 'B28002_010E', 'B28002_013E', 'B28002_016E',\
           'B28002_020E', 'B28002_021E']
usecols_from2016 = ['GEO_ID', 'NAME', 'B28002_001E', 'B28002_002E','B28002_003E', 'B28002_004E', 'B28002_012E', 'B28002_013E']

## Load the data
list_df_int = []
for year in range(2013, 2019+1):
    if year < 2016:
        df_int_load =  pd.read_csv(path_int + file_int.format(year), skiprows = [1], usecols = usecols)
        df_int_load['broadband_load'] = df_int_load[['B28002_007E', 'B28002_010E', 'B28002_013E', 'B28002_016E']].sum(axis = 1)
        df_int_load.drop(columns = ['B28002_007E', 'B28002_010E', 'B28002_013E', 'B28002_016E'], inplace = True)
    else:
        df_int_load =  pd.read_csv(path_int + file_int.format(year), skiprows = [1], usecols = usecols_from2016)
    df_int_load['date'] = year
    list_df_int.append(df_int_load)

df_int = pd.concat(list_df_int)

## drop columns, make columns and rename the rest
df_int['broadband'] = df_int[['B28002_004E', 'broadband_load']].sum(axis = 1)
df_int['int_nosub'] = df_int[['B28002_020E', 'B28002_012E']].sum(axis = 1)
df_int['noint']  = df_int[['B28002_021E', 'B28002_013E']].sum(axis = 1)

df_int.drop(columns = ['B28002_004E', 'broadband_load', 'B28002_020E', 'B28002_012E', 'B28002_021E', 'B28002_013E'], inplace = True)

names = ['geo_id', 'name', 'int_total', 'int_sub', 'dailup','date', 'broadband', 'int_nosub', 'noint']
df_int.columns = names

## Change Geo-id
df_int['geo_id'] = df_int.geo_id.str[-5:].astype(int)

#------------------------------------------------------------
# Load Instruments to dataframes
ls_other = pd.read_csv('Data/df_ls_other.csv')
             
## Add instruments to columns
columns.append('ls_other')

#------------------------------------------------------------
# Make df for the years 2018-2019 (automated lending and credit scoring)

# Make df
df1819 = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')

# Remove date < 2018
df1819 = df1819[df1819.date.astype(int) >= 2018]

# Reset index
df1819.reset_index(drop = True, inplace = True)

# Check dataset
#desc_df1819 = df1819.describe().T # All good

# Merge dataframes            
df1819 = df1819.merge(ls_other[['fips','date','cert','ls_other']], how = 'inner', on = ['fips','date','cert'])

# Transform data
df1819_grouped_fips = df1819.groupby('fips')[columns[4:]].transform('mean')
df1819_grouped_lender = df1819.groupby('cert')[columns[4:]].transform('mean')
df1819_mean = df1819[columns[4:]].mean()

df1819_trans = df1819[columns[4:]] - df1819_grouped_fips - df1819_grouped_lender + df1819_mean 

df1819_trans['msamd'] = df1819.msamd
df1819_trans.dropna(subset = columns[4:], inplace = True)
#------------------------------------------------------------
# Make df for the years 2013-2019 (Internet subscriptions)

# Make df
df1319 = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')

# Remove date < 2013
df1319 = df1319[df1319.date.astype(int) >= 2013]

# Reset index
df1319.reset_index(drop = True, inplace = True)

# Merge with df_int
df1319 = df1319.merge(df_int, how = 'inner', left_on = ['fips','date'], right_on = ['geo_id','date'])

# Check dataset
#desc_df1319 = df1319.describe().T # All good

# Merge dataframes            
df1319 = df1319.merge(ls_other[['fips','date','cert','ls_other']], how = 'inner', on = ['fips','date','cert'])

# Transform df1319 according to the FE structure (eq. (1))
df1319 = df1319.assign(msat = df1319.date.astype(str) + df1319.msamd.astype(str))

df1319_grouped_msat = df1319.groupby('msat')[columns[4:] + ['dailup', 'broadband', 'noint']].transform('mean')
df1319_grouped_fips = df1319.groupby('fips')[columns[4:] + ['dailup', 'broadband', 'noint']].transform('mean')
df1319_grouped_lender = df1319.groupby('cert')[columns[4:] + ['dailup', 'broadband', 'noint']].transform('mean')
df1319_mean = df1319[columns[4:] + ['dailup', 'broadband', 'noint']].mean()

df1319_trans = df1319[columns[4:] + ['dailup', 'broadband', 'noint']] - df1319_grouped_msat - df1319_grouped_fips - df1319_grouped_lender + 2 * df1319_mean 

df1319_trans['msamd'] = df1319.msamd
df1319_trans.dropna(subset = columns[4:], inplace = True)
#------------------------------------------------------------
# Drop dask dataframe from memory
del dd_main, df1319, df1819

#------------------------------------------------------------
# Set variables, parameters and file names
#------------------------------------------------------------

# Set variables
## Dependent variable
y = 'log_min_distance'

## Regressors 1819
x1819 = [['ls'],['automated'],['csm'],['ls','automated'],['ls','csm'],['automated','csm'],['ls','automated','csm']]

## Regressors 1319
x1319 = [['ls'],['dailup'],['broadband'],['noint'],['ls','dailup'],['ls','broadband'],['ls','noint'],['dailup','broadband'],['ls','dailup','broadband']]

## Regressors all models
x_rest = ['lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
    
# Set regression parameters
##1819
cov_type1819 = 'clustered'
cluster_cols_var1819 = 'msamd'

##1319
cov_type1319 = 'clustered'
cluster_cols_var1319 = 'msamd'

# File names
## 1819
file_1819 = 'Robustness_checks/Benchmark_techinno_1819_{}.{}'

## 1319
file_1319 = 'Robustness_checks/Benchmark_techinno_1319_{}.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------

# Run model 1819
for x,i in zip(x1819,range(len(x1819))):
    
    # Compute first stage results
    if 'ls' in x:
        results_1819_fs = MultiDimensionalOLS().fit(df1819_trans[x[0]], df1819_trans[['ls_other'] + x[1:] + x_rest],\
                        cov_type = cov_type1819, cluster_cols = df1819_trans[cluster_cols_var1819])
    
        ## Transform to pandas dataframe and save as csv
        df_results_1819_fs = results_1819_fs.to_dataframe()
        df_results_1819_fs['fixed effects'] = 'FIPS \& Lender'
        partial_filename = 'fs_{}'.format(i)
        df_results_1819_fs.to_csv(file_1819.format(partial_filename,'csv'))
    
        ## Append fitted values and residuals to dataframe
        df1819_trans['ls_hat'] = results_1819_fs.fittedvalues
        df1819_trans['ls_resid'] = results_1819_fs.resid
        
        del results_1819_fs
        
    # Compute Second Stage results
    if 'ls' in x:
        results_1819 = MultiDimensionalOLS().fit(df1819_trans[y], df1819_trans[['ls_hat'] + x[1:] + x_rest],\
                        cov_type = cov_type1819, cluster_cols = df1819_trans[cluster_cols_var1819])
    
        ## Transform to dataframe
        df_results_1819 = results_1819.to_dataframe()
        df_results_1819['fixed effects'] = 'FIPS \& Lender'
    
        ## Perform Hausman test and append to results dataframe
        results_1819_hw = MultiDimensionalOLS().fit(df1819_trans[y], df1819_trans[x + x_rest + ['ls_resid']],\
                    cov_type = cov_type1819, cluster_cols = df1819_trans[cluster_cols_var1819])
    
        df_results_1819_hw = results_1819_hw.to_dataframe() 
        del results_1819_hw

        ### Ad hw-stat to df_results_ss
        df_results_1819['hw_pval'] = df_results_1819_hw.loc['ls_resid','p']
    
        ## Save to csv
        df_results_1819.to_csv(file_1819.format(i,'csv'))
        
        del results_1819
    else:
        results_1819 = MultiDimensionalOLS().fit(df1819_trans[y], df1819_trans[x + x_rest],\
                        cov_type = cov_type1819, cluster_cols = df1819_trans[cluster_cols_var1819])
    
        ## Transform to pandas df
        df_results_1819 = results_1819.to_dataframe()
        df_results_1819['fixed effects'] = 'FIPS \& Lender'
    
        ## Save to excel and csv
        df_results_1819.to_csv(file_1819.format(i,'csv'))
        del results_1819

# Run model 1319
for x,i in zip(x1319,range(len(x1319))):
    
    # Compute first stage results
    if 'ls' in x:
        results_1319_fs = MultiDimensionalOLS().fit(df1319_trans[x[0]], df1319_trans[['ls_other'] + x[1:] + x_rest],\
                        cov_type = cov_type1319, cluster_cols = df1319_trans[cluster_cols_var1319])
    
        ## Transform to pandas dataframe and save as csv
        df_results_1319_fs = results_1319_fs.to_dataframe()
        df_results_1319_fs['fixed effects'] = 'MSA-year, FIPS \& Lender'
        partial_filename = 'fs_{}'.format(i)
        df_results_1319_fs.to_csv(file_1319.format(partial_filename,'csv'))
    
        ## Append fitted values and residuals to dataframe
        df1319_trans['ls_hat'] = results_1319_fs.fittedvalues
        df1319_trans['ls_resid'] = results_1319_fs.resid
        
        del results_1319_fs
    # Compute Second Stage results
    if 'ls' in x:
        results_1319 = MultiDimensionalOLS().fit(df1319_trans[y], df1319_trans[['ls_hat'] + x[1:] + x_rest],\
                        cov_type = cov_type1319, cluster_cols = df1319_trans[cluster_cols_var1319])
    
        ## Transform to dataframe
        df_results_1319 = results_1319.to_dataframe()
        df_results_1319['fixed effects'] = 'MSA-year, FIPS \& Lender'
    
        ## Perform Hausman test and append to results dataframe
        results_1319_hw = MultiDimensionalOLS().fit(df1319_trans[y], df1319_trans[x + x_rest + ['ls_resid']],\
                    cov_type = cov_type1319, cluster_cols = df1319_trans[cluster_cols_var1319])
    
        df_results_1319_hw = results_1319_hw.to_dataframe() 
        del results_1319_hw

        ### Ad hw-stat to df_results_ss
        df_results_1319['hw_pval'] = df_results_1319_hw.loc['ls_resid','p']
    
        ## Save to csv
        df_results_1319.to_csv(file_1319.format(i,'csv'))
        
        del results_1319
    else:
        results_1319 = MultiDimensionalOLS().fit(df1319_trans[y], df1319_trans[x + x_rest],\
                        cov_type = cov_type1319, cluster_cols = df1319_trans[cluster_cols_var1319])
    
        ## Transform to pandas df
        df_results_1319 = results_1319.to_dataframe()
        df_results_1319['fixed effects'] = 'FIPS \& Lender'
    
        ## Save to excel and csv
        df_results_1319.to_csv(file_1319.format(i,'csv'))
        del results_1319