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
# Make df for the years 2018-2019 (automated lending and credit scoring)

# Make df
df1819 = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')

# Remove date < 2018
df1819 = df1819[df1819.date.astype(int) >= 2018]

# Reset index
df1819.reset_index(drop = True, inplace = True)

# Check dataset
#desc_df1819 = df1819.describe().T # All good

#------------------------------------------------------------
# Make df for the years 2013-2019 (Internet subscriptions)

# Make df
df1319 = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')

# Remove date < 2013
df1319 = df1319[df1819.date.astype(int) >= 2013]

# Reset index
df1319.reset_index(drop = True, inplace = True)

# Merge with df_int
df1319 = df1319.merge(df_int, how = 'inner', left_on = ['fips','date'], right_on = ['geo_id','date'])

# Check dataset
#desc_df1319 = df1319.describe().T # All good

# Transform df1319 according to the FE structure (eq. (1))
df1319 = df1319.assign(msat = df1319.date.astype(str) + df1319.msamd.astype(str))

df1319_grouped_msat = df1319.groupby('msat')[columns[4:]].transform('mean')
df1319_grouped_fips = df1319.groupby('fips')[columns[4:]].transform('mean')
df1319_grouped_lender = df1319.groupby('cert')[columns[4:]].transform('mean')
df1319_mean = df1319[columns[4:]].mean()

df1319_trans = df1319[columns[4:]] - df1319_grouped_msat - df1319_grouped_fips - df1319_grouped_lender + 2 * df1319_mean 

#------------------------------------------------------------
# Drop dask dataframe from memory
del dd_main

#------------------------------------------------------------
# Set variables, parameters and file names
#------------------------------------------------------------

# Set variables
## Dependent variable
y = 'log_min_distance'

## Regressors 1819
x1819 = [['ls'],['automated'],['csm'],['ls','automated'],['ls','csm'],['automated','csm'],['ls','automated','csm']]

## Regressors 1319
x1319 = [['ls'],['dailup'],['broadband'],['noint'],['ls','dailup'],['ls','broadband'],['ls','noint'],['dailup','broadband','noint'],['ls','dailup','broadband','noint']]

## Regressors all models
x_rest = ['lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
    
# Set regression parameters
##1819
cov_type1819 = 'clustered'
cluster_cols_var1819 = 'msamd'
FE_cols_vars1819 = ['fips','cert']
how1819 = 'fips, cert'

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
    results_1819 = MultiDimensionalOLS().fit(df1819[y], df1819[x + x_rest],\
                        cov_type = cov_type1819, cluster_cols = df1819[cluster_cols_var1819],\
                        transform_data = True, FE_cols = df1819[FE_cols_vars1819], how = how1819)
    
    ## Transform to pandas df
    df_results_1819 = results_1819.to_dataframe()
    
    ## Save to excel and csv
    df_results_1819.to_excel(file_1819.format(i,'xlsx'))
    df_results_1819.to_csv(file_1819.format(i,'csv'))

# Run model 1819
for x,i in zip(x1319,range(len(x1319))):
    results_1319 = MultiDimensionalOLS().fit(df1319[y], df1319[x + x_rest],\
                        cov_type = cov_type1319, cluster_cols = df1319[cluster_cols_var1319])
    
    ## Transform to pandas df
    df_results_1319 = results_1319.to_dataframe()
    
    ## Save to excel and csv
    df_results_1319.to_excel(file_1319.format(i,'xlsx'))
    df_results_1319.to_csv(file_1319.format(i,'csv'))