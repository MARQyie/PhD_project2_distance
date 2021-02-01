# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    the benchmark model in Working paper 2
    '''

#------------------------------------------------------------
# Import necessary packages
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

columns = ['date','fips','msamd','cert','log_min_distance','ls',\
           'lti','ln_loanamout','ln_appincome','subprime',\
           'lien','owner','preapp', 'coapp','loan_originated',\
            'ln_ta','ln_emp','ln_num_branch','cb']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Interact msamd and date
dd_main = dd_main.assign(msat = dd_main.date.astype(str) + dd_main.msamd.astype(str))
        
# Transform to pandas df
## FULL
df = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)

df_grouped_msat = df.groupby('msat')[columns[4:]].transform('mean')
df_grouped_fips = df.groupby('fips')[columns[4:]].transform('mean')
df_grouped_lender = df.groupby('cert')[columns[4:]].transform('mean')
df_mean = df[columns[4:]].mean()

df_trans = df[columns[4:]] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

### Add MSA
df_trans['msamd'] = df.msamd

del df_grouped_msat, df_grouped_fips, df_grouped_lender, df

## After 2009
df_1019 = dd_main[(dd_main.loan_originated == 1) & (dd_main.date.astype(int) >= 2010)].compute(scheduler = 'threads')
df_1019.reset_index(drop = True, inplace = True)

df_grouped_msat = df_1019.groupby('msat')[columns[4:]].transform('mean')
df_grouped_fips = df_1019.groupby('fips')[columns[4:]].transform('mean')
df_grouped_lender = df_1019.groupby('cert')[columns[4:]].transform('mean')
df_mean = df_1019[columns[4:]].mean()

df_trans_1019 = df_1019[columns[4:]] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

### Add MSA
df_trans_1019['msamd'] = df_1019.msamd


del dd_main, df_grouped_msat, df_grouped_fips, df_grouped_lender, df_1019

#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y = 'log_min_distance'
x = ['ls','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
#FE_cols_vars = ['msat','fips','cert']
#how = 'msat, fips, cert'

# Set File names
file = 'Results/Distance_results_benchmark.{}'
file_1019 = 'Results/Distance_results_benchmark_1019.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------
    
# Run
## Full
'''
results_local = MultiDimensionalOLS().fit(df[y], df[x],\
                    cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                    transform_data = True, FE_cols = df[FE_cols_vars], how = how) '''
results = MultiDimensionalOLS().fit(df_trans[y], df_trans[x],\
                    cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])

### Transform to pandas df
df_results = results.to_dataframe()

### Save to excel and csv
df_results.to_excel(file.format('xlsx'))
df_results.to_csv(file.format('csv'))

## > 2009
results = MultiDimensionalOLS().fit(df_trans_1019[y], df_trans_1019[x],\
                    cov_type = cov_type, cluster_cols = df_trans_1019[cluster_cols_var])

### Transform to pandas df
df_results = results.to_dataframe()

### Save to excel and csv
df_results.to_excel(file_1019.format('xlsx'))
df_results.to_csv(file_1019.format('csv'))