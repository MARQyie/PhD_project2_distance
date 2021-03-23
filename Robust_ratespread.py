# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    the rate spread model in Working paper 2
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

columns = ['date','fips','msamd','cert','rate_spread','log_min_distance','local',\
           'ls','ltv','lti','ln_loanamout','ln_appincome','int_only','balloon',\
           'lien','mat','hoepa','owner','preapp', 'coapp','loan_originated',\
           'loan_term','log_min_distance_cdd','ls_ever']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Add intercept and interaction term
## Subset data
dd_main = dd_main[(dd_main.date.astype(int) >= 2018) & (dd_main.loan_originated == 1)]

## Add remote dummy
dd_main['remote'] = (dd_main['local'] - 1).abs()

## Interaction terms
dd_main = dd_main.assign(log_min_distance_cdd_ls = dd_main['log_min_distance_cdd'] * dd_main['ls'])
dd_main = dd_main.assign(log_min_distance_ls = dd_main['log_min_distance'] * dd_main['ls'])
dd_main = dd_main.assign(remote_ls_ever = dd_main['remote'] * dd_main['ls_ever'])

## Remove outliers
dd_main = dd_main[(dd_main.rate_spread > -150) & (dd_main.rate_spread < 10) &\
                  (dd_main.loan_term < 2400) & (dd_main.ltv < 87)]

# Transform to pandas df
df = dd_main.compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)

## Describe df
df_desc = df.describe().T
''' NOTE very little variation in HOEPA'''


#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y = 'rate_spread'
x1 = ['ls','log_min_distance','log_min_distance_ls','lti','ltv','ln_loanamout',\
     'ln_appincome','int_only','balloon','mat','lien', 'coapp']
x2 = ['ls','log_min_distance_cdd','log_min_distance_cdd_ls','lti','ltv','ln_loanamout',\
     'ln_appincome','int_only','balloon','mat','lien', 'coapp']
x3 = ['ls_ever','remote','remote_ls_ever','lti','ltv','ln_loanamout',\
     'ln_appincome','int_only','balloon','mat','lien', 'coapp']

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['fips','cert']
how = 'fips, cert'

# Set File names
file1 = 'Robustness_checks/Ratespread_robust_distance.{}'
file2 = 'Robustness_checks/Ratespread_robust_cdd.{}'
file3 = 'Robustness_checks/Ratespread_robust_lsever.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------
    
# Run
for x,file in zip([x1,x2,x3],[file1,file2,file3]):
    results_local = MultiDimensionalOLS().fit(df[y], df[x],\
                        cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                        transform_data = True, FE_cols = df[FE_cols_vars], how = how)
    
    ### Transform to pandas df
    df_results_local = results_local.to_dataframe()
    
    ### Save to excel and csv
    df_results_local.to_excel(file.format('xlsx'))
    df_results_local.to_csv(file.format('csv'))
    
    del results_local
