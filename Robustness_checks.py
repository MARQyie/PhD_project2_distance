# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the robustness checks of
    Working paper 2
    
    The following models are estimated:
        For Distance CDD:
        1) Pooled OLS -- loan sales + controls -- Full sample
        2) Pooled OLS -- loan sales + controls -- Reduced sample
        3) Pooled OLS -- internet + controls -- Reduced sample
        4) Pooled OLS -- loan sales + internet + controls -- Reduced sample
    
        5) FE: cert and date*msamd -- loan sales + controls -- Full sample
        6) FE: cert and date*msamd -- loan sales + controls -- Reduced sample
    
        7) FE: cert and date -- loan sales + controls -- Full sample
        8) FE: cert and date -- loan sales + controls -- Reduced sample
        9) FE: cert and date -- internet + controls -- Reduced sample
        10) FE: cert and date -- loan sales + internet + controls -- Reduced sample
    
        For loan sales, value
        11) Pooled OLS -- loan sales + controls -- Full sample
        12) Pooled OLS -- loan sales + controls -- Reduced sample
        13) Pooled OLS -- loan sales + internet + controls -- Reduced sample
    
        14) FE: cert and date*msamd -- loan sales + controls -- Full sample
        15) FE: cert and date*msamd -- loan sales + controls -- Reduced sample
    
        16) FE: cert and date -- loan sales + controls -- Full sample
        17) FE: cert and date -- loan sales + controls -- Reduced sample
        18) FE: cert and date -- loan sales + internet + controls -- Reduced sample
        
        For loan sales split
        19) Pooled OLS -- loan sales + controls -- Full sample
        20) Pooled OLS -- loan sales + controls -- Reduced sample
        21) Pooled OLS -- loan sales + internet + controls -- Reduced sample
    
        22) FE: cert and date*msamd -- loan sales + controls -- Full sample
        23) FE: cert and date*msamd -- loan sales + controls -- Reduced sample
    
        24) FE: cert and date -- loan sales + controls -- Full sample
        25) FE: cert and date -- loan sales + controls -- Reduced sample
        26) FE: cert and date -- loan sales + internet + controls -- Reduced sample
        
        For Internet banking
        27) Pooled OLS -- internet + controls -- Reduced sample
        28) Pooled OLS -- loan sales + internet + controls -- Reduced sample
       
        29) FE: cert and date -- internet + controls -- Reduced sample
        30) FE: cert and date -- loan sales + internet + controls -- Reduced sample
        
    Full sample: 2010 -- 2017
    Reduced sample: 2013 -- 2017
    '''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization
from sklearn.utils import resample
from Code_docs.Help_functions.MD_panel_estimation import MultiDimensionalOLS, Transformation, Metrics 

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')

#------------------------------------------------------------
# Make full and restricted df and set variables
#------------------------------------------------------------

# Make dfs
df_full = df.copy()
df_red = df.dropna()

# Set variables
## Y vars
y_var = 'log_min_distance'
y_var_robust = 'log_min_distance_cdd'

## X vars interest
ls_num = ['ls_num']
perc_broadband = ['perc_broadband']
ls_val = ['ls_val']
ls_split = ['ls_gse_num', 'ls_priv_num', 'sec_num']
perc_noint = ['perc_noint']

## X vars control

## X vars
x_var = ['lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'ln_ta', 'ln_emp',  'ln_num_branch',  'cb', 'ln_density','ln_pop_area', 'ln_mfi',  'hhi']

## Other vars
msat_invar = ['ln_density', 'ln_pop_area', 'ln_mfi', 'hhi'] # Used to remove msat-fixed vars. Example: [x for x in x_int_var if x not in msat_invar]


# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['msamd','date','cert']
how = 'msamd x date, cert'
FE_cols_vars_t = ['date','cert']
how_t = 'date, cert'

msamd_full = df_full.msamd.nunique()
msamd_red = df_red.msamd.nunique()
cert_full = df_full.cert.nunique()
cert_red = df_red.cert.nunique()

# Set File names (in list, for numbers see top of script)
list_file_names = ['Robustness_checks/Robust_{}.csv'.format(i) for i in range(1,31)]

#------------------------------------------------------------
# Loop over all robustness checks
#------------------------------------------------------------

counter = 0
for file in list_file_names:
    
    # Prelims
    ## Set correct df, msamd, cert
    if not counter in (0,4,6,10,13,15,18,21,24):
        data = df_red
        msamd = msamd_red
        cert = cert_red
    else:
        data = df_full
        msamd = msamd_full
        cert = cert_full
    
    ## Set y_var
    if counter <= 9:
        y = y_var_robust
    else:
        y = y_var
    
    ## Set x_var
    if counter in (0,1,4,5,6,7):
        x = ls_num + x_var
    elif counter in (2,8):
        x = perc_broadband + x_var
    elif counter in (3,9):
        x = ls_num + perc_broadband + x_var
    elif counter in (10,11,13,14,15,16):
        x = ls_val + x_var
    elif counter in (12,17):
        x =ls_val + perc_broadband + x_var
    elif counter in (18,19,21,22,23,24):
        x = ls_split + x_var
    elif counter in (20,25):
        x = ls_split + perc_broadband + x_var
    elif counter in (26,28):
        x = perc_noint + x_var
    elif counter in (27,29):
        x = ls_num + perc_noint + x_var
    
    ## Set FE and how
    if counter in (6,7,8,9,15,16,17,23,24,25,28,29):
        FE_cols = FE_cols_vars_t
        h = how_t
    else:
        FE_cols = FE_cols_vars
        h = how
    
    # Run Model
    if counter in (0,1,2,3,10,11,12,18,19,20,26,27): # pooled OLS
        results = MultiDimensionalOLS().fit(data[y], data[x], cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    elif counter in (4,5,13,14,21,22): # MSA-time and lender
        results = MultiDimensionalOLS().fit(data[y], data[[var for var in x if x not in msat_invar]],\
                                            cov_type = cov_type, cluster_cols = data[cluster_cols_var],\
                                            transform_data = True, FE_cols = data[FE_cols], how = h)
    else:
        results = MultiDimensionalOLS().fit(data[y], data[x],\
                                            cov_type = cov_type, cluster_cols = data[cluster_cols_var],\
                                            transform_data = True, FE_cols = data[FE_cols], how = h)
    
    # Transform results to pd.DataFrame
    df_results = results.to_dataframe()
    
    # Add count for msamd en cert
    df_results['msamd'] = msamd
    df_results['cert'] = cert
    
    # Save to csv
    df_results.to_csv(file)
    
    # Add one to counter    
    counter += 1

#------------------------------------------------------------
# 1) Pooled OLS -- loan sales + controls -- Full sample
#------------------------------------------------------------
