# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the robustness checks of
    Working paper 2
    
    The following models are estimated:
        1) FE: cert and date -- loan sales + v + w + z -- Reduced sample
        2) FE: cert and date -- loan sales + v + w + z + internet banking-- Reduced sample
        3) FE: cert and date*MSA -- loan sales split + v + w  -- Full sample
        4) FE: cert and date*MSA -- CCD Distance ~ loan sales + v + w  -- Full sample
        5) Pooled OLS -- loan sales split + v + w + z -- Full sample
        
    Full sample: 2010 -- 2019
    Reduced sample: 2013 -- 2018
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
import copy 

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')
df['intercept'] = 1

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
ls_split = ['ls_gse_num', 'ls_priv_num', 'sec_num']

## X vars control

## X vars
x_vw = ['lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'ln_ta', 'ln_emp',  'ln_num_branch',  'cb']
x_z = ['ln_density', 'ln_pop_area', 'ln_mfi', 'hhi']

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
list_file_names = ['Robustness_checks/Robust_{}.csv'.format(i) for i in range(1,6)]

#------------------------------------------------------------
# Loop over all robustness checks
#------------------------------------------------------------

counter = 0
for file in list_file_names:
    
    # Prelims
    ## Set correct df, msamd, cert
    if  counter in (0,1):
        data = df_red
        msamd = msamd_red
        cert = cert_red
    else:
        data = df_full
        msamd = msamd_full
        cert = cert_full
    
    ## Set y_var
    if counter <= 3:
        y = y_var_robust
    else:
        y = y_var
    
    ## Set x_var
    if counter == 0:
        x = ls_num + x_vw + x_z
    elif counter == 1:
        x = ls_num + x_vw + x_z + perc_broadband
    elif counter == 2:
        x = ls_split + x_vw
    elif counter == 3:
        x = ls_num + x_vw
    elif counter == 4:
        x = ls_num + x_vw + x_z + ['intercept']
    
    ## Set FE and how
    if counter in (0,1):
        FE_cols = FE_cols_vars_t
        h = how_t
    else:
        FE_cols = FE_cols_vars
        h = how
    
    # Run Model
    if counter == 4: # pooled OLS
        results = MultiDimensionalOLS().fit(data[y], data[x], cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    else:
        results = MultiDimensionalOLS().fit(data[y], data[x],\
                                            cov_type = cov_type, cluster_cols = data[cluster_cols_var],\
                                            transform_data = True, FE_cols = data[FE_cols], how = h)
    if counter == 2:
        results_split = copy.deepcopy(results)
        
    # Transform results to pd.DataFrame
    df_results = results.to_dataframe()
    
    # Add count for msamd en cert
    df_results['msamd'] = msamd
    df_results['cert'] = cert
    
    # Save to csv
    df_results.to_csv(file)
    
    # Add one to counter    
    counter += 1

# Test the three different loan sale variables in model 2
# Joint Wald test of b1 = b2 and b2 = b3
R = pd.DataFrame([[1, -1, 0],[0, 1, -1]])
h_beta = R @ pd.DataFrame(results_split.params[:3])
C = results_split.nobs * results_split.cov.iloc[:3,:3]
test_stat = results_split.nobs * h_beta.T @ np.linalg.inv(R @ C @ R.T) @ h_beta

## F test
pval_wald = stats.chi2.sf(test_stat, R.shape[0])