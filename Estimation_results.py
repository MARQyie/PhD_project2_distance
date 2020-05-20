# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    Working paper 2
    
    The following models are estimated:
        1) Pooled OLS -- loan sales + controls -- Full sample
        2) Pooled OLS -- internet + controls -- Full sample
        3) Pooled OLS -- loan sales + internet + controls -- Full sample
        4) FE: cert and date*msamd -- loan sales + controls -- Full sample
        5) FE: cert and date*msamd -- loan sales + controls -- Reduced sample
        6) FE: cert and date*msamd -- internet + controls -- Reduced sample
        7) FE: cert and date*msamd -- loan sales + internet + controls -- Reduced sample
    
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
df_res = df.dropna()

# Set variables
y_var = 'log_min_distance'
x_ls_var = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'ln_ta', 'ln_emp',  'ln_num_branch',  'cb', 'ln_density','ln_pop_area', 'ln_mfi',  'hhi']
x_int_var = ['perc_broadband', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
              'ln_ta', 'ln_emp',  'ln_num_branch', 'cb', 'ln_density', 'ln_pop_area', 'ln_mfi', 'hhi']
x_tot_var = ['ls_num', 'perc_broadband', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'ln_ta', 'ln_emp', 'ln_num_branch', 'cb', 'ln_density', 'ln_pop_area', 'ln_mfi', 'hhi']

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['msamd','date','cert']
how = 'msamd x date, cert'

msamd_full = df_full.msamd.nunique()
msamd_res = df_res.msamd.nunique()
cert_full = df_full.cert.nunique()
cert_res = df_res.cert.nunique()

# Set File names
file_ols_full = 'Results/Results_ols_full.{}'
file_olsint = 'Results/Results_olsint_full.{}'
file_olslsint = 'Results/Results_olslsint_full.{}'
file_fels_full = 'Results/Results_fels_full.{}'
file_fels_res = 'Results/Results_fels_res.{}'
file_feint_res = 'Results/Results_feint_res.{}'
file_felsint_res = 'Results/Results_felsint_res.{}'

#------------------------------------------------------------
# 1) Pooled OLS -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
results_ols_full = MultiDimensionalOLS().fit(df_full[y_var], df_full[x_ls_var], cov_type = cov_type, cluster_cols = df_full[cluster_cols_var])

# Transform to pandas df
df_results_ols_full = results_ols_full.to_dataframe()

# Add count for msamd en cert
df_results_ols_full['msamd'] = msamd_full
df_results_ols_full['cert'] = cert_full

# Save to excel and csv
df_results_ols_full.to_excel(file_ols_full.format('xlsx'))
df_results_ols_full.to_csv(file_ols_full.format('csv'))

#------------------------------------------------------------
# 2) Pooled OLS -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
results_olsint = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_int_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var])

# Transform to pandas df
df_results_olsint = results_olsint.to_dataframe()

# Add count for msamd en cert
df_results_olsint['msamd'] = msamd_res
df_results_olsint['cert'] = cert_res

# Save to excel and csv
df_results_olsint.to_excel(file_olsint.format('xlsx'))
df_results_olsint.to_csv(file_olsint.format('csv'))

#------------------------------------------------------------
# 3) Pooled OLS -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
results_olslsint = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_tot_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var])

# Transform to pandas df
df_results_olslsint = results_olslsint.to_dataframe()

# Add count for msamd en cert
df_results_olslsint['msamd'] = msamd_res
df_results_olslsint['cert'] = cert_res

# Save to excel and csv
df_results_olslsint.to_excel(file_olslsint.format('xlsx'))
df_results_olslsint.to_csv(file_olslsint.format('csv'))

#------------------------------------------------------------
# 4) FE: cert and date*msamd -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
results_fels_full = MultiDimensionalOLS().fit(df_full[y_var], df_full[x_ls_var], cov_type = cov_type, cluster_cols = df_full[cluster_cols_var],\
                                       transform_data = True, FE_cols = df_full[FE_cols_vars], how = how)

# Transform to pandas df
df_results_fels_full = results_fels_full.to_dataframe()

# Add count for msamd en cert
df_results_fels_full['msamd'] = msamd_full
df_results_fels_full['cert'] = cert_full

# Save to excel and csv
df_results_fels_full.to_excel(file_fels_full.format('xlsx'))
df_results_fels_full.to_csv(file_fels_full.format('csv'))

#------------------------------------------------------------
# 5) FE: cert and date*msamd -- loan sales + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_fels_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_ls_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                       transform_data = True, FE_cols = df_res[FE_cols_vars], how = how)

# Transform to pandas df
df_results_fels_res = results_fels_res.to_dataframe()

# Add count for msamd en cert
df_results_fels_res['msamd'] = msamd_res
df_results_fels_res['cert'] = cert_res

# Save to excel and csv
df_results_fels_res.to_excel(file_fels_res.format('xlsx'))
df_results_fels_res.to_csv(file_fels_res.format('csv'))

#------------------------------------------------------------
# 6) FE: cert and date*msamd -- internet + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_feint_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_int_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                       transform_data = True, FE_cols = df_res[FE_cols_vars], how = how)

# Transform to pandas df
df_results_feint_res = results_feint_res.to_dataframe()

# Add count for msamd en cert
df_results_feint_res['msamd'] = msamd_res
df_results_feint_res['cert'] = cert_res

# Save to excel and csv
df_results_feint_res.to_excel(file_feint_res.format('xlsx'))
df_results_feint_res.to_csv(file_feint_res.format('csv'))

#------------------------------------------------------------
# 7) FE: cert and date*msamd -- loan sales + internet + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_felsint_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_tot_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                       transform_data = True, FE_cols = df_res[FE_cols_vars], how = how)

# Transform to pandas df
df_results_felsint_res = results_felsint_res.to_dataframe()

# Add count for msamd en cert
df_results_felsint_res['msamd'] = msamd_res
df_results_felsint_res['cert'] = cert_res

# Save to excel and csv
df_results_felsint_res.to_excel(file_felsint_res.format('xlsx'))
df_results_felsint_res.to_csv(file_felsint_res.format('csv'))