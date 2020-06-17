# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    Working paper 2
    
    The following models are estimated:
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
    
        11) RE: cert and date*msamd -- loan sales + controls -- Full sample
        12) RE: cert and date*msamd -- loan sales + controls -- Reduced sample
        13) RE: cert and date*msamd -- internet + controls -- Reduced sample
        14) RE: cert and date*msamd -- loan sales + internet + controls -- Reduced sample
    
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
msat_invar = ['ln_density', 'ln_pop_area', 'ln_mfi', 'hhi'] # Used to remove msat-fixed vars. Example: [x for x in x_int_var if x not in msat_invar]

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['msamd','date','cert']
how = 'msamd x date, cert'
FE_cols_vars_t = ['date','cert']
how_t = 'date, cert'

msamd_full = df_full.msamd.nunique()
msamd_res = df_res.msamd.nunique()
cert_full = df_full.cert.nunique()
cert_res = df_res.cert.nunique()

# Set File names
## OLS
file_ols_ls_full = 'Results/Results_ols_ls_full.{}'
file_ols_ls_res = 'Results/Results_ols_ls_res.{}'
file_ols_int_res = 'Results/Results_ols_int_res.{}'
file_ols_lsint_res = 'Results/Results_ols_lsint_res.{}'

## FE: MSAT, Lender
file_fe_msatcert_ls_full = 'Results/Results_fe_msatcert_ls_full.{}'
file_fe_msatcert_ls_res = 'Results/Results_fe_msatcert_ls_res.{}'

## FE: T, Lender
file_fe_tcert_ls_full = 'Results/Results_fe_tcert_ls_full.{}'
file_fe_tcert_ls_res = 'Results/Results_fe_tcert_ls_res.{}'
file_fe_tcert_int_res = 'Results/Results_fe_tcert_int_res.{}'
file_fe_tcert_lsint_res = 'Results/Results_fe_tcert_lsint_res.{}'

## RE: MSAT, Lender
file_re_msatcert_ls_full = 'Results/Results_re_msatcert_ls_full.{}'
file_re_msatcert_ls_res  = 'Results/Results_re_msatcert_ls_res.{}'
file_re_msatcert_int_res  = 'Results/Results_re_msatcert_int_res.{}'
file_re_msatcert_lsint_res  = 'Results/Results_re_msatcert_lsint_res{}'

#------------------------------------------------------------
# 1) Pooled OLS -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
results_ols_ls_full = MultiDimensionalOLS().fit(df_full[y_var], df_full[x_ls_var], cov_type = cov_type, cluster_cols = df_full[cluster_cols_var])

# Transform to pandas df
df_results_ols_ls_full = results_ols_ls_full.to_dataframe()

# Add count for msamd en cert
df_results_ols_ls_full['msamd'] = msamd_full
df_results_ols_ls_full['cert'] = cert_full

# Save to excel and csv
df_results_ols_ls_full.to_excel(file_ols_ls_full.format('xlsx'))
df_results_ols_ls_full.to_csv(file_ols_ls_full.format('csv'))

#------------------------------------------------------------
# 2) Pooled OLS -- loan sales + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_ols_ls_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_ls_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var])

# Transform to pandas df
df_results_ols_ls_res = results_ols_ls_res.to_dataframe()

# Add count for msamd en cert
df_results_ols_ls_res['msamd'] = msamd_res
df_results_ols_ls_res['cert'] = cert_res

# Save to excel and csv
df_results_ols_ls_res.to_excel(file_ols_ls_res.format('xlsx'))
df_results_ols_ls_res.to_csv(file_ols_ls_res.format('csv'))

#------------------------------------------------------------
# 3) Pooled OLS -- Internet + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_ols_int_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_int_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var])

# Transform to pandas df
df_results_ols_int_res = results_ols_int_res.to_dataframe()

# Add count for msamd en cert
df_results_ols_int_res['msamd'] = msamd_res
df_results_ols_int_res['cert'] = cert_res

# Save to excel and csv
df_results_ols_int_res.to_excel(file_ols_int_res.format('xlsx'))
df_results_ols_int_res.to_csv(file_ols_int_res.format('csv'))

#------------------------------------------------------------
# 4) Pooled OLS -- Loan Sales + Internet + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_ols_lsint_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_tot_var], cov_type = cov_type, cluster_cols = df_res[cluster_cols_var])

# Transform to pandas df
df_results_ols_lsint_res = results_ols_lsint_res.to_dataframe()

# Add count for msamd en cert
df_results_ols_lsint_res['msamd'] = msamd_res
df_results_ols_lsint_res['cert'] = cert_res

# Save to excel and csv
df_results_ols_lsint_res.to_excel(file_ols_lsint_res.format('xlsx'))
df_results_ols_lsint_res.to_csv(file_ols_lsint_res.format('csv'))

#------------------------------------------------------------
# 5) FE: cert and date*msamd -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
## NOTE: remove msat-invariant variables 
results_fe_msatcert_ls_full = MultiDimensionalOLS().fit(df_full[y_var], df_full[[x for x in x_ls_var if x not in msat_invar]],\
                                                 cov_type = cov_type, cluster_cols = df_full[cluster_cols_var],\
                                                 transform_data = True, FE_cols = df_full[FE_cols_vars], how = how)

# Transform to pandas df
df_results_fe_msatcert_ls_full = results_fe_msatcert_ls_full.to_dataframe()

# Add count for msamd en cert
df_results_fe_msatcert_ls_full['msamd'] = msamd_full
df_results_fe_msatcert_ls_full['cert'] = cert_full

# Save to excel and csv
df_results_fe_msatcert_ls_full.to_excel(file_fe_msatcert_ls_full.format('xlsx'))
df_results_fe_msatcert_ls_full.to_csv(file_fe_msatcert_ls_full.format('csv'))

#------------------------------------------------------------
# 6) FE: cert and date*msamd -- loan sales + controls -- Reduced sample
#------------------------------------------------------------

# Run
## NOTE: remove msat-invariant variables 
results_fe_msatcert_ls_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[[x for x in x_ls_var if x not in msat_invar]],\
                                                cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                                transform_data = True, FE_cols = df_res[FE_cols_vars], how = how)

# Transform to pandas df
df_results_fe_msatcert_ls_res = results_fe_msatcert_ls_res.to_dataframe()

# Add count for msamd en cert
df_results_fe_msatcert_ls_res['msamd'] = msamd_res
df_results_fe_msatcert_ls_res['cert'] = cert_res

# Save to excel and csv
df_results_fe_msatcert_ls_res.to_excel(file_fe_msatcert_ls_res.format('xlsx'))
df_results_fe_msatcert_ls_res.to_csv(file_fe_msatcert_ls_res.format('csv'))

#------------------------------------------------------------
# 7) FE: cert and date -- loan sales + controls -- Full sample
#------------------------------------------------------------

# Run
results_fe_tcert_ls_full = MultiDimensionalOLS().fit(df_full[y_var], df_full[x_ls_var],\
                                             cov_type = cov_type, cluster_cols = df_full[cluster_cols_var],\
                                             transform_data = True, FE_cols = df_full[FE_cols_vars_t], how = how_t)

# Transform to pandas df
df_results_fe_tcert_ls_full = results_fe_tcert_ls_full.to_dataframe()

# Add count for msamd en cert
df_results_fe_tcert_ls_full['msamd'] = msamd_full
df_results_fe_tcert_ls_full['cert'] = cert_full

# Save to excel and csv
df_results_fe_tcert_ls_full.to_excel(file_fe_tcert_ls_full.format('xlsx'))
df_results_fe_tcert_ls_full.to_csv(file_fe_tcert_ls_full.format('csv'))

#------------------------------------------------------------
# 8) FE: cert and date -- loan sales + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_fe_tcert_ls_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_ls_var],\
                                             cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                             transform_data = True, FE_cols = df_res[FE_cols_vars_t], how = how_t)

# Transform to pandas df
df_results_fe_tcert_ls_res = results_fe_tcert_ls_res.to_dataframe()

# Add count for msamd en cert
df_results_fe_tcert_ls_res['msamd'] = msamd_res
df_results_fe_tcert_ls_res['cert'] = cert_res

# Save to excel and csv
df_results_fe_tcert_ls_res.to_excel(file_fe_tcert_ls_res.format('xlsx'))
df_results_fe_tcert_ls_res.to_csv(file_fe_tcert_ls_res.format('csv'))

#------------------------------------------------------------
# 9) FE: cert and date -- Internet + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_fe_tcert_int_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_int_var],\
                                              cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                              transform_data = True, FE_cols = df_res[FE_cols_vars_t], how = how_t)

# Transform to pandas df
df_results_fe_tcert_int_res = results_fe_tcert_int_res.to_dataframe()

# Add count for msamd en cert
df_results_fe_tcert_int_res['msamd'] = msamd_res
df_results_fe_tcert_int_res['cert'] = cert_res

# Save to excel and csv
df_results_fe_tcert_int_res.to_excel(file_fe_tcert_int_res.format('xlsx'))
df_results_fe_tcert_int_res.to_csv(file_fe_tcert_int_res.format('csv'))

#------------------------------------------------------------
# 10) FE: cert and date -- Loan Sales + Internet + controls -- Reduced sample
#------------------------------------------------------------

# Run
results_fe_tcert_lsint_res = MultiDimensionalOLS().fit(df_res[y_var], df_res[x_tot_var],\
                                                cov_type = cov_type, cluster_cols = df_res[cluster_cols_var],\
                                                transform_data = True, FE_cols = df_res[FE_cols_vars_t], how = how_t)

# Transform to pandas df
df_results_fe_tcert_lsint_res = results_fe_tcert_lsint_res.to_dataframe()

# Add count for msamd en cert
df_results_fe_tcert_lsint_res['msamd'] = msamd_res
df_results_fe_tcert_lsint_res['cert'] = cert_res

# Save to excel and csv
df_results_fe_tcert_lsint_res.to_excel(file_fe_tcert_lsint_res.format('xlsx'))
df_results_fe_tcert_lsint_res.to_csv(file_fe_tcert_lsint_res.format('csv'))

#------------------------------------------------------------
# 11) RE: cert and date*msamd -- Loan Sales + controls -- Full sample
#------------------------------------------------------------

#------------------------------------------------------------
# 12) RE: cert and date*msamd -- Loan Sales + controls -- Reduced sample
#------------------------------------------------------------


#------------------------------------------------------------
# 13) RE: cert and date*msamd -- Internet + controls -- Reduced sample
#------------------------------------------------------------

#------------------------------------------------------------
# 14) RE: cert and date*msamd -- Loan Sales + Internet +controls -- Reduced sample
#------------------------------------------------------------
