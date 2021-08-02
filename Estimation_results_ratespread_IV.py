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

# Import packages for the Sargan-Hausman test
from linearmodels.iv._utility import annihilate
from linearmodels.utility import WaldTestStatistic

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

columns = ['date','fips','msamd','cert','rate_spread','log_min_distance','local',\
           'ls','ltv','lti','ln_loanamout','ln_appincome','int_only','balloon',\
           'lien','mat','hoepa','owner','preapp', 'coapp','loan_originated',\
           'loan_term']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Add intercept and interaction term
## Subset data
dd_main = dd_main[(dd_main.date.astype(int) >= 2018) & (dd_main.loan_originated == 1)]

## Add remote dummy
dd_main['remote'] = (dd_main['local'] - 1).abs()

## Intercept
#dd_main['intercept'] = 1

## Interaction terms
dd_main = dd_main.assign(remote_ls = dd_main['remote'] * dd_main['ls'])
#dd_main = dd_main.assign(log_min_distance_ls = dd_main['log_min_distance'] * dd_main['ls'])

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
# Add instruments to dataframe
#------------------------------------------------------------

# Load ls_other
ls_other = pd.read_csv('Data/df_ls_other.csv')
             
# Merge dataframes            
df = df.merge(ls_other[['fips','date','cert','ls_other']], how = 'inner', on = ['fips','date','cert'])

# Drop nans
df.dropna(inplace = True)

# Make interaction term
df = df.assign(ls_other_remote = df.ls_other * df.remote)

del dd_main

#------------------------------------------------------------
# Split dataframe into three and demean
#------------------------------------------------------------

def DataDemean(data):
    df_grouped_fips = data.groupby('fips').transform('mean')
    df_grouped_lender = data.groupby('cert').transform('mean')
    df_mean = data.mean()
    
    ## Transform data
    df_trans = data - df_grouped_fips - df_grouped_lender + df_mean 
    
    ## Add MSA variable
    df_trans['msamd'] = data.msamd
    
    ## Dropna
    #df_trans.dropna(inplace = True)
    
    return df_trans

# Full data
df_trans_full = DataDemean(df)
df_trans_2018 = DataDemean(df[df.date == 2018])
df_trans_2019 = DataDemean(df[df.date == 2019])
#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y = 'rate_spread'
x_endo = ['ls', 'remote_ls']
x_exo = ['remote','lti','ltv','ln_loanamout',\
     'ln_appincome','int_only','balloon','mat','lien', 'coapp']
z = ['ls_other','ls_other_remote']
    
# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'

# Set File names
file = 'Results/ratespread_results_local_IV_{}_{}.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------

for data, doc_string in zip([df_trans_full, df_trans_2018, df_trans_2019],['full','2018','2019']):
    
    #------------------------------------------------------------
    # Compute first stage results
    results_fs1 = MultiDimensionalOLS().fit(data[x_endo[0]], data[z + x_exo],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    results_fs2 = MultiDimensionalOLS().fit(data[x_endo[1]], data[z + x_exo],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    
    ## Perform partial f-test
    ### First calculate results without instrument
    results_fs1_noinstr = MultiDimensionalOLS().fit(data[x_endo[0]], data[x_exo],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    results_fs2_noinstr = MultiDimensionalOLS().fit(data[x_endo[1]], data[x_exo],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
        
    ### Do f-test
    f_stat_fs1 = ((results_fs1_noinstr.rss - results_fs1.rss) / 2) / results_fs1.mse_resid
    f_stat_fs2 = ((results_fs2_noinstr.rss - results_fs2.rss) / 2) / results_fs2.mse_resid
    
    ## Transform to dataframe and save
    df_results_fs1 = results_fs1.to_dataframe()
    df_results_fs2 = results_fs2.to_dataframe()
    
    ### Add F-test to dataframe
    df_results_fs1['fstat'] = f_stat_fs1
    df_results_fs2['fstat'] = f_stat_fs2
    
    ### Save 
    df_results_fs1.to_csv(file.format(doc_string,'fs1','csv'))
    df_results_fs2.to_csv(file.format(doc_string,'fs2','csv'))
    
    ## Add fitted values to df
    data['ls_hat'] = results_fs1.fittedvalues
    data['ls_other_remote_hat'] = results_fs2.fittedvalues
    
    ## Add residuals to df
    data['ls_res'] = results_fs1.resid
    data['ls_other_remote_res'] = results_fs2.resid
    
    #------------------------------------------------------------
    # Compute Second Stage results
    results_ss = MultiDimensionalOLS().fit(data[y], data[['ls_hat'] + [x_exo[0]] + ['ls_other_remote_hat'] + x_exo[1:]],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
        
    ## Transform to dataframe and save
    df_results_ss = results_ss.to_dataframe()
    
    ## Perform Hausman-Hu test
    results_ss_hw = MultiDimensionalOLS().fit(data[y], data[['ls_res','ls_other_remote_res'] + ['ls'] + [x_exo[0]] + ['ls_other_remote'] + x_exo[1:]],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    
    ## To dataframe
    df_results_ss_hw = results_ss_hw.to_dataframe() 
    
    ### Perform joint F-test
    results_ss_hw_nores = MultiDimensionalOLS().fit(data[y], data[['ls'] + [x_exo[0]] + ['ls_other_remote'] + x_exo[1:]],\
                        cov_type = cov_type, cluster_cols = data[cluster_cols_var])
    
    f_stat_ss_hw_nores = ((results_ss_hw_nores.rss - results_ss_hw.rss) / 2) / results_ss_hw.mse_resid
    pval_f_stat_ss_hw_nores = 1 - stats.f.cdf(f_stat_ss_hw_nores, data.shape[0], data.shape[0])
    
    ## Add to results df
    df_results_ss['dwh_value'] = f_stat_ss_hw_nores
    df_results_ss['dwh_p'] = pval_f_stat_ss_hw_nores
    
    # Save to
    df_results_ss.to_csv(file.format(doc_string,'ss','csv'))
    df_results_ss_hw.to_csv(file.format(doc_string,'ss_hw','csv'))
