# Robustness results

''' This script makes the following robustness estimations:
        1) LPM with CDD distance
        2) LPM with local dummy
        3) LPM with distance * subprime, LS * subprime
        4) LPM with split sample subprime
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

columns = ['date','fips','msamd','cert','log_min_distance','lti','ln_loanamout','ln_appincome',\
           'subprime','lien','hoepa','owner','preapp', 'coapp','ethnicity_0',\
           'ethnicity_1', 'ethnicity_2','ethnicity_3', 'ethnicity_4', 'ethnicity_5',\
           'sex_1','sex_2','ls_ever','loan_originated','loan_type_2', 'loan_type_3',\
           'loan_type_4','local','log_min_distance_cdd']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Add interaction terms
dd_main = dd_main.assign(log_min_distance_ls_ever = dd_main['log_min_distance'] * dd_main['ls_ever'])
dd_main = dd_main.assign(log_min_distance_cdd_ls_ever = dd_main['log_min_distance_cdd'] * dd_main['ls_ever'])
dd_main = dd_main.assign(local_ls_ever = dd_main['local'] * dd_main['ls_ever'])
dd_main = dd_main.assign(suprime_ls_ever = dd_main['subprime'] * dd_main['ls_ever'])
dd_main = dd_main.assign(suprime_log_min_distance = dd_main['subprime'] * dd_main['log_min_distance'])


#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y = 'loan_originated'
x0 = ['log_min_distance','log_min_distance_ls_ever','lti','ln_loanamout',\
     'ln_appincome','lien','owner','preapp', 'coapp',\
     'ethnicity_1', 'ethnicity_2','ethnicity_3', 'ethnicity_4', 'ethnicity_5',\
     'sex_1','loan_type_2', 'loan_type_3', 'loan_type_4']
x1 = ['log_min_distance_cdd','log_min_distance_cdd_ls_ever','lti','ln_loanamout',\
     'ln_appincome','lien','owner','preapp', 'coapp',\
     'ethnicity_1', 'ethnicity_2','ethnicity_3', 'ethnicity_4', 'ethnicity_5',\
     'sex_1','loan_type_2', 'loan_type_3', 'loan_type_4']
x2 = ['local','local_ls_ever','lti','ln_loanamout',\
     'ln_appincome','lien','owner','preapp', 'coapp',\
     'ethnicity_1', 'ethnicity_2','ethnicity_3', 'ethnicity_4', 'ethnicity_5',\
     'sex_1','loan_type_2', 'loan_type_3', 'loan_type_4']
    
# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['fips','cert']
how = 'fips, cert'

# Set File names
## OLS
file1 = 'Robustness_checks/lpm_robust_cdd_{}.{}'
file2 = 'Robustness_checks/lpm_robust_local_{}.{}'

#------------------------------------------------------------
# Run benchmark Model
#------------------------------------------------------------
# Loop over all years
for year in range(2004,2019+1):
# Set df
    df = dd_main[dd_main.date == year].compute(scheduler = 'threads')
    df.reset_index(drop = True, inplace = True)
    
    for x, file in zip([x1,x2], [file1,file2]):
    # Run first three in a loop
        results = MultiDimensionalOLS().fit(df[y], df[x],\
                            cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                            transform_data = True, FE_cols = df[FE_cols_vars], how = how)
        
        
        # Transform to pandas df
        df_results = results.to_dataframe()
        
        
        # Save to excel and csv
        df_results.to_excel(file.format(year,'xlsx'))
        df_results.to_csv(file.format(year,'csv'))
        
        del results, df_results
    
    del df
