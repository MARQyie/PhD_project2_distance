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
           'ln_ta','ln_emp','ln_num_branch','cb','local','log_min_distance_cdd',\
           'ls_gse', 'ls_priv', 'sec','ls_ever','loan_costs']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Interact msamd and date
dd_main = dd_main.assign(msat = dd_main.date.astype(str) + dd_main.msamd.astype(str))

## Add remote dummy
dd_main['remote'] = (dd_main['local'] - 1).abs()

        
# Transform to pandas df
## FULL
df = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)

df_grouped_msat = df.groupby('msat')[columns[4:] + ['remote']].transform('mean')
df_grouped_fips = df.groupby('fips')[columns[4:] + ['remote']].transform('mean')
df_grouped_lender = df.groupby('cert')[columns[4:] + ['remote']].transform('mean')
df_mean = df[columns[4:] + ['remote']].mean()

df_trans = df[columns[4:] + ['remote']] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

### Add MSA
df_trans['msamd'] = df.msamd

del df_grouped_msat, df_grouped_fips, df_grouped_lender, df, dd_main

# 2018-2019
dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

df1819 = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')
df1819 = df1819[df1819.date.astype(int) >= 2018]
df1819.reset_index(drop = True, inplace = True)
df1819.dropna(inplace = True)
df1819['loan_costs'] = df1819['loan_costs'] / 1e3

del dd_main

#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y_lst = ['log_min_distance_cdd','remote','log_min_distance','log_min_distance']
x0 = ['ls','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
x1 = ['ls_gse', 'ls_priv', 'sec','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
x2 = ['ls_ever','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'

##1819
cov_type1819 = 'clustered'
cluster_cols_var1819 = 'msamd'
FE_cols_vars1819 = ['fips','cert']
how1819 = 'fips, cert'

# Set File names
file1 = 'Robustness_checks/Distance_robust_cdd.{}'
file2 = 'Robustness_checks/Distance_robust_remote.{}'
file3 = 'Robustness_checks/Distance_robust_lssplit.{}'
file4 = 'Robustness_checks/Distance_robust_lsever.{}'
file5 = 'Robustness_checks/Distance_robust_loancosts.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------
    
# Run
for y,x,file in zip(y_lst,[x0,x0,x1,x2],[file1,file2,file3,file4]):
    results = MultiDimensionalOLS().fit(df_trans[y], df_trans[x],\
                    cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
    
    ### Transform to pandas df
    df_results = results.to_dataframe()
    
    ## Do wald test for file3
    if file == file3:
        R = pd.DataFrame([[1, -1, 0],[0, 1, -1]])
        h_beta = R @ pd.DataFrame(results.params[:3])
        C = results.nobs * results.cov.iloc[:3,:3]
        test_stat = results.nobs * h_beta.T @ np.linalg.inv(R @ C @ R.T) @ h_beta
        
        ## F test
        pval_wald = stats.chi2.sf(test_stat, R.shape[0])
    
        df_results['wald_stat'] = test_stat.iloc[0,0]
        df_results['wald_pval'] = float(pval_wald)
    
    ### Save to excel and csv
    df_results.to_excel(file.format('xlsx'))
    df_results.to_csv(file.format('csv'))

# Run 1819
results_1819 = MultiDimensionalOLS().fit(df1819['loan_costs'], df1819[x0],\
                        cov_type = cov_type1819, cluster_cols = df1819[cluster_cols_var1819],\
                        transform_data = True, FE_cols = df1819[FE_cols_vars1819], how = how1819)

# To df
df_results_1819 = results_1819.to_dataframe()
    
## Save to excel and csv
df_results_1819.to_excel(file5.format('xlsx'))
df_results_1819.to_csv(file5.format('csv'))