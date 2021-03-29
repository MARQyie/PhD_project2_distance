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

# Parallel functionality pandas via Dask
import dask.dataframe as dd 

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
df = dd_main[(dd_main.loan_originated == 1)].compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)
df['date'] = df.date.astype(int)

del dd_main
#------------------------------------------------------------
# Add instruments
#------------------------------------------------------------

# (Geographic) diversification
#df['div'] = df.groupby(['cert','fips','date']).fips.transform('count') / df.groupby(['cert','date']).fips.transform('count')

# Marketability
#df['marketability'] = (df[['loan_type_2', 'loan_type_3', 'loan_type_4']].sum(axis = 1) >= 1) *1

# Market share
#df['market_share'] = df.groupby(['cert','msamd','date']).fips.transform('count') / df.groupby(['msamd','date']).fips.transform('count')

# Loan sales liquidity a la Loutskina (2011)
## Calculate portfolio shares prior to the Great Recession
#portfolio_share = df[df.date.astype(int) < 2007].groupby(['msamd','cert']).ln_loanamout.sum() / df[df.date.astype(int) < 2007].groupby(['cert']).ln_loanamout.sum()

## Calculate loan sales market depth
#market_depth = df.groupby(['msamd','date']).ls.sum() / df.groupby(['msamd','date']).ls.count()

## Calculate loan sales liquidity and rename
#ls_liquidity = market_depth * portfolio_share
#ls_liquidity = ls_liquidity.rename('ls_liq')

## Merge with df
#df = df.merge(ls_liquidity, how = 'inner', left_on = ['msamd','date','cert'], right_index = True)

'''TURN ON IF RAN FOR THE FIRST TIME
# Percentage loans sold per county by other banks
unique_certs = df.cert.unique().sort_values().tolist()

# Set and sortindex for df
df.set_index(['cert','fips','date'], inplace = True)
df.sort_index(inplace = True)

# Get sum and count of total df
df_grouped_sumcount = df.groupby(['fips','date']).ls.agg(['sum','count'])

# Set mask indexer
mask_index = df.index.get_loc

# Set function
def soldOther(lender):
    mask = mask_index(lender)
    
    return (df_grouped_sumcount - df[mask].groupby(['fips','date']).ls.agg(['sum','count'])).assign(cert = lender)

# Loop over function
ls_other = []
append = ls_other.append

for lender in unique_certs:
    append(soldOther(lender))

# Make pandas dataframe and add mean measure
ls_other = pd.concat(ls_other).dropna()
ls_other['ls_other'] = ls_other['sum'] / ls_other['count']

## reset index
ls_other.reset_index(inplace = True)

# Save to csv
ls_other.to_csv('Data/df_ls_other.csv') '''

# Load ls_other
ls_other = pd.read_csv('Data/df_ls_other.csv')
             
# Merge dataframes            
df = df.merge(ls_other[['fips','date','cert','ls_other']], how = 'inner', on = ['fips','date','cert'])

del ls_other

## Add instruments to columns
#columns.append('div')
#columns.append('marketability')
#columns.append('market_share')
#columns.append('ls_liq')
columns.append('ls_other')

#------------------------------------------------------------
# Transform data (within transformation)
#------------------------------------------------------------

#------------------------------------------------------------
# Full sample
## Get group means
df_grouped_msat = df.groupby('msat')[columns[4:]].transform('mean')
df_grouped_fips = df.groupby('fips')[columns[4:]].transform('mean')
df_grouped_lender = df.groupby('cert')[columns[4:]].transform('mean')
df_mean = df[columns[4:]].mean()

## Transform data
df_trans = df[columns[4:]] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

## Add MSA variable
df_trans['msamd'] = df.msamd

## Dropna
df_trans.dropna(inplace = True)

del df_grouped_msat, df_grouped_fips, df_grouped_lender, df_mean

#------------------------------------------------------------
# After 2009
## Subset full dataset
df_1019 = df[df.date.astype(int) >= 2010]

## Get group means
df_grouped_msat = df_1019.groupby('msat')[columns[4:]].transform('mean')
df_grouped_fips = df_1019.groupby('fips')[columns[4:]].transform('mean')
df_grouped_lender = df_1019.groupby('cert')[columns[4:]].transform('mean')
df_mean = df_1019[columns[4:]].mean()

## Transform data
df_trans_1019 = df_1019[columns[4:]] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

## Add MSA variable
### Add MSA
df_trans_1019['msamd'] = df_1019.msamd

## Dropna
df_trans_1019.dropna(inplace = True)

del df, df_grouped_msat, df_grouped_fips, df_grouped_lender, df_mean, df_1019

#------------------------------------------------------------
# Set variables, parameters and file names
#------------------------------------------------------------

# Set variables
y = 'log_min_distance'
x_endo = 'ls'
x_exo = ['lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
z = 'ls_other'

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'

# Set File names
file = 'Robustness_checks/Distance_robust_benchmark_IV_{}.{}'
file_1019 = 'Robustness_checks/Distance_robust_benchmark_IV_{}_1019.{}'

#------------------------------------------------------------
# Perform 2SLS
#------------------------------------------------------------

#------------------------------------------------------------
#------------------------------------------------------------
# FULL SAMPLE 

#------------------------------------------------------------
# First Stage

## Compute first stage results
results_fs = MultiDimensionalOLS().fit(df_trans[x_endo], df_trans[[z] + x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
    
## Transform to df
df_results_fs = results_fs.to_dataframe()

''' TURN ON IF MULTIPLE INSTRUMENTS
## Perform partial f-test
### First calculate results without instrument
results_fs_noinstr = MultiDimensionalOLS().fit(df_trans[x_endo], df_trans[x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])

### Do f-test
f_stat = ((results_fs_noinstr.rss - results_fs.rss) / 1) / results_fs.mse_resid '''

## Save to csv
df_results_fs.to_csv(file.format('fs','csv'))

## Calculate LS hat and append to df
df_trans['ls_hat'] = results_fs.fittedvalues

## Add residuals to df_trans (for Hausman Wu test later on)
df_trans['resids_fs'] = results_fs.resid

del results_fs
#------------------------------------------------------------
# Second Stage

## Compute second stage results
results_ss = MultiDimensionalOLS().fit(df_trans[y], df_trans[['ls_hat'] + x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
    
## Transform to df
df_results_ss = results_ss.to_dataframe()
del results_ss

## Perform Hausman-Hu test
### Perform HW test 
results_ss_hw = MultiDimensionalOLS().fit(df_trans[y], df_trans[['ls','resids_fs'] + x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
    
df_results_ss_hw = results_ss_hw.to_dataframe() # Highly significant
del results_ss_hw

### Ad hw-stat to df_results_ss
df_results_ss['hw_pval'] = df_results_ss_hw.loc['resids_fs','p']

## Save to csv
df_results_ss.to_csv(file.format('ss','csv'))

#------------------------------------------------------------
#------------------------------------------------------------
# 1019 SAMPLE 

#------------------------------------------------------------
# First Stage

## Compute first stage results
results_fs = MultiDimensionalOLS().fit(df_trans_1019[x_endo], df_trans_1019[[z] + x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans_1019[cluster_cols_var])
    
## Transform to df
df_results_fs = results_fs.to_dataframe()

''' TURN ON IF MULTIPLE INSTRUMENTS
## Perform partial f-test
### First calculate results without instrument
results_fs_noinstr = MultiDimensionalOLS().fit(df_trans_1019[x_endo], df_trans_1019[x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans_1019[cluster_cols_var])

### Do f-test
f_stat = ((results_fs_noinstr.rss - results_fs.rss) / 1) / results_fs.mse_resid '''

## Save to csv
df_results_fs.to_csv(file_1019.format('fs','csv'))

## Calculate LS hat and append to df
df_trans_1019['ls_hat'] = results_fs.fittedvalues

## Add residuals to df_trans_1019 (for Hausman Wu test later on)
df_trans_1019['resids_fs'] = results_fs.resid

del results_fs
#------------------------------------------------------------
# Second Stage

## Compute second stage results
results_ss = MultiDimensionalOLS().fit(df_trans_1019[y], df_trans_1019[['ls_hat'] + x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans_1019[cluster_cols_var])
    
## Transform to df
df_results_ss = results_ss.to_dataframe()
del results_ss

## Perform Hausman-Hu test
### Perform HW test 
results_ss_hw = MultiDimensionalOLS().fit(df_trans_1019[y], df_trans_1019[['ls','resids_fs'] + x_exo],\
                    cov_type = cov_type, cluster_cols = df_trans_1019[cluster_cols_var])
    
df_results_ss_hw = results_ss_hw.to_dataframe() # Highly significant
del results_ss_hw

### Ad hw-stat to df_results_ss
df_results_ss['hw_pval'] = df_results_ss_hw.loc['resids_fs','p']

## Save to csv
df_results_ss.to_csv(file_1019.format('ss','csv'))