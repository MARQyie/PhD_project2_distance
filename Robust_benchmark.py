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
           'ls_gse', 'ls_priv', 'sec']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Interact msamd and date
dd_main = dd_main.assign(msat = dd_main.date.astype(str) + dd_main.msamd.astype(str))

## Add remote dummy
dd_main['remote'] = (dd_main['local'] - 1).abs()

## Add LS non-securitized
dd_main['ls_ns'] = dd_main[['ls_gse','ls_priv']].sum(axis = 1)
columns.extend(['remote','ls_ns'])

# Transform to pandas df
## FULL
df = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)

del dd_main

# Drop column local
df.drop(columns = ['local'], inplace = True)

#------------------------------------------------------------
# Add instruments to dataframe
#------------------------------------------------------------

# Load ls_other
ls_other = pd.read_csv('Data/df_ls_other.csv', na_values = ['count','sum']) # dirty trick to remove strings from float columns

# Make ls_ns_other
ls_other['ls_ns_other'] = ls_other[['ls_gse','ls_priv']].sum(axis = 1) / ls_other[['ls_gse.1','ls_priv.1']].sum(axis = 1)
         
# Merge dataframes            
df = df.merge(ls_other[['fips','date','cert','ls_other','ls_gse_other','ls_priv_other','sec_other','ls_ns_other']], how = 'inner', on = ['fips','date','cert']) # 'sec_other'

# Drop nans to make 2018-2019 sample
df.dropna(subset = ['ls_other'],inplace = True)

# Add instruments to columns
columns.extend(['ls_other','ls_gse_other','ls_priv_other','sec_other','ls_ns_other'])
columns.remove('local')
del ls_other

#------------------------------------------------------------
# Split dataframe into three and demean
#------------------------------------------------------------

#------------------------------------------------------------
# df
# Demean data
df_grouped_msat = df.groupby('msat')[columns].transform('mean')
df_grouped_fips = df.groupby('fips')[columns].transform('mean')
df_grouped_lender = df.groupby('cert')[columns].transform('mean')
df_mean = df[columns].mean()

## Transform data
df_trans = df[columns] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

## Add MSA variable
df_trans['msamd'] = df.msamd

del df_grouped_msat, df_grouped_fips, df_grouped_lender, df_mean, df

''' OLD
# Entry into remote markets
## Load and subset data
dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)
df = dd_main[(dd_main.loan_originated == 1) & (dd_main.log_min_distance > 0)].compute(scheduler = 'threads')

## Get lender/year list of unique fips, lag year and merge with df
df_unique_fips = df.groupby(['date','cert']).fips.unique()
df_unique_fips = df_unique_fips.reset_index()
df_unique_fips.date = (df_unique_fips.date.astype(int) + 1).astype('category')
df_unique_fips.rename(columns = {'fips':'fips_list'}, inplace = True)

df_entry = df.merge(df_unique_fips, how = 'inner', on = ['date','cert'])

## Make entry variable
def entryFinder(fips, fips_list):
    try:
        boolean = (fips not in fips_list) * 1
    except:
        boolean = (fips != fips_list) * 1
    return boolean

vecEntryFinder = np.vectorize(entryFinder)

df_entry['entry'] =  vecEntryFinder(df_entry.fips, df_entry.fips_list)

# transform data
df_entry = df_entry.assign(msat = df_entry.date.astype(str) + df_entry.msamd.astype(str))
df_grouped_msat = df_entry.groupby('msat')[columns[4:] + ['entry']].transform('mean')
df_grouped_fips = df_entry.groupby('fips')[columns[4:] + ['entry']].transform('mean')
df_grouped_lender = df_entry.groupby('cert')[columns[4:] + ['entry']].transform('mean')
df_mean = df_entry[columns[4:] + ['entry']].mean()

df_entry_trans = df_entry[columns[4:] + ['entry']] - df_grouped_msat - df_grouped_fips - df_grouped_lender + 2 * df_mean 

df_entry_trans['msamd'] = df_entry.msamd

del dd_main, df, df_unique_fips, df_entry

results = MultiDimensionalOLS().fit(df_entry_trans['entry'], df_entry_trans[['ls','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']],\
                    cov_type = 'clustered', cluster_cols = df_entry_trans['msamd'])
df_results = results.to_dataframe() '''


#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y_lst = ['log_min_distance_cdd','remote','log_min_distance','log_min_distance']
x0 = ['ls','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
x1 = ['ls_gse', 'ls_priv','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']
x2 = ['ls_ns','lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']

z_lst = ['ls_other','ls_other',['ls_gse_other','ls_priv_other'],'ls_ns_other']
    
# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'

# Set File names
file1 = 'Robustness_checks/Distance_robust_cdd_{}.{}'
file2 = 'Robustness_checks/Distance_robust_remote_{}.{}'
file3 = 'Robustness_checks/Distance_robust_lssplit_{}.{}'
file4 = 'Robustness_checks/Distance_robust_lsnonsec_{}.{}'
file5 = 'Robustness_checks/Distance_robust_loancosts_{}.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------
    
''' Notes:
    - Splitting the instruments leads to correlated instruments which are more correlated amongst
        each other than with their respective endogenous variables
    - Especially Securitization looks problematic
    - Adding GSE and Priv leads to believable estimates; gse < 0, priv >0
    - Solo adding (FS: +, +, -): GSE < 0 (P<.05), Priv > 0 (P<.05), Sec PROBLEMATIC > 0 (P<.05)
    - Securitization liquidity (Loutskina 2011) performs very poor
    - LS Other on securitization performs poor, T < 10 (first stage); Second stage is terrible
    - LS minus sec works very well, similar results as benchmark, slightly lower estimate
    - sec_other * sec_ever and sec_ever perform poorly as instrument for sec
    
    Conclusion: Due to the low variation in securitization (only 2.8% securitized) our instrument
    does not have the level of detail to make a strong instrument. As a result, we cannot directly
    estimate the effects of securitization. We can, however, filter out securitization and study the
    changes in the parameter estimates.
'''
#------------------------------------------------------------
# FULL SAMPLE 

for y,x,z,file in zip(y_lst,[x0,x0,x1,x2],z_lst,[file1,file2,file3,file4]):
    #------------------------------------------------------------
    # First Stage
    
    ## Compute first stage results
    if file == file3:
        for i in range(len(z)):
            results_fs = MultiDimensionalOLS().fit(df_trans[x[i]], df_trans[z + x[len(z):]],\
                        cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var]) # NOTE instruments are more correlated to each other than the endogenous variables. Securitization is the most troublesome variable.

            ## Perform partial f-test
            ### First calculate results without instrument
            results_fs_noinstr = MultiDimensionalOLS().fit(df_trans[x[i]], df_trans[x[len(z):]],\
                                cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
            
            ### Do f-test
            f_stat = ((results_fs_noinstr.rss - results_fs.rss) / len(z)) / results_fs.mse_resid
            
            ## Transform to df
            df_results_fs = results_fs.to_dataframe()
            
            ## Add fstat to df
            df_results_fs['fstat'] = f_stat
            
            ## Save to csv
            df_results_fs.to_csv(file.format('fs_{}'.format(i),'csv'))
            
            ## Calculate LS hat and append to df
            df_trans['ls_hat_{}'.format(i)] = results_fs.fittedvalues
            
            ## Add residuals to df_trans (for Hausman Wu test later on)
            df_trans['ls_res_{}'.format(i)] = results_fs.resid
            
            del results_fs, results_fs_noinstr
    else:
        results_fs = MultiDimensionalOLS().fit(df_trans[x[0]], df_trans[[z] + x[1:]],\
                        cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
        
        ## Transform to df
        df_results_fs = results_fs.to_dataframe()
        
        ## Save to csv
        df_results_fs.to_csv(file.format('fs','csv'))
        
        ## Calculate LS hat and append to df
        df_trans['ls_hat'] = results_fs.fittedvalues
        
        ## Add residuals to df_trans (for Hausman Wu test later on)
        df_trans['ls_res'] = results_fs.resid
        
        del results_fs
    #------------------------------------------------------------
    # Second Stage
    if file == file3:
        results_ss = MultiDimensionalOLS().fit(df_trans[y], df_trans[['ls_hat_{}'.format(i) for i in range(len(z))] + x[len(z):]],\
                            cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
         ## Transform to df
        df_results_ss = results_ss.to_dataframe()
        del results_ss
        
        ## Perform Hausman-Hu test
        ### Perform HW test 
        results_ss_hw = MultiDimensionalOLS().fit(df_trans[y], df_trans[x  + ['ls_res_{}'.format(i) for i in range(len(z))] ],\
                            cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
        
        # To dataframe
        df_results_ss_hw = results_ss_hw.to_dataframe() 
                
        ### Perform joint F-test
        results_ss_hw_nores = MultiDimensionalOLS().fit(df_trans[y], df_trans[x],\
                            cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
        
        f_stat_ss_hw_nores = ((results_ss_hw_nores.rss - results_ss_hw.rss) / len(z)) / results_ss_hw.mse_resid
        pval_f_stat_ss_hw_nores = 1 - stats.f.cdf(f_stat_ss_hw_nores, df_trans.shape[0], df_trans.shape[0])
        del results_ss_hw
        
        ## Wald test for estimate equality
        # TODO

        ## Add to results df
        df_results_ss['dwh_value'] = f_stat_ss_hw_nores
        df_results_ss['dwh_p'] = pval_f_stat_ss_hw_nores
        
        # Save to
        df_results_ss.to_csv(file.format('ss','csv'))
        df_results_ss_hw.to_csv(file.format('ss_hw','csv'))
        
    else:
        ## Compute second stage results
        results_ss = MultiDimensionalOLS().fit(df_trans[y], df_trans[['ls_hat'] + x[1:]],\
                            cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
            
        ## Transform to df
        df_results_ss = results_ss.to_dataframe()
        del results_ss
        
        ## Perform Hausman-Hu test
        ### Perform HW test 
        results_ss_hw = MultiDimensionalOLS().fit(df_trans[y], df_trans[['ls','ls_res'] + x[1:]],\
                            cov_type = cov_type, cluster_cols = df_trans[cluster_cols_var])
            
        df_results_ss_hw = results_ss_hw.to_dataframe()
        del results_ss_hw
        
        ### Ad hw-stat to df_results_ss
        df_results_ss['hw_pval'] = df_results_ss_hw.loc['ls_res','p']
        
        ## Change name ls_hat if file == file4
        if file == file4:
            df_results_ss.index = ['ls_ns_hat'] + x[1:]
        
        ## Save to csv
        df_results_ss.to_csv(file.format('ss','csv'))

#------------------------------------------------------------
# 1819

# delete previous data
del df_trans, df_results_fs, df_results_ss, df_results_ss_hw

# Reload data and transform
columns = ['date','fips','msamd','cert','log_min_distance','ls',\
           'lti','ln_loanamout','ln_appincome','subprime',\
           'lien','owner','preapp', 'coapp','loan_originated',\
           'ln_ta','ln_emp','ln_num_branch','cb','local','log_min_distance_cdd',\
           'ls_gse', 'ls_priv', 'sec']
columns.append('loan_costs')
dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Transform to pandas df
## FULL
df1819 = dd_main[(dd_main.date.astype(int) >= 2018) & (dd_main.loan_originated == 1)].compute(scheduler = 'threads')
df1819.reset_index(drop = True, inplace = True)

del dd_main

# Merge with ls_other
ls_other = pd.read_csv('Data/df_ls_other.csv')
             
# Merge dataframes            
df1819 = df1819.merge(ls_other[['fips','date','cert','ls_other']], how = 'inner', on = ['fips','date','cert'])

# Drop nans and transform loan costs
df1819 = df1819.dropna()
df1819.loc[:,'loan_costs'] = df1819.loc[:,'loan_costs'] / 1e3

# Demean data 
df_grouped_fips = df1819.groupby('fips')[columns + ['ls_other']].transform('mean')
df_grouped_lender = df1819.groupby('cert')[columns + ['ls_other']].transform('mean')
df_mean = df1819.mean()

## Transform data
df1819_trans = df1819[columns + ['ls_other']] - df_grouped_fips - df_grouped_lender + df_mean 

## Add MSA variable
df1819_trans['msamd'] = df1819.msamd

del df_grouped_fips, df_grouped_lender, df_mean, df1819

#------------------------------------------------------------
# First Stage
## Compute first stage results
results_fs = MultiDimensionalOLS().fit(df1819_trans[x0[0]], df1819_trans[['ls_other'] + x0[1:]],\
                cov_type = cov_type, cluster_cols = df1819_trans[cluster_cols_var])

## Transform to df
df_results_fs = results_fs.to_dataframe()

## Save to csv
df_results_fs.to_csv(file5.format('fs','csv'))

## Calculate LS hat and append to df
df1819_trans['ls_hat'] = results_fs.fittedvalues

## Add residuals to df_trans (for Hausman Wu test later on)
df1819_trans['ls_res'] = results_fs.resid

del results_fs

#------------------------------------------------------------
# Second Stage
## Compute second stage results
results_ss = MultiDimensionalOLS().fit(df1819_trans['loan_costs'], df1819_trans[['ls_hat'] + x0[1:]],\
                    cov_type = cov_type, cluster_cols = df1819_trans[cluster_cols_var])
    
## Transform to df
df_results_ss = results_ss.to_dataframe()
del results_ss

## Perform Hausman-Hu test
### Perform HW test 
results_ss_hw = MultiDimensionalOLS().fit(df1819_trans['loan_costs'], df1819_trans[['ls','ls_res'] + x0[1:]],\
                    cov_type = cov_type, cluster_cols = df1819_trans[cluster_cols_var])
    
df_results_ss_hw = results_ss_hw.to_dataframe() 
del results_ss_hw

### Ad hw-stat to df_results_ss
df_results_ss['hw_pval'] = df_results_ss_hw.loc['ls_res','p']

## Save to csv
df_results_ss.to_csv(file5.format('ss','csv'))

