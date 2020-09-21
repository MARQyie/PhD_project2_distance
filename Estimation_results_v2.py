# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    Working paper 2
    
    The following model is estimated:
        1) FE: cert and date*msamd -- loan sales + controls -- Full sample
    
    Full sample: 2010 -- 2019
    
    Furthermore we do a Hausman specification test (OLS vs. FE), and test 
    the variance inflation factor
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
df['intercept'] = 1
num_cores = mp.cpu_count()

#------------------------------------------------------------
# Hausman specification test
#------------------------------------------------------------

def HausmanSpecificationTest(b0, b1, v0, v1):
    ''' This method implements the Hausman specification test, based on the 
        formula:
            
            H = (b0 - b1)'(v0 - v1)^-1(b0 - b1),
            
        where b0, v0 are the parameter estimates, variance of the consistent model (FE),
        and b1 are the parameter estimates, variance of the efficient model (POLS/RE)
        
        The degrees of freedom for the statistic is the rank of the difference
        in the variance matrices. 
        '''
        
    # Prelims
    b_diff = b0 - b1
    v_diff = v0 - v1
    v_inv = np.linalg.inv(v_diff)
    dof = np.linalg.matrix_rank(v_diff)
        
    # Calculate the H-stat
    H = b_diff.T @ v_inv @ b_diff
        
    # Calculate significance
    pval = stats.chi2.sf(H, dof)
        
    return H, pval, dof
  
def VIFTest(df, x):
    
    # import vif and parallel packages
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    
    # Set data
    data = df[['intercept'] + x]
    
    # Set prelim
    column_shape = data.shape[1]
    
    # Run
    if __name__ == '__main__':
        vif_list =  Parallel(n_jobs = num_cores, prefer="threads")(delayed(vif)(data.values, i) for i in range(column_shape))
        
    # Make pandas series and sort
    df_vif = pd.Series(vif_list, index = data.columns).sort_values(ascending = False)
    
    return df_vif

#------------------------------------------------------------
# Set variables
#------------------------------------------------------------

# Set variables
y = 'log_min_distance'
x = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'ln_ta', 'ln_emp',  'ln_num_branch',  'cb']

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['msamd','date','cert']
how = 'msamd x date, cert'

msamd = df.msamd.nunique()
cert = df.cert.nunique()

# Set File names
## OLS
file = 'Results/Benchmark_results.{}'
file_nc = 'Results/Benchmark_results_nc.{}'
file_nlen = 'Results/Benchmark_results_nlen.{}'

#------------------------------------------------------------
# Run benchmark Model
#------------------------------------------------------------

# Run
## NOTE: remove msat-invariant variables 
results_benchmark = MultiDimensionalOLS().fit(df[y], df[x],\
                    cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                    transform_data = True, FE_cols = df[FE_cols_vars], how = how)

## OLS Benchmark
### First demean the data (because we cannot add a constant)
df_demean = df - df.mean()

### Run benchmark
ols_benchmark = MultiDimensionalOLS().fit(df_demean[y], df_demean[x],\
                cov_type = cov_type, cluster_cols = df[cluster_cols_var])

# Hausman test
haus_H, haus_pval, haus_dof = HausmanSpecificationTest(results_benchmark.params,\
                              ols_benchmark.params, results_benchmark.cov,\
                              ols_benchmark.cov)

# Transform to pandas df
df_results_benchmark = results_benchmark.to_dataframe()

# Add count for hausman pval, msamd en cert
df_results_benchmark['haus'] = haus_pval
df_results_benchmark['msamd'] = msamd
df_results_benchmark['cert'] = cert

# Save to excel and csv
df_results_benchmark.to_excel(file.format('xlsx'))
df_results_benchmark.to_csv(file.format('csv'))

test_res = MultiDimensionalOLS().fit(df[df.date < 2018][y], df[df.date < 2018][x],\
                    cov_type = cov_type, cluster_cols = df[df.date < 2018][cluster_cols_var],\
                    transform_data = True, FE_cols = df[df.date < 2018][FE_cols_vars], how = how)
'''OLD
#------------------------------------------------------------
# Run benchmark Model -- no control
#------------------------------------------------------------

# Run
## NOTE: remove msat-invariant variables 
results_benchmark_nc = MultiDimensionalOLS().fit(df[y], df['ls_num'],\
                    cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                    transform_data = True, FE_cols = df[FE_cols_vars], how = how)

ols_benchmark_nc = MultiDimensionalOLS().fit(df[y], df['ls_num'],\
                cov_type = cov_type, cluster_cols = df[cluster_cols_var])

# Hausman test
haus_H_nc, haus_pval_nc, haus_dof_nc = HausmanSpecificationTest(results_benchmark_nc.params,\
                              ols_benchmark_nc.params, results_benchmark_nc.cov,\
                              ols_benchmark_nc.cov)

# Transform to pandas df
df_results_benchmark_nc = results_benchmark_nc.to_dataframe()

# Add count for hausman pval, msamd en cert
df_results_benchmark_nc['haus'] = haus_pval_nc
df_results_benchmark_nc['msamd'] = msamd
df_results_benchmark_nc['cert'] = cert

# Save to excel and csv
df_results_benchmark_nc.to_excel(file_nc.format('xlsx'))
df_results_benchmark_nc.to_csv(file_nc.format('csv'))

#------------------------------------------------------------
# Run benchmark Model -- no lender controls
#------------------------------------------------------------

# Run
## NOTE: remove msat-invariant variables 
results_benchmark_nlen = MultiDimensionalOLS().fit(df[y], df[['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime']],\
                    cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                    transform_data = True, FE_cols = df[FE_cols_vars], how = how)

ols_benchmark_nlen = MultiDimensionalOLS().fit(df[y], df[['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime']],\
                cov_type = cov_type, cluster_cols = df[cluster_cols_var])

# Hausman test
haus_H_nlen, haus_pval_nlen, haus_dof_nlen = HausmanSpecificationTest(results_benchmark_nlen.params,\
                              ols_benchmark_nlen.params, results_benchmark_nlen.cov,\
                              ols_benchmark_nlen.cov)

# Transform to pandas df
df_results_benchmark_nlen = results_benchmark_nlen.to_dataframe()

# Add count for hausman pval, msamd en cert
df_results_benchmark_nlen['haus'] = haus_pval_nlen
df_results_benchmark_nlen['msamd'] = msamd
df_results_benchmark_nlen['cert'] = cert

# Save to excel and csv
df_results_benchmark_nlen.to_excel(file_nlen.format('xlsx'))
df_results_benchmark_nlen.to_csv(file_nlen.format('csv'))
'''