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

## Intercept
dd_main['intercept'] = 1

## Interaction terms
dd_main = dd_main.assign(local_ls = dd_main['local'] * dd_main['ls'])
dd_main = dd_main.assign(log_min_distance_ls = dd_main['log_min_distance'] * dd_main['ls'])

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
y = 'rate_spread'
x_local = ['ls','local','local_ls','lti','ltv','ln_loanamout',\
     'ln_appincome','int_only','balloon','mat','lien', 'coapp']

# Set other parameters
cov_type = 'clustered'
cluster_cols_var = 'msamd'
FE_cols_vars = ['fips','cert']
how = 'fips, cert'

# Set File names
file_local = 'Results/ratespread_results_local.{}'
file_local_year = 'Results/ratespread_results_local_{}.{}'

#------------------------------------------------------------
# Run Model
#------------------------------------------------------------
    
# Run
## Local
results_local = MultiDimensionalOLS().fit(df[y], df[x_local],\
                    cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                    transform_data = True, FE_cols = df[FE_cols_vars], how = how)

### Transform to pandas df
df_results_local = results_local.to_dataframe()

### Save to excel and csv
df_results_local.to_excel(file_local.format('xlsx'))
df_results_local.to_csv(file_local.format('csv'))

del results_local

## 2018 and 2019 separate
for year in range(2018,2019+1):
# Set df
    df_year = df[df.date == year]
    
    # Run
    results_yearly = MultiDimensionalOLS().fit(df_year[y], df_year[x_local],\
                        cov_type = cov_type, cluster_cols = df_year[cluster_cols_var],\
                        transform_data = True, FE_cols = df_year[FE_cols_vars], how = how)
    
   
    # Transform to pandas df
    df_results_yearly = results_yearly.to_dataframe()
    
    # Add count for hausman pval, msamd en cert
    #msamd = df.msamd.nunique()
    #cert = df.cert.nunique()
    #df_results_benchmark['haus'] = haus_pval
    #df_results_benchmark['msamd'] = msamd
    #df_results_benchmark['cert'] = cert
    
    # Save to excel and csv
    df_results_yearly.to_excel(file_local_year.format(year,'xlsx'))
    df_results_yearly.to_csv(file_local_year.format(year,'csv'))
    
    del results_yearly#, ols_benchmark


