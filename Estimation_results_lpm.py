# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    the linear probability model in Working paper 2
    
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
           'sex_1','sex_2','ls_ever','loan_originated','loan_type_2', 'loan_type_3', 'loan_type_4']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Add intercept and interaction term
## Intercept
dd_main['intercept'] = 1

## Interaction term
dd_main = dd_main.assign(log_min_distance_ls_ever = dd_main['log_min_distance'] * dd_main['ls_ever'])

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
y = 'loan_originated'
x = ['log_min_distance','log_min_distance_ls_ever','lti','ln_loanamout',\
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
file = 'Results/lpm_results_{}.{}'

#------------------------------------------------------------
# Run benchmark Model
#------------------------------------------------------------
# Loop over all years
for year in range(2004,2019+1):
# Set df
    df = dd_main[dd_main.date == year].compute(scheduler = 'threads')
    df.reset_index(drop = True, inplace = True)
    
    # Run
    results_benchmark = MultiDimensionalOLS().fit(df[y], df[x],\
                        cov_type = cov_type, cluster_cols = df[cluster_cols_var],\
                        transform_data = True, FE_cols = df[FE_cols_vars], how = how)
    
    # Transform to pandas df
    df_results_benchmark = results_benchmark.to_dataframe()
    
    # Save to excel and csv
    df_results_benchmark.to_excel(file.format(year,'xlsx'))
    df_results_benchmark.to_csv(file.format(year,'csv'))
    
    del df, results_benchmark#, ols_benchmark
