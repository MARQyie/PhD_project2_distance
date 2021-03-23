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
           'ln_ta','ln_emp','ln_num_branch','cb','points']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)
    
# Interact msamd and date
dd_main = dd_main.assign(msat = dd_main.date.astype(str) + dd_main.msamd.astype(str))

# Transform to pandas df
df = dd_main[(dd_main.loan_originated == 1) & (dd_main.date.astype(int) == 2019)].compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)

del dd_main
#------------------------------------------------------------
# Add instruments
#------------------------------------------------------------

# (Geographic) diversification
df['div'] = df.groupby(['cert','fips','date']).fips.transform('count') / df.groupby(['cert','date']).fips.transform('count')

#------------------------------------------------------------
# Test 2sls
#------------------------------------------------------------
from linearmodels import IV2SLS

x_exo = ['lti','ln_loanamout','ln_appincome','subprime','lien','coapp','ln_ta',\
           'ln_emp', 'ln_num_branch','cb']

model1 = IV2SLS(df['log_min_distance'], df[x_exo], df['ls'], df['div'])
results1 = model1.fit()
