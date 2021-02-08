# Cleaning the df 
''' This script opens the full and aggregated datasets and checks them for outliers
    and inconsistencies 

    We use dask.dataframes to cut back on RAM-usage and utilize fast parallelization
    of loading the dataframes.
 '''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/')

# Load packages
## Data manipulation packages
import numpy as np
import pandas as pd

## Parallel functionality pandas via Dask
import dask.dataframe as dd 
import dask 
#dask.config.set({'temporary_directory': '/data/p285325/WP2_distance/'})
dask.config.set({'temporary_directory': 'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Data/'})

## Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale = 1.75)
import hvplot.dask
import hvplot

## Parallel processing
#import multiprocessing as mp # For parallelization
#from joblib import Parallel, delayed # For parallelization

## Parquet packages
#import pyarrow as pa
#import pyarrow.parquet as pq
#import fastparquet 

#------------------------------------------------------------
# Load data
#------------------------------------------------------------

# Main data
#------------------------------------------------------------
dd_main = dd.read_parquet(path = 'Data/data_main.parquet',\
                       engine = 'fastparquet')

# Aggregated data
#------------------------------------------------------------
#dd_agg = dd.read_parquet(path = 'Data/data_agg.parquet',\
#                       engine = 'fastparquet')

#------------------------------------------------------------
# Check for weird FIPS, MSAMD and CERT
#------------------------------------------------------------

''' NOTE
    FIPS == 00000 equals a county outside of a US state of D.C.
    FIPS == 99999 equals not in a place
    MSA/MD == 99999 equals county falls outside a Metropolitian statistical area
    '''

# Main data
#------------------------------------------------------------
'''Turn on if necessary
# Check nans
nans_main = dd_main.loc[:,['date','fips','msamd','cert']].isnull().sum().compute(scheduler = 'threads') # No missings

# FIPS
fips_zero_main = 0 in dd_main.fips
fips_99999_main = 99999 in dd_main.fips

# MSAMD
msamd_zero_main = 0 in dd_main.msamd
msamd_99999_main = 99999 in dd_main.msamd

# CERT
cert_zero_main = 0 in dd_main.cert
'''
''' NOTE: No weird FIPS, MSAMDs or CERTs '''

# Aggregated data
#------------------------------------------------------------
''' Nothing to remove '''

#------------------------------------------------------------
# Check for outliers in all variables; remove if necessary
#------------------------------------------------------------

# Main data
#------------------------------------------------------------

# Get a description of the necessary variables
vars_needed = [val for val in dd_main.columns.tolist() if val not in ['respondent_id','agency_code','date', 'cert', 'msamd']]
dd_main_desc = dd_main[vars_needed].describe().compute()
dd_main_desc.T.to_excel('Data/Describe_data_main.xlsx')

'''NOTE:    ln_emp = 0
            min rate_spread = -9999997 
            max loan_term = 8888
            max ltv 1056818
    
    DO:
            Check and remove outliers for rate_spread, loan_term and ltv
            Take logs of the following: num_branch, pop_area, rate_spread, loan_costs,
                points, ori_charges, lender_credit, loan_term,
            divide by 1e2: ltv
    '''

# Check and remove outliers
## Check outliers
vars_needed = ['rate_spread','loan_term','ltv']

min_rate_spread = dd_main[vars_needed[0]].nsmallest(n = 20).compute() # 8 rows are lower than -150 --> remove
max_rate_spread = dd_main[vars_needed[0]].nlargest(n = 20).compute() # Many rows with 1111 --> remove
max_loan_term = dd_main[vars_needed[1]].nlargest(n = 40).compute() # Many 8888s --> remove
max_ltv = dd_main[vars_needed[2]].nlargest(n = 40).compute() # 1e2 = 100%. The remaining high values are high, but there is no clear cut-off. We assume that all LTV >100% are implausible

''' NOTE: Bar ln_emp == 0, these missings only pose a problem in the third analysis.
    Remove then!
    '''
## Remove outliers 
dd_main = dd_main[dd_main.ln_emp > 0]

''' Copy when necessary: 
dd_main = dd_main[(dd_main.rate_spread > -150) & (dd_main.rate_spread < 1111) &\
                  (dd_main.loan_term < 8888) & (dd_main.ltv < 88)] '''

# Take logs
dd_main = dd_main.assign(ln_num_branch = np.log(1 + dd_main.num_branch))
dd_main = dd_main.assign(ln_pop_area = np.log(1 + dd_main.pop_area))

''' NOTE: Take the logs of the rest of the in the third analysis.
    It's easier to first remove outliers and then take the log'
    '''

# divide by 1e2
dd_main['ltv'] = dd_main.ltv.divide(1e2)

# Save data
dd_main.to_parquet(path = 'Data/data_main_clean.parquet',\
                   engine = 'fastparquet',\
                   compression = 'snappy',\
                   partition_on = ['date'])

# Aggregated data
#------------------------------------------------------------

#------------------------------------------------------------
# Check number of originated and not-originated loans per year
#------------------------------------------------------------

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet')

# Main data
#------------------------------------------------------------

# Setup the data for plotting
df_originations = dd_main.groupby(['date','loan_originated']).log_min_distance.count().compute()
df_originations = df_originations.unstack(level = 1)
df_originations.index = df_originations.index.astype(int)
df_originations = df_originations.sort_index(ascending = True)

# plot
labels = ['Not Originated', 'Originated']
colors = ['k','dimgrey']

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Number of Total Applications')
ax.stackplot(df_originations.index, df_originations[0], df_originations[1],\
             colors = colors, labels = labels)
ax.axvspan(2007,2009, alpha = 0.5, color = 'lightgray')
ax.set_xlim(2004, 2019)
ax.legend()
plt.tight_layout()

fig.savefig('Figures/Number_originations.png')

# Distance
df_distance = dd_main.groupby(['date','loan_originated','ls']).min_distance.mean().compute()
df_distance = df_distance.unstack(level = [1,2])
df_distance.index = df_distance.index.astype(int)
df_distance = df_distance.sort_index(ascending = True)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Mean lending distance (in km)')
ax.plot(df_distance.index, df_distance.iloc[:,0], color = 'black', linestyle = '-', label = 'Not Originated')
ax.plot(df_distance.index, df_distance.iloc[:,2], color = 'black', linestyle = '--', label = 'Originated, Not Sold')
ax.plot(df_distance.index, df_distance.iloc[:,3], color = 'black', linestyle = '-.', label = 'Originated, Sold')
ax.axvspan(2007,2009, alpha = 0.5, color = 'lightgray')
ax.set_xlim(2004, 2019)
ax.legend()
plt.tight_layout()

fig.savefig('Figures/Mean_distance_yearly_split.png')

# Get mean distance trhough the years
mean_distance = dd_main.groupby('date').min_distance.mean().compute()
mean_distance.index = mean_distance.index.astype(int)
mean_distance = mean_distance.sort_index(ascending = True)

mean_distance.iloc[-1] - mean_distance.iloc[6]
(mean_distance.iloc[-1] - mean_distance.iloc[6]) / mean_distance.iloc[6]