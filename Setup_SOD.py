# Setup the Summary of Deposits dataframe
'''
    This script sets up the total dataframe for SODs.
    
    The script does the following
    1. Reads in the SOD dataframes for each year-end from 2010 to 2017
    2. Select the relevant variables
    3. Clean up the dataframe
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d'

#------------------------------------------------------------
# Set Parameters
#------------------------------------------------------------
start = 2010
end = 2017

#------------------------------------------------------------
# Load the data in one df
#------------------------------------------------------------

# Prelims
path_sod = r'X:/My Documents/Data/Data_SOD/'
file_sod = r'ALL_{}.csv'
vars_sod = ['CERT','MSABR','RSSDID','STCNTYBR','SIMS_LATITUDE','SIMS_LONGITUDE',\
            'STALPBR','BRSERTYP','UNINUMBR','DEPSUMBR']
dtypes_col_sod = {'MSABR':'str','STCNTYBR':'str'}

# Load data
list_df_sod = []

for year in range(start,end + 1):
    # Load the dataframe in a temporary frame
    df_sod_load = pd.read_csv(path_sod + file_sod.format(year),\
                              usecols = vars_sod, dtype = dtypes_col_sod)
    # Correct STCNTYBR
    df_sod_load['STCNTYBR'] = df_sod_load.STCNTYBR.str.zfill(5)
    
    # Fix DEPSUMBR (reads in as object, but should be an int)
    df_sod_load['DEPSUMBR'] = df_sod_load.DEPSUMBR.str.replace(',','').astype(int)
    
    # Force lowercase column names
    df_sod_load.columns = map(str.lower, df_sod_load.columns)
    
    # Rename columns
    df_sod_load.rename(columns = {'stcntybr':'fips'}, inplace = True)
    
    # Add a date var
    df_sod_load['date'] = year
    
    # Append to list
    list_df_sod.append(df_sod_load)

# Make a dataframe from the list
df_sod = pd.concat(list_df_sod)

#------------------------------------------------------------
# Clean the dataframe
#------------------------------------------------------------

# Drop all missings in cert, numemp, asset, cb
df_sdi.dropna(subset = ['cert', 'fips', 'sims_latitude', 'sims_longitude', 'depsumbr'], inplace = True)

# Restrict deposits
df_sod = df_sod[(df_sod.depsumbr >= 0.0)]

# Check for outliers in assets and numemp
## Box plots
dict_var_names = {'sims_latitude':'Log Total Assets',\
                  'sims_longitude':'Log Total Employees',\
                  'depsumbr':'Branch Deposits'}
vars_needed = ['sims_latitude', 'sims_longitude', 'depsumbr']

for var in vars_needed:
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))
    
    data = df_sdi[var]
    ax.boxplot(data)
    
    plt.xticks([1], ['Full Sample SOD'])
    
    fig.savefig('Figures\Box_plots\Box_SOD_{}.png'.format(var)) # MAKE FOLDER!!!
    plt.clf()

## remove outliers
#TODO

#------------------------------------------------------------
# Save the dataframe
#------------------------------------------------------------
    
df_sdi.to_csv('Data/df_sod_wp2.csv')