# Setup the HMDA dataframe
'''
    This script sets up the total dataframe for HMDA. This script also reads
    the HMDA lender file (avery file)
    
    The script does the following
    1. Reads in the HMDA dataframes for each year-end from 2010 to 2017
    2. Select the relevant variables
    3. Clean up the dataframe
    
    Concatenating all HMDA files might lead to an overload of memory usage. 
    Strategy: load all HMDA files separately, clean them and save them as 
    zipped csv (saves space).
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
## Lender File
path_lf = r'X:/My Documents/Data/Data_HMDA_lenderfile/'
file_lf = r'hmdpanel17.dta'
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['RSSD{}'.format(str(year)[2:4]) for year in range(start, end +1)]

## HMDA
path_hmda = r'X:/My Documents/Data/Data_HMDA/LAR/'
file_hmda = r'hmda_{}_nationwide_originated-records_codes.zip'
dtypes_col_hmda = {'state_code':'str', 'county_code':'str','msamd':'str',\
                   'census_tract_number':'str'}
na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA']

# Load data
## Lender file
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)

# HMDA
for year in range(start,end + 1):
     #Load the dataframe in a temporary frame
     df_chunk = pd.read_csv(path_hmda + file_hmda.format(year)), index_col = 0, chunksize = 1e6, \
                  na_values = na_values, dtype = dtypes_col_hmda)

    
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