# Setup the Satistics on Deposit institutions dataframe
'''
    This script sets up the total dataframe for statistics on deposit insitutions.

    The script does the following
    1. Reads in the SDI dataframes for each year-end from 2004 to 2019;
       Reads the table Assets and Liabilities and Bank Assets Sold and Securitized 
    2. Select the relevant variables
    3. Adds variables to the dataframe
    4. Clean up the dataframe
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

#------------------------------------------------------------
# Set Parameters
#------------------------------------------------------------

start = 2004
end = 2019

#------------------------------------------------------------
# Load the data in one df
#------------------------------------------------------------

# Prelims
path_sdi = r'D:/RUG/Data/Data_SDI/{}/'
file_sdi = r'All_Reports_{}1231_Assets and Liabilities.csv'
file_sec = r'All_Reports_{}1231_Bank Assets Sold and Securitized.csv'
file_sec_def = r'D:/RUG/Data/Data_SDI/Definitions/Bank Assets Sold and Securitized.csv'
vars_sdi = ['cert','fed_rssd','numemp','asset','cb']

# Load definition file securitization and loan sales
df_sec_def = pd.read_csv(file_sec_def, header = 1, index_col = 0)

## Get variable names for all sec and loan sales
names_sec_ls = df_sec_def.loc[df_sec_def.Variable.str.contains('SZLN|ASDR'),'Variable'].tolist()

# Load data
list_df_sdi = []

for year in range(start,end + 1):
    # Load the dataframe in a temporary frame
    df_sdi_load = pd.read_csv(path_sdi.format(year) + file_sdi.format(year), usecols = vars_sdi, encoding = 'ISO-8859-1')
    df_sec_load = pd.read_csv(path_sdi.format(year) + file_sec.format(year), encoding = 'ISO-8859-1')
    
    # Add a date var
    df_sdi_load['date'] = year
    
    # Add sec and ls dummies
    df_sdi_load['sec_sdi'] = (df_sec_load.loc[:,names_sec_ls[:-2]].sum(axis = 1) > 0)* 1
    df_sdi_load['ls_sdi'] = (df_sec_load.loc[:,names_sec_ls[-2:]].sum(axis = 1) > 0)* 1
    df_sdi_load['lssec_sdi'] = (df_sec_load.loc[:,names_sec_ls].sum(axis = 1) > 0)* 1

    # Append to list
    list_df_sdi.append(df_sdi_load)

# Make a dataframe from the list
df_sdi = pd.concat(list_df_sdi)

#------------------------------------------------------------
# Add variables to the dataframe
#------------------------------------------------------------

df_sdi['ln_ta'] = np.log(df_sdi.asset)
df_sdi['ln_emp'] = np.log(1 + df_sdi.numemp)

# Add boolean if bank has ever sold or securitized loans in the past
df_grouped = df_sdi.groupby('cert')
list_grouped = []

for name, group in df_grouped:
    for year in group.date.tolist():
        # Get boolean
        ## sec
        group.loc[group.date == year, 'sec_ever'] = (group.loc[group.date <= year,'sec_sdi'].sum() > 0) * 1 
        
        ## ls
        group.loc[group.date == year, 'ls_ever'] = (group.loc[group.date <= year,'ls_sdi'].sum() > 0) * 1
        
        ## lssec
        group.loc[group.date == year, 'lssec_ever'] = (group.loc[group.date <= year,'lssec_sdi'].sum() > 0) * 1
        
        # append to list
    list_grouped.append(group)

# Make new df
df_sdi = pd.concat(list_grouped)
#------------------------------------------------------------
# Clean the dataframe
#------------------------------------------------------------

# Drop all missings in cert, numemp, asset, cb
df_sdi.dropna(subset = ['cert', 'numemp', 'asset','cb'], inplace = True)

# Restrict assets and numemp to be positive
df_sdi = df_sdi[(df_sdi.asset >= 0.0) | (df_sdi.numemp >= 0.0)]

# Check for outliers in assets and numemp
## Box plots
dict_var_names = {'ln_ta':'Log Total Assets',\
                  'ln_emp':'Log Total Employees'}
vars_needed = ['ln_ta', 'ln_emp']

for var in vars_needed:
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))

    data = df_sdi[var]
    ax.boxplot(data)

    plt.xticks([1], ['Full Sample SDI'])

    fig.savefig('Figures\Box_plots\Box_SDI_{}.png'.format(var))
    plt.clf()

## remove outliers
# NOTE: No outliers to remove

#------------------------------------------------------------
# Save the dataframe
#------------------------------------------------------------
df_sdi.to_csv('Data/df_sdi_wp2.csv')