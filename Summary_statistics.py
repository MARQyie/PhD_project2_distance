# Summary statistics
''' This script returns a table with summary statistics for WP2 '''

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# prelims
vars_needed = ['log_min_distance', 'log_min_distance_cdd','ls_num', 'perc_broadband',\
               'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'ln_ta', 'ln_emp',\
               'ln_num_branch', 'cb', 'ln_density', 'ln_pop_area', 'ln_mfi', 'hhi']
all_vars = ['msamd','cert','date']

# Subset dfs
df_full = df[all_vars + vars_needed]
df_reduced = df.loc[df.date >= 2013,all_vars + vars_needed]

#------------------------------------------------------------
# Make table
#------------------------------------------------------------

# Get summary statistics
ss_full = df_full[vars_needed].describe().T[['mean','std']]
ss_reduced = df_reduced[vars_needed].describe().T[['mean','std']]

# Add extra stats
## MSA-Lender-Years
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.shape[0], 'std':np.nan}, index = ['MSA-lender-years']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.shape[0], 'std':np.nan}, index = ['MSA-lender-years']))

## MSAs
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.msamd.nunique(), 'std':np.nan}, index = ['MSA']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.msamd.nunique(), 'std':np.nan}, index = ['MSA']))

## Lenders
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.cert.nunique(), 'std':np.nan}, index = ['Lender']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.cert.nunique(), 'std':np.nan}, index = ['Lender']))

## Years
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.date.nunique(), 'std':np.nan}, index = ['Years']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.date.nunique(), 'std':np.nan}, index = ['Years']))

# Change name of columns
cols_full = [('2010--2017','Mean'), ('2010--2017','S.E.')]
cols_reduced = [('2013--2017','Mean'), ('2013--2017','S.E.')]

ss_full.columns = cols_full
ss_reduced.columns = cols_reduced

# Concat ss_full and ss_reduced
ss_tot = pd.concat([ss_full, ss_reduced], axis = 1)

# Change index
index_col = ['Distance (pop. weighted)', 'Distance (CDD)','Loan Sales', 'Internet',\
               'LTI', 'Loan Value', 'Income', 'subprime', 'Size', 'Employees',\
               'Branches', 'Bank', 'Density', 'Population', 'MFI', 'HHI','MSA-lender-years',
               'MSAs','Lenders','Years']

ss_tot.index = index_col

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss_tot.to_excel('Tables/Summary_statistics.xlsx')

