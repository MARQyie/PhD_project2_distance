# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-lender-level dataset per year
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS

# Import a method to make nice tables
from Code_docs.help_functions.summary3 import summary_col

#------------------------------------------------------------
# Load data
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg.csv')

# Set multiindex
## First combine date and msamd
df['date'] = ['{}-12-31'.format(entry) for entry in df.date]
df['date'] = pd.to_datetime(df.date)
df.set_index(['cert','date'],inplace=True)

# Drop na
df.dropna(inplace = True)

#------------------------------------------------------------
# Make dummies and add to df
#------------------------------------------------------------

# Make dummies 
## Time dummies
df['dum_t'] = pd.Categorical(df.index.get_level_values(1))

''' NOT NECESSARY YET
## MSA-Time dummies
## Get all unique dummies
dum_msat = pd.get_dummies(df.index.get_level_values(1).year.astype(str) + df.msamd.astype(str),prefix = 'd')

## Drop all dummies from 2010
dum_msat.drop(labels = dum_msat.columns[dum_msat.columns.str.contains('d_2010')],axis = 1, inplace = True)

## Rename dummies
dum_msat.columns = ['dum_msat_{}'.format(i) for i in range(dum_msat.shape[1])]

## Add dummies to the frame
df = pd.concat([df, dum_msat], axis = 1)
'''
#------------------------------------------------------------
# Set Variables
#------------------------------------------------------------

# Set vars
## Y
y = df['log_min_distance'] 

## X
x_list = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'secured', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'ln_pop', 'density', 'hhi', 'ln_mfi',\
               'mean_distance']
x = sm.add_constant(df[x_list]) 

'''
x_msat_list = x_list + ['dum_msat_{}'.format(i) for i in range(dum_msat.shape[1])]
x_msat = sm.add_constant(df[x_msat_list])         
'''
#------------------------------------------------------------
# Run regression
#------------------------------------------------------------

# Run no dum
res_nd = PanelOLS(y, x, entity_effects = True).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_nodummy.txt", "w")
text_file.write(res_nd.summary.as_text())
text_file.close()

# Run dum_t
res_t = PanelOLS(y, x, entity_effects = True, time_effects = True).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_t.txt", "w")
text_file.write(res_t.summary.as_text())
text_file.close()

'''
# Run dum_msat
res_msat = PanelOLS(y, x_msat, entity_effects = True).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_msat.txt", "w")
text_file.write(res_msat.summary.as_text())
text_file.close()
'''
#------------------------------------------------------------
# Run test FD model
#------------------------------------------------------------

# Subset data
df_fd = df.groupby([df.index.get_level_values(0),'msamd'])[['log_min_distance'] + x_list].diff(periods = 1).dropna()

## add time dummy
df_fd['dum_t'] = pd.Categorical(df_fd.index.get_level_values(1))

# Set vars (with time dummies)
y_fd = df_fd['log_min_distance'] 
x_fd_list = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'secured', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'ln_pop', 'density', 'hhi', 'ln_mfi',\
               'mean_distance', 'dum_t']
x_fd = sm.add_constant(df_fd[x_fd_list])  

# Run no dum
res_fd = PanelOLS(y_fd, x_fd).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_fd.txt", "w")
text_file.write(res_fd.summary.as_text())
text_file.close() 

#------------------------------------------------------------
# Make nice table
#------------------------------------------------------------

#TODO

#------------------------------------------------------------
# Save table
#------------------------------------------------------------

#TODO
