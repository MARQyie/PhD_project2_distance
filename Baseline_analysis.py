# Baseline analysis 
''' This script performs the first baseline analyses '''

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
from linearmodels import PanelOLS

# Import a method to make nice tables
from Code_docs.Help_functions.summary3 import summary_col

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

''' OLD
## MSA-Time dummies
## Get all unique dummies
dum_msat = pd.get_dummies(df.index.get_level_values(1).year.astype(str) + df.msamd.astype(str),prefix = 'd')

## Drop all dummies from 2010
dum_msat.drop(labels = dum_msat.columns[dum_msat.columns.str.contains('d_2010')],axis = 1, inplace = True)

## Rename dummies
dum_msat.columns = ['dum_msat_{}'.format(i) for i in range(dum_msat.shape[1])]

## Add dummies to the frame
df[dum_msat.columns.tolist()] = dum_msat
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
x = df[x_list]

'''
x_msat_list = x_list + ['dum_msat_{}'.format(i) for i in range(dum_msat.shape[1])]
x_msat = sm.add_constant(df[x_msat_list])         
'''
#------------------------------------------------------------
# Run regression
#------------------------------------------------------------

# Run no dum
res_nd = PanelOLS(y, x).fit(cov_type = 'clustered', cluster_entity = True)

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

# Manual test of FE
y_hat = y.groupby(y.index.get_level_values(0)).apply(lambda df: df - df.mean())
x_hat = x.groupby(x.index.get_level_values(0)).apply(lambda df: df - df.mean())
dum_t = pd.get_dummies(x_hat.index.get_level_values(1), drop_first = True)


df[dummies.columns] = dummies

x_man_list = x_list + dummies.columns.tolist()
x_man = df[x_man_list]

import scipy as sp
beta = sp.linalg.inv(x_man.T @ x_man) @ (x_man.T @ y)

res_man = PanelOLS(y, x_man).fit(cov_type = 'clustered', cluster_entity = True)
text_file = open("Results/Results_baseline_manualcheck.txt", "w")
text_file.write(res_man.summary.as_text())
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
df_fd = df.groupby(df.index.get_level_values(0))[['log_min_distance'] + x_list].diff(periods = 1).dropna()

## add time dummy
df_fd['dum_t'] = pd.Categorical(df_fd.index.get_level_values(1))

# Set vars (with time dummies)
y_fd = df_fd['log_min_distance'] 
x_fd_list = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'secured', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'ln_pop', 'density', 'hhi', 'ln_mfi',\
               'mean_distance', 'dum_t']
x_fd = df_fd[x_fd_list]  

# Run no dum
res_fd = PanelOLS(y_fd, x_fd).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_fd.txt", "w")
text_file.write(res_fd.summary.as_text())
text_file.close() 

#------------------------------------------------------------
# Make nice table
#------------------------------------------------------------

# Load functions
def adjR2(r2, n, k): 
    ''' Calculates the adj. R2'''
    adj_r2 = 1 - (1 - r2) * ((n - 1)/(n - (k + 1)))
    return(round(adj_r2, 3))

vecAdjR2 = np.vectorize(adjR2)  

# Call summary col
res_table = summary_col([res_nd, res_t, res_fd], show = 'se', regressor_order = x_fd_list)

# Calculate Adj. R2
r2 = res_table.tables[2].iloc[5,:].astype(float)
n = res_table.tables[2].iloc[4,:].astype(float)
k = [mod.model.exog.shape[0] for mod in [res_nd, res_t, res_fd]]

adj_r2 = vecAdjR2(r2,n,k)

# Edit layout
## Make dict that contains all variable names
dict_var = {'':'',
            'const':'Constant',
            'ls_num':'Loan Sales',
            'lti':'Loan-to-Income',
            'ln_loanamout':'Loan Amount',
            'ln_appincome':'Appl. Income',
            'subprime':'Subprime',
            'secured':'Secured',
            'cb':'Bank Dummy',
            'ln_ta':'Size',
            'ln_emp':'Num. Employees',
            'num_branch':'Num. Branches',
            'ln_pop':'Population',
            'density':'Density',
            'hhi':'HHI',
            'ln_mfi':'Median Fam. Income',
            'mean_distance': 'Mean Distance',
            'No. Observations:':'N',
            'R-squared:':'$R^2$',
            '$Adj. R^2$':'$Adj. R^2$',
            'N Banks':'N Banks',
            'Fixed Effects':'Fixed Effects'}

## Change Lower part of the table
lower_table = res_table.tables[2].iloc[[4],:]
lower_table.loc[-1] = [df.index.get_level_values(0).nunique(),\
               df.index.get_level_values(0).nunique(),\
               df_fd.index.get_level_values(0).nunique()]
lower_table.rename({-1:'N Banks'}, axis = 'index', inplace = True)
lower_table.loc[-1] = adj_r2
lower_table.rename({-1:'$Adj. R^2$'}, axis = 'index', inplace = True)
lower_table.loc[-1] = ['None','Bank + Time','Time']
lower_table.rename({-1:'Fixed Effects'}, axis = 'index', inplace = True)

### Add to the table
res_table.tables[2] = lower_table

## Delete dummies from main table
main_table = res_table.tables[1].iloc[:-np.sum((res_table.tables[1].index.str.contains('dum') * 1))*2,:]
res_table.tables[1] = main_table

## Make new table
res_table_pretty = pd.concat(res_table.tables[1:3])

## Change column names
res_table_pretty.columns = ['FE','FE','FD']

## Change index 
res_table_pretty.index = [dict_var[key] for key in res_table_pretty.index]

#------------------------------------------------------------
# Save table
#------------------------------------------------------------

res_table_pretty.to_excel('Results\Baseline_results.xlsx')
