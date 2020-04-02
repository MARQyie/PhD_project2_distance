# Baseline analysis MSAT
# Set working directory
import os
os.chdir(r'/home/p285325/WP2_distance/')

# Load packages
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS

#------------------------------------------------------------
# Load data
#------------------------------------------------------------

df = pd.read_csv('data_agg.csv')

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
df['dum_msat'] = pd.Categorical(df.index.get_level_values(1).year.astype(str) + df.msamd.astype(str))

''' OLD
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
               'mean_distance', 'dum_msat']
x = sm.add_constant(df[x_list]) 

''' OLD
x_msat_list = x_list + ['dum_msat_{}'.format(i) for i in range(dum_msat.shape[1])]
x_msat = sm.add_constant(df[x_msat_list])         
'''

#------------------------------------------------------------
# Run regression
#------------------------------------------------------------

# Run no dum
res = PanelOLS(y, x, entity_effects = True).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results_baseline_msat.txt", "w")
text_file.write(res.summary.as_text())
text_file.close()
