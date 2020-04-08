# Baseline analysis MSAT
# Set working directory
import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/')

# Load packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels import PanelOLS

#------------------------------------------------------------
# Load data
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg.csv')


# Set multiindex
## First combine date and msamd
df['date'] = ['{}-12-31'.format(entry) for entry in df.date]
df['date'] = pd.to_datetime(df.date)
df.set_index(['cert','msamd','date'],inplace=True) 

# Drop na
df.dropna(inplace = True)

#------------------------------------------------------------
# Transform the data
#------------------------------------------------------------

# Select necessary variables
vars_trans = ['log_min_distance', 'ls_num', 'lti', 'ln_loanamout', 'ln_appincome',\
              'subprime', 'secured', 'cb', 'ln_ta', 'ln_emp', 'num_branch',\
              'ln_pop', 'density', 'hhi', 'ln_mfi', 'mean_distance']

# Group on msamd and time
msat = df.groupby([df.index.get_level_values(1),df.index.get_level_values(2)])[vars_trans].transform('mean')

# Group on msamd and bank
msabank = df.groupby([df.index.get_level_values(0),df.index.get_level_values(1)])[vars_trans].transform('mean')

# Group on time
t  = df.groupby([df.index.get_level_values(2)])[vars_trans].transform('mean')

# Group on msa
msa  = df.groupby([df.index.get_level_values(1)])[vars_trans].transform('mean')

# Group on bank
bank = df.groupby([df.index.get_level_values(0)])[vars_trans].transform('mean')

# Total mean
mean = df[vars_trans].mean()

# Make a transformed dfs
df_msat = df[vars_trans] - msat - bank + mean
df_msabank = df[vars_trans] - msabank - t + mean
df_bank = df[vars_trans] - bank - t + mean

# Drop level from multi index
df_msat = df_msat.droplevel(level = 1)
df_msabank = df_msabank.droplevel(level = 1)
df_bank = df_bank.droplevel(level = 1)

# Add original index from df to the frame as colum
df_msat['cert'], df_msat['msamd'], df_msat['date'] = df.index.get_level_values(0), df.index.get_level_values(1), df.index.get_level_values(2)


#------------------------------------------------------------
# Set Variables
#------------------------------------------------------------

# Set vars
## Y
y = 'log_min_distance'

## X
x_list = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'secured', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'ln_pop', 'density', 'hhi', 'ln_mfi',\
               'mean_distance']
x = ' + '.join(x_list)

#------------------------------------------------------------
# Run regressions
#------------------------------------------------------------

# Run Bank + msat dummies
res_msat = PanelOLS.from_formula('{} ~ {}'.format(y,x), data = df_msat).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_msat.txt", "w")
text_file.write(res_msat.summary.as_text())
text_file.close()

# Run Bankmsa + t dummies
res_msabank = PanelOLS.from_formula('{} ~ {}'.format(y,x), data = df_msabank).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_msabank.txt", "w")
text_file.write(res_msabank.summary.as_text())
text_file.close()

# Run Bank + t dummies
res_bank = PanelOLS.from_formula('{} ~ {}'.format(y,x), data = df_bank).fit(cov_type = 'clustered', cluster_entity = True)

## Save output to txt
text_file = open("Results/Results_baseline_bank.txt", "w")
text_file.write(res_bank.summary.as_text())
text_file.close()
