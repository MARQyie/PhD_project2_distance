# Pilot regression
# Set working directory
import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np
import statsmodels.api as sm

#------------------------------------------------------------
# Load data
#------------------------------------------------------------
# Load data 
df = pd.read_csv('Data/df_agg_2010_test.csv', index_col = 0)

# Drop missings
df.dropna(inplace = True)

#------------------------------------------------------------
# Setup the data for the regression
#------------------------------------------------------------
# Make dummies for banks/thrifts
dum_bank = pd.get_dummies(df.cert, prefix = 'dum_bank')

# Make dummies for MSA 
dum_msa = pd.get_dummies(df.msamd, prefix = 'dum_msa')

#------------------------------------------------------------
# Do the regression
#------------------------------------------------------------
# Setup the regression columns
y = df['distance']

list_x_vars = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'secured', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'ln_pop', 'density', 'hhi', 'ln_mfi',\
               'ave_distance'] #dum_min is only 1 in this subsample
x_nodum = sm.add_constant(df[list_x_vars])
x_dum = sm.add_constant(pd.concat([df[list_x_vars], dum_bank.iloc[:,:-1], dum_msa.iloc[:,:-1]], axis = 1))

# Regress y on x (without dummies)
res_nodum = sm.OLS(y,x_nodum).fit(cov_type='HC3')
print(res_nodum.summary())

# Regress y on x (with dummies)
res_nodum = sm.OLS(y,x_dum).fit()

#------------------------------------------------------------
# Estimate Correlation Matrix
#------------------------------------------------------------
# Corr matrix
data_corr = pd.concat([df['distance'], df[list_x_vars]], axis = 1)
corr_matrix = data_corr.corr(method = 'spearman')
