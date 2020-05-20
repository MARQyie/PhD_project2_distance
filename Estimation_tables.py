# Make estimation table

''' This script uses the estimation results to make nice estimation tables.
'''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import numpy as np
import pandas as pd

#------------------------------------------------------------
# Load the data
#------------------------------------------------------------

df_ols_ls = pd.read_csv('Results/Results_ols_full.csv', index_col = 0)
df_ols_int = pd.read_csv('Results/Results_olsint_full.csv', index_col = 0)
df_ols_lsint = pd.read_csv('Results/Results_olslsint_full.csv', index_col = 0)
df_fe_ls_full = pd.read_csv('Results/Results_fels_full.csv', index_col = 0)
df_fe_ls = pd.read_csv('Results/Results_fels_res.csv', index_col = 0)
df_fe_int = pd.read_csv('Results/Results_feint_res.csv', index_col = 0)
df_fe_lsint = pd.read_csv('Results/Results_felsint_res.csv', index_col = 0)