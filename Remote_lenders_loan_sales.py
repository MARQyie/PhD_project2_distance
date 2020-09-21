# Estimation results

''' This script tests whether whether remote lenders sell a higher fraction of 
    their residential loans
    '''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')

#------------------------------------------------------------
# Group the lenders based on lending distance
#------------------------------------------------------------

#------------------------------------------------------------
# Quartile grouping

# Group the data
df['quart_distance'] = pd.qcut(df.log_min_distance, q = 4, labels = False)
df_grouped_quart = df.groupby('quart_distance')[['log_min_distance', 'ls_num',\
                             'ls_gse_num', 'ls_priv_num', 'sec_num']]

# Describe groups
desc_quart = []
for i in range(4):
    desc_quart.append(df_grouped_quart.get_group(i).describe())
    print(desc_quart[i])
    
# Get differences in mean for all ls variables
diffmean_abs_quart = df_grouped_quart.get_group(3)[['ls_num', 'ls_gse_num', 'ls_priv_num', 'sec_num']].mean() -\
                 df_grouped_quart.get_group(0)[['ls_num', 'ls_gse_num', 'ls_priv_num', 'sec_num']].mean()
                 
diffmean_perc_quart = diffmean_abs_quart / \
                      ((df_grouped_quart.get_group(3)[['ls_num', 'ls_gse_num', 'ls_priv_num', 'sec_num']].mean() +\
                        df_grouped_quart.get_group(0)[['ls_num', 'ls_gse_num', 'ls_priv_num', 'sec_num']].mean()) / 2)
    
# Welch's t-test for q1 and q4
wstat_quart, wpval_quart = stats.ttest_ind(df_grouped_quart.get_group(0)[['ls_num', 'ls_gse_num', 'ls_priv_num', 'sec_num']],\
                df_grouped_quart.get_group(3)[['ls_num', 'ls_gse_num', 'ls_priv_num', 'sec_num']],\
                equal_var = False)

''' NOTE: The simpel Welch's test indicate the for all 4 ls variables the means are
    significantly different. This means that Q4 sells more in total and for GSE and 
    private. It securitizes less -- interesting.
'''

# Kruskal–Wallis one-way analysis of variance
hstat_quart_lsnum, hpval_quart_lsnum = stats.kruskal(*[df_grouped_quart.get_group(i).ls_num for i in range(4)])
hstat_quart_lsgsenum, hpval_quart_lsgsenum = stats.kruskal(*[df_grouped_quart.get_group(i).ls_gse_num for i in range(4)])
hstat_quart_lsprivnum, hpval_quart_lsprivnum = stats.kruskal(*[df_grouped_quart.get_group(i).ls_priv_num for i in range(4)])
hstat_quart_secnum, hpval_quart_secnum = stats.kruskal(*[df_grouped_quart.get_group(i).sec_num for i in range(4)])

'''NOTE: The Kruskal-Wallis test is to try out stuff. At least one group is dominating the other.
    As another try-out, let's try a conover test.
'''

# Conover test, p-adjust????
cpval_quart_lsnum = sp.posthoc_conover(df, val_col = 'ls_num', group_col = 'quart_distance')
cpval_quart_lsgsenum = sp.posthoc_conover(df, val_col = 'ls_gse_num', group_col = 'quart_distance')
cpval_quart_lsprivnum = sp.posthoc_conover(df, val_col = 'ls_priv_num', group_col = 'quart_distance')
cpval_quart_secnum = sp.posthoc_conover(df, val_col = 'sec_num', group_col = 'quart_distance')

#------------------------------------------------------------
# Decile grouping
#df['dec_distance'] = pd.qcut(df.log_min_distance, q = 10, labels = False)
