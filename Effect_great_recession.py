# Estimation results

''' This script uses the 3D_panel_estimation procedure to estimate the results of
    the benchmark model in Working paper 2
    '''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Data manipulation packages
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

# Parallel functionality pandas via Dask
import dask.dataframe as dd 
import dask 

# Other parallel packages
import multiprocessing as mp 
from joblib import Parallel, delayed 
num_cores = mp.cpu_count()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2.5)

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

columns = ['date','fips','msamd','cert','log_min_distance','ls','loan_originated',\
            'ln_ta','ln_emp','ln_num_branch','cb','ln_loanamout']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

# Transform to pandas df
df = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')
df.reset_index(drop = True, inplace = True)
df['date'] = df.date.astype(int)
df.sort_values(by = ['cert','date'], ascending = True, inplace = True)

del dd_main

#------------------------------------------------------------
# Group data on date/cert and date/msamd
#------------------------------------------------------------

# date/cert
## Groupby by and take mean
df_cert_mean = df.groupby(['date','cert']).mean()

## Take first difference and sort on log_min_distance
df_cert_diff = df_cert_mean.groupby('cert').diff()
df_cert_diff.sort_values(by = ['log_min_distance','date'], ascending = True, inplace = True)
df_cert_diff.dropna(subset = ['log_min_distance'], inplace = True)

## Check quantiles
cert_q = df_cert_diff.describe(percentiles = [x/20 for x in range(1,20)]) # Outliers are q < 0.05

### Check the lenders in q <.1. 
cert_q01 = df_cert_diff[df_cert_diff.log_min_distance <= df_cert_diff.log_min_distance.quantile(0.1)].index.get_level_values('cert').unique()

cert_qrest = df_cert_diff[df_cert_diff.log_min_distance > df_cert_diff.log_min_distance.quantile(0.1)].index.get_level_values('cert').unique()

difference_cert = set(cert_q01).intersection(set(cert_qrest)) # Most certs in q01 are not in qrest

## Check certs in difference_cert on lender characteristics
## NOTE: NO considerable differences
cert_q01_char = df_cert_mean[df_cert_mean.index.get_level_values('cert').isin(difference_cert)].describe()
cert_qrest_char = df_cert_mean[~df_cert_mean.index.get_level_values('cert').isin(difference_cert)].describe()

## Take pct_change and sort on log_min_distance
df_cert_pct = df_cert_mean.groupby('cert').pct_change()
df_cert_pct.sort_values(by = ['log_min_distance','date'], ascending = True, inplace = True)
df_cert_pct.dropna(subset = ['log_min_distance'], inplace = True)
cert_q_pct = df_cert_pct.describe(percentiles = [x/20 for x in range(1,20)]) # Outliers are q < 0.05

# TODO
# date/MSAMD

#------------------------------------------------------------
# < 2008 vs >=2008
#------------------------------------------------------------

# Get dfs
df_pre = df[df.date < 2008]
df_post = df[df.date >= 2008]

## Get decribes
df_pre_desc = df_pre.describe(percentiles = [x/20 for x in range(1,20)])
df_post_desc = df_post.describe(percentiles = [x/20 for x in range(1,20)])

# cert
## Get mean
df_pre_cert_mean = df_pre.groupby('cert').mean()
df_post_cert_mean = df_post.groupby('cert').mean()

## Get change
df_change_cert_mean = df_post_cert_mean - df_pre_cert_mean
df_change_cert_mean_desc = df_change_cert_mean.describe(percentiles = [x/20 for x in range(1,20)])

## Get certs with NANs and q < 0.01 and check 
cert_change_nan = df_change_cert_mean[df_change_cert_mean.log_min_distance.isnull()].index.get_level_values(0).tolist()
cert_change_q01 = df_change_cert_mean[df_change_cert_mean.log_min_distance <= df_change_cert_mean.log_min_distance.quantile(0.1)].index.get_level_values(0).tolist()

df_pre_nan = df_pre[df_pre.index.get_level_values(0).isin(cert_change_nan)].describe(percentiles = [x/20 for x in range(1,20)])
df_pre_nonan = df_pre[~df_pre.index.get_level_values(0).isin(cert_change_nan)].describe(percentiles = [x/20 for x in range(1,20)])
'''NOTE: NAN-lenders have much smaller mean lending distances, and originate few loans. Do not drive the bump '''

df_pre_q01 = df_pre[df_pre.cert.isin(cert_change_q01)].describe(percentiles = [x/20 for x in range(1,20)])
df_post_q01 = df_post[df_post.cert.isin(cert_change_q01)].describe(percentiles = [x/20 for x in range(1,20)])

df_pre_qrest = df_pre[~df_pre.cert.isin(cert_change_q01)].describe(percentiles = [x/20 for x in range(1,20)])
df_post_qrest = df_post[~df_post.cert.isin(cert_change_q01)].describe(percentiles = [x/20 for x in range(1,20)])
''' NOTE: q01 lenders indeed see a larger drop in lending distances. Pre 2008 lending distances two groups 
    are close, post difference is more significant. q01 lenders are larger, have more employees and branches,
    both pre and post 2008. Size effect? Nationwide banks? --> Question is how much these banks influence the results'''

#------------------------------------------------------------
# Plot
#------------------------------------------------------------
# On distance
df['q_distance'] = pd.qcut(df.log_min_distance, [0, .6, .7, .8, .9, 1.], labels = [0, 1, 2, 3, 4])
df['q_distance'] = df['q_distance'].astype(int)

df_dist = df.groupby(['date','q_distance']).log_min_distance.mean()
df_dist = df_dist.unstack(level = 1)
df_dist = df_dist.sort_index(ascending = True)

# plot
labels = ['<60','60-70','70-80','80-90','>90']

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Log distance (km)')
for col, label in zip(df_dist.iloc[:,1:].iteritems(), labels[1:]):
    ax.plot(col[1], label = label)

ax.axvspan(2007,2009, alpha = 0.5, color = 'lightgray')
ax.set_xlim(2004, 2019)
ax.legend()
plt.tight_layout()
'''NOTE: Nothing special '''

#fig.savefig('Figures/Number_originations.png')

# On lender size
df['q_ln_ta'] = pd.qcut(df.ln_ta, [0, .2, .4, .6, .8, 1], labels = [0, 1, 2, 3, 4])
df['q_ln_ta'] = df['q_ln_ta'].astype(int)

df_ln_ta = df.groupby(['date','q_ln_ta']).log_min_distance.mean()
df_ln_ta = df_ln_ta.unstack(level = 1)
df_ln_ta = df_ln_ta.sort_index(ascending = True)

# plot
labels = ['<20','20-40','40-60','60-80','>80']

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Log distance (km)')
for col, label in zip(df_ln_ta.iteritems(), labels):
    ax.plot(col[1], label = label)

ax.axvspan(2007,2009, alpha = 0.5, color = 'lightgray')
ax.set_xlim(2004, 2019)
ax.legend()
plt.tight_layout()

'''NOTE: The mid categories decrease the most. Small lenders and large lenders bearly
    see a difference. Small lenders probably are relationship lenders, large lenders
    are the mega lenders. Largest lenders have the smallest distance'''

# On loan_amount
df['q_ln_loanamout'] = pd.qcut(df.ln_loanamout, [0, .2, .4, .6, .8, 1], labels = [0, 1, 2, 3, 4])
df['q_ln_loanamout'] = df['q_ln_loanamout'].astype(int)

df_ln_loanamout = df.groupby(['date','q_ln_loanamout']).log_min_distance.mean()
df_ln_loanamout = df_ln_loanamout.unstack(level = 1)
df_ln_loanamout = df_ln_loanamout.sort_index(ascending = True)

# plot
labels = ['<20','20-40','40-60','60-80','>80']

fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = 'Log distance (km)')
for col, label in zip(df_ln_loanamout.iteritems(), labels):
    ax.plot(col[1], label = label)

ax.axvspan(2007,2009, alpha = 0.5, color = 'lightgray')
ax.set_xlim(2004, 2019)
ax.legend()
plt.tight_layout()

'''NOTE: Bearly any differences '''