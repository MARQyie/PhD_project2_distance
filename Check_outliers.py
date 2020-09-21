# Check for outliers
''' This script checks for outliers '''

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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')
import multiprocessing as mp 
from joblib import Parallel, delayed 

#------------------------------------------------------------
# Load data
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg.csv')
#df.dropna(inplace = True)

columns = ['cert', 'msamd', 'fips', 'date', \
           'ln_ta', 'ln_emp', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
           'secured', 'ls', 'ls_gse', 'ls_priv', 'sec', 'min_distance',\
           'min_distance_cdd', 'log_min_distance', 'log_min_distance_cdd',\
           'ln_mfi', 'hhi', 'density', 'pop_area']
#df_main = pd.read_csv('Data/data_main.csv', usecols = columns)

#------------------------------------------------------------
# Make boxplots
#------------------------------------------------------------
#------------------------------------------------------------
# df 
# Prelims
## Setup dict with vars
vars_needed = [val for val in df.columns.tolist() if val not in ['date', 'cert', 'msamd']]
vars_names = ['Loan-to-Income', 'Loan Amount', 'Appl. Income', 'Min. Distance', 'Min. Distance CDD',\
              'Density', 'Population', 'Mean Fam. Income', 'HHI', \
              'Subprime', 'Secured', 'Loan Sales (num)', 'Loan Sales GSE (num)',\
              'Loan Sales Priv. (num)', 'Securitized (num)','Loan Sales (val)',\
              'Loan Sales GSE (val)', 'Loan Sales Priv. (val)', 'Securitized (val)',\
              'Loan Sales (dum)', 'Loan Sales GSE (dum)', 'Loan Sales Priv. (dum)',\
              'Securitized (dum)', 'Bank', 'Size', 'Total Empl.', 'Num. Branches']
dict_var_names = dict(zip(vars_needed, vars_names))

## Make function to parallelize
def boxPlots(var):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))
    
    data = df[var].dropna()
    ax.boxplot(data)
    
    plt.xticks([1], ['Full Sample'])
    
    fig.savefig('Figures\Box_plots\Box_finaldf_{}.png'.format(var)) 
    plt.clf()

# Run
num_cores = mp.cpu_count()
'''
if __name__ == '__main__':
    Parallel(n_jobs = num_cores)(delayed(boxPlots)(var) for var in vars_needed)
'''    
#------------------------------------------------------------
# df_main
'''
## Make function to parallelize
def boxPlotsMain(var):
    global df_main
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(var))
    
    data = df_main[var].dropna()
    ax.boxplot(data)
    
    plt.xticks([1], ['Main Sample'])
    
    fig.savefig('Figures\Box_plots\Box_maindf_{}.png'.format(var)) 
    plt.clf() 
    
# Run
vars_needed_main = [val for val in df_main.columns.tolist() if val not in ['date', 'cert', 'msamd']] 
'''   
'''   
if __name__ == '__main__':
    Parallel(n_jobs = num_cores)(delayed(boxPlotsMain)(var) for var in vars_needed_main)
'''
'''Nothing out of the ordinary '''
    
#------------------------------------------------------------
# Remove Outliers and save df
#------------------------------------------------------------
'''NOTE
    The boxplots show that total employees has some zero values. 
    This is improbable and thus we remove those and save a cleaned df.'''

# limit vars    
df_cleaned = df[df.ln_emp != 0.0]
df_cleaned = df_cleaned[df_cleaned.lti < np.inf]    
df_cleaned = df_cleaned[df_cleaned.ln_appincome > -np.inf]
 
# Add log of pop / area + num_branches
df_cleaned['ln_pop_area'] = np.log(df_cleaned.pop_area + 1)
df_cleaned['ln_num_branch'] = np.log(df_cleaned.num_branch + 1)
df_cleaned['ln_density'] = np.log(df_cleaned.density * 1e3 + 1)

# Check describe
df_desc = df_cleaned.describe()

# Save df
df_cleaned.to_csv('Data/data_agg_clean.csv', index = False)