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
df.dropna(inplace = True)

#------------------------------------------------------------
# Make boxplots
#------------------------------------------------------------
# Prelims
## Setup dict with vars
vars_needed = [val for val in df.columns.tolist() if val not in ['date', 'cert', 'msamd']]
vars_names = ['Loan-to-Income', 'Loan Amount', 'Appl. Income', 'Min. Distance',\
              'Density', 'Population', 'Mean Fam. Income', 'HHI', 'Dummy Minority',\
              'Subprime', 'Secured', 'Loan Sales (num)', 'Loan Sales GSE (num)',\
              'Loan Sales Priv. (num)', 'Securitized (num)','Loan Sales (val)',\
              'Loan Sales GSE (val)', 'Loan Sales Priv. (val)', 'Securitized (val)',\
              'Loan Sales (dum)', 'Loan Sales GSE (dum)', 'Loan Sales Priv. (dum)',\
              'Securitized (dum)', 'Bank', 'Size', 'Total Empl.', 'Num. Branches',\
              'Mean Distance']
dict_var_names = dict(zip(vars_needed, vars_names))

## Make function to parallelize
def boxPlots(var):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.title('{}'.format(dict_var_names[var]))
    
    data = df[var]
    ax.boxplot(data)
    
    plt.xticks([1], ['Full Sample'])
    
    fig.savefig('Figures\Box_plots\Box_finaldf_{}.png'.format(var)) 
    plt.clf()

# Run
num_cores = mp.cpu_count()

if __name__ == '__main__':
    Parallel(n_jobs = num_cores)(delayed(boxPlots)(var) for var in vars_needed)
    
#------------------------------------------------------------
# Remove Outliers and save df
#------------------------------------------------------------
'''NOTE
    The boxplots show that total employees has some zero values. 
    This is improbable and thus we remove those and save a cleaned df.'''
    
df_cleaned = df[df.ln_emp != 0.0]

# Save df
df_cleaned.to_csv('Data/data_agg_clean.csv', index = False)