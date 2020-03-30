# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-lender-level dataset per year
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'/data/p285325/WP2_distance/')
#os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Data')

# Load packages
import numpy as np
import pandas as pd
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

start = 2010
end = 2017
num_cores = mp.cpu_count()

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Function for parallelization pandas read_csv for the HMDA data
def readHMDA(year):
    
    filename = 'data_hmda_{}.csv'.format(year)
    
    return (pd.read_csv(filename, dtype = {'msamd':'str', 'fips':'str'}))
    
# Merge SDI LF   
def mergeSDILF(year):
       
    ## Merge on year
    df_load = df_sdi[df_sdi.date == year].merge(df_lf,\
                            how = 'left', left_on = 'cert',\
                            right_on = 'CERT{}'.format(str(year)[2:4]))
    
    
    ## Return concatenated pd DataFrame
    return (df_load)

#------------------------------------------------------------
# Load all dfs that do not need to be read in the loop
#------------------------------------------------------------

# df MSA
df_msa = pd.read_csv('data_msa_popcenter.csv', index_col = 0, dtype = {'fips':'str','cbsa_code':'str'}) 
          
# SOD df
df_sod = pd.read_csv('df_sod_wp2.csv', index_col = 0, dtype = {'fips':'str'})

# df SDI
df_sdi = pd.read_csv('df_sdi_wp2.csv', dtype = {'cert':'float64'})

# df LF
## Prelims
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end +1)]

## Load df LF
df_lf = pd.read_csv('hmdpanel17.csv', usecols = vars_lf)

# Merge df SDI and df LF   
if __name__ == "__main__":
    list_sdilf = Parallel(n_jobs=num_cores)(delayed(mergeSDILF)(year) for year in range(start,end + 1)) 

## Concat to pd DataFrame
df_sdilf = pd.concat(list_sdilf, ignore_index = True)

## Drop CERT.. columns and drop nans in hmprid
df_sdilf = df_sdilf[df_sdilf.columns[~df_sdilf.columns.str.contains('CERT')]].dropna(subset = ['hmprid'])

# Load HMDA
if __name__ == "__main__":
    list_hmda = Parallel(n_jobs=num_cores)(delayed(readHMDA)(year) for year in range(start,end + 1)) 

## Concat to pd DataFrame
df_hmda = pd.concat(list_hmda, ignore_index = True) 

# Make main df
df_main = df_sdilf.merge(df_hmda, how = 'left', left_on = ['date','hmprid'],\
                             right_on = ['date','respondent_id'])

# Load min_distances df
df_mindist = pd.read_csv('data_min_distances.csv', dtype = {'fips':'str'})

## Take log of min_distances
df_mindist['log_min_distance'] = np.log(df_mindist.min_distance + 1)

#------------------------------------------------------------
# Aggregate to MSA-lender level
#------------------------------------------------------------
# Only keep MSA that are in the df_msa
df_main = df_main[df_main.fips.isin(df_msa.fips.unique())]

# Remove all certs in df_main that are not in df_sod
df_main = df_main[df_main.cert.isin(df_sod.cert)]
  
# Merge distance on fips and cert
df_main = df_main.merge(df_mindist, how = 'left', on = ['fips','cert'])

#------------------------------------------------------------
# Aggregate data to MSA-lender-level

# Setup functions to parallelize functions
## Wrapper
def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs = num_cores)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

## Percentage
def tmpPerc(df, col_name):
    return (np.sum(df[col_name]) / len(df[col_name]))
    
## Loan sales on value
def tmpLSVal(df, col_name):
    return(np.sum(df[col_name] * df.ln_loanamout) / np.sum(df.ln_loanamout))
    
## Loan sales dummy
def tmpLSDum(df, col_name):
    return((df[col_name].sum() > 0.0) * 1)
    
# Group the data
df_grouped = df_main.groupby(['date','cert','msamd'])    

# 1) Aggregate portfolio-specific variables
df_agg = df_grouped[['lti','ln_loanamout','ln_appincome','log_min_distance']].mean()

if __name__ == '__main__':
    df_agg['subprime'] = Parallel(n_jobs = num_cores)(delayed(tmpPerc)(group, 'subprime') for name, group in df_grouped)
    df_agg['secured'] = Parallel(n_jobs = num_cores)(delayed(tmpPerc)(group, 'secured') for name, group in df_grouped)

    # 2) Add Loan sales variables
    ## Fraction based on number of loans   
    df_agg['ls_num'] = Parallel(n_jobs = num_cores)(delayed(tmpPerc)(group, 'ls') for name, group in df_grouped)
    df_agg['ls_gse_num'] = Parallel(n_jobs = num_cores)(delayed(tmpPerc)(group, 'ls_gse') for name, group in df_main.groupby(['date','cert','msamd']))
    df_agg['ls_priv_num'] = Parallel(n_jobs = num_cores)(delayed(tmpPerc)(group, 'ls_priv') for name, group in df_main.groupby(['date','cert','msamd']))
    df_agg['sec_num'] = Parallel(n_jobs = num_cores)(delayed(tmpPerc)(group, 'sec') for name, group in df_main.groupby(['date','cert','msamd']))

    ## Fraction based on value of loans
    df_agg['ls_val'] = Parallel(n_jobs = num_cores)(delayed(tmpLSVal)(group, 'ls') for name, group in df_grouped)
    df_agg['ls_gse_val'] = Parallel(n_jobs = num_cores)(delayed(tmpLSVal)(group, 'ls_gse') for name, group in df_grouped)
    df_agg['ls_priv_val'] = Parallel(n_jobs = num_cores)(delayed(tmpLSVal)(group, 'ls_priv') for name, group in df_grouped)
    df_agg['sec_val'] = Parallel(n_jobs = num_cores)(delayed(tmpLSVal)(group, 'sec') for name, group in df_grouped)
     
## Dummy
    df_agg['ls_dum' ] = Parallel(n_jobs = num_cores)(delayed(tmpLSDum)(group, 'ls') for name, group in df_grouped)
    df_agg['ls_gse_dum' ] = Parallel(n_jobs = num_cores)(delayed(tmpLSDum)(group, 'ls_gse') for name, group in df_grouped)
    df_agg['ls_priv_dum' ] = Parallel(n_jobs = num_cores)(delayed(tmpLSDum)(group, 'ls_priv') for name, group in df_grouped)
    df_agg['sec_dum' ] = Parallel(n_jobs = num_cores)(delayed(tmpLSDum)(group, 'sec') for name, group in df_grouped)

# 3) Bank/Thrift level controls
## Reset index before merge
df_agg.reset_index(inplace = True)
    
## add ln_ta, ln_emp and bank indicator
df_agg = df_agg.merge(df_sdi[['date','cert','cb','ln_ta','ln_emp']],\
                      how = 'left', on = ['date','cert'])

## Add number of branches
num_branches = df_sod.groupby(['date','cert']).cert.count().rename('num_branch').to_frame()
df_agg = df_agg.merge(num_branches, how = 'left', left_on = ['date','cert'],\
                      right_on = [num_branches.index.get_level_values(0), num_branches.index.get_level_values(1)])

# 4) MSA level
# Add vars from SOD to df_msa
density = df_sod.groupby(['date','fips']).fips.count().rename('density').to_frame()
df_msa = df_msa.merge(density, how = 'left', left_on = ['date','fips'], right_on = [density.index.get_level_values(0), density.index.get_level_values(1)])

hhi = df_sod.groupby(['date','fips']).depsumbr.apply(lambda x: ((x / x.sum())**2).sum()).rename('hhi').to_frame()
df_msa = df_msa.merge(hhi, how = 'left', left_on = ['date','fips'], right_on = [hhi.index.get_level_values(0), hhi.index.get_level_values(1)])

## HMDA vars
ln_mfi = df_main.groupby(['date','fips']).hud_median_family_income.mean().rename('ln_mfi').to_frame()
df_msa = df_msa.merge(ln_mfi, how = 'left', left_on = ['date','fips'], right_on = [ln_mfi.index.get_level_values(0), ln_mfi.index.get_level_values(1)])

dum_min = df_main.groupby(['date','fips']).minority_population.apply(lambda x: (x.mean() > 0.5) * 1).rename('dum_min').to_frame()
df_msa = df_msa.merge(dum_min, how = 'left', left_on = ['date','fips'], right_on = [dum_min.index.get_level_values(0), dum_min.index.get_level_values(1)])

# from df_msa
'''NOTE
There is a mistake in this block. The merge is not done correctly --> missings
in the variables of df_msa. FIX

Check what exactly the MSAMD is and whether we need cbsa or csa.

ANSWER: MSA are not the same as CBSA or CSA. The latter two could be a collection of MSA.
We therefore have to fips the MSA for each fips in the df_msa databased. Do this in a different
script.
'''

df_msa_agg = df_msa.groupby(['date','cbsa_code'])[['ln_pop', 'density', 'hhi', 'ln_mfi', 'dum_min']].mean()
df_agg = df_agg.merge(df_msa_agg, how = 'left', left_on = ['date','msamd'], right_on = [df_msa_agg.index.get_level_values(0), df_msa_agg.index.get_level_values(1)])

# Add average lending distance per MSA
distance_msa = df_main.groupby(['date','msamd']).log_min_distance.apply(lambda x: np.log(1 + x.mean())).rename('mean_distance').to_frame()
df_agg = df_agg.merge(distance_msa, how = 'left', left_on = ['date','msamd'], right_on = [distance_msa.index.get_level_values(0), distance_msa.index.get_level_values(1)])

#------------------------------------------------------------
# Save df_agg
# Make list of which columns to keep

# Save
df_agg.to_csv('data_agg.csv', index = False)