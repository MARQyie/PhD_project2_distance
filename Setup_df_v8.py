# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-lender-level dataset per year
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Data/')

# Load packages
import numpy as np
import pandas as pd
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

start = 2004
end = 2019
num_cores = mp.cpu_count()

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Function for parallelization pandas read_csv for the HMDA data
def readHMDA(year):
    
    filename = 'data_hmda_{}.csv'.format(year)
    
    return (pd.read_csv(filename, dtype = {'msamd':'uint16', 'fips':'uint16',\
                                           'respondent_id':'str'}))
    
# Merge SDI LF   
def mergeSDILF(year):
    
    ## Prelims
    if year < 2018:
        cert = 'CERT{}'.format(str(year)[2:4])
    else:
        cert = 'CERT17'
       
    ## Merge on year
    df_load = df_sdi[df_sdi.date == year].merge(df_lf.dropna(subset = [cert]),\
                            how = 'left', left_on = 'cert',\
                            right_on = cert)
    
    ## Return concatenated pd DataFrame
    return (df_load)

# Setup functions to parallelize functions
## Wrapper
def applyParallel(dfGrouped, func, col_name = None):
    retLst = Parallel(n_jobs = num_cores, prefer="threads")(delayed(func)(group, col_name) for name, group in dfGrouped)
    return retLst  

## Percentage
def tmpPerc(df, col_name):
    return (np.sum(df[col_name]) / len(df[col_name]))
    
## Loan sales on value
def tmpLSVal(df, col_name):
    return(np.sum(df[col_name] * df.ln_loanamout) / np.sum(df.ln_loanamout))
    
## Loan sales dummy
def tmpLSDum(df, col_name):
    return((df[col_name].sum() > 0.0) * 1)

## Ever loan sales dummies
def tmpLSEver(df, col_name):
    return((1 in df[col_name]) * 1)
    
#------------------------------------------------------------
# Load all dfs that do not need to be read in the loop
#------------------------------------------------------------

# df MSA
df_msa = pd.read_csv('data_msa_popcenter.csv', index_col = 0, dtype = {'fips':'uint16'})

## Remove duplicate rows
df_msa.drop_duplicates(subset = ['date','population', 'latitude', 'longitude', 'fips', 'ln_pop'], inplace = True)
                  
# SOD df
df_sod = pd.read_csv('df_sod_wp2.csv', index_col = 0, dtype = {'fips':'uint16'})

# df SDI
df_sdi = pd.read_csv('df_sdi_wp2.csv', index_col = 0)

# df pop
df_pop = pd.read_csv('data_pop.csv', index_col = 0)

# df_int
df_int = pd.read_csv('data_internet.csv', index_col = 0)

# df LF
## Prelims
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end - 1)]

## Load df LF
df_lf = pd.read_csv('hmdpanel17.csv', usecols = vars_lf)

### reduce dimensions df_lf
df_lf.dropna(how = 'all', subset = vars_lf[1:], inplace = True) # drop rows with all na
df_lf = df_lf[~(df_lf[vars_lf[1:]] == 0.).any(axis = 1)] # Remove cert with value 0.0
df_lf = df_lf[df_lf[vars_lf[1:]].all(axis = 1)] # Drop rows that have different column values (nothing gets deleted: good)

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

## Concat to pd DataFrame and drop population column
df_hmda = pd.concat(list_hmda, ignore_index = True) 
df_hmda = df_hmda[df_hmda.loan_purpose == 1].drop(columns = ['population'])

# Make main df
df_main = df_sdilf.merge(df_hmda, how = 'left', left_on = ['date','hmprid'],\
                             right_on = ['date','respondent_id'])

# Load min_distances df
df_mindist_2004 = pd.read_csv('data_min_distances_2004.csv', dtype = {'fips':'uint16'})
df_mindist_2010 = pd.read_csv('data_min_distances_2010.csv', dtype = {'fips':'uint16'})

## Take log of min_distances
df_mindist_2004['log_min_distance'] = np.log(df_mindist_2004.min_distance + 1)
df_mindist_2004['log_min_distance_cdd'] = np.log(df_mindist_2004.min_distance_cdd + 1)

df_mindist_2010['log_min_distance'] = np.log(df_mindist_2010.min_distance + 1)
df_mindist_2010['log_min_distance_cdd'] = np.log(df_mindist_2010.min_distance_cdd + 1)

#------------------------------------------------------------
# Add control variables
#------------------------------------------------------------

# Only keep MSA that are in the df_msa
df_main = df_main[df_main.fips.isin(df_msa.fips.unique())] 

# Remove all certs in df_main that are not in df_sod
df_main = df_main[df_main.cert.isin(df_sod.cert)]
  
# Merge distance on fips and cert
df_main_2004 = df_main[df_main.date < 2010].merge(df_mindist_2004, how = 'left', on = ['fips','cert'])
df_main_2010 = df_main[df_main.date >= 2010].merge(df_mindist_2010, how = 'left', on = ['fips','cert'])
df_main = pd.concat([df_main_2004,df_main_2010], ignore_index=True)

# Merge population on fips and date
df_main = df_main.merge(df_pop, how = 'left', on = ['fips','date'])

# Merge internet on fips and date
df_main = df_main.merge(df_int[['fips','date', 'perc_intsub', 'perc_broadband', 'perc_noint']], how = 'left', on = ['fips','date'])

# Add FIPS-level vars
## Branch Density
density = df_sod.groupby(['date','fips']).fips.count().rename('density').to_frame().reset_index()
df_msasod = density.merge(df_msa, how = 'left', on = ['date','fips'])

## HHI
hhi = df_sod.groupby(['date','fips']).depsumbr.apply(lambda x: ((x / x.sum())**2).sum()).rename('hhi').to_frame().reset_index()
df_msasod = hhi.merge(df_msasod, how = 'left', on = ['date','fips'])

## Merge msasod with main
df_main = df_main.merge(df_msasod.drop('population', axis = 1), how = 'left', on = ['date','fips'])

# population weigh density
df_main['density'] = df_main.density / df_main.population

# Add SDI data
## ln_ta, ln_emp, bank indicator, and loan sales indicators
df_main = df_main.merge(df_sdi[['date','cert','cb','ln_ta','ln_emp',\
                        'sec_sdi','ls_sdi', 'lssec_sdi','sec_ever', 'ls_ever','lssec_ever']],\
                      how = 'left', on = ['date','cert'])

## Add number of branches
num_branches = df_sod.groupby(['date','cert']).cert.count().rename('num_branch').to_frame()
df_main = df_main.merge(num_branches, how = 'left', on = ['date','cert'])

# Add local (lender) variable
df_main['local']

#df_main.loc[0,'fips'] in df_sod.loc[df_sod.cert == df_main.loc[0,'cert'],'fips']
#------------------------------------------------------------
# Aggregate to MSA-lender level
#------------------------------------------------------------

# Group the data
df_grouped = df_main.groupby(['date','cert','msamd'])    

# 1) Aggregate portfolio-specific variables
df_agg = df_grouped[['lti','ln_loanamout','ln_appincome','log_min_distance',\
'log_min_distance_cdd']].mean()

if __name__ == '__main__':
    df_agg['subprime'] = applyParallel(df_grouped, tmpPerc, 'subprime')
    df_agg['lien'] = applyParallel(df_grouped, tmpPerc, 'lien')

    # 2) Add Loan sales variables
    ## Fraction based on number of loans   
    df_agg['ls_num'] = applyParallel(df_grouped, tmpPerc, 'ls')
    df_agg['ls_gse_num'] = applyParallel(df_grouped, tmpPerc, 'ls_gse')
    df_agg['ls_priv_num'] = applyParallel(df_grouped, tmpPerc, 'ls_priv')
    df_agg['sec_num'] = applyParallel(df_grouped, tmpPerc, 'sec')

    ## Fraction based on value of loans
    df_agg['ls_val'] = applyParallel(df_grouped, tmpLSVal, 'ls')
    df_agg['ls_gse_val'] = applyParallel(df_grouped, tmpLSVal, 'ls_gse')
    df_agg['ls_priv_val'] = applyParallel(df_grouped, tmpLSVal, 'ls_priv')
    df_agg['sec_val'] = applyParallel(df_grouped, tmpLSVal, 'sec')
     
    ## Dummy
    df_agg['ls_dum'] = applyParallel(df_grouped, tmpLSDum, 'ls')
    df_agg['ls_gse_dum'] = applyParallel(df_grouped, tmpLSDum, 'ls_gse')
    df_agg['ls_priv_dum'] = applyParallel(df_grouped, tmpLSDum, 'ls_priv')
    df_agg['sec_dum'] = applyParallel(df_grouped, tmpLSDum, 'sec')
    
    ## Ever lsers
    df_agg['sec_ever_dum'] = applyParallel(df_grouped, tmpLSEver, 'sec_ever')
    df_agg['ls_ever_dum'] = applyParallel(df_grouped, tmpLSEver, 'ls_ever')
    df_agg['lssec_ever_dum'] = applyParallel(df_grouped, tmpLSEver, 'lssec_ever')

## Reset index before merge
df_agg.reset_index(inplace = True)

# 2) Transform MSA specific variables
df_grouped = df_main.groupby(['date','msamd'])    
msa_data = df_grouped[['density', 'pop_area', 'hhi', 'perc_intsub', 'perc_broadband', 'perc_noint']].mean().reset_index()

df_agg = df_agg.merge(msa_data, how = 'left', on = ['date','msamd'])

'''OLD
# Add average lending distance per MSA
distance_msa = df_main.groupby(['date','msamd']).log_min_distance.mean().rename('mean_distance').to_frame()
df_agg = df_agg.merge(distance_msa, how = 'left', left_on = ['date','msamd'], right_on = [distance_msa.index.get_level_values(0), distance_msa.index.get_level_values(1)])
'''

#------------------------------------------------------------
# Remove all respective MSA/banks with only one observation and drop na (not in internet variables)
drop_msamd = set(df_agg.msamd.value_counts()[df_agg.msamd.value_counts() > 1].index)
drop_cert = set(df_agg.cert.value_counts()[df_agg.cert.value_counts() > 1].index)

df_agg = df_agg[(df_agg.msamd.isin(drop_msamd)) & (df_agg.cert.isin(drop_cert))]

## Drop na
subset = ['log_min_distance', 'ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'pop_area', 'density', 'hhi', 'ln_mfi']
df_agg.dropna(subset = subset, inplace = True)

#------------------------------------------------------------
# Save df_agg
# Make list of which columns to keep

# Save
df_main.to_csv('data_main.csv', index = False)
df_agg.to_csv('data_agg.csv', index = False)