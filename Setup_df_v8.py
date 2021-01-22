# Setup the df 
''' This script makes the final df (before data cleaning). 

    We use dask.dataframes to cut back on RAM-usage and utilize fast parallelization
    of loading the dataframes.
    
    Final dfs: 
        1) df_main is a loan-level dataset
        2) df_agg is df_main aggregated at the MSAMD level.
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Data/')

# Load packages
## Data manipulation packages
import numpy as np
import pandas as pd

## Parallel functionality pandas via Dask
import dask.dataframe as dd 
import dask 
#dask.config.set({'temporary_directory': '/data/p285325/WP2_distance/'})
dask.config.set({'temporary_directory': 'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Data/'})

## Parallel processing
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

## Parquet packages
#import pyarrow as pa
#import pyarrow.parquet as pq
#import fastparquet 

#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

start = 2004
end = 2019
num_cores = mp.cpu_count()

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Merge SDI LF   
def mergeSDILF(year):
    
    ## Prelims
    if year < 2018:
        cert = 'CERT{}'.format(str(year)[2:4])
    else:
        cert = 'CERT17'
       
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

## Local lender dummy
def findLocal(cert, msamd):
    return  any(item in dict_cert[cert] for item in dict_msamd[msamd]) * 1

#------------------------------------------------------------
# Load all dfs that do not need to be read in the loop
#------------------------------------------------------------
                  
# SOD df
df_sod = pd.read_csv('df_sod_wp2.csv', dtype = {'fips':'int32'},index_col = 0)

# df SDI
df_sdi = pd.read_csv('df_sdi_wp2.csv', dtype = {'cert':'float'},index_col = 0)

# df pop
df_pop = pd.read_csv('data_pop.csv',index_col = ['fips','date'], usecols = ['fips','date','population','pop_area'])

# df_int
df_int = pd.read_csv('data_internet.csv',index_col = ['fips','date'], usecols = ['fips','date', 'perc_intsub', 'perc_broadband', 'perc_noint'])

# df LF
## Prelims
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end - 1)]

## Load df LF
df_lf = pd.read_csv('hmdpanel17.csv', usecols = vars_lf, dtype = {'hmprid':'str'})

### reduce dimensions df_lf
df_lf = df_lf.dropna(how = 'all', subset = vars_lf[1:]) # drop rows with all na
df_lf = df_lf[~(df_lf[vars_lf[1:]] == 0.).any(axis = 1)] # Remove cert with value 0.0
df_lf = df_lf[df_lf[vars_lf[1:]].all(axis = 1)] # Drop rows that have different column values (nothing gets deleted: good)

# Merge df SDI and df LF   
if __name__ == "__main__":
    list_sdilf = Parallel(n_jobs=num_cores,prefer="threads")(delayed(mergeSDILF)(year) for year in range(start,end + 1)) 

## Concat to pd DataFrame
df_sdilf = pd.concat(list_sdilf, ignore_index = True)

## Drop CERT.. columns and drop nans in hmprid
df_sdilf = df_sdilf[df_sdilf.columns[~df_sdilf.columns.str.contains('CERT')]].dropna(subset = ['hmprid'])

del df_sdi, df_lf, list_sdilf

# Load min_distances df
df_mindist_2004 = pd.read_csv('data_min_distances_2004.csv', dtype = {'fips':'int64'})
df_mindist_2010 = pd.read_csv('data_min_distances_2010.csv', dtype = {'fips':'int64'})

## Take log of min_distances
df_mindist_2004['log_min_distance'] = np.log(df_mindist_2004.min_distance + 1)
df_mindist_2004['log_min_distance_cdd'] = np.log(df_mindist_2004.min_distance_cdd + 1)

df_mindist_2010['log_min_distance'] = np.log(df_mindist_2010.min_distance + 1)
df_mindist_2010['log_min_distance_cdd'] = np.log(df_mindist_2010.min_distance_cdd + 1)

# Make fips-level dataframe
## Merge df_pop and df_int
df_fips = df_pop.merge(df_int, how = 'left', left_index = True, right_index = True)

## Add Branch Density
density = df_sod.groupby(['fips','date']).fips.count().rename('density').to_frame()

## Add HHI and merge
hhi = df_sod.groupby(['fips','date']).depsumbr.apply(lambda x: ((x / x.sum())**2).sum()).rename('hhi').to_frame()
df_fips_vars = density.merge(hhi, how = 'left', on = ['fips','date'])

df_fips = df_fips.merge(df_fips_vars, left_index = True, right_index = True)

## Delete datasets from working memory
del df_pop, df_int, df_fips_vars

# Make cert-level dataframe
## Add number of branches
df_cert = df_sod.groupby(['date','cert']).cert.count().rename('num_branch').to_frame().reset_index()

#------------------------------------------------------------
# Make main df
#------------------------------------------------------------
## NOTE: use dask to parallelize and prevent a RAM-overload
lst_hmda = []
#lambda_local = lambda data: 1 if any(item in dict_cert[data.cert] for item in dict_msamd[data.msamd]) * 1 else 0

for year in range(start, end + 1):
    dd_hmda = dd.read_csv('data_hmda_{}.csv'.format(year), dtype = {'respondent_id':'str','sex_4': 'float64','ln_appincome':'float64','ethnicity_0': 'int8', 'ethnicity_1': 'int8','ethnicity_2': 'int8','ethnicity_3': 'int8','ethnicity_4': 'int8','ethnicity_5': 'int8','loan_type_1': 'int8','loan_type_2': 'int8','loan_type_3': 'int8','loan_type_4': 'int8','sex_1': 'int8','sex_2': 'int8','sex_3': 'int8'})
    
    # Drop population column
    #dd_hmda = dd_hmda.drop('population', axis = 1)
    
    # Merge with df_sdilf
    dd_hmda = dd.merge(dd_hmda,\
                       df_sdilf[df_sdilf.date == year].drop(columns = ['date']).set_index('hmprid'),\
                       left_on='respondent_id', right_index=True)
    
    # Merge with fips-level data
    dd_hmda = dd.merge(dd_hmda,\
                       df_fips[df_fips.index.get_level_values('date') == year].droplevel('date'),\
                       left_on='fips', right_index=True) 
    
    # Merge with cert-level data
    dd_hmda = dd.merge(dd_hmda,\
                       df_cert[df_cert.date == year].drop(columns = ['date']),\
                       left_on='cert', right_on='cert')
        
    # Merge with distance dfs
    if year < 2010:
        dd_hmda = dd.merge(dd_hmda, df_mindist_2004, left_on=['fips','cert'], right_on=['fips','cert'])
    else:
        dd_hmda = dd.merge(dd_hmda, df_mindist_2010, left_on=['fips','cert'], right_on=['fips','cert'])
    
    # Remove all certs that are not in df_sod
    dd_hmda = dd_hmda[dd_hmda.cert.isin(df_sod.cert.unique())]
    
    # Add variables
    ## population weigh density
    dd_hmda['density'] = dd_hmda.density / dd_hmda.population
    
    lst_hmda.append(dd_hmda)

# Concat dask dataframes
dd_main = dd.concat(lst_hmda)

# Save and compute the dd_main (prevent RAM overload)
## NOTE: We partition on date such that we get smaller and useful partitions
dd_main.to_parquet(path = 'data_main_pre_clean1.parquet',\
                   engine = 'fastparquet',\
                   compression = 'snappy',\
                   partition_on = ['date'])
    
# Load dd_main
dd_main = dd.read_parquet(path = 'data_main_pre_clean1.parquet',\
                       engine = 'fastparquet')

# Set dictionaries for local lender calculation
dict_msamd = dict(dd_main.groupby('msamd').fips.unique().compute(scheduler = "threads"))
dict_cert = dict(df_sod.groupby('cert').fips.unique())

# Remove all respective MSA/banks with only one observation
drop_msamd = set(dd_main.msamd.value_counts()[dd_main.msamd.value_counts() > 1].index.compute())
drop_cert = set(dd_main.cert.value_counts()[dd_main.cert.value_counts() > 1].index.compute())

dd_main = dd_main[(dd_main.msamd.isin(drop_msamd)) & (dd_main.cert.isin(drop_cert))]
npartitions_main = dd_main.npartitions

# Compute main dataset and remove unneeded datasets
dd_main.to_parquet(path = 'data_main_pre_clean2.parquet',\
                   engine = 'fastparquet',\
                   compression = 'snappy',\
                   partition_on = ['date'])
dd_main = dd.read_parquet(path = 'data_main_pre_clean2.parquet',\
                       engine = 'fastparquet')

del dd_hmda, lst_hmda, df_sdilf, df_fips, df_cert, df_mindist_2004, df_mindist_2010

#------------------------------------------------------------
# Add local lender variable and prelim cleaning
#------------------------------------------------------------

# Add local (lender) variable
dd_main['local'] = dd_main.apply(lambda x: 1 if any(item in dict_cert[x.cert] for item in dict_msamd[x.msamd]) else 0, axis = 1, meta = ('local','i4'))

# Compute main dataset and remove unneeded datasets
dd_main.to_parquet(path = 'data_main_pre_clean3.parquet',\
                   engine = 'fastparquet',\
                   compression = 'snappy',\
                   partition_on = ['date'])
dd_main = dd.read_parquet(path = 'data_main_pre_clean3.parquet',\
                       engine = 'fastparquet')

'''OLD
lst_cert, list_msamd = np.array(dd_main.loc[:,['cert','msamd']].compute(scheduler = 'threads').T)

if __name__ == '__main__':
    local = Parallel(n_jobs = num_cores, prefer="threads")(delayed(findLocal)(cert, msamd) for cert, msamd in zip(lst_cert,list_msamd))
    
## Transform to dd array
df_local = pd.DataFrame(local, index = dd_main.index.compute(scheduler = 'threads').tolist(), columns = ['local'])
                        
## Add to dd_main
test = dd_main.merge(df_local, left_index = True, right_index = True)
    
# Save to text file
with open('local_lender.txt', 'w') as filehandle:
    for listitem in local:
        filehandle.write('%s\n' % listitem)'''

      
# Drop na (variables before 2018)
subset = ['log_min_distance', 'ls', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'pop_area', 'density', 'hhi']
dd_main = dd_main.dropna(subset = subset) 

# Save
dd_main.to_parquet(path = 'data_main.parquet',\
                   engine = 'fastparquet',\
                   compression = 'snappy',\
                   partition_on = ['date'])
dd_main = dd.read_parquet(path = 'data_main.parquet',\
                       engine = 'fastparquet')

#------------------------------------------------------------
# Aggregate to MSA-lender level
#------------------------------------------------------------

# Select only originated loans
df_orig = dd_main[dd_main.loan_originated == 1].compute(scheduler = 'threads')

# Group the data and only select originated loans
df_grouped = df_orig.groupby(['date','cert','msamd'])

# 1) Aggregate portfolio-specific variables
## NOTE 'cb','ln_ta','ln_emp','num_branch' are only unique per date-cert group. Taking the mean of a date-cert-msamd gives us the same as a merge
df_agg = df_grouped[['lti','ln_loanamout','ln_appincome','log_min_distance',\
'log_min_distance_cdd','cb','ln_ta','ln_emp','num_branch']].mean()

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
df_grouped = dd_main.groupby(['date','msamd'])    
msa_data = df_grouped[['density', 'pop_area', 'hhi', 'perc_intsub', 'perc_broadband', 'perc_noint']].mean().reset_index()

df_agg = df_agg.merge(msa_data, how = 'left', on = ['date','msamd'])

# Remove all respective MSA/banks with only one observation
drop_msamd = set(df_agg.msamd.value_counts()[df_agg.msamd.value_counts() > 1].index)
drop_cert = set(df_agg.cert.value_counts()[df_agg.cert.value_counts() > 1].index)

df_agg = df_agg[(df_agg.msamd.isin(drop_msamd)) & (df_agg.cert.isin(drop_cert))]

## Drop na
subset = ['log_min_distance', 'ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'pop_area', 'density', 'hhi', 'ln_mfi']
df_agg.dropna(subset = subset, inplace = True)

# Save (Use apache parquet to reduce file size)
df_agg.to_parquet(path = 'data_agg.parquet',\
                   engine = 'fastparquet',\
                   compression = 'snappy',\
                   partition_on = ['date']) 

#df_agg.to_csv('data_agg.csv', index = False)