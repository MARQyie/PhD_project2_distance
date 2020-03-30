# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-lender-level dataset per year
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'E:/2-Working_paper_competition')

# Load packages
import numpy as np
import pandas as pd
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization
from tqdm import tqdm

#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

start = 2010
end = 2017
num_cores = mp.cpu_count()

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Minimum distance       
def minDistanceLenderBorrower(hmda_cert,hmda_fips):
    ''' This methods calculates the minimum distance between a lender (a specific
        branche of a bank or thrift) and a borrower (a unknown individual) based on
        the respective fips codes. Calls upon the distance matrix calculated by the
        haversine formula.
    
    Parameters
    ----------
    hmda : pandas DataFrame 
        one row and n columns
    sod : pandas DataFrame
    distances : pandas DataFrame

    Returns
    -------
    Float

    '''
    
    # Make a subset of branches where the FDIC certificates in both datasets match  
    branches = sod_fips[sod_cert == hmda_cert]
        
    # Get the minimum distance
    try:
        output = np.min(dist_dist[(dist_fips1 == hmda_fips) & (np.isin(dist_fips2,branches))])
    except:
        output = np.nan
    
    return(output)

## Vectorize the function
vecMinDistanceLenderBorrower = np.vectorize(minDistanceLenderBorrower)

#------------------------------------------------------------
# Load all dfs that do not need to be read in the loop
#------------------------------------------------------------

# df MSA
df_msa = pd.read_csv('Data/data_msa_popcenter.csv', index_col = 0, dtype = {'fips':'str','cbsa_code':'str'}) 
        
# df SOD
df_sod = pd.read_csv('Data/df_sod_wp2.csv', index_col = 0, dtype = {'fips':'str'})

# df Distances
df_distances = pd.read_csv('Data/data_all_distances.csv.gz', dtype = {'fips_1':'str','fips_2':'str'})

# df SDI
df_sdi = pd.read_csv('Data/df_sdi_wp2.csv', index_col = 0)

# df LF
## Prelims
#path_lf = r'X:/My Documents/Data/Data_HMDA_lenderfile/'
file_lf = r'Data/hmdpanel17.dta'
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['RSSD{}'.format(str(year)[2:4]) for year in range(start, end +1)]

## Load df LF
df_lf = pd.read_stata(file_lf, columns = vars_lf)

#------------------------------------------------------------
# Loop over the HMDA data and aggregate to MSA-lender level
#------------------------------------------------------------

# Prelims
file_hmda = r'Data/data_hmda_{}.csv.gz'
dtypes_col_hmda = {'msamd':'str','fips':'str'}

# Loop
for year in range(start,end + 1):
    
    ## Read HMDA file
    df_hmda_load = pd.read_csv(file_hmda.format(year), dtype = dtypes_col_hmda)
    
    ## Merge HMDA, LF and SDI
    ### 1) SDI and LF
    df_sdilf = df_sdi[df_sdi.date == year].merge(df_lf[df_lf['CERT{}'.format(str(year)[2:4])].isin(df_sdi.cert)],\
                            how = 'outer', left_on = 'cert',\
                            right_on = 'CERT{}'.format(str(year)[2:4]), indicator = True)
    
    #### Drop left_only in _merge
    df_sdilf = df_sdilf[df_sdilf._merge == 'both'].drop(columns = '_merge')
    
    ### 2) SDILF with HMDA
    df_main = df_sdilf.merge(df_hmda_load, how = 'outer', left_on = ['date','hmprid'],\
                             right_on = ['date','respondent_id'], indicator = True)
    
    #### Del df_sdilf
    del df_sdilf
    
    #### Drop left_only in _merge
    df_main = df_main[df_main._merge == 'both'].drop(columns = '_merge')
    
    ## Only keep MSA that are in the df_msa
    df_main = df_main[df_main.fips.isin(df_msa[df_msa.date == year].fips.unique())]
    
    ## Remove all certs in df_main that are not in df_sod
    df_main = df_main[df_main.cert.isin(df_sod.cert)]
    
    ## Calculate the minimum lender-borrower distance for each entry
    ### Reduce the dimension of df_distances 
    df_distances_load = df_distances[(df_distances.fips_1.isin(df_hmda_load.fips.unique())) &\
                 (df_distances.fips_2.isin(df_sod.fips.unique()))]
    
    ### Get the unique combinations of fips/cert in df_main
    unique_fipscert_hmda = df_main.groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0]).to_numpy().T
    
    ### Set other numpy arrays
    sod_fips = df_sod[df_sod.date == year].fips.to_numpy(dtype = str)
    sod_cert = df_sod[df_sod.date == year].cert.to_numpy(dtype = float)
    dist_fips1 = df_distances_load['fips_1'].to_numpy(dtype = str)
    dist_fips2 = df_distances_load['fips_2'].to_numpy(dtype = str)
    dist_dist = df_distances_load['distance'].to_numpy(dtype = float)
    
    ### Parallelize the function and calculate distance
    inputs = tqdm(zip(unique_fipscert_hmda[1], unique_fipscert_hmda[0]))

    if __name__ == "__main__":
        distance_list = Parallel(n_jobs=num_cores)(delayed(minDistanceLenderBorrower)(cert, fips) for cert, fips in inputs)     
           
    ### Merge distance on fips and cert
    df_main = df_main.merge(unique_fipscert_hmda, how = 'left', on = ['fips','cert'])
    
    ## Delete some data to save RAM
    del df_hmda_load, df_distances_load, unique_fipscert_hmda, sod_fips, sod_cert,\
        dist_fips1, dist_fips2, dist_dist
    
    #------------------------------------------------------------
    # Aggregate data to MSA-lender-level
    
    # 1) Aggregate portfolio-specific variables
    df_agg = df_main.groupby(['date','cert','msamd'])[['lti','ln_loanamout','ln_appincome','distance']].mean()
    df_agg['subprime'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.subprime) / len(x.subprime))
    df_agg['secured'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.secured) / len(x.secured))
    
    # 2) Add Loan sales variables
    ## Fraction based on number of loans
    df_agg['ls_num'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.ls) / len(x.ls))
    df_agg['ls_gse_num'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.ls_gse) / len(x.ls_gse))
    df_agg['ls_priv_num'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.ls_priv) / len(x.ls_priv))
    df_agg['sec_num'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.sec) / len(x.sec))
    
    ## Fraction based on value of loans
    df_agg['ls_val'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.ls * x.loan_amount_000s) \
                                                               / np.sum(x.loan_amount_000s))
    df_agg['ls_gse_val'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.ls_gse * x.loan_amount_000s) \
                                                                   / np.sum(x.loan_amount_000s))
    df_agg['ls_priv_val'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.ls_priv * x.loan_amount_000s) \
                                                                    / np.sum(x.loan_amount_000s))
    df_agg['sec_val'] = df_main.groupby(['date','cert','msamd']).apply(lambda x: np.sum(x.sec * x.loan_amount_000s) \
                                                                / np.sum(x.loan_amount_000s))   
    ## Dummy
    df_agg['ls_dum' ] = (df_main.groupby(['date','cert','msamd']).ls.sum() > 0.0) * 1
    df_agg['ls_gse_dum' ] = (df_main.groupby(['date','cert','msamd']).ls_gse.sum() > 0.0) * 1
    df_agg['ls_priv_dum' ] = (df_main.groupby(['date','cert','msamd']).ls_priv.sum() > 0.0) * 1
    df_agg['sec_dum' ] = (df_main.groupby(['date','cert','msamd']).sec.sum() > 0.0) * 1
    
    # 3) Bank/Thrift level controls
    ## Reset index before merge
    df_agg.reset_index(inplace = True)
        
    ## add ln_ta, ln_emp and bank indicator
    df_agg = df_agg.merge(df_sdi[df_sdi.date == year][['cert','cb','ln_ta','ln_emp']],\
                          how = 'left', on = ['date','cert'])
    
    ## Add number of branches
    num_branches = df_sod[df_sod.date == year].groupby(['date','cert']).CERT.count().rename('num_branch')
    df_agg = df_agg.merge(num_branches, how = 'left', left_on = ['date','cert'],\
                          right_on = [num_branches.index[0], num_branches.index[1]])
    
    # 4) MSA level
    # from df_msa
    df_msa_agg = df_msa[df_msa.date == year].groupby(['date','cbsa_code'])[['ln_pop', 'density', 'hhi', 'ln_mfi', 'dum_min']].mean()
    df_agg = df_agg.merge(df_msa_agg, how = 'left', left_on = ['date','msamd'], right_on = [df_msa_agg.index[0], df_msa_agg.index[1]])
    
    # Add average lending distance per MSA
    distance_msa = df_main.groupby('msamd').distance.apply(lambda x: np.log(1 + x.mean())).rename('mean_distance')
    df_agg = df_agg.merge(distance_msa, how = 'left', left_on = 'msamd', right_on = distance_msa.index)
    
    #------------------------------------------------------------
    # Save df_agg
    
    df_agg.to_csv('Data/data_agg_{}.csv.gz'.format(year), index = False, compression = 'gzip')