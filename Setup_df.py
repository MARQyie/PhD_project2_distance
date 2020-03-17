# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-lender-level dataset per year
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

start = 2010
end = 2017

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Minimum distance

def minDistanceLenderBorrower(hmda_cert,hmda_fips,msa_fips,sod_stcntybr,sod_cert,distances):
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
    branches = sod_stcntybr[sod_cert == hmda_cert]
        
    # Get the minimum distance
    if not hmda_cert in sod_cert:
        return (np.nan)
    elif hmda_fips in branches:
        return (0.0)
    else:
        output = np.min(distances[(distances.fips_1 == hmda_fips) & (distances.fips_2.isin(branches))].distance)

        return(output)

#------------------------------------------------------------
# Load all dfs that do not need to be read in the loop
#------------------------------------------------------------

# df MSA
df_msa = pd.read_csv('Data/data_msa_popcenter.csv', index_col = 0, dtype = {'fips':'str','cbsa_code':'str'}) 
        
# df Distances
df_distances = pd.read_csv('Data/data_all_distances.csv.gz', dtype = {'fips_1':'str','fips_2':'str'})

# df SOD
df_sod = pd.read('Data/df_sod_wp2.csv', index_col = 0, dtype = {'fips':'str'})

# df SDI
df_sdi = pd.read('Data/df_sdi_wp2.csv', index_col = 0)

# df LF
## Prelims
path_lf = r'X:/My Documents/Data/Data_HMDA_lenderfile/'
file_lf = r'hmdpanel17.dta'
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['RSSD{}'.format(str(year)[2:4]) for year in range(start, end +1)]

## Load df LF
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)

#------------------------------------------------------------
# Loop over the HMDA data and aggregate to MSA-lender level
#------------------------------------------------------------

# Prelims
file_hmda = r'Data/data_hmda_{}.csv.gz'
dtypes_col_hmda = {'msamd':'str'}

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
    
    #### Drop left_only in _merge
    df_main = df_main[df_main._merge == 'both'].drop(columns = '_merge')
    
    ## Only keep MSA that are in the df_msa
    df_main = df_main[df_main.fips.isin(df_msa[df_msa.date == year].fips.unique())]
    
    ## Calculate the minimum lender-borrower distance for each entry
    df_main['distance'] = df_main.apply(lambda data: \
           minDistanceLenderBorrower(data.cer, data.fips, df_msa.fips, df_sod.fips,\
                                     df_sod.cert, df_distances), axis = 1)
    
    ## Delete df_hmda_load to save RAM
    del df_hmda_load
    
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