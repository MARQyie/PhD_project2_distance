# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-bank/thrift-level dataset
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
    ### First SDI and LF
    df_sdilf = df_sdi[df_sdi.date == year].merge(df_lf[df_lf['CERT{}'.format(str(year)[2:4])].isin(df_sdi.cert)],\
                            how = 'left', left_on = 'cert',\
                            right_on = 'CERT{}'.format(str(year)[2:4]), indicator = True)
    #### Drop left_only
    df_sdilf = df_sdilf[df_sdilf._merge == 'both'].drop(columns = '_merge')
    
    ### Second SDILF with HMDA
    df_main = df_sdilf.merge(df_hmda, how = 'outer', left_on = 'hmprid', right_on = 'respondent_id',\
                       indicator = True)
