# Calculate the minimum distance between all CERT and fips
''' This script calculates the minimum distances between a certain fin. institution
    cert and all possible fips. This data can then be merged with the final 
    dataset. The script is parallelized where possible. Do this for the 2000 
    and 2010 US Census.
    
    INPUT
    -----
    
    OUTPUT
    -----
    Dataframe with cert, fips(, year) and minimum distance
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory

import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Data')

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
#num_cores = mp.cpu_count()
num_cores = -1

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Function for parallelization pandas read_csv for the HMDA data
def readHMDA(year):
    
    filename = 'data_hmda_{}.csv'.format(year)
    
    return (pd.read_csv(filename,\
                       dtype = {'msamd':'uint16', 'fips':'uint16'}, 
                       usecols = ['date', 'respondent_id', 'fips']))
    
# Merge SDI LF   
def mergeSDILF(year):
       
    ## Merge on year
    if year < 2018:
        df_load = df_sdi[df_sdi.date == year].merge(df_lf,\
                            how = 'left', left_on = 'cert',\
                            right_on = 'CERT{}'.format(str(year)[2:4]))
    else:
        df_load = df_sdi[df_sdi.date == year].merge(df_lf,\
                            how = 'left', left_on = 'cert',\
                            right_on = 'CERT{}'.format(str(17)))    
    
    ## Return concatenated pd DataFrame
    return (df_load)

# Minimum distance       
def minDistanceLenderBorrower(hmda_fips, hmda_cert, sod_fips, sod_cert, dist_fips1, dist_fips2, dist_dist, dist_dist_cdd):
    ''' This methods calculates the minimum distance between a lender (a specific
        branche of a bank or thrift) and a borrower (a unknown individual) based on
        the respective fips codes. Calls upon the distance matrix calculated by the
        haversine formula.
    '''

    # Make a subset of branches where the FDIC certificates in both datasets match  
    branches = sod_fips[sod_cert == hmda_cert]
    
    # If lender and borrower are in the same county
    if hmda_fips in branches:
        return 0.0, 0.0
        
    # Get the minimum distance
    boolean = (dist_fips1 == hmda_fips) & (np.isin(dist_fips2,branches))
          
    if np.any(boolean):
        dist_list = dist_dist[boolean]
        dist_cdd_list = dist_dist_cdd[boolean]
        
        return min(dist_list), min(dist_cdd_list)
      
    else:
        return np.nan, np.nan
    
#------------------------------------------------------------
# Load the dfs
#------------------------------------------------------------
    
# Distance df
df_distances = pd.read_csv('data_all_distances_v3.csv',\
                           dtype = {'fips_1':'uint16','fips_2':'uint16'})
   
# SOD df
df_sod = pd.read_csv('df_sod_wp2.csv', index_col = 0, dtype = {'fips':'uint16'},\
                     usecols = ['date','fips','cert'])

# df SDI
df_sdi = pd.read_csv('df_sdi_wp2.csv', usecols = ['date','cert'])

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
    list_sdilf = Parallel(n_jobs=num_cores, prefer="threads")(delayed(mergeSDILF)(year) for year in range(start,end + 1)) 

## Concat to pd DataFrame
df_sdilf = pd.concat(list_sdilf, ignore_index = True)

## Drop CERT.. columns and drop nans in hmprid
df_sdilf = df_sdilf[df_sdilf.columns[~df_sdilf.columns.str.contains('CERT')]].dropna(subset = ['hmprid'])

# Load HMDA
if __name__ == "__main__":
    list_hmda = Parallel(n_jobs=num_cores, prefer="threads")(delayed(readHMDA)(year) for year in range(start,end + 1)) 

## Concat to pd DataFrame
df_hmda = pd.concat(list_hmda, ignore_index = True) 

#------------------------------------------------------------
# Get all possible cert-fips combinations
#------------------------------------------------------------
# Merge HMDA with SDILF
df_main = df_sdilf.merge(df_hmda, how = 'inner', left_on = ['date','hmprid'],\
                             right_on = ['date','respondent_id'])

# Get the unique combinations of fips/cert in df_main
hmda_fips_2004, hmda_cert_2004 = np.array(df_main[df_main.date < 2010].groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0])).T
hmda_fips_2010, hmda_cert_2010 = np.array(df_main[df_main.date >= 2010].groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0])).T


# Get the unique combinations of fips/cert in df_sod
sod_fips_2004, sod_cert_2004 = np.array(df_sod[df_sod.date < 2010].groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0])).T
sod_fips_2010, sod_cert_2010 = np.array(df_sod[df_sod.date >= 2010].groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0])).T
    
# Split distances and reduce dimensions
df_distances_2004 = df_distances.loc[(df_distances.fips_1.isin(df_hmda.fips.unique())) &\
                 (df_distances.fips_2.isin(df_sod.fips.unique())) & (df_distances.date < 2010)]
df_distances_2010 = df_distances.loc[(df_distances.fips_1.isin(df_hmda.fips.unique())) &\
                 (df_distances.fips_2.isin(df_sod.fips.unique())) & (df_distances.date >= 2010)]

## Make np_array out of it
dist_fips1_2004, dist_dist_cdd_2004, dist_fips2_2004, _, dist_dist_2004 = np.array(df_distances_2004).T
dist_fips1_2010, dist_dist_cdd_2010, dist_fips2_2010, _, dist_dist_2010 = np.array(df_distances_2010).T

#------------------------------------------------------------
# Calculate distances
#------------------------------------------------------------

# Set inputs 
inputs_2004 = zip(hmda_fips_2004, hmda_cert_2004)
inputs_2010 = zip(hmda_fips_2010, hmda_cert_2010)

kwargs_2004 = [sod_fips_2004, sod_cert_2004, dist_fips1_2004, dist_fips2_2004, dist_dist_2004, dist_dist_cdd_2004]
kwargs_2010 = [sod_fips_2010, sod_cert_2010, dist_fips1_2010, dist_fips2_2010, dist_dist_2010, dist_dist_cdd_2010]

if __name__ == "__main__":
    distance_list_2004, distance_cdd_list_2004 = zip(*Parallel(n_jobs=num_cores, prefer="threads")(delayed(minDistanceLenderBorrower)(fips, cert, *kwargs_2004) for fips, cert in inputs_2004))
    distance_list_2010, distance_cdd_list_2010 = zip(*Parallel(n_jobs=num_cores, prefer="threads")(delayed(minDistanceLenderBorrower)(fips, cert, *kwargs_2010) for fips, cert in inputs_2010))       
           
#------------------------------------------------------------
# Make a pandas DataFrame and save to csv
#------------------------------------------------------------
# Make pandas DataFrame
df_2004 = pd.DataFrame({'fips':hmda_fips_2004, 'cert':hmda_cert_2004, 'min_distance':distance_list_2004, 'min_distance_cdd':distance_cdd_list_2004})
df_2010 = pd.DataFrame({'fips':hmda_fips_2010, 'cert':hmda_cert_2010, 'min_distance':distance_list_2010, 'min_distance_cdd':distance_cdd_list_2010})

# Save df
df_2004.to_csv('data_min_distances_2004.csv', index = False)
df_2010.to_csv('data_min_distances_2010.csv', index = False)




