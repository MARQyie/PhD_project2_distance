# Calculate the minimum distance between all CERT and fips
''' This script calculates the minimum distances between a certain fin. institution
    cert and all possible fips. This data can then be merged with the final 
    dataset. The script is parallelized where possible.
    
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

start = 2010
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
def minDistanceLenderBorrower(hmda_fips, hmda_cert):
    ''' This methods calculates the minimum distance between a lender (a specific
        branche of a bank or thrift) and a borrower (a unknown individual) based on
        the respective fips codes. Calls upon the distance matrix calculated by the
        haversine formula.
    '''
    global sod_fips, sod_cert, dist_fips1, dist_fips2, dist_dist
    
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
df_distances = pd.read_csv('data_all_distances_v2.csv',\
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
hmda_fips, hmda_cert = np.array(df_main.groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0])).T

# Get the unique combinations of fips/cert in df_sod
sod_fips, sod_cert = np.array(df_sod.groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0])).T

# Reduce the dimension of df_distances 
df_distances = df_distances[(df_distances.fips_1.isin(df_hmda.fips.unique())) &\
                 (df_distances.fips_2.isin(df_sod.fips.unique()))]

## Make np_array out of it
dist_fips1, dist_dist_cdd, dist_fips2, dist_dist = np.array(df_distances).T

#------------------------------------------------------------
# Calculate distances
#------------------------------------------------------------

# Set inputs 
inputs = zip(hmda_fips, hmda_cert)

if __name__ == "__main__":
    distance_list, distance_cdd_list = zip(*Parallel(n_jobs=num_cores, prefer="threads")(delayed(minDistanceLenderBorrower)(fips, cert) for fips, cert in inputs))     
           
#------------------------------------------------------------
# Make a pandas DataFrame and save to csv
#------------------------------------------------------------
# Make pandas DataFrame
df = pd.DataFrame({'fips':hmda_fips, 'cert':hmda_cert, 'min_distance':distance_list, 'min_distance_cdd':distance_cdd_list})

# Save df
df.to_csv('data_min_distances.csv', index = False)




