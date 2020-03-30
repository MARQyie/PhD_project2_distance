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
'''
import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')
'''

# Load packages
import numpy as np
import pandas as pd
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

import sys
# Check usage
if len(sys.argv) < 2:
    print("Usage: {} <csv file>".format(sys.argv[0]))
    sys.exit()

#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

start = 2010
end = 2017
num_cores = mp.cpu_count()

# Fetch csvfiles
csv_distances = sys.argv[1]
csv_sod = sys.argv[2]
csv_sdi = sys.argv[3]
csv_lf = sys.argv[4]

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Function for parallelization pandas read_csv for the HMDA data
def readHMDA(year):
    
    idx = year - start + 5
    filename = sys.argv[idx]
    
    return (pd.read_csv(filename, compression = 'gzip',\
                       dtype = {'msamd':'str', 'fips':'str'}, 
                       usecols = ['date', 'respondent_id', 'fips']))
    
# Merge SDI LF   
def mergeSDILF(year):
       
    ## Merge on year
    df_load = df_sdi[df_sdi.date == year].merge(df_lf,\
                            how = 'left', left_on = 'cert',\
                            right_on = 'CERT{}'.format(str(year)[2:4]))
    
    
    ## Return concatenated pd DataFrame
    return (df_load)

# Minimum distance       
def minDistanceLenderBorrower(hmda_fips, hmda_cert):
    ''' This methods calculates the minimum distance between a lender (a specific
        branche of a bank or thrift) and a borrower (a unknown individual) based on
        the respective fips codes. Calls upon the distance matrix calculated by the
        haversine formula.
    '''
    
    # Make a subset of branches where the FDIC certificates in both datasets match  
    branches = sod_fips[sod_cert == hmda_cert]
        
    # Get the minimum distance
    try:
        output = np.min(dist_dist[(dist_fips1 == hmda_fips) & (np.isin(dist_fips2,branches))])
    except:
        output = np.nan
    
    return(output)
#------------------------------------------------------------
# Load the dfs
#------------------------------------------------------------
    
# Distance df
df_distances = pd.read_csv(csv_distances,\
                           dtype = {'fips_1':'str','fips_2':'str'},
                           usecols = ['fips_1', 'fips_2', 'distance'])
   
# SOD df
df_sod = pd.read_csv(csv_sod, index_col = 0, dtype = {'fips':'str'},\
                     usecols = ['date','fips','cert'])

# df SDI
df_sdi = pd.read_csv('Data/df_sdi_wp2.csv', usecols = ['date','cert'], dtype = {'cert':'float64'})

# df LF
## Prelims
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end +1)]

## Load df LF
df_lf = pd.read_stata(csv_lf, columns = vars_lf)

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

#------------------------------------------------------------
# Get all possible cert-fips combinations
#------------------------------------------------------------
# Merge HMDA with SDILF
df_main = df_sdilf.merge(df_hmda, how = 'left', left_on = ['date','hmprid'],\
                             right_on = ['date','respondent_id'])

# Get the unique combinations of fips/cert in df_main
hmda_fips, hmda_cert = df_main.groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0]).to_numpy().T
                           
# Get the unique combinations of fips/cert in df_sod
sod_fips, sod_cert = df_sod.groupby(['fips','cert']).size().reset_index().\
                           drop(columns = [0]).to_numpy().T

# Reduce the dimension of df_distances 
df_distances = df_distances[(df_distances.fips_1.isin(df_hmda.fips.unique())) &\
                 (df_distances.fips_2.isin(df_sod.fips.unique()))]

## Make np_array out of it
dist_fips1, dist_fips2, dist_dist = df_distances.to_numpy().T

#------------------------------------------------------------
# Calculate distances
#------------------------------------------------------------

# Set inputs 
inputs = zip(hmda_fips, hmda_cert)

if __name__ == "__main__":
    distance_list = Parallel(n_jobs=num_cores)(delayed(minDistanceLenderBorrower)(cert, fips) for cert, fips in inputs)     
           
#------------------------------------------------------------
# Make a pandas DataFrame and save to csv
#------------------------------------------------------------
# Make pandas DataFrame
df = pd.DataFrame([hmda_fips, hmda_cert, distance_list], columns = ['fips','cert','min_distance'])

# Save df
df.to_csv('/data/p285325/WP2_distance/data_min_distances.csv.gz', index = False, compression = 'gzip')




