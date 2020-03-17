# Calculate distance between all fips
''' This script calculates the distance between all fips in the US.
    This data is later used to calculate the minimum distance between the 
    borrower and the lending bank.
    
    The script calculates the distance between the population centers as 
    specified by the US Census Bureau.
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'C:\Users\mark0\Documents\RUG\PhD\PhD_project2_distance')
#os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np
from itertools import combinations
from numba import cuda, jit, prange, vectorize, guvectorize

#------------------------------------------------------------
# Load the data 
#------------------------------------------------------------

df_msa = pd.read_csv('data_msa_popcenter.csv', index_col = 0, \
                     dtype = {'fips':'str','cbsa_code':'str'}) 
#df_msa = pd.read_csv('Data/data_msa_popcenter.csv', index_col = 0, \
#                     dtype = {'fips':'str','cbsa_code':'str'}) 

#------------------------------------------------------------
# Get the unique fips and the corresponding coordinates
#------------------------------------------------------------

#unique_fips = df_msa.groupby(['date','fips','latitude','longitude']).size().reset_index().rename(columns={0:'count'})
# NOTE: All entries are unique

# Check whether population centers change
#unique_fips = df_msa.groupby(['fips','latitude','longitude']).size().reset_index().rename(columns={0:'count'})
# NOTE: Not the case. 

# Get the unique fips
unique_fips = df_msa.groupby(['fips','latitude','longitude']).size().reset_index().drop(columns = [0])

#------------------------------------------------------------
# Get all unique combinations of fips
#------------------------------------------------------------

unique_combinations = pd.DataFrame([list(map(str, comb)) for comb in combinations(unique_fips.fips.tolist(), 2)],
                                    columns = ['fips_1','fips_2'])

# Add the coordinates
## Fips 1
unique_coorcomb = unique_combinations.merge(unique_fips, how = 'left', left_on = 'fips_1', right_on = 'fips').rename(columns = {'latitude':'lat1','longitude':'long1'})
unique_coorcomb.drop(columns = ['fips'], inplace = True)

## Fips 2
unique_coorcomb = unique_coorcomb.merge(unique_fips, how = 'left', left_on = 'fips_2', right_on = 'fips').rename(columns = {'latitude':'lat2','longitude':'long2'})
unique_coorcomb.drop(columns = ['fips'], inplace = True)

# Make a numpy array with coordinates only (is faster and compatible with numba)
unique_coor1_np = unique_coorcomb[['lat1','long1']].to_numpy()
unique_coor2_np = unique_coorcomb[['lat2','long2']].to_numpy()

#------------------------------------------------------------
# Setup functions to calculate the distances
#------------------------------------------------------------
@jit(nopython=True)
def haversine(s_lat,s_lng,e_lat,e_lng):
    # approximate radius of earth in km
    R = 6372.8

    s_lat = np.deg2rad(s_lat)                    
    s_lng = np.deg2rad(s_lng)     
    e_lat = np.deg2rad(e_lat)                       
    e_lng = np.deg2rad(e_lng)  

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    return 2 * R * np.arcsin(np.sqrt(d))

@jit(nopython=True, parallel=True)
def get_distance_numba(coord1, coord2):

    # initialize
    n = coord1.shape[0]
    output = np.zeros(n, dtype=np.float64)
    
    # prange is an explicit parallel loop, use when operations are independent and no race conditions exist
    for i in prange(n):
        
        # returns an array of length k
        output[i] = haversine(coord1[i,0], coord1[i,1], coord2[i,0], coord1[2,1])

    return output

#------------------------------------------------------------
# get all distances and append to unique_coorcomb
#------------------------------------------------------------

unique_coorcomb['distance'] = get_distance_numba(unique_coor1_np, unique_coor2_np)

#------------------------------------------------------------
# Save the df
#------------------------------------------------------------
# Save
unique_coorcomb.to_csv('data_all_distances.csv.gz', index=False, compression = 'gzip')
#unique_coorcomb.to_csv('Data/data_all_distances.csv.gz', index=False, compression= 'gzip')
