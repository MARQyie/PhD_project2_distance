# Calculate distance between all fips
''' This script calculates the distance between all fips in the US.
    This data is later used to calculate the minimum distance between the 
    borrower and the lending bank.
    
    The script calculates the distance between the population centers as 
    specified by the US Census Bureau.
    
    We do this for the 2000 and 2010 US census. Since our dataset starts in 2004
    we keep that as our first date
    
    The script appends these distances to the county distance database of the
    NBER.
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
#os.chdir(r'C:\Users\mark0\Documents\RUG\PhD\PhD_project2_distance')
os.chdir(r'D:\RUG\PhD\Materials_papers\2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np
from itertools import combinations
from numba import jit, prange

#------------------------------------------------------------
# Load the data 
#------------------------------------------------------------

df_msa = pd.read_csv('Data/data_msa_popcenter.csv', index_col = 0, \
                     dtype = {'fips':'str','cbsa_code':'str'})
df_cdd_2000 = pd.read_csv('Data/sf12000countydistancemiles.csv',\
                     dtype = {'county1':'str','county2':'str'})  
df_cdd_2010 = pd.read_csv('Data/sf12010countydistancemiles.csv',\
                     dtype = {'county1':'str','county2':'str'})

#------------------------------------------------------------
# Prepare df_cdd_2000 and 2010
#------------------------------------------------------------

# Subset data
## Get all unique fips in MSA
msa_fips = df_msa.fips.unique().tolist()

## Only keep rows in which county1 & county2 isin msa_fips
df_cdd_2000 = df_cdd_2000[(df_cdd_2000.county1.isin(msa_fips)) & (df_cdd_2000.county2.isin(msa_fips))]
df_cdd_2010 = df_cdd_2010[(df_cdd_2010.county1.isin(msa_fips)) & (df_cdd_2010.county2.isin(msa_fips))]

# Transform mi to km
mi_to_km = 1 / 0.62137
df_cdd_2000.mi_to_county = df_cdd_2000.mi_to_county * mi_to_km
df_cdd_2010.mi_to_county = df_cdd_2010.mi_to_county * mi_to_km

# Rename columns
df_cdd_2000 = df_cdd_2000.rename(columns = {'county1':'fips_1', 'mi_to_county':'distance_cdd', 'county2':'fips_2'})
df_cdd_2010 = df_cdd_2010.rename(columns = {'county1':'fips_1', 'mi_to_county':'distance_cdd', 'county2':'fips_2'})

#------------------------------------------------------------
# Get the unique fips and the corresponding coordinates
#------------------------------------------------------------

#unique_fips = df_msa.groupby(['date','fips','latitude','longitude']).size().reset_index().rename(columns={0:'count'})
# NOTE: All entries are unique

# Check whether population centers change
#unique_fips = df_msa.groupby(['fips','latitude','longitude']).size().reset_index().rename(columns={0:'count'})
# NOTE: Not the case. 

# Get the unique fips
unique_fips = df_msa.groupby(['date','fips','latitude','longitude']).size().reset_index().drop(columns = [0])

#------------------------------------------------------------
# Attach coordinates to df_cdd
#------------------------------------------------------------

# First make panel
list_df_cdd = []

for year in [2004,2010]:
    if year < 2010:
        df_load = df_cdd_2000.copy()
        df_load['date'] = year
    else:
        df_load = df_cdd_2010.copy()
        df_load['date'] = year
    
    list_df_cdd.append(df_load)
    
df_cdd = pd.concat(list_df_cdd)

# Add the coordinates
## Fips 1
df_cdd = df_cdd.merge(unique_fips, how = 'left', left_on = ['date','fips_1'], right_on = ['date','fips']).rename(columns = {'latitude':'lat1','longitude':'long1'})
df_cdd.drop(columns = ['fips'], inplace = True)

## Fips 2
df_cdd = df_cdd.merge(unique_fips, how = 'left', left_on = ['date','fips_2'], right_on = ['date','fips']).rename(columns = {'latitude':'lat2','longitude':'long2'})
df_cdd.drop(columns = ['fips'], inplace = True)

# Make a numpy array with coordinates only (is faster and compatible with numba)
unique_coor1_np = df_cdd[['lat1','long1']].to_numpy()
unique_coor2_np = df_cdd[['lat2','long2']].to_numpy()

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
        output[i] = haversine(coord1[i,0], coord1[i,1], coord2[i,0], coord2[i,1])

    return output

#------------------------------------------------------------
# get all distances and append to unique_coorcomb
#------------------------------------------------------------

df_cdd['distance'] = get_distance_numba(unique_coor1_np, unique_coor2_np)

# Drop coordinate columns
df_cdd.drop(columns = ['lat1', 'long1', 'lat2', 'long2'], inplace = True)

#------------------------------------------------------------
# Save the df
#------------------------------------------------------------

df_cdd.to_csv('Data/data_all_distances_v3.csv.gz', index = False, compression = 'gzip')

