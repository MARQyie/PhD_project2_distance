# Setup df internet and population
# Setup the Summary of Deposits dataframe
'''
    This script sets up the total dataframe for SODs.
    
    The script does the following
    1. Reads in the SOD dataframes for each year-end from 2010 to 2017
    2. Select the relevant variables
    3. Clean up the dataframe
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'D:\RUG\PhD\Materials_papers\2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

#------------------------------------------------------------
# Set Parameters
#------------------------------------------------------------

start = 2010
end = 2019

#------------------------------------------------------------
# Load the data 
#------------------------------------------------------------

#------------------------------------------------------------
# df_internet

path_int = r'D:/RUG/Data/US_Census_Bureau/Internet_subscriptions/County/'
file_int = r'ACSDT1Y{}.B28002_data_with_overlays_2020-05-04T063923.csv'
usecols = ['GEO_ID', 'NAME', 'B28002_001E', 'B28002_002E', 'B28002_003E', 'B28002_004E',\
           'B28002_007E', 'B28002_010E', 'B28002_013E', 'B28002_016E',\
           'B28002_020E', 'B28002_021E']
usecols_from2016 = ['GEO_ID', 'NAME', 'B28002_001E', 'B28002_002E','B28002_003E', 'B28002_004E', 'B28002_012E', 'B28002_013E']


list_df_int = []
for year in range(start + 3, end):
    if year < 2016:
        df_int_load =  pd.read_csv(path_int + file_int.format(year), skiprows = [1], usecols = usecols)
        df_int_load['broadband_load'] = df_int_load[['B28002_007E', 'B28002_010E', 'B28002_013E', 'B28002_016E']].sum(axis = 1)
        df_int_load.drop(columns = ['B28002_007E', 'B28002_010E', 'B28002_013E', 'B28002_016E'], inplace = True)
    else:
        df_int_load =  pd.read_csv(path_int + file_int.format(year), skiprows = [1], usecols = usecols_from2016)
    df_int_load['date'] = year
    list_df_int.append(df_int_load)

df_int = pd.concat(list_df_int)

## drop columns, make columns and rename the rest
df_int['broadband'] = df_int[['B28002_004E', 'broadband_load']].sum(axis = 1)
df_int['int_nosub'] = df_int[['B28002_020E', 'B28002_012E']].sum(axis = 1)
df_int['noint']  = df_int[['B28002_021E', 'B28002_013E']].sum(axis = 1)

df_int.drop(columns = ['B28002_004E', 'broadband_load', 'B28002_020E', 'B28002_012E', 'B28002_021E', 'B28002_013E'], inplace = True)

names = ['geo_id', 'name', 'int_total', 'int_sub', 'dailup','date', 'broadband', 'int_nosub', 'noint']
df_int.columns = names

#------------------------------------------------------------
# df_pop 

path_pop = r'D:/RUG/Data/US_Census_Bureau/Population_estimates/'
file_pop = r'co-est2019-alldata.csv'
usecols = ['STATE', 'COUNTY'] + ['POPESTIMATE{}'.format(year) for year in range(start, end + 1)]
df_pop = pd.read_csv(path_pop + file_pop, encoding = "ISO-8859-1", usecols = usecols)

# Drop state only obs
df_pop = df_pop[df_pop.COUNTY != 0]

# Merge State and county to fips. Drop State and county columns
df_pop['fips'] = df_pop.STATE.astype(str).str.zfill(2) + df_pop.COUNTY.astype(str).str.zfill(3)
df_pop.drop(columns = ['STATE', 'COUNTY'], inplace = True)

# Reshape the data
df_pop.set_index('fips', inplace = True)
df_pop_res = df_pop.stack().reset_index()
df_pop_res.rename(columns = {'level_1':'date', 0:'population'}, inplace = True)
df_pop_res['date'] = [year for year in range(start, end + 1)] * df_pop_res.fips.nunique()

#------------------------------------------------------------
# df_area
path_area = r'D:/RUG/Data/US_Census_Bureau/State-based_files/'
file_area = r'State-based_files_census2010.csv'
df_area = pd.read_csv(path_area + file_area, dtype = {'GEOID':'str'})

# Correct GEOID
df_area.GEOID = df_area.GEOID.str.zfill(5)

#------------------------------------------------------------
# Add variables 
#------------------------------------------------------------

#------------------------------------------------------------
# df_internet

# fips
df_int['fips'] = df_int.geo_id.str[-5:]

# Percentage with internet subscription
df_int['perc_intsub'] = df_int.int_sub / df_int.int_total

# Percentage fast internet subscription
df_int['perc_broadband'] = df_int.broadband / df_int.int_total

# Percentage no internet acces
df_int['perc_noint'] = df_int.noint / df_int.int_total

# Drop columns
df_int.drop(columns = ['geo_id', 'name'], inplace = True)

#------------------------------------------------------------
# df_area

# Make area var
df_area['area'] = df_area.AREALAND.divide(1e6)

#------------------------------------------------------------
# df_pop

# Correct population with area
df_pop_area = df_pop_res.merge(df_area[['GEOID', 'area']], how = 'inner', left_on = 'fips', right_on = 'GEOID')
df_pop_area['pop_area'] = df_pop_area['population'] / df_pop_area['area']
df_pop_area.drop(columns = ['GEOID', 'area'], inplace = True)

#------------------------------------------------------------
# Save dfs
#------------------------------------------------------------
df_pop_area.to_csv('Data/data_pop.csv')
df_int.to_csv('Data/data_internet.csv')