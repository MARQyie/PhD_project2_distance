# Make County/MSA file

''' This files makes a dataframe will all county fips and matches the to
    MSA yes/no. We then add the lat and long of the center of population of each county
    
    We get all county fips and the centers of population from the US Census Bureau
        https://www.census.gov/geographies/reference-files/time-series/geo/centers-population.html
   
    For all the MSA codes, we take data from the US census bureau (delineation files). The files
    hold from the publishing data until the next file
'''

#------------------------------------------------------------
# Prelims
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np
import requests

from bs4 import BeautifulSoup
from urllib.request import urlopen

#------------------------------------------------------------
# Load Centers of Population from the US Census Bureau (2010 census)
#------------------------------------------------------------

# Prelims
url = 'https://www2.census.gov/geo/docs/reference/cenpop2010/county/CenPop2010_Mean_CO{}.txt'
dtypes_centers = {'STATEFP':'str','COUNTYFP':'str'}
range_statecodes = range(1,56+1)
list_df_load = []

# Loop over all dfs
for state in range_statecodes:
    ''' Loop over all state codes. Since the codes are not a perfect sequence,
        we use a try-except structure.
        
        Next we create a single fips variable.
    '''
    try:
        df_load = pd.read_csv(url.format(str(state).zfill(2)), dtype = dtypes_centers)
    except:
        continue
    
    df_load['fips'] = df_load.STATEFP + df_load.COUNTYFP
    df_load.drop(columns = ['STATEFP','COUNTYFP'], inplace = True)
    
    list_df_load.append(df_load)

# Concat all dfs    
df_centers = pd.concat(list_df_load)

# Force lowercase in columns
df_centers.columns = map(str.lower, df_centers.columns)

#------------------------------------------------------------
# Load all MSA delineation files from the US Census Bureau
#------------------------------------------------------------
# Prelims
path_msa = r'X:/My Documents/Data/Data_MSA/'
col_names_msa = ['cbsa_code', 'metro_div', 'csa_code', 'cbsa_title', 'level_cbsa',\
             'status', 'metro_div_title', 'csa_title', 'comp_name', 'state',\
             'fips', 'county_status']
dtypes_msa = {'fips':'str'}

# Load the dfs
df_msa2009 = pd.read_excel(path_msa + 'msa_2009.xls', header = 3, names = col_names_msa, dtype = dtypes_msa)
df_msa2013 = pd.read_excel(path_msa + 'msa_2013.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)
df_msa2015 = pd.read_excel(path_msa + 'msa_2015.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)
df_msa2017 = pd.read_excel(path_msa + 'msa_2017.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)
df_msa2018 = pd.read_excel(path_msa + 'msa_2018.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)

#------------------------------------------------------------
# Make a panel data set
#------------------------------------------------------------

# Add date to df_centers and make a new df
list_df_center = []

for year in range(10,18+1):  
    list_df_center.append(pd.concat([df_centers, pd.DataFrame(['20{}'.format(year)] * df_centers.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    
df_centers_panel = pd.concat(list_df_center)

# Make a panel from the msa dfs
list_df_msa = []

for year in range(10,18+1):
    if np.isin(year,[10,11,12]):
        list_df_msa.append(pd.concat([df_msa2009, pd.DataFrame(['20{}'.format(year)] * df_msa2009.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif np.isin(year,[13,14]):
        list_df_msa.append(pd.concat([df_msa2013, pd.DataFrame(['20{}'.format(year)] * df_msa2013.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif np.isin(year,[15,16]):
        list_df_msa.append(pd.concat([df_msa2015, pd.DataFrame(['20{}'.format(year)] * df_msa2015.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif year == 17:
        list_df_msa.append(pd.concat([df_msa2017, pd.DataFrame(['20{}'.format(year)] * df_msa2017.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    else:
        list_df_msa.append(pd.concat([df_msa2018, pd.DataFrame(['20{}'.format(year)] * df_msa2018.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))

df_msa_panel = pd.concat(list_df_msa)

#------------------------------------------------------------
# Merge the data sets and save to csv
#------------------------------------------------------------
# Merge the dfs
df = df_centers_panel.merge(df_msa_panel, how = 'left', on = ['fips','date'])

## Drop double colums
df.drop(columns = ['comp_name','state'], inplace = True)

# Save df
df.to_csv('Data/data_msa_popcenter.csv')




























'''OLD
#------------------------------------------------------------
# Scrape web page of USDA to get all county fips
#------------------------------------------------------------
# Prelim
## Set and open URL
url = 'https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697'
page = urlopen(url)

## Read the page with beautiful soup and find the table
soup = BeautifulSoup(page)
msa_table_html = soup.find('table',{'class':'data'})
#print(msa_table_html)

## Make a pd.DataFrame
table_rows = msa_table_html.find_all('tr')
table_vector = []

for tr in table_rows:
    td = tr.find_all('td')
    row = [tr.text.replace('\r\n\t\t\t\t','') for tr in td] # This webpage adds weird, extra strings. Remove them
    table_vector.append(row)

df_allfips = pd.DataFrame(table_vector,columns = ['fips','name','state']).dropna().reset_index(drop = True)
'''

