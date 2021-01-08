# Make County/MSA file

''' This files loads all US Census 2010 population center files and makes one df
    from it. 
    
    We get all county fips and the centers of population from the US Census Bureau
    https://www.census.gov/geographies/reference-files/time-series/geo/centers-population.html
'''

#------------------------------------------------------------
# Prelims
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np
#import requests
#
#from bs4 import BeautifulSoup
#from urllib.request import urlopen

#------------------------------------------------------------
# Load Centers of Population from the US Census Bureau (2000 and 2010 census)
#------------------------------------------------------------

# Prelims
url2000 = 'https://www2.census.gov/geo/docs/reference/cenpop2000/county/cou_{}_{}.txt'
url2010 = 'https://www2.census.gov/geo/docs/reference/cenpop2010/county/CenPop2010_Mean_CO{}.txt'
columns_2000 = ['STATEFP','COUNTYFP','COUNAME','POPULATION','LATITUDE','LONGITUDE']
dtypes_centers = {'STATEFP':'str','COUNTYFP':'str'}
statecodes = list(range(1,56+1))

for elem in statecodes:
    if elem in [3,7,14,43,52]:
        statecodes.remove(elem)
        
states =[x.lower() for x in ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]]
list_df_load_2000 = []
list_df_load_2010 = []

# Loop over all dfs
for state, code in zip(states,statecodes):
    ''' Loop over all state codes. Since the codes are not a perfect sequence,
        we use a try-except structure.
        
        Next we create a single fips variable.
    '''
    try:
        df_load_2000 = pd.read_csv(url2000.format(str(code).zfill(2), state), dtype = dtypes_centers, encoding = 'ISO-8859-1', names = columns_2000)
        df_load_2010 = pd.read_csv(url2010.format(str(code).zfill(2)), dtype = dtypes_centers, encoding = 'ISO-8859-1')
    except:
        continue
    
    df_load_2000['fips'] = df_load_2000.STATEFP + df_load_2000.COUNTYFP
    df_load_2000.drop(columns = ['STATEFP','COUNTYFP'], inplace = True)
    df_load_2010['fips'] = df_load_2010.STATEFP + df_load_2010.COUNTYFP
    df_load_2010.drop(columns = ['STATEFP','COUNTYFP'], inplace = True)
    
    list_df_load_2000.append(df_load_2000)
    list_df_load_2010.append(df_load_2010)

# Concat all dfs    
df_centers_2000 = pd.concat(list_df_load_2000)
df_centers_2010 = pd.concat(list_df_load_2010)

# Force lowercase in columns
df_centers_2000.columns = map(str.lower, df_centers_2000.columns)
df_centers_2010.columns = map(str.lower, df_centers_2010.columns)

#------------------------------------------------------------
# Load all MSA delineation files from the US Census Bureau
#------------------------------------------------------------
# Prelims
path_msa = r'D:/RUG/Data/Data_MSA/'
col_names_msa = ['cbsa_code', 'metro_div', 'csa_code', 'cbsa_title', 'level_cbsa',\
             'status', 'metro_div_title', 'csa_title', 'comp_name', 'state',\
             'fips', 'county_status']
dtypes_msa = {'fips':'str','state':'str'}

# Load the dfs
df_msa2003 = pd.read_excel(path_msa + 'msa_2003.xls', header = 2, names = col_names_msa[:-1], dtype = dtypes_msa)
df_msa2004 = pd.read_excel(path_msa + 'msa_2004.xls', header = 7, names = col_names_msa[:-1], dtype = dtypes_msa)
df_msa2005 = pd.read_excel(path_msa + 'msa_2005.xls', header = 3, names = col_names_msa[:-1], dtype = dtypes_msa)
df_msa2006 = pd.read_excel(path_msa + 'msa_2006.xls', header = 3, names = col_names_msa[:-1], dtype = dtypes_msa)
df_msa2007 = pd.read_excel(path_msa + 'msa_2007.xls', header = 3, names = col_names_msa, dtype = dtypes_msa)
df_msa2008 = pd.read_excel(path_msa + 'msa_2008.xls', header = 3, names = col_names_msa, dtype = dtypes_msa)
df_msa2009 = pd.read_excel(path_msa + 'msa_2009.xls', header = 3, names = col_names_msa, dtype = dtypes_msa)
df_msa2013 = pd.read_excel(path_msa + 'msa_2013.xls', header = 2, names = col_names_msa, dtype = dtypes_msa) # FIPS change
df_msa2015 = pd.read_excel(path_msa + 'msa_2015.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)
df_msa2017 = pd.read_excel(path_msa + 'msa_2017.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)
df_msa2018 = pd.read_excel(path_msa + 'msa_2018.xls', header = 2, names = col_names_msa, dtype = dtypes_msa)

# Fix fips >= 2013
df_msa2013['fips'] = df_msa2013.state + df_msa2013.fips
df_msa2015['fips'] = df_msa2015.state + df_msa2015.fips
df_msa2017['fips'] = df_msa2017.state + df_msa2017.fips
df_msa2018['fips'] = df_msa2018.state + df_msa2018.fips

#------------------------------------------------------------
# Make a panel data set
#------------------------------------------------------------

# Add date to df_centers and make a new df
list_df_center = []

for year in range(4,19+1):
    if year < 10:
        list_df_center.append(pd.concat([df_centers_2000, pd.DataFrame(['200{}'.format(year)] * df_centers_2000.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    else:
        list_df_center.append(pd.concat([df_centers_2010, pd.DataFrame(['20{}'.format(year)] * df_centers_2010.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    
df_centers_panel = pd.concat(list_df_center)

# Make a panel from the msa dfs
list_df_msa = []

for year in range(4,19+1):
    if year == 4:
        list_df_msa.append(pd.concat([df_msa2003, pd.DataFrame(['200{}'.format(year)] * df_msa2003.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif year == 5:
        list_df_msa.append(pd.concat([df_msa2004, pd.DataFrame(['200{}'.format(year)] * df_msa2004.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif year ==6:
        list_df_msa.append(pd.concat([df_msa2005, pd.DataFrame(['200{}'.format(year)] * df_msa2005.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif year == 7:
        list_df_msa.append(pd.concat([df_msa2006, pd.DataFrame(['200{}'.format(year)] * df_msa2006.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif year == 8:
        list_df_msa.append(pd.concat([df_msa2007, pd.DataFrame(['200{}'.format(year)] * df_msa2007.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif year == 9:
        list_df_msa.append(pd.concat([df_msa2008, pd.DataFrame(['200{}'.format(year)] * df_msa2008.shape[0],\
                                                              columns = ['date'])], axis = 1, join = 'inner'))
    elif np.isin(year,[10,11,12]):
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

df_msa_panel = pd.concat(list_df_msa, ignore_index=True)

#------------------------------------------------------------
# Merge the data sets and save to csv
#------------------------------------------------------------
# Merge the dfs
df = df_centers_panel.merge(df_msa_panel, how = 'left', on = ['fips','date'])

## Drop double colums
df.drop(columns = ['comp_name','state'], inplace = True)

# Add variables
## ln pop
df['ln_pop'] = np.log(df.population)

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

