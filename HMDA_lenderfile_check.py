och
#HMDA lender file check
import pandas as pd
import numpy as np

import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load data
## Prelims
path = r'X:/My Documents/Data/Data_HMDA_lenderfile/'

## Load the stata-file
df_lf = pd.read_stata(path + 'hmdpanel17.dta')

#------------------------------------------------------------
# Test with 2010 data
#------------------------------------------------------------
# Select the variables from df_lf
vars_lf_10 = ['hmprid','CERT10','ENTITY10','RSSD10']

# Subset the data
df_lf_10 = df_lf[vars_lf_10]

# Select only FDIC insured institutions and get the unique hmprid
unique_fdicinsured = df_lf_10[df_lf_10.CERT10 > 0.0].hmprid.unique().tolist()

#------------------------------------------------------------
# Match with HMDA data
# Load data
## Prelims
path_hmda = r'X:/My Documents/Data/Data_HMDA/LAR/'

## Define column dtypes for specific columns
dtypes_col = {'state_code':'str', 'county_code':'str'}

## Load the data 
df_chunk = pd.read_csv(path_hmda + 'hmda_2010_nationwide_originated-records_codes.zip', index_col = 0, chunksize = 1e6, \
                 na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    '], dtype = dtypes_col)
   
## Preprocess the data   
### Define method to make one column for ethnicity
def determineEthnicity(dataframe):
    '''This method defines the ethnicity of the borrower (see Avery et al 2007)
       The hierachy is as follows (if either app or co-app has declared):
           Black, Hispanic, American Indian, Hawaiian/Pacific
           Islander, Asian, white Non-Hispanic'''
    
    #Prelims
    list_race = ['applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',
                 'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4', 'co_applicant_race_5']
    list_eth = ['applicant_ethnicity', 'co_applicant_ethnicity']
    vector = pd.DataFrame(index = dataframe.index, columns = ['ethnicity_borrower'])
    
    # Setup the boolean vectors
    black = dataframe.loc[:,list_race].isin([3]).any(axis = 1)
    hispanic = dataframe.loc[:,list_eth].isin([1]).any(axis = 1)
    amer_ind = dataframe.loc[:,list_race].isin([1]).any(axis = 1)
    hawaiian = dataframe.loc[:,list_race].isin([4]).any(axis = 1)
    asian = dataframe.loc[:,list_race].isin([2]).any(axis = 1)
    white_nh = dataframe.loc[:,list_race].isin([5]).any(axis = 1) & dataframe.loc[:,list_eth].isin([2]).any(axis = 1)
    
    # Fill the vector
    vector[black] = 1
    vector[~black & hispanic] = 2
    vector[~black & ~hispanic & amer_ind] = 3
    vector[~black & ~hispanic & ~amer_ind & hawaiian] = 4
    vector[~black & ~hispanic & ~amer_ind & ~hawaiian & asian] = 5
    vector[~black & ~hispanic & ~amer_ind & ~hawaiian & ~asian & white_nh] = 0
    vector[~black & ~hispanic & ~amer_ind & ~hawaiian & ~asian & ~white_nh] = np.nan
    
    return(np.array(vector))     

### Prelims     
chunk_list = []  # append each chunk df here 

### Loop over the chunks
for chunk in df_chunk:
    # Filter data
    ## Remove credit unions and mortgage institutions (CHECK WITH AVERY ET AL 2007 LATER), and remove NA in msamd
    chunk_filtered = chunk[chunk.respondent_id.isin(unique_fdicinsured)].\
                     dropna(axis = 0, how = 'any', subset = ['msamd'])
    
    ## Make a fips column and remove separate state and county
    chunk_filtered['fips'] = chunk_filtered['state_code'] + chunk_filtered['county_code']
    chunk_filtered.drop(columns = ['state_code', 'county_code'], inplace = True)
    
    ## Make one ethnicity column and drop the originals 
    list_eth_race = ['applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',\
                     'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4',\
                     'co_applicant_race_5', 'applicant_ethnicity', 'co_applicant_ethnicity']
    chunk_filtered['ethnicity_borrower'] = determineEthnicity(chunk_filtered)
    chunk_filtered.drop(columns = list_eth_race, inplace = True)
    
    ## Drop denail columns
    chunk_filtered.drop(columns = ['denial_reason_1', 'denial_reason_2', 'denial_reason_3'], inplace = True)
    
    # Add the chunk to the list
    chunk_list.append(chunk_filtered)
    
# concat the list into dataframe 
df_hmda = pd.concat(chunk_list)

#------------------------------------------------------------
# Make a geography plot
#------------------------------------------------------------
# Prelims
fips_count = pd.DataFrame(df_hmda.groupby('fips').fips.count())
fips_count = fips_count.rename(columns = {'fips':'count'}).reset_index()
fips_count['ln_count'] = np.log(fips_count['count'])

## figure prelims
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Make the image    
fig = px.choropleth(fips_count, geojson=counties, locations='fips', color='ln_count',
                           color_continuous_scale="inferno",
                           range_color=(0, fips_count.iloc[:,2].max()),
                           scope="usa")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width = 1500, height = 900,
                   coloraxis_colorbar=dict(
                   title="Num. Originations",
                   tickvals=[0,np.log(np.floor(fips_count.iloc[:,1].max() / 1e3) * 1e3)],
                   ticktext=['0','{}k'.format(np.floor(fips_count.iloc[:,1].max() / 1e3).astype(str)[:-2])]),
                   title=go.layout.Title(
                   text='Number of Mortgage Originations per County (2010)',
                   xref='paper',
                   x=0.1,
                   y=0.95)
                  )
pio.write_image(fig, 'Figures/Number_originations_map_2010_alt.png') 
