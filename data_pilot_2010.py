# Pilot HMDA, SDI, SOD for 2010

#------------------------------------------------------------
# Prelims
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

from Code_docs.determine_ethnicity import determineEthnicity

#------------------------------------------------------------
# Load data
#------------------------------------------------------------

#------------------------------------------------------------
# Statistics on Dep. Institutions

# Prelims
path_sdi = r'X:/My Documents/Data/Data_SDI/2010/'
file_sdi = r'All_Reports_20101231_Assets and Liabilities.csv'
vars_sdi = ['cert','fed_rssd','rssdhcr','name','city','stalp','address','zip','numemp','asset']

# Load df
df_sdi = pd.read_csv(path_sdi + file_sdi, usecols = vars_sdi)

# Prelim data inspection
print(df_sdi.info())
print(df_sdi.describe())

''' NOTE: numemp has 9 missings. For now that does not look like a problem. Other
    variables are complete.
'''

#------------------------------------------------------------
# HMDA Lender File

# Prelims
path_lf = r'X:/My Documents/Data/Data_HMDA_lenderfile/'
file_lf = r'hmdpanel17.dta'
vars_lf = ['hmprid','CERT10','ENTITY10','RSSD10']

# Load df
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)

# Prelim data inspection
print(df_lf.info())
print(df_lf.describe())

''' NOTE: There are less institutions in the HMDA LF than in the SDI. No surprise.
    After preselection, no missings.
'''

# Preselect CERT10 in cert (from df_sdi)
df_lf = df_lf[df_lf.CERT10.isin(df_sdi.cert)]

#------------------------------------------------------------
# HMDA LAR

# Prelims
path_hmda = r'X:/My Documents/Data/Data_HMDA/LAR/'
file_hmda = r'hmda_2010_nationwide_originated-records_codes.zip'
dtypes_col_hmda = {'state_code':'str', 'county_code':'str','msamd':'str',\
                   'census_tract_number':'str'} # Force the correct column type

## Load the data 
df_chunk = pd.read_csv(path_hmda + file_hmda, index_col = 0, chunksize = 1e6, \
                 na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA'], dtype = dtypes_col_hmda)

### Prelims     
chunk_list = []  # append each chunk df here 

### Loop over the chunks
for chunk in df_chunk:
    # Filter data
    ## Remove credit unions and mortgage institutions (CHECK WITH AVERY ET AL 2007 LATER), and remove NA in msamd
    chunk_filtered = chunk[chunk.respondent_id.isin(df_lf.hmprid)].\
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

# Delete some variables to keep RAM usage to a minimum
del chunk_filtered, chunk_list, chunk

# Prelim data inspection
print(df_hmda.info())
print(df_hmda.isna().sum())

''' NOTE: missings in the following variabels.
    applicant_income_000s              218167
    population                            602
    minority_population                   651
    hud_median_family_income              542
    tract_to_msamd_income                1441
    number_of_owner_occupied_units       2308
    number_of_1_to_4_family_units        1575
    ethnicity_borrower                 319243
'''
#------------------------------------------------------------
# Summary of Deposits 

# Prelim
path_sod = r'X:/My Documents/Data/Data_SOD/'
file_sod = r'ALL_2010.csv'
vars_sod = ['CERT','MSABR','RSSDID','STCNTYBR','SIMS_LATITUDE','SIMS_LONGITUDE',\
            'ADDRESBR','STALPBR','ZIPBR','BRSERTYP','CITYBR','UNINUMBR']
dtypes_col_sod = {'MSABR':'str','STCNTYBR':'str'}
    
# Load df
df_sod = pd.read_csv(path_sod + file_sod, usecols = vars_sod, encoding = "ISO-8859-1",\
                     dtype = dtypes_col_sod) 
    
# Prelim data inspection
print(df_sod.info())

''' NOTE: Quite some missings in lat and long. Do nothing for now. 
'''

# STCNTYBR (probles with the fips)
''' OLD NOTE: The following states have not-recognizable entries for STCNTYBR:
    California, Arizona, Colorado, Arkansas, Alabama, Alaska, Coneticut.
    
    All these state fips should start with a 0
'''
## Make a list of state codes 
state_codes_gaps = ['AZ','CA','CO','AK','AL','AR','CT']

## Check values in STCNTYBR
vals_fips = df_sod[df_sod.STALPBR.isin(state_codes_gaps)][['CERT','STALPBR','STCNTYBR']]
    
## use zfill to force the correct fips
df_sod['STCNTYBR'] = df_sod.STCNTYBR.str.zfill(5)

#------------------------------------------------------------
# Make a geography plot of SOD
#------------------------------------------------------------
'''
# Prelims
fips_count = pd.DataFrame(df_sod.groupby('STCNTYBR').STCNTYBR.count())
fips_count = fips_count.rename(columns = {'STCNTYBR':'count'}).reset_index()
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
fig = px.choropleth(fips_count, geojson=counties, locations='STCNTYBR', color='ln_count',
                            color_continuous_scale="inferno",
                            range_color=(0, fips_count.iloc[:,2].max()),
                            scope="usa")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width = 1500, height = 900,
                    coloraxis_colorbar=dict(
                    title="Num. Branches",
                    tickvals=[0,np.log(np.floor(fips_count.iloc[:,1].max() / 1e3) * 1e3)],
                    ticktext=['0','{}k'.format(np.floor(fips_count.iloc[:,1].max() / 1e3).astype(str)[:-2])]),
                    title=go.layout.Title(
                    text='Number Bank and Thrift Branches per County (2010)',
                    xref='paper',
                    x=0.1,
                    y=0.95)
                  )
pio.write_image(fig, 'Figures/Number_branches_map_2010.png')
'''

#------------------------------------------------------------
# Match the data
#------------------------------------------------------------

#------------------------------------------------------------
# First SDI and SOD

# Merge the two dataframes
df_fdic = df_sod.merge(df_sdi, how = 'outer', left_on = 'CERT', right_on = 'cert',\
                       indicator = True)

# Add date variable
df_fdic['date'] = 2010

# Check the merge
print(df_fdic.info())
print(df_fdic._merge.value_counts())

# Check left_only in _merge
left_only_fdic = df_fdic[df_fdic._merge == 'left_only']
ids_inleft_fdic = df_fdic[df_fdic.CERT.isin(left_only_fdic.CERT.unique().tolist())] # Is the same as left_only_fdic
rssdid_inleft_fdic = df_fdic[df_fdic.RSSDID.isin(left_only_fdic.RSSDID.unique().tolist())] # Is the same as ids_inleft_fdic

# Check right_only in _merge
right_only_fdic = df_fdic[df_fdic._merge == 'right_only']
ids_inright_fdic = df_fdic[df_fdic.CERT.isin(right_only_fdic.CERT.unique().tolist())] # Is the same as right_only_fdic
rssdid_inright_fdic = df_fdic[df_fdic.RSSDID.isin(right_only_fdic.RSSDID.unique().tolist())] # Is the same as ids_inright_fdic

#------------------------------------------------------------
# Second SDI, LF and LAR

# Merge SDI and LF
df_sdilf = df_sdi.merge(df_lf, how = 'outer', left_on = 'cert', right_on = 'CERT10',\
                       indicator = True)
    
# Add date variable
df_sdilf['date'] = 2010

# Check the merge
print(df_sdilf.info())
print(df_sdilf._merge.value_counts()) 

# Drop left_only
df_sdilf = df_sdilf[df_sdilf._merge == 'both'].drop(columns = '_merge')

# Merge sdilf and HMDA
df_main = df_sdilf.merge(df_hmda, how = 'outer', left_on = 'hmprid', right_on = 'respondent_id',\
                       indicator = True)
    
# Check the merge
print(df_main.info())
print(df_main._merge.value_counts())  # 480 are left only, dorp them

df_main = df_main[df_main._merge == 'both'].drop(columns = '_merge')
