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
vars_sdi = ['cert','fed_rssd','rssdhcr','name','city','stalp','address','zip','numemp','asset','cb']

# Load df
df_sdi = pd.read_csv(path_sdi + file_sdi, usecols = vars_sdi)

# Prelim data inspection
print(df_sdi.info())
print(df_sdi.describe())

''' NOTE: numemp has 9 missings. For now that does not look like a problem. Other
    variables are complete.
'''

# Add variables to the df
''' List of new variables
        ln total assets
        ln 1 + number of employees
'''
df_sdi['ln_ta'] = np.log(df_sdi.asset)
df_sdi['ln_emp'] = np.log(1 + df_sdi.numemp)

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
df_chunk = pd.read_csv(path_hmda + file_hmda, index_col = 0, chunksize = 1e6, nrows = 1e6,\
                 na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA'], dtype = dtypes_col_hmda)    
# df_chunk = pd.read_csv(path_hmda + file_hmda, index_col = 0, chunksize = 1e6, \
#                  na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA'], dtype = dtypes_col_hmda)

### Prelims     
chunk_list = []  # append each chunk df here 

### Loop over the chunks
for chunk in df_chunk:
    # Filter data
    ## Remove credit unions and mortgage institutions (CHECK WITH AVERY ET AL 2007 LATER), and remove NA in msamd
    chunk_filtered = chunk[chunk.respondent_id.isin(df_lf.hmprid)].\
                     dropna(axis = 0, how = 'any', subset = ['msamd','loan_amount_000s','applicant_income_000s'])
    
    ## Make a fips column and remove separate state and county
    chunk_filtered['fips'] = chunk_filtered['state_code'] + chunk_filtered['county_code']
    chunk_filtered.drop(columns = ['state_code', 'county_code'], inplace = True)
    
    ## Make one ethnicity column and drop the originals 
    list_eth_race = ['applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',\
                     'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4',\
                     'co_applicant_race_5', 'applicant_ethnicity', 'co_applicant_ethnicity']
    chunk_filtered['ethnicity_borrower'] = determineEthnicity(chunk_filtered)
    chunk_filtered.drop(columns = list_eth_race, inplace = True)
       
    ## Add variables
    ''' List of new vars
            Loan to income
            ln 1 + loan amount
            ln 1 + income
            Dummy Subprime
            Dummy Secured
            Total loan sales, loan sales to GSE, loan sales to private parties, private securitization
    '''
    ### Loan to income
    chunk_filtered['lti'] = np.log(1 + (chunk_filtered.loan_amount_000s / chunk_filtered.applicant_income_000s))
    
    ### ln loan amount
    chunk_filtered['ln_loanamout'] = np.log(1 + chunk_filtered.loan_amount_000s)
    
    ### ln income
    chunk_filtered['ln_appincome'] = np.log(1 + chunk_filtered.applicant_income_000s)
    
    ### Dummy subprime
    chunk_filtered['subprime'] = (chunk_filtered.rate_spread > 0.0) * 1
    
    ### Dummy secured
    chunk_filtered['secured'] = (chunk_filtered.lien_status.isin([1,2])) * 1
    
    ### Loan sale dummies
    chunk_filtered['ls'] = (chunk_filtered.purchaser_type.isin(range(1,9+1))) * 1
    chunk_filtered['ls_gse'] = (chunk_filtered.purchaser_type.isin(range(1,4+1))) * 1
    chunk_filtered['ls_priv'] = (chunk_filtered.purchaser_type.isin(range(6,9+1))) * 1
    chunk_filtered['sec'] = (chunk_filtered.purchaser_type == 5) * 1
       
    ## Drop unneeded columns
    chunk_filtered.drop(columns = ['denial_reason_1', 'denial_reason_2', 'denial_reason_3',\
                                   'hoepa_status', 'lien_status', 'edit_status', 'action_taken',\
                                   'preapproval'], inplace = True)
    
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
            'ADDRESBR','STALPBR','ZIPBR','BRSERTYP','CITYBR','UNINUMBR','DEPSUMBR']
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

## Fix DEPSUMBR (reads in as object, but should be an int)
df_sod.DEPSUMBR = df_sod.DEPSUMBR.str.replace(',','').astype(int)

#------------------------------------------------------------
# MSA file
df_msa = pd.read_csv('Data/data_msa_popcenter.csv', index_col = 0, dtype = {'fips':'str','cbsa_code':'str'}) 

# Select date == 2010
df_msa = df_msa[df_msa.date == 2010]

# Add variables to the df
''' Variables added + data source
        Density of bank branches (ln)       SOD
        Deposit-based HHI                   SOD
        ln mfi                              HMDA
        dum_min                             HMDA
'''

## SOD vars
density = df_sod.groupby('STCNTYBR').STCNTYBR.count().rename('density')
df_msa = df_msa.merge(density, how = 'left', left_on = 'fips', right_on = density.index)
df_msa.density = df_msa.density.fillna(value = 0)

hhi = df_sod.groupby('STCNTYBR').DEPSUMBR.apply(lambda x: ((x / x.sum())**2).sum()).rename('hhi')
df_msa = df_msa.merge(hhi, how = 'left', left_on = 'fips', right_on = hhi.index)
df_msa.hhi = df_msa.hhi.fillna(value = 0)

## HMDA vars
ln_mfi = df_hmda.groupby('fips').hud_median_family_income.apply(lambda x: np.log(x.mean())).rename('ln_mfi')
df_msa = df_msa.merge(ln_mfi, how = 'left', left_on = 'fips', right_on = ln_mfi.index)
df_msa.ln_mfi = df_msa.ln_mfi.fillna(value = 0)

dum_min = df_hmda.groupby('fips').minority_population.apply(lambda x: (x.mean() > 0.5) * 1).rename('dum_min')
df_msa = df_msa.merge(dum_min, how = 'left', left_on = 'fips', right_on = dum_min.index)
df_msa.dum_min = df_msa.dum_min.fillna(value = 0)

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
# SDI, LF and LAR

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
print(df_main._merge.value_counts())  # 480 are left only, drop them

df_main = df_main[df_main._merge == 'both'].drop(columns = '_merge')

# Only keeps fips that are in df_msa
df_main = df_main[df_main.fips.isin(df_msa.fips.unique())]

#------------------------------------------------------------
# Calculate the distance variable
#------------------------------------------------------------
from numpy import sin, cos, sqrt, arctan2, radians
from numba import njit, jit

@jit(nopython = True, fastmath = True) #speed things up a little
def haversine(lat1, lon1, lat2, lon2):
    
    R = 6372.8  # Earth radius in km
    
    phi1, phi2, dphi, dlambda = map(radians, [lat1, lat2, lat2 - lat1, lon2 - lon1])
    
    a = sin(dphi/2)**2 + \
        cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    
    return 2*R*arctan2(sqrt(a), sqrt(1 - a))

# Vectorize the haversine
vecHaversine = np.vectorize(haversine)

def minDistanceLenderBorrower(hmda_cert,hmda_fips,msa_fips,msa_lat,msa_long,sod_stcntybr,sod_cert):
    ''' This methods calculates the minimum distance between a lender (a specific
        branche of a bank or thrift) and a borrower (a unknown individual) based on
        the respective fips codes. Uses the haversine method to calculate distances.
    
    Parameters
    ----------
    hmda : pandas DataFrame 
        one row and n columns
    msa : pandas DataFrame
    sod : pandas DataFrame

    Returns
    -------
    Float

    '''

    # Make a subset of branches where the FDIC certificates in both datasets match  
    branches = sod_stcntybr[sod_cert == hmda_cert]
    
    if (not hmda_cert in sod_cert):
        return (np.nan)
    elif hmda_fips in branches:
        return (0.0)
    else:
        distances = vecHaversine(msa_lat[msa_fips == hmda_fips], msa_long[msa_fips == hmda_fips],\
                                 msa_lat[np.isin(msa_fips,branches)], msa_long[np.isin(msa_fips,branches)])
        
        return(np.min(distances))

vecMinDistanceLenderBorrower = np.vectorize(minDistanceLenderBorrower, excluded = (2,3,4,5,6))

df_main['distance'] = vecMinDistanceLenderBorrower(*df_main[['cert','fips']].T.to_numpy(),\
                       df_msa.fips.to_numpy(),df_msa.latitude.to_numpy(),df_msa.longitude.to_numpy(),\
                       df_sod.STCNTYBR.to_numpy(), df_sod.CERT.to_numpy())
  

#------------------------------------------------------------
# Aggregate data to MSA-level
#------------------------------------------------------------

# Begin with the portfolio specific variables    
df_agg = df_main.groupby(['cert','msamd'])[['lti','ln_loanamout','ln_appincome','distance']].mean() #Some missings in distance, check later
df_agg['subprime'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.subprime) / len(x.subprime))
df_agg['secured'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.secured) / len(x.secured))

# Add the loan sale variables
## Fraction based on number of loans
df_agg['ls_num'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.ls) / len(x.ls))
df_agg['ls_gse_num'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.ls_gse) / len(x.ls_gse))
df_agg['ls_priv_num'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.ls_priv) / len(x.ls_priv))
df_agg['sec_num'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.sec) / len(x.sec))

## Fraction based on value of loans
df_agg['ls_val'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.ls * x.loan_amount_000s) \
                                                           / np.sum(x.loan_amount_000s))
df_agg['ls_gse_val'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.ls_gse * x.loan_amount_000s) \
                                                               / np.sum(x.loan_amount_000s))
df_agg['ls_priv_val'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.ls_priv * x.loan_amount_000s) \
                                                                / np.sum(x.loan_amount_000s))
df_agg['sec_val'] = df_main.groupby(['cert','msamd']).apply(lambda x: np.sum(x.sec * x.loan_amount_000s) \
                                                            / np.sum(x.loan_amount_000s))   
## Dummy
df_agg['ls_dum' ] = (df_main.groupby(['cert','msamd']).ls.sum() > 0.0) * 1
df_agg['ls_gse_dum' ] = (df_main.groupby(['cert','msamd']).ls_gse.sum() > 0.0) * 1
df_agg['ls_priv_dum' ] = (df_main.groupby(['cert','msamd']).ls_priv.sum() > 0.0) * 1
df_agg['sec_dum' ] = (df_main.groupby(['cert','msamd']).sec.sum() > 0.0) * 1

# Bank/Thrift level controls
## First reset index before merge
df_agg.reset_index(inplace = True)
 
## add ln_ta, ln_emp and bank indicator
df_agg = df_agg.merge(df_sdi[['cert','cb','ln_ta','ln_emp']], how = 'left', on = 'cert')

## Add number of branches
num_branches = df_sod.groupby('CERT').CERT.count().rename('num_branch')
df_agg = df_agg.merge(num_branches, how = 'left', left_on = 'cert', right_on = num_branches.index)

# MSA level
# from df_msa
df_msa_agg = df_msa.groupby('cbsa_code')[['ln_pop', 'density', 'hhi', 'ln_mfi', 'dum_min']].mean()
df_agg = df_agg.merge(df_msa_agg, how = 'left', left_on = 'msamd', right_on = df_msa_agg.index)

# Add average lending distance per MSA
distance_msa = df_main.groupby('msamd').distance.apply(lambda x: np.log(1 + x.mean())).rename('ave_distance')
df_agg = df_agg.merge(distance_msa, how = 'left', left_on = 'msamd', right_on = distance_msa.index)

#------------------------------------------------------------
# Save df
#------------------------------------------------------------

df_agg.to_csv('Data/df_agg_2010_test.csv')
