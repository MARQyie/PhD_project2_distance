# Setup the HMDA dataframe
'''
    This script sets up the total dataframe for HMDA. This script also reads
    the HMDA lender file (avery file)
    
    The script does the following
    1. Reads in the HMDA dataframes for each year-end from 2010 to 2017
    2. Select the relevant variables
    3. Clean up the dataframe
    
    Concatenating all HMDA files might lead to an overload of memory usage. 
    Strategy: load all HMDA files separately, clean them and save them as 
    zipped csv (saves space).
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
#os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

from Code_docs.determine_ethnicity import determineEthnicity

# For parallelization
import multiprocessing as mp 
from joblib import Parallel, delayed 

#------------------------------------------------------------
# Set Parameters
#------------------------------------------------------------

start = 2010
end = 2019

num_cores = mp.cpu_count()

#------------------------------------------------------------
# Load the data, clean and save
#------------------------------------------------------------

# Prelims
## Lender File
path_lf = r'D:/RUG/Data/Data_HMDA_lenderfile/'
file_lf = r'hmdpanel17.dta'
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end -1)] \
          + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end -1)] \
          + ['RSSD{}'.format(str(year)[2:4]) for year in range(start, end -1)]

## HMDA LAR
path_hmda = r'D:/RUG/Data/Data_HMDA/LAR/'
file_hmda_04 = 'f{}lar.public.dat.zip'
file_hmda_0506 = 'LARS.FINAL.{}.DAT.zip'
file_hmda_0717 = r'hmda_{}_nationwide_all-records_codes.zip'
file_hmda_1819 = r'year_{}.csv'
dtypes_col_hmda = {'respondent_id':'object','state_code':'str', 'county_code':'str','msamd':'str',\
                   'census_tract_number':'str', 'derived_msa-md':'str'}
col_width_0406 = [4, 10, 1, 1, 1, 1, 5, 1, 5, 2, 3, 7, 1, 1, 4, 1, 1, 1, 1, 1, 1,\
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 7]
col_names_0406 = ['date','respondent_id', 'agency_code', 'loan_type',\
             'loan_purpose', 'owner_occupancy', 'loan_amount_000s',\
             'action_taken', 'msamd', 'state_code', 'county_code',\
             'census_tract_number', 'applicant_sex', 'co_applicant_sex',\
             'applicant_income_000s', 'purchaser_type', 'denial_reason_1',\
             'denial_reason_2', 'denial_reason_3', 'edit_status', 'property_type',\
             'preapproval', 'applicant_ethnicity', 'co_applicant_ethnicity',\
             'applicant_race_1', 'applicant_race_2', 'applicant_race_3',\
             'applicant_race_4', 'applicant_race_5', 'co_applicant_race_1',\
             'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4',\
             'co_applicant_race_5', 'rate_spread', 'hoepa_status', 'lien_status',\
             'sequence_number']
na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA', 'Exempt', 'N/AN/', 'na']

## HMDA panel
path_hmda_panel = r'D:/RUG/Data/Data_HMDA/Panel/'
file_hmda_panel = r'{}_public_panel_csv.csv'

# Load data
## Lender file
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)

# HMDA
def cleanHMDA(year):
    #Load the dataframe in a temporary frame
    if year == 2004:
        df_chunk = pd.read_fwf(path_hmda + file_hmda_04.format(year), widths = col_width_0406, names = col_names_0406, chunksize = 1e6, na_values = na_values, dtype = dtypes_col_hmda, header=None)
    elif year < 2007:
        df_chunk = pd.read_fwf(path_hmda + file_hmda_0506.format(year), widths = col_width_0406, names = col_names_0406, chunksize = 1e6, na_values = na_values, dtype = dtypes_col_hmda, header=None)
    elif year < 2018:
        df_chunk = pd.read_csv(path_hmda + file_hmda_0717.format(year), index_col = 0, chunksize = 1e6, na_values = na_values, dtype = dtypes_col_hmda)
    else: # From 2018 onward structure of the data changes
        df_chunk = pd.read_csv(path_hmda + file_hmda_1819.format(year), index_col = 0, chunksize = 1e6, na_values = na_values, dtype = dtypes_col_hmda)
        df_panel = pd.read_csv(path_hmda_panel + file_hmda_panel.format(year))
        
    chunk_list = []  # append each chunk df here 

    ### Loop over the chunks
    for chunk in df_chunk:
        # Merge with df_panel and change column names if year is 2018 or 2019
        if year >= 2018:
            ## Merge
            chunk = chunk.merge(df_panel.loc[:,['lei','agency_code','id_2017','arid_2017','tax_id']], how = 'left', on = 'lei')
            
            ## Change column names
            dict_columns = {'derived_msa-md':'msamd',
                'tract_population':'population',
                'tract_minority_population_percent':'minority_population',
                'ffiec_msa_md_median_family_income':'hud_median_family_income',
                'tract_to_msa_income_percentage':'tract_to_msamd_income',
                'tract_one_to_four_family_homes':'number_of_1_to_4_family_units',
                'income':'applicant_income_000s',
                'applicant_race-1':'applicant_race_1',
                'applicant_race-2':'applicant_race_2',
                'applicant_race-3':'applicant_race_3',
                'applicant_race-4':'applicant_race_4',
                'applicant_race-5':'applicant_race_5',
                'co-applicant_race-1':'co_applicant_race_1',
                'co-applicant_race-2':'co_applicant_race_2',
                'co-applicant_race-3':'co_applicant_race_3',
                'co-applicant_race-4':'co_applicant_race_4',
                'co-applicant_race-5':'co_applicant_race_5',
                'applicant_ethnicity_observed':'applicant_ethnicity',
                'co-applicant_ethnicity_observed':'co_applicant_ethnicity',
                'co-applicant_sex':'co_applicant_sex'}
            chunk.rename(columns = dict_columns, inplace = True)     
            
            ## Add column of loan amount in thousand USD
            chunk['loan_amount_000s'] = chunk.loan_amount / 1e3
            
            ## Format respondent_id column  (form arid_2017: remove agency code (first string), and zero fill. Replace -1 with non)
            ## NOTE Fill the missings with the tax codes. After visual check, some arid_2017 are missing. However, the tax_id corresponds to the resp_id pre 2018
            chunk['respondent_id'] = chunk.apply(lambda x: x.tax_id if x.arid_2017 == '-1' else str(x.arid_2017)[1:], axis = 1).replace('-1',np.nan).str.zfill(10)
            
        # Filter data
        ## Remove credit unions and mortgage institutions, and remove NA in msamd
        if year < 2018:
            chunk_filtered = chunk[chunk.respondent_id.isin(df_lf[df_lf['CERT{}'.format(str(year)[2:4])] >= 0.0].hmprid)].\
                         dropna(axis = 0, how = 'any', subset = ['msamd','loan_amount_000s','applicant_income_000s'])
        else:
            chunk_filtered = chunk[chunk.respondent_id.isin(df_lf[df_lf['CERT{}'.format(17)] >= 0.0].hmprid)].\
                         dropna(axis = 0, how = 'any', subset = ['msamd','loan_amount_000s','applicant_income_000s'])
                         
        ## Drop all purchased loans, denied preapproval requests and non-home purchase loans
        chunk_filtered = chunk_filtered[(chunk_filtered.action_taken < 6) & (chunk_filtered.loan_purpose == 1)]
        
        ## Make a fips column and remove separate state and county
        if year < 2018:
            chunk_filtered['fips'] = chunk_filtered['state_code'].str.zfill(2) + chunk_filtered['county_code'].str.zfill(3)
        else:
            chunk_filtered['fips'] =  chunk_filtered['county_code'].str.zfill(5)
        chunk_filtered.drop(columns = ['state_code', 'county_code'], inplace = True)
        
        ## Make one ethnicity column
        ethnicity = determineEthnicity(chunk_filtered)
        dummies_ethnicity = pd.get_dummies(ethnicity, prefix = 'ethnicity')
        chunk_filtered[dummies_ethnicity.columns] = dummies_ethnicity
        
        ## Add a date var
        chunk_filtered['date'] = year
    
        ## Add variables
        ### Loan to income
        chunk_filtered['lti'] = np.log(1 + (chunk_filtered.loan_amount_000s / chunk_filtered.applicant_income_000s))
        
        ### ln loan amount
        chunk_filtered['ln_loanamout'] = np.log(1 + chunk_filtered.loan_amount_000s)
        
        ### ln income
        chunk_filtered['ln_appincome'] = np.log(1 + chunk_filtered.applicant_income_000s)
        
        ### Dummy subprime
        if year < 2018:
            chunk_filtered['subprime'] = (chunk_filtered.rate_spread > 0.0) * 1
        else:
            # NOTE: we can use swifter to speed up the process. Supported by peregrine?
            lambda_subprime = lambda data: (data.rate_spread - 0.03) > 0.0 if data.lien_status == 1 else (data.rate_spread - 0.05) > 0.0 if data.lien_status == 2 else np.nan
            chunk_filtered['subprime'] = (chunk_filtered.apply(lambda_subprime)) * 1
        
        
        ### Dummy first lien
        chunk_filtered['lien'] = (chunk_filtered.lien_status == 1) * 1
        
        ### Dummy HOEPA loan
        chunk_filtered['hoepa'] = (chunk_filtered.hoepa_status == 1)
        
        ### Loan type dummies
        dummies_loan = pd.get_dummies(chunk_filtered.loan_type, prefix = 'loan_type')
        chunk_filtered[dummies_loan.columns] = dummies_loan
        
        ### Owner occupancy dummy
        if year < 2018:
            chunk_filtered['owner'] = (chunk_filtered.owner_occupancy == 1) * 1
        else:
            chunk_filtered['owner'] = (chunk_filtered.occupancy_type == 1) * 1
        
        ### Dummy Pre-approval
        chunk_filtered['preapp'] = (chunk_filtered.preapproval == 1) * 1
        
        ### Dummies Sex (possibilities: male, female, non-disclosed)
        dummies_sex = pd.get_dummies(chunk_filtered.applicant_sex, prefix = 'sex')
        chunk_filtered[dummies_sex.columns] = dummies_sex
        
        ### Dummy co-applicant
        chunk_filtered['coapp'] = (chunk_filtered.co_applicant_sex & chunk_filtered.co_applicant_ethnicity == 5) * 1
                
        ### Loan sale dummies
        chunk_filtered['ls'] = (chunk_filtered.purchaser_type.isin(list(range(1,9+1)) + [71, 72])) * 1
        chunk_filtered['ls_gse'] = (chunk_filtered.purchaser_type.isin(range(1,4+1))) * 1
        chunk_filtered['ls_priv'] = (chunk_filtered.purchaser_type.isin(list(range(1,9+1)) + [71, 72])) * 1
        chunk_filtered['sec'] = (chunk_filtered.purchaser_type == 5) * 1
           
        ## Drop unneeded columns
        if year < 2018:
            columns = ['respondent_id', 'agency_code', 'loan_type', 'loan_purpose', 'msamd',\
                   'applicant_sex', 'co_applicant_sex', 'population', 'minority_population',\
                   'hud_median_family_income', 'tract_to_msamd_income', 'number_of_1_to_4_family_units',\
                   'fips', 'ethnicity_borrower', 'date', 'lti', 'ln_loanamout', 'ln_appincome',\
                   'subprime', 'lien', 'hoepa', 'owner', 'preapp', 'coapp','ls',\
                   'ls_gse', 'ls_priv', 'sec'] + dummies_ethnicity.columns.tolist() +\
                    dummies_loan.columns.tolist() + dummies_sex.columns.tolist()
        else:
            # Todo: add extra 2018-2019 variables
            columns = ['respondent_id', 'agency_code', 'loan_type', 'loan_purpose', 'msamd',\
                   'applicant_sex', 'co_applicant_sex', 'population', 'minority_population',\
                   'hud_median_family_income', 'tract_to_msamd_income', 'number_of_1_to_4_family_units',\
                   'fips', 'ethnicity_borrower', 'date', 'lti', 'ln_loanamout', 'ln_appincome',\
                   'subprime', 'lien', 'hoepa', 'owner', 'preapp', 'coapp','ls',\
                   'ls_gse', 'ls_priv', 'sec'] + dummies_ethnicity.columns.tolist() +\
                    dummies_loan.columns.tolist() + dummies_sex.columns.tolist()
        
        chunk_filtered = chunk_filtered[columns]
        
        ## Drop na in fips
        chunk_filtered.dropna(subset = ['fips'], inplace = True)
        
        # Add the chunk to the list
        chunk_list.append(chunk_filtered)
        
    # concat the list into dataframe 
    df_hmda = pd.concat(chunk_list)
    
    # Save as zipped dataframe
    df_hmda.to_csv('Data/data_hmda_{}.csv.gz'.format(year), index = False, compression = 'gzip')
    df_hmda.to_csv('Data/data_hmda_{}.csv'.format(year), index = False)
    
#
if __name__ == '__main__':
    Parallel(n_jobs=num_cores)(delayed(cleanHMDA)(year) for year in range(start,end + 1))
    