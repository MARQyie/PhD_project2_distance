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

#------------------------------------------------------------
# Set Parameters
#------------------------------------------------------------

start = 2010
end = 2017

#------------------------------------------------------------
# Load the data, clean and save
#------------------------------------------------------------

# Prelims
## Lender File
path_lf = r'D:/RUG/Data/Data_HMDA_lenderfile/'
file_lf = r'hmdpanel17.dta'
vars_lf = ['hmprid'] + ['CERT{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['ENTITY{}'.format(str(year)[2:4]) for year in range(start, end +1)] \
          + ['RSSD{}'.format(str(year)[2:4]) for year in range(start, end +1)]

## HMDA
path_hmda = r'D:/RUG/Data/Data_HMDA/LAR/'
file_hmda = r'hmda_{}_nationwide_originated-records_codes.zip'
dtypes_col_hmda = {'state_code':'str', 'county_code':'str','msamd':'str',\
                   'census_tract_number':'str'}
na_values = ['NA   ', 'NA ', '...',' ','NA  ','NA      ','NA     ','NA    ','NA']

# Load data
## Lender file
df_lf = pd.read_stata(path_lf + file_lf, columns = vars_lf)

# HMDA
for year in range(start,end + 1):
    #Load the dataframe in a temporary frame
    df_chunk = pd.read_csv(path_hmda + file_hmda.format(year), index_col = 0, chunksize = 1e6, na_values = na_values, dtype = dtypes_col_hmda)
    
    chunk_list = []  # append each chunk df here 

    ### Loop over the chunks
    for chunk in df_chunk:
        # Filter data
        ## Remove credit unions and mortgage institutions (CHECK WITH AVERY ET AL 2007 LATER), and remove NA in msamd
        chunk_filtered = chunk[chunk.respondent_id.isin(df_lf[df_lf['CERT{}'.format(str(year)[2:4])] >= 0.0].hmprid)].\
                         dropna(axis = 0, how = 'any', subset = ['msamd','loan_amount_000s','applicant_income_000s'])
        
        ## Make a fips column and remove separate state and county
        chunk_filtered['fips'] = chunk_filtered['state_code'].str.zfill(2) + chunk_filtered['county_code'].str.zfill(3)
        chunk_filtered.drop(columns = ['state_code', 'county_code'], inplace = True)
        
        ## Make one ethnicity column and drop the originals 
        list_eth_race = ['applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',\
                         'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4',\
                         'co_applicant_race_5', 'applicant_ethnicity', 'co_applicant_ethnicity']
        chunk_filtered['ethnicity_borrower'] = determineEthnicity(chunk_filtered)
        chunk_filtered.drop(columns = list_eth_race, inplace = True)
        
        ## Add a date var
        chunk_filtered['date'] = year
    
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
                                       'preapproval','loan_amount_000s','applicant_income_000s',\
                                       'rate_spread', 'lien_status','purchaser_type'], inplace = True)
        
        # Add the chunk to the list
        chunk_list.append(chunk_filtered)
        
    # concat the list into dataframe 
    df_hmda = pd.concat(chunk_list)
    
    # Save as zipped dataframe
    df_hmda.to_csv('Data/data_hmda_{}.csv.gz'.format(year), index = False, compression = 'gzip')