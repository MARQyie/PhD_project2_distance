# Summary statistics
''' This script returns a table with summary statistics for WP2 '''

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd

# Parallel functionality pandas via Dask
import dask.dataframe as dd 
import dask 

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

columns = ['date','msamd','fips','cert','date','loan_originated','log_min_distance', 'log_min_distance_cdd','min_distance',\
           'min_distance_cdd','local','ls','ls_gse','ls_priv','sec','lti','ln_loanamout',\
           'ln_appincome','subprime','lien','owner', 'coapp', 'ln_ta',\
           'ln_emp', 'ln_num_branch','cb','rate_spread','ltv','int_only','balloon','mat','loan_term']

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# Make full dataset
## All applications
df_full_app = pd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

## Only originations
df_full_ori = df_full_app[df_full_app.loan_originated == 1].drop('loan_originated', axis = 1)

## Add remote dummy
df_full_ori['remote'] = (df_full_ori['local'] - 1).abs()

## Change columns
columns[10] = 'remote'

#------------------------------------------------------------
# Make table
#------------------------------------------------------------

# Get summary statistics
## Full, only originations and denied
sumstat_full_ori = df_full_ori[columns[6:-6]].describe(percentiles = [.5]).T[['mean','50%','std']]

## >2017, only originations
'''
columns_1819 = ['rate_spread','log_min_distance','local',\
           'ls','ltv','lti','ln_loanamout','ln_appincome','int_only','balloon',\
           'lien','mat','hoepa','owner','preapp', 'coapp','loan_originated',\
           'loan_term']
sumstat_full_ori_1819 = df_full_ori_1819[columns_1819].describe(percentiles = [.5]).T[['mean','50%','std']] '''

# Add extra stats
## Full, only originations and denied
sumstat_full_ori = sumstat_full_ori.append(pd.DataFrame({'mean':df_full_ori.shape[0], 'std':np.nan}, index = ['Observations'])) # MSA-Lender-Years
sumstat_full_ori = sumstat_full_ori.append(pd.DataFrame({'mean':df_full_ori.fips.nunique(), 'std':np.nan}, index = ['FIPS'])) # MSAs
sumstat_full_ori = sumstat_full_ori.append(pd.DataFrame({'mean':df_full_ori.msamd.nunique(), 'std':np.nan}, index = ['MSA'])) # MSAs
sumstat_full_ori = sumstat_full_ori.append(pd.DataFrame({'mean':df_full_ori.cert.nunique(), 'std':np.nan}, index = ['Lender'])) # Lenders
sumstat_full_ori = sumstat_full_ori.append(pd.DataFrame({'mean':df_full_ori.date.nunique(), 'std':np.nan}, index = ['Years'])) # Years

## <2017, only originations

# Make new df
cols = ['Mean','50\%','S.E.']

index_names = ['Distance (pop. weighted; log)', 'Distance (CDD; log)','Distance (pop. weighted; km)',\
           'Distance (CDD; km)','Remote','Sold','Sold to GSE','Sold to private','Securitized','LTI','Loan Value (log)',\
           'Income (log)','Subprime','Lien','Owner Occ.', 'Co-applicant','Size (log)',\
           'Employees (log)', 'Branches (log)','Bank','Observations','FIPS','MSA','Lender','Years']

sumstat_full_ori.index = index_names
sumstat_full_ori.columns = cols

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{4cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
    
    # Add table placement
    location_place = latex_table.find('\begin{table}\n')
    latex_table = latex_table[:location_place + len('\begin{table}\n') + 1] + '[th!]' + latex_table[location_place + len('\begin{table}\n') + 1:]
    
    # Add script size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + note_string + latex_table[location_note + len('\end{tabular}\n'):]
            
    # Add midrule above 'Observations'
    if latex_table.find('\nObservations') >= 0:
        size_midrule = '\\midrule'
        location_mid = latex_table.find('\nObservations')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
        
    # Add headers for dependent var, ls vars, control vars 
    ## Set strings
    from string import Template
    #template_firstsubheader = Template('\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    template_subheaders = Template('& ' * results.shape[1] + '\\\\\n' + '\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    
    txt_distance = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Distance Variables')
    txt_ls = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Loan Sales Variables')
    txt_loan = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Loan Control Variables')
    txt_lend = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Lender Control Variables')
    
    ## Get locations and insert strings
    location_distance = latex_table.find('Distance (pop. weighted; log)')
    latex_table = latex_table[:location_distance] + txt_distance + latex_table[location_distance:]
    
    location_ls = latex_table.find('Sold')
    latex_table = latex_table[:location_ls] + txt_ls + latex_table[location_ls:]
    
    location_loan = latex_table.find('LTI')
    latex_table = latex_table[:location_loan] + txt_loan + latex_table[location_loan:]
    
    location_lend = latex_table.find('Size')
    latex_table = latex_table[:location_lend] + txt_lend + latex_table[location_lend:]
    
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('table','sidewaystable',2)
    
    return latex_table

# Call function
caption = 'Summary Statistics Full Sample'
label = 'tab:summary_statistics'
size_string = '\\scriptsize \n'
note = '\justify\n\\scriptsize{\\textit{Notes.} Summary statistics of the full sample. Mean, \%50, and S.E. stand for the mean, median and standard deviation, respectively. FIPS stands for Federal Information Processing Standard, which is a five-digit code that uniquely  identifies counties. For a description of all variables see subsection~\\ref{subsec:variable_description} and Table~\\ref{tab:variableconstruction}.}'

## Change data type of certain columns/rows
sumstat_full_ori.iloc[:-5,:] = sumstat_full_ori.iloc[:-5,:].round(4)
sumstat_full_ori.iloc[-5:,0] = sumstat_full_ori.iloc[-5:,0].astype(str).str[:-2]

sumstats_latex = resultsToLatex(sumstat_full_ori, caption, label,\
                                 size_string = size_string, note_string = note)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

sumstat_full_ori.to_excel('Tables/Summary_statistics_full.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics_full.tex', 'w')
text_ss_tot_latex.write(sumstats_latex)
text_ss_tot_latex.close()

#------------------------------------------------------------
# Summary Statistics > 2017
#------------------------------------------------------------

del df_full_app

# Load and setup data
df_ori_1819 = df_full_ori[df_full_ori.date.astype(int) >= 2018]
df_ori_1819 = df_ori_1819[(df_ori_1819.rate_spread > -150) & (df_ori_1819.rate_spread < 10) &\
                  (df_ori_1819.loan_term < 2400) & (df_ori_1819.ltv < 87)]
    
# Get summary statistics
## >2017, only originations
sumstat_ori_1819 = df_ori_1819[columns[-6:]].describe(percentiles = [.5]).T[['mean','50%','std']]

# Add extra stats
## < 2017, only originations
sumstat_ori_1819 = sumstat_ori_1819.append(pd.DataFrame({'mean':df_ori_1819.shape[0], 'std':np.nan}, index = ['Observations'])) # MSA-Lender-Years
sumstat_ori_1819 = sumstat_ori_1819.append(pd.DataFrame({'mean':df_ori_1819.fips.nunique(), 'std':np.nan}, index = ['FIPS'])) # MSAs
sumstat_ori_1819 = sumstat_ori_1819.append(pd.DataFrame({'mean':df_ori_1819.msamd.nunique(), 'std':np.nan}, index = ['MSA'])) # MSAs
sumstat_ori_1819 = sumstat_ori_1819.append(pd.DataFrame({'mean':df_ori_1819.cert.nunique(), 'std':np.nan}, index = ['Lender'])) # Lenders
sumstat_ori_1819 = sumstat_ori_1819.append(pd.DataFrame({'mean':df_ori_1819.date.nunique(), 'std':np.nan}, index = ['Years'])) # Years

# Make df pretty
index_names = ['Rate Spread', 'LTV', 'IO', 'Balloon', 'MAT', 'Loan Term','Observations','FIPS','MSA','Lender','Years']

sumstat_ori_1819.index = index_names
sumstat_ori_1819.columns = cols

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{3cm}' + 'p{1.5cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
    
    # Add table placement
    location_place = latex_table.find('\begin{table}\n')
    latex_table = latex_table[:location_place + len('\begin{table}\n') + 1] + '[th!]' + latex_table[location_place + len('\begin{table}\n') + 1:]
    
    # Add script size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + note_string + latex_table[location_note + len('\end{tabular}\n'):]
            
    # Add midrule above 'Observations'
    if latex_table.find('\nObservations') >= 0:
        size_midrule = '\\midrule'
        location_mid = latex_table.find('\nObservations')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
        
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('table','sidewaystable',2)
    
    return latex_table

# Call function
caption = 'Summary Statistics 2018--2019'
label = 'tab:summary_statistics_1819'
size_string = '\\scriptsize \n'
note = '\justify\n\\scriptsize{\\textit{Notes.} Summary statistics of the variables starting in 2018. Mean, \%50, and S.E. stand for the mean, median and standard deviation, respectively. FIPS stands for Federal Information Processing Standard, which is a five-digit code that uniquely  identifies counties. For a description of all variables see subsection~\\ref{subsec:variable_description} and Table~\\ref{tab:variableconstruction}.}'

## Change data type of certain columns/rows
sumstat_ori_1819.iloc[:-5,:] = sumstat_ori_1819.iloc[:-5,:].round(4)
sumstat_ori_1819.iloc[-5:,0] = sumstat_ori_1819.iloc[-5:,0].astype(str).str[:-2]

sumstats_latex = resultsToLatex(sumstat_ori_1819, caption, label,\
                                 size_string = size_string, note_string = note)

sumstat_ori_1819.to_excel('Tables/Summary_statistics_1819.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics_1819.tex', 'w')
text_ss_tot_latex.write(sumstats_latex)
text_ss_tot_latex.close()