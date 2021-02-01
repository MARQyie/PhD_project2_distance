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

columns = ['date','msamd','date','loan_originated','log_min_distance', 'log_min_distance_cdd','min_distance',\
           'min_distance_cdd','ls','ls_gse','ls_priv','sec','lti','ln_loanamout',\
           'ln_appincome','subprime','lien','owner','preapp', 'coapp',\
           'ethnicity_1', 'ethnicity_2','ethnicity_3', 'ethnicity_4',\
           'ethnicity_5','sex_1', 'loan_type_2', 'loan_type_3', 'loan_type_4','ln_ta',\
           'ln_emp', 'ln_num_branch','cb','rate_spread','ltv','int_only','balloon','mat','loan_term']

dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# Make full dataset
## All applications
df_full_app = dd_main[columns[:-6]].compute()

## Only originations
df_full_ori = dd_main[dd_main.loan_originated == 1][columns[:-6]].drop('loan_originated', axis = 1).compute()

# Make dataset >2017
## Only originations
# TODO

#------------------------------------------------------------
# Make table
#------------------------------------------------------------

# Get summary statistics
ss_full = df_full[vars_needed].describe(percentiles = [.01, .05, .5, .95, .99]).T[['mean','50%','std']]

''' # Turn on to check percentiles
percentiles_df = df_full[vars_needed].describe(percentiles = np.arange(0.01, 1, 0.01)).T
'''
''' NOTES ON PERCENTILES
    LS starts at 29%
    ls GSE and LS PRIVE start at 58%
    SEC starts at 98%
    Both distances become positive at 15%
    BANK about 53% of the lenders are banks (little over half)
    Subprime starts at 68%
'''

# Add extra stats
## MSA-Lender-Years
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.shape[0], 'std':np.nan}, index = ['MSA-lender-years']))

## MSAs
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.msamd.nunique(), 'std':np.nan}, index = ['MSA']))

## Lenders
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.cert.nunique(), 'std':np.nan}, index = ['Lender']))

## Years
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.date.nunique(), 'std':np.nan}, index = ['Years']))

# Change name of columns
cols_full = ['Mean','Median','S.E.']

ss_full.columns = cols_full

# Change index
index_col = ['Distance (pop. weighted; log)', 'Distance (CDD; log)','Distance (pop. weighted; km)', 'Distance (CDD; km)', 'Loan Sales', \
             'LS GSE', 'LS Private', 'Securitization',\
               'LTI', 'Loan Value', 'Income', 'Subprime', 'Size', 'Employees',\
               'Branches', 'Bank', 'Internet', 'Density', 'Population', 'MFI', 'HHI','MSA-lender-years',
               'MSAs','Lenders','Years']

ss_full.index = index_col

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------

# Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{3.5cm}' + 'p{1.25cm}' * results.shape[1],
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
    if latex_table.find('\nMSA-lender-years') >= 0:
        size_midrule = '\\midrule'
        location_mid = latex_table.find('\nMSA-lender-years')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
        
    # Add headers for dependent var, ls vars, control vars 
    ## Set strings
    from string import Template
    template_firstsubheader = Template('\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    template_subheaders = Template('& ' * results.shape[1] + '\\\\\n' + '\multicolumn{$numcols}{l}{\\textbf{$variable}}\\\\\n')
    
    txt_distance = template_firstsubheader.substitute(numcols = results.shape[1] + 1, variable = 'Dependent Variable')
    txt_ls = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Loan Sales Variables')
    txt_loan = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Loan Control Variables')
    txt_lend = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Lender Control Variables')
    txt_msa = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'MSA Control Variables')
    
    ## Get locations and insert strings
    location_distance = latex_table.find('Distance (pop. weighted; log)')
    latex_table = latex_table[:location_distance] + txt_distance + latex_table[location_distance:]
    
    location_ls = latex_table.find('Loan Sales')
    latex_table = latex_table[:location_ls] + txt_ls + latex_table[location_ls:]
    
    location_loan = latex_table.find('LTI')
    latex_table = latex_table[:location_loan] + txt_loan + latex_table[location_loan:]
    
    location_lend = latex_table.find('Size')
    latex_table = latex_table[:location_lend] + txt_lend + latex_table[location_lend:]
    
    location_msa = latex_table.find('Internet')
    latex_table = latex_table[:location_msa] + txt_msa + latex_table[location_msa:]    
    
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('table','sidewaystable',2)
    
    return latex_table

# Call function
caption = 'Summary Statistics'
label = 'tab:summary_statistics'
size_string = '\\tiny \n'
note = '\justify\n\\scriptsize{\\textit{Notes.} All variables are in logs except Loan Sales, Subprime, Bank, Internet, and HHI. For a description of all variables see subsection~\\ref{subsec:variable_description} and Table~\\ref{tab:variableconstruction}.}'

## Change data type of certain columns/rows
ss_full = ss_full.round(decimals = 4)
ss_full = ss_full.astype(str)
ss_full.iloc[-4:,0] = ss_full.iloc[-4:,0].str[:-2]
ss_full.iloc[-1,:] = ss_full.iloc[-1,:].str.replace('nan','')
ss_full.iloc[-2,:] = ss_full.iloc[-2,:].str.replace('nan','')
ss_full.iloc[-3,:] = ss_full.iloc[-3,:].str.replace('nan','')
ss_full.iloc[-4,:] = ss_full.iloc[-4,:].str.replace('nan','')
ss_full.iloc[3,:] = ss_full.iloc[3,:].astype(str).str.replace('nan','')

ss_full_latex = resultsToLatex(ss_full, caption, label,\
                                 size_string = size_string, note_string = note)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss_full.to_excel('Tables/Summary_statistics.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_tot_latex.write(ss_full_latex)
text_ss_tot_latex.close()