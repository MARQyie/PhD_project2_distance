# Summary statistics
''' This script returns a table with summary statistics for WP2 '''

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# prelims
vars_needed = ['log_min_distance', 'log_min_distance_cdd','ls_num', 'perc_broadband',\
               'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'ln_ta', 'ln_emp',\
               'ln_num_branch', 'cb', 'ln_density', 'ln_pop_area', 'ln_mfi', 'hhi']
all_vars = ['msamd','cert','date']

# Subset dfs
df_full = df[all_vars + vars_needed]
df_reduced = df.loc[df.date >= 2013,all_vars + vars_needed]

#------------------------------------------------------------
# Make table
#------------------------------------------------------------

# Get summary statistics
ss_full = df_full[vars_needed].describe().T[['mean','std']]
ss_reduced = df_reduced[vars_needed].describe().T[['mean','std']]

# Add extra stats
## MSA-Lender-Years
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.shape[0], 'std':np.nan}, index = ['MSA-lender-years']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.shape[0], 'std':np.nan}, index = ['MSA-lender-years']))

## MSAs
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.msamd.nunique(), 'std':np.nan}, index = ['MSA']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.msamd.nunique(), 'std':np.nan}, index = ['MSA']))

## Lenders
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.cert.nunique(), 'std':np.nan}, index = ['Lender']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.cert.nunique(), 'std':np.nan}, index = ['Lender']))

## Years
ss_full = ss_full.append(pd.DataFrame({'mean':df_full.date.nunique(), 'std':np.nan}, index = ['Years']))
ss_reduced = ss_reduced.append(pd.DataFrame({'mean':df_reduced.date.nunique(), 'std':np.nan}, index = ['Years']))

# Change name of columns
cols_full = [('2010--2017','Mean'), ('2010--2017','S.E.')]
cols_reduced = [('2013--2017','Mean'), ('2013--2017','S.E.')]

ss_full.columns = cols_full
ss_reduced.columns = cols_reduced

# Concat ss_full and ss_reduced
ss_tot = pd.concat([ss_full, ss_reduced], axis = 1)

# Change index
index_col = ['Distance (pop. weighted)', 'Distance (CDD)','Loan Sales', 'Internet',\
               'LTI', 'Loan Value', 'Income', 'Subprime', 'Size', 'Employees',\
               'Branches', 'Bank', 'Density', 'Population', 'MFI', 'HHI','MSA-lender-years',
               'MSAs','Lenders','Years']

ss_tot.index = index_col

# Add 2010 and 2017 columns + difference column and Welch test
ss_2013 = df_full.loc[df_full.date == 2013,vars_needed].describe().T['mean']
ss_2017 = df_full.loc[df_full.date == 2017,vars_needed].describe().T['mean']
ss_diff = ss_2017 - ss_2013

## Welch t-test
for var in vars_needed:
    stat, pval = ttest_ind(df_full.loc[df_full.date == 2017,vars_needed], \
                           df_full.loc[df_full.date == 2013,vars_needed], \
                           equal_var = False, nan_policy = 'omit')
    
## Concat to one df
df_ttest = pd.concat([ss_2013, ss_2017, ss_diff, pd.Series(pval, index = ss_2013.index)], axis = 1)

## Change col names
cols_ttest = ['2013','2017','$\Delta$','P-value']
df_ttest.columns = cols_ttest

## Change index names
df_ttest.index = [x for x in index_col if x not in ['MSA-lender-years', 'MSAs', 'Lenders', 'Years']]

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
    latex_table = latex_table[:location_place + len('\begin{table}\n') + 1] + '[th]' + latex_table[location_place + len('\begin{table}\n') + 1:]
    
    # Add script size
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        note = '\\\\\scriptsize{\\textit{Notes.} ' + note_string + '} \n'
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + note + latex_table[location_note + len('\end{tabular}\n'):]
            
    # Add midrule above 'Observations'
    if latex_table.find('\nMSA-lender-years') >= 0:
        size_midrule = '\\midrule'
        location_mid = latex_table.find('\nMSA-lender-years')
        latex_table = latex_table[:location_mid] + size_midrule + latex_table[location_mid:]
    
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('table','sidewaystable',2)
    
    return latex_table

# Call function
caption = 'Summary Statistics'
label = 'tab:summary_statistics'
size_string = '\\scriptsize \n'
note = None

caption_ttest = "Welch's T-test"
label_ttest = 'tab:welch_ttest'
note_ttest = "Differences in means are tested by Welch's t-test with unequal variances"

## Change data type of certain columns/rows
ss_tot = ss_tot.round(decimals = 4)
ss_tot = ss_tot.astype(str)
ss_tot.iloc[-4:,0] = ss_tot.iloc[-4:,0].str[:-2]
ss_tot.iloc[-4:,2] = ss_tot.iloc[-4:,2].str[:-2]
ss_tot.iloc[-1,:] = ss_tot.iloc[-1,:].str.replace('nan','')
ss_tot.iloc[-2,:] = ss_tot.iloc[-2,:].str.replace('nan','')
ss_tot.iloc[-3,:] = ss_tot.iloc[-3,:].str.replace('nan','')
ss_tot.iloc[-4,:] = ss_tot.iloc[-4,:].str.replace('nan','')
ss_tot.iloc[3,:] = ss_tot.iloc[3,:].astype(str).str.replace('nan','')

ss_tot_latex = resultsToLatex(ss_tot, caption, label,\
                                 size_string = size_string, note_string = note)
df_ttest_latex = resultsToLatex(df_ttest.round(decimals = 4), caption_ttest, label_ttest,\
                                 size_string = size_string, note_string = note_ttest)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

ss_tot.to_excel('Tables/Summary_statistics.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics.tex', 'w')
text_ss_tot_latex.write(ss_tot_latex)
text_ss_tot_latex.close()

text_df_ttest_latex= open('Tables/Welchs_ttest.tex', 'w')
text_df_ttest_latex.write(df_ttest_latex)
text_df_ttest_latex.close()
