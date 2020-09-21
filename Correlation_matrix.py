# Correlation matrix
''' This script returns a table with summary statistics for WP2 '''

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')

#------------------------------------------------------------
# Subset dfs
#------------------------------------------------------------

# prelims
vars_needed = ['log_min_distance', 'log_min_distance_cdd','ls_num', \
               'lti', 'ln_loanamout', 'ln_appincome', 'subprime', 'ln_ta', 'ln_emp',\
               'ln_num_branch', 'cb', 'perc_broadband', 'ln_density', 'ln_pop_area', 'ln_mfi', 'hhi']
# Subset dfs
df_full = df[vars_needed]
df_reduced = df.loc[df.date >= 2013, vars_needed]

#------------------------------------------------------------
# Make correlation matrices
#------------------------------------------------------------

corr_full = df_full.corr(method = 'spearman').round(decimals = 4)
corr_reduced = df_reduced.corr(method = 'spearman').round(decimals = 4)

#------------------------------------------------------------
# Layout correlation matrices
#------------------------------------------------------------
# Change column and index corr matrices
col_names = ['Distance (pop. weighted)', 'Distance (CDD)','Loan Sales',\
               'LTI', 'Loan Value', 'Income', 'Subprime', 'Size', 'Employees',\
               'Branches', 'Bank',  'Internet', 'Density', 'Population', 'MFI', 'HHI']

corr_full.columns = col_names
corr_full.index = col_names

corr_reduced.columns = col_names
corr_reduced.index = col_names

# To laTeX
## Set function
def resultsToLatex(results, caption = '', label = '', size_string = '\\scriptsize \n',  note_string = None, sidewaystable = False):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = True,
                               column_format = 'p{2cm}' + 'p{0.81cm}' * results.shape[1],
                               escape = False,
                               caption = caption,
                               label = label)
  
    # To Latex
    latex_table = results.to_latex(**function_parameters)
    
    # Add table placement
    location_place = latex_table.find('\begin{table}\n')
    latex_table = latex_table[:location_place + len('\begin{table}\n') + 1] + '[th]' + latex_table[location_place + len('\begin{table}\n') + 1:]
    
    # Add scriptsize
    location_size = latex_table.find('\centering\n')
    latex_table = latex_table[:location_size + len('\centering\n')] + size_string + latex_table[location_size + len('\centering\n'):]
    
    # Add note to the table
    if note_string is not None:
        location_note = latex_table.find('\end{tabular}\n')
        note = '\scriptsize{\\textit{Notes.} ' + note_string + '} \n'
        latex_table = latex_table[:location_note + len('\end{tabular}\n')]\
            + note + latex_table[location_note + len('\end{tabular}\n'):]
    
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('table','sidewaystable',2)
    
    return latex_table

## Call function
caption_full = 'Correlation Matrix Full Sample'
label_full = 'tab:corr_full'
size_string = '\\tiny \n'
note_full = "Spearman's rank correlation coefficient"
sidewaystable = True
corr_full_latex = resultsToLatex(corr_full, caption_full, label_full,\
                                 size_string = size_string, note_string = note_full,\
                                 sidewaystable = sidewaystable)

caption_reduced = 'Correlation Matrix Reduced Sample'
label_reduced = 'tab:corr_reduced'
note_reduced = "Spearman's rank correlation coefficient"
corr_reduced_latex = resultsToLatex(corr_reduced, caption_reduced, label_reduced,\
                                    size_string = size_string, note_string = note_reduced,\
                                    sidewaystable = sidewaystable)
                   
#------------------------------------------------------------
# Save
#------------------------------------------------------------

text_corr_full_latex = open('Tables/Correlation_full.tex', 'w')
text_corr_full_latex.write(corr_full_latex)
text_corr_full_latex.close()

text_corr_reduced_latex = open('Tables/Correlation_reduced.tex', 'w')
text_corr_reduced_latex.write(corr_reduced_latex)
text_corr_reduced_latex.close()
