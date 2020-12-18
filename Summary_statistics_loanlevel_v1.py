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


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 1.75, palette = 'Greys_d')

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_main.csv', usecols = ['fips', 'cert', 'date', 'ls',\
                                                  'ls_gse', 'ls_priv', 'sec',\
                                    'min_distance', 'min_distance_cdd',\
                                    'log_min_distance', 'log_min_distance_cdd',\
                                    'lti', 'ln_loanamout', 'ln_appincome',\
                                    'subprime', 'secured', 'pop_area',\
                                    'ln_mfi', 'hhi', 'density'])
# Drop NaNs
df.dropna(inplace = True)

# Drop inf in LTI
df = df[df.lti != np.inf]

#------------------------------------------------------------
# Make table sold vs. not sold
#------------------------------------------------------------
    
# Get summary statistics
sum_stat_mean = df.groupby(df.ls)[['log_min_distance', 'log_min_distance_cdd',\
                     'min_distance', 'min_distance_cdd',\
                     'lti', 'ln_loanamout', 'ln_appincome',\
                     'subprime']].mean().T

# Welch's T test 
vars_test = ['log_min_distance', 'log_min_distance_cdd',\
                     'min_distance', 'min_distance_cdd',\
                     'lti', 'ln_loanamout', 'ln_appincome',\
                     'subprime']
    
welch_stat = np.zeros(len(vars_test))
welch_p = np.zeros(len(vars_test))

for variable,i in zip(vars_test,list(range(len(vars_test)))):
    welch_stat[i], welch_p[i] = ttest_ind(df[df.ls == 0][variable],\
                                df[df.ls == 1][variable],\
                                equal_var = False, nan_policy = 'omit') # p == 0.0

## Add to the table
sum_stat_mean['P-value'] = welch_p
    
# Change name of columns
cols = ['Not Sold','Sold', 'P-value']

sum_stat_mean.columns = cols

# Change index
index_col = ['Distance (pop. weighted; log)', 'Distance (CDD; log)','Distance (pop. weighted; km)', 'Distance (CDD; km)',\
               'LTI', 'Loan Value', 'Income', 'Subprime']

sum_stat_mean.index = index_col

#------------------------------------------------------------
# To Latex
#------------------------------------------------------------
# TODO: Save to latex


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
    latex_table = latex_table[:location_place + len('\begin{table}\n') + 1] + '[ht]' + latex_table[location_place + len('\begin{table}\n') + 1:]
    
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
    txt_loan = template_subheaders.substitute(numcols = results.shape[1] + 1, variable = 'Loan Control Variables')
    
    ## Get locations and insert strings
    location_distance = latex_table.find('Distance (pop. weighted; log)')
    latex_table = latex_table[:location_distance] + txt_distance + latex_table[location_distance:]
    
    location_loan = latex_table.find('LTI')
    latex_table = latex_table[:location_loan] + txt_loan + latex_table[location_loan:]
    
    # Makes sidewaystable
    if sidewaystable:
        latex_table = latex_table.replace('table','sidewaystable',2)
    
    return latex_table

# Call function
caption = 'Summary Statistics -- Loan Level'
label = 'tab:summary_statistics_loan'
size_string = '\\tiny \n'
note = '\justify\n\\scriptsize{\\textit{Notes.} All variables are in logs except Subprime. The P-value is the P-value from a t-test with unequal means and variance. For a description of all variables see subsection~\\ref{subsec:variable_description} and Table~\\ref{tab:variableconstruction}.}'

## Change data type of certain columns/rows
sum_stat_mean = sum_stat_mean.round(decimals = 4)

sum_stat_mean_full_latex = resultsToLatex(sum_stat_mean, caption, label,\
                                 size_string = size_string, note_string = note)

#------------------------------------------------------------
# Save
#------------------------------------------------------------

sum_stat_mean.to_excel('Tables/Summary_statistics_loan.xlsx')

text_ss_tot_latex = open('Tables/Summary_statistics_loan.tex', 'w')
text_ss_tot_latex.write(sum_stat_mean_full_latex)
text_ss_tot_latex.close()

#------------------------------------------------------------
# Plot through time
#------------------------------------------------------------

# Set df_plot
df_plot = df.groupby(['ls','date']).log_min_distance.mean()

# Divide df_plot
df_plot_nonls = df_plot[df_plot.index.get_level_values(0) == 0]
df_plot_ls = df_plot[df_plot.index.get_level_values(0) == 1]

# Drop index level == 0
df_plot_nonls.index = df_plot_nonls.index.droplevel()
df_plot_ls.index = df_plot_ls.index.droplevel()

# Make plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = "Average (in logs)")
ax.plot(df_plot_nonls, linestyle = '-', color = 'black', label = 'Held Loans')
ax.plot(df_plot_ls, linestyle = '--', color = 'black', label = 'Sold Loans')
ax.legend()

fig.savefig('Figures\Log_min_distance_2010_2019.png')

# Plot where Loan Sales are split
df_plot_gse = df[df.ls_gse > 0.0].groupby(['date']).log_min_distance.mean()
df_plot_priv = df[df.ls_priv > 0.0].groupby(['date']).log_min_distance.mean()
df_plot_sec = df[df.sec > 0.0].groupby(['date']).log_min_distance.mean()
df_plot_non = df[df.ls == 0.0].groupby(['date']).log_min_distance.mean()

## Make plot
fig, ax = plt.subplots(figsize=(14, 8))
ax.set(xlabel='Year', ylabel = "Average (in logs)")
ax.plot(df_plot_non, linestyle = '-', color = 'black', label = 'Held Loans')
ax.plot(df_plot_gse, linestyle = '--', color = 'black', label = 'GSE Loans')
ax.plot(df_plot_priv, linestyle = ':', color = 'black', label = 'Priv Loans')
ax.plot(df_plot_sec, linestyle = '-.', color = 'black', label = 'Sec. Loans')
ax.legend()

