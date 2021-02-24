# Make estimation table

''' This script uses the estimation results to make nice estimation tables.
'''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Data manipulation
import numpy as np
import pandas as pd

# Plot packages
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set(style = 'ticks', font_scale = 1.5, palette = 'Greys_d')

#------------------------------------------------------------
# Make functions
#------------------------------------------------------------

def estimationTable(df, show = 'pval', stars = False, col_label = 'Est. Results'):
    ''' This function takes a df with estimation results and returns 
        a formatted column. 
        
        Input:  df = pandas dataframe estimation results
                show = Str indicating what to show (std, tval, or pval)
                stars = Boolean, show stars based on pval yes/no
                col_label = Str, label for the resulting pandas dataframe
        
        Output: Pandas dataframe
        '''
    # Prelims
    ## Set dictionary for index and columns
    dictionary = {'ls_num':'Loan Sales',
                  'log_min_distance':'Distance',
                  'log_min_distance_cdd':'Distance (CDD)',
                  'local':'Local',
                  'ls_ever':'Loan Seller',
                  'log_min_distance_ls_ever':'LS x Distance',
                  'log_min_distance_cdd_ls_ever':'LS x Distance (CDD)',
                  'local_ls_ever':'LS x Local',
                  'suprime_ls_ever':'LS X Subprime',
                  'suprime_log_min_distance':'Subprime x Distance',
                  'perc_broadband':'Internet',
                  'lti':'LTI',
                  'ln_loanamout':'Loan Value',
                  'ln_appincome':'Income',
                  'subprime':'Subprime',
                  'lien':'Lien',
                  'owner':'Owner',
                  'preapp':'Pre-application',
                  'coapp':'Co-applicant',
                  'ethnicity_1':'Ethnicity 1',
                  'ethnicity_2':'Ethnicity 2',
                  'ethnicity_3':'Ethnicity 3',
                  'ethnicity_4':'Ethnicity 4',
                  'ethnicity_5':'Ethnicity 5',
                  'loan_type_2':'Loan Type 1',
                  'loan_type_3':'Loan Type 2',
                  'loan_type_4':'Loan Type 3',
                  'sex_1':'Sex',
                  'ln_ta':'Size',
                  'ln_emp':'Employees',
                  'ln_num_branch':'Branches',
                  'cb':'Bank',
                  'ln_density':'Density',
                  'ln_pop_area':'Population',
                  'ln_mfi':'MFI',
                  'hhi':'HHI',
                  'params':'Parameter',
                  'std':'Standard Deviation',
                  't':'$t$-value',
                  'p':'$p$-value',
                  'nobs':'Observations',
                  'adj_rsquared':'Adj. $R^2$',
                  'depvar_notrans_mean':'Depvar mean',
                  'depvar_notrans_median':'Depvar median',
                  'depvar_notrans_std':'Depvar SD',
                  'fixed effects':'FE',
                  'msamd':'MSAs/MDs',
                  'cert':'Lenders',
                  'intercept':'Intercept'}
    
    # Get parameter column and secondary columns (std, tval, pval)
    params = df.params.round(4)
    
    if show == 'std':
        secondary = df.std
    elif show == 'tval':
        secondary = df.t
    else:
        secondary = df.p

    # Transform secondary column 
    # If stars, then add stars to the parameter list
    if stars:
        stars_count = ['*' * i for i in sum([df.p <0.10, df.p <0.05, df.p <0.01])]
        params = ['{:.4f}{}'.format(val, stars) for val, stars in zip(params,stars_count)]
    secondary_formatted = ['({:.4f})'.format(val) for val in secondary]
    
    # Zip lists to make one list
    results = [val for sublist in list(zip(params, secondary_formatted)) for val in sublist]
    
    # Make pandas dataframe
    ## Make index col (make list of lists and zip)
    lol_params = list(zip([dictionary[val] for val in params.index],\
                          ['{} {}'.format(show, val) for val in [dictionary[val] for val in params.index]]))
    index_row = [val for sublist in lol_params for val in sublist]
    
    # Make df
    results_df = pd.DataFrame(results, index = index_row, columns = [col_label])    
    
    # append N, lenders, MSAs, adj. R2, Depvar, and FEs
    ## Make stats lists and maken index labels pretty
    stats = df[['nobs', 'adj_rsquared', 'fixed effects']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    stats.index = [dictionary[val] for val in stats.index]
    
    ### Make df from stats
    stats_df = pd.DataFrame(stats)
    stats_df.columns = [col_label]
    
    ## Append to results_df
    results_df = results_df.append(stats_df)

    return results_df  

def resultsToLatex(results, caption = '', label = ''):
    # Prelim
    function_parameters = dict(na_rep = '',
                               index_names = False,
                               column_format = 'p{2.5cm}' + 'p{1cm}' * results.shape[1],
                               escape = False,
                               multicolumn = True,
                               multicolumn_format = 'c',
                               caption = caption,
                               label = label)
  
    # To Latex
    return results.to_latex(**function_parameters)


def concatResults(path_list, show = 'pval', stars = False, col_label = None, caption = '', label = ''):
    '''Calls estimationTable and returns a concatenated table '''
    
    list_of_results = []
    for df_path, lab in zip(path_list, col_label):
        # Read df
        df = pd.read_csv(df_path, index_col = 0, dtype = {'nobs':'str'})
        
        # Call estimationTable and append to list
        list_of_results.append(estimationTable(df, show = 'pval', stars = False,\
                                               col_label = lab))

    # Concat all list of dfs to a single df
    results = pd.concat(list_of_results, axis = 1)
    
    # Order results
    results = results.loc[list_of_results[-1].index.to_numpy(),:]

    # Rename index
    results.index = [result if not show in result else '' for result in results.index]
        
    # Rename columns if multicolumn
    if '|' in results.columns:
        col_names = np.array([string.split('|') for string in results.columns])
        results.columns = pd.MultiIndex.from_arrays([col_names[:,0], col_names[:,1]], names = ['Method','Number'])
    
    # To latex
    results_latex = resultsToLatex(results, caption, label)
    
    ## Add table placement
    location = results_latex.find('\begin{table}\n')
    results_latex = results_latex[:location + len('\begin{table}\n') + 1] + '[th!]' + results_latex[location + len('\begin{table}\n') + 1:]
    
    ## Make the font size of the table footnotesize
    size_string = '\\tiny \n'
    location = results_latex.find('\centering\n')
    results_latex = results_latex[:location + len('\centering\n')] + size_string + results_latex[location + len('\centering\n'):]
    
    # Add midrule above 'Observations'
    size_midrule = '\\midrule'
    location = results_latex.find('\nObservations')
    results_latex = results_latex[:location] + size_midrule + results_latex[location:]
    
    ## Add note to the table
    # TODO: Add std, tval and stars option
    note_string = '\justify\n\\scriptsize{\\textit{Notes.} Robustness check of the linear probability model. The model is estimated with the within estimator and includes clustered standard errors on the MSA-level. P-value in parentheses. LTI = loan-to-income ratio, LS = Loan Seller.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------
# NOTE Since we have 16 results tables, we split it into two separate tables

# Set path lists
path_cdd_list0411 = ['Robustness_checks/lpm_robust_cdd_{}.csv'.format(year) for year in range(2004,2011+1)]
path_cdd_list1219 = ['Robustness_checks/lpm_robust_cdd_{}.csv'.format(year) for year in range(2012,2019+1)]

path_local_list0411 = ['Robustness_checks/lpm_robust_local_{}.csv'.format(year) for year in range(2004,2011+1)]
path_local_list1219 = ['Robustness_checks/lpm_robust_local_{}.csv'.format(year) for year in range(2012,2019+1)]

# Set column labels
col_label0411 = ['({})'.format(year) for year in range(2004,2011+1)]
col_label1219 = ['({})'.format(year) for year in range(2012,2019+1)]

# Set titles and labels
caption_cdd_0411 = 'Robustness Linear Probability Model -- CDD Distance (2004-2011)'
label_cdd_0411 = 'tab:robust_cdd_lpm_0411'
caption_cdd_1219 = 'Robustness Linear Probability Model -- CDD Distance  (2012-2019)'
label_cdd_1219 = 'tab:robust_cdd_lpm_1219'

caption_local_0411 = 'Robustness Linear Probability Model -- Local (2004-2011)'
label_local_0411 = 'tab:robust_local_lpm_0411'
caption_local_1219 = 'Robustness Linear Probability Model -- Local  (2012-2019)'
label_local_1219 = 'tab:robust_local_lpm_1219'

# Call function
df_results_cdd_0411, latex_results_cdd_0411 = concatResults(path_cdd_list0411, col_label = col_label0411,\
                                                  caption = caption_cdd_0411, label = label_cdd_0411)
df_results_cdd_1219, latex_results_cdd_1219 = concatResults(path_cdd_list1219, col_label = col_label1219,\
                                                  caption = caption_cdd_1219, label = label_cdd_1219)

df_results_local_0411, latex_results_local_0411 = concatResults(path_local_list0411, col_label = col_label0411,\
                                                  caption = caption_local_0411, label = label_local_0411)
df_results_local_1219, latex_results_local_1219 = concatResults(path_local_list1219, col_label = col_label1219,\
                                                  caption = caption_local_1219, label = label_local_1219)

# Save df and latex file
#------------------------------------------------------------

df_results_cdd_0411.to_csv('Robustness_checks/Table_robust_cdd_lpm0411.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_cdd_lpm0411.tex', 'w')
text_file_latex_results.write(latex_results_cdd_0411)
text_file_latex_results.close()

df_results_cdd_1219.to_csv('Robustness_checks/Table_robust_cdd_lpm1219.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_cdd_lpm1219.tex', 'w')
text_file_latex_results.write(latex_results_cdd_1219)
text_file_latex_results.close()

df_results_local_0411.to_csv('Robustness_checks/Table_robust_local_lpm0411.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_local_lpm0411.tex', 'w')
text_file_latex_results.write(latex_results_local_0411)
text_file_latex_results.close()

df_results_local_1219.to_csv('Robustness_checks/Table_robust_local_lpm1219.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_local_lpm1219.tex', 'w')
text_file_latex_results.write(latex_results_local_1219)
text_file_latex_results.close()



