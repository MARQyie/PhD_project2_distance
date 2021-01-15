# Make estimation table

''' This script uses the estimation results to make nice estimation tables.
'''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import numpy as np
import pandas as pd

#------------------------------------------------------------
# Load the data
#------------------------------------------------------------
''' OLD
df_ols_ls = pd.read_csv('Results/Results_ols_full.csv', index_col = 0)
df_ols_int = pd.read_csv('Results/Results_olsint_full.csv', index_col = 0)
df_ols_lsint = pd.read_csv('Results/Results_olslsint_full.csv', index_col = 0)
df_fe_ls_full = pd.read_csv('Results/Results_fels_full.csv', index_col = 0)
df_fe_ls = pd.read_csv('Results/Results_fels_res.csv', index_col = 0)
df_fe_int = pd.read_csv('Results/Results_feint_res.csv', index_col = 0)
df_fe_lsint = pd.read_csv('Results/Results_felsint_res.csv', index_col = 0)
'''
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
                  'perc_broadband':'Internet',
                  'lti':'LTI',
                  'ln_loanamout':'Loan Value',
                  'ln_appincome':'Income',
                  'subprime':'Subprime',
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
    stats = df[['nobs', 'adj_rsquared', 'depvar_notrans_mean',\
                'depvar_notrans_median', 'depvar_notrans_std',\
                'fixed effects', 'msamd', 'cert']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
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
                               column_format = 'p{2.5cm}' + 'p{1.35cm}' * results.shape[1],
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
        df = pd.read_csv(df_path, index_col = 0)
    
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
    size_string = '\\scriptsize \n'
    location = results_latex.find('\centering\n')
    results_latex = results_latex[:location + len('\centering\n')] + size_string + results_latex[location + len('\centering\n'):]
    
    # Add midrule above 'Observations'
    size_midrule = '\\midrule'
    location = results_latex.find('\nObservations')
    results_latex = results_latex[:location] + size_midrule + results_latex[location:]
    
    ## Add note to the table
    # TODO: Add std, tval and stars option
    note_string = '\\\\\scriptsize{\\textit{Notes.} P-value in parentheses. All models are estimated with clustered standard errors on the MSA-level}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------
    
# Set path list
path_list_ols = ['Results/Results_ols_ls_full.csv', 'Results/Results_ols_ls_res.csv',\
                 'Results/Results_ols_int_res.csv', 'Results/Results_ols_lsint_res.csv']
path_list_fe = ['Results/Results_fe_msatcert_ls_full.csv', 'Results/Results_fe_msatcert_ls_res.csv',\
                'Results/Results_fe_tcert_ls_full.csv', 'Results/Results_fe_tcert_ls_res.csv',\
                'Results/Results_fe_tcert_int_res.csv', 'Results/Results_fe_tcert_lsint_res.csv'  ]  

col_label_ols = ['({})'.format(i) for i in range(1,len(path_list_ols) + 1)]
col_label_fe = ['({})'.format(i) for i in range(1,len(path_list_fe) + 1)]

# Set title and label
caption_ols = 'Estimation Results Pooled OLS'
label_ols = 'tab:main_results_ols'
caption_fe = 'Estimation Results Fixed Effects'
label_fe = 'tab:main_results_fe'

# Call function
df_results_ols, latex_results_ols = concatResults(path_list_ols, col_label = col_label_ols,\
                                                  caption = caption_ols, label = label_ols)
df_results_fe, latex_results_fe = concatResults(path_list_fe, col_label = col_label_fe,\
                                                caption = caption_fe, label = label_fe)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results_ols.to_csv('Results/Results_main_ols.csv')
df_results_fe.to_csv('Results/Results_main_fe.csv')

text_file_latex_results = open('Results/Results_main_ols.tex', 'w')
text_file_latex_results.write(latex_results_ols)
text_file_latex_results.close()
text_file_latex_results = open('Results/Results_main_fe.tex', 'w')
text_file_latex_results.write(latex_results_fe)
text_file_latex_results.close()