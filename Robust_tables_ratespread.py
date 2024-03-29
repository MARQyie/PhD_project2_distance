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
    dictionary = {'log_min_distance':'Distance',
                  'log_min_distance_cdd':'Distance (CDD)',
                  'ls':'Loan Sold',
                  'ls_other':'MD',
                  'ls_ever':'Loan Seller',
                  'log_min_distance_ls':'LS x Distance',
                  'log_min_distance_cdd_ls':'LS x Distance (CDD)',
                  'ls_hat':'$\hat{LS}$',
                  'ls_other_log_min_distance_hat':'$\hat{LS}$ x Distance',
                  'ls_other_log_min_distance_cdd_hat':'$\hat{LS}$ x Distance (CDD)',
                  'ls_other_log_min_distance':'MD x Distance',
                  'ls_other_log_min_distance_cdd':'MD x Distance (CDD)',
                  'local':'Local',
                  'local_ls':'Local X LS',
                  'remote':'Remote ',
                  'remote_ls':'Remote X LS',
                  'local_ls_ever':'Local X Loan Seller',
                  'remote_ls_ever':'Remote X Loan Seller',
                  'perc_broadband':'Internet',
                  'lti':'LTI',
                  'ltv':'LTV',
                  'ln_loanamout':'Loan Value',
                  'ln_appincome':'Income',
                  'subprime':'Subprime',
                  'lien':'Lien',
                  'owner':'Owner',
                  'preapp':'Pre-application',
                  'coapp':'Co-applicant',
                  'int_only':'IO',
                  'balloon':'Balloon',
                  'mat':'MAT',
                  'ethnicity_1':'Ethnicity 1',
                  'ethnicity_2':'Ethnicity 2',
                  'ethnicity_3':'Ethnicity 3',
                  'ethnicity_4':'Ethnicity 4',
                  'ethnicity_5':'Ethnicity 5',
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
                  'intercept':'Intercept',
                  'dwh_p':'DHW p-val',
                  'fstat':'F-stat'}
    
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
    stats = df[['nobs', 'adj_rsquared', 'fstat','dwh_p','fixed effects']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
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
                               column_format = 'p{3cm}' + 'p{1.5cm}' * results.shape[1],
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
        if 'fs' in df_path:
            df['dwh_p'] = np.nan
        else:
            df['fstat'] = np.nan
        df['fixed effects'] = 'FIPS \& Lender'
        # Call estimationTable and append to list
        list_of_results.append(estimationTable(df, show = 'pval', stars = False,\
                                               col_label = lab))

    # Concat all list of dfs to a single df
    results = pd.concat(list_of_results, axis = 1)
    
    # Order results
    ## Get column indexes that are not in fist column and insert in index column 0
    missing_cols = [var for i in range(0,len(list_of_results)-1,1) for var in list_of_results[i+1].index if var not in list_of_results[0].index]
    target_cols = list_of_results[0].index.tolist()
    for i in range(len(missing_cols)):
        target_cols.insert(i + 6, missing_cols[i])
        
    ## Remove double Distance (CDD)
    target_cols = target_cols[:10] + target_cols[14:18] + target_cols[22:]
    
    # order results    
    results = results.loc[target_cols,:]

    # Rename index
    results.index = [result if not show in result else '' for result in results.index]
    
    
    # Rename columns if multicolumn
    if '|' in results.columns[0]:
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
    note_string = '\justify\n\\scriptsize{\\textit{Notes.} Robustness results of the rate spread model. The model is estimated with the within estimator and includes clustered standard errors on the MSA-level. P-value in parentheses. LS = Loan Sold, LTI = loan-to-income ratio, LTV = loan-to-value ratio, IO = Interest Only, MAT = Maturity loan 30 years and longer.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------
    
# Set path list
path_list = ['Robustness_checks/Ratespread_robust_distance_fs1.csv',\
             'Robustness_checks/Ratespread_robust_distance_fs2.csv',\
             'Robustness_checks/Ratespread_robust_cdd_fs1.csv',\
             'Robustness_checks/Ratespread_robust_cdd_fs2.csv',\
             'Robustness_checks/Ratespread_robust_distance_ss.csv',\
             'Robustness_checks/Ratespread_robust_cdd_ss.csv']

col_label = ['First Stage|(1)','First Stage|(2)','First Stage|(3)','First Stage|(4)','Second Stage|(5)','Second Stage|(6)']

# Set title and label
caption = 'Robustness Results Rate Spread Model'
label = 'tab:robust_ratespread'

# Call function
df_results, latex_results = concatResults(path_list, col_label = col_label,\
                                                  caption = caption, label = label)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results.to_csv('Robustness_checks/Table_robust_ratespread.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_ratespread.tex', 'w')
text_file_latex_results.write(latex_results)
text_file_latex_results.close()


