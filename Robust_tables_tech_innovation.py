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
                  'ls':'Loan Sold',
                  'ls_hat':'$\hat{LS}$',
                  'ls_gse':'LS GSE',
                  'ls_priv':'LS Private',
                  'ls_other':'MD',
                  'sec':'Securitization',
                  'ls_ever':'Loan Seller',
                  'log_min_distance_ls':'LS x Distance',
                  'local':'Local',
                  'local_ls':'Local X LS',
                  'dailup':'Dail-up',
                  'broadband':'Broadband',
                  'noint':'No Internet',
                  'automated':'Automated',
                  'csm':'CSM',
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
                  'hw_pval':'DHW p-val'}
    
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
    try:
        stats = df[['nobs', 'adj_rsquared','hw_pval', 'fixed effects']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    except:
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
        df['nobs'] = df.nobs.str[:-2]
        #if '1319' in df_path:
        #    df['fixed effects'] = 'MSA-year, FIPS \& Lender'
    
        # Call estimationTable and append to list
        list_of_results.append(estimationTable(df, show = 'pval', stars = False,\
                                               col_label = lab))

    # Concat all list of dfs to a single df
    results = pd.concat(list_of_results, axis = 1)
    
    # Order results

    ## Get column indexes that are not in fist column and insert in index column 0
    if '1319' in df_path:
        missing_cols = [var for i in range(0,3) for var in list_of_results[i+1].index if var not in list_of_results[0].index]
    else:
        missing_cols = [var for i in range(0,2) for var in list_of_results[i+1].index if var not in list_of_results[0].index]
    target_cols = list_of_results[0].index.tolist()
    for i in range(len(missing_cols)):
        target_cols.insert(i + 2, missing_cols[i])

    # order results    
    results = results.loc[target_cols,:]

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
    note_string = '\justify\n\\scriptsize{\\textit{Notes.} Robustness results of the benchmark model. The model includes dummies for technological innovation, and is estimated with the within estimator and includes clustered standard errors on the MSA-level. The dependent variable is Distance. P-value in parentheses. LTI = loan-to-income ratio.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------
    
# Set path list
path_list1819 = ['Robustness_checks/Benchmark_techinno_1819_{}.csv'.format(i) for i in range(7)]
path_list1819_fs = ['Robustness_checks/Benchmark_techinno_1819_fs_{}.csv'.format(i) for i in [0,3,4,6]]
path_list1319 = ['Robustness_checks/Benchmark_techinno_1319_{}.csv'.format(i) for i in range(9)]
path_list1319_fs = ['Robustness_checks/Benchmark_techinno_1319_fs_{}.csv'.format(i) for i in [0,4,5,6,8]]

col_label1819 = ['({})'.format(i + 1) for i in range(7)]
col_label1819_fs = ['({})'.format(i + 1) for i in range(4)]
col_label1319 = ['({})'.format(i + 1) for i in range(9)]
col_label1319_fs = ['({})'.format(i + 1) for i in range(5)]

# Set title and label
caption1819 = 'Robustness Results Technological Innovation: Automated Lending and Credit Scoring'
label1819 = 'tab:robust_techinno1'
caption1819_fs = 'Robustness Results Technological Innovation: Automated Lending and Credit Scoring (First Stage)'
label1819_fs = 'tab:robust_techinno1_fs'
caption1319 = 'Robustness Results Technological Innovation: Internet Subscriptions'
label1319 = 'tab:robust_techinno2'
caption1319_fs = 'Robustness Results Technological Innovation: Internet Subscriptions (First Stage)'
label1319_fs = 'tab:robust_techinno2_fs'

# Call function
df_results1819, latex_results1819 = concatResults(path_list1819, col_label = col_label1819,\
                                                  caption = caption1819, label = label1819)
df_results1819_fs, latex_results1819_fs = concatResults(path_list1819_fs, col_label = col_label1819_fs,\
                                                  caption = caption1819_fs, label = label1819_fs)
df_results1319, latex_results1319 = concatResults(path_list1319, col_label = col_label1319,\
                                                  caption = caption1319, label = label1319)
df_results1319_fs, latex_results1319_fs = concatResults(path_list1319_fs, col_label = col_label1319_fs,\
                                                  caption = caption1319_fs, label = label1319_fs)
#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results1819.to_csv('Robustness_checks/Table_robust_techinno1.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_techinno1.tex', 'w')
text_file_latex_results.write(latex_results1819)
text_file_latex_results.close()

df_results1819_fs.to_csv('Robustness_checks/Table_robust_techinno1_fs.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_techinno1_fs.tex', 'w')
text_file_latex_results.write(latex_results1819_fs)
text_file_latex_results.close()

df_results1319.to_csv('Robustness_checks/Table_robust_techinno2.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_techinno2.tex', 'w')
text_file_latex_results.write(latex_results1319)
text_file_latex_results.close()

df_results1319_fs.to_csv('Robustness_checks/Table_robust_techinno2_fs.csv')

text_file_latex_results = open('Robustness_checks/Table_robust_techinno2_fs.tex', 'w')
text_file_latex_results.write(latex_results1319_fs)
text_file_latex_results.close()
