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
                  'ls_ever':'Loan Seller',
                  'log_min_distance_ls_ever':'LS x Distance',
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
    stats = df[['nobs', 'adj_rsquared', 'depvar_notrans_mean',\
                'depvar_notrans_median', 'depvar_notrans_std',\
                'fixed effects']].iloc[0,:].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
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
    note_string = '\justify\n\\scriptsize{\\textit{Notes.} P-value in parentheses. LTI = loan-to-income ratio, LS = Loan Seller. The model is estimated with clustered standard errors on the MSA-level.}\n'
    location = results_latex.find('\end{tabular}\n')
    results_latex = results_latex[:location + len('\end{tabular}\n')] + note_string + results_latex[location + len('\end{tabular}\n'):]
    
    return results,results_latex
    

#------------------------------------------------------------
# Call concatResults
#------------------------------------------------------------
# NOTE Since we have 16 results tables, we split it into two separate tables
    
# Set path list
path_list0411 = ['Results/lpm_results_{}.csv'.format(year) for year in range(2004,2011+1)]
path_list1219 = ['Results/lpm_results_{}.csv'.format(year) for year in range(2012,2019+1)]

col_label0411 = ['({})'.format(year) for year in range(2004,2011+1)]
col_label1219 = ['({})'.format(year) for year in range(2012,2019+1)]

# Set title and label
caption0411 = 'Estimation Results Linear Probability Model (2004-2011)'
label0411 = 'tab:results_lpm_0411'

caption1219 = 'Estimation Results Linear Probability Model (2012-2019)'
label1219 = 'tab:results_lpm_1219'

# Call function
# TODO: sidewards table
df_results0411, latex_results0411 = concatResults(path_list0411, col_label = col_label0411,\
                                                  caption = caption0411, label = label0411)

df_results1219, latex_results1219 = concatResults(path_list1219, col_label = col_label1219,\
                                                  caption = caption1219, label = label1219)

#------------------------------------------------------------
# Save df and latex file
#------------------------------------------------------------

df_results0411.to_csv('Results/Table_results_benchmark0411.csv')

text_file_latex_results = open('Results/Table_results_benchmark0411.tex', 'w')
text_file_latex_results.write(latex_results0411)
text_file_latex_results.close()

df_results1219.to_csv('Results/Table_results_benchmark1219.csv')

text_file_latex_results = open('Results/Table_results_benchmark1219.tex', 'w')
text_file_latex_results.write(latex_results1219)
text_file_latex_results.close()

#------------------------------------------------------------
# Plot estimates log_min_distance and log_min_distance_ls_ever
#------------------------------------------------------------

# set plotting function
def PlotParams(params, std, param_names, file_name, aplha = 0.05):
    global dictionary
    
    for var in range(len(params)):
        # Set fig size
        fig, ax = plt.subplots(figsize=(10,8))
        
        # Plot QR
        x_axis = list(range(2004,2019+1))
        ## Params
        ax.plot(x_axis, params[var])
        
        ## Confidence interval (10%)
        ciub = [params[var][i] + 1.960 * std[var][i] for i in range(len(params[var]))]
        cilb = [params[var][i] - 1.960 * std[var][i] for i in range(len(params[var]))]
        
        ax.plot(x_axis, ciub, color = 'steelblue')
        ax.plot(x_axis, cilb, color = 'steelblue')
        ax.fill_between(x_axis, ciub,\
                        cilb, color = 'deepskyblue', alpha = 0.3)
               
        ## Accentuate y = 0.0 
        ax.axhline(0, color = 'darkred', alpha = 0.75)
        
        # Round yticks
        ax.set_yticklabels(ax.get_yticks(), rotation=90, va = 'center')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
        
        # Fix layout
        ax.set_xlim(2004, 2019)
        plt.tight_layout()
        plt.savefig('Figures\{}_{}.png'.format(file_name, param_names[var]))
        
        plt.close()

# Load data and append to list
df_lst = []
for year in range(2004,2019+1):
    df_lst.append(pd.read_csv('Results/lpm_results_{}.csv'.format(year), index_col = 0))
    
# Set params and std list
params = [table.loc[['log_min_distance','log_min_distance_ls_ever'],'params'] for table in df_lst]
stds = [table.loc[['log_min_distance','log_min_distance_ls_ever'],'std'] for table in df_lst]

## Change shape of the lists
params = [[params[i][0] for i in range(len(params))],[params[i][1] for i in range(len(params))]]
stds = [[stds[i][0] for i in range(len(stds))],[stds[i][1] for i in range(len(stds))]]

# Plot
## Prelims
param_names = ['Distance','LS X Distance']
file_name = 'Estimates_lpm'

## Plot
PlotParams(params, stds, param_names, file_name)
