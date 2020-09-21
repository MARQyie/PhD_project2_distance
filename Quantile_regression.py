# Quantile regression

''' 
    '''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.regression.quantile_regression as sqr
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set(style = 'ticks', font_scale = 1.5, palette = 'Greys_d')


#------------------------------------------------------------
# Parameters
#------------------------------------------------------------

num_cores = mp.cpu_count()

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')
df['intercept'] = 1

#------------------------------------------------------------
# Make full and restricted df and set variables
#------------------------------------------------------------

# Make dfs
df_full = df.copy()
df_red = df.dropna()

# Set variables
## Y vars
y_var = 'log_min_distance'
y_var_robust = 'log_min_distance_cdd'

## X vars interest
ls_num = ['ls_num']
perc_broadband = ['perc_broadband']
ls_val = ['ls_val']
ls_split = ['ls_gse_num', 'ls_priv_num', 'sec_num']
perc_noint = ['perc_noint']

## X vars control

## X vars
x_var = ['lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'ln_ta', 'ln_emp',  'ln_num_branch',  'cb', 'ln_density','ln_pop_area', 'ln_mfi',  'hhi']

#------------------------------------------------------------
# Prelims
#------------------------------------------------------------

## Set dictionary for index and columns
dictionary = {'ls_num':'Loan Sales',
                  'ls_val':'Loan Sales (\$)',
                  'ls_gse_num':'GSE',
                  'ls_priv_num':'Private',
                  'sec_num':'Securitization',
                  'perc_broadband':'Internet',
                  'perc_noint':'No Internet',
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

#------------------------------------------------------------
# Quantile Regressions
#------------------------------------------------------------

# Setup functions that loops over quantiles
def fit_model(data, y, x, q):
    mod = sqr.QuantReg(data[y], data[x])
    res = mod.fit(q=q)
    return res.params.tolist(), res.conf_int()[0].tolist(), res.conf_int()[1].tolist()

# Load ols data
df_ols_ls_full = pd.read_csv('Results/Results_ols_ls_full.csv', index_col = 0)
#df_ols_ls_red = pd.read_csv('Results/Results_ols_ls_res.csv', index_col = 0)
#df_ols_lsint_red = pd.read_csv('Results/Results_ols_lsint_res.csv', index_col = 0)

# Set quantiles
quantiles = np.arange(.01, .99, .01)
    
# Run models
if __name__ == '__main__':
    params_quant_lsnum_full, cilb_quant_lsnum_full, ciub_quant_lsnum_full = zip(*Parallel(n_jobs=num_cores)(delayed(fit_model)(df_full, y_var, ls_num + x_var + ['intercept'], q) for q in quantiles))
    #params_quant_lsnum_red, cilb_quant_lsnum_red, ciub_quant_lsnum_red = zip(*Parallel(n_jobs=num_cores)(delayed(fit_model)(df_red, y_var, ls_num + x_var + ['intercept'], q) for q in quantiles))
    #params_quant_lsint_red, cilb_quant_lsint_red, ciub_quant_lsint_red = zip(*Parallel(n_jobs=num_cores)(delayed(fit_model)(df_red, y_var, ls_num + perc_broadband + x_var + ['intercept'], q) for q in quantiles))

#------------------------------------------------------------
# Plots
#------------------------------------------------------------

# Function to plot all vars at once
def QRPlotAll(params_qr, cilb_qr, ciub_qr, res_ols, file_name, aplha = 0.05):
    global dictionary
    
    # Set fig size
    plt.figure(figsize=(16, 20))
    
    for var in range(len(params_qr[0])):
        # Get subplot
        plt.subplot(4,4,var + 1)
        
        # Plot QR
        ## Params
        plt.plot(quantiles, [params_qr[i][var] for i in range(len(params_qr))])
        
        ## Confidence interval
        plt.plot(quantiles, [ciub_qr[i][var] for i in range(len(ciub_qr))], color = 'steelblue')
        plt.plot(quantiles, [cilb_qr[i][var] for i in range(len(cilb_qr))], color = 'steelblue')
        plt.fill_between(quantiles, [ciub_qr[i][var] for i in range(len(ciub_qr))],\
                                     [cilb_qr[i][var] for i in range(len(cilb_qr))],\
                                     color = 'deepskyblue', alpha = 0.3)
        
        # plot OLS 
        plt.axhline(res_ols.params[var], color = 'black')
        plt.axhline(res_ols.params[var] - res_ols['std'][var] * 1.960, color = 'black', linestyle = '--')
        plt.axhline(res_ols.params[var] + res_ols['std'][var] * 1.960, color = 'black', linestyle = '--')
        
        ## Accentuate y = 0.0 
        plt.axhline(0, color = 'darkred', alpha = 0.75)
        
        # Set title
        #plt.title(dictionary[res_ols.index[var]])
    
    plt.tight_layout()
    plt.savefig('Figures\{}.png'.format(file_name))

# Function to plot everything separately
def QRPlotSep(params_qr, cilb_qr, ciub_qr, res_ols, file_name, aplha = 0.05):
    global dictionary
    
    for var in range(len(params_qr[0])):
        # Set fig size
        fig, ax = plt.subplots(figsize=(10,8))
        
        # Plot QR
        ## Params
        ax.plot(quantiles, [params_qr[i][var] for i in range(len(params_qr))])
        
        ## Confidence interval
        ax.plot(quantiles, [ciub_qr[i][var] for i in range(len(ciub_qr))], color = 'steelblue')
        ax.plot(quantiles, [cilb_qr[i][var] for i in range(len(cilb_qr))], color = 'steelblue')
        ax.fill_between(quantiles, [ciub_qr[i][var] for i in range(len(ciub_qr))],\
                                     [cilb_qr[i][var] for i in range(len(cilb_qr))],\
                                     color = 'deepskyblue', alpha = 0.3)
        
        # plot OLS 
        ax.axhline(res_ols.params[var], color = 'black')
        ax.axhline(res_ols.params[var] - res_ols['std'][var] * 1.960, color = 'black', linestyle = '--')
        ax.axhline(res_ols.params[var] + res_ols['std'][var] * 1.960, color = 'black', linestyle = '--')
        
        ## Accentuate y = 0.0 
        ax.axhline(0, color = 'darkred', alpha = 0.75)
        
        # Set title
        #ax.set(title = dictionary[res_ols.index[var]])
        
        # Round yticks
        ax.set_yticklabels(ax.get_yticks(), rotation=90, va = 'center')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    
        plt.tight_layout()
        plt.savefig('Figures\{}_{}.png'.format(file_name, res_ols.index[var]))
        
        plt.close()
    
# Plot
## All
QRPlotAll(params_quant_lsnum_full, cilb_quant_lsnum_full, ciub_quant_lsnum_full, df_ols_ls_full, 'QR_lsnum_full')
#QRPlotAll(params_quant_lsnum_red, cilb_quant_lsnum_red, ciub_quant_lsnum_red, df_ols_ls_red, 'QR_lsnum_red')
#QRPlotAll(params_quant_lsint_red, cilb_quant_lsint_red, ciub_quant_lsint_red, df_ols_lsint_red, 'QR_lsint_red')
        
## Separately
QRPlotSep(params_quant_lsnum_full, cilb_quant_lsnum_full, ciub_quant_lsnum_full, df_ols_ls_full, 'QR_lsnum_full')
#QRPlotSep(params_quant_lsnum_red, cilb_quant_lsnum_red, ciub_quant_lsnum_red, df_ols_ls_red, 'QR_lsnum_red')
#QRPlotSep(params_quant_lsint_red, cilb_quant_lsint_red, ciub_quant_lsint_red, df_ols_lsint_red, 'QR_lsint_red')