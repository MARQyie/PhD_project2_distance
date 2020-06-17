# OLS estimation
''' This script contains all the methods to estimate a three-dimensional
    panel data with OLS. The script contains the following methods:
        
        1. Fixed effects data transformation
        2. OLS estimation of the transformed data
        3. (Multi-way) standard error clustering
'''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization
from sklearn.utils import resample

#------------------------------------------------------------
# Metrics Class (Calculates metrics after estimation)
#------------------------------------------------------------
class Metrics:
    
    def RSS(self):
        '''returns sum of squared errors (model vs actual)'''
        if self._transform_data:
            squared_errors = (self._depvar_trans - self.fittedvalues) ** 2
            self.rss = np.sum(squared_errors)
            return self.rss
        
        else:
            squared_errors = (self._depvar - self.fittedvalues) ** 2
            self.rss = np.sum(squared_errors)
            return self.rss
        
    def TSS(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        if self._transform_data:
            avg_y = np.mean(self._depvar_trans)
            squared_errors = (self._depvar_trans - avg_y) ** 2
            self.tss = np.sum(squared_errors)
            return self.tss
        
        else:
            avg_y = np.mean(self._depvar)
            squared_errors = (self._depvar - avg_y) ** 2
            self.tss = np.sum(squared_errors)
            return self.tss
    
    def rSquared(self):
        '''returns calculated value of the (centered) r^2'''
        self.rsquared = 1 - self.rss / self.tss
        return self.rsquared
    
    def adjRSquared(self): 
        ''' Returns the adj. R2'''
        self.adj_rsquared = 1 - (1 - self.rsquared) * ((self.nobs - 1)/(self.nobs - (self.nvars + 1)))
        return self.adj_rsquared
    
    def dfModel(self):
        ''' Calculates the model degrees of freedom.
            rank(X).'''
        # NOTE: self._data or self._data_trans never includes an intercept
        if self._transform_data:
            self.df_model = np.linalg.matrix_rank(self._data_trans)
            return self.df_model
        
        else:
            self.df_model = np.linalg.matrix_rank(self._data)
            return self.df_model
            
    def dfResid(self):
        ''' Calculates the model degrees of freedom.
            N - rank(X)'''
        if self._transform_data:
            self.df_resid = self.nobs - np.linalg.matrix_rank(self._data_trans)
            return self.df_resid
        
        else:
            self.df_resid = self.nobs - np.linalg.matrix_rank(self._data)
            return self.df_resid
    
    def mseModel(self):
        """
        Mean squared error the model.

        The explained sum of squares divided by the model degrees of freedom.
        """
        if np.all(self.df_model == 0.0):
            return np.full_like(self.ess, np.nan)
        return self.ess / self.df_model

    def mseResid(self):
        """
        Mean squared error of the residuals.

        The sum of squared residuals divided by the residual degrees of
        freedom.
        """
        if np.all(self.df_resid == 0.0):
            return np.full_like(self.rss, np.nan)
        return self.rss / self.df_resid
    
    def fValue(self):
        """
        F-statistic and the p-value of the fully specified model.

        Calculated as the mean squared error of the model divided by the mean
        squared error of the residuals if the nonrobust covariance is used.
        Otherwise computed using a Wald-like quadratic form that tests whether
        all coefficients (excluding the constant) are zero.
        """
        if self.cov_type != 'nonrobust': #TODO
            return(np.nan, np.nan)
        else:
            # for standard homoscedastic case
            fstat = self.mse_model / self.mse_resid
        
        # P-value
        if self.df_model == 0:
            pval_f = np.full_like(fstat, np.nan)
        else:
            pval_f = stats.f.sf(fstat, self.df_model, self.df_resid)    
    
        return (fstat, pval_f)
    
    def tStat(self):
        ''' Returns an array of t-statistics '''
        self.tstat = self.params / self.std
        return self.tstat
    
    def pValue(self, two_sided = True):
        ''' Returns an array of p-values for the t-stats '''
        if two_sided:
            self.pval = 2 * (1 - stats.t.cdf(np.abs(self.tstat), df = self.df_resid))
        else:
            self.pval = 1 - stats.t.cdf(np.abs(self.tstat), df = self.df_resid)
        return self.pval

#------------------------------------------------------------
# Transformation class
#------------------------------------------------------------

class Transformation:

    def inventorizeFE(self):
        ''' Function to determine how to transform '''
        
        # Check whether there are multiple FE
        self._multFE = ',' in self._how
        
        # If multiple FE split string
        if self._multFE:
            self._sepFE = self._how.split(', ')
        else:
             self._sepFE = self._how.copy()
        
        # Check whether there are combined FE
        if self._multFE:
            self._boolCombFE = ['x' in FE for FE in self._sepFE]
        else:
            self._boolCombFE = 'x' in self._sepFE
        
        return self
        
    def FEColumns(self):
        ''' Creates the correct FE columns. If there are any combinations of 
        FE the method creates a new column '''
        
        # Check whether there is only one FE and whether that has a combination
        if not self._multFE:
            if self._boolCombFE: # has a FE combination               
                # Make new FE column and return to self
                self._FE = self._FE_cols[self._sepFE.split(' x ')].astype(str).agg('-'.join, axis = 1)
                
                return self
            elif not self._boolCombFE: # has no FE combination
                self._FE = self._FE_cols.copy()
                
                return self
            else: # if something goes wrong
                raise ValueError('Column label not in FE columns')
            
        elif self._multFE: # There are more than one FE
            if self._boolCombFE:
                # Get FE that need to be combined
                to_combine = [FE for FE,comb in zip(self._sepFE, self._boolCombFE) if comb]
                
                # Loop over the list entries, combine, and add to a list
                list_comb_FE = []
                for combination in to_combine:
                    list_comb_FE.append(self._FE_cols[combination.split(' x ')].astype(str).agg('-'.join, axis = 1))
                    
                # Make df from list_comb_FE and add non-combined colums
                ## Add non-combined FEs to the list
                not_combine = [FE for FE,comb in zip(self._sepFE, self._boolCombFE) if not comb]
                for non in not_combine:
                    list_comb_FE.append(self._FE_cols[non])
                
                ## Make df
                self._FE = pd.DataFrame(list_comb_FE).T
                
                return self
            elif not self._boolCombFE:
                self._FE = self._FE_cols.copy()
                
            else: # if something goes wrong
                raise ValueError('Column label not in FE columns')
        
        else: # is something goes wrong
            raise ValueError('Column label not in FE columns')
    
    def dataTransformation(self):
        ''' Transforms that data '''
        # NOTE does not work on three FE yet
  
        # Make df with _dep_var and _data
        df = pd.concat([self._depvar, self._data], axis = 1)
        
        # Transform the data
        if self._FE.shape[1] == 1:
            # Get the mean of the df
            df_mean_fe1 = df.groupby(self._FE).transform('mean')
            
            # Demean the df
            df_demeaned = df - df_mean_fe1
            
            # Add _dep_var_demeaned and _data_demeaned to self and return
            self._depvar_demeaned = df_demeaned.iloc[:,0]
            self._data_demeaned = df_demeaned.iloc[:,1:]
            
            return self
            
        elif self._FE.shape[1] == 2:
            # Get the means of the df
            df_mean_fe1 = df.groupby(self._FE.iloc[:,0]).transform('mean')
            df_mean_fe2 = df.groupby(self._FE.iloc[:,1]).transform('mean')
            df_mean = df.mean()
            
            # Demean the df
            df_demeaned = df - df_mean_fe1 - df_mean_fe2 + df_mean
            
            # Add _dep_var_demeaned and _data_demeaned to self and return
            self._depvar_demeaned = df_demeaned.iloc[:,0]
            self._data_demeaned = df_demeaned.iloc[:,1:]
            
            return self
        elif self._FE.shape[1] == 3:
            # Get the means of the df
            df_mean_fe1 = df.groupby(self._FE.iloc[:,0]).transform('mean')
            df_mean_fe2 = df.groupby(self._FE.iloc[:,1]).transform('mean')
            df_mean_fe3 = df.groupby(self._FE.iloc[:,2]).transform('mean')
            df_mean = df.mean()
            
            # Demean the df
            df_demeaned = df - df_mean_fe1 - df_mean_fe2 - df_mean_fe3 + 2 * df_mean
            
            # Add _dep_var_demeaned and _data_demeaned to self and return
            self._depvar_demeaned = df_demeaned.iloc[:,0]
            self._data_demeaned = df_demeaned.iloc[:,1:]
            
        else:
            raise RuntimeError('Too many FE') 
    
    def transform(self):
        ''' Returns transformed data as df '''
        # Run inventorizeFE
        self = self.inventorizeFE()
                     
        # Call FEColumns
        self = self.FEColumns()
        
        #Run dataTransformation
        self = self.dataTransformation()
               
        return (self._depvar_demeaned, self._data_demeaned)
         
#------------------------------------------------------------
# Regression class and nested methods
#------------------------------------------------------------

class MultiDimensionalOLS(Metrics, Transformation):
    ''' MAIN CLASS. Performs the estimation of the model. 
    
    Calls the following subclasses:
        1. Metrics
    '''
    
    # Initialize the class
    def __init__(self):
        self.nobs = None
        self.nvars = None
        self.params = None
        self.std = None
        self.resid = None
        self.cov_type = None
        self._name = 'Code_docs.Help_functions.MD_panel_estimation'
    
    # Calculate Residuals
    def residuals(self):
        ''' Returns vector of residuals '''
        if self._transform_data:
            data = self._data_trans.copy()
            depvar = self._depvar_trans.copy()
        
        else:
            data = self._data.copy()
            depvar = self._depvar.copy()
            
        if self.fit_intercept:
            if isinstance(self._data, pd.DataFrame):
                data = self.intercept.join(data, how = 'outer')
                self.resid = np.subtract(depvar, np.dot(data, self.params))
            else:
                self.resid = np.subtract(depvar, np.dot(np.c_[np.ones(data.shape[0]), data], self.params))
        
        else:
            self.resid = np.subtract(depvar, np.dot(data, self.params))
        return self.resid
    
    # non-robust std function
    def nonRobustSTD(self):
        ''' Returns non-robust standard errors of the parameters.
            
            std = sqrt(SSR/(N - K)); see Wooldridge p.60
        '''
        resid_sqr = np.power(self.resid, 2)
        sigma = np.divide(np.sum(resid_sqr), (self.nobs - self.nvars))
        if self._transform_data:
            xtx = np.dot(self._data_trans.T,self._data_trans)
            xtx_inv = np.linalg.inv(xtx)
        else:
            xtx = np.dot(self._data.T,self._data)
            xtx_inv = np.linalg.inv(xtx)
        cov_matrix = np.dot(sigma, xtx_inv)
        self.std = np.sqrt(np.diag(cov_matrix))
        return self.std
    
    def clusteredSTD(self, wild = False):
        ''' Returns an array of clustered standard errors based on the columns 
            given. '''
            
        if self.cov_type == 'clustered' and not (isinstance(self.cluster_cols, pd.Series) or isinstance(self.cluster_cols, pd.DataFrame)):
            raise TypeError('Cluster columns should be pandas.Series or pandas.DataFrame')
        
        # if cluster_cols is a pandas.Series, make pandas.DataFrame
        if isinstance(self.cluster_cols, pd.Series):
            self.cluster_cols = pd.DataFrame(self.cluster_cols)
               
        # Set column labels to use
        col_labels = self._data.columns
        
        # Check whether data is transformed and add residuals to data
        # NOTE: we adjust the residuals for the number of clusters. 
        # TODO: Adjustments for 2 clusters might not be correct
        if self.cluster_cols.shape[1] == 1.:
            G = float(self.cluster_cols.nunique())
            
        elif self.cluster_cols.shape[1] == 2.:
            G = float(self.cluster_cols.nunique().sum())
            
        else:
            raise RuntimeError('Does not support clustering on more than two levels')
        c = np.sqrt(G / (G - 1))        
        
        if self._transform_data:
            data = self._data_trans.copy()
            if wild:
                data['resid'] = self._resid_wild * c
            else:
                data['resid'] = self.resid * c
        else:
            data = self._data.copy()
            if wild:
                data['resid'] = self._resid_wild * c
            else:
                data['resid'] = self.resid * c

        # define methods that allows for parallel estimation of the clusters (Sum_g=1^G X'_g u_g u'_g Xg)
        num_cores = mp.cpu_count() # Get number of cores
        
        def clusterErrors(df, resid, col_names):
            df_vars = df[col_names]
            df_resid = pd.DataFrame(df[resid])
            return df_vars.T @ df_resid @ df_resid.T @ df_vars
        
        # Cluster the standard errors. For the moment it only accepts 1D or 2D cluster_cols
        if self.cluster_cols.shape[1] == 1.:
            # Add cluster columns to data and set index 
            data = data.join(self.cluster_cols, how = 'outer')
            data.set_index(pd.DataFrame(self.cluster_cols).columns[0], inplace = True)
            
            # Formula (see Cameron & Miller, 2015): (X'X)^-1 B (X'X)^-1
            # where B = Sum_g=1^G X'_g u_g u'_g Xg
            ## Calculate the inverse XX matrix
            xtx = np.dot(data[col_labels].T, data[col_labels])
            xtx_inv = np.linalg.inv(xtx)
            
            ## Calculate the clustered errors
            data_grouped = data.groupby(data.index)
            
            if __name__ == '__main__' or __name__ == self._name:
                error_matrices = Parallel(n_jobs = num_cores)(delayed(clusterErrors)(group, 'resid', col_labels) for name, group in data_grouped)
            
            ## Sum the matrices elementwise
            error_matrix_sum = 0
            for matrix in error_matrices:
                error_matrix_sum += matrix
             
            ## Get the variance-covariance matrix
            cov_matrix = xtx_inv @ error_matrix_sum @ xtx_inv
            
        elif self.cluster_cols.shape[1] == 2.:
            # Identify unique groups and add to data
            group_comb = pd.DataFrame({'group_comb':self.cluster_cols.iloc[:,0].astype(str) + self.cluster_cols.iloc[:,1].astype(str)})
            unique_groups = self.cluster_cols.join(group_comb, how = 'outer')
            data = data.join(unique_groups, how = 'outer')
            
            # Set index
            data.set_index(unique_groups.columns, inplace = True)
            
            ## Calculate the inverse XX matrix
            xtx = np.dot(data[col_labels].T, data[col_labels])
            xtx_inv = np.linalg.inv(xtx)
            
            # Estimate the variance matrix for each unique group
            if __name__ == '__main__' or __name__ == self._name:
                error_matrices_group1 = Parallel(n_jobs = num_cores)(delayed(clusterErrors)(group, 'resid', col_labels) for name, group in data.groupby(data.index.get_level_values(0)))
                error_matrices_group2 = Parallel(n_jobs = num_cores)(delayed(clusterErrors)(group, 'resid', col_labels) for name, group in data.groupby(data.index.get_level_values(1)))
                error_matrices_group12 = Parallel(n_jobs = num_cores)(delayed(clusterErrors)(group, 'resid', col_labels) for name, group in data.groupby(data.index.get_level_values(2)))
                
            ## Sum the matrices elementwise
            error_matrix_sum_group1 = 0
            error_matrix_sum_group2 = 0
            error_matrix_sum_group12 = 0
            for matrix1, matrix2, matrix12 in zip(error_matrices_group1,error_matrices_group2,error_matrices_group12):
                error_matrix_sum_group1 += matrix1
                error_matrix_sum_group2 += matrix2
                error_matrix_sum_group12 += matrix12
                
            ## Get variance-covariance matrices
            cov_matrix_group1 = xtx_inv @ error_matrix_sum_group1 @ xtx_inv
            cov_matrix_group2 = xtx_inv @ error_matrix_sum_group2 @ xtx_inv
            cov_matrix_group12 = xtx_inv @ error_matrix_sum_group12 @ xtx_inv
            
            # Calculate two-way variance matrix
            cov_matrix = cov_matrix_group1 + cov_matrix_group2 - cov_matrix_group12
            # Note that this matrix might not be positive semidefinite. 
            # Cameron, Gelbach and Miller (2011) present a eigendecomp. technique
            # that zeros out the negative eigenvalues in the cov_matrix.
            # TODO add eigendecomp.
            
        else:
            raise RuntimeError('Does not support clustering on more than two levels')
        
        # Calculate the standard errors and return
        self.std = np.sqrt(np.diag(cov_matrix))
        return self.std
    
    def clusterErrorsBootstrap(self, B = 1000):
        ''' Uses a boostrap procedure to estimate a cluster robust variance matrix.
            Returns the std (srqt of the diag of the var matrix). See: (pp.327-328; Cameron & Miller, 2015) '''
        
        # if cluster_cols is a pandas.Series, make pandas.DataFrame
        if isinstance(self.cluster_cols, pd.Series):
            self.cluster_cols = pd.DataFrame(self.cluster_cols)
            
        # Set the correct data
        if self._transform_data:
            depvar = self._depvar_trans.copy()
            data = self._data_trans.copy()
        else:
            depvar = self._depvar.copy()
            data = self._data.copy()
        
        # Define the bootstrap method
        def resampleProcedure(data):
            data_r = resample(data)
            return data_r
        
        def ProcedureCEB(y_grouped, X_grouped):
            
            # Resample data
            y_r = y_grouped.apply(resampleProcedure)
            X_r = X_grouped.apply(resampleProcedure)
            
            # Calculate beta_b and return
            xtx = np.dot(X_r.T,X_r)
            xtx_inv = np.linalg.inv(xtx)
            xty = np.dot(X_r.T,y_r)
            beta_b = np.dot(xtx_inv, xty)  
            
            return beta_b
        
        if self.cluster_cols.shape[1] == 1.:
            # Set index
            depvar = depvar.join(self.cluster_cols, how = 'outer')
            depvar.set_index(pd.DataFrame(self.cluster_cols).columns[0], inplace = True)
            
            data = data.join(self.cluster_cols, how = 'outer')
            data.set_index(pd.DataFrame(self.cluster_cols).columns[0], inplace = True)
            
            # Group data
            depvar_grouped = depvar.groupby(depvar.index)
            data_grouped = data.groupby(depvar.index)
            
            # Run bootstrap
            num_cores = mp.cpu_count()
            
            if __name__ == '__main__' or __name__ == self._name:
                beta_b = Parallel(n_jobs = num_cores, prefer = 'threads')(delayed(ProcedureCEB)(depvar_grouped, data_grouped) for i in range(B))
            
            # Calculate variance matrix
            ## Calculate beta_b_bar
            beta_b_bar = (1 / B) * np.sum(beta_b, axis = 0)
            
            ## Calculate cov_matrix
            cov_matrix = (1 / (B - 1)) * np.sum([(b - beta_b_bar) @ (b - beta_b_bar).T for b in beta_b], axis = 0)
                        
            # Calculate std
            self.std = np.sqrt(np.diag(cov_matrix))
            
            return self.std
            
        elif self.cluster_cols.shape[1] == 2.:
            raise RuntimeError('Two-levels is not implemented yet for bootstrap variance estimation')
            
        else:
            raise RuntimeError('Does not support clustering on more than two levels')    
          
    def clusterErrorsJackknife(self):
        # TODO
        pass
    
    def wildClusterBootstrap(self, B = 1000, alpha = 0.05, weights = 'Rademacher'):
        ''' Returns refined test statistics (pp.344-345; Cameron & Miller, 2015) ''' 

        # Prelim
        if self._transform_data:
            y = self._depvar_trans.copy()
            X = self._data_trans.copy()
        else:
            y = self._depvar.copy()
            X = self._data.copy()
            
        # Step 1: setup
        ## Set Wald t-stat
        wald = self.tStat()
        
        ## Regress y on restricted X for each param left out
        def restrictedEstimations(y, X, col_name):
            ### Get restricted X
            X_res = X.drop(columns = col_name)
            
            ### regress y on X_res
            xtx = np.dot(X_res.T,X_res)
            xtx_inv = np.linalg.inv(xtx)
            xty = np.dot(X_res.T,y)
            beta_res = np.dot(xtx_inv, xty)
            
            ### calculate residuals
            u_res = np.subtract(y, np.dot(X_res, beta_res))

            return X_res, beta_res, u_res
        
        ### Run the method parallel
        num_cores = mp.cpu_count()
        col_names = X.columns.to_numpy()
        
        if __name__ == '__main__' or __name__ == self._name:
            X_res, beta_res, u_res = zip(*Parallel(n_jobs = num_cores, prefer = 'threads')(delayed(restrictedEstimations)(y, X, col) for col in col_names))
        
        # Step 2: Bootstrap procedure
        ## Set up the procedure
        def bootstrapProcedureWild(self, y, x, x_res, beta_res, resids_grouped):           
            ### Form new residuals
            if weights == 'Rademacher':
                resids_as = resids_grouped.apply(lambda x: x * np.random.choice([-1,1], p = [0.5, 0.5]))
            elif weights == 'Mammen':
                p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
                resids_as = resids_grouped.apply(lambda x: x * np.random.choice([(1 - np.sqrt(5)) / 2, (1 + np.sqrt(5)) / 2], p = [p, 1 - p]))
                
            ### Calculate y_asterisk (y_as)
            y_as = x_res @ beta_res + resids_as
            
            ### Regress y_as on X (unrestricted)
            xtx = np.dot(x.T,x)
            xtx_inv = np.linalg.inv(xtx)
            xty = np.dot(x.T,y_as)
            beta_as = np.dot(xtx_inv, xty)        
            
            ### Calculate residuals (not adjusted) and attatch to self
            resid_as = y - x @ beta_as
            self._resid_wild = resid_as
            
            ### Compute standard deviations (std_as)
            std_as = self.clusteredSTD(wild = True)
            
            ### Compute Wald statistic (wald_as)    
            wald_as = beta_as / std_as
            
            return wald_as
        
        ## Loop over all variables
        ### Prelims
        wald_as_list = []
        
        for var, par, resids in zip(X_res, beta_res, u_res):
            ### Group the sample on clusters
            resids_grouped = resids.groupby(self.cluster_cols.iloc[:,0])
                        
            ## Enter loop B = 1000
            if __name__ == '__main__' or __name__ == self._name:
                wald_as = Parallel(n_jobs = num_cores, prefer = 'threads')(delayed(bootstrapProcedureWild)(self, y, X, var, par, resids_grouped) for i in range(B))
            
            wald_as_list.append(wald_as)
            
        # Step 3: Reject Wald statistics
        wild_wald = []
        
        for test_list in wald_as_list:
            ## Calculate the quantiles
            lower, upper = np.quantile(test_list, [alpha / 2, 1 - alpha / 2])
            if wald < lower or wald > upper:
                wild_wald.append(1)
            else:
                wild_wald.append(0)
        
        # Add wild_wald to self and return
        self.wild_wald = wild_wald
        
        return self
        
    def to_dataframe(self):
        ''' Returns a pandas DataFrame containing the estimates, stds, 
            tstats, pvals and, if available, wild_wald '''
        index = self._data.columns.to_numpy() 
        columns = ['params', 'std', 't', 'p', 'nobs', 'adj_rsquared' ,'depvar_notrans_mean' ,'depvar_notrans_median', 'depvar_notrans_std','fixed effects']
        nobs_col = [self.nobs] + [np.nan] * (index.shape[0] - 1)
        r2_col = [self.adj_rsquared] + [np.nan] * (index.shape[0] - 1)
        if self._transform_data:
            depvar_mean_col = [np.exp(self._depvar_trans.mean() - 1)] + [np.nan] * (index.shape[0] - 1)
            depvar_median_col = [np.exp(self._depvar_trans.median() - 1)] + [np.nan] * (index.shape[0] - 1)
            depvar_std_col = [np.exp(self._depvar_trans.std())] + [np.nan] * (index.shape[0] - 1)
        else:
            depvar_mean_col = [np.exp(self._depvar.mean() - 1)] + [np.nan] * (index.shape[0] - 1)
            depvar_median_col = [np.exp(self._depvar.median() - 1)] + [np.nan] * (index.shape[0] - 1)
            depvar_std_col = [np.exp(self._depvar.std())] + [np.nan] * (index.shape[0] - 1)
        if not(self._FE_cols is None):
            if self._FE_cols.shape[1] == 3:
                fixed_effects = ['MSA-Year \& Lender'] + [np.nan] * (index.shape[0] - 1)
            elif self._FE_cols.shape[1] == 2:
                fixed_effects = ['Year \& Lender'] + [np.nan] * (index.shape[0] - 1)
            else:
                fixed_effects = ['Lender'] + [np.nan] * (index.shape[0] - 1)
        else:
            fixed_effects = [np.nan] + [np.nan] * (index.shape[0] - 1)
        
        data = np.array([self.params, self.std, self.tstat, self.pval, nobs_col, r2_col, depvar_mean_col, depvar_median_col, depvar_std_col, fixed_effects]).T
        
        # Add the bootstrap results if available
        if self._bootstrap_residuals:
            columns.append('wild_waldstat')
            data.append(self.wild_wald)
        
        # Make pandas dataframe
        df = pd.DataFrame(data = data, index = index, columns = columns)        
        
        return df
    
    def OLSestimator(self, y, X):
        ''' Returns params with OLS estimation '''
        xtx = np.dot(X.T,X)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(X.T,y)
        params = np.dot(xtx_inv, xty)
        
        return params
    
    # Calculate parameters, residuals, and standard deviation
    def fit(self, y, X, fit_intercept = False, cov_type = 'nonrobust', cluster_cols = None, bootstrap_residuals = False, B = 1000, alpha = 0.05, weights = 'Rademacher', transform_data = False, FE_cols = None, how = None):
        ''' Calculates the parameters by OLS, and the residuals and 
            standard deviation of the parameters. The method allows for
            clustering of the standard errors; this only works with pandas.
                
            Arguments:
                X: 1D or 2D numpy array/pandas DataFrame
                y: 1D numpy array/pandas Series
        '''
                
        # Set attributes
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type     
        self.cluster_cols = cluster_cols
        self._FE_cols = FE_cols
        self._how = how
        self._depvar_trans = None
        self._data_trans = None
        self._transform_data = transform_data
        self._bootstrap_residuals = bootstrap_residuals
        
        # Check whether instance is pandas and transform X to DataFrame if
        # it is a Series. If data is numpy, check dimensions of X, fit intercept
        # and check whether cov_type = 'clustered'
        if isinstance(y, pd.Series) and isinstance(X, pd.Series):
            X = pd.DataFrame(X)
            
            self._depvar = y.copy()
            self._data = X.copy()
            
            if transform_data: # Transform the data
                y_trans, X_trans = self.transform()
                
                if fit_intercept: # Fit intercept to data if necessary
                    self.intercept = pd.DataFrame({'cons':np.ones(X_trans.shape[0])})
                    X_trans = self.intercept.join(X_trans, how = 'outer')
                
                self._depvar_trans = y_trans.copy()
                self._data_trans = X_trans.copy()
                    
            elif not transform_data and fit_intercept:
                self.intercept = pd.DataFrame({'cons':np.ones(X.shape[0])})
                X = self.intercept.join(X, how = 'outer')
                
        elif isinstance(y, pd.Series) and isinstance(X, pd.DataFrame):
            self._depvar = y.copy()
            self._data = X.copy()
            
            if transform_data: # Transform the data
                y_trans, X_trans = self.transform()
                
                if fit_intercept: # Fit intercept to data if necessary
                    self.intercept = pd.DataFrame({'cons':np.ones(X_trans.shape[0])})
                    X_trans = self.intercept.join(X_trans, how = 'outer')
                
                self._depvar_trans = y_trans.copy()
                self._data_trans = X_trans.copy()
            
            elif not transform_data and fit_intercept:
                self.intercept = pd.DataFrame({'cons':np.ones(X.shape[0])})
                X = self.intercept.join(X, how = 'outer')
                
        elif isinstance(y, np.ndarray) and isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1,1)
            
            self._depvar = y.copy()
            self._data = X.copy()
            
            if fit_intercept:
                self.intercept = np.ones(X.shape[0])
                X = np.c_[self.intercept, X]
            
            if cov_type == 'clustered':
                raise TypeError('Clustering only works with pandas DataFrames or Series')
        
        else:
            raise TypeError('Type should be numpy.ndarray, pandas.Series, or pandas.DataFrame')

        # Check column rank of X
        if transform_data:
            if np.linalg.matrix_rank(X_trans) != X_trans.shape[1]:
                raise RuntimeError("Transformed X does not have full column rank")
        else:
            if np.linalg.matrix_rank(X) != X.shape[1]:
                raise RuntimeError("X does not have full column rank")
            
        # Get N and K from X
        self.nobs, self.nvars = X.shape
        
        # Calculate parameters with OLS and set attribute
        # TODO FGLS
        if transform_data:
            self.params = self.OLSestimator(self._depvar_trans, self._data_trans)
        else:
            self.params = self.OLSestimator(self._depvar, self._data)
        
        # Calculate the residuals
        #self.resid = np.subtract(y, np.dot(X, params))
        self.resid = self.residuals()
        
        # Calculate standard deviation
        if cov_type == 'nonrobust':
            self.nonRobustSTD()
            
        elif cov_type == 'clustered':
            self.clusteredSTD()
            if self._bootstrap_residuals:
                self.wild_wald = self.wildClusterBootstrap(B = B, alpha = alpha, weights = weights)
        
        # Add metrics
        if transform_data:
            self.fittedvalues = self.predict(X_trans)
        else:
            self.fittedvalues = self.predict(X)
        self.rss = self.RSS()
        self.tss = self.TSS()
        self.ess = self.tss - self.rss        
        self.rsquared = self.rSquared()
        self.adj_rsquared = self.adjRSquared()
        self.df_model = self.dfModel()
        self.df_resid = self.dfResid()
        self.mse_model = self.mseModel()
        self.mse_resid = self.mseResid()
        self.fvalue = self.fValue()
        self.tstat = self.tStat()
        self.pval = self.pValue()
        
        # Delete constant column
        if fit_intercept:
            X.drop(columns = 'cons', inplace = True)
        
        return self
    
    def predict(self, X):
        """
        Output model prediction.

        Arguments:
        X: 1D or 2D numpy array 
        """
        
        # check if X is 1D or 2D array
        if X.ndim == 1 and isinstance(X, np.ndarray):
            X = X.reshape(-1,1)
        elif X.ndim == 1 and isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        return np.dot(X, self.params) 

#------------------------------------------------------------
# Test
#------------------------------------------------------------

'''
#Load packages
import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Load data and drop na
df = pd.read_csv('Data/data_agg_clean.csv')
df.dropna(inplace = True)

# Set y and X
## Y
y = df['log_min_distance'] 

## X
x_list = ['ls_num', 'lti', 'ln_loanamout', 'ln_appincome', 'subprime', \
               'cb', 'ln_ta', 'ln_emp', 'num_branch', 'ln_pop', 'density', 'hhi', 'ln_mfi'] #secured is only 1
x = df[x_list]

# Get the estimates
## Run model
results = MultiDimensionalOLS().fit(y, x, cov_type = 'clustered', cluster_cols = df.msamd, transform_data = True, FE_cols = df[['msamd','date','cert']], how = 'msamd x date, cert')
results_pooled = MultiDimensionalOLS().fit(y, x, cov_type = 'clustered', cluster_cols = df.msamd)

## Save to df and write to csv file and excel
df_results = results.to_dataframe()
df_results_pooled = results_pooled.to_dataframe()

path = r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition/Results/'
df_results.to_excel(path + 'Results_mainmodel.xlsx')
df_results_pooled.to_excel(path + 'Results_pooledmodel.xlsx')
'''
            