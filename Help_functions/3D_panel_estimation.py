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

#------------------------------------------------------------
# Metrics Class (Calculates metrics after estimation)
#------------------------------------------------------------
class Metrics:
    
    def RSS(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.depvar - self.fittedvalues) ** 2
        self.rss = np.sum(squared_errors)
        return self.rss
        
    def TSS(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.depvar)
        squared_errors = (self.depvar - avg_y) ** 2
        self.tss = np.sum(squared_errors)
        return self.tss
    
    def rSquared(self):
        '''returns calculated value of r^2'''
        self.rsquared = 1 - self.rss / self.tss
        return self.rsquared
    
    def adjRSquared(self): 
        ''' Returns the adj. R2'''
        self.adj_rsquared = 1 - (1 - self.rsquared) * ((self.nobs - 1)/(self.nobs - (self.nvars + 1)))
        return self.adj_rsquared
    
    def dfModel(self):
        ''' Calculates the model degrees of freedom.
            rank(X) - intercept'''
        if self.fit_intercept:
            self.df_model = np.linalg.matrix_rank(self.data) - 1
        else:
            self.df_model = np.linalg.matrix_rank(self.data)
        return self.df_model
            
    def dfResid(self):
        ''' Calculates the model degrees of freedom.
            N - rank(X)'''
        self.df_resid = self.nobs - np.linalg.matrix_rank(self.data)
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
    
    def __init__(self, y, X, FE_cols, how):
        self._depvar = y.copy()
        if X.ndim == 1:
            X = pd.DataFrame(X)
        self._data = X.copy()
        self._FE_cols = FE_cols.copy()
        self._how = how
    
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
        
        # Run inventorizeFE
        self = self.inventorizeFE()
        
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
        
        # Call FEColumns
        self = self.FEColumns()
        
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
            raise RuntimeError('3FE does not work yet')
        else:
            raise RuntimeError('Too many FE') 
    
    def transform(self):
        ''' Returns transformed data as df '''
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
    
    # Calculate Residuals
    def residuals(self):
        ''' Returns vector of residuals '''
        if self.fit_intercept:
            if isinstance(self.data, pd.DataFrame):
                data = self.data
                data['cons'] = 1
                self.resid = np.subtract(self.depvar, np.dot(data, self.params))
            else:
                self.resid = np.subtract(self.depvar, np.dot(np.c_[np.ones(self.data.shape[0]), self.data], self.params))
        else:
            self.resid = np.subtract(self.depvar, np.dot(self.data, self.params))
        return self.resid
    
    # non-robust std function
    def nonRobustSTD(self, xtx_inv):
        ''' Returns non-robust standard errors of the parameters.
            
            std = sqrt(SSR/(N - K)); see Wooldridge p.60
        '''
        resid_sqr = np.power(self.resid, 2)
        sigma = np.divide(np.sum(resid_sqr), (self.nobs - self.nvars))
        cov_matrix = np.dot(sigma, xtx_inv)
        self.std = np.sqrt(np.diag(cov_matrix))
        return self.std
    
    def clusteredSTD(self):
        ''' Returns an array of clustered standard errors based on the columns 
            given. '''
            
        if self.cov_type == 'clustered' and not (isinstance(self.cluster_cols, pd.Series) or isinstance(self.cluster_cols, pd.DataFrame)):
            raise TypeError('Cluster columns should be pandas.Series or pandas.DataFrame')
        
        # if cluster_cols is a pandas.Series, make pandas.DataFrame
        if isinstance(self.cluster_cols, pd.Series):
            self.cluster_cols = pd.DataFrame(self.cluster_cols)
        
        # Set column labels to use
        col_labels = self.data.columns
        
        # Add residuals to data
        data = self.data.join(self.resid, how = 'outer')

        # define methods that allows the parallel estimation of the clusters (Sum_g=1^G X'_g u_g u'_g Xg)
        num_cores = mp.cpu_count() # Get number of cores
        
        def applyParallel(dfGrouped, func):
            ''' First runs the function parallel '''
            retLst = Parallel(n_jobs = num_cores)(delayed(func)(group) for name, group in dfGrouped)
            return retLst

        def clusterErrors(df, resid, col_names):
            df_vars = df[col_names]
            df_resid = pd.DataFrame(df[resid])
            return df_vars.T @ df_resid @ df_resid.T @ df_vars
        
        # Cluster the standard errors. For the moment it only accepts 1D or 2D cluster_cols
        if self.cluster_cols.shape[1] == 1.:
            # Add cluster columns to data and set index 
            data = data.join(self.cluster_cols, how = 'outer')
            data.set_index(self.cluster_cols.columns[0], inplace = True)
            
            # Formula (see Cameron & Miller, 2015): (X'X)^-1 B (X'X)^-1
            # where B = Sum_g=1^G X'_g u_g u'_g Xg
            ## Calculate the inverse XX matrix
            xtx = np.dot(data[col_labels].T, data[col_labels])
            xtx_inv = np.linalg.inv(xtx)
            
            ## Calculate the clustered errors
            data_grouped = data.groupby(data.index)
            
            if __name__ == '__main__':
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
            
            # Estimate the variance matrix for each unique group
            if __name__ == '__main__':
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
    
    # TODO: cluster-bootstrap variance matrix (pp.327-328; Cameron & Miller, 2015)
    
    # Calculate parameters, residuals, and standard deviation
    def fit(self, y, X, fit_intercept = False, cov_type = 'nonrobust', cluster_cols = None):
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
        
        # Check whether instance is pandas and transform X to DataFrame if
        # it is a Series. If data is numpy, check dimensions of X, fit intercept
        # and check whether cov_type = 'clustered'
        if isinstance(y, pd.Series) and isinstance(X, pd.Series):
            X = pd.DataFrame(X)
            
            self.depvar = y.copy()
            self.data = X.copy()
            
            if fit_intercept:
                self.intercept = pd.DataFrame({'cons':np.ones(X.shape[0])})
                X = self.intercept.join(X, how = 'outer')
                
        elif isinstance(y, pd.Series) and isinstance(X, pd.DataFrame):
            self.depvar = y.copy()
            self.data = X.copy()
            
            if fit_intercept:
                self.intercept = pd.DataFrame({'cons':np.ones(X.shape[0])})
                X = self.intercept.join(X, how = 'outer')
                
        elif isinstance(y, np.ndarray) and isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1,1)
            
            self.depvar = y.copy()
            self.data = X.copy()
            
            if fit_intercept:
                self.intercept = np.ones(X.shape[0])
                X = np.c_[self.intercept, X]
            
            if cov_type == 'clustered':
                raise TypeError('Clustering only works with pandas DataFrames or Series')
        
        else:
            raise TypeError('Type should be numpy.ndarray, pandas.Series, or pandas.DataFrame')

        # Check column rank of X
        if np.linalg.matrix_rank(X) != X.shape[1]:
            raise RuntimeError("X does not have full column rank")
            
        # Get N and K from X
        self.nobs, self.nvars = X.shape
        
        # Calculate parameters with OLS and set attribute
        # TODO FGLS
        xtx = np.dot(X.T,X)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(X.T,y)
        params = np.dot(xtx_inv, xty)
        
        self.params = params
        
        # Calculate the residuals
        #self.resid = np.subtract(y, np.dot(X, params))
        self.residuals()
        
        # Calculate standard deviation
        if cov_type == 'nonrobust':
            self.nonRobustSTD(xtx_inv)
            
        elif cov_type == 'clustered':
            self.clusteredSTD()

        # Add metrics
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


            