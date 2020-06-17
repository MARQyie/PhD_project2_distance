# Random Effects estimation
''' This script contains all the methods to estimate a three-dimensional
    panel data with Random Effects. The procedure can only estimate a model
    with two Random effects. Many methods are from MultiDimensional OLS.
'''

import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

#------------------------------------------------------------
# Import MD_panel_estimation for access to the metrics class
#------------------------------------------------------------
import MD_panel_estimation as MDOLS

#------------------------------------------------------------
# Transformation class
#------------------------------------------------------------

class Transformation(MDOLS.Transformation):
    ''' Class transforms a given data vector or data matrics along a certain
    axis
    '''
    
    def dataTransformation_RE(self, data, axis):
        if self._FE.shape[1] == 2:
            data_mean = data.groupby(self._FE.iloc[:,axis]).transform('mean')
            data_trans = data - data_mean
        else:
            raise RuntimeError('Other possible Random Effects not possible yet.')
            
        return data_trans
    
    def transform_RE(self, data, axis):
        ''' Returns transformed data as df '''
        # Run inventorizeFE
        self = self.inventorizeFE()
                     
        # Call FEColumns
        self = self.FEColumns()
        
        #Run dataTransformation
        data_trans = self.dataTransformation_RE(data, axis)
               
        return (data_trans)
    
#------------------------------------------------------------
# Random effects estimation class
#------------------------------------------------------------

class MultiDimensionalRE(MDOLS.Metrics, Transformation):
    
    # Initialize the class
    def __init__(self, y, X):
        self._depvar = y.copy()
        self._data = X.copy()
        self.nobs = None
        self.nvars = None
        self.params = None
        self.std = None
        self.resid = None
        self._name = 'Code_docs.Help_functions.MD_panel_estimation'
        
        # Check data type of y and X
        if isinstance(y, pd.Series) and isinstance(X, pd.Series):
            X = pd.DataFrame(X)
            
            self._depvar = y.copy()
            self._data = X.copy()
                
        elif isinstance(y, pd.Series) and isinstance(X, pd.DataFrame):
            self._depvar = y.copy()
            self._data = X.copy()
                
        elif isinstance(y, np.ndarray) and isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1,1)
            
            self._depvar = y.copy()
            self._data = X.copy()
        
        else:
            raise TypeError('Type should be numpy.ndarray, pandas.Series, or pandas.DataFrame')
            
        # Get N and K from X
        self.nobs, self.nvars = X.shape
    
    @staticmethod
    def OLS(y,X):
        '''OLS estimation, returns parameters '''
        # Check column rank:
        if np.linalg.matrix_rank(X) != X.shape[1]:
            raise RuntimeError("X does not have full column rank")
            
        xtx = np.dot(X.T,X)
        xtx_inv = np.linalg.inv(xtx)
        xty = np.dot(X.T,y)
        params = np.dot(xtx_inv, xty)
        
        return params
    
    @staticmethod
    def residuals(y,X):
        '''Calculates OLS residuals. y is the dependent variable, X is a matrix
            of independent variables '''
        params = MultiDimensionalRE.OLS(y,X)
        resid = np.subtract(y, np.dot(X, params))
        
        return resid
    
    def fit(self, FE_cols = None, how = None):       
        
        # Prelims
        self._FE_cols = FE_cols
        self._how = how
        
        # OLS estimation model and get the residuals
        resid = MultiDimensionalRE.residuals(self._depvar, self._data)
    
        # Transform residuals
        resid_trans_a = self.transform_RE(resid, axis = 0)
        resid_trans_b = self.transform_RE(resid, axis = 1)
       
        # Calculate variance components
        sigma2_u = np.mean(resid**2)
        sigma2_alpha = sigma2_u - np.mean(resid_trans_a.groupby(self._FE.iloc[:,0]).apply(lambda x: (1 / (x.shape[0] - 1)) * np.sum(x**2)))
        sigma2_delta = sigma2_u - np.mean(resid_trans_b.groupby(self._FE.iloc[:,1]).apply(lambda x: (1 / (x.shape[0] - 1)) * np.sum(x**2)))
        sigma2_epsilon = sigma2_u - sigma2_alpha - sigma2_delta
        
        
        self._test = [resid, resid_trans_a, resid_trans_b, sigma2_u, sigma2_alpha, sigma2_delta, sigma2_epsilon]
        return self
    
        # Calculate the thetas-hat
        theta_hat12 = sigma2_epsilon / (sigma2_epsilon + self._FE.iloc[:,0].nunique() * sigma2_alpha)
        theta_hat13 = sigma2_epsilon / (sigma2_epsilon + self._FE.iloc[:,1].nunique() * sigma2_delta)
        theta_hat14 = sigma2_epsilon / (sigma2_epsilon + self._FE.iloc[:,0].nunique() * sigma2_alpha + self._FE.iloc[:,1].nunique() * sigma2_delta)
        
        # Transfrom the data
        ## Calculate the scalars
        sqrt_theta_hat12 = (1 - np.sqrt(theta_hat12))
        sqrt_theta_hat13 = (1 - np.sqrt(theta_hat13))
        sqrt_theta_hat14 = (1 - np.sqrt(theta_hat12) - np.sqrt(theta_hat13) - np.sqrt(theta_hat14))
       
        ## Calculate transformed data
        ### dependent variable
        y_demeaned_ax0 = self.transform_RE(self._depvar, axis = 0) 
        y_demeaned_ax1 = self.transform_RE(self._depvar, axis = 1)
        y_mean = self._depvar.mean()
       
        ### regressores
        X_demeaned_ax0 = self.transform_RE(self._data, axis = 0) 
        X_demeaned_ax1 = self.transform_RE(self._data, axis = 1)
        X_mean = self._data.mean()       
       
        ### y_tilde and X_tilde
        y_tilde = self._depvar - sqrt_theta_hat12 * y_demeaned_ax0 - sqrt_theta_hat13 * y_demeaned_ax1 - sqrt_theta_hat14 * y_mean
        X_tilde = self._data - sqrt_theta_hat12 * X_demeaned_ax0 - sqrt_theta_hat13 * X_demeaned_ax1 - sqrt_theta_hat14 * X_mean
       
        # Estimated transformed model with OLS (FGLS)
        self.params = MultiDimensionalRE.OLS(y_tilde,X_tilde)
       
        #Calculate residuals
        self.resid = MultiDimensionalRE.residuals(y_tilde, X_tilde)
       
        # Calculate standard deviation
        self.nonRobustSTD()
       
        # Add metrics
        self.fittedvalues = self.predict(X_tilde)
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
        
        return self