# Estimation results

''' This script tests whether whether remote lenders sell a higher fraction of 
    their residential loans
    '''

#------------------------------------------------------------
# Import necessary packages
#------------------------------------------------------------

import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import numpy as np
import pandas as pd
from scipy import stats

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

df = pd.read_csv('Data/data_agg_clean.csv')