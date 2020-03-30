# Set working directory
import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import numpy as np
import pandas as pd
import multiprocessing as mp # For parallelization
from joblib import Parallel, delayed # For parallelization

#---------------------------------------
# Load Data
file_lf = r'Data/hmdpanel17.dta'

## Load df LF
df_lf = pd.read_stata(file_lf)

#---------------------------------------
# Save df
df_lf.to_csv(r'Data/hmdpanel17.csv', index = False)