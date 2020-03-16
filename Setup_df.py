# Setup the df 
''' This script makes the final df (before data cleaning). 
    
    Final df: MSA-bank/thrift-level dataset
'''

#------------------------------------------------------------
# Load packages and set working directory
#------------------------------------------------------------

# Set working directory
import os
os.chdir(r'X:/My Documents/PhD/Materials_papers/2-Working_paper_competition')

# Load packages
import pandas as pd
import numpy as np

#------------------------------------------------------------
# Setup necessary functions
#------------------------------------------------------------

# Minimum distance

def minDistanceLenderBorrower(hmda_cert,hmda_fips,msa_fips,sod_stcntybr,sod_cert,distances):
    ''' This methods calculates the minimum distance between a lender (a specific
        branche of a bank or thrift) and a borrower (a unknown individual) based on
        the respective fips codes. Uses the haversine method to calculate distances.
    
    Parameters
    ----------
    hmda : pandas DataFrame 
        one row and n columns
    sod : pandas DataFrame
    distances : pandas DataFrame

    Returns
    -------
    Float

    '''

    # Make a subset of branches where the FDIC certificates in both datasets match  
    branches = sod_stcntybr[sod_cert == hmda_cert]
    
    
    # Get the minimum distance
    if not hmda_cert in sod_cert:
        return (np.nan)
    elif hmda_fips in branches:
        return (0.0)
    else:
        output = np.min(distances[(distances.fips_1 == hmda_fips) & (distances.fips_2.isin(branches))].distance)

        return(output)
