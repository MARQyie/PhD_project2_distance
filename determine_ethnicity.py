def determineEthnicity(dataframe):
    '''This method defines the ethnicity of the borrower (see Avery et al 2007)
       The hierachy is as follows (if either app or co-app has declared):
           Black, Hispanic, American Indian, Hawaiian/Pacific
           Islander, Asian, white Non-Hispanic'''
           
    import pandas as pd
    import numpy as np
    
    #Prelims
    list_race = ['applicant_race_1', 'applicant_race_2', 'applicant_race_3', 'applicant_race_4', 'applicant_race_5',
                 'co_applicant_race_1', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_4', 'co_applicant_race_5']
    list_eth = ['applicant_ethnicity', 'co_applicant_ethnicity']
    vector = pd.DataFrame(index = dataframe.index, columns = ['ethnicity_borrower'])
    
    # Setup the boolean vectors
    black = dataframe.loc[:,list_race].isin([3]).any(axis = 1)
    hispanic = dataframe.loc[:,list_eth].isin([1]).any(axis = 1)
    amer_ind = dataframe.loc[:,list_race].isin([1]).any(axis = 1)
    hawaiian = dataframe.loc[:,list_race].isin([4]).any(axis = 1)
    asian = dataframe.loc[:,list_race].isin([2]).any(axis = 1)
    white_nh = dataframe.loc[:,list_race].isin([5]).any(axis = 1) & dataframe.loc[:,list_eth].isin([2]).any(axis = 1)
    
    # Fill the vector
    vector[black] = 1
    vector[~black & hispanic] = 2
    vector[~black & ~hispanic & amer_ind] = 3
    vector[~black & ~hispanic & ~amer_ind & hawaiian] = 4
    vector[~black & ~hispanic & ~amer_ind & ~hawaiian & asian] = 5
    vector[~black & ~hispanic & ~amer_ind & ~hawaiian & ~asian & white_nh] = 0
    vector[~black & ~hispanic & ~amer_ind & ~hawaiian & ~asian & ~white_nh] = np.nan
    
    return(np.array(vector))   