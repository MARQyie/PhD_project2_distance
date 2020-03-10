# Import packages
import pandas as pd
import numpy as np

import os
os.chdir(r'C:\Users\p285325\Desktop')

import dask.dataframe as dd


fix_width_lar = [4,10,1,1,1,1,5,1,4,2,3,7,1,1,1,1,4,1,1,1,1,1,7]
header_names_lar = ['year','id','agency','type_loan','purchase_loan','occ','amount','action',\
                'property','state','county','census','race_app','race_coapp',\
                'sex_app','sex_coapp','inc_app','purchaser','denial1','denial2',\
                'denial3','edit_status','seq_num']
df = pd.read_fwf('HMS.U1994.LARS', widths = fix_width_lar, names = header_names_lar)

#test
%time dfpd = pd.read_fwf('HMS.U1994.LARS', widths = fix_width_lar, nrows= 10000, names = header_names_lar)

%time dfpd = pd.read_fwf('HMS.U1994.LARS', widths = fix_width_lar, nrows= 100000, names = header_names_lar, engine = 'c')
%time dfdd = dd.read_fwf('HMS.U1994.LARS', widths = fix_width_lar, names = header_names)

%time dfpd = pd.read_fwf('HMS.U1994.LARS', widths = fix_width_lar, names = header_names_lar, chunksize = 1e6)

chunk_list = []  # append each chunk df here 

for chunk in dfpd:  
    # perform data filtering 
    chunk_filter = pd.DataFrame(chunk)
    
    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk_filter)
    
# concat the list into dataframe 
df_concat = pd.concat(chunk_list)

## TS
fix_width_tts = [4,1,10,30,40,25,2,10,30,40,25,2,10,1,10]
header_names_tts = ['year','agency','id','resp_name','resp_addr','resp_city','resp_state',\
                    'resp_zip','par_name','par_addr','par_city','par_state',\
                    'par_zip','edit_status','tax_id']
dftts = pd.read_fwf('HMS.U1994.TTS', widths = fix_width_tts, names = header_names_tts)

#NOTES: Chucksize werkt erg goed. Dask werkt ook, maar de computaties zijn langzaam.