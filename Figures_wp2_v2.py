# Figures
''' This script makes a variety of figures for working paper #2
    ''' 
    
import pandas as pd
import dask.dataframe as dd
import numpy as np    
    
import os
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

from urllib.request import urlopen
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', font_scale = 2.5, palette = 'Greys_d')

#------------------------------------------------------------
# Load the df
#------------------------------------------------------------

#df_main
columns = ['cert','msamd','fips','date','min_distance','log_min_distance']
dd_main = dd.read_parquet(path = 'Data/data_main_clean.parquet',\
                       engine = 'fastparquet',\
                       columns = columns)

#------------------------------------------------------------
# Geo Choropleth
#------------------------------------------------------------

# Define function
def choroplethUS(df, counties, loc, color, title, text, path):
    global df_main
    
    if color == 'log_min_distance' and df.name == 'difference':
        range_color = (df.log_min_distance.min(), df.log_min_distance.max())
        color_continuous_scale = 'balance'
    elif color == 'log_min_distance':
        range_color = (0, df_main.log_min_distance.max())
        color_continuous_scale = 'Inferno'
    elif color == 'log_density':
        range_color = (0, df_main.log_density.max())
        color_continuous_scale = 'Inferno'
    elif color == 'ls' or color == 'perc_intsub':
        range_color = (0,1)
        color_continuous_scale = 'Inferno'
    else:
        range_color = None
        color_continuous_scale = 'Inferno'
        
    # Make the figure
    fig = px.choropleth(df, geojson = counties, locations = loc, color = color,
                            color_continuous_scale = color_continuous_scale,
                            range_color = range_color,
                            scope = "usa")
    
    # Update the layout
    fig.update_layout(margin = {"r":0,"t":0,"l":0,"b":0}, width = 1500, height = 800,
                      font = {'family':'Times New Roman','size':30},
                    coloraxis_colorbar = dict(
                    x = 0.9,
                    title = dict(text = title,
                             side = 'right')),
                  )
                    
    # Save static image
    pio.write_image(fig, path)
    
#------------------------------------------------------------
# difference 2004 -- 2019 

'''NOTE: Over the entire dataset the mean lending distance decreases, as is the mean
    distance for most counties. We see an increase only in smaller MSAs'''

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Setup data
df_2004 = dd_main[dd_main.date == 2004].compute(scheduler = 'threads')
df_2004.fips = df_2004.fips.astype(int).astype(str).str.zfill(5)
df_2004_mean = df_2004.groupby('fips').mean()

df_2019 = dd_main[dd_main.date == 2019].compute(scheduler = 'threads')
df_2019.fips = df_2019.fips.astype(int).astype(str).str.zfill(5)
df_2019_mean = df_2019.groupby('fips').mean()

df_20192004 = (df_2019_mean - df_2004_mean).dropna(subset = ['min_distance', 'log_min_distance']).reset_index()
df_20192004.name = 'difference'

    
# Make figure
## Prelims
loc = 'fips'
color_distance = 'log_min_distance'
title_distance = 'Log distance (in km)'
text_distance = 'Difference in mean residential lending distance from U.S. Banks and Thrifts (2004-2019)'
path_distance = 'Figures/Difference_mean_lendingdistance_20192004.png'  

## Plot
choroplethUS(df_20192004, counties = counties, loc = loc, color = color_distance, title = title_distance, text = text_distance, path = path_distance)

#------------------------------------------------------------
# difference 2010 -- 2019 
'''NOTE: In the past decade we observe a small increase in the mean lending distance.
     In a far greater number of MSAs we observe an increase, especially on the east and
     west coast, Florida, and the coast of the Mexican golf'''

# Setup data
df_2010 = dd_main[dd_main.date == 2010].compute(scheduler = 'threads')
df_2010.fips = df_2010.fips.astype(int).astype(str).str.zfill(5)
df_2010_mean = df_2010.groupby('fips').mean()

df_20192010 = (df_2019_mean - df_2010_mean).dropna(subset = ['min_distance', 'log_min_distance']).reset_index()
df_20192010.name = 'difference'

    
# Make figure
## Prelims
text_distance = 'Difference in mean residential lending distance from U.S. Banks and Thrifts (2010-2019)'
path_distance = 'Figures/Difference_mean_lendingdistance_20192010.png'  

## Plot
choroplethUS(df_20192010, counties = counties, loc = loc, color = color_distance, title = title_distance, text = text_distance, path = path_distance)



