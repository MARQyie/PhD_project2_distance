# Geoplots
''' This script makes various geoplots with aggregate HMDA data
    '''

#------------------------------------------------------------
# Load packages
#------------------------------------------------------------
import pandas as pd
import dask.dataframe as dd
import numpy as np    
    
import os
#os.chdir(r'/data/p285325/WP2_distance/')
os.chdir(r'D:/RUG/PhD/Materials_papers/2-Working_paper_competition')

import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go

from urllib.request import urlopen
import json

#------------------------------------------------------------
# Load the dfs
#------------------------------------------------------------

#df_popcenter
df_popcenter = pd.read_csv('Data/data_msa_popcenter.csv', index_col = 0, \
                     dtype = {'fips':'str','cbsa_code':'str'}) 


# df_main (loaded with dask)
df_main = pd.read_csv('Data/data_main.csv')
#df_main.dropna(inplace = True)

#------------------------------------------------------------
# Plot distance chloropleths for 2010 and 2017 (works with HMDA-level data)
#------------------------------------------------------------

# Define function
def choroplethUS(df, counties, loc, color, title, text, path):
    # Make the figure
    fig = px.choropleth(df, geojson = counties, locations = loc, color = color,
                            color_continuous_scale="inferno",
                            scope="usa")
    
    # Update the layout
    fig.update_layout(margin = {"r":0,"t":0,"l":0,"b":0}, width = 1500, height = 900,
                    coloraxis_colorbar = dict(
                    title = title),
                    title = go.layout.Title(
                    text = text,
                    xref = 'paper',
                    x=0.1,
                    y=0.95)
                  )
    # Save static image
    pio.write_image(fig, path)
    
#------------------------------------------------------------
# 2010

# Setup data
df_main_2010 = df_main[df_main.date == 2010][['fips', 'min_distance', 'log_min_distance']].compute()
df_main_2010.fips = df_main_2010.fips.astype(int).astype(str).str.zfill(5)

# Setup prelims
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
loc = 'fips'
color = 'log_min_distance'
title = 'Mean lending distance'
text = 'Mean residential lending distance from U.S. Banks and Thrifts (2010)'
path = 'Figures/Mean_lendingdistance_2010.png'  
    
# Make figure
choroplethUS(df_main_2010, counties = counties, loc = loc, color = color, title = title, text = text, path = path)

#------------------------------------------------------------
# 2017

# Setup data
df_main_2017 = df_main[df_main.date == 2017][['fips', 'min_distance', 'log_min_distance']].compute()
df_main_2017.fips = df_main_2017.fips.astype(int).astype(str).str.zfill(5)

# Setup prelims
loc = 'fips'
color = 'log_min_distance'
title = 'Mean lending distance'
text = 'Mean residential lending distance from U.S. Banks and Thrifts (2017)'
path = 'Figures/Mean_lendingdistance_2017.png'  
    
# Make figure
choroplethUS(df_main_2017, counties = counties, loc = loc, color = color, title = title, text = text, path = path)

#------------------------------------------------------------
# Plot population-weighted 
#------------------------------------------------------------

# Define function
def plotPopCenter(df, title, path):
    colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
    fig = go.Figure()
    
    # Make the Figure
    data = df_popcenter[df_popcenter.stname == 'New York']
       
    fig.add_trace(go.Scattergeo(locationmode = 'USA-states',
            lon = df['longitude'],
            lat = df['latitude'],
            text = df['couname'],
            marker = dict(
                    size = np.log(df['population']),
                    line_color='rgb(40,40,40)',
                    line_width=0.5,
                    sizemode = 'area'  
                    )
            ))
    
    # Update Layout
    fig.update_layout(
            title = title,
            geo = go.layout.Geo(
                    scope = 'usa',
                lonaxis_range= [ -150.0, -200.0 ],
                lataxis_range= [ 0.0, 0.0 ])
        )

    # Save static image
    pio.write_image(fig, path)

#------------------------------------------------------------
# Plot the data
    
# Prelim
title = 'Population-weighted U.S. county centroids'
path = 'Figures/pop_county_centroids.png'

# Plot
plotPopCenter(df_popcenter, title, path)







''' OLD
# Prelims
fips_count = pd.DataFrame(df_sod.groupby('STCNTYBR').STCNTYBR.count())
fips_count = fips_count.rename(columns = {'STCNTYBR':'count'}).reset_index()
fips_count['ln_count'] = np.log(fips_count['count'])

## figure prelims

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Make the image    
fig = px.choropleth(fips_count, geojson=counties, locations='STCNTYBR', color='ln_count',
                            color_continuous_scale="inferno",
                            range_color=(0, fips_count.iloc[:,2].max()),
                            scope="usa")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width = 1500, height = 900,
                    coloraxis_colorbar=dict(
                    title="Num. Branches",
                    tickvals=[0,np.log(np.floor(fips_count.iloc[:,1].max() / 1e3) * 1e3)],
                    ticktext=['0','{}k'.format(np.floor(fips_count.iloc[:,1].max() / 1e3).astype(str)[:-2])]),
                    title=go.layout.Title(
                    text='Number Bank and Thrift Branches per County (2010)',
                    xref='paper',
                    x=0.1,
                    y=0.95)
                  )
pio.write_image(fig, 'Figures/Number_branches_map_2010.png')
'''

