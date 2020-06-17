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
# Load the dfs
#------------------------------------------------------------

#df_agg_clean
df_agg = pd.read_csv('Data/data_agg_clean.csv')
#df_agg.dropna(inplace = True)

#df_main
columns = ['cert','msamd','fips','date','log_min_distance','density','ls', 'perc_intsub']
df_main = pd.read_csv('Data/data_main.csv', usecols = columns)

## Add normal distance
df_main['min_distance'] = np.exp(df_main.log_min_distance - 1)
df_main['log_density'] = np.log(df_main.density)

#------------------------------------------------------------
# Plot largest 3 and smallest 3 msa through time
#------------------------------------------------------------

#------------------------------------------------------------
# Distance

# Make new df
df_msadate = df_agg.groupby(['msamd','date']).mean().reset_index()

## Balance the df
list_msa = df_msadate.msamd.value_counts()[df_msadate.msamd.value_counts() == 8].index.to_numpy()
df_msadate = df_msadate[df_msadate.msamd.isin(list_msa)]

# Get top and bottom MSAs
top = df_msadate.sort_values(by = 'ln_pop_area', ascending = False).msamd.unique()[:3]
bottom = df_msadate.sort_values(by = 'ln_pop_area', ascending = True).msamd.unique()[:3]

# Make new df for plot
df_largest_smallest = df_msadate[df_msadate.msamd.isin(np.append(top, bottom))].loc[:,['msamd','date','log_min_distance']]

## Make date a datetime
df_largest_smallest.date =  pd.to_datetime(df_largest_smallest.date.astype(str) + '-01-01')

#------------------------------------------------------------
# Plot

# Prelims
linestyles = ['-','-','-',':',':',':']
markers = ['o','s','^','o','s','^']
colors = ['darkblue','darkblue','darkblue','crimson','crimson','crimson']

labels = ['Philadelphia, PA','San Francisco-San Mateo-Redwood City, CA','Chicago-Naperville-Joliet, IL',\
         'Flagstaff, AZ','Fairbanks, AK','Casper, WY']
msa_strings = [str(int(x)) for x in np.append(top, bottom)]

# Plot
fig, ax = plt.subplots(figsize=(24, 16))
ax.set(xlabel='Year', ylabel = 'Log Mean Lending Distance')
for msa, linestyle, marker, color, label in zip(np.append(top, bottom), linestyles, markers, colors, labels):
    df_plot = df_largest_smallest[df_largest_smallest.msamd == msa]
    ax.plot(df_plot.date, df_plot.log_min_distance, linestyle = linestyle, marker = marker, color = color, label = label, linewidth = 3, markersize = 20)   
ax.legend()
plt.tight_layout()
plt.show()

fig.savefig('Figures\Fig_meandistance_over_time.png')

# calculate average distance in 2010 and 2017 resp.
mean_all_2010 = np.exp(df_largest_smallest[df_largest_smallest.date == pd.Timestamp(2010,1,1)].log_min_distance.mean() - 1)
mean_all_2017 = np.exp(df_largest_smallest[df_largest_smallest.date == pd.Timestamp(2017,1,1)].log_min_distance.mean() - 1)

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
# 2010
    
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Setup data
df_main_2010 = df_main[df_main.date == 2010][['fips', 'min_distance', 'log_min_distance','density', 'log_density','ls']].dropna()
df_main_2010.fips = df_main_2010.fips.astype(int).astype(str).str.zfill(5)
df_2010 = df_main_2010.groupby('fips').mean().reset_index()
df_2010.name = np.nan

# Setup prelims
loc = 'fips'
color_distance = 'log_min_distance'
color_density = 'log_density'
color_ls = 'ls'

title_distance = 'Log distance (in km)'
title_density = 'Log number of branches'
title_ls = 'Percentage'

text_distance = 'Mean residential lending distance from U.S. Banks and Thrifts (2010)'
text_density = 'Number of Branches U.S. Banks and Thrifts (2010)'
text_ls = 'Percentage of loan sold in the U.S. (2010)'

path_distance = 'Figures/Mean_lendingdistance_2010.png'  
path_density = 'Figures/Mean_density_2010.png'
path_ls = 'Figures/Mean_ls_2010.png'
    
# Make figures
choroplethUS(df_2010, counties = counties, loc = loc, color = color_distance, title = title_distance, text = text_distance, path = path_distance)
choroplethUS(df_2010, counties = counties, loc = loc, color = color_density, title = title_density, text = path_density, path = path_density)
choroplethUS(df_2010, counties = counties, loc = loc, color = color_ls, title = title_ls, text = text_ls, path = path_ls)

#------------------------------------------------------------
# 2013

# Setup data
df_main_2013 = df_main[df_main.date == 2013][['fips', 'min_distance', 'log_min_distance', 'density', 'log_density', 'ls', 'perc_intsub']].dropna()
df_main_2013.fips = df_main_2013.fips.astype(int).astype(str).str.zfill(5)
df_2013 = df_main_2013.groupby('fips').mean().reset_index()
df_2013.name = np.nan

# Setup prelims
color_intsub = 'perc_intsub'
title_intsub = 'Percentage'
text_intsub = 'Percentage of internet subscriptions in the U.S. (2013)'
path_intsub = 'Figures/Mean_intsub_2013.png'

# Make figure
choroplethUS(df_2013, counties = counties, loc = loc, color = color_intsub, title = title_intsub, text = text_intsub, path = path_intsub)

#------------------------------------------------------------
# 2017

# Setup data
df_main_2017 = df_main[df_main.date == 2017][['fips', 'min_distance', 'log_min_distance','density', 'log_density', 'ls','perc_intsub']].dropna(subset = ['min_distance', 'log_min_distance','density', 'log_density', 'ls'])
df_main_2017.fips = df_main_2017.fips.astype(int).astype(str).str.zfill(5)
df_2017 = df_main_2017.groupby('fips').mean().reset_index()
df_2017.name = np.nan

# Setup prelims
text_distance = 'Mean residential lending distance from U.S. Banks and Thrifts (2017)'
text_density = 'Number of Branches U.S. Banks and Thrifts (2017)'
text_ls = 'Percentage of loan sold in the U.S. (2017)'
text_intsub = 'Percentage of internet subscriptions in the U.S. (2017)'

path_distance = 'Figures/Mean_lendingdistance_2017.png'  
path_density = 'Figures/Mean_density_2017.png' 
path_ls = 'Figures/Mean_ls_2017.png'
path_intsub = 'Figures/Mean_intsub_2017.png'
    
# Make figures
choroplethUS(df_2017, counties = counties, loc = loc, color = color_distance, title = title_distance, text = text_distance, path = path_distance)
choroplethUS(df_2017, counties = counties, loc = loc, color = color_density, title = title_density, text = text_density, path = path_density)
choroplethUS(df_2017, counties = counties, loc = loc, color = color_ls, title = title_ls, text = text_ls, path = path_ls)
choroplethUS(df_2017, counties = counties, loc = loc, color = color_intsub, title = title_intsub, text = text_intsub, path = path_intsub)

#------------------------------------------------------------
# difference 2010 -- 2017 DISTANCE ONLY

# Setup data
df_20172010 = (df_2017.set_index('fips') - df_2010.set_index('fips')).dropna(subset = ['min_distance', 'log_min_distance', 'density', 'log_density', 'ls']).reset_index()
df_20172010.name = 'difference'

# Setup prelims
text_distance = 'Difference in mean residential lending distance from U.S. Banks and Thrifts (2010-2017)'

path_distance = 'Figures/Difference_mean_lendingdistance_2017.png'  
    
# Make figures
choroplethUS(df_20172010, counties = counties, loc = loc, color = color_distance, title = title_distance, text = text_distance, path = path_distance)

## Make list of the counties/MSA with a decrease in lending distance and compare these with other MSAs
df_agg_20172010 = (df_agg[df_agg.date == 2017].set_index(['cert','msamd']).log_min_distance - df_agg[df_agg.date == 2010].set_index(['cert','msamd']).log_min_distance).dropna()
msa_negative = np.array(df_agg_20172010[df_agg_20172010 < 0].index.get_level_values(1).unique())

df_agg_negative = df_agg.loc[df_agg.msamd.isin(msa_negative),['cert','msamd','date','lti', 'ln_loanamout',\
                'ln_appincome', 'density', 'pop_area', 'cb', 'ln_ta', 'ln_emp',\
                'num_branch', 'ln_pop_area', 'ln_num_branch', 'ln_density','perc_broadband']].set_index(['cert','msamd','date'])
df_agg_positive = df_agg.loc[~df_agg.msamd.isin(msa_negative),['cert','msamd','date','lti', 'ln_loanamout',\
                'ln_appincome', 'density', 'pop_area', 'cb', 'ln_ta', 'ln_emp',\
                'num_branch', 'ln_pop_area', 'ln_num_branch', 'ln_density','perc_broadband']].set_index(['cert','msamd','date'])
df_agg_difference = df_agg_positive.mean() - df_agg_negative.mean() ## Population density looks much greater for the negative sample: TEST

### Welch's t-test for pop_area
from scipy.stats import ttest_ind
stat, pval = ttest_ind(df_agg_positive.ln_pop_area, \
                           df_agg_negative.ln_pop_area, \
                           equal_var = False, nan_policy = 'omit') #Statistically significantly different

### Welch's t-test for num_branches
stat, pval = ttest_ind(df_agg_positive.pop_area, \
                           df_agg_negative.pop_area, \
                           equal_var = False, nan_policy = 'omit') #Statistically significantly different

## Percentage change
perc_change = ((df_2017.set_index('fips').log_min_distance - df_2010.set_index('fips').log_min_distance) / df_2010.set_index('fips').log_min_distance).mean()

