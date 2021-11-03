# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python [conda env:q5]
#     language: python
#     name: conda-env-q5-py
# ---

# %% [markdown] pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
# - results are shown here
#   - [z010_cluster_analysis.ipynb](./nb/z010_cluster_analysis.md)
#   - we try maultple cluster groups [2, 3, 5, 9, 15, 20]
#     - the number of cluster groups is always a topic of debate 
#     - it depends on the complexity and specificity of the intended analysis 
#     - also simplicity is important as simple results are easy to understand 
#     - i suggest 5 clusters as a good compromise.  
#
# region cluster analysis for the mosaic campaign 
# - loosely based on the method described in
#   - https://acp.copernicus.org/preprints/acp-2021-126/
# - data and flexpart analysis obtained from 
#   - https://srvx1.img.univie.ac.at/webdata/mosaic/mosaic.html
# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2

# import matplotlib as mpl
# import matplotlib.colors
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
import xarray as xr

import funs as fu
import os 

# %% md [markdown]
#
# ## constants and functions

# %% jupyter={"outputs_hidden": false}

PATH_TO_FILE = '../data_in/flexpart-mosaic-data-alias/AIRTRACER_100m.nc'

LA = 'lat'
LO = 'lon'
TI = 'time'
AG = 'age'
AT = 'AIRTRACER'
L = 'lab'
CM = 'tab20'

DATA_OUT = '../data_out'


# %% md [markdown]
#
# # code
# %%


# def main():
# %%

ds = xr.open_mfdataset(PATH_TO_FILE)

# %%
# lets check lat and lon borders
for l in LO, LA:
    a = ds[l].diff(l).to_series().describe()
    print(f'{l}\n{a}\n\n')

# %% [markdown]
# ## overall residence time
# Lets start by plotting the the sum over time of the flexpart output

# %%

# sum over time and age and load the file in memory
d1 = ds.sum([TI, AG])[AT]
d1.load()
# plot a sum over the campaign
fu.plot_map(d1)

# %% [markdown]
# ## the ship's path
# Lets identify the path taken by the boat. We can euristically determine this 
# by using the maximum residence time of the flexpart output at each time step.

# %%
# find the ship path on a euristic basis
d = ds[AT][{AG: 0}].load()
am = d.argmax(dim=[LO, LA])
df = d[am].to_dataframe()
fu.plot_path(df, LO, LA)

# %% [markdown]
# ## some distributions
# Lets also explore the distributions of single 'pixels'.
# They seem to be log distributions. 

# %%
d2 = ds[AT].sum(AG).load()
fu.plot_some_distributions(d2, LA, LO)

# %%

# %%

# plot the residence times
fu.plot_residence_time(d2, LA, LO)

# %%
# coarse the array to reduce number of data points and also transform the
# residence from seconds to days
d3 = d2.coarsen({LA: 5, LO: 10}, boundary='exact').sum() / 3600 / 24
# plot the coarse array
fu.plot_map(d3.sum(TI))

# %%
qta = fu.get_quantiles(d3, TI)

# %% [markdown]
# # plot and save cluster from [2, 3, 5, 6, 9, 15, 20]

# %%

for N in [2, 3, 5, 6, 9, 15, 20]:
# for N in [2]:
    d4 = fu.kmeans_cluster(N, qta, d3, L, TI, )
    fu.save_cluster_csv(d4, N, DATA_OUT, L)
    lax = d4[L]
#     fu.plot_kmap(lax, N, CM)
    fu.plot_hatch(d4, L, N, d1, LA, LO, CM)
    fu.plot_cluster_bar(N, d4, L, CM)
    fu.plot_month_means(d4, L, CM, N)
    fu.plot_cluster_ts(N, d4, L, CM)
# %%
