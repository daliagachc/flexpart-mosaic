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
# lets cluster 100 so that it is easy to apply the elastic net 
# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# %%
d2 = ds[AT].sum(AG).load()


# %%

# %%
d2.shape

# %%
# coarse the array to reduce number of data points and also transform the
# residence from seconds to days
d3 = d2.coarsen({LA: 5, LO: 10}, boundary='exact').sum() / 3600 / 24


# %%
qta = fu.get_quantiles(d3, TI)

# %%
NN = 1000
fu.LETTERS = np.arange(NN)
for N in [NN]:
# for N in [2]:
    d4 = fu.kmeans_cluster(N, qta, d3, L, TI, )
#     fu.save_cluster_csv(d4, N, DATA_OUT, L)
#     lax = d4[L]

#     fu.plot_hatch(d4, L, N, d1, LA, LO, CM)
#     fu.plot_cluster_bar(N, d4, L, CM)
#     fu.plot_month_means(d4, L, CM, N)
#     fu.plot_cluster_ts(N, d4, L, CM)
# %%


import cartopy.crs as ccrs

# air = xr.tutorial.open_dataset("air_temperature").air

dic = dict(projection=ccrs.Orthographic(0, 90), facecolor="gray")

f, ax = plt.subplots(subplot_kw=dic, )

p = d4[L].plot(
    # subplot_kws=dic,
    transform=ccrs.PlateCarree(),
#     norm=mpl.colors.LogNorm(vmin, vmax),
    robust=True,
    cmap='Reds_r'
)
#   p.axes.set_global()
p.axes.coastlines()

# %%
d4.groupby(L).sum().plot(norm=mpl.colors.LogNorm(),robust=True)

# %%
d4.to_netcdf(f'../data_out/cluster{NN}.nc')

# %%
d4

# %%

# %%

# %%

# %%
