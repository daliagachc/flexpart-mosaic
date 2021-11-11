# ---
# jupyter:
#   jupytext:
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

# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import regression_funs as rfu

# %% md [markdown]
#
# ## constants and functions

# %% jupyter={"outputs_hidden": false}

PATH_TO_FILE = '../data_in/flexpart-mosaic-data_alias/AIRTRACER_100m.nc'

PATH_TO_SO2 = '../data_in/flexpart-mosaic-data_alias/MSAQSO2L4_2005-2018_v01-04_20190314.h5'

PATH_TO_ST = '../data_in/ciapitof_masked_filtered.csv'

PATH_200_CLUS = '../data_out/cluster200.nc'

LA = 'lat'
LO = 'lon'
TI = 'time'
AG = 'age'
AT = 'AIRTRACER'
L = 'lab'
CM = 'tab20'

SA = 'sa'
MSA = 'msa'
IA = 'ia'

LSA = 'log10(sa)'
LMSA = 'log10(msa)'
LIA = 'log10(ia)'

COLS = [SA, MSA, IA]
LCOLS = [LSA, LMSA, LIA]

DATA_OUT = '../data_out'

COL = plt.get_cmap('Dark2')

DC = {
    SA  : COL(0),
    MSA : COL(1),
    IA  : COL(2),
    LSA : COL(0),
    LMSA: COL(1),
    LIA : COL(2),
}

# %%
df = pd.read_csv(PATH_TO_ST, index_col=0, parse_dates=True)

for l, c in zip(COLS, LCOLS):
    df[c] = np.log10(df[l])

# %% [markdown]
# # distributions
#
#
# lets find what kind of dist. do we have. They seem to be log dists.

# %%
for c in COLS:
    df[c].plot.hist(color=DC[c])

# %%
for c in COLS:
    np.log10(df[c]).plot.hist(alpha=.5, bins=100, label=c, color=DC[c])
ax = plt.gca()
ax.legend()
ax.set_xlabel('log10')

# %% [markdown]
# # timeseries of the trace gases

# %%

# %% [markdown]
# # open and merge flex 200 clusters

# %%
ds = xr.open_dataset(PATH_200_CLUS)

# %%
dsf = ds.groupby(L).sum().to_dataframe()[AT].unstack(L)

q0, q1 = np.quantile(dsf.values.flatten(), [.01, .99])



# f, axs = plt.subplots(10, 20, sharex=True, sharey=True, figsize=(20, 20))
# axf = axs.flatten()

# for i, ax in zip(dsf.columns, axf):
#     ax.hist(np.log10(dsf[i] + q0), bins=np.linspace(np.log10(q0), np.log10(q1 + q0), 20))
#     ax.set_xlabel('')
#     ax.set_ylabel('')

# %%
dsfn = dsf / dsf.mean()

# %%
q0, q1 = dsfn.stack().quantile([.01, .99])

# %%
# f, axs = plt.subplots(10, 20, sharex=True, sharey=True, figsize=(20, 20))
# axf = axs.flatten()

# for i, ax in zip(dsfn.columns, axf):
#     ax.hist(dsfn[i], bins=np.linspace(q0, q1, 20))
#     ax.set_xlabel('')
#     ax.set_ylabel('')

# %%
df1 = df

# %%
df2 = df1.resample('3H').median()

# %%
dm = pd.merge(df2, dsf, left_index=True, right_index=True, how='inner', validate="1:1")


# %%


# %%

# %% [markdown]
# # Invers modeling elastic NET

# %%
# for PAR in [SA,MSA,IA]:
for PAR in [SA]:
    pred, cdf, y, yn, dp = rfu.elastic_net_reg(dsf, dm, PAR)
    MEA = f'measured {PAR}'
    MOD = f'inverse modeled {PAR}'
#     rfu.scatter_plot(y, yn, pred, MEA, MOD, PAR)
#     rfu.mea_vs_mod_plot(y, yn, dp, pred, MEA, MOD)
#     rfu.rank_cluster_inf(cdf)
    dd2 = rfu.get_plot_inf(cdf, L, ds,PAR)
#     rfu.plot_map(dd2)
#     rfu.plot_map_rect(dd2)

# %%
cdf.sort_values(ascending=False)

# %%
import funs as fu

# %%
d5 = fu.get_bounds(LA, ds[AT])
d6 = fu.get_bounds(LO, d5)

df = d6[L].to_dataframe()

df1 = df.loc[:, ~df.columns.duplicated()]

G = 'geometry'

# %%
df1[G] = df.apply(fu.get_pol, axis=1)

# %%
import geopandas
dg = geopandas.GeoDataFrame(df1).reset_index()

dg1 = dg[[L, G]].dissolve(by=L)

# %%
cdf.name = 'infl'

# %%
df2 = pd.merge(dg1,cdf,left_index=True,right_index=True)

# %%
df3 = df2.sort_values('infl',ascending=False).iloc[:20]

# %%
df3['infl']

# %%
dm1 = dm[df3.index]

# %%
dm2 = dm1/dm1.sum()
dm2 = (dm1 * df3['infl'])/100000

# %%

# %%
cmap = plt.get_cmap('tab20')

# %%
co = dm2.columns

# %%
q1,q2=dm2.stack().quantile([.001,.999])

# %%

# %%
lco = len(co)

f = plt.figure(constrained_layout=True,figsize=(2*7,lco*.6),dpi=200)

import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(lco,2)




# f,axs=plt.subplots(lco,dpi=200,sharex=True,sharey=True,figsize=(7,lco*.6))


ax0 = None
for i,c in enumerate(co):
    ax = f.add_subplot(gs[i, 1],sharex=ax0)
    if i == 0: ax0 = ax
    (dm2[c]).plot(c=cmap(i),ax=ax)
    ax.text(1,.5,f' {i}',transform=ax.transAxes,c=cmap(i))
    ax.grid()
    ax.set_ylim(0,q2*1.1)
# ax=plt.gca()
# ax.set_yscale('log')
# ax.set_ylim(q1,q2)
# ax.legend(bbox_to_anchor=(1,1))


import cartopy.crs as ccrs
ax = f.add_subplot(gs[:,0],projection=ccrs.Orthographic(0, 90), facecolor="gray")
rfu.plot_map(dd2,ax=ax)
# ax=plt.gca()
for i,c in enumerate(co):
    p =df3.centroid[c]
    ax.scatter( p.x,p.y,transform=ccrs.PlateCarree(),c='w',s=105,alpha=1,zorder=19)
    ax.text( p.x,p.y,i,transform=ccrs.PlateCarree(),c=cmap(i),va='center',ha='center',fontsize=7,zorder = 20)



# %%



# %%

# %%

# %%

# %%
#convert to mrakdown
# !jupyter-nbconvert --to markdown z030_inverse_modeling_time_series.ipynb

# %%
