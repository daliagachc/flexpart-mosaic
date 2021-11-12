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
    rfu.scatter_plot(y, yn, pred, MEA, MOD, PAR)
    rfu.mea_vs_mod_plot(y, yn, dp, pred, MEA, MOD)
    rfu.rank_cluster_inf(cdf)
    dd2 = rfu.get_plot_inf(cdf, L, ds,PAR)
    rfu.plot_map(dd2)
    rfu.plot_map_rect(dd2)

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
    from sklearn.linear_model import ElasticNetCV,ElasticNet

    c200 = dsf.columns

    X = dm[c200]

    xn = np.sqrt((X ** 2).sum())

    XX = X.divide(xn, axis=1)


    y = dm[PAR]
    yn = y.notna()
    y = y[yn]
    ii = [.1, .5, .7, .9, .95, .99, 1]
    regr = ElasticNetCV(cv=5, random_state=0, positive=True, l1_ratio=ii, fit_intercept=False)
#     regr = ElasticNet( random_state=0, positive=True, l1_ratio=ii, fit_intercept=False)

    regr.fit(XX[yn], y[yn])

    pred = regr.predict(XX[yn])
    dp = pd.Series(pred, index=y[yn].index)
    cdf = pd.Series(regr.coef_, index=XX.columns)
    cdf = cdf / xn

# %%
regr = ElasticNetCV(alphas=.5,cv=5, random_state=0, positive=True, l1_ratio=ii, fit_intercept=False)

# %%

# %%
ccs = {}
r = {}
for a in np.geomspace(.1,8e3,201):
    from sklearn.linear_model import ElasticNetCV,ElasticNet

    c200 = dsf.columns

    X = dm[c200]

    xn = np.sqrt((X ** 2).sum())

    XX = X.divide(xn, axis=1)


    y = dm[PAR]
    yn = y.notna()
    y = y[yn]
    ii = [.1, .5, .7, .9, .95, .99, 1]
#     
    regr = ElasticNet(alpha=a, random_state=0, positive=True, l1_ratio=1, fit_intercept=False)

    regr.fit(XX[yn], y[yn])

    pred = regr.predict(XX[yn])
    dp = pd.Series(pred, index=y[yn].index)
    cdf = pd.Series(regr.coef_, index=XX.columns)
    cdf = cdf
    cdf = cdf/cdf.sum()
    ccs[a]=cdf
    r[a]=regr.score(XX[yn], y[yn])

# %%
regr = ElasticNetCV(cv=5, random_state=0, positive=True, l1_ratio=ii, fit_intercept=False)
regr.fit(XX[yn], y[yn])

# %%
regr.alpha_

# %%
pd.Series(r).plot()

# %%
ddd = pd.DataFrame(ccs).T

# %%

# %%
ddd

# %%
l = []
for a,r in ddd.iloc[::-1].iloc[::].iterrows():
    
    rr = r[~r.index.isin(l)]

    try:
        iii = rr.idxmax()
    #     print(iii)
        if rr.sum() == 0:
            continue
#             print(a)
        l.append(iii)
    except:
        break
    
    

# %%
len(l)

# %%
ddd[l].plot.area()
ax = plt.gca()
ax.legend(bbox_to_anchor=(1,1))
# ax.legend().remove()

# %%
rem = set(ddd.columns)-set(l)
l1 = [*l,*rem]

# %%
dd = pd.Series(l1).reset_index().set_index(0)['index'].to_dict()

# %%
ds2 = ds[L].to_series().replace(dd).to_xarray()

# %%
ds

# %%
q1,q2 = ds[AT].sum('time').quantile([.01,.5])

# %%
import matplotlib as mpl
dic = dict(projection=ccrs.Orthographic(0, 90), facecolor=".8")


f, axs = plt.subplots(6,6,subplot_kw=dic, sharex=True,sharey=True,figsize=(15,15),dpi=150)
for i in range(36):
    ax = axs.flatten()[i]
    p=ds[AT].sum('time').where(ds2<i).plot(
        # subplot_kws=dic,
        transform=ccrs.PlateCarree(),
    #     norm=mpl.colors.LogNorm(vmin, vmax),
    #     robust=True,
        cmap='viridis',
        vmax=q2,
        vmin=0,
        ax=ax
        
    )
    ax.set_title(i+1)
    if i ==15:    
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(5)
    p.axes.coastlines(lw=.5)



# %%
ds16 = ds[AT].sum('time').where(ds2<i)

# %%
ds16.to_dataframe()

# %%

# %%

# %%
import geopandas

d5 = fu.get_bounds(LA, ds[AT])
d6 = fu.get_bounds(LO, d5)

df = d6[L].to_dataframe()

df1 = df.loc[:, ~df.columns.duplicated()]

G = 'geometry'

df1[G] = df.apply(fu.get_pol, axis=1)


dg = geopandas.GeoDataFrame(df1).reset_index()

dg1 = dg[[L, G]].dissolve(by=L)

# %%
dg2=dg1[dg1.index.isin(l[:16])]
dg2.plot()

# %%
mpol = dg2.unary_union

# %%
pols = list(mpol)

# %%
pols_se = geopandas.GeoDataFrame(geometry=pols)

# %%
pols_se = pols_se[:]

# %%
pols_se['area'] = pols_se.area

# %%
ps1 = pols_se.sort_values('area')[::-1].reset_index(drop=True)

# %%
from shapely import geometry
ps1.contains(geometry.Point([.1,.1]))

# %%
df= ds.to_dataframe()

# %%
iv = {}
for i,v in dg2.centroid.items():
    n = ps1[ps1.contains(v)].index[0]
    iv[i]=n


# %%
iv

# %%
dss =ds.to_dataframe()


# %%
dss

# %%
dss1 = dss[dss[L].isin(l[:16])].copy()

# %%

NL = 'nl'

# %%
dss1[NL] = dss1[L].replace(iv)

# %%
dss2 = dss1.where(dss1[NL]<9)

# %%
ndf = dss2.reset_index().groupby([NL,'time'])[AT].sum().unstack().T

# %%
ndf

# %%
dss1.reset_index().groupby([NL,'time'])[AT].sum().unstack().sum().plot()

# %%

# %%

# %%

# %%
ndf.columns

# %%
ndm = ndf.copy()

# %%
ndm[SA]=dm[SA]

# %%
PAR = SA

    # %%
    from sklearn.linear_model import ElasticNetCV,ElasticNet, LinearRegression

    c200 = ndf.columns

    X = ndm[c200]

    xn = np.sqrt((X ** 2).sum())

    XX = X.divide(xn, axis=1)


    y = ndm[PAR]
    yn = y.notna()
    y = y[yn]
    ii = [.1, .5, .7, .9, .95, .99, 1]
#     
    regr = LinearRegression( positive=True, fit_intercept=False)

    regr.fit(XX[yn], y[yn])

    pred = regr.predict(XX[yn])
    dp = pd.Series(pred, index=y[yn].index)
    cdf = pd.Series(regr.coef_, index=XX.columns)
    cdf = cdf
#     cdf = cdf/cdf.sum()

    r2=regr.score(XX[yn], y[yn])

# %%
f,ax=plt.subplots(figsize=(20,5))
mod_ = (ndf/xn * cdf).sum(axis=1).where(yn)
ndm[SA].where(yn).plot(label=f"{SA} measured",lw=.5)
mod_.plot(label=f"{SA} inverse modeled",lw=.5)
plt.gca().set_yscale('log')

# %%
dum = pd.DataFrame()
dum['mea'] = ndm[SA].where(yn)
dum['mod'] = mod_

# %%

# %%
plt.scatter(dum['mea'],dum['mod'],alpha=.2, edgecolor='k', facecolor='none', )
ax= plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')

q0, q1 = np.quantile([*dum[yn]['mea'].values,*dum[yn]['mod'].values], [.001, .999])

ax.plot(
    [q0, q1],
    [q0, q1],
    label=('1 to 1')
)

# %% tags=[]
eight_=(ndf/xn * cdf)
eight_.plot.area(figsize=(10,10),subplots=True,sharey=True,ylim=(0,2.e6),cmap='tab10')
ax = plt.gca()


# %%
eight_.sum().to_frame().T.plot.bar()

# %%
order = eight_.sum().sort_values()[::-1]
order.plot.bar()

# %%
eight_.to_csv('../data_out/eight_sa_sources_inverse.csv')

# %%

# %% tags=[]
dic = dict(projection=ccrs.Orthographic(0, 90), facecolor=".8")
f, ax = plt.subplots(subplot_kw=dic, sharex=True,sharey=True,dpi=150)
ps1.reset_index().plot(column='index',cmap='tab10',ax=ax,transform=ccrs.PlateCarree())
ax.coastlines(lw=.5)

# %%
dic = dict(projection=ccrs.PlateCarree(), facecolor=".8")
f, ax = plt.subplots(subplot_kw=dic, sharex=True,sharey=True,dpi=150)
ps1.reset_index().plot(column='index',cmap='tab10',ax=ax,transform=ccrs.PlateCarree())
ax.coastlines(lw=.5)
ax.set_ylim(0,90)

# %%
dss1.to_xarray()[AT][{'time':2}]

# %%
#convert to mrakdown
# !jupyter-nbconvert --to markdown z030_inverse_modeling_time_series.ipynb

# %%

# %%

# %%
