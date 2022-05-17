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

PATH_TO_BC = '../data_in/black_carbon_channel6_masked_5min.csv'
# PATH_TO_BC = '../data_in/black_carbon_channel6_raw_5min.csv'

PATH_1000_CLUS = '../data_out/cluster1000.nc'

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

BC = 'bc_masked_ngm3'

# BC = 'bc_raw_ngm3'

LBC = 'log(bc)'

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
    BC  : COL(3),
    LBC : COL(3)
}


# %%
df = pd.read_csv(PATH_TO_BC, index_col=0, parse_dates=True)
(df[BC]>0).value_counts()

# %%
df[BC].plot.hist(bins=[df[BC].min()-1,*np.geomspace(.01,10000)])
plt.gca().set_xscale('log')

# %%

# %%
df=df[df[BC]>0]

# %%
p = (df[BC]**1)
q1,q2 = p.quantile([.001,.999])
p.plot.hist(bins=np.linspace(q1,q2))
ax = plt.gca()
ax.set_xlim(q1,q2)

# %%
p = (df[BC]**.3)
q1,q2 = p.quantile([.001,.999])
p.plot.hist(bins=np.linspace(q1,q2))
ax = plt.gca()
ax.set_xlim(q1,q2)

# %%
df[LBC] = np.log10(df[BC])

# %%

# %% [markdown]
# # distributions
#
#
# lets find what kind of dist. do we have. They seem to be log dists.

# %%

# %%
for c in [BC]:
    np.log10(df[c]).plot.hist(alpha=.5, bins=100, label=c, color=DC[c])
ax = plt.gca()
ax.legend()
ax.set_xlabel('log10')

# %% [markdown]
# # timeseries of the trace gases

# %%

# %% [markdown]
# # open and merge flex 1000 clusters

# %%
ds = xr.open_dataset(PATH_1000_CLUS)

# %%
dsf = ds.groupby(L).sum().to_dataframe()[AT].unstack(L)

# %%
dsf.sum()

# %%
from sklearn.preprocessing import RobustScaler

# %%
RS = RobustScaler(
    with_centering=False,
    with_scaling=True,
    quantile_range=(25.0, 75.0),
    copy=True,
    unit_variance=False,
)

# %%
dsf2 = dsf/np.sqrt((dsf**2).sum())

# %%
dsf2.sum()

# %%
df1 = df[[BC]]

# %%

# %%
df2 = df1.resample('3H').median()
df3 = df2[~df2[BC].isna()]

# %%

# %%
dm = pd.merge(df3, dsf2, left_index=True, right_index=True, how='inner', validate="1:1")


# %%

# %% [markdown]
# # Linear Regression

# %%
from sklearn.linear_model import ElasticNetCV,ElasticNet, LinearRegression

# %%
lreg = LinearRegression(
    fit_intercept=False,
    normalize='deprecated',
    copy_X=True,
    n_jobs=None,
    positive=True,
)

# %%
X = dm[dsf2.columns]
y = dm[BC]
lreg.fit(X,y)
y_pred = lreg.predict(X)

# %%
plt.scatter(y_pred,y,alpha=.1)
ax=plt.gca()
ax.set_xlim(0,y.quantile(.99))
ax.set_ylim(0,y.quantile(.99))
ax.set_aspect('equal')

# %%
f,ax = plt.subplots(figsize=(20,5))
y.plot(lw=0,marker='x')
(y*0 + y_pred).plot(lw=0,marker='v')

# %%
tot_sd = (y**2).sum()**.5

# %%
exp_sd = ((y-y_pred)**2).sum()**.5

# %%
best_sd = exp_sd/tot_sd
best_sd

# %%
np.corrcoef(y,y_pred)[0,1]**2

# %%
plt.hist(lreg.coef_,bins=40);

# %% [markdown]
# # Elastic net regression cross validated
#
# $L_{\text {enet }}(\hat{\beta})=\frac{\sum_{i=1}^{n}\left(y_{i}-x_{i}^{\prime} \hat{\beta}\right)^{2}}{2 n}+\lambda\left(\frac{1-\alpha}{2} \sum_{j=1}^{m} \hat{\beta}_{j}^{2}+\alpha \sum_{j=1}^{m}\left|\hat{\beta}_{j}\right|\right)$

# %%
e_reg_cv=ElasticNetCV(
    l1_ratio=[.1,0.5,0.7,.9,.99,.999,.9999,.99999,.999999,.9999999],
    eps=0.001,
    n_alphas=100,
    alphas=None,
    fit_intercept=False,
    normalize='deprecated',
    precompute='auto',
    max_iter=1000,
    tol=0.0001,
    cv=5,
    copy_X=True,
    verbose=0,
    n_jobs=None,
    positive=True,
    random_state=None,
    selection='cyclic',
)

# %%
X = dm[dsf2.columns]
y = dm[BC]
e_reg_cv.fit(X,y)
y_pred = e_reg_cv.predict(X)

# %%
exp_sd = ((y-y_pred)**2).sum()**.5

# %%
exp_sd/tot_sd

# %%
cv_a = e_reg_cv.alpha_
cv_a

# %%
cv_l1 = e_reg_cv.l1_ratio_
cv_l1

# %%
plt.hist(e_reg_cv.coef_,bins=40);

# %%
plt.scatter(y_pred,y,alpha=.1)
ax=plt.gca()
ax.set_xlim(0,y.quantile(.99))
ax.set_ylim(0,y.quantile(.99))
ax.set_aspect('equal')

# %%
f,ax = plt.subplots(figsize=(20,5))
y.plot(lw=0,marker='x')
(y*0 + y_pred).plot(lw=0,marker='v')

# %% [markdown]
# alpha = a + b and l1_ratio = a / (a + b)

# %%
cv_a

# %%
B = cv_a * (1-cv_l1)
B

# %%
A = cv_a - B
A


# %%

# %% [markdown]
# # selection

# %%
def elastic_net(_a, _l1, dsf2, dm, BC,tot_sd):
    e_reg = ElasticNet(
        alpha=_a,
        l1_ratio=_l1,
        fit_intercept=False,
        normalize='deprecated',
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=0.0001,
        warm_start=False,
        positive=True,
        random_state=None,
        selection='cyclic',
    )

    X = dm[dsf2.columns]
    y = dm[BC]
    e_reg.fit(X, y)
    y_pred = e_reg.predict(X)



    exp_sd = ((y - y_pred) ** 2).sum() ** .5

    ratio = exp_sd / tot_sd

    coef = X.sum().T * 0 + e_reg.coef_
    return coef, ratio, exp_sd
# %%
def to_min(fac):
    nA = fac*A

    _a = B + nA 

    _l1 = nA/(nA + B)


    coef, ratio, exp_sd = elastic_net(_a, _l1, dsf2, dm, BC,tot_sd)
#     print(ratio)
    return np.abs(ratio - .9999)

# %%
from scipy.optimize import minimize_scalar

# %% tags=[]
res = minimize_scalar(to_min,method='Bounded',bounds=(1,1000))

# %%
aas = np.linspace(1,res.x,100)

# %%
aas

# %% tags=[] jupyter={"outputs_hidden": true}
dic = {}
for aa in aas:
#     print(aa)
    d2 = {}
    nA = aa * A
#     print(aa)

    _a = B + nA 

    _l1 = nA/(nA + B)


    coef, ratio, exp_sd = elastic_net(_a, _l1, dsf2, dm, BC,tot_sd)

    inn=coef[coef>0].index


    e_reg_cv=ElasticNetCV(
        l1_ratio=0.01,
        eps=0.001,
    #     n_alphas=100,
        alphas=[.0001,.001,.01,.1,1,10,100,1000,10000],
        fit_intercept=False,
        normalize='deprecated',
        precompute='auto',
        max_iter=100000,
        tol=0.0001,
        cv=5,
        copy_X=True,
        verbose=0,
        n_jobs=None,
        positive=True,
        random_state=None,
        selection='cyclic',
    )

    X = dm[inn]
    y = dm[BC]
    e_reg_cv.fit(X, y)
    y_pred = e_reg_cv.predict(X)

    exp_sd = ((y - y_pred) ** 2).sum() ** .5

    ratio = exp_sd / tot_sd

    coef = X.sum().T * 0 + e_reg_cv.coef_

    d2['coef'] = coef
    d2['ratio'] = ratio
    d2['exp_sd'] = exp_sd
    dic[aa] = d2
        


# %%
dk = {}
for k,r in dic.items():
    dk[k] = r['coef'].to_dict()
ddf = pd.DataFrame(dk).T

# %%

# %%

ddf.plot.area()
ax = plt.gca()
ax.get_legend().remove()

# %%
dk = {}
for k,r in dic.items():
    dk[k] = 1-r['ratio']
pd.Series(dk).plot()

# %%
dk = {}
for k,r in dic.items():
    dk[1-dic[k]['ratio']] = r['coef'].to_dict()
ddf = pd.DataFrame(dk).T

# %%

# %%
la = []
rr = []
# lr = {0.1+i/100000:0 for i in range(1,100)}
lr = {0:0}
# lr = {}
for l,r in ddf[::-1].iterrows():
    ii = r[r>0].index
    bo = ~ii.isin(la)
    la = [*la,*list(ii[bo])]
    rr = [*rr,*r[list(ii[bo])].values]
    lr[len(la)] = l
    
    

# %%
n_sd = pd.Series(lr)

# %%

# %%
from pygam import LinearGAM, s

# X = np.log10(n_sd.reset_index()[['index']])
X = (n_sd.reset_index()[['index']])
y = n_sd

gam1 = LinearGAM(s(0, constraints='concave',n_splines=20,spline_order=3)).fit(X, y)
# gam1 = LinearGAM(s(0,n_splines=20,spline_order=3)).fit(X, y)

# %%
X1=10**X
X1=X

# %%
f,ax = plt.subplots()
ax.plot(X1,n_sd,lw=0,marker='x')
ax.plot(X1,gam1.predict(X))

def ffit(x): 
#     if x ==0:
#         return 0
#     else:
#         return gam1.predict([x])[0]
    return gam1.predict([x])[0]

from scipy.misc import derivative

x = np.arange(0,500,1)

y = [derivative(ffit,_) for _ in x ]

axx = ax.twinx()
axx.plot(x,y,lw=1,marker='o')
# axx.set_yscale('log')

# %%

# %%
ni = pd.Series(rr,index=la)

# %%

# %%
ni1 = ni[::-1].reset_index().reset_index().set_index('index')['level_0'].to_dict()
ni1 = ni[::].reset_index().reset_index().set_index('index')['level_0'].to_dict()

# %%
ni2 = {i:derivative(ffit,_)  for i,_ in ni1.items()}
ni2 = {i:ffit(_)  for i,_ in ni1.items()}

# %%
_ds = ds['lab'].to_series()

# %%
_ds1 = _ds.where(_ds.isin(ni2)).replace(ni2).to_xarray()

# %%
_ds1

# %%

# %%
# plt.scatter(y_pred,y,alpha=.1)
# ax=plt.gca()
# ax.set_xlim(0,y.quantile(.99))
# ax.set_ylim(0,y.quantile(.99))
# ax.set_aspect('equal')

# %%
# f,ax = plt.subplots(figsize=(20,5))
# y.plot(lw=0,marker='x')
# (y*0 + y_pred).plot(lw=0,marker='v')

# %%
_ds2 = (_ds1/_ds1.max())**10
_ds2 = _ds1/_ds1.max()
_ds2 = _ds1

# %%
# inn = coef[coef>0].index

# pl = ds['lab'].where(ds['lab'].isin(inn))* 0 + 1




import cartopy.crs as ccrs
ax = plt.subplot(projection=ccrs.Orthographic(0, 90), facecolor=plt.get_cmap('inferno_r')(0))
(_ds2).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    robust=True,
    levels=6,
    cmap='inferno',
#     vmin=0,

)
ax.coastlines(color='red')

# %%
ds

# %%

# %%
