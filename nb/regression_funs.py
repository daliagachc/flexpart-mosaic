import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr




def plot_ts(SA,df,DC):
    l1 = np.log10(df[SA])

    l2 = l1.resample('1H').mean()

    l3 = l2.rolling('10D', center=True).mean();

    f, ax = plt.subplots(figsize=(10, 3), dpi=100)
    q0, q1 = l1.quantile([.01, .99])
    l1.plot(lw=0, marker=',', color='k', alpha=.2)
    l3.plot(lw=2, alpha=.5, label='10D rolling mean', color=DC[SA])
    ax = plt.gca()
    ax.legend()
    ax.set_xlabel(f'log10[{SA}]')
    ax.set_ylim(q0, q1)
def elastic_net_reg(dsf, dm, PAR):
    from sklearn.linear_model import ElasticNetCV

    c200 = dsf.columns

    X = dm[c200]

    xn = np.sqrt((X ** 2).sum())

    XX = X.divide(xn, axis=1)


    y = dm[PAR]
    yn = y.notna()
    y = y[yn]
    ii = [.1, .5, .7, .9, .95, .99, 1]
    regr = ElasticNetCV(cv=5, random_state=0, positive=True, l1_ratio=ii, fit_intercept=False)

    regr.fit(XX[yn], y[yn])

    pred = regr.predict(XX[yn])
    dp = pd.Series(pred, index=y[yn].index)
    cdf = pd.Series(regr.coef_, index=XX.columns)
    cdf = cdf / xn
    return pred, cdf, y, yn, dp


def scatter_plot(y, yn, pred, MEA, MOD, PAR):

    q0, q1 = np.quantile([*y[yn], *pred], [.001, .999])
    f, ax = plt.subplots()
    ax.scatter(y[yn], pred, alpha=.2, edgecolor='k', facecolor='none', )
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(MEA)
    ax.set_ylabel(MOD)
    ax.plot(
        [q0, q1],
        [q0, q1],
        label=('1 to 1')
    )
    ax.legend()
    ax.set_ylim(q0, q1)
    ax.set_xlim(q0, q1)
    ax.set_aspect('equal')
    ax.set_title(PAR)

def mea_vs_mod_plot(y, yn, dp, pred, MEA, MOD):
    q0, q1 = np.quantile([*y[yn], *pred], [.001, .999])
    f, ax = plt.subplots(figsize=(20, 3))

    y.resample('3H').median().plot(label=MEA,ax =ax)
    dp.resample('3H').median().plot(label=MOD, ax =ax )
    ax = plt.gca()
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.legend()
    ax.grid()
    ax.set_ylim(q0, q1)

def rank_cluster_inf(cdf):
    f, ax = plt.subplots(figsize=(20, 3))
    f1 = cdf.sort_values(ascending=False).reset_index(drop=True)

    f1.sort_values().plot(figsize=(20, 3), lw=1, marker='o',ax =ax )
    ax = plt.gca()
    ax.set_xlabel('''ranked influence of plots
    above: here we plot the influence of clusters order from highest to lowest.''')
    ax.grid()

def get_plot_inf(cdf, L, ds, PAR):
    cdf.index.name = L

    cdic = cdf.to_dict()

    PO = potential_influence = f'Potential influence of {PAR} [arbitrary units]'

    dd1 = ds[L].to_series().replace(cdic)
    dd1.name = PO

    dd2 = dd1.to_xarray()

    return dd2


def plot_map(dd2,ax=None):
    import cartopy.crs as ccrs

    # air = xr.tutorial.open_dataset("air_temperature").air

    dic = dict(projection=ccrs.Orthographic(0, 90), facecolor="gray")

    if ax is None:
        f, ax = plt.subplots(subplot_kw=dic, dpi=200)
    else:
        ax =ax
        f = ax.figure

    p = dd2.plot(
        # subplot_kws=dic,
        transform=ccrs.PlateCarree(),
        #     norm=mpl.colors.LogNorm(vmin, vmax),
        robust=True,
        cmap='Reds',
        ax=ax,
        cbar_kwargs = {'orientation':"horizontal"}
    )
    #   p.axes.set_global()
    p.axes.coastlines()


def plot_map_rect(dd2):
    import cartopy.crs as ccrs

    # air = xr.tutorial.open_dataset("air_temperature").air

    dic = dict(projection=ccrs.PlateCarree(), facecolor="gray")

    f, ax = plt.subplots(subplot_kw=dic, dpi=200)

    p = dd2.plot(
        # subplot_kws=dic,
        transform=ccrs.PlateCarree(),
        #     norm=mpl.colors.LogNorm(vmin, vmax),
        robust=True,
        ax=ax,
        cmap = 'Reds',
    )
    p.axes.set_global()
    p.axes.coastlines()