import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


# CM = plt.get_cmap('tab20')


LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQ'

def plot_map(da, vmin=None, vmax=None, ax=None, grid=True):
    import cartopy.crs as ccrs

    # air = xr.tutorial.open_dataset("air_temperature").air

    dic = dict(projection=ccrs.Orthographic(0, 90), facecolor="gray")

    if ax is None:
        f, ax = plt.subplots(subplot_kw=dic, )

    p = da.plot(
        # subplot_kws=dic,
        transform=ccrs.PlateCarree(),
        norm=mpl.colors.LogNorm(vmin, vmax),
        robust=True
    )
    #   p.axes.set_global()
    p.axes.coastlines()
    if grid:
        p.axes.gridlines()
    f = plt.gcf()
    f.set_dpi(150)
    return p


def plot_kmap(da, n, CM, ax=None):
    import cartopy.crs as ccrs
    dic = dict(projection=ccrs.Orthographic(0, 90), facecolor="gray")

    if ax is None:
        f, ax = plt.subplots(subplot_kw=dic)

    cmap = plt.get_cmap(CM, n)

    # air = xr.tutorial.open_dataset("air_temperature").air

    p = da.plot(
        # subplot_kws=dic,
        transform=ccrs.PlateCarree(),
        #         norm=mpl.colors.LogNorm(vmin,vmax),
        robust=True,
        cmap=cmap,
        vmin=-.5,
        vmax=n - .5,
        ax=ax
    )
    #   p.axes.set_global()
    p.axes.coastlines()
    p.axes.gridlines()
    f = plt.gcf()
    f.set_dpi(150)
    return p


def plot_path(df, LO, LA, ax=None):
    import cartopy.crs as ccrs
    if ax is None:
        dic = dict(projection=ccrs.Orthographic(0, 90), facecolor="white")
        fig, ax = plt.subplots(subplot_kw=dic, dpi=150)
    else:
        fig = ax.figure
    l1 = 15
    ll = [plt.get_cmap('magma', l1)(i + 2) for i in range(l1 - 2)]
    cmap = mpl.colors.ListedColormap(ll)
    smap = ax.scatter(
        df[LO], df[LA],
        s=20, c=df.index,
        edgecolors='none', marker='o', cmap=cmap,
        transform=ccrs.PlateCarree()
    )

    cb = fig.colorbar(smap, orientation='vertical')
    ax.coastlines()
    ax.gridlines()

    ii = df.index.round('1D').drop_duplicates()
    d0 = ii.day == 1
    ii = ii[d0]

    cb.ax.set_yticks(ii)

    print(cb.get_ticks())

    cb.ax.set_yticklabels(pd.to_datetime(cb.get_ticks()).strftime(date_format='%b %Y'))


def plot_some_distributions(d2, LA, LO, ):
    sub = d2[{LA: slice(None, None, 21), LO: slice(None, None, 21)}]
    la = sub[LA]
    lo = sub[LO]
    lalo = sub.reset_coords()[[LA, LO]].to_dataframe().reset_index()
    ll = len(lalo)
    r = int(np.sqrt(ll))

    c = int(np.ceil(ll / r))

    f, axs = plt.subplots(nrows=r, ncols=c, figsize=(10, 10),
                          sharex=False, sharey=True)
    axf = axs.flatten()

    for i in range(ll):
        la, lo = lalo.iloc[i]

        se = d2.loc[{LO: lo, LA: la}].to_series()

        for j in np.linspace(.005, 1):
            q0 = se.quantile(j)
            if q0 > 0:
                break

        gs = np.geomspace(
            q0
            ,
            se.quantile(.99)
        )
        bs = [0, *gs]

        ax = axf[i]
        sns.histplot(se, bins=bs, ax=ax)
        ax.set_xscale('log')

    for ax in axf:
        ax.set_xlabel(None)
        ax.set_ylabel(None)


def kmeans_cluster(N, qta, d3, L, TI, ):
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=N,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm='auto',
    )

    q1 = qta.to_dataframe()

    q2 = q1.unstack(TI)

    km.fit(q2.values)

    lab = pd.Series(km.labels_, index=q2.index)

    lax = lab.to_xarray()

    d4 = d3

    d4 = d4.assign_coords({L: lax})

    ldf = d4.groupby(L).sum().sum(TI).to_series()

    lab_dic0 = ldf.sort_values(ascending=False
                               ).reset_index()[L].reset_index().set_index(L)['index'].to_dict()


    lab_dic = {k: LETTERS[v] for k, v in lab_dic0.items()}

    lab = lab.replace(lab_dic)
    lax = lab.to_xarray()

    d4 = d3

    d4 = d4.assign_coords({L: lax})

    return d4


def plot_month_means(d4, L, CM, N, ):
    cm = plt.get_cmap(CM, N)
    lt = d4.groupby(L).sum()

    lt2 = lt.to_series().unstack(L)
    # ma = matplotlib.markers
    from matplotlib.lines import Line2D
    unfilled_markers = [m for m, func in Line2D.markers.items()
                        if func != 'nothing' and m not in Line2D.filled_markers]
    mm = unfilled_markers[2:8] * 4

    f, ax = plt.subplots(dpi=100)
    mea = lt2.resample('M').mean()
    for i in range(N):
        l = LETTERS[i]
        mea[l].plot(ax=ax, lw=2, alpha=1, marker=mm[i], markersize=10, c=cm(i))

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='cluster')
    ax.grid()


# %%
def plot_cluster_ts(N, d4, L, CM, fig=None):
    cm = plt.get_cmap(CM, N)
    fs = (10, N * 2)
    if fig is None:
        fig = plt.figure(figsize=fs)

    axs = fig.subplots(N, 1, sharex=True, sharey=True).flatten()

    df = d4.groupby(L).sum().to_series().unstack(L)

    for i in range(N):
        l = LETTERS[i]
        df.loc[:, l].plot(ax=axs[i], c=cm(i), label=f'{l}')

    for ax in axs:
        ax.grid()
        ax.set_ylabel('cluster residence\ntime [days]')
        ax.legend()


def plot_cluster_bar(N, d4, L, CM, ax=None):
    cm = plt.get_cmap(CM, N)
    if ax is None:
        f, ax = plt.subplots()
    dd = d4.groupby(L).sum().to_series().unstack(L)
    dd.mean().plot.bar(
        #     subplots=True,
        #     figsize=(10,N*1),
        color=[cm(c) for c in range(N)],
        ax=ax
    )
    ax.set_ylabel('cluster mean influence [days]')
    ax.set_xlabel('cluster')


def plot_residence_time(d2, LA, LO):
    d2s = d2.sum([LA, LO]) / 3600 / 24
    mm = d2s.mean().item()
    d2s.plot(marker=',', lw=0)
    d2s.to_series().rolling(40).mean().plot()
    ax = plt.gca()
    l = f'mean = {mm:0.1f} days'
    ax.axhline(mm, label=l, c='k', ls='--')
    ax.legend()


def get_quantiles(d3, TI):
    # aplplt a quantile transform across the time dimension so that
    # we can cluster the values
    def quantiler(a, b=None):
        # global ii
        #     print(ii)
        # ii += 1
        from sklearn.preprocessing import QuantileTransformer

        b = a.reshape(1, -1).T
        qt = QuantileTransformer()
        res = qt.fit_transform(b)
        aa = res
        return res[:, 0]

    qta = xr.apply_ufunc(quantiler, d3,
                         input_core_dims=[[TI]],
                         output_core_dims=[[TI]],
                         vectorize=True)
    return qta


def get_pol(r):
    from shapely.geometry import Polygon
    global rr
    rr = r

    lat0, lat1, lon0, lon1 = r.loc[['lat0', 'lat1', 'lon0', 'lon1']]
    ll = [
        [lon0, lat0],
        [lon1, lat0],
        [lon1, lat1],
        [lon0, lat1],
        [lon0, lat0]
    ]
    p = Polygon(ll)
    return p


def get_bounds(LL, d):
    d4 = d
    from xarray.plot.plot import _infer_interval_breaks as infer_interval_breaks
    # las = [f'{LA}00', f'{LA}10', f'{LA}11', f'{LA}01']
    lamm = [f'{LL}0', f'{LL}1']
    ib = infer_interval_breaks(d4[LL])
    ib0 = ib[:-1]
    x0 = xr.DataArray(ib0, dims=LL, coords={LL: d4[LL].values})
    d4[lamm[0]] = x0
    ib1 = ib[1:]
    x1 = xr.DataArray(ib1, dims=LL, coords={LL: d4[LL].values})
    d4[lamm[1]] = x1
    return d4


# import geopandas
# from geopandas import GeoSeries

def plot_hatch(d4, L, N, d1, LA, LO, CM):
    import geopandas

    import cartopy.crs as ccrs

    d5 = get_bounds(LA, d4)
    d6 = get_bounds(LO, d5)

    df = d6[L].to_dataframe()

    df = df.loc[:, ~df.columns.duplicated()]

    G = 'geometry'

    df[G] = df.apply(get_pol, axis=1)

    dg = geopandas.GeoDataFrame(df).reset_index()

    dg1 = dg[[L, G]].dissolve(by=L).reset_index()

    hs = ['\\\\', '//', '|', '-', '+'] * 5

    import matplotlib.patches as mpatches

    dic = dict(projection=ccrs.Orthographic(0, 90), facecolor="gray")
    fig, ax = plt.subplots(subplot_kw=dic, dpi=150)

    plot_map(d1, ax=ax, grid=False)

    # cm = plt.get_cmap('tab20', N)
    labs = []

    cm = plt.get_cmap(CM, N)

    for i in range(N):
        # print(dg1)
        l = LETTERS[i]
        boo = dg1[L] == l
        dg1[boo].plot(column=L, hatch=hs[i] * 2, ax=ax, transform=ccrs.PlateCarree(),
                      facecolor='none', edgecolor=cm(i), zorder=10
                      )
        p = mpatches.Patch(label=f'{l}', hatch=hs[i] * 3, edgecolor=cm(i), facecolor='none')
        labs.append(p)
    ax.coastlines()
    ax.legend(handles=labs, loc='upper right', bbox_to_anchor=(0, 1))

    ax.set_title(f'{N} CLUSTERS')


def get_out_path(i, DATA_OUT):
    import os
    fname = f'{i:02d}_clusters_ts.csv'
    path_out = os.path.join(DATA_OUT, fname)
    return path_out


def save_cluster_csv(d4, N, DATA_OUT, L):
    pout = get_out_path(N, DATA_OUT)

    csv = d4.groupby(L).sum().to_series().unstack(L)

    csv.to_csv(pout)

def compressed_netcdf_save(ds, path, shuffle=True, complevel=4, fletcher32=True, encode_u=False):
    encoding = {}
    for k, v in ds.variables.items():
        encoding[k] = {
            'zlib'      : True,
            'shuffle'   : shuffle,
            'fletcher32': fletcher32,
            'complevel' : complevel
        }
        if encode_u:
            if v.dtype.kind == 'U':
                encoding[k]['dtype'] = 'S1'
    ds.to_netcdf(path, encoding=encoding)