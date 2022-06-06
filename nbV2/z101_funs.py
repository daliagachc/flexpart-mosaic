import numpy as np
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from scipy.misc import derivative
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.linear_model import ElasticNet



def elastic_net(_a, _l1, dsf2, dm, PAR,tot_sd):
    e_reg = ElasticNet(
        alpha=_a,
        l1_ratio=_l1,
        fit_intercept=False,
        normalize='deprecated',
        precompute=False,
        max_iter=10000,
        copy_X=True,
        tol=0.0001,
        warm_start=False,
        positive=True,
        random_state=None,
        selection='cyclic',
    )

    X = dm[dsf2.columns]
    y = dm[PAR]
    e_reg.fit(X, y)
    y_pred = e_reg.predict(X)



    exp_sd = ((y - y_pred) ** 2).sum() ** .5

    ratio = exp_sd / tot_sd

    coef = X.sum().T * 0 + e_reg.coef_
    return coef, ratio, exp_sd


def plot_15(_ds2):
    dic = dict(projection=ccrs.Orthographic(0, 90))
    f,ax = plt.subplots(facecolor='w', figsize=(10,10), subplot_kw=dic)
    (_ds2).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        robust=True,
        levels=6,
        cmap='inferno',
        #     vmin=0,

    )
    ax.coastlines(color='red')


def get_ds2(ds, ffit, la, rr):
    ni = pd.Series(rr, index=la)


    ni1 = ni[::-1].reset_index().reset_index().set_index('index')['level_0'].to_dict()
    ni1 = ni[::].reset_index().reset_index().set_index('index')['level_0'].to_dict()

    ni2 = {i: derivative(ffit, _) for i, _ in ni1.items()}
    ni2 = {i: ffit(_) for i, _ in ni1.items()}

    _ds = ds['lab'].to_series()

    _ds1 = _ds.where(_ds.isin(ni2)).replace(ni2).to_xarray()




    _ds2 = (_ds1/_ds1.max())**10
    _ds2 = _ds1/_ds1.max()
    _ds2 = _ds1
    return _ds2


def plot_11_13(X, X1, ffit, gam1, n_sd):
    x = np.arange(0, 500, 1)
    y = [derivative(ffit, _) for _ in x]
    ax = plot_11(X, X1, gam1, n_sd)
    plot_13(ax, x, y)


def plot_13(ax, x, y):
    axx = ax.twinx()
    axx.plot(x, y, lw=1, marker='o')


def plot_11(X, X1, gam1, n_sd):
    f, ax = plt.subplots()
    ax.plot(X1, n_sd, lw=0, marker='x')
    ax.plot(X1, gam1.predict(X))
    return ax


def do_some_GAM(n_sd):
    from pygam import LinearGAM, s
    # X = np.log10(n_sd.reset_index()[['index']])
    X = (n_sd.reset_index()[['index']])
    y = n_sd
    gam1 = LinearGAM(s(0, constraints='concave', n_splines=20, spline_order=3)).fit(X, y)
    # gam1 = LinearGAM(s(0,n_splines=20,spline_order=3)).fit(X, y)

    X1 = 10 ** X
    X1 = X

    def ffit(x):
        #     if x ==0:
        #         return 0
        #     else:
        #         return gam1.predict([x])[0]
        return gam1.predict([x])[0]

    return X, X1, gam1, ffit


def do_a_loop(ddf2):
    la = []
    rr = []
    # lr = {0.1+i/100000:0 for i in range(1,100)}
    lr = {0: 0}
    # lr = {}
    for l, r in ddf2[::].iterrows():
        ii = r[r > 0].index
        bo = ~ii.isin(la)
        la = [*la, *list(ii[bo])]
        rr = [*rr, *r[list(ii[bo])].values]
        lr[len(la)] = l
    #     if len(la)>37: break
    return la, lr, rr


def get_ddf2(ddf, dic):
    dk = {}
    for k, r in dic.items():
        dk[1 - dic[k]['ratio']] = r['coef'].to_dict()
    ddf = pd.DataFrame(dk).T
    return ddf


def plot_9(dic):
    dk = {}
    for k, r in dic.items():
        dk[k] = 1 - r['ratio']
    pd.Series(dk).plot()


def plot_8(ddf):
    ddf.plot.area()
    ax = plt.gca()
    ax.get_legend().remove()


def get_ddf(dic):
    dk = {}
    for k, r in dic.items():
        dk[k] = r['coef'].to_dict()
    ddf = pd.DataFrame(dk).T
    return ddf


def do_something(A, B, PAR, dm, dsf2, res, tot_sd):

    aas = np.linspace(1, res.x, 100)

    dic = {}
    for aa in aas[::-1]:
        #     print(aa)
        d2 = {}
        nA = aa * A
        #     print(aa)

        _a = B + nA

        _l1 = nA / (nA + B)

        coef, ratio, exp_sd = elastic_net(_a, _l1, dsf2, dm, PAR, tot_sd)


        d2['coef'] = coef
        d2['ratio'] = ratio
        d2['exp_sd'] = exp_sd
        dic[aa] = d2
    return dic


def get_to_min(cv_a, cv_l1,dsf2, dm, PAR, tot_sd ):
    B = cv_a * (1 - cv_l1)

    A = cv_a - B

    def to_min(fac):
        nA = fac * A

        _a = B + nA

        _l1 = nA / (nA + B)

        coef, ratio, exp_sd = elastic_net(_a, _l1, dsf2, dm, PAR, tot_sd)
        #     print(ratio)
        #         g.append([ratio, fac])

        return np.abs(ratio - .999)

    return A, B, to_min


def plot_7(y, y_pred):
    f, ax = plt.subplots(figsize=(20, 5))
    y.plot(lw=0, marker='x', alpha=.5)
    (y * 0 + y_pred).plot(lw=0, marker='v', alpha=.5)


def get_e_reg_cv(PAR, dm, dsf2, y, y_pred):
    e_reg_cv = ElasticNetCV(
        l1_ratio=[.1, 0.5, 0.7, .9, .99, .999, .9999, .99999, .999999, .9999999],
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

    X = dm[dsf2.columns]
    y = dm[PAR]
    e_reg_cv.fit(X, y)
    y_pred = e_reg_cv.predict(X)
    return e_reg_cv, y, y_pred


def get_best_sd(y, y_pred):
    tot_sd = (y ** 2).sum() ** .5

    exp_sd = ((y - y_pred) ** 2).sum() ** .5

    best_sd = exp_sd / tot_sd
    return best_sd, tot_sd


def plot_ts_simple_linear_reg(y, y_pred):
    f, ax = plt.subplots(figsize=(20, 5))
    y.plot(lw=0, marker='x')
    (y * 0 + y_pred).plot(lw=0, marker='v')


def get_simple_linear_regression(PAR, dm, dsf2):
    lreg = LinearRegression(
        fit_intercept=False,
        normalize='deprecated',
        copy_X=True,
        n_jobs=None,
        positive=True,
    )

    X = dm[dsf2.columns]
    y = dm[PAR]
    lreg.fit(X, y)
    y_pred = lreg.predict(X)
    return lreg, y, y_pred


def plot_results_simple_linear_regression(y, y_pred):
    plt.scatter(y_pred, y, alpha=.1)
    ax = plt.gca()
    ax.set_xlim(0, y.quantile(.99))
    ax.set_ylim(0, y.quantile(.99))
    ax.set_aspect('equal')


def open_dm(AT, L, PAR, PATH_1000_CLUS, df):
    ds = xr.open_dataset(PATH_1000_CLUS)

    dsf = ds.groupby(L).sum().to_dataframe()[AT].unstack(L)


    dsf2 = dsf / np.sqrt((dsf ** 2).sum())

    dsf2.sum()

    df1 = df[[PAR]]


    df2 = df1.resample('3H').median()
    df3 = df2[~df2[PAR].isna()]


    dm = pd.merge(df3, dsf2, left_index=True, right_index=True, how='inner', validate="1:1")
    return dm, ds, dsf2



def plot_dist_3(PAR, df):
    p = (df[PAR] ** .3)
    q1, q2 = p.quantile([.001, .999])
    p.plot.hist(bins=np.linspace(q1, q2))
    ax = plt.gca()
    ax.set_xlim(q1, q2)
    plt.gca().set_title(PAR)


def plot_dist_2(PAR, df):
    p = (df[PAR] ** 1)
    q1, q2 = p.quantile([.001, .999])
    p.plot.hist(bins=np.linspace(q1, q2))
    ax = plt.gca()
    ax.set_xlim(q1, q2)
    plt.gca().set_title(PAR)


def plot_dist_1(PAR, df):
    q1, q2 = df[PAR].quantile([.001, .999])
    df[PAR].plot.hist(bins=[df[PAR].min() - 1, *np.geomspace(q1, q2)])
    plt.gca().set_xscale('log')
    plt.gca().set_title(PAR)