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

# %% [markdown]
# ## readme 
#
# Work in progress, needs better naming of the funcs

# %% jupyter={"outputs_hidden": false}
# %load_ext autoreload
# %autoreload 2

# %% jupyter={"outputs_hidden": false}
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import minimize_scalar

# %% md [markdown]
# ## constants and functions

# %%
import z101_funs as f1


# %% jupyter={"outputs_hidden": false}
def fake_main():


    # %% jupyter={"outputs_hidden": false}
    PATH_TO_ST = '../data_in/ciapitof_masked_filtered.csv'

    PATH_1000_CLUS = '../data_out/cluster1000.nc'

    AT = 'AIRTRACER'
    L = 'lab'

    SA = 'sa'
    MSA = 'msa'
    IA = 'ia'

    BC = 'bc_masked_ngm3'


    LBC = 'log(bc)'

    LSA = 'log10(sa)'
    LMSA = 'log10(msa)'
    LIA = 'log10(ia)'

    PAR = MSA



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


    # %%
    df = pd.read_csv(PATH_TO_ST, index_col=0, parse_dates=True)
    (df[PAR]>0).value_counts()
    df = df[df[PAR]>0][[PAR]]


    # %%
    f1.plot_dist_1(PAR, df)

    # %%
    f1.plot_dist_2(PAR, df)

    # %%
    f1.plot_dist_3(PAR, df)


# %%

# %% [markdown]
#     # # distributions
#     # lets find what kind of dist. do we have. They seem to be log dists.
#
#


# %% [markdown]
# ## open and merge flex 1000 clusters

    # %%
    dm, ds, dsf2 = f1.open_dm(AT, L, PAR, PATH_1000_CLUS, df)

# %%

# %% [markdown]
# ## Linear Regression

# %%


    # %%
    lreg, y, y_pred = f1.get_simple_linear_regression(PAR, dm, dsf2)

    # %%
    f1.plot_results_simple_linear_regression(y, y_pred)

    # %%
    f1.plot_ts_simple_linear_reg(y, y_pred)

    # %%
    best_sd, tot_sd = f1.get_best_sd(y, y_pred)
    best_sd


    # %%
    plt.hist(lreg.coef_,bins=40);

# %% [markdown]
#    ## Elastic net regression cross validated   

# %% [markdown]
#    # $L_{\text {enet }}(\hat{\beta})=\frac{\sum_{i=1}^{n}\left(y_{i}-x_{i}^{\prime} \hat{\beta}\right)^{2}}{2 n}+\lambda\left(\frac{1-\alpha}{2} \sum_{j=1}^{m} \hat{\beta}_{j}^{2}+\alpha \sum_{j=1}^{m}\left|\hat{\beta}_{j}\right|\right)$

    # %%
    e_reg_cv, y, y_pred = f1.get_e_reg_cv(PAR, dm, dsf2, y, y_pred)

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
    f1.plot_results_simple_linear_regression(y, y_pred)

    # %%
    f1.plot_ts_simple_linear_reg(y, y_pred)

# %% [markdown]
#     # alpha = a + b and l1_ratio = a / (a + b)

    # %%
    cv_a

    # %%
    A, B, to_min = f1.get_to_min(cv_a, cv_l1, dsf2, dm, PAR, tot_sd)

# %%


    # %% tags=[]
    res = minimize_scalar(to_min,method='Bounded',bounds=(1,300))

    dic = f1.do_something(A, B, PAR, dm, dsf2, res, tot_sd)

    # %%
    ddf = f1.get_ddf(dic)

# %%

# %%

    f1.plot_8(ddf)

    # %%
    f1.plot_9(dic)

    # %%
    ddf2 = f1.get_ddf2(ddf, dic)

    la, lr, rr = f1.do_a_loop(ddf2)

    # %%
    n_sd = pd.Series(lr)



    # %%
    X, X1, gam1, ffit = f1.do_some_GAM(n_sd)

# %%

    f1.plot_11_13(X, X1, ffit, gam1, n_sd)
    # axx.set_yscale('log')

# %%

    # %%
    _ds2 = f1.get_ds2(ds, ffit, la, rr)

    # %%
    f1.plot_15(_ds2)

# %%

# %%
