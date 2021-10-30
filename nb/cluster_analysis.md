- results are shown here
  - [cluster_analysis.ipynb](./nb/cluster_analysis.md)
  - we try maultple cluster groups [2, 3, 5, 9, 15, 20]
    - the number of cluster groups is always a topic of debate 
    - it depends on the complexity and specificity of the intended analysis 
    - also simplicity is important as simple results are easy to understand 
    - i suggest 5 clusters as a good compromise.  

region cluster analysis for the mosaic campaign 
- loosely based on the method described in
  - https://acp.copernicus.org/preprints/acp-2021-126/
- data and flexpart analysis obtained from 
  - https://srvx1.img.univie.ac.at/webdata/mosaic/mosaic.html


```python
%load_ext autoreload
%autoreload 2

# import matplotlib as mpl
# import matplotlib.colors
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
import xarray as xr

import funs as fu
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



## constants and functions


```python

PATH_TO_FILE = '/Users/aliaga/Downloads/AIRTRACER_100m.nc'

LA = 'lat'
LO = 'lon'
TI = 'time'
AG = 'age'
AT = 'AIRTRACER'
L = 'lab'
CM = 'tab20'
```


# code


```python


# def main():
```


```python

ds = xr.open_mfdataset(PATH_TO_FILE)
```


```python
# lets check lat and lon borders
for l in LO, LA:
    a = ds[l].diff(l).to_series().describe()
    print(f'{l}\n{a}\n\n')
```

    lon
    count    719.0
    mean       0.5
    std        0.0
    min        0.5
    25%        0.5
    50%        0.5
    75%        0.5
    max        0.5
    Name: lon, dtype: float64
    
    
    lat
    count    119.00
    mean       0.25
    std        0.00
    min        0.25
    25%        0.25
    50%        0.25
    75%        0.25
    max        0.25
    Name: lat, dtype: float64
    
    


## overall residence time
Lets start by plotting the the sum over time of the flexpart output


```python

# sum over time and age and load the file in memory
d1 = ds.sum([TI, AG])[AT]
d1.load()
# plot a sum over the campaign
fu.plot_map(d1)
```




    <cartopy.mpl.geocollection.GeoQuadMesh at 0x2a8656670>




    
![png](cluster_analysis_files/cluster_analysis_9_1.png)
    


## the ship's path
Lets identify the path taken by the boat. We can euristically determine this 
by using the maximum residence time of the flexpart output at each time step.


```python
# find the ship path on a euristic basis
d = ds[AT][{AG: 0}].load()
am = d.argmax(dim=[LO, LA])
df = d[am].to_dataframe()
fu.plot_path(df, LO, LA)
```

    /var/folders/xl/wx70j1c50bsdzdy63_wpbhrxpglmw2/T/ipykernel_82253/3423378844.py:70: UserWarning: Use the colorbar set_ticks() method instead.
      cb.ax.set_yticks(ii)
    /var/folders/xl/wx70j1c50bsdzdy63_wpbhrxpglmw2/T/ipykernel_82253/3423378844.py:74: UserWarning: FixedFormatter should only be used together with FixedLocator
      cb.ax.set_yticklabels(pd.to_datetime(cb.get_ticks()).strftime(date_format='%b %Y'));


    [1.570e+18 1.575e+18 1.580e+18 1.585e+18 1.590e+18 1.595e+18 1.600e+18]



    
![png](cluster_analysis_files/cluster_analysis_11_2.png)
    


## some distributions
Lets also explore the distributions of single 'pixels'.
They seem to be log distributions. 


```python
d2 = ds[AT].sum(AG).load()
fu.plot_some_distributions(d2, LA, LO)
```


    
![png](cluster_analysis_files/cluster_analysis_13_0.png)
    



```python

```


```python

# plot the residence times
fu.plot_residence_time(d2, LA, LO)
```


    
![png](cluster_analysis_files/cluster_analysis_15_0.png)
    



```python
# coarse the array to reduce number of data points and also transform the
# residence from seconds to days
d3 = d2.coarsen({LA: 5, LO: 10}, boundary='exact').sum() / 3600 / 24
# plot the coarse array
fu.plot_map(d3.sum(TI))
```




    <cartopy.mpl.geocollection.GeoQuadMesh at 0x292eab5b0>




    
![png](cluster_analysis_files/cluster_analysis_16_1.png)
    



```python
qta = fu.get_quantiles(d3, TI)
```


```python
for N in [2, 3, 5, 9, 15, 20]:
    d4 = fu.kmeans_cluster(N, qta, d3, L, TI, )

    lax = d4[L]
#     fu.plot_kmap(lax, N, CM)
    fu.plot_hatch(d4, L, N, d1, LA, LO, CM)
    fu.plot_cluster_bar(N, d4, L, CM)
    fu.plot_month_means(d4, L, CM, N)
    fu.plot_cluster_ts(N, d4, L, CM)
```

    /Users/aliaga/py-packs/flexpart-mosaic/nb/funs.py:337: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, ax = plt.subplots(subplot_kw=dic, dpi=150)



    
![png](cluster_analysis_files/cluster_analysis_18_1.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_2.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_3.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_4.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_5.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_6.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_7.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_8.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_9.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_10.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_11.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_12.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_13.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_14.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_15.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_16.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_17.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_18.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_19.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_20.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_21.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_22.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_23.png)
    



    
![png](cluster_analysis_files/cluster_analysis_18_24.png)
    



```python

```
