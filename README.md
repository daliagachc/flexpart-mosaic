# Results 
- results are shown here
  - [cluster_analysis.ipynb](./nb/cluster_analysis.md)
  - we try multiple cluster groups [2, 3, 5, 6, 9, 15, 20]
    - the number of cluster groups is always a topic of debate 
    - it depends on the complexity and specificity of the intended analysis 
    - also simplicity is important as simple results are easy to understand 
    - i suggest 5 clusters as a good compromise.  
# data output 
- timeseries values for each of the clusters is stored at
  - [data_out](./data_out)
  - each values is the residence time of the airtracers at the cluster in units of days
  - notice that the input data only considers the first 100 meters for the analysis. This means tha we only report the residence time for particles in the first layer of the model (100 m)

# references 
region cluster analysis for the mosaic campaign 
- loosely based on the method described in
  - https://acp.copernicus.org/preprints/acp-2021-126/
- data and flexpart analysis obtained from 
  - https://srvx1.img.univie.ac.at/webdata/mosaic/mosaic.html



