# Results 
- results are shown here
  - [cluster_analysis.ipynb](./nb/z010_cluster_analysis.md)
  - we try multiple cluster groups [2, 3, 5, 6, 9, 15, 20]
    - the number of cluster groups is always a topic of debate 
    - it depends on the complexity and specificity of the intended analysis 
    - also simplicity is important as simple results are easy to understand 
    - i suggest 5 clusters as a good compromise.  
- source region identification with inverse modeling for SA, MSA ans IA
  - results are shown here
    - [z020_inverse_modeling.ipynb](./nb/z020_inverse_modeling.md)
  - couple of assumptions that in reality dont hold: 
    - we assume assumptions from each region are constant in time
      - this seems to hold for SA and MSA in this case (based on results)
      - but it does not for IA and therefore "bad region identification"
  - It is super nice how for
    - SA we identify the russian big so2 smelters 
    - MSA almost all of the source regions are oceanic
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



