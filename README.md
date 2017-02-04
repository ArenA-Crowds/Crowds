[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.268639.svg)](https://doi.org/10.5281/zenodo.268639)

# Crowds

This repository contains the Jupyter notebook scripts that correspond to the data analysis and the methodology described in this draft [paper](https://github.com/sonjageorgievska/Arena/blob/master/PaperOnSmoothing/draft-31052016.pdf).

Because of the law for privacy preservation we are not allowed to publicly display the data, but provide a description of the data format [here](https://github.com/sonjageorgievska/Crowds/blob/master/density-estimation-version-2.md). An example of one line of the input data:

```json
{"measurementTimestamp":1436220156106,"value":{"sourceMac":"2d1ff804-c499-4163-b90f-003df1e4ec65","regionsNodesIds":[],"averagecoordinate":{"avg":{"coordinates":[2.47162,-13.851751,0.0],"type":"Point"},"error":{"coordinates":[8.52464,15.449013,1000.0],"type":"Point"}},"trackeeHistory":{"nMeasurements":8,"errState":{"sigmaY":15.449013,"sigmaX":8.52464,"sigmaP0":1.217258},"seqNr":3654,"chi2":29.534072,"fitStatus":"FITTED","state":{"y":-13.851751,"x":2.47162,"p0":-51.537968},"localMac":0,"nOutliers":5,"type":2,"probChi2":1.8E-5,"chi2PerDof":5.906814,"subType":0,"retryID":0}},"processingTimestamp":1436220164741}
```
We also provide most of the output generated from the original dataset in the .md files. 
