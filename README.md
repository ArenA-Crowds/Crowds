[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.268639.svg)](https://doi.org/10.5281/zenodo.268639)

# Crowds

This repository contains the Jupyter notebook scripts together with output that correspond to the data analysis and the methodology for estimating crowd density based on WiFi positioning.

Because of the law for privacy protection we are not allowed to publicly display the data (we can only display aggregated results). 
An example of one line of the "fitted" input data for ArenaDataAnalysis.ipynb and density_estimation.ipynb :

```json
{
    "measurementTimestamp": 1436025068309,
    "value": {
        "sourceMac": "6cab229a-e15d-48b7-ab1e-5ddca3d7e283",
        "regionsNodesIds": [],
        "averagecoordinate": {
            "avg": {
                "coordinates": [253.417258, -224.352813, 0.0],
                "type": "Point"
            },
            "error": {
                "coordinates": [645.814525, 342.331268, 1000.0],
                "type": "Point"
            }
        },
        "trackeeHistory": {
            "nMeasurements": 4,
            "errState": {
                "sigmaY": 342.331268,
                "sigmaX": 645.814525,
                "sigmaP0": 23.142437
            },
            "seqNr": 527,
            "chi2": 0.543771,
            "fitStatus": "FITTED",
            "state": {
                "y": -224.352813,
                "x": 253.417258,
                "p0": -39.64037
            },
            "localMac": 0,
            "nOutliers": 0,
            "type": 0,
            "probChi2": 0.460874,
            "chi2PerDof": 0.543771,
            "subType": 4,
            "retryID": 0
        }
    },
    "processingTimestamp": 1436025078040
    }
```

Example of one line of the raw input data for ArenaRawDataAnalysis.ipynb:
```json
{
    "measurementTimestamp": 1436220095136,
    "value": {
        "typeNr": 0,
        "seqNr": 772,
        "droneId": "117",
        "sourceMac": "d41181a2-d8a0-45d3-a145-58ef960d778f",
        "localMac": 0,
        "signal": -86,
        "subTypeNr": 4,
        "retryFlag": 0
    },
    "processingTimestamp": 1436220098267
    }
```

