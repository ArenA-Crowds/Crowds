
# Import and rebuild data set

Below we import the fitted positions data and build a Pandas DataFrame called 'frame'.
In 'frame' each row is a fitted position.
The columns contain the separate pieces of information accompanying each fit, such as the coordinates, 
timestamp, and uncertainty values.
The rows are ordered by timestamp. 


```python
# real dataset

import json

data = []
with open("F:/ArenaData/arena_fits/2015-07-05.json") as f:
    data = f.readlines()

json_lines = []

for line in data:
    jsline = json.loads(line)
    json_lines.append(jsline)
```


```python
import pandas as pd

frame = pd.DataFrame.from_dict(json_lines)
```


```python
# rebuild dataframe
# make dataframe of dicts nested in 'value' column
value = pd.DataFrame(list(frame['value']))
del frame['value']

# make dataframe of dicts nested in 'trackeeHistory' column
trackee = pd.DataFrame(list(value['trackeeHistory']))
del value['trackeeHistory']

chi2PerDof = pd.DataFrame(list(trackee['chi2PerDof']))
chi2PerDof.columns = ['chi2PerDof']
probChi2 = pd.DataFrame(list(trackee['probChi2']))
probChi2.columns = ['probChi2']
nMeasurements = pd.DataFrame(list(trackee['nMeasurements']))
nMeasurements.columns = ['nMeasurements']
localMac = pd.DataFrame(list(trackee['localMac']))
localMac.columns = ['localMac']
```


```python
# make dataframe with a 'coordinates' column
averagecoordinate = pd.DataFrame(list(value['averagecoordinate']))
coordinates = pd.DataFrame(list(averagecoordinate['avg']))
averagecoordinate = averagecoordinate.join(coordinates)
error = pd.DataFrame(list(averagecoordinate['error']))
errorcoordinates = pd.DataFrame(list(error['coordinates']))
del errorcoordinates[2]
errorcoordinates.columns = ['x_error','y_error']

del averagecoordinate['avg']
del value['averagecoordinate']

# join dataframes
frame = frame.join(value.join(averagecoordinate))
frame = frame.join(chi2PerDof)
frame = frame.join(probChi2)
frame = frame.join(errorcoordinates)
frame = frame.join(localMac)
frame = frame.join(nMeasurements)
del frame['regionsNodesIds']
del frame['error']
del frame['type']
```


```python
frame = frame.sort_values(by='measurementTimestamp')
```


```python
frame[:3]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>measurementTimestamp</th>
      <th>processingTimestamp</th>
      <th>sourceMac</th>
      <th>coordinates</th>
      <th>chi2PerDof</th>
      <th>probChi2</th>
      <th>x_error</th>
      <th>y_error</th>
      <th>localMac</th>
      <th>nMeasurements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>1436047367297</td>
      <td>1436047381262</td>
      <td>62e72aeb-2c41-44ed-88a6-3423267c1cb5</td>
      <td>[-7.459665, -75.723003, 0.0]</td>
      <td>0.101141</td>
      <td>0.750465</td>
      <td>3.838867</td>
      <td>0.298622</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>142</th>
      <td>1436047367330</td>
      <td>1436047381441</td>
      <td>7c824b77-a38b-4c3d-bb6a-3e40f5373d97</td>
      <td>[46.207027, -29.514564, 0.0]</td>
      <td>1.323366</td>
      <td>0.249989</td>
      <td>3.778767</td>
      <td>2.691939</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>150</th>
      <td>1436047367683</td>
      <td>1436047381455</td>
      <td>4ff97883-b9e8-4e2e-8b22-9bd093cfa401</td>
      <td>[-110.189892, -2.635529, 0.0]</td>
      <td>3.053348</td>
      <td>0.080571</td>
      <td>5.761471</td>
      <td>5.406795</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame = frame[frame['localMac'] == 0]
```

# Start data analysis code

The density estimation code works according to the following steps:
- First a time window is selected from the data
- The set of unique MAC addresses in the time window is determined
- A bunch of dictionairies are created which hold for each MAC (as key), the required values to construct
the density estimate, such as the fitted position coordinates, and the associated uncertainty values
- After the first time window, the dictionairies are recreated from the previous set, so the set of MAC addresses
can only expand
- The density estimates are calculated
- The density estimates are summed, to create the total crowd density estimate
- After each iteration, a deep copy is made of all the density estimates 
- If a MAC address is not detected in a new time window, the previous density estimate is smoothed (stored in the deep copy)
- The smoothing is only done if the history value associated with the MAC address does not exceed the memory parameter
- If the history value does exceed the memory parameter, the density estimate remains zero until the MAC is detected again

From here it is assumed that the data is stored in a Pandas DataFrame called 'frame'.
The function 'selectWindow' selects the part of the dataset with timestamps falling within the interval specified by variables 'start' and 'stop', and returns a DataFrame with the same structure as 'frame'.
Start and stop are specified by the iterator k and the parameters 'interval' and 'timestep'.
If timestep > interval, the time windows are non-overlapping.


```python
def selectWindow(k):
    start = min(frame['measurementTimestamp']) + k * timestep
    stop = start + interval

    window = frame[(frame['measurementTimestamp'] >= start) & 
                       (frame['measurementTimestamp'] < stop)]

    return window
```

The function createDataStructures returns a bunch of dictionairies for the MAC addresses detected in the 
selected time window, required to do the density estimation later on.
It creates a Python dictionairy called 'histos' with the MAC addresses as keys. Each address gets an empty grid (zeros), 
which is the two-dimensional probability distribution yet to be evaluated.
It creates a second dictionairy called 'positions' where each MAC address gets an empty list.
It creates two separate dictionairies for the uncertainty values in x and y direction,
where each MAC address gets an empty list.
It creates a dictionairy called 'history', which for each MAC keeps track of the time that has passed since the last update, given by the number of time windows.


```python
def createDataStructures(window):
    grids = np.zeros((len(set(window['sourceMac'])), height,width))

    # dictionary of histograms (with mac addresses as keys)
    histos = dict(zip(set(window['sourceMac']), grids))
    
    emptylist = [[] for i in range(len(set(window['sourceMac'])))]
    positions = dict(zip(set(window['sourceMac']), [[] for i in range(len(set(window['sourceMac'])))]))
    emptylist = [[] for i in range(len(set(window['sourceMac'])))]
    x_errors = dict(zip(set(window['sourceMac']), emptylist))
    emptylist = [[] for i in range(len(set(window['sourceMac'])))]
    y_errors = dict(zip(set(window['sourceMac']), emptylist))
    
    history = dict(zip(set(window['sourceMac']), np.zeros(len(set(window['sourceMac'])))))
    
    return histos, positions, x_errors, y_errors, history
```

In the function resetDataStructures all the dictionairies created in createDataStructures are reset: all the MAC addresses get an empty list again.
The dictionairy containing the calculated density estimates is deep copied, for possible smoothing.


```python
def resetDataStructures(histos):
    
    histos_old = copy.deepcopy(histos)
    
    grids = np.zeros((len(histos), height,width))
    histos = dict(zip(histos.keys(), grids))
    
    emptylist = [[] for i in range(len(histos))]
    positions = dict(zip(histos.keys(), emptylist))
    emptylist = [[] for i in range(len(histos))]
    x_errors = dict(zip(histos.keys(), emptylist))
    emptylist = [[] for i in range(len(histos))]
    y_errors = dict(zip(histos.keys(), emptylist))
    
    return histos, histos_old, positions, x_errors, y_errors
```

In the function updateDataStructures all the empty lists in the dictionairies created in createDataStructures are filled with values from the data in the selected time window.
If a MAC address is not yet in the dictionary, it is added.


```python
def updateDataStructures(window, histos, positions, x_errors, y_errors, history):
    for i in range(len(window)):
        if not window['sourceMac'].values[i] in positions:
            histos[window['sourceMac'].values[i]] = np.zeros((height,width))
            positions[window['sourceMac'].values[i]] = []
            x_errors[window['sourceMac'].values[i]] = []
            y_errors[window['sourceMac'].values[i]] = []
            history[window['sourceMac'].values[i]] = 0
            
        positions[window['sourceMac'].values[i]].append(window['coordinates'].values[i])
        x_errors[window['sourceMac'].values[i]].append(window['x_error'].values[i])
        y_errors[window['sourceMac'].values[i]].append(window['y_error'].values[i])
        
    return histos, positions, x_errors, y_errors, history
```

In the function createDensityEstimates all the actual magic happens.
We iterate over the MAC addresses, for each MAC address we iterate over the evaluation grid.
At each evaluation grid point, we iterate over the number of fitted positions (data points) found in the time window for that MAC address.
For each data point, we collect the x and y coordinates and the x and y uncertainty values,
and calculate the contribution we get from that data point by evaluating the kernel function.
The origin (0,0) of the grid is centered on the center of the football field.
The kernel function takes as arguments the x and y coordinate of the distance between the evaluation
grid point and the data point in meters.
If the errors are zero, we check whether the data point is in the same cell as the grid point we are evaluating.
If that is the case, the data point contributes 1 (unity) to our evaluation grid point.
We multiply (scale) by the cell area of our evalution grid to get probabilities per cell area (and not per square meter).
Finally, we normalize our density estimate by the value we get from integrating over the evaluation grid field (and omit normalizing by the number N of data points),
because we assume that the probability is unity that the mobile device is somewhere inside the Arena.


```python
def createDensityEstimates(window, histos, positions, x_errors, y_errors):

    for mac in histos.keys():
        if len(positions[mac]) > 0:          
            for u in range(width):
                for v in range(height):
                    for j in range(len(positions[mac])):
                        xi = positions[mac][j][0]
                        yi = positions[mac][j][1]
                        
                        x = u * cellsize - 120
                        y = v * cellsize - 90

                        sigma_x = x_errors[mac][j]
                        sigma_y = y_errors[mac][j]

                        if sigma_x > 0 and sigma_y > 0:
                            histos[mac][v][u] += cellsize**2 *\
                            kernel((x - xi), sigma_x) * kernel((y - yi), sigma_y)
                        else:
                            if abs((x - xi)) < cellsize/2. and abs((y - yi)) < cellsize/2.:
                                 histos[mac][v][u] = 1
                        
                    #histos[mac][v][u] /= len(positions[mac])
            if histos[mac].sum() > 0:
                histos[mac] /= histos[mac].sum()
    return histos
```

Kernel is the Gaussian kernel function, without normalization term, because the density estimates are normalized
by the value resulting from integrating over the evaluation grid.


```python
def kernel(x, sigma): 
    return exp(-(x**2)/(2*sigma**2))
```

In the function smoothNonUpdatedEstimates previously calculated density estimates are smoothed.
We first construct a two-dimensional Gaussian bump  of which the width (sigma) is based on by pedestrian walking speed.
The two-dimensional function is created using a scipy.stats library function.
If there were no detections for a MAC address in the time window, its density estimate from the previous time window 
(stored in a deep copy) is convoluted with the Gaussian bivariate bump, using a library function from the scipy signal processing module.
Each time this is done, the history value associated with the MAC address is incremented.
If the history value exceeds the memory parameter value, the density estimate remains zero.
The density estimate remains zero, untill the MAC is detected again.


```python
def smoothNonUpdatedEstimates(histos, histos_old, positions, history):
    
    # generate weighting function with dispersion set to 
    # Brownian motion with v = 0.5 m/s and t = interval time
    # diffusion constant D = (v^2)/2
    D = 0.5 # this assumes a walking speed of 0.71 m/s 
    t = interval / 1000
    sigma = sqrt(2*D*t) / cellsize
    
    var = multivariate_normal(mean=[width/2 - 1,height/2 - 1], cov=[[sigma**2,0],[0,sigma**2]])
    
    weights = np.zeros((height,width))
    for i in np.arange(width):
        for j in np.arange(height):
            weights[j][i] += var.pdf([i,j])
    
    for mac in histos.keys():
        if len(positions[mac]) == 0:
            if history[mac] < memory:
                # smooth existing pdf from previous time interval
                # apply a convolution

                conv = signal.convolve2d(histos_old[mac], weights, boundary='wrap', mode='same')
                
                histos[mac] += conv
                history[mac] += 1
            else:
                history[mac] = 0
    
    return histos
```

The sumHistograms function simply sums the histograms in the dictionairy 'histos', 
and returns the total density estimate in the form of a numpy array.


```python
def sumHistograms(histos):
    # total density histogram per period
    total_dens_histo = np.zeros((height, width))
    
    for mac in histos.keys():
        total_dens_histo += histos[mac]
                
    return total_dens_histo
```

This is the __main__ function. It runs all the steps, and writes the total density estimate to file.
It differentiates between the first and later iterations, in order to initialize the dictionairies,
and then only to reset them.


```python
def runDataAnalysis():
    
    for k in range(periods):
        window = selectWindow(k)
        
        if k < 1:
            histos, positions, x_errors, y_errors, history = createDataStructures(window)
            histos, positions, x_errors, y_errors, history =\
            updateDataStructures(window, histos, positions, x_errors, y_errors, history)
            histos = createDensityEstimates(window, histos, positions, x_errors, y_errors)
        else:
            histos, histos_old, positions, x_errors, y_errors = resetDataStructures(histos)
            histos, positions, x_errors, y_errors, history =\
            updateDataStructures(window, histos, positions, x_errors, y_errors, history)
            histos = createDensityEstimates(window, histos, positions, x_errors, y_errors)
            histos = smoothNonUpdatedEstimates(histos, histos_old, positions, history)
        
        total_dens_histo = sumHistograms(histos)
        
        #print(len(histos), total_dens_histo.sum())
        
        np.savetxt('output/dens_histo_%d.csv' %  k, total_dens_histo, delimiter=',')
        print('Time window:', k)      
```

In this cell the parameters are set to run runDataAnalysis. The variable 'cellsize' sets the distance between
grid points in the evaluation grid.
The variables height and width follow from dividing the size of the evaluation grid (240x180 meter), 
which is the rectangle containing the Arena stadium, by the cellsize.
The variable 'periods' sets the number of time windows to run the analysis.
Interval is the length of the time window (in milliseconds). Timestep is the amount of time the time window is moved at each iteration.
The memory variable determines the amount of time, measured by the number of time windows,
a density estimate is held in memory.


```python
from math import sqrt, pi, exp
import numpy as np
from scipy.stats import multivariate_normal
from scipy import signal
import copy

# cell size (bin size)
cellsize = 3;
# size of binned region (number of cells)
width = int(240/cellsize); height = int(180/cellsize)

# numbers of time intervals
periods = 60
timestep = 30000 # 10000
interval = 120000 # 30000
memory = 0

runDataAnalysis()
```

    Time window: 0
    Time window: 1
    Time window: 2
    Time window: 3
    Time window: 4
    Time window: 5
    Time window: 6
    Time window: 7
    Time window: 8
    Time window: 9
    Time window: 10
    Time window: 11
    Time window: 12
    Time window: 13
    Time window: 14
    Time window: 15
    Time window: 16
    Time window: 17
    Time window: 18
    Time window: 19
    Time window: 20
    Time window: 21
    Time window: 22
    Time window: 23
    Time window: 24
    Time window: 25
    Time window: 26
    Time window: 27
    Time window: 28
    Time window: 29
    Time window: 30
    Time window: 31
    Time window: 32
    Time window: 33
    Time window: 34
    Time window: 35
    Time window: 36
    Time window: 37
    Time window: 38
    Time window: 39
    Time window: 40
    Time window: 41
    Time window: 42
    Time window: 43
    Time window: 44
    Time window: 45
    Time window: 46
    Time window: 47
    Time window: 48
    Time window: 49
    Time window: 50
    Time window: 51
    Time window: 52
    Time window: 53
    Time window: 54
    Time window: 55
    Time window: 56
    Time window: 57
    Time window: 58
    Time window: 59
    

# Code for plotting


```python
# check maximum value for z-axis limit

from math import ceil

maxValue = 0

for i in range(periods):
    temp = np.loadtxt('output/brownian-smoothing-test_%d.csv' % i, delimiter=',').max()
    if temp > maxValue:
        maxValue = temp
        
#maxValue = ceil(maxValue)
```


```python
maxValue
```


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16,10))

#col = ['r', 'y', 'c', 'k', 'c','r'] * height * width
col = ['w','r','w','w','w','w'] * height * width
# colors = np.random.choice(col, height*width)

for k in range(periods):
    
    ax = fig.add_subplot(111, projection='3d')

    x_data, y_data = np.meshgrid( np.arange(width),
                                  np.arange(height)*(-1) )

    x_data = x_data.flatten()
    y_data = y_data.flatten()

    z_data = np.loadtxt('output/brownian-smoothing-test_%s.csv' % k, delimiter=',').flatten()
    #z_data = total_dens_histos[k].flatten()
    ax.set_zlim3d(0, maxValue)
    ax.bar3d( x_data,
              y_data,
              np.zeros(len(z_data)),
              1, 1, z_data, color=col) # 
    if k < 10:
        number = '000' + str(k)
    elif k > 9:
        number = '00' + str(k)
    elif k > 99:
        number = '0' + str(k)
    plt.savefig('output/brownian-smoothing-test-%s.png' % number)

#plt.show()
```


```python

```
