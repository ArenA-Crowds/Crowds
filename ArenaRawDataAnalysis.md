
Packet rate analysis + signal strengths plotting


```python

#reading from raw data
import fileinput
import json 
json_lines_raw = []
for line in fileinput.input(["F:/ArenaData/arena_raw_data/2015-07-05.json/2015-07-05_raw.json"]):
    jsline = json.loads(line)
    data = []
    data.append(jsline["value"]["sourceMac"])
    data.append(jsline["value"]["localMac"])
    data.append(jsline["measurementTimestamp"])
    data.append(jsline["value"]["droneId"])
    data.append(jsline["value"]["signal"])
    json_lines_raw.append(data)
print(json_lines_raw[0])
              
```

Finding the most overloaded drone (112)


```python
numberOfMeasurementsForDrone = dict()
for line in json_lines_raw:
    droneId = line[3]
    if droneId not in numberOfMeasurementsForDrone.keys():
        numberOfMeasurementsForDrone[droneId] = 0
    numberOfMeasurementsForDrone[droneId] +=1    
maxMeasurements = 0

maxDroneId = 0
for droneId in numberOfMeasurementsForDrone.keys():
    if numberOfMeasurementsForDrone[droneId] > maxMeasurements:
        maxMeasurements = numberOfMeasurementsForDrone[droneId]
        maxDroneId = droneId
print(maxDroneId)
print(maxMeasurements)
```

    112
    2920472
    

Computing the timestamps for every address visible at drone 112


```python
timestamps112 = dict()
for line in json_lines_raw:
    if line[3]=="112":
        address = line[0]
        if address not in timestamps112.keys():
            timestamps112[address] = []        
        timestamps112[address].append(line[2])
for address in timestamps112.keys():
    timestamps112[address].sort()  
```

Computing the timestamps for every address visible at any drone


```python

timestamps = dict()
minTimestamp = 2000000000000
for line in json_lines_raw:    
    address = line[0]
    if address not in timestamps.keys():
        timestamps[address] = []        
    timestamps[address].append(line[2])
for address in timestamps.keys():
    timestamps[address].sort()  
    if timestamps[address][0] < minTimestamp:
        minTimestamp = timestamps[address][0]
```


```python
# this cleans the json lines to make room in memory for the 'visible' dictionary
json_lines_raw = []
isVisible = dict()

#converts a timestamp from a ms to a second 
def getSecond(timestamp):
    return int(math.floor((timestamp- minTimestamp)/1000)) 

#converts all timestamps to seconds
import math 
for address in timestamps.keys():   
    for i in range (0, len(timestamps[address])-1):
        timestamps[address][i] = getSecond(timestamps[address][i])
                  

#removes the duplicates from the list of timestamps. The lists need to be sorted next
for address in timestamps.keys():   
    timestamps[address] = list(set(timestamps[address]))

#sort the sets 
for address in timestamps.keys():   
    timestamps[address].sort()
    
#for some reasons the last one remains in ms, therefore I get rid of it. I don't bother to check why. 
for address in timestamps.keys():  
    del timestamps[address][-1]

visible = dict()
for address in timestamps.keys(): 
    if len(timestamps[address]) > 0:
        rangee =  max(timestamps[address])-min(timestamps[address]) +1
        visible[address] =[0]* rangee
        for timestamp in timestamps[address]:
            visible[address][timestamp - min(timestamps[address]) ] = 1        
       
```


```python
import matplotlib.pyplot as plt

#plt.plot(visible['e1deda99-163f-4b68-9ade-e1f05d070bf2'], 'o')
plt.plot(visible['8b8a2356-d11e-4bd5-bb35-d8370bf48b1e'], 'o')

axes = plt.gca()
axes.set_ylim([-0.2,1.2])
plt.show()
```


```python
#plotting packet arrival rate at drone 112
import matplotlib.pyplot as plt
#myAddress = 'e1deda99-163f-4b68-9ade-e1f05d070bf2'#the largest one
myAddress = '8b8a2356-d11e-4bd5-bb35-d8370bf48b1e'#randomized
times112MyAddress = []
minTime = 1436047367297
for timestamp in timestamps112[myAddress]:
    time = (timestamp - minTime)/1000
    times112MyAddress.append(time)
plt.hist(times112MyAddress, bins = 2000)
plt.ylabel('number of packets')
plt.xlabel('time(sec)')
axes = plt.gca()
plt.show()

```


```python
#plotting packet arrival rate at any drone
myAddress = 'e1deda99-163f-4b68-9ade-e1f05d070bf2'#the largest one
#myAddress = '8b8a2356-d11e-4bd5-bb35-d8370bf48b1e'#randomized
timesMyAddress = []
minTime = 1436047367297
for timestamp in timestamps[myAddress]:
    time = (timestamp - minTime)/1000
    timesMyAddress.append(time)
plt.hist(timesMyAddress, bins = 2000)
plt.ylabel('number of packets')
plt.xlabel('time(sec)')
axes = plt.gca()
plt.show()
```


```python
#computing the interarrival times for every address
delays = dict()
for address in timestamps.keys():
    if address not in delays.keys():
        delays[address] = []
    oldTimeStamp = 0        
    for timestamp in timestamps[address]:
        if oldTimeStamp > 0:
            delay = (timestamp - oldTimeStamp)
            delays[address].append(delay)
        oldTimeStamp = timestamp
                            
```


```python
#computing the interarrival times for every address at drone 112
delays112 = dict()
for address in timestamps112.keys():
    if address not in delays112.keys():
        delays112[address] = []
    oldTimeStamp = 0        
    for timestamp in timestamps112[address]:
        if oldTimeStamp > 0:
            delay = (timestamp - oldTimeStamp)
            delays112[address].append(delay)
        oldTimeStamp = timestamp
```


```python
print(json_lines_raw[30000])
print(len(delays))
address1 = '32bf72d3-d707-4c37-b9b5-6318187de63f'
address2 = '3be779a3-1e1a-4b79-8ad0-c555f5124e7c'
print(localMac[address1])
print(localMac[address2])
print(delays[address1])
print(delays[address2])
```


```python
#print(json_lines_raw[30000])
#print(len(delays))
address1 = '32bf72d3-d707-4c37-b9b5-6318187de63f'
address2 = '3be779a3-1e1a-4b79-8ad0-c555f5124e7c'

print(delays112[address1])
print(delays112[address2])
```


```python
#computing statistics of packet interarrival times of all addresses
import numpy as np
averageDelay= dict()
medianDelay = dict()
stderrDelay = dict()
for address in delays.keys():
    if address not in averageDelay.keys():
        if len(delays[address]) > 1:
            delaysArray = np.array(delays[address])
            averageDelay[address] = np.mean(delaysArray)
            medianDelay[address] = np.median(delaysArray)
            stderrDelay[address] = np.std(delaysArray)
            
```


```python
#computing statistics of non-randomized addresses
averageDelay0= dict()
medianDelay0 = dict()
stderrDelay0 = dict()
for address in delays.keys():
    if localMac[address] == 0:
        if address not in averageDelay0.keys():
            if len(delays[address]) > 1:
                delaysArray = np.array(delays[address])
                averageDelay0[address] = np.mean(delaysArray)
                medianDelay0[address] = np.median(delaysArray)
                stderrDelay0[address] = np.std(delaysArray)
               
```


```python
#computing statistics of packet interarrival times of randomized addresses
averageDelay1= dict()
medianDelay1 = dict()
stderrDelay1 = dict()
for address in delays.keys():
    if localMac[address] == 1:
        if address not in averageDelay0.keys():
            if len(delays[address]) > 1:
                delaysArray = np.array(delays[address])
                averageDelay1[address] = np.mean(delaysArray)
                medianDelay1[address] = np.median(delaysArray)
                stderrDelay1[address] = np.std(delaysArray)
```


```python
#computing statistics of non-randomized addresses for drone 112 (most overloaded drone)
import numpy as np
averageDelay0_112= dict()
medianDelay0_112 = dict()
stderrDelay0_112 = dict()
for address in delays112.keys():
    if localMac[address] == 0:
        if address not in averageDelay0_112.keys():
            if len(delays112[address]) > 1:
                delaysArray = np.array(delays112[address])
                averageDelay0_112[address] = np.mean(delaysArray)
                medianDelay0_112[address] = np.median(delaysArray)
                stderrDelay0_112[address] = np.std(delaysArray)
                if len(delays112[address]) > 200:
                    print(address)
                if len(delays112[address])> 1000:    
                    print("long " +  address)
print('**********************************************************************')                
#computing statistics of randomized addresses for drone 112
averageDelay1_112= dict()
medianDelay1_112 = dict()
stderrDelay1_112 = dict()
for address in delays112.keys():
    if localMac[address] == 1:
        if address not in averageDelay0_112.keys():
            if len(delays112[address]) > 1:
                delaysArray = np.array(delays112[address])
                averageDelay1_112[address] = np.mean(delaysArray)
                medianDelay1_112[address] = np.median(delaysArray)
                stderrDelay1_112[address] = np.std(delaysArray)       
                if len(delays112[address]) > 200:
                    print(address)
                if len(delays112[address])> 1000:    
                    print("long " +  address)    
```


```python
#drawing the delays of some non-randomized address
import matplotlib.pyplot as plt
#plt.plot(delays112['62042a84-f904-4b2c-b79b-1e06c1ac980e'])
plt.plot(delays['e1deda99-163f-4b68-9ade-e1f05d070bf2'])
plt.plot()
plt.ylabel('packet interarrival time (ms) at any drone')
plt.xlabel('# packet from address e1deda99-163f-4b68-9ade-e1f05d070bf2 ')
#axes.set_ylim([0,200])
axes = plt.gca()

plt.show()
```


```python
#drawing the delays of some randomized address
import matplotlib.pyplot as plt
plt.plot(delays112['7a43a795-b538-4563-b702-6f0256588479'])
plt.ylabel('interarrival time (ms)')
plt.xlabel('packet')
#axes.set_ylim([0,200])
axes = plt.gca()
print(delays112['7a43a795-b538-4563-b702-6f0256588479'])
plt.show()
```


```python
print(delays112['443200e5-5752-4e5e-92f4-00af19a293be'])#a randomized address

```


```python
print(delays112['8b8a2356-d11e-4bd5-bb35-d8370bf48b1e'])# a randomized address
```


```python
arr = np.array(delays['32bf72d3-d707-4c37-b9b5-6318187de63f'])
```


```python
plt.hist(arr, bins = 500)
plt.ylabel('frequency')
plt.xlabel('delay')
axes.set_xlim([0,200000])

axes = plt.gca()

plt.show()
```


```python
#plotting histogram of average delay for non-randomized
averagedelaysList = []
for address in averageDelay0.keys():
    averagedelaysList.append(averageDelay[address])
averagedelays = np.array(averagedelaysList)
plt.hist(averagedelays, bins = 2000)
plt.ylabel('frequency')
plt.xlabel('average packet delay for non-randomized addresses')

#axes.set_xlim([0,200000])

axes = plt.gca()

plt.show()
```


```python
#plotting histogram of average delay for randomized
averagedelaysList = []
for address in averageDelay1.keys():
    averagedelaysList.append(averageDelay[address])
averagedelays = np.array(averagedelaysList)
plt.hist(averagedelays, bins = 2000)
plt.ylabel('frequency')
plt.xlabel('average packet delay for randomized addresses')

#axes.set_xlim([0,200000])

axes = plt.gca()

plt.show()
```


```python
def plotHistogramOfDictionary(dictionary, xlabel, ylabel, nbins):
    dictionaryList = []
    for address in dictionary.keys():
        dictionaryList.append(dictionary[address])
    dictArray = np.array(dictionaryList)
    plt.hist(dictArray, bins = nbins)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    axes = plt.gca()
    plt.show()
```


```python
plotHistogramOfDictionary(averageDelay0,  'average packet delay for non-randomized addresses ', 'freq', 2000 )
```


```python
plotHistogramOfDictionary(medianDelay0_112,  'median packet delay for non-randomized addresses for drone 112', 'freq', 2000 )
```


```python
plotHistogramOfDictionary(stderrDelay0_112, 'stdev packet delay for non-randomized addresses for drone 112', 'freq' 2000 )
```


```python
plt.hist(arr, bins = 500)
plt.ylabel('frequency')
plt.xlabel('delay')
axes.set_xlim([0,200000])

axes = plt.gca()

plt.show()
```


```python
#let us check the stability of the signal strengths
# line: ['32bf72d3-d707-4c37-b9b5-6318187de63f', 0, 1436047299132, '107', -68]
# line: [address, localmac, time, drone, signal]
def GetSignalStrengthPerDroneForAddress(address):
    signalsDictionary = dict()
    for line in json_lines_raw: 
        mac = line[0]
        time = line[2]
        drone = line[3]
        signal = line[4]
        if mac == address:            
            pair=[time, signal]
            if drone not in signalsDictionary.keys():
                signalsDictionary[drone] = []                
            signalsDictionary[drone].append(pair)
    return signalsDictionary
    

```


```python
signalsDict = GetSignalStrengthPerDroneForAddress('e1deda99-163f-4b68-9ade-e1f05d070bf2')
print(len(signalsDict))
print (signalsDict['112'])
```


```python
#drawing signal strengths for drone 126 and address e1deda99-163f-4b68-9ade-e1f05d070bf2
import numpy as np
import matplotlib.pyplot as plt

pairs = signalsDict['126']
xx = []
yy=[]
for pair in pairs:
    xx.append(pair[0])
    yy.append(pair[1])
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
ax0.errorbar(xx, yy, yerr=0, fmt='o')
ax0.set_title('signal')
plt.xlabel('timestamp(ms)')=

ax1.errorbar(xx, yy, yerr=0, fmt='o')
ax1.set_title('signal')

plt.show()
```


```python
print(signalsDict.keys())
```


```python
#computing the timestamps for every address visible at drone 122 -- on the other side of the stadium
timestamps122 = dict()
for line in json_lines_raw:
    if line[3]=="122":
        address = line[0]
        if address not in timestamps122.keys():
            timestamps122[address] = []        
        timestamps122[address].append(line[2])
for address in timestamps122.keys():
    timestamps122[address].sort() 
```


```python
for address in timestamps122.keys():
    if len(timestamps122[address]) > 1000:
        print(address + " " + str(localMac[address]))

```


```python

```
