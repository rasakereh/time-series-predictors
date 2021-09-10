import pandas as pd 
from dfply import X, arrange, mutate, drop
import numpy as np

from os import listdir
from os.path import isfile

csvFiles = [f for f in listdir('.') if isfile(f) and f[-4:] == '.csv']

def loadPriceIndex(dataFile):
    priceIndex = pd.read_csv(dataFile) >> arrange(X.Date)
    dates = priceIndex['Date'].values
    deltas = (dates[1:] - dates[:-1])/1000
        
    return deltas


allDeltas = []

for csvFile in csvFiles:
    print(csvFile, end=':\t\t')
    try:
        deltas = loadPriceIndex(csvFile)
        pIndex = (np.min(deltas), np.max(deltas), np.median(deltas), np.mean(deltas), np.std(deltas), (np.sum(deltas) / 3600 / 24), '%d,%d'%(deltas.shape[0] // 1000, deltas.shape[0] % 1000))
        print(pIndex)
        allDeltas.append(np.log(deltas + 1))
    except:
        print('Failed')
    
import matplotlib.pyplot as plt
plt.boxplot(allDeltas)
plt.show()
