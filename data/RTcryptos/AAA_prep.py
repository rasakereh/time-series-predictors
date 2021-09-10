import pandas as pd 
from dfply import X, arrange, mutate, drop

from os import listdir
from os.path import isfile

csvFiles = [f for f in listdir('.') if isfile(f) and f[-4:] == '.csv']

header = 'id|Price|volume|timestamp|buy\n'

def loadPriceIndex(dataFile):
    original = open(dataFile, 'r')
    data = header + original.read()
    original.close()
    modified = open(dataFile, 'w')
    modified.write(data)
    modified.close()
    priceIndex = pd.read_csv(dataFile, sep='|')
    # priceIndex['Date'] = pd.to_datetime(priceIndex['Date'])
    priceIndex = priceIndex >> mutate(Date = priceIndex.timestamp) >> drop(X.timestamp) >> arrange(X.Date)
    
    return priceIndex

for csvFile in csvFiles:
    print(csvFile, end=':\t\t')
    try:
        pIndex = loadPriceIndex(csvFile)
        pIndex.to_csv(csvFile)
        print('Done')
    except:
        print('Failed')
