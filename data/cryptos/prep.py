import pandas as pd 
from dfply import X, arrange

def loadPriceIndex(dataFile):
    priceIndex = pd.read_csv(dataFile)
    priceIndex["Price"] = priceIndex.apply(lambda x: x.Price if isinstance(x.Price, float) else float(x.Price.replace(',', '')), axis=1)
    priceIndex["Date"] = priceIndex["Date"].astype("datetime64[ns]")
    priceIndex = priceIndex >> arrange(X.Date)
    
    return priceIndex

for i in range(1, 4):
    coins = loadPriceIndex('coin%d.csv'%i)
    coins.to_csv('coin%d.csv'%i)
