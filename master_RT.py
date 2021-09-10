import numpy as np
import pandas as pd
from dfply import *
from pprint import pprint
from os import listdir
from os.path import isfile, join
from time import time
import datetime
import matplotlib.pyplot as plt
from src.utils import rolling_window, NumpyEncoder
from src.GBL import GBLM
from src.MLShepard import MLShepard
from src.MondrianForest import MondrianForest
from src.OARIMA import OARIMA
from src.OSVR import OSVR
from src.RandConLSTM import RandConLSTM
from src.WHLR import WHLR

import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 100)

TRAIN_PORTION = .8

DIM =  200
# DIM = 15

# Uncomment the method and its parameters to include the corresponding result
methods = {
    'GBLM': {
        'class': GBLM,
        'options': {
            'dimension': DIM,
            'epsilon': 5e-3,
            'forgetting_rate': .59,
            'p_learning_rate': .008,
            's_learning_rate': .001,
            'decay_rate': .25,
            'oe_penalty': -1.5,
            'ue_penalty': -1.5,
            'reward': 1,
            'epochs': 1
        }
    },
    # 'MLShepard': {
    #     'class': MLShepard,
    #     'options': {
    #         'future_scope': 3,
    #         'dimension': DIM,
    #         'minor_days': 3,
    #         'trust_treshold': 4,
    #         'max_point_usage': 5,
    #         'avr_elemwise_dist': 0.04,
    #         'epsilon': 1e-10
    #     }
    # },
    # 'OARIMA (ogd)': {
    #     'class': OARIMA,
    #     'options': {
    #         'dimension': DIM,
    #         'lrate': 1e-2,
    #         'epsilon': 1e-10,
    #         'method': 'ogd'
    #     }
    # },
    # 'OARIMA (ons)': {
    #     'class': OARIMA,
    #     'options': {
    #         'dimension': DIM,
    #         'lrate': 1e-2,
    #         'epsilon': 1e-10,
    #         'method': 'ons'
    #     }
    # },
    # 'OSVR': {
    #     'class': OSVR,
    #     'options': {
    #         'future_scope': 3,
    #         'dimension': DIM,
    #         'C': 10,
    #         'kernelParam': 30,
    #         'epsilon': 1e-10
    #     }
    # }, RUNNING TIME IS: [ 9.84e-002, -3.39e-003,  2.63e-005,  5.94e-007] @ [n, n**2, n**3, n**4]
    # 'LSTM': {
    #     'class': RandConLSTM,
    #     'options': {
    #         'future_scope': 3,
    #         'dimension': DIM,
    #         'epochs': 2,
    #         'batch_size': 128,
    #         'num_layers': 1,
    #         'epsilon': 1e-10,
    #         'hidden_size': 100,
    #         'connectivity': 1
    #     }
    # },
    # 'RandConLSTM': {
    #     'class': RandConLSTM,
    #     'options': {
    #         'future_scope': 3,
    #         'dimension': DIM,
    #         'epochs': 2,
    #         'batch_size': 128,
    #         'num_layers': 1,
    #         'epsilon': 1e-10,
    #         'hidden_size': 100,
    #         'connectivity': .2
    #     }
    # },
    # 'WHLR': {
    #     'class': WHLR,
    #     'options': {
    #         'future_scope': 3,
    #         'dimension': DIM,
    #         'avr_elemwise_dist': 0.04,
    #         'learning_rate': 1e-2
    #     }
    # },
    # 'MondrianForest': {
    #     'class': MondrianForest,
    #     'options': {
    #         'future_scope': 3,
    #         'dimension': DIM
    #     }
    # },
}

print('Preparing dataset...')
# Here is the data directory. Each stock/crypto must be stored in a seperated csv file
dataDir = 'data/RTstocks'
file_name = '25336820825905643.csv'
dataFiles = {f: join(dataDir, f)  for f in [file_name] if isfile(join(dataDir, f)) and f[-4:] == '.csv' and f not in ['stock_metadata.csv', 'NIFTY50_all.csv']}
# print(list(dataFiles.keys()))
priceIndices = {f: pd.read_csv(dataFiles[f]) for f in dataFiles}

# dataFiles = {'dummy1': 1, 'dummy2': 1, 'dummy3': 1, 'dummy4': 1, 'dummy5': 1, 'dummy6': 1}
# T_SIZE = 3000
# priceIndices = {
#     f: pd.DataFrame({
#         'Date': list(range(T_SIZE)),
#         'Price': np.random.normal(
#             np.random.uniform(70, 300),
#             np.random.uniform(1, 1.5),
#             (T_SIZE,)
#         )
#     }) for f in dataFiles
# }

prices = {}
pricePartitions = {'train': {}, 'test': {}}
trueVals = {}
intervalLength = float('Inf')
# intervalLength = 0

for cryptoID in priceIndices:
    priceIndices[cryptoID].fillna(method='ffill')
    priceIndices[cryptoID]["Date"] = priceIndices[cryptoID]["Date"].astype("datetime64[ns]")
    priceIndices[cryptoID] = priceIndices[cryptoID] >> arrange(X.Date)
    indexLength = priceIndices[cryptoID].shape[0]
    print('============%d================'%indexLength)
    indexMean = mean(priceIndices[cryptoID]["Price"].values)
    prices[cryptoID] = priceIndices[cryptoID]["Price"].values + np.random.normal(loc=0, scale=indexMean/500, size=indexLength)
    intervalLength = min(indexLength, intervalLength)
    # intervalLength = min(2000, intervalLength)

cutOff = int(intervalLength * TRAIN_PORTION)

for cryptoID in priceIndices:
    # if intervalLength != prices[cryptoID].shape[0]:
    #     prices[cryptoID] = np.concatenate((
    #         prices[cryptoID],
    #         np.repeat(prices[cryptoID][-1], intervalLength - prices[cryptoID].shape[0])
    #     ))
    
    pricePartitions['train'][cryptoID] = prices[cryptoID][:cutOff]
    pricePartitions['test'][cryptoID] = rolling_window(prices[cryptoID][cutOff:intervalLength], (DIM+1))[:-1]
    trueVals[cryptoID] = prices[cryptoID][cutOff:intervalLength][(DIM+1):]


MSE = lambda truth, estimate, _prices: np.sqrt(np.mean((truth-estimate)**2))
PMSE = lambda truth, estimate, _prices: np.sqrt(np.mean(((truth-estimate)/truth)**2))
PASE = lambda truth, estimate, _prices: np.mean((np.abs(truth-estimate)/truth))
DMSE = lambda truth, estimate, prices: np.sqrt(np.mean((np.heaviside(-(truth - prices[:,-1])*(estimate - prices[:,-1]), [0]) * (truth-estimate)/truth)**2))
wrongs = lambda truth, estimate, prices: np.sqrt(np.mean(np.heaviside(-(truth - prices[:,-1])*(estimate - prices[:,-1]), [0])))
# DMSESD = lambda truth, estimate, prices: np.sqrt(np.std((np.heaviside(-(truth - prices[:,-1])*(estimate - prices[:,-1]), [0]) * (truth-estimate)/truth)**2))
# DMSE = lambda truth, estimate, prices: print(*[truth, estimate, prices], sep='\n')

# methods['MondrianForest']['later_values'] = {'X': pricePartitions['test'], 'f': trueVals}
import json
for method_name in methods:
    print("==================== %s ===================="%(method_name))
    method = methods[method_name]
    pClass, options = method['class'], method['options']
    model = pClass(**options)

    print('Fitting model...')
    startTime = time()
    model.fit({f: pricePartitions['train'][f] for f in dataFiles})
    fittedTime = time()

    print('Predicting values...')
    predStartTime = time()
    res = model.predict(pricePartitions['test'], update=True, true_values=trueVals,
        loss_functions={'MSE': MSE, 'PMSE': PMSE, 'PASE': PASE, 'DMSE': DMSE, 'wrongs': wrongs})
    finishedTime = time()

    pprint({coin: {l: np.mean(res[1][coin][l]) for l in res[1][coin]} for coin in res[1]})

    print('Plotting results...')
    indices = np.random.choice(list(dataFiles.keys()), 1, False)
    plt.plot(range((DIM+1)+cutOff, (DIM+1)+cutOff+res[0][indices[0]].shape[0]), res[0][indices[0]])
    plt.plot(range(prices[indices[0]].shape[0]), prices[indices[0]])

    learnT = (fittedTime - startTime) * 1000
    predT = (finishedTime - predStartTime) * 1000
    avrPredT = (finishedTime - predStartTime) / (intervalLength-cutOff) * 1000
    totalT = learnT + predT
    timingString = '''
        learning time:\t%.1f ms
        predicting time:\t%.1f ms
        prediction/test:\t%.1f ms
        total time:\t%.1fms
    '''%(learnT, predT, avrPredT, totalT)
    print(timingString)
    
    print('saving dump...')
    currentTime = datetime.datetime.now()
    dump_file = open('dumps/Results-%s-%s_RT.dmp'%(method_name, currentTime), 'w')
    json.dump(res, dump_file, cls=NumpyEncoder)
    dump_file.close()
    dump_file = open('dumps/Timing-%s-%s_RT.txt'%(method_name, currentTime), 'w')
    dump_file.write(timingString)
    dump_file.close()
    


plt.show()

