from os import listdir
from os.path import join, isfile
import json
import numpy as np

dataDir = '../dumps'
dataFiles = {f: join(dataDir, f)  for f in listdir(dataDir) if isfile(join(dataDir, f)) and f[-4:] == '.dmp'}
# print(list(dataFiles.keys()))

for f in dataFiles:
    dump = open(dataFiles[f], 'r')
    res = json.load(dump)
    print('============ %s ============'%(f))
    loss = res[1]
    loss_funcs = list(loss[list(loss.keys())[0]].keys())
    loss_vals = {loss_func: [] for loss_func in loss_funcs}
    for loss_func in loss_funcs:
        for capital in loss:
            if not np.isnan(loss[capital][loss_func]):
                loss_vals[loss_func].append(loss[capital][loss_func])
        print('''loss: %s
        (mean, sd, n#)
        (%f, %f, %d)'''%(loss_func, np.mean(loss_vals[loss_func]), np.std(loss_vals[loss_func]), len(loss_vals[loss_func])))
    print()
    
    
