results_dir = '/user/home/sh14603/Deep-music-classification/results'
import os
import glob
files = glob.glob(os.path.join(results_dir, '**', '*'), recursive=True)
import re
files = [f for f in files if re.search(r'val_accuracies_\w+_\w+_\d+.npy', f)]
import numpy as np
print(len(files))
from collections import defaultdict
accs = defaultdict(list)
for f in files:
    print(f)
    run, model, tag = re.search(r'(\d+)/val_accuracies_(\w+)_(\w+)_\d+.npy', f).groups()
    val_accuracies = np.load(f)
    print(run, model, val_accuracies[-1])
    accs[model+'_'+tag].append(val_accuracies[-1])

# print(dict(accs))
# print({k:len(v) for k,v in accs.items()})
for k,v in accs.items():
    print(k)
    print('n = ',len(v))
    print('val_acc =',np.mean(v))