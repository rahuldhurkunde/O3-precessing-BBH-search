import numpy as np
import pycbc
from pycbc.types import TimeSeries, FrequencySeries
import gwpy
from gwpy.timeseries import TimeSeries as gwpyTimeSeries
from pycbc.waveform import get_fd_waveform
import h5py
import sys
from tqdm import tqdm

run = '../.'
data = np.loadtxt(f'{run}/all_comps.dat')

num_comps = np.full([len(data)], 5, dtype=int)
reverse_flag = np.zeros([len(data)], dtype=int)

for k in tqdm(range(len(data))):
    sum1 = 0.0
    sum_rev = 0.0
    for i in range(5):
        sum1 = data[k,i+1]
        sum_rev = data[k,i+6]
        if np.logical_or(sum1 > 0.97, sum_rev > 0.97):
            num_comps[k] = i+1
            if sum_rev > sum1:
                if sum_rev > 0.97:
                    reverse_flag[k] = 1
            break

filename = sys.argv[1]

h5fp = h5py.File(filename, 'a')

if 'num_comps' in h5fp.keys():
    print('Deleting num_comps')
    del h5fp['num_comps']
if 'reverse_flag' in h5fp.keys():
    print('Deleting reverse_flag')
    del h5fp['reverse_flag']

h5fp['num_comps'] = num_comps
h5fp['reverse_flag'] = reverse_flag
params = h5fp.attrs['parameters']
params = np.append(params, ['num_comps'])
params = np.append(params, ['reverse_flag'])
params = np.array(params, dtype=h5py.string_dtype(encoding='utf-8'))
del h5fp.attrs['parameters']
h5fp.attrs.create('parameters', params, dtype=h5py.string_dtype(encoding='utf-8'))
h5fp.close()
