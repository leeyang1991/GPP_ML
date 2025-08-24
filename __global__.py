
import platform
import os
from os.path import join

os_version = platform.version()
centimeter_factor = 1 / 2.54
if 'Ubuntu' in os_version:
    # Dell
    this_root = '/home/yangli/HDD/GPP_ML/'
elif 'Darwin' in os_version:
    import matplotlib
    this_root = '/Volumes/HDD/GPP_ML/'
    matplotlib.use('TkAgg')
else:
    raise ValueError('os not recognized')
if not os.path.isdir(this_root):
    raise ValueError(f'working directory not found: {this_root}')

print('this_root:', this_root)
data_root = join(this_root, 'data')
results_root = join(this_root, 'results')
temp_root = join(this_root, 'temp')
conf_root = join(this_root, 'conf')