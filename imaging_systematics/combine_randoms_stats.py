# Combine random counts and systematics

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
# import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

import healpy as hp

min_nobs = 2

# maskbits_list = [sorted([1, 8, 9, 11, 12, 13]), sorted([1, 11, 12, 13]), sorted([1, 13])]

maskbits_list = [[]]
apply_lrgmask = True

if apply_lrgmask:
    lrgmask_str = '_lrgmask_v1'
else:
    lrgmask_str = ''

randoms_ver_str = '0.49.0'

randoms_counts_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/{}/resolve/counts'.format(randoms_ver_str)
randoms_systematics_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/{}/resolve/systematics'.format(randoms_ver_str)
stardens_dir = '/global/cfs/cdirs/desi/users/rongpu/useful/healpix_maps'

randoms_combined_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/{}/resolve/combined'.format(randoms_ver_str)

if not os.path.isdir(randoms_combined_dir):
    os.makedirs(randoms_combined_dir)

for maskbits in maskbits_list:

    for nside in [64, 128, 256, 512, 1024]:

        stardens = np.load(os.path.join(stardens_dir, 'pixweight-dr7.1-0.22.0_stardens_{}_ring.npy'.format(nside)))

        for field in ['north', 'south']:

            output_path = os.path.join(randoms_combined_dir, 'pixmap_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str))
            if os.path.isfile(output_path):
                continue

            maps = Table.read(os.path.join(randoms_counts_dir, 'counts_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str)))
            maps = maps[maps['n_randoms']>0]
            maps1 = Table.read(os.path.join(randoms_systematics_dir, 'systematics_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str)))
            maps1.remove_columns(['RA', 'DEC'])
            maps = join(maps, maps1, join_type='inner', keys='HPXPIXEL')
            if not np.all(np.diff(maps['HPXPIXEL'])>0):
                raise ValueError

            maps['STARDENS'] = stardens[maps['HPXPIXEL']]
            
            maps.write(output_path)

