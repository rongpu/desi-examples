# Combine random counts and systematics

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
# import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

import healpy as hp

min_nobs = 1

maskbits = []
# custom_mask_name = 'lrgmask_v1.1'
custom_mask_name = 'elgmask_v1'

mask_str = ''.join([str(tmp) for tmp in maskbits])
if custom_mask_name!='':
    mask_str += '_' + custom_mask_name

randoms_ver_str = '0.49.0'

randoms_counts_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/{}/resolve/counts'.format(randoms_ver_str)
randoms_systematics_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/{}/resolve/systematics'.format(randoms_ver_str)
stardens_dir = '/global/cfs/cdirs/desi/users/rongpu/useful/healpix_maps'

randoms_combined_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/{}/resolve/combined'.format(randoms_ver_str)

if not os.path.isdir(randoms_combined_dir):
    os.makedirs(randoms_combined_dir)

for nside in [64, 128, 256, 512]:

    stardens = np.load(os.path.join(stardens_dir, 'pixweight-dr7.1-0.22.0_stardens_{}_ring.npy'.format(nside)))

    for field in ['north', 'south']:

        output_path = os.path.join(randoms_combined_dir, 'pixmap_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, mask_str))
        if os.path.isfile(output_path):
            continue

        maps = Table(fitsio.read(os.path.join(randoms_counts_dir, 'counts_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, mask_str))))
        maps = maps[maps['n_randoms']>0]
        maps1 = Table(fitsio.read(os.path.join(randoms_systematics_dir, 'systematics_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, mask_str))))
        maps1.remove_columns(['RA', 'DEC'])
        maps = join(maps, maps1, join_type='inner', keys='HPXPIXEL')
        if not np.all(np.diff(maps['HPXPIXEL'])>0):
            raise ValueError

        maps['STARDENS'] = stardens[maps['HPXPIXEL']]

        maps.write(output_path)

