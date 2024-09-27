from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

tmp1 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_basic.fits'), columns=['TARGETID']))
tmp2 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_lrgmask_v1.1.fits.gz')))
lrgmask = hstack([tmp1, tmp2])
lrgmask.write('/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_lrgmask_v1.1_with_targetid.fits')

tmp1 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_basic.fits'), columns=['TARGETID']))
tmp2 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_elgmask_v1.fits.gz')))
elgmask = hstack([tmp1, tmp2])
elgmask.write('/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_elgmask_v1_with_targetid.fits')

