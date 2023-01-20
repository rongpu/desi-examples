# Get PIXEL_NOBS_GRZ values for a catalog
# Examples:
# srun -N 1 -C haswell -c 64 -t 04:00:00 -q interactive python read_pixel_nexp.py --input catalog.fits --output catalog_nexp.fits
# srun -N 1 -C haswell -c 64 -t 04:00:00 -q interactive python read_pixel_nexp.py --input /global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits --output $CSCRATCH/temp/randoms-1-0-nexp.fits
# srun -N 1 -C haswell -c 64 -t 04:00:00 -q interactive python read_pixel_nexp.py --input /global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.0.0/resolve/dr9_lrg_south_1.0.0_basic.fits --output $CSCRATCH/temp/dr9_lrg_south_1.0.0_nexp.fits

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

from astropy.io import fits
from astropy import wcs

from multiprocessing import Pool
import argparse


time_start = time.time()

data_dir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9'

n_processes = 32

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

input_path = args.input
output_path = args.output

# input_path = '/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits'
# output_path = '/global/cscratch1/sd/rongpu/temp/randoms-1-0-lrgmask_v1.fits'

if os.path.isfile(output_path):
    raise ValueError(output_path+' already exists!')


def bitmask_radec(brickid, ra, dec):

    brick_index = np.where(bricks['BRICKID']==brickid)[0][0]

    brickname = str(bricks['BRICKNAME'][brick_index])
    if bricks['PHOTSYS'][brick_index]=='N':
        field = 'north'
    elif bricks['PHOTSYS'][brick_index]=='S':
        field = 'south'
    else:
        # Outside DR9 footprint; assign NEXP=0
        n_g, n_r, n_z = np.full((3, len(ra)), 0, dtype=np.int16)
        return n_g, n_r, n_z

    # bitmask_fn = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{}/coadd/{}/{}/legacysurvey-{}-maskbits.fits.fz'.format(field, brickname[:3], brickname, brickname)
    nexp_g_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-g.fits.fz'.format(field, brickname[:3], brickname, brickname))
    nexp_r_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-r.fits.fz'.format(field, brickname[:3], brickname, brickname))
    nexp_z_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-z.fits.fz'.format(field, brickname[:3], brickname, brickname))

    nexp_g = fitsio.read(nexp_g_fn)
    nexp_r = fitsio.read(nexp_r_fn)
    nexp_z = fitsio.read(nexp_z_fn)

    header = fits.open(nexp_g_fn)[1].header
    w = wcs.WCS(header)

    coadd_x, coadd_y = w.wcs_world2pix(ra, dec, 0)
    coadd_x, coadd_y = np.round(coadd_x).astype(int), np.round(coadd_y).astype(int)

    n_g = nexp_g[coadd_y, coadd_x]
    n_r = nexp_r[coadd_y, coadd_x]
    n_z = nexp_z[coadd_y, coadd_x]

    return n_g, n_r, n_z


def wrapper(bid_index):

    idx = bidorder[bidcnts[bid_index]:bidcnts[bid_index+1]]
    brickid = bid_unique[bid_index]

    ra, dec = cat['RA'][idx], cat['DEC'][idx]

    n_g, n_r, n_z = bitmask_radec(brickid, ra, dec)

    data = Table()
    data['idx'] = idx
    data['PIXEL_NOBS_G'] = n_g
    data['PIXEL_NOBS_R'] = n_r
    data['PIXEL_NOBS_Z'] = n_z

    return data


# bricks = Table(fitsio.read('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/survey-bricks.fits.gz'))
bricks = Table(fitsio.read('/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/randoms/survey-bricks-dr9-randoms-0.48.0.fits'))

try:
    cat = Table(fitsio.read(input_path, rows=None, columns=['RA', 'DEC', 'BRICKID']))
except ValueError:
    cat = Table(fitsio.read(input_path, rows=None, columns=['RA', 'DEC']))

print(len(cat))

for col in cat.colnames:
    cat.rename_column(col, col.upper())

if 'TARGET_RA' in cat.colnames:
    cat.rename_columns(['TARGET_RA', 'TARGET_DEC'], ['RA', 'DEC'])

if 'BRICKID' not in cat.colnames:
    from desiutil import brick
    tmp = brick.Bricks(bricksize=0.25)
    cat['BRICKID'] = tmp.brickid(cat['RA'], cat['DEC'])

# Just some tricks to speed up things up
bid_unique, bidcnts = np.unique(cat['BRICKID'], return_counts=True)
bidcnts = np.insert(bidcnts, 0, 0)
bidcnts = np.cumsum(bidcnts)
bidorder = np.argsort(cat['BRICKID'])

# start multiple worker processes
with Pool(processes=n_processes) as pool:
    res = pool.map(wrapper, np.arange(len(bid_unique)))

res = vstack(res)
res.sort('idx')
res.remove_column('idx')

res.write(output_path)

print('Done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
