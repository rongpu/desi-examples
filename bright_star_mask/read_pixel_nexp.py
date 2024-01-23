# Get PIXEL_NOBS_GRZ values for a catalog
# Examples:
# srun -N 1 -C cpu -c 256 -t 04:00:00 -q interactive python read_pixel_nexp.py --dr 9 --input catalog.fits --output catalog_nexp.fits
# srun -N 1 -C cpu -c 256 -t 04:00:00 -q interactive python read_pixel_nexp.py --dr 9 --input /global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-0.fits --output $CSCRATCH/temp/randoms-1-0-nexp.fits
# srun -N 1 -C cpu -c 256 -t 04:00:00 -q interactive python read_pixel_nexp.py --dr 9 --input /global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.0.0/resolve/dr9_lrg_south_1.0.0_basic.fits --output $CSCRATCH/temp/dr9_lrg_south_1.0.0_nexp.fits

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

n_processes = 256

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('--dr', default='9')
args = parser.parse_args()

data_dir = '/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr'+args.dr

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
        # Outside the survey footprint; assign NEXP=0
        if float(args.dr)>=10:
            n_g, n_r, n_i, n_z = np.full((4, len(ra)), 0, dtype=np.int16)
            return n_g, n_r, n_i, n_z
        else:
            n_g, n_r, n_z = np.full((3, len(ra)), 0, dtype=np.int16)
            return n_g, n_r, n_z

    # bitmask_fn = '/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr{}/{}/coadd/{}/{}/legacysurvey-{}-maskbits.fits.fz'.format(dr, field, brickname[:3], brickname, brickname)
    # Example: /dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/coadd/196/1963p287
    nexp_g_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-g.fits.fz'.format(field, brickname[:3], brickname, brickname))
    nexp_r_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-r.fits.fz'.format(field, brickname[:3], brickname, brickname))
    nexp_z_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-z.fits.fz'.format(field, brickname[:3], brickname, brickname))
    if float(args.dr)>=10:
        nexp_i_fn = os.path.join(data_dir, '{}/coadd/{}/{}/legacysurvey-{}-nexp-i.fits.fz'.format(field, brickname[:3], brickname, brickname))

    if os.path.isfile(nexp_g_fn):
        nexp_g = fitsio.read(nexp_g_fn)
        nexp_good_fn = nexp_g_fn
    else:
        n_g = np.full(len(ra), 0, dtype=np.int16)

    if os.path.isfile(nexp_r_fn):
        nexp_r = fitsio.read(nexp_r_fn)
        nexp_good_fn = nexp_r_fn
    else:
        n_r = np.full(len(ra), 0, dtype=np.int16)

    if os.path.isfile(nexp_z_fn):
        nexp_z = fitsio.read(nexp_z_fn)
        nexp_good_fn = nexp_z_fn
    else:
        n_z = np.full(len(ra), 0, dtype=np.int16)

    if float(args.dr)>=10:
        if os.path.isfile(nexp_i_fn):
            nexp_i = fitsio.read(nexp_i_fn)
            nexp_good_fn = nexp_i_fn
        else:
            n_i = np.full(len(ra), 0, dtype=np.int16)

    header = fits.open(nexp_good_fn)[1].header
    w = wcs.WCS(header)

    coadd_x, coadd_y = w.wcs_world2pix(ra, dec, 0)
    coadd_x, coadd_y = np.round(coadd_x).astype(int), np.round(coadd_y).astype(int)

    if 'n_g' not in locals():
        n_g = nexp_g[coadd_y, coadd_x]
    if 'n_r' not in locals():
        n_r = nexp_r[coadd_y, coadd_x]
    if 'n_z' not in locals():
        n_z = nexp_z[coadd_y, coadd_x]
    if float(args.dr)>=10:
        if 'n_i' not in locals():
            n_i = nexp_i[coadd_y, coadd_x]

    if float(args.dr)>=10:
        return n_g, n_r, n_i, n_z
    else:
        return n_g, n_r, n_z


def wrapper(bid_index):

    idx = bidorder[bidcnts[bid_index]:bidcnts[bid_index+1]]
    brickid = bid_unique[bid_index]

    ra, dec = cat['RA'][idx], cat['DEC'][idx]

    if float(args.dr)>=10:
        n_g, n_r, n_i, n_z = bitmask_radec(brickid, ra, dec)
    else:
        n_g, n_r, n_z = bitmask_radec(brickid, ra, dec)

    data = Table()
    data['idx'] = idx
    data['PIXEL_NOBS_G'] = n_g
    data['PIXEL_NOBS_R'] = n_r
    if float(args.dr)>=10:
        data['PIXEL_NOBS_I'] = n_i
    data['PIXEL_NOBS_Z'] = n_z

    return data


# bricks = Table(fitsio.read('/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/survey-bricks.fits.gz'))
if args.dr=='10':
    bricks = Table(fitsio.read('/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr10/randoms/survey-bricks-dr10-randoms-2.6.0.fits'))
elif args.dr=='9':
    bricks = Table(fitsio.read('/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/randoms/survey-bricks-dr9-randoms-0.48.0.fits'))
else:
    raise ValueError('survey-bricks path for DR{} not specified'.format(args.dr))

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
