# Create combined main cumulative LRG catalog
# For Eddie's email "Redshift success rates & speed, redux"

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits


coadd_type = 'cumulative'

tiles = Table.read('/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-specstatus.ecsv')
print(len(tiles), len(np.unique(tiles['TILEID'])))

mask = tiles['SURVEY']=='main'
mask &= tiles['FAPRGRM']=='dark'
tiles = tiles[mask]
print(len(tiles))

# mask = tiles['OBSSTATUS']=='obsend'
mask = tiles['QA']=='good'
tiles = tiles[mask]
print(len(tiles))

mask = tiles['LASTNIGHT']>20210801  # post shutdown
tiles = tiles[mask]
print(len(tiles))

columns_1 = ['TARGETID', 'CHI2', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'DELTACHI2']
columns_2 = ['TARGETID', 'PETAL_LOC', 'DEVICE_LOC', 'LOCATION', 'FIBER', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'MORPHTYPE', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'PARALLAX', 'EBV', 'FLUX_W1', 'FLUX_W2', 'FIBERFLUX_Z', 'MASKBITS', 'PHOTSYS', 'DESI_TARGET', 'BGS_TARGET', 'TILEID', 'COADD_NUMEXP', 'COADD_EXPTIME', 'COADD_NUMNIGHT', 'COADD_NUMTILE', 'FIBERASSIGN_X', 'FIBERASSIGN_Y']
columns_4 = ['TARGETID', 'TSNR2_ELG', 'TSNR2_BGS', 'TSNR2_QSO', 'TSNR2_LRG']

tileid_list = tiles['TILEID']

data_dir = '/global/cfs/cdirs/desi/spectro/redux/daily/tiles/{}'.format(coadd_type)

cat_stack = []

for tileid in tileid_list:

    print(tileid)

    fn_list = sorted(glob.glob(os.path.join(data_dir, str(tileid), '*/redrock-*.fits')))
    for fn in fn_list:
        tmp1 = Table(fitsio.read(fn, ext=1, columns=columns_1))
        tmp2 = Table(fitsio.read(fn, ext=2, columns=columns_2))
        tmp4 = Table(fitsio.read(fn, ext=4, columns=columns_4))
        if not (np.all(tmp1['TARGETID']==tmp2['TARGETID']) and np.all(tmp1['TARGETID']==tmp4['TARGETID'])):
            raise ValueError
        tmp = tmp1.copy()
        tmp = join(tmp, tmp2, keys='TARGETID')
        tmp = join(tmp, tmp4, keys='TARGETID')
        
        mask = tmp['DESI_TARGET'] & 2**0 > 0
        tmp = tmp[mask]
        
        cat_stack.append(tmp)

cat = vstack(cat_stack)
print()
print(len(cat))

############################ Add LRG mask ############################

main_dir = '/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.0.0/resolve'

lrg = []
for field in ['north', 'south']:
    tmp = Table(fitsio.read(os.path.join(main_dir, 'dr9_lrg_{}_1.0.0_basic.fits'.format(field)), columns=['TARGETID']))
    tmp1 = Table(fitsio.read(os.path.join(main_dir, 'dr9_lrg_{}_1.0.0_lrgmask_v1.fits'.format(field))))
    lrg.append(hstack([tmp, tmp1]))
lrg = vstack(lrg)

mask = np.in1d(lrg['TARGETID'], cat['TARGETID'])
lrg = lrg[mask]

cat = join(cat, lrg, keys='TARGETID')

#############################################################################

cat.write('/global/cfs/cdirs/desi/users/rongpu/spectro/daily/main_done_{}_lrg_good_20211027.fits'.format(coadd_type), overwrite=False)

