# Add tractor columns to an LS catalog
# This script only works for DR9

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack
import fitsio

from multiprocessing import Pool

from desitarget.targets import decode_targetid, encode_targetid

n_processes = 128

cat_basic_path = '/dvs_ro/cfs/cdirs/desicollab/users/rongpu/targets/dr9.0/zp_offset_corrected/dr9_bgs_basic.fits'
output_path = '/pscratch/sd/r/rongpu/tmp/dr9_bgs_apflux_r.fits'

# Add these columns from tractor catalogs
tractor_columns = ['apflux_r', 'apflux_ivar_r']

cat_basic = Table(fitsio.read(cat_basic_path, columns=['TARGETID']))

objid_list, brickid_list, release_list = decode_targetid(cat_basic['TARGETID'])[:3]
cat_basic['PHOTSYS'] = ' '
mask = (release_list==9010) | (release_list==9012)
cat_basic['PHOTSYS'][mask] = 'S'
cat_basic['PHOTSYS'][~mask] = 'N'
assert np.sum(cat_basic['PHOTSYS']==' ')==0

bricks = Table.read('/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/survey-bricks.fits.gz')


print('Start!')
time_start = time.time()

cat_stack = []

for field in ['north', 'south']:

    if field=='south':
        photsys = 'S'
    elif field=='north':
        photsys = 'N'

    mask = cat_basic['PHOTSYS']==photsys
    if np.sum(mask)==0:
        continue
    objid_list, brickid_list, release_list = decode_targetid(cat_basic['TARGETID'][mask])[:3]

    def get_tractor_columns(brickid):

        brickname = bricks['BRICKNAME'][bricks['BRICKID']==brickid][0]
        tractor_path = os.path.join(tractor_dir, brickname[:3], 'tractor-'+brickname+'.fits')
        tractor_objid = fitsio.read(tractor_path, columns=['objid'])['objid']
        mask = brickid_list==brickid
        idx = np.where(np.in1d(tractor_objid, objid_list[mask]))[0]
        cat = Table(fitsio.read(tractor_path, columns=tractor_columns+['objid', 'brickid', 'release'], rows=idx))
        cat['TARGETID'] = encode_targetid(cat['objid'], cat['brickid'], cat['release'])
        cat.remove_columns(['objid', 'brickid', 'release'])

        if len(cat)!=np.sum(brickid_list==brickid):
            print(len(cat), np.sum(brickid_list==brickid))
            raise ValueError('different catalog length')

        return cat

    tractor_dir = '/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/{}/tractor'.format(field)

    # start multiple worker processes
    with Pool(processes=n_processes) as pool:
        res = pool.map(get_tractor_columns, np.unique(brickid_list))

    # # Remove None elements from the list
    # for index in range(len(res)-1, -1, -1):
    #     if res[index] is None:
    #         res.pop(index)

    cat_more = vstack(res, join_type='exact')

    cat_stack.append(cat_more)

cat_more = vstack(cat_stack)

if len(cat_more)!=len(cat_basic):
    print(len(cat_more), len(cat_basic))
    raise ValueError('different catalog length')

# Here matching cat_more to cat_basic
t1_reverse_sort = np.array(cat_basic['TARGETID']).argsort().argsort()
cat_more = cat_more[np.argsort(cat_more['TARGETID'])[t1_reverse_sort]]
if not np.all(cat_more['TARGETID']==cat_basic['TARGETID']):
    raise ValueError('different targetid')
cat_more.remove_column('TARGETID')

cat_more.write(output_path)

print(time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
