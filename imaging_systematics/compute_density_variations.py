# Example:
# python compute_density_variations.py LRG south

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio

# from multiprocessing import Pool
import healpy as hp


target_class, field = str(sys.argv[1]), str(sys.argv[2])
target_class = target_class.upper()
field = field.lower()

min_nobs = 1

use_combined_catalog = True

target_bits = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS_ANY': 60, 'BGS_BRIGHT': 1}
# maskbits_dict = {'LRG': [1, 8, 9, 11, 12, 13], 'ELG': [1, 11, 12, 13], 'QSO': [1, 8, 9, 11, 12, 13], 'BGS_ANY': [1, 13], 'BGS_BRIGHT': [1, 13]}
maskbits_dict = {'LRG': [1, 12, 13], 'ELG': [1, 12, 13], 'QSO': [1, 12, 13], 'BGS_ANY': [1, 13], 'BGS_BRIGHT': [1, 13]}  # The maskbits in desitarget
# maskbits_dict = {'LRG': [], 'ELG': [1, 11, 12, 13], 'QSO': [1, 8, 9, 11, 12, 13], 'BGS_ANY': [1, 13], 'BGS_BRIGHT': [1, 13]}
apply_lrgmask = False
# apply_lrgmask = True

if apply_lrgmask:
    lrgmask_str = '_lrgmask_v1'
else:
    lrgmask_str = ''

nsides = [64, 128, 256, 512, 1024]

target_columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS']

if 'BGS' in target_class:
    target_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/1.0.0/targets/main/resolve/bright'
else:
    target_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/1.0.0/targets/main/resolve/dark'

cat_dir = '/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.0.0/resolve'

output_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/density_maps/1.0.0/resolve'

target_bit = target_bits[target_class]
maskbits = maskbits_dict[target_class]

if field=='south':
    photsys = 'S'
else:
    photsys = 'N'


def apply_mask(cat, min_nobs, maskbits):

    mask = (cat['NOBS_G']>=min_nobs) & (cat['NOBS_R']>=min_nobs) & (cat['NOBS_Z']>=min_nobs)

    mask_clean = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask_clean &= (cat['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))

    mask &= mask_clean

    return mask


def get_systematics(pix_list):

    hp_table = Table()
    hp_table['HPXPIXEL'] = pix_list
    hp_table['RA'], hp_table['DEC'] = hp.pixelfunc.pix2ang(nside, pix_list, nest=False, lonlat=True)

    # if target_class=='LRG' or target_class=='QSO':
    #     hp_columns = ['NOBS_W1', 'NOBS_W2']
    #     arr = np.zeros([len(pix_list), len(hp_columns)])
    #     hp_table = hstack([hp_table, Table(arr, names=hp_columns)])
    #     for index in range(len(pix_list)):
    #         mask = pix_allobj==pix_list[index]
    #         hp_table['NOBS_W1'][index] = np.mean(cat['NOBS_W1'][mask])
    #         hp_table['NOBS_W2'][index] = np.mean(cat['NOBS_W2'][mask])

    return hp_table


if __name__ == '__main__':

    print('Start!')
    print(target_class, field)

    time_start = time.time()

    if not use_combined_catalog or apply_lrgmask:
        cat_path = os.path.join(cat_dir, 'dr9_{}_{}_1.0.0_basic.fits'.format(target_class.lower(), field))
        cat = Table(fitsio.read(cat_path))
        if apply_lrgmask:
            lrgmask_path = os.path.join(cat_dir, 'dr9_{}_{}_1.0.0_lrgmask_v1.fits'.format(target_class.lower(), field))
            lrgmask = Table(fitsio.read(lrgmask_path))
            cat = hstack([cat, lrgmask], join_type='exact')
            mask = cat['lrg_mask']==0
            cat = cat[mask]
    else:
        target_path_list = glob.glob(os.path.join(target_dir, 'targets-*.fits'))
        cat = []
        for target_path in target_path_list:
            # print(target_path)
            if target_class!='BGS_BRIGHT':
                tmp = fitsio.read(target_path, columns=['DESI_TARGET', 'PHOTSYS'])
                mask = ((tmp["DESI_TARGET"] & (2**target_bit))!=0) & (tmp['PHOTSYS']==photsys)
            else:
                tmp = fitsio.read(target_path, columns=['BGS_TARGET', 'PHOTSYS'])
                mask = ((tmp["BGS_TARGET"] & (2**target_bit))!=0) & (tmp['PHOTSYS']==photsys)
            idx = np.where(mask)[0]
            if len(idx)==0:
                continue
            # print(len(idx)/len(tmp), len(idx), len(tmp))
            cat.append(Table(fitsio.read(target_path, columns=target_columns, rows=idx)))
        cat = vstack(cat)

    print('Loading complete!')

    mask = apply_mask(cat, min_nobs, maskbits)
    cat = cat[mask]

    for nside in nsides:

        output_path = os.path.join(output_dir, 'density_map_{}_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(target_class.lower(), field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str))
        if os.path.isfile(output_path):
            continue

        npix = hp.nside2npix(nside)
        pix_allobj = hp.pixelfunc.ang2pix(nside, cat['RA'], cat['DEC'], lonlat=True)
        pix_unique, pix_count = np.unique(pix_allobj, return_counts=True)
        hp_table = get_systematics(pix_unique)
        hp_table['n_targets'] = pix_count
        hp_table.write(output_path)

    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
