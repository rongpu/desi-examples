# srun -N 1 -C cpu -c 256 -t 04:00:00 -L cfs -q interactive python compute_systematics_maps.py south

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio
from astropy.io import fits

from multiprocessing import Pool
import healpy as hp


# resolve = 'noresolve'
resolve = 'resolve'

field = str(sys.argv[1])
field = field.lower()

if field=='south':
    photsys = 'S'
elif field=='north':
    photsys = 'N'

min_nobs = 1
# maskbits = sorted([1, 13])
# maskbits = sorted([1, 12, 13])
# maskbits = sorted([1, 11, 12, 13])
# maskbits = sorted([1, 8, 9, 11, 12, 13])
# custom_mask_name = ''

maskbits = []
# custom_mask_name = 'lrgmask_v1.1'
custom_mask_name = 'elgmask_v1'

mask_str = ''.join([str(tmp) for tmp in maskbits])
if custom_mask_name!='':
    mask_str += '_' + custom_mask_name

n_randoms_catalogs = 8

n_processes = 128

# nsides = [64, 128, 256, 512, 1024]
nsides = [64, 128, 256, 512]

randoms_columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS', 'PHOTSYS',
                   'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
                   'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                   'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'EBV']

if resolve=='resolve':
    randoms_paths = sorted(glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
elif resolve=='noresolve':
    randoms_paths = sorted(glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/noresolve/{}/randoms-noresolve-*.fits'.format(field)))
randoms_paths = randoms_paths[:n_randoms_catalogs]

randoms_density = 2500

mask_dir = os.path.join('/global/cfs/cdirs/desi/users/rongpu/desi_mask/randoms/', custom_mask_name)

output_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/{}/systematics'.format(resolve)

hp_columns = ['EBV', 'galdepth_gmag', 'galdepth_rmag', 'galdepth_zmag', 'psfdepth_gmag', 'psfdepth_rmag', 'psfdepth_zmag', 'psfdepth_w1mag', 'psfdepth_w2mag', 'galdepth_gmag_ebv', 'galdepth_rmag_ebv', 'galdepth_zmag_ebv', 'psfdepth_gmag_ebv', 'psfdepth_rmag_ebv', 'psfdepth_zmag_ebv', 'psfdepth_w1mag_ebv', 'psfdepth_w2mag_ebv', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'NOBS_G', 'NOBS_R', 'NOBS_Z']


def apply_mask(randoms, min_nobs, maskbits, custom_mask_name):

    mask = (randoms['NOBS_G']>=min_nobs) & (randoms['NOBS_R']>=min_nobs) & (randoms['NOBS_Z']>=min_nobs)

    mask_clean = np.ones(len(randoms), dtype=bool)
    for bit in maskbits:
        mask_clean &= (randoms['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))

    if custom_mask_name!='':
        mask_col = custom_mask_name[: custom_mask_name.find("mask")]+'_mask'
        mask_clean &= randoms[mask_col]==0

    mask &= mask_clean

    return mask


def get_systematics(pix_idx):

    pix_list = pix_unique[pix_idx]

    hp_table = Table()
    hp_table['HPXPIXEL'] = pix_list
    hp_table['RA'], hp_table['DEC'] = hp.pixelfunc.pix2ang(nside, pix_list, nest=False, lonlat=True)

    arr = np.zeros([len(pix_list), len(hp_columns)])
    hp_table = hstack([hp_table, Table(arr, names=hp_columns)])

    for index in np.arange(len(pix_idx)):

        idx = pixorder[pixcnts[pix_idx[index]]:pixcnts[pix_idx[index]+1]]

        for hp_column in hp_columns:
            if 'NOBS_' in hp_column:
                hp_table[hp_column][index] = np.mean(randoms[hp_column][idx])
            else:
                hp_table[hp_column][index] = np.median(randoms[hp_column][idx])

    return hp_table


if __name__ == '__main__':

    print('Start!')

    time_start = time.time()

    randoms_stack = []

    for randoms_path in randoms_paths:

        # print(randoms_path)
        # randoms_index_str = os.path.basename(randoms_path).replace('randoms-{}-'.format(resolve), '').replace('.fits', '')

        # randoms = Table(fitsio.read(randoms_path, columns=randoms_columns))
        hdu = fits.open(randoms_path)
        randoms = Table()
        for col in randoms_columns:
            randoms[col] = np.copy(hdu[1].data[col])

        if custom_mask_name!='':
            mask_path = os.path.join(mask_dir, os.path.basename(randoms_path).replace('.fits', '-{}.fits.gz'.format(custom_mask_name)))
            custom_mask = Table(fitsio.read(mask_path))
            randoms = hstack([randoms, custom_mask], join_type='exact')

        # print(len(randoms))

        if fitsio.read_header(randoms_path, ext=1)['DENSITY']!=randoms_density:
            raise ValueError

        if resolve=='resolve':
            mask = randoms['PHOTSYS']==photsys
            randoms = randoms[mask]

        mask = apply_mask(randoms, min_nobs, maskbits, custom_mask_name)
        randoms = randoms[mask]

        randoms_stack.append(randoms)

    randoms = vstack(randoms_stack)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        randoms['galdepth_gmag'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_G'])))-9)
        randoms['galdepth_rmag'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_R'])))-9)
        randoms['galdepth_zmag'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_Z'])))-9)
        randoms['psfdepth_gmag'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_G'])))-9)
        randoms['psfdepth_rmag'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_R'])))-9)
        randoms['psfdepth_zmag'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_Z'])))-9)
        randoms['psfdepth_w1mag'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_W1'])))-9)
        randoms['psfdepth_w2mag'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_W2'])))-9)
        randoms['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_G'])))-9) - 3.214*randoms['EBV']
        randoms['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_R'])))-9) - 2.165*randoms['EBV']
        randoms['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['GALDEPTH_Z'])))-9) - 1.211*randoms['EBV']
        randoms['psfdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_G'])))-9) - 3.214*randoms['EBV']
        randoms['psfdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_R'])))-9) - 2.165*randoms['EBV']
        randoms['psfdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_Z'])))-9) - 1.211*randoms['EBV']
        randoms['psfdepth_w1mag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_W1'])))-9) - 0.184*randoms['EBV']
        randoms['psfdepth_w2mag_ebv'] = -2.5*(np.log10((5/np.sqrt(randoms['PSFDEPTH_W2'])))-9) - 0.113*randoms['EBV']

    print('Loading complete!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

    for nside in nsides:

        output_path = os.path.join(output_dir, 'systematics_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, mask_str))
        if os.path.isfile(output_path):
            continue

        npix = hp.nside2npix(nside)

        pix_allobj = hp.pixelfunc.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)
        pix_unique, pixcnts = np.unique(pix_allobj, return_counts=True)

        pixcnts = np.insert(pixcnts, 0, 0)
        pixcnts = np.cumsum(pixcnts)

        pixorder = np.argsort(pix_allobj)

        # split among the Cori processors
        pix_idx_split = np.array_split(np.arange(len(pix_unique)), n_processes)

        # start multiple worker processes
        with Pool(processes=n_processes) as pool:
            res = pool.map(get_systematics, pix_idx_split)

        hp_table = vstack(res)
        hp_table.sort('HPXPIXEL')

        hp_table.write(output_path)

    print('Done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
