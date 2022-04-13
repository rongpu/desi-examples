# srun -N 1 -C haswell -c 64 -t 04:00:00 -q interactive python count_randoms.py south

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
maskbits = sorted([1, 12, 13])
# maskbits = sorted([1, 11, 12, 13])
# maskbits = sorted([1, 8, 9, 11, 12, 13])
apply_lrgmask = False

# maskbits = []
# apply_lrgmask = True

if apply_lrgmask:
    lrgmask_str = '_lrgmask_v1'
else:
    lrgmask_str = ''

n_processes = 32

n_randoms_catalogs = 64  # There are 200 random catalogs in total

nsides = [64, 128, 256, 512, 1024]
randoms_columns = ['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS', 'PHOTSYS']

if resolve=='resolve':
    randoms_paths = sorted(glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-[0-9]*.fits'))
elif resolve=='noresolve':
    randoms_paths = sorted(glob.glob('/global/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/noresolve/{}/randoms-noresolve-*.fits'.format(field)))
print(len(randoms_paths))

randoms_paths = randoms_paths[:n_randoms_catalogs]
print(len(randoms_paths))

randoms_density = 2500

lrgmask_dir = '/global/cfs/cdirs/desi/users/rongpu/desi_mask/lrgmask_v1/randoms'

output_dir = '/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/{}/counts'.format(resolve)


def apply_mask(randoms, min_nobs, maskbits):

    mask = (randoms['NOBS_G']>=min_nobs) & (randoms['NOBS_R']>=min_nobs) & (randoms['NOBS_Z']>=min_nobs)

    mask_clean = np.ones(len(randoms), dtype=bool)
    for bit in maskbits:
        mask_clean &= (randoms['MASKBITS'] & 2**bit)==0
    # print(np.sum(~mask_clean)/len(mask_clean))

    mask &= mask_clean

    return mask


def count_randoms(randoms_path):

    print(randoms_path)
    if resolve=='resolve':
        randoms_index_str = os.path.basename(randoms_path).replace('randoms-', '').replace('.fits', '')
    elif resolve=='noresolve':
        randoms_index_str = os.path.basename(randoms_path).replace('randoms-noresolve-', '').replace('.fits', '')

    all_exist = True
    for nside in nsides:
        output_path = os.path.join(output_dir, 'minobs_{}_maskbits_{}'.format(min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str), '{}_nside_{}_minobs_{}_maskbits_{}_{}.npy'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str, randoms_index_str))
        if not os.path.isfile(output_path):
            all_exist = False
    if all_exist:
        return None

    # randoms = Table(fitsio.read(randoms_path, columns=randoms_columns))
    hdu = fits.open(randoms_path)
    randoms = Table()
    for col in randoms_columns:
        randoms[col] = np.copy(hdu[1].data[col])

    if apply_lrgmask:
        lrgmask_path = os.path.join(lrgmask_dir, os.path.basename(randoms_path).replace('.fits', '-lrgmask_v1.fits'))
        lrgmask = Table(fitsio.read(lrgmask_path))
        randoms = hstack([randoms, lrgmask], join_type='exact')
        mask = randoms['lrg_mask']==0
        randoms = randoms[mask]

    # print(len(randoms))

    if fitsio.read_header(randoms_path, ext=1)['DENSITY']!=randoms_density:
        raise ValueError

    if resolve=='resolve':
        mask = randoms['PHOTSYS']==photsys
        randoms = randoms[mask]

    mask = apply_mask(randoms, min_nobs, maskbits)
    randoms = randoms[mask]

    for nside in nsides:
        npix = hp.nside2npix(nside)

        # pix_area = hp.pixelfunc.nside2pixarea(nside, degrees=True)
        # print('Healpix size = {:.5f} sq deg'.format(pix_area))

        pix = hp.pixelfunc.ang2pix(nside, randoms['RA'], randoms['DEC'], lonlat=True)
        pix_unique, pix_count = np.unique(pix, return_counts=True)
        pix_count_all = np.zeros(npix, dtype=int)
        pix_count_all[pix_unique] = pix_count

        output_path = os.path.join(output_dir, 'minobs_{}_maskbits_{}'.format(min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str), '{}_nside_{}_minobs_{}_maskbits_{}_{}.npy'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str, randoms_index_str))

        if not os.path.isdir(os.path.dirname(output_path)):
            try:
                os.makedirs(os.path.dirname(output_path))
            except:
                pass

        np.save(output_path, pix_count_all)

    return None


if __name__ == '__main__':

    print('Start!')

    time_start = time.time()

    # start multiple worker processes
    with Pool(processes=n_processes) as pool:
        pool.map(count_randoms, randoms_paths)

    # for randoms_path in randoms_paths:
    #     count_randoms(randoms_path)

    # Combine the results into a single table

    for nside in nsides:

        print(nside)

        final_output_path = os.path.join(output_dir, 'counts_{}_nside_{}_minobs_{}_maskbits_{}.fits'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str))
        if os.path.isfile(final_output_path):
            continue

        npix = hp.nside2npix(nside)
        pix_area = hp.pixelfunc.nside2pixarea(nside, degrees=True)

        hp_table = Table()
        hp_table['HPXPIXEL'] = np.arange(npix)
        hp_table['RA'], hp_table['DEC'] = hp.pixelfunc.pix2ang(nside, hp_table['HPXPIXEL'], nest=False, lonlat=True)
        hp_table['n_randoms'] = 0

        output_paths = sorted(glob.glob(os.path.join(output_dir, 'minobs_{}_maskbits_{}'.format(min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str), '{}_nside_{}_minobs_{}_maskbits_{}_*.npy'.format(field, nside, min_nobs, ''.join([str(tmp) for tmp in maskbits])+lrgmask_str))))
        print(len(output_paths))

        for output_path in output_paths:
            n_randoms = np.load(output_path)
            hp_table['n_randoms'] += n_randoms

        total_randoms_density = randoms_density * len(output_paths)
        hp_table['FRACAREA'] = hp_table['n_randoms']/(total_randoms_density*pix_area)

        hp_table.write(final_output_path)

    print('Done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))
