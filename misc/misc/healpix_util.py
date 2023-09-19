from __future__ import division, print_function
import numpy as np
from astropy.table import Table, vstack
# from astropy.io import fits

import healpy as hp
from multiprocessing import Pool


def count_in_healpix(nside, ra, dec, weights=None, n_processes=1):

    pix_allobj = np.array(hp.pixelfunc.ang2pix(nside, ra, dec, nest=False, lonlat=True))
    pix_unique, pix_count = np.unique(pix_allobj, return_counts=True)

    if weights is None:
        hp_table = Table()
        hp_table['HPXPIXEL'] = pix_unique
        hp_table['count'] = pix_count
    else:
        global weights_, pix_unique_, pixorder_, pixcnts_
        pix_unique_ = pix_unique
        pixcnts_ = pix_count.copy()
        pixcnts_ = np.insert(pixcnts_, 0, 0)
        pixcnts_ = np.cumsum(pixcnts_)
        pixorder_ = np.argsort(pix_allobj)
        # split among the processors
        pix_idx_split = np.array_split(np.arange(len(pix_unique_)), n_processes)

        weights_ = weights

        with Pool(processes=n_processes) as pool:
            res = pool.map(weighted_count_in_parallel, pix_idx_split)
        hp_table = vstack(res)
        hp_table.sort('HPXPIXEL')

        del weights_, pix_unique_, pixorder_, pixcnts_

    return hp_table


def weighted_count_in_parallel(pix_idx):
    pix_list = pix_unique_[pix_idx]
    hp_table = Table()
    hp_table['HPXPIXEL'] = pix_list
    # hp_table['RA'], hp_table['DEC'] = hp.pixelfunc.pix2ang(nside, pix_list, nest=False, lonlat=True)
    hp_table['count'] = np.zeros(len(hp_table))
    for index in np.arange(len(pix_idx)):
        idx = pixorder_[pixcnts_[pix_idx[index]]:pixcnts_[pix_idx[index]+1]]
        hp_table['count'][index] = np.sum(weights_[idx])

    return hp_table


def downsize_hp_map(nside_in, nside_out, hp_in, stats_dict=None, weights=None, n_processes=1):
    '''
    Downsize a masked (nan) healpix map.
    Example:
    stats_dict = {'n_randoms': np.sum, 'EBV': np.average, 'GALDEPTH_G': np.average, 'GALDEPTH_R': np.average, 'GALDEPTH_Z': np.average, 'PSFDEPTH_G': np.average, 'PSFDEPTH_R': np.average, 'PSFDEPTH_Z': np.average, 'PSFDEPTH_W1': np.average, 'PSFDEPTH_W2': np.average, 'PSFSIZE_G': np.average, 'PSFSIZE_R': np.average, 'PSFSIZE_Z': np.average, 'STARDENS': np.average}
    columns = ['HPXPIXEL', 'n_randoms', 'EBV', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'STARDENS']
    hp_table_new = downsize_hp_map(1024, 128, hp_table[columns], stats_dict=stats_dict, weights=hp_table['n_randoms'], n_processes=128)
    '''

    global pix_unique_, pixorder_, pixcnts_, colnames_, stats_dict_, hp_in_, weights_
    hp_in_ = hp_in
    stats_dict_ = stats_dict
    weights_ = weights
    colnames_ = hp_in.colnames
    colnames_.remove('HPXPIXEL')

    ra, dec = hp.pix2ang(nside_in, hp_in['HPXPIXEL'], nest=False, lonlat=True)
    pix_allobj = hp.pixelfunc.ang2pix(nside_out, ra, dec, lonlat=True)
    pix_unique_, pixcnts_ = np.unique(pix_allobj, return_counts=True)

    pixcnts_ = np.insert(pixcnts_, 0, 0)
    pixcnts_ = np.cumsum(pixcnts_)
    pixorder_ = np.argsort(pix_allobj)
    # split among the processors
    pix_idx_split = np.array_split(np.arange(len(pix_unique_)), n_processes)

    with Pool(processes=n_processes) as pool:
        res = pool.map(get_stats_in_parallel, pix_idx_split)
    hp_table = vstack(res)
    hp_table.sort('HPXPIXEL')

    del pix_unique_, pixorder_, pixcnts_, colnames_, stats_dict_, hp_in_, weights_

    return hp_table


# def simple_updownsizing(nside_in, nside_out, v, pix=None):
#     '''
#     Simple up/downsizing of unmasked map.
#     Example:
#     pix_new, v_new = simple_updownsizing(128, 64, full_sky['EBV'], full_sky['HPXPIXEL'])
#     '''
#     if pix is None:
#         pix = np.arange(hp.nside2npix(nside_in))



def get_stats_in_parallel(pix_idx):
    pix_list = pix_unique_[pix_idx]
    hp_table = Table()
    hp_table['HPXPIXEL'] = pix_list

    for col in colnames_:
        hp_table[col] = np.zeros(len(hp_table), dtype=hp_in_[col].dtype)
        if stats_dict_ is None or col not in stats_dict_.keys():
            stats = np.mean
        else:
            stats = stats_dict_[col]
        for index in np.arange(len(pix_idx)):
            idx = pixorder_[pixcnts_[pix_idx[index]]:pixcnts_[pix_idx[index]+1]]
            if weights_ is None or stats==np.sum:
                hp_table[col][index] = stats(hp_in_[col][idx])
            else:
                hp_table[col][index] = stats(hp_in_[col][idx], weights=weights_[idx])

    return hp_table



