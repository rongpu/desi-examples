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
            res = pool.map(count_in_parallel, pix_idx_split)
        hp_table = vstack(res)
        hp_table.sort('HPXPIXEL')

        del weights_, pix_unique_, pixorder_, pixcnts_

    return hp_table


def count_in_parallel(pix_idx):
    pix_list = pix_unique_[pix_idx]
    hp_table = Table()
    hp_table['HPXPIXEL'] = pix_list
    # hp_table['RA'], hp_table['DEC'] = hp.pixelfunc.pix2ang(nside, pix_list, nest=False, lonlat=True)
    hp_table['count'] = np.zeros(len(hp_table))
    for index in np.arange(len(pix_idx)):
        idx = pixorder_[pixcnts_[pix_idx[index]]:pixcnts_[pix_idx[index]+1]]
        hp_table['count'][index] = np.sum(weights_[idx])

    return hp_table

