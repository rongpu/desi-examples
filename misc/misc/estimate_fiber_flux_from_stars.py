# Estimate the flux in DESI fibers from scattered light of stars

# Example:
# from estimate_fiber_flux_from_stars import get_stellar_flux
# cat = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/xmm_lae/odin_xmm_n419_lae_targets.fits'))
# ffcat = get_stellar_flux(cat)
# print brightest magnitudes:
# for col in ['star_flux_g', 'star_flux_r', 'star_flux_i', 'star_flux_z']:
#     mask = ffcat[col]!=0
#     print(col, np.min(22.5 - 2.5*np.log10(ffcat[col][mask])))

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

import healpy as hp

from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import stats


def get_stellar_flux(cat):

    bands = ['g', 'r', 'i', 'z']

    cat['original_id'] = np.arange(len(cat))

    gaia_dir = '/dvs_ro/cfs/cdirs/cosmo/data/gaia/edr3/healpix'

    nside = 32
    hp_pix = hp.ang2pix(nside, cat['ra'], cat['dec'], nest=True, lonlat=True)
    hp_include_neighbors = np.unique(hp.get_all_neighbours(nside, np.unique(hp_pix), nest=True))
    hp_include_neighbors = np.unique(np.concatenate([hp_pix, hp_include_neighbors]))

    gaia = []
    for hp_index in np.unique(hp_include_neighbors):
        gaia_fn = str(hp_index).zfill(5)
        tmp = Table(fitsio.read(os.path.join(gaia_dir, 'healpix-{}.fits'.format(gaia_fn)), columns=['PHOT_G_MEAN_MAG', 'PHOT_G_MEAN_FLUX_OVER_ERROR']))
        mask = (tmp['PHOT_G_MEAN_MAG']<18.0) & (tmp['PHOT_G_MEAN_FLUX_OVER_ERROR']>0)
        idx = np.where(mask)[0]
        tmp = Table(fitsio.read(os.path.join(gaia_dir, 'healpix-{}.fits'.format(gaia_fn)), rows=idx, columns=['RA', 'DEC', 'PHOT_G_MEAN_MAG', 'PHOT_BP_MEAN_MAG', 'PHOT_RP_MEAN_MAG']))
        gaia.append(tmp)

    gaia = vstack(gaia)
    print(len(gaia))


    # Coefficients for EDR3
    coeffs = dict(
        g = [-0.1125681175, 0.3506376997, 0.9082025788, -1.0078309266,
            -1.4212131445, 4.5685722177, -4.5719415419, 2.3816887292,
            -0.7162270722, 0.1247021438, -0.0114938710, 0.0003949585,
            0.0000051647],
        r = [0.1431278873, -0.2999797766, -0.0553379742, 0.1544273115,
            0.3068634689, -0.9499143903, 0.9769739362, -0.4926704528,
            0.1272539574, -0.0133178183, -0.0008153813, 0.0003094116,
            -0.0000198891],
        i = [0.3396481660, -0.6491867119, -0.3330769819, 0.4381097294,
            0.5752125977, -1.4746570523, 1.2979140762, -0.6371018151,
            0.1948940062, -0.0382055596, 0.0046907449, -0.0003296841,
            0.0000101480],
        z = [0.5173814296, -1.0450176704, 0.1529797809, 0.1856005222,
            -0.2366580132, 0.1018331214, -0.0189673240, 0.0012988354])

    bprp_min, bprp_max = -0.5, 4.7

    for band in bands:
        mag = np.copy(gaia['PHOT_G_MEAN_MAG'])
        for order, c in enumerate(coeffs[band]):
            x = gaia['PHOT_BP_MEAN_MAG']-gaia['PHOT_RP_MEAN_MAG']
            x = np.clip(x, bprp_min, bprp_max)
            mag += c * (x)**order
        gaia['decam_mag_'+band] = mag

    mask = (gaia['PHOT_BP_MEAN_MAG']==0) | (gaia['PHOT_RP_MEAN_MAG']==0)
    for band in bands:
        gaia['decam_mag_'+band][mask] = np.nan

    gaia['mask_mag'] = np.nanmin([gaia['PHOT_G_MEAN_MAG'], gaia['decam_mag_z']+1], axis=0)


    mask_radius_factor = 3

    g_bins = np.arange(int(np.floor(gaia['mask_mag'].min())), 19, 1)
    print(g_bins)

    ra2 = cat['ra']
    dec2 = cat['dec']
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')

    newcat_stack = []

    for index in range(len(g_bins)-1):

        gmin, gmax = g_bins[index], g_bins[index+1]
        print('{} < G < {}'.format(gmin, gmax))

        mask = (gaia['mask_mag']>gmin) & (gaia['mask_mag']<gmax)
        gaia1 = gaia[mask]
        print(np.sum(mask), np.sum(mask)/len(mask))

        if np.sum(mask)==0:
            continue

        ra1 = gaia1['RA']
        dec1 = gaia1['DEC']
        sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')

        max_mask_radius = (1630 * 1.396**(-gaia1['mask_mag'])).max()
        search_radius = max_mask_radius * mask_radius_factor

        idx1, idx2, d2d, _ = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
        print('%d nearby objects'%len(idx1))
        if len(idx1)==0:
            continue

        d_ra = (ra2[idx2]-ra1[idx1])*3600.    # in arcsec
        d_dec = (dec2[idx2]-dec1[idx1])*3600. # in arcsec

        # Convert d_ra to actual arcsecs
        mask = d_ra > 180*3600
        d_ra[mask] = d_ra[mask] - 360.*3600
        mask = d_ra < -180*3600
        d_ra[mask] = d_ra[mask] + 360.*3600
        d_ra = d_ra * np.cos(dec1[idx1]/180*np.pi)

        # convert distances to numpy array in arcsec
        d2d = np.array(d2d.to(u.arcsec))

        gaia_g = gaia1['mask_mag'][idx1]
        mask_radius = 1630 * 1.396**(-gaia_g)
        mask = d2d < mask_radius * mask_radius_factor
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        d2d = d2d[mask]
        d_ra = d_ra[mask]
        d_dec = d_dec[mask]
        mask_radius = mask_radius[mask]

        newcat = hstack([cat[['original_id']][idx2], gaia1[idx1]])
        newcat['d_ra'] = d_ra.copy()
        newcat['d_dec'] = d_dec.copy()
        newcat['d2d'] = d2d.copy()
        newcat['mask_radius'] = mask_radius.copy()

        newcat_stack.append(newcat)

    newcat_stack = vstack(newcat_stack)
    newcat = newcat_stack


    def get_sb_moffat(r, alpha, beta):
        """
        Calculate the surface brightness of light at radius r of a Moffat profile.
        The integral (i.e., total flux) is unity by definition. 
        Radius in arcsec. Returns SB in nmgy/sq arcsec.
        """
        i = (beta-1)/(np.pi * alpha**2)*(1 + (r/alpha)**2)**(-beta)
        return i


    def get_sb_moffat_plus_power_law(r, alpha1, beta1, plexp2, weight2):
        """
        Calculate the surface brightness of a 22.5 mag star in nanomaggies per sq arcsec.
        Radius in arcsec. Returns SB in nmgy/sq arcsec.
        """
        i = (beta1-1)/(np.pi * alpha1**2)*(1 + (r/alpha1)**2)**(-beta1) \
            + weight2 *r**(plexp2)
        return i


    def get_sb_double_moffat(r, alpha1, beta1, alpha2, beta2, weight2):
        """
        Calculate the surface brightness of a 22.5 mag star in nanomaggies per sq arcsec.
        Radius in arcsec. Returns SB in nmgy/sq arcsec.
        """
        i = (beta1-1)/(np.pi * alpha1**2)*(1 + (r/alpha1)**2)**(-beta1) \
            + weight2 * (beta2-1)/(np.pi * alpha2**2)*(1 + (r/alpha2)**2)**(-beta2)
        return i

    fiber_diameter = 1.5  # arcsec
    fiber_area = np.pi * (fiber_diameter/2)**2  # sq.arcsec.
    print(fiber_area)

    # Extended PSF parameters
    params = {
    'g_weight2': 0.00045, 'g_plexp2': -2.,
    'r_weight2': 0.00033, 'r_plexp2': -2.,
    'i_weight2': 0.00033, 'i_plexp2': -2.,
    'z_alpha2': 16, 'z_beta2': 2.3, 'z_weight2': 0.0095,
    }

    # Medians of the Moffat parameters
    moffat_medians = {'g_alpha': 0.86, 'g_beta': 2.48, 'r_alpha': 0.67, 'r_beta': 2.23, 'i_alpha': 0.619, 'i_beta': 2.173, 'z_alpha': 0.51, 'z_beta': 2.00}

    for band in bands:

        alpha, beta = moffat_medians[band+'_alpha'], moffat_medians[band+'_beta']

        # get total fiber flux in nanomaggies
        norm = fiber_area * 10**(-0.4 * (newcat['decam_mag_'+band] - 22.5))

        if band!='z':
            plexp2, weight2 = params[band+'_plexp2'], params[band+'_weight2']
            newcat['flux_'+band] = norm * get_sb_moffat_plus_power_law(newcat['d2d'], alpha, beta, plexp2, weight2)
            # newcat['flux_inner_'+band] = norm * get_sb_moffat(newcat['d2d'], alpha, beta)
            # newcat['flux_outer_'+band] = norm * weight2 * newcat['d2d']**(plexp2)
        else:
            alpha2, beta2, weight2 = params[band+'_alpha2'], params[band+'_beta2'],  params[band+'_weight2']
            newcat['flux_'+band] = norm * get_sb_double_moffat(newcat['d2d'], alpha, beta, alpha2, beta2, weight2)
            # newcat['flux_inner_'+band] = norm * get_sb_moffat(newcat['d2d'], alpha, beta)
            # newcat['flux_outer_'+band] = norm * weight2 * get_sb_moffat(newcat['d2d'], alpha2, beta2)

        # Assign the exact Gaia-based magnitude if within 1 arcsec
        mask = newcat['d2d']<1.
        newcat['flux_'+band][mask] = 10**(-0.4*(newcat['decam_mag_'+band][mask]-22.5))
        newcat['gaia_g_mag'] = 999.
        newcat['gaia_g_mag'][mask] = newcat['PHOT_G_MEAN_MAG'][mask]

        newcat['flux_'+band][~np.isfinite(newcat['flux_'+band])] = 0.


    ffcat = Table()

    for band in bands:
        newcat1 = newcat.copy()
        newcat1.sort('flux_'+band, reverse=True)
        _, idx_keep = np.unique(newcat1['original_id'], return_index=True)
        newcat1 = newcat1[idx_keep]
        newcat1.sort('original_id')
        ffcat['star_flux_'+band] = np.full(len(cat), 0., dtype=np.float32)
        ffcat['star_flux_'+band][newcat1['original_id']] = newcat1['flux_'+band].copy()

    newcat1 = newcat.copy()
    newcat1 = newcat[newcat1['gaia_g_mag']!=999.]
    newcat1.sort('gaia_g_mag', reverse=False)
    _, idx_keep = np.unique(newcat1['original_id'], return_index=True)
    newcat1 = newcat1[idx_keep]
    newcat1.sort('original_id')
    ffcat['gaia_g_mag'] = np.full(len(cat), np.nan, dtype=np.float32)
    ffcat['gaia_g_mag'][newcat1['original_id']] = newcat1['gaia_g_mag'].copy()

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     ffcat['stellar_gmag'] = 22.5 - 2.5*np.log10(ffcat['star_flux_g'])
    #     ffcat['stellar_rmag'] = 22.5 - 2.5*np.log10(ffcat['star_flux_r'])
    #     ffcat['stellar_imag'] = 22.5 - 2.5*np.log10(ffcat['star_flux_i'])
    #     ffcat['stellar_zmag'] = 22.5 - 2.5*np.log10(ffcat['star_flux_z'])

    print(len(ffcat))

    return ffcat
