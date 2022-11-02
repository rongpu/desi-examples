# Obtain sweep, sweep-extra, and photo-z columns for a catalog
# Python example (in interactive node when using >1 processes):
# from get_sweep_columns import get_sweep_columns
# columns = ['FLUX_G', 'NEA_R', 'Z_PHOT_MEDIAN']
# new = get_sweep_columns(cat, columns, n_processes=128)

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio

from multiprocessing import Pool

from desitarget.targets import encode_targetid, decode_targetid


def decode_sweep_name(sweepname):
    # taken from desihub/desitarget

    sweepname = os.path.basename(sweepname)

    ramin, ramax = float(sweepname[6:9]), float(sweepname[14:17])
    decmin, decmax = float(sweepname[10:13]), float(sweepname[18:21])

    if sweepname[9] == 'm':
        decmin *= -1
    if sweepname[17] == 'm':
        decmax *= -1

    return [ramin, ramax, decmin, decmax]


def is_in_box(objs, radecbox, ra_col='RA', dec_col='DEC'):
    # taken from desihub/desitarget

    ramin, ramax, decmin, decmax = radecbox

    # ADM check for some common mistakes.
    if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
        msg = 'Strange input: [ramin, ramax, decmin, decmax] = {}'.format(radecbox)
        raise ValueError(msg)

    ii = ((objs[ra_col] >= ramin) & (objs[ra_col] < ramax)
          & (objs[dec_col] >= decmin) & (objs[dec_col] < decmax))

    return ii


def read_sweep_columns(sweep_fn, field, pz_dir=None):

    cat = Table(fitsio.read(sweep_fn, columns=['OBJID', 'BRICKID', 'RELEASE']))
    targetid = encode_targetid(cat['OBJID'], cat['BRICKID'], cat['RELEASE'])
    if field=='north':
        idx = np.where(np.in1d(targetid, cat_basic_north['TARGETID']))[0]
    else:
        idx = np.where(np.in1d(targetid, cat_basic_south['TARGETID']))[0]
    if len(idx)==0:
        return None
    targetid = targetid[idx]

    if len(sweep_columns)!=0:
        cat = Table(fitsio.read(sweep_fn, rows=idx, columns=sweep_columns))
    else:
        cat = Table()

    if len(sweep_extra_columns)!=0:
        sweep_extra_fn = sweep_fn.replace('/sweep/9.0/', '/sweep/9.0-extra/').replace('.fits', '-ex.fits')
        cat_extra = Table(fitsio.read(sweep_extra_fn, rows=idx, columns=sweep_extra_columns))
    else:
        cat_extra = Table()

    if len(pz_columns)!=0:
        if pz_dir is None:
            pz_fn = sweep_fn.replace('sweep/9.0/', 'sweep/9.0-photo-z/').replace('.fits', '-pz.fits')
        else:
            pz_fn = os.path.basename(sweep_fn).replace('.fits', '-pz.fits')
            pz_fn = os.path.join(pz_dir, field, pz_fn)
        pz = Table(fitsio.read(pz_fn, rows=idx, columns=pz_columns))

    cat = hstack([cat, cat_extra, pz], join_type='outer')
    cat['TARGETID'] = targetid

    return cat


def get_sweep_columns(cat, columns, n_processes=128, pz_dir=None):
    '''
    Return catalog with additional sweep columns.

    Args:
       cat: astropy table with at minimum these columns:
            RA, DEC and TARGETID (or OBJID, BRICKID and RELEASE if TARGETID is not available)
       columns: list of str, columns to be obtained

    Options:
       n_processes: int, number of processes in multiprocessing
       pz_dir: str (default None), directory path of photo-z files; by default it takes DR9 photo-z's
    '''

    print('Start!')
    time_start = time.time()

    sweep_columns_all = ['RELEASE', 'BRICKID', 'BRICKNAME', 'OBJID', 'TYPE', 'RA', 'DEC', 'RA_IVAR', 'DEC_IVAR', 'DCHISQ', 'EBV', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3', 'FLUX_IVAR_W4', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'NOBS_W1', 'NOBS_W2', 'NOBS_W3', 'NOBS_W4', 'RCHISQ_G', 'RCHISQ_R', 'RCHISQ_Z', 'RCHISQ_W1', 'RCHISQ_W2', 'RCHISQ_W3', 'RCHISQ_W4', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z', 'FRACFLUX_W1', 'FRACFLUX_W2', 'FRACFLUX_W3', 'FRACFLUX_W4', 'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACIN_G', 'FRACIN_R', 'FRACIN_Z', 'ANYMASK_G', 'ANYMASK_R', 'ANYMASK_Z', 'ALLMASK_G', 'ALLMASK_R', 'ALLMASK_Z', 'WISEMASK_W1', 'WISEMASK_W2', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2', 'WISE_COADD_ID', 'SHAPE_R', 'SHAPE_R_IVAR', 'SHAPE_E1', 'SHAPE_E1_IVAR', 'SHAPE_E2', 'SHAPE_E2_IVAR', 'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z', 'FIBERTOTFLUX_G', 'FIBERTOTFLUX_R', 'FIBERTOTFLUX_Z', 'REF_CAT', 'REF_ID', 'REF_EPOCH', 'GAIA_PHOT_G_MEAN_MAG', 'GAIA_PHOT_G_MEAN_FLUX_OVER_ERROR', 'GAIA_PHOT_BP_MEAN_MAG', 'GAIA_PHOT_BP_MEAN_FLUX_OVER_ERROR', 'GAIA_PHOT_RP_MEAN_MAG', 'GAIA_PHOT_RP_MEAN_FLUX_OVER_ERROR', 'GAIA_ASTROMETRIC_EXCESS_NOISE', 'GAIA_DUPLICATED_SOURCE', 'GAIA_PHOT_BP_RP_EXCESS_FACTOR', 'GAIA_ASTROMETRIC_SIGMA5D_MAX', 'GAIA_ASTROMETRIC_PARAMS_SOLVED', 'PARALLAX', 'PARALLAX_IVAR', 'PMRA', 'PMRA_IVAR', 'PMDEC', 'PMDEC_IVAR', 'MASKBITS', 'FITBITS', 'SERSIC', 'SERSIC_IVAR']

    sweep_extra_columns_all = ['BRICK_PRIMARY', 'BX', 'BY', 'MJD_MIN', 'MJD_MAX', 'GAIA_PHOT_G_N_OBS', 'GAIA_PHOT_BP_N_OBS', 'GAIA_PHOT_RP_N_OBS', 'GAIA_PHOT_VARIABLE_FLAG', 'GAIA_ASTROMETRIC_EXCESS_NOISE_SIG', 'GAIA_ASTROMETRIC_N_OBS_AL', 'GAIA_ASTROMETRIC_N_GOOD_OBS_AL', 'GAIA_ASTROMETRIC_WEIGHT_AL', 'GAIA_A_G_VAL', 'GAIA_E_BP_MIN_RP_VAL', 'APFLUX_G', 'APFLUX_R', 'APFLUX_Z', 'APFLUX_RESID_G', 'APFLUX_RESID_R', 'APFLUX_RESID_Z', 'APFLUX_BLOBRESID_G', 'APFLUX_BLOBRESID_R', 'APFLUX_BLOBRESID_Z', 'APFLUX_IVAR_G', 'APFLUX_IVAR_R', 'APFLUX_IVAR_Z', 'APFLUX_MASKED_G', 'APFLUX_MASKED_R', 'APFLUX_MASKED_Z', 'APFLUX_W1', 'APFLUX_W2', 'APFLUX_W3', 'APFLUX_W4', 'APFLUX_RESID_W1', 'APFLUX_RESID_W2', 'APFLUX_RESID_W3', 'APFLUX_RESID_W4', 'APFLUX_IVAR_W1', 'APFLUX_IVAR_W2', 'APFLUX_IVAR_W3', 'APFLUX_IVAR_W4', 'NEA_G', 'NEA_R', 'NEA_Z', 'BLOB_NEA_G', 'BLOB_NEA_R', 'BLOB_NEA_Z', 'PSFDEPTH_W3', 'PSFDEPTH_W4', 'WISE_X', 'WISE_Y']

    pz_columns_all = ['Z_PHOT_MEAN', 'Z_PHOT_MEDIAN', 'Z_PHOT_STD', 'Z_PHOT_L68', 'Z_PHOT_U68', 'Z_PHOT_L95', 'Z_PHOT_U95', 'Z_SPEC', 'SURVEY', 'TRAINING', 'KFOLD']

    global sweep_columns, sweep_extra_columns, pz_columns
    sweep_columns, sweep_extra_columns, pz_columns = [], [], []
    for col in columns:
        if col in sweep_columns_all:
            sweep_columns.append(col)
        elif col in sweep_extra_columns_all:
            sweep_extra_columns.append(col)
        elif col in pz_columns_all:
            pz_columns.append(col)
        else:
            raise ValueError('Column {} does not exist!'.format(col))

    if 'TARGETID' not in cat.colnames:
        cat['TARGETID'] = encode_targetid(cat['OBJID'], cat['BRICKID'], cat['RELEASE'])
    if 'RELEASE' not in cat.colnames:
        _, _, cat['RELEASE'], _, _, _ = decode_targetid(cat['TARGETID'])

    mask_north = cat['RELEASE']==9011
    mask_south = (cat['RELEASE']==9010) | (cat['RELEASE']==9012)
    global cat_basic_north, cat_basic_south
    cat_basic_north = cat[mask_north].copy()
    cat_basic_south = cat[mask_south].copy()

    sweep_dir_north = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/sweep/9.0'
    sweep_fn_list_north = np.array(sorted(glob.glob(os.path.join(sweep_dir_north, '*.fits'))))
    sweep_radec_list_north = [decode_sweep_name(sweep_fn) for sweep_fn in sweep_fn_list_north]
    mask = np.array([np.any(is_in_box(cat_basic_north, sweep_radec)) for sweep_radec in sweep_radec_list_north])
    sweep_fn_list_north = np.unique(sweep_fn_list_north[mask])

    sweep_dir_south = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0'
    sweep_fn_list_south = np.array(sorted(glob.glob(os.path.join(sweep_dir_south, '*.fits'))))
    sweep_radec_list_south = [decode_sweep_name(sweep_fn) for sweep_fn in sweep_fn_list_south]
    mask = np.array([np.any(is_in_box(cat_basic_south, sweep_radec)) for sweep_radec in sweep_radec_list_south])
    sweep_fn_list_south = np.unique(sweep_fn_list_south[mask])

    # zipped_arg_list = list(zip(sweep_fn_list_north, ['north']*len(sweep_fn_list_north)))
    # zipped_arg_list += list(zip(sweep_fn_list_south, ['south']*len(sweep_fn_list_south)))
    zipped_arg_list = list(zip(sweep_fn_list_north, ['north']*len(sweep_fn_list_north), [pz_dir]*len(sweep_fn_list_north)))
    zipped_arg_list += list(zip(sweep_fn_list_south, ['south']*len(sweep_fn_list_south), [pz_dir]*len(sweep_fn_list_south)))

    with Pool(processes=n_processes) as pool:
        res = pool.starmap(read_sweep_columns, zipped_arg_list, chunksize=1)

    # Remove None elements from the list
    for index in range(len(res)-1, -1, -1):
        if res[index] is None:
            res.pop(index)

    new = vstack(res, join_type='exact')
    if len(new)!=len(cat):
        print(len(new), len(cat))
        raise ValueError('different catalog length')

    # Here matching new to cat
    t1_reverse_sort = np.array(cat['TARGETID']).argsort().argsort()
    new = new[np.argsort(new['TARGETID'])[t1_reverse_sort]]
    if not np.all(new['TARGETID']==cat['TARGETID']):
        raise ValueError('different targetid')

    new = new[['TARGETID']+columns]

    print(time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))

    return new
