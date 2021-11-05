from __future__ import division, print_function
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
import fitsio


def get_rr_model(coadd_fn, index, use_targetid=False, restframe=False, z=None, return_z=False):
    '''
    Return redrock model spectrum.

    Args:
       coadd_fn: str, path of coadd FITS file
       index: int, index of coadd FITS file if use_targetid=False, or TARGETID if use_targetid=True

    Options:
       use_targetid: bool, if True, index is TARGETID
       restframe: bool, if True, return restframe spectrum in template wavelength grid; if False,
       return spectrum in three cameras in observed frame
       z: bool, if None, use redrock best-fit redshift
       return_z: bool, if true, include redshift in output
    '''
    # If use_targetid=False, index is the index of coadd file; if True, index is TARGETID.

    from desispec.interpolation import resample_flux
    import redrock.templates
    from desispec.io import read_spectra

    spec = read_spectra(coadd_fn)
    redshifts = Table(fitsio.read(coadd_fn.replace('/coadd-', '/redrock-'), ext='REDSHIFTS'))

    if use_targetid:
        coadd_index = np.where(redshifts['TARGETID']==index)[0][0]
    else:
        coadd_index = index
    if z is None:
        z = redshifts['Z'][coadd_index]

    templates = dict()
    for filename in redrock.templates.find_templates():
        tx = redrock.templates.Template(filename)
        templates[(tx.template_type, tx.sub_type)] = tx

    tx = templates[(redshifts['SPECTYPE'][coadd_index], redshifts['SUBTYPE'][coadd_index])]
    coeff = redshifts['COEFF'][coadd_index][0:tx.nbasis]

    if restframe==False:
        wave = dict()
        model_flux = dict()
        for camera in ['B', 'R', 'Z']:
            wave[camera] = spec.wave[camera.lower()]
            model_flux[camera] = np.zeros(wave[camera].shape)
            model = tx.flux.T.dot(coeff).T
            mx = resample_flux(wave[camera], tx.wave*(1+z), model)
            model_flux[camera] = spec.R[camera.lower()][coadd_index].dot(mx)
    else:
        wave = tx.wave
        model_flux = tx.flux.T.dot(coeff).T

    if return_z:
        return wave, model_flux, z
    else:
        return wave, model_flux


def plot_spectrum(coadd_fn, index, use_targetid=False, show_lines=True, show_restframe=True, show_model=True,
    figsize=(20, 8), lw=1.2, gauss_smooth=3, label=None, show=True, return_ax=False, xlim=[3400, 10000], ylim=[-1., 2.]):
    '''
    Plot DESI spectrum.

    Args:
       coadd_fn: str, path of coadd FITS file
       index: int, index of coadd FITS file if use_targetid=False, or TARGETID if use_targetid=True
       
    Options:
       use_targetid: bool, if True, index is TARGETID
    '''

    from scipy.ndimage import gaussian_filter1d

    lines = {
        'Ha'      : 6562.8,
        'Hb'       : 4862.68,
        'Hg'       : 4340.464,
        'Hd'       : 4101.734,
        # 'OIII-b'       :  5006.843,
        # 'OIII-a'       : 4958.911,
        'OIII': 4982.877,
        'MgII'    : 2799.49,
        'OII'         : 3728,
        'CIII'  : 1909.,
        'CIV'    : 1549.06,
        'SiIV'  : 1393.76018,
        'LYA'         : 1215.67,
        'LYB'         : 1025.72
    }

    tmp = fitsio.read(coadd_fn, columns=['TARGETID'], ext='FIBERMAP')

    if use_targetid:
        tid = index
        coadd_index = np.where(tmp['TARGETID']==index)[0][0]
    else:
        tid = tmp['TARGETID'][index]
        coadd_index = index

    # Get model spectrum
    _, model_flux, z = get_rr_model(coadd_fn, coadd_index, return_z=True)

    redshifts = Table(fitsio.read(coadd_fn.replace('/coadd-', '/redrock-'), ext='REDSHIFTS'))
        
    fig, ax1 = plt.subplots(figsize=figsize)
    for camera in ['B', 'R', 'Z']:
        wave = fitsio.read(coadd_fn, ext=camera+'_WAVELENGTH')
        # wave_rest = wave/(1+z)
        flux = fitsio.read(coadd_fn, ext=camera+'_FLUX')[coadd_index]
        mask = fitsio.read(coadd_fn, ext=camera+'_MASK')[coadd_index]
        flux[mask] = 0.
        # if np.sum(mask!=0)!=0:
        #     print('{} masked pixels: {}'.format(camera, np.sum(mask!=0)))

        if gauss_smooth==0 or gauss_smooth is None:
            flux_smooth = flux.copy()
        elif gauss_smooth>0:
            flux_smooth = gaussian_filter1d(flux, 3, mode='constant', cval=0)

        if camera=='B':
            if label is not None:
                plot_label = label
            else:
                plot_label = 'TID={}\nZ={:.4f}  TYPE={}  ZWARN={}  DELTACHI2={:.1f}'.format(tid, z, redshifts['SPECTYPE'][coadd_index], redshifts['ZWARN'][coadd_index], redshifts['DELTACHI2'][coadd_index])
        else:
            plot_label = None

        ax1.plot(wave, flux_smooth, lw=lw, label=plot_label)
        if show_model:
            if gauss_smooth==0 or gauss_smooth is None:
                model_flux_smooth = model_flux[camera].copy()
            elif gauss_smooth>0:
                model_flux_smooth = gaussian_filter1d(model_flux[camera], 3, mode='constant', cval=0)
            ax1.plot(wave, model_flux_smooth, lw=lw, color='r', alpha=0.65)

    if show_lines:
        for line in lines.keys():
            if (lines[line]*(1+z)>3400) & (lines[line]*(1+z)<10000):
                ax1.axvline(lines[line]*(1+z), lw=lw, color='r', alpha=0.3)
                ax1.text(lines[line]*(1+z), -0.8, line)
    ax1.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    ax1.set_xlabel('observed wavelength ($\AA$)')
    # plt.axvline(4000, ls='--', lw=1, color='k')
    ax1.legend(loc='upper left', handletextpad=.0, handlelength=0)
    ax1.grid()
    if show_restframe:
        ax2 = ax1.twiny()
        ax2.set_xlim(3400/(1+z), 10000/(1+z))
        ax2.set_xlabel('restframe wavelength ($\AA$)')
    plt.tight_layout()
    # plt.savefig('/global/cfs/cdirs/desi/users/rongpu/plots/lrg_speed/spectra_low_speed_failures/{}_deep.png'.format(tid))
    if show:
        plt.show()

    if return_ax:
        if show_restframe:
            return ax1, ax2
        else:
            return ax1


# coadd_fn = '/global/cfs/cdirs/desi/spectro/redux/everest/tiles/cumulative/80605/20210205/coadd-0-80605-thru20210205.fits'
# tid = 39627640566453451
# plot_spectrum(coadd_fn, tid, use_targetid=True)

