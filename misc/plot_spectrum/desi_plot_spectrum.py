from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize': 'x-large',
         'xtick.labelsize': 'x-large',
         'ytick.labelsize': 'x-large',
         'figure.facecolor': 'w'}
plt.rcParams.update(params)


def get_rr_model(coadd_fn, index, redrock_fn=None, use_targetid=False, coadd_cameras=False, restframe=False, z=None, return_z=False):
    '''
    Return redrock model spectrum.

    Args:
       coadd_fn: str, path of coadd FITS file
       index: int, index of coadd FITS file if use_targetid=False, or TARGETID if use_targetid=True

    Options:
       redrock_fn, str, path of redrock FITS file
       use_targetid: bool, if True, index is TARGETID
       coadd_cameras: bool, if True, the BRZ cameras are coadded together
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

    if redrock_fn is None:
        redrock_fn = coadd_fn.replace('/coadd-', '/redrock-')
    redshifts = Table(fitsio.read(redrock_fn, ext='REDSHIFTS'))

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
        if coadd_cameras:
            cameras = ['BRZ']
        else:
            cameras = ['B', 'R', 'Z']
        for camera in cameras:
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


def plot_spectrum(coadd_fn, index, redrock_fn=None, use_targetid=False, coadd_cameras=False,
    show_lines=True, show_restframe=True, show_model=True, figsize=(22, 5), lw=1.2, gauss_smooth=3,
    label=None, title=None, show=True, return_ax=False, xlim=[3400, 10000], ylim=None, grid=False,
    save_path=None, mags=None):
    '''
    Plot DESI spectrum.

    Args:
       coadd_fn: str, path of coadd FITS file
       index: int, index of coadd FITS file if use_targetid=False, or TARGETID if use_targetid=True

    Options:
       use_targetid: bool, if True, index is TARGETID
       redrock_fn, str, path of redrock FITS file
       use_targetid: bool, if True, index is TARGETID
       coadd_cameras: bool, if True, the BRZ cameras are coadded together
    '''

    # from scipy.ndimage import gaussian_filter1d
    from astropy.convolution import Gaussian1DKernel, convolve
    from matplotlib.ticker import AutoMinorLocator

    # lines = {
    #     'Ha'      : 6562.8,
    #     'Hb'       : 4862.68,
    #     'Hg'       : 4340.464,
    #     'Hd'       : 4101.734,
    #     # 'OIII-b'       :  5006.843,
    #     # 'OIII-a'       : 4958.911,
    #     'OIII': 4982.877,
    #     'MgII'    : 2799.49,
    #     'OII'         : 3728,
    #     'CIII'  : 1909.,
    #     'CIV'    : 1549.06,
    #     'SiIV'  : 1393.76018,
    #     'LYA'         : 1215.67,
    #     'LYB'         : 1025.72
    # }
    
    lines = [
    # Major absorption lines
    ['K', 3933.7, 1],
    ['H', 3968.5, 2],
    ['G', 4307.74, 0],
    ['Mg I', 5175.0, 0],
    ['D2', 5889.95, 0],
    ['D1', 5895.92, 1],

    # Major emission lines
    [r'Ly$\alpha$', 1215.67, 0],
    ['C IV', 1549.48, 0],
    ['C III]', 1908.734, 0],
    ['Mg II', 2796.3543, 0],
    ['Mg II', 2803.5315, 1],
    # ['[O II]', 3726.032, 0],
    # ['[O II]', 3728.815, 1],
    ['[O II]', 3727.424, 0],
    ['[Ne III]', 3868.76, 0],
    [r'H$\delta$', 4101.734, 0],
    [r'H$\gamma$', 4340.464, 1],
    [r'H$\beta$', 4861.325, 0],
    ['[O III]', 4958.911, 0],
    ['[O III]', 5006.843, 1],
    [r'H$\alpha$', 6562.801, 0],
    ['[N II]', 6583.45, 1],
    ['[S II]', 6716.44, 0],
    ['[S II]', 6730.82, 1],
    ]

    line_names = [tmp[0] for tmp in lines]

    tmp = fitsio.read(coadd_fn, columns=['TARGETID'], ext='FIBERMAP')

    if use_targetid:
        tid = index
        coadd_index = np.where(tmp['TARGETID']==index)[0][0]
    else:
        tid = tmp['TARGETID'][index]
        coadd_index = index

    if show_model:
        # Get model spectrum
        _, model_flux = get_rr_model(coadd_fn, coadd_index, redrock_fn=redrock_fn, coadd_cameras=coadd_cameras)

    if redrock_fn is None:
        redrock_fn = coadd_fn.replace('/coadd-', '/redrock-')
    redshifts = Table(fitsio.read(redrock_fn, ext='REDSHIFTS'))
    fibermap = Table(fitsio.read(redrock_fn, ext='FIBERMAP'))
    fibermap.remove_column('TARGETID')
    redshifts = hstack([redshifts, fibermap])
    z = redshifts['Z'][coadd_index]

    if mags is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmag = 22.5 - 2.5*np.log10(redshifts['FLUX_G']) - 3.214 * redshifts['EBV']
            rmag = 22.5 - 2.5*np.log10(redshifts['FLUX_R']) - 2.165 * redshifts['EBV']
            zmag = 22.5 - 2.5*np.log10(redshifts['FLUX_Z']) - 1.211 * redshifts['EBV']
            w1mag = 22.5 - 2.5*np.log10(redshifts['FLUX_W1']) - 0.184 * redshifts['EBV']
            w2mag = 22.5 - 2.5*np.log10(redshifts['FLUX_W2']) - 0.113 * redshifts['EBV']
            gfibermag = 22.5 - 2.5*np.log10(redshifts['FIBERFLUX_G']) - 3.214 * redshifts['EBV']
            rfibermag = 22.5 - 2.5*np.log10(redshifts['FIBERFLUX_R']) - 2.165 * redshifts['EBV']
            zfibermag = 22.5 - 2.5*np.log10(redshifts['FIBERFLUX_Z']) - 1.211 * redshifts['EBV']
    else:
        gmag, rmag, zmag, w1mag, w2mag, gfibermag, rfibermag, zfibermag = mags
    ymin, ymax = 0., 0.

    fig, ax1 = plt.subplots(figsize=figsize)
    if coadd_cameras:
        cameras = ['BRZ']
    else:
        cameras = ['B', 'R', 'Z']
    for camera in cameras:
        wave = fitsio.read(coadd_fn, ext=camera+'_WAVELENGTH')
        # wave_rest = wave/(1+z)
        flux = fitsio.read(coadd_fn, ext=camera+'_FLUX')[coadd_index]
        msk = fitsio.read(coadd_fn, ext=camera+'_MASK')[coadd_index]
        ivar = fitsio.read(coadd_fn, ext=camera+'_IVAR')[coadd_index]
        bad = msk!=0
        # flux[bad] = 0.
        flux[bad] = np.nan
        if show_model:
            model_flux[camera][bad] = np.nan
        # if np.sum(bad)!=0:
        #     print('{} masked pixels: {}'.format(camera, np.sum(bad)))

        if gauss_smooth==0 or gauss_smooth is None:
            flux_smooth = flux.copy()
        elif gauss_smooth>0:
            # flux_smooth = gaussian_filter1d(flux, gauss_smooth, mode='constant', cval=0)
            gauss_kernel = Gaussian1DKernel(stddev=gauss_smooth)
            flux_smooth = convolve(flux, gauss_kernel, boundary='extend')

        if label is not None:
            plot_label = label
        else:
            plot_label = 'TARGETID={}'.format(tid)
            plot_label += '  g={:.2f} r={:.2f} z={:.2f} W1={:.2f} zfiber={:.2f}'.format(gmag, rmag, zmag, w1mag, zfibermag)
            plot_label += '\nRedshift={:.4f}  TYPE={}  ZWARN={}  DELTACHI2={:.1f}'.format(
                z, redshifts['SPECTYPE'][coadd_index], redshifts['ZWARN'][coadd_index], redshifts['DELTACHI2'][coadd_index])

        if camera=='B' or camera=='BRZ':
            label_data, label_model = 'data', 'best-fit model'
        else:
            label_data, label_model = None, None
        if camera=='BRZ':
            color = 'C0'
        else:
            color = None
        ax1.plot(wave, flux_smooth, lw=lw, label=label_data, color=color)
        if show_model:
            if gauss_smooth==0 or gauss_smooth is None:
                model_flux_smooth = model_flux[camera].copy()
            elif gauss_smooth>0:
                # model_flux_smooth = gaussian_filter1d(model_flux[camera], gauss_smooth, mode='constant', cval=0)
                model_flux_smooth = convolve(model_flux[camera], gauss_kernel, boundary='extend')
            ax1.plot(wave, model_flux_smooth, lw=3, color='r', alpha=0.4, label=label_model)
        ymin = np.minimum(ymin, 1.3 * np.percentile(flux_smooth[np.isfinite(flux_smooth)], 1.))
        ymax = np.maximum(ymax, 1.3 * np.percentile(flux_smooth[np.isfinite(flux_smooth)], 99.))

    if ylim is None:
        ylim = [ymin, ymax]
    if show_lines:
        for line_index in range(len(lines)):
            line_name, line_wavelength, text_offset = lines[line_index]
            if (line_wavelength*(1+z)>3400) & (line_wavelength*(1+z)<10000):
                ax1.axvline(line_wavelength*(1+z), lw=lw, color='C2', alpha=1, ls='--')
                text_yposition = 0.96*ylim[0]+0.04*ylim[1]
                text_yposition += 0.05*(ylim[1]-ylim[0])*text_offset
                ax1.text(line_wavelength*(1+z)+7, text_yposition, line_name, fontsize=plt.rcParams['legend.fontsize'])
    ax1.text(3450, 0.04*ylim[0]+0.96*ylim[1], plot_label, verticalalignment='top',
            fontsize=plt.rcParams['legend.fontsize'], bbox=dict(facecolor='white', alpha=0.7))
    ax1.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    ax1.set_xlabel('observed wavelength ($\AA$)')
    # plt.axvline(4000, ls='--', lw=1, color='k')
    # ax1.legend(loc='upper left', handletextpad=.0, handlelength=0)
    ax1.legend(loc=(0.885, 0.16))
    if grid:
        ax1.grid()
    if title is not None:
        ax1.set_title(title)
    if show_restframe:
        ax2 = ax1.twiny()
        ax2.set_xlim(3400/(1+z), 10000/(1+z))
        ax2.set_xlabel('restframe wavelength ($\AA$)')
        ax1.set_ylabel('flux ($10^{-17}$ ergs/s/cm$^2$/$\AA$)')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="both", which='both', direction="in", top=False, right=True)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(which='both', direction="in")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
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


