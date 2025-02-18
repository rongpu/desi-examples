# Example:
# sys.path.append(os.path.expanduser('~/git/desi-examples/imaging_systematics'))
# from plot_healpix_map import plot_map
# nside = 128
# maps = Table(fitsio.read('/global/cfs/cdirs/desi/users/rongpu/data/imaging_sys/randoms_stats/0.49.0/resolve/combined/pixmap_south_nside_128_minobs_1_maskbits__lrgmask_v1.1.fits'))
# plot_map(nside, maps['PSFSIZE_G'], pix=maps['HPXPIXEL'], cmap='viridis', nest=False, vmin=1.0, vmax=2., show=True)

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

from astropy.coordinates import SkyCoord
from astropy import units
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

from IPython.display import Image, display

# params = {'figure.facecolor': 'w'}
# plt.rcParams.update(params)

default_dpi = {32: 100, 64: 200, 128: 400, 256: 600, 512: 1200}
default_xsize = {32: 1500, 64: 4000, 128: 4000, 256: 6000, 512: 12000}


def plot_map(nside, v, pix=None, vmin=None, vmax=None, cmap='jet', title=None, save_path=None,
             xsize=None, dpi=None, show=True, timing=True, nest=False, coord=None, cbar_label='',
             fontsize=9.5, overwrite=True):

    if os.path.isfile(save_path) and overwrite is False:
        print('Plot exists:', save_path, '  Skip')
        return None

    if xsize is None:
        xsize = default_xsize[nside]

    if dpi is None:
        dpi = default_dpi[nside]

    # Font sizes for healpix maps
    fontsize_dict = {
        "xlabel": fontsize,
        "ylabel": fontsize,
        "title": fontsize,
        "xtick_label": fontsize,
        "ytick_label": fontsize,
        "cbar_label": fontsize,
        "cbar_tick_label": fontsize,
    }

    npix = hp.nside2npix(nside)

    v = np.array(v).astype(float)

    # Density map
    hp_mask = np.zeros(npix, dtype=bool)
    if pix is None:
        map_values = v.copy()
        hp_mask[np.isfinite(map_values)] = True
    else:
        map_values = np.zeros(npix, dtype=v.dtype)
        map_values[pix] = v
        hp_mask[pix] = True
    mplot = hp.ma(map_values)
    mplot.mask = ~hp_mask

    # Galactic plane
    org = 120
    tmpn = 1000
    cs = SkyCoord(l=np.linspace(0, 360, tmpn) * units.deg, b=np.zeros(tmpn) * units.deg, frame="galactic")
    ras, decs = cs.icrs.ra.degree, cs.icrs.dec.degree
    ras = np.remainder(ras + 360 - org, 360)  # shift ra values
    ras[ras > 180] -= 360  # scale conversion to [-180, 180]
    ii = ras.argsort()
    ras, decs = ras[ii], decs[ii]

    if timing:
        time_start = time.time()

    projview(mplot, min=vmin, max=vmax,
             rot=(120, 0, 0), coord=coord, cmap=cmap, xsize=xsize,
             graticule=True, graticule_labels=True, projection_type="mollweide", nest=nest,
             title=title,
             xlabel='RA', ylabel='Dec',
             custom_xtick_labels=[r'$240\degree$', r'$180\degree$', r'$120\degree$', r'$60\degree$', r'$0\degree$'],
             fontsize=fontsize_dict, unit=cbar_label)
    newprojplot(theta=np.radians(90-decs), phi=np.radians(ras), color='k', lw=1)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        # if save_path is None:
        #     plt.savefig("tmp.png", bbox_inches="tight", dpi=dpi)
        #     tmp = Image("tmp.png")
        #     display(tmp)
        #     plt.close()
        # else:
        #     plt.show()
        plt.show()
    else:
        plt.close()

    if timing:
        print('Done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

