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

# Font sizes for healpix maps
fontsize_dict = {
    "xlabel": 9.5,
    "ylabel": 9.5,
    "title": 9.5,
    "xtick_label": 9.5,
    "ytick_label": 9.5,
    "cbar_label": 9.5,
    "cbar_tick_label": 9.5,
}

default_dpi = {64: 200, 128: 400, 256: 600, 512: 1600}
default_xsize = {64: 4000, 128: 4000, 256: 6000, 512: 16000}


def plot_map(nside, pix, v, vmin=None, vmax=None, cmap='jet', title=None, save_path=None,
             xsize=None, dpi=None, show=True, timing=True, nest=False):

    if xsize is None:
        xsize = default_xsize[nside]

    if dpi is None:
        dpi = default_dpi[nside]

    npix = hp.nside2npix(nside)

    v = np.array(v)

    # Density map
    map_values = np.zeros(npix, dtype=v.dtype)
    hp_mask = np.zeros(npix, dtype=bool)
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
             rot=(120, 0, 0), cmap=cmap, xsize=xsize,
             graticule=True, graticule_labels=True, projection_type="mollweide", nest=nest,
             title=title,
             xlabel='RA (deg)', ylabel='Dec (deg)',
             custom_xtick_labels=[r'$240\degree$', r'$180\degree$', r'$120\degree$', r'$60\degree$', r'$0\degree$'],
             fontsize=fontsize_dict)
    newprojplot(theta=np.radians(90-decs), phi=np.radians(ras), color='k', lw=1)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

    if timing:
        print('Done!', time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

