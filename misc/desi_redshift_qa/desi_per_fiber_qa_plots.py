# salloc -N 1 -C cpu -t 04:00:00 -q interactive python desi_per_fiber_qa_plots.py -v kibo -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo --tracer LRG --stats /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo/per_fiber_qa_stats.fits
# salloc -N 1 -C cpu -t 04:00:00 -q interactive python desi_per_fiber_qa_plots.py -v kibo -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo --tracer ELG_LOP --stats /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo/per_fiber_qa_stats.fits
# salloc -N 1 -C cpu -t 04:00:00 -q interactive python desi_per_fiber_qa_plots.py -v kibo -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo --tracer ELG_VLO --stats /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo/per_fiber_qa_stats.fits
# salloc -N 1 -C cpu -t 04:00:00 -q interactive python desi_per_fiber_qa_plots.py -v kibo -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo --tracer QSO --stats /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo/per_fiber_qa_stats.fits
# salloc -N 1 -C cpu -t 04:00:00 -q interactive python desi_per_fiber_qa_plots.py -v kibo -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo --tracer BGS_BRIGHT --stats /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo/per_fiber_qa_stats.fits
# salloc -N 1 -C cpu -t 04:00:00 -q interactive python desi_per_fiber_qa_plots.py -v kibo -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo --tracer BGS_FAINT --stats /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo/per_fiber_qa_stats.fits

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

from datetime import datetime, timedelta
from desitarget.targetmask import desi_mask, bgs_mask

import argparse

import time
time_start = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', type=str, help="redux version", required=True)
parser.add_argument('-o', '--output', type=str, help="output directory path", required=True)
parser.add_argument('--tracer', type=str, help="tracer type, e.g., LRG", required=True)
parser.add_argument('--stats', type=str, help="per-fiber stats path", required=False)
args = parser.parse_args()

output_dir = args.output
version = args.version
tracer = args.tracer
stats_fn = args.stats

pvalue_threshold = 1e-4
save_pdf = False
save_png = True

if stats_fn is not None:
    stats = Table(fitsio.read(stats_fn))

# dirname = '/dvs_ro/cfs/cdirs/desi/spectro/redux/{}/zcatalog/v1/ztile-main-dark-cumulative.fits'.format(version)
dirname = '/pscratch/sd/r/rongpu/zcatalog'

print(tracer)

if tracer in ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_VLO', 'BGS_ANY']:
    fn = os.path.join(dirname, 'ztile-main-dark-cumulative-basic.fits')
    cat = Table(fitsio.read(fn))
    mask = np.where(cat['DESI_TARGET'] & desi_mask[tracer] > 0)[0]
    cat = cat[mask]
else:
    fn = os.path.join(dirname, 'ztile-main-bright-cumulative-basic.fits')
    cat = Table(fitsio.read(fn))
    mask = np.where(cat['BGS_TARGET'] & bgs_mask[tracer] > 0)[0]
    cat = cat[mask]

print(tracer, 'zcat', len(cat))
stats_zcat = Table()
stats_zcat['FIBER'], stats_zcat[tracer.lower()+'_zcat_n_tot'] = np.unique(cat['FIBER'], return_counts=True)

cat['EFFTIME_BGS'] = 0.1400 * cat['TSNR2_BGS']
cat['EFFTIME_LRG'] = 12.15 * cat['TSNR2_LRG']

# Remove FIBERSTATUS!=0 fibers
mask = cat['COADD_FIBERSTATUS']==0
print('FIBERSTATUS   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
cat = cat[mask]

# Remove "no data" fibers
mask = cat['ZWARN'] & 2**9==0
print('No data   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
cat = cat[mask]

# Require a minimum depth
if tracer in ['BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT']:
    min_depth = 160
    mask = cat['EFFTIME_BGS']>min_depth
else:
    min_depth = 800.
    mask = cat['EFFTIME_LRG']>min_depth
print('Min depth   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
cat = cat[mask]

# Apply masks
if tracer=='LRG':
    tmp1 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_basic.fits'), columns=['TARGETID']))
    tmp2 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_lrgmask_v1.1.fits.gz')))
    lrgmask = hstack([tmp1, tmp2])
    lrgmask = lrgmask[lrgmask['lrg_mask']==0]
    mask = np.in1d(cat['TARGETID'], lrgmask['TARGETID'])
    print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]
elif tracer in ['ELG', 'ELG_LOP', 'ELG_VLO']:
    tmp1 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_basic.fits'), columns=['TARGETID']))
    tmp2 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_elgmask_v1.fits.gz')))
    elgmask = hstack([tmp1, tmp2])
    elgmask = elgmask[elgmask['elg_mask']==0]
    mask = np.in1d(cat['TARGETID'], elgmask['TARGETID'])
    print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

# Remove QSO reobservations
if tracer=='QSO':
    mask = cat['PRIORITY']==3400
    print('Remove QSO reobservations', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

good_z_col = 'GOOD_' + tracer.split('_')[0]
print(tracer, 'average failure rate', np.sum(~cat[good_z_col])/len(cat))

datemin = datetime.strptime(str(cat['LASTNIGHT'].min()), '%Y%m%d') - timedelta(days=20)
datemax = datetime.strptime(str(cat['LASTNIGHT'].max()), '%Y%m%d') + timedelta(days=20)

if tracer=='QSO':
    bin_size = 0.04
else:
    bin_size = 0.02
bins = np.arange(-0.01, 10, bin_size)
bin_centers = (bins[1:]+bins[:-1])/2

for apply_good_z_cut in [True, False]:

    if apply_good_z_cut:
        output_fn = os.path.join(output_dir, 'per_fiber_redshifts_{}_goodz.pdf'.format(tracer.lower()))
    else:
        output_fn = os.path.join(output_dir, 'per_fiber_redshifts_{}_allz.pdf'.format(tracer.lower()))
    print(output_fn)

    if stats_fn is not None:
        if apply_good_z_cut:
            pvalue_col = tracer.lower()+'_ks_pvalue_goodz'
        else:
            pvalue_col = tracer.lower()+'_ks_pvalue_allz'
        mask_outlier = stats[pvalue_col]<pvalue_threshold
        outliers = np.array(np.sort(stats['FIBER'][mask_outlier]))
    else:
        outliers = []

    if save_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output_fn) as pdf:
            # for ii in range(1):
            for ii in range(1000):
                fig, axes = plt.subplots(5, 2, figsize=(20, 24), gridspec_kw={'width_ratios': [0.55, 1.5]})
                for fiber in range(ii*5, ii*5+5):
                    if fiber%100==0:
                        print('FIBER {}'.format(fiber))
                    mask = cat['FIBER']==fiber
                    if apply_good_z_cut:
                        mask &= cat[good_z_col]
                    if np.sum(mask)==0:
                        ax = axes[fiber%5, 0]
                        ax.set_title('FIBER {} {}'.format(fiber, tracer))
                        ax = axes[fiber%5, 1]
                        ax.set_title('0 objects')
                        continue

                    ax = axes[fiber%5, 0]
                    vmin, vmax = cat['Z'][mask].min(), cat['Z'][mask].max()
                    vmin = vmin - (vmax-vmin)*0.05
                    vmax = vmax + (vmax-vmin)*0.05
                    ax.hist(bin_size*np.digitize(cat['Z'][mask], bins=bins), bins=bins, label='single fiber', rasterized=True)
                    mask1 = ~np.in1d(cat['FIBER'], outliers)
                    if apply_good_z_cut:
                        mask1 &= cat[good_z_col]
                    ax.hist(bin_size*np.digitize(cat['Z'][mask1], bins=bins), bins=bins, histtype='step', weights=np.full(np.sum(mask1), np.sum(mask)/np.sum(mask1)), color='red', alpha=1., label='all good fibers', rasterized=True)

                    if stats_fn is not None:
                        index = np.where(stats['FIBER']==fiber)[0][0]
                        ax.set_title('FIBER {} {}  p-value = {:.5g}'.format(fiber, tracer, stats[pvalue_col][index]))
                    else:
                        ax.set_title('FIBER {} {}'.format(fiber, tracer))
                    ax.set_xlabel('Redshift')
                    ax.set_xlim(vmin, vmax)
                    ax.legend(loc='upper right')

                    ax = axes[fiber%5, 1]
                    cat1 = cat[mask]
                    dates = np.asarray([datetime.strptime(str(night), '%Y%m%d') for night in cat1['LASTNIGHT']])
                    ax.plot(dates, cat1['Z'], '.', alpha=np.minimum(0.5, 1.*600/len(cat1)), rasterized=True)
                    ax.set_title('{} objects'.format(len(cat1)))
                    ax.set_xlabel('date')
                    ax.set_ylabel('Redshift')
                    ax.set_xlim(datemin, datemax)
                    plt.tight_layout()
                pdf.savefig(dpi=50)
                plt.close()
        print('PDF done!', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))

    if save_png:

        if apply_good_z_cut:
            png_dir = os.path.join(output_dir, 'png', '{}_goodz'.format(tracer.lower()))
        else:
            png_dir = os.path.join(output_dir, 'png', '{}_allz'.format(tracer.lower()))
        if not os.path.isdir(png_dir):
            os.makedirs(png_dir)

        def write_png(fiber):
            print('FIBER {}'.format(fiber))
            fig, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [0.55, 1.5]})
            mask = cat['FIBER']==fiber
            if apply_good_z_cut:
                mask &= cat[good_z_col]

            if np.sum(mask)==0:
                ax = axes[0]
                ax.set_title('FIBER {} {}'.format(fiber, tracer))
                ax = axes[1]
                ax.set_title('0 objects')
            else:
                ax = axes[0]
                vmin, vmax = cat['Z'][mask].min(), cat['Z'][mask].max()
                vmin = vmin - (vmax-vmin)*0.05
                vmax = vmax + (vmax-vmin)*0.05
                ax.hist(bin_size*np.digitize(cat['Z'][mask], bins=bins), bins=bins, label='single fiber')
                mask1 = ~np.in1d(cat['FIBER'], outliers)
                if apply_good_z_cut:
                    mask1 &= cat[good_z_col]
                ax.hist(bin_size*np.digitize(cat['Z'][mask1], bins=bins), bins=bins, histtype='step', weights=np.full(np.sum(mask1), np.sum(mask)/np.sum(mask1)), color='red', alpha=1., label='all good fibers')

                if stats_fn is not None:
                    index = np.where(stats['FIBER']==fiber)[0][0]
                    ax.set_title('FIBER {} {}  p-value = {:.5g}'.format(fiber, tracer, stats[pvalue_col][index]))
                else:
                    ax.set_title('FIBER {} {}'.format(fiber, tracer))
                ax.set_xlabel('Redshift')
                ax.set_xlim(vmin, vmax)
                ax.legend(loc='upper right')

                ax = axes[1]
                cat1 = cat[mask]
                dates = np.asarray([datetime.strptime(str(night), '%Y%m%d') for night in cat1['LASTNIGHT']])
                ax.plot(dates, cat1['Z'], '.', alpha=np.minimum(0.5, 1.*600/len(cat1)))
                ax.set_title('{} objects'.format(len(cat1)))
            ax.set_xlabel('Date')
            ax.set_ylabel('Redshift')
            ax.set_xlim(datemin, datemax)
            plt.tight_layout()
            plt.savefig(os.path.join(png_dir, 'FIBER_{}.png'.format(fiber)))
            plt.close()

            return None

        from multiprocessing import Pool
        n_process = 128
        with Pool(processes=n_process) as pool:
            res = pool.map(write_png, np.arange(5000), chunksize=1)

        print('PNGs done!', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))

print('Done!', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))