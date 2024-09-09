from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

from scipy.stats import kstest
from scipy.interpolate import interp1d

from datetime import datetime, timedelta

params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'figure.facecolor': 'w'}
plt.rcParams.update(params)

min_nobs = 50

tracer = 'LRG'
cat = Table(fitsio.read('/dvs_ro/cfs/cdirs/desicollab/users/rongpu/redshift_qa/kibo_data/{}.fits'.format(tracer.lower())))

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

# Require a minimum depth for the cat coadd
if tracer=='BGS_ANY':
    min_depth = 160
    mask = cat['EFFTIME_BGS']>min_depth
else:
    min_depth = 800.
    mask = cat['EFFTIME_LRG']>min_depth
print('Min depth   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
cat = cat[mask]

# if tracer=='LRG':
#     # Apply maskbits
#     maskbits = [1, 8, 9, 11, 12, 13]
#     mask = np.ones(len(cat), dtype=bool)
#     for bit in maskbits:
#         mask &= (cat['MASKBITS'] & 2**bit)==0
#     print('MASKBITS  ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
#     cat = cat[mask]
# elif tracer=='ELG':
#     # Apply maskbits
#     maskbits = [1, 11, 12, 13]
#     mask = np.ones(len(cat), dtype=bool)
#     for bit in maskbits:
#         mask &= (cat['MASKBITS'] & 2**bit)==0
#     print('MASKBITS  ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
#     cat = cat[mask]

# Apply LRG mask
main_dir = '/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve'
tmp = Table(fitsio.read(os.path.join(main_dir, 'dr9_lrg_1.1.1_basic.fits'), columns=['TARGETID']))
tmp2 = Table(fitsio.read(os.path.join(main_dir, 'dr9_lrg_1.1.1_lrgmask_v1.1.fits.gz')))
lrg = hstack([tmp, tmp2])

print(len(lrg))
mask = lrg['lrg_mask']==0
lrg = lrg[mask]
print(len(lrg))

mask = np.in1d(cat['TARGETID'], lrg['TARGETID'])
cat = cat[mask]
print('LRG mask   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))

fiberstats = Table()
fiberstats['FIBER'], fiberstats['n_tot'] = np.unique(cat['FIBER'], return_counts=True)

print(np.median(fiberstats['n_tot']))

pvalue_threshold = 1e-4

# # outliers = []
# outliers = [552, 553, 725, 1008, 3234, 3235, 3250, 3504, 3969, 3994]

# for ii in range(2):

#     print('iteration', ii)

#     mask = ~np.in1d(cat['FIBER'], outliers)
#     allz = np.sort(np.array(cat['Z'][mask]))
#     x = allz.copy()
#     y = np.linspace(0, 1, len(x))
#     cdf = interp1d(x, y, fill_value=(0, 1), bounds_error=False)

#     fiberstats['pvalue'] = 0.
#     for index, fiber in enumerate(fiberstats['FIBER']):
#         # if index%1000==0:
#         #     print(index, len(fiberstats))
#         mask = cat['FIBER']==fiberstats['FIBER'][index]
#         fiberstats['pvalue'][index] = kstest(cat['Z'][mask], cdf).pvalue

#     mask_outlier = fiberstats['pvalue']<pvalue_threshold
#     mask_outlier &= fiberstats['n_tot']>=min_nobs
#     print(np.sum(mask_outlier))
#     outliers = np.array(np.sort(fiberstats['FIBER'][mask_outlier]))
#     print('Outlier fibers:', list(outliers))
#     print()

# fiberstats.write('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/kibo_data/ks_fiberstats_{}.fits'.format(tracer.lower()))

fiberstats = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/kibo_data/ks_fiberstats_{}.fits'.format(tracer.lower())))
print(len(fiberstats))

mask_outlier = fiberstats['pvalue']<pvalue_threshold
mask_outlier &= fiberstats['n_tot']>=min_nobs
outliers = np.array(np.sort(fiberstats['FIBER'][mask_outlier]))
print(len(outliers))

mask_outlier = np.in1d(fiberstats['FIBER'], outliers)
print(np.sum(mask_outlier))

fiberstats.sort('pvalue')
fiberstats[mask_outlier]

fiberstats.sort('FIBER')
fiberstats[mask_outlier]

datemin = datetime.strptime(str(cat['LASTNIGHT'].min()), '%Y%m%d') - timedelta(days=20)
datemax = datetime.strptime(str(cat['LASTNIGHT'].max()), '%Y%m%d') + timedelta(days=20)

bin_size = 0.02
bins = np.arange(-0.01, 10, bin_size)
bin_centers = (bins[1:]+bins[:-1])/2

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/per_fiber_qa/kibo/per_fiber_redshifts_{}_zcatalog.pdf'.format(tracer.lower())) as pdf:
    # for ii in range(1):
    for ii in range(1000):
        fig, axes = plt.subplots(5, 2, figsize=(20, 24), gridspec_kw={'width_ratios': [0.55, 1.5]})
        for fiber in range(ii*5, ii*5+5):
            if fiber%100==0:
                print('FIBER {}'.format(fiber))
            mask = cat['FIBER']==fiber
            if np.sum(mask)==0:
                ax = axes[fiber%5, 0]
                ax.set_title('FIBER {} {}'.format(fiber, tracer))
                ax = axes[fiber%5, 1]
                ax.set_title('0 objects')
                continue
            index = np.where(fiberstats['FIBER']==fiber)[0][0]

            ax = axes[fiber%5, 0]
            vmin, vmax = cat['Z'][mask].min(), cat['Z'][mask].max()
            vmin = vmin - (vmax-vmin)*0.05
            vmax = vmax + (vmax-vmin)*0.05
            ax.hist(bin_size*np.digitize(cat['Z'][mask], bins=bins), bins=bins, label='single fiber', rasterized=True)
            mask1 = ~np.in1d(cat['FIBER'], fiberstats['FIBER'][mask_outlier])
            ax.hist(bin_size*np.digitize(cat['Z'][mask1], bins=bins), bins=bins, histtype='step', weights=np.full(np.sum(mask1), np.sum(mask)/np.sum(mask1)), color='red', alpha=1., label='all good fibers', rasterized=True)
            ax.set_title('FIBER {} {}  p-value = {:.5g}'.format(fiber, tracer, fiberstats['pvalue'][index]))
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

