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

tracer = 'ELG'
cat = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/jura_data/{}.fits'.format(tracer.lower())))

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

# Apply ELG mask
main_dir = '/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve'
tmp = Table(fitsio.read(os.path.join(main_dir, 'dr9_elg_1.1.1_basic.fits'), columns=['TARGETID']))
tmp2 = Table(fitsio.read(os.path.join(main_dir, 'dr9_elg_1.1.1_elgmask_v1.fits.gz')))
elg = hstack([tmp, tmp2])

print(len(elg))
mask = elg['elg_mask']==0
elg = elg[mask]
print(len(elg))

mask = np.in1d(cat['TARGETID'], elg['TARGETID'])
cat = cat[mask]
print('ELG mask   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))

fiberstats = Table()
fiberstats['FIBER'], fiberstats['n_tot'] = np.unique(cat['FIBER'], return_counts=True)

print(np.median(fiberstats['n_tot']))

pvalue_threshold = 1e-7

# # # outliers = []
# outliers = [53, 55, 56, 59, 64, 69, 185, 299, 333, 436, 551, 552, 553, 556, 562, 650, 675, 700, 725, 793, 817, 918, 919, 920, 997, 998, 1008, 1089, 1098, 1113, 1124, 1149, 1157, 1174, 1261, 1269, 1295, 1296, 1659, 1711, 1725, 1733, 1735, 1740, 1743, 1750, 1754, 1774, 1824, 1825, 1849, 1850, 1874, 1875, 1899, 1900, 1950, 1971, 1972, 2022, 2098, 2099, 2124, 2131, 2138, 2148, 2170, 2199, 2249, 2263, 2315, 2487, 2623, 2627, 2775, 2782, 2824, 2841, 2875, 2899, 2967, 3234, 3235, 3237, 3238, 3240, 3241, 3245, 3246, 3250, 3251, 3253, 3275, 3358, 3359, 3399, 3504, 3506, 3510, 3514, 3536, 3618, 3700, 3825, 3875, 3892, 3969, 3983, 3993, 3994, 4275, 4300, 4324, 4325, 4374, 4524, 4549, 4574, 4649, 4699, 4720, 4775, 4787, 4799, 4874, 4875, 4891, 4899, 4915]

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

# fiberstats.write('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/jura_data/ks_fiberstats_{}.fits'.format(tracer.lower()))

fiberstats = Table(fitsio.read('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/jura_data/ks_fiberstats_{}.fits'.format(tracer.lower())))
print(len(fiberstats))

mask_outlier = fiberstats['pvalue']<pvalue_threshold
mask_outlier &= fiberstats['n_tot']>=min_nobs
outliers = np.array(np.sort(fiberstats['FIBER'][mask_outlier]))
print(len(outliers))

# mask_outlier = np.in1d(fiberstats['FIBER'], outliers)
# print(np.sum(mask_outlier))

# fiberstats.sort('pvalue')
# fiberstats[mask_outlier]

# fiberstats.sort('FIBER')
# fiberstats[mask_outlier]

datemin = datetime.strptime(str(cat['LASTNIGHT'].min()), '%Y%m%d') - timedelta(days=20)
datemax = datetime.strptime(str(cat['LASTNIGHT'].max()), '%Y%m%d') + timedelta(days=20)

bin_size = 0.02
bins = np.arange(-0.01, 10, bin_size)
bin_centers = (bins[1:]+bins[:-1])/2

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/per_fiber_qa/jura/per_fiber_redshifts_{}_zcatalog.pdf'.format(tracer.lower())) as pdf:
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

