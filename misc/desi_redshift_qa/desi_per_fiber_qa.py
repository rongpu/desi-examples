# Plot QA plots per-fiber redshift performance
# Examples:
# python desi_per_fiber_qa.py -t BGS_ANY -v kibo -o /global/cfs/cdirs/desi/users/rongpu/redshift_qa/per_fiber_qa/kibo
# python desi_per_fiber_qa.py -t LRG -v kibo -o /global/cfs/cdirs/desi/users/rongpu/redshift_qa/per_fiber_qa/kibo
# # python desi_per_fiber_qa.py -t ELG -v kibo -o /global/cfs/cdirs/desi/users/rongpu/redshift_qa/per_fiber_qa/kibo
# # python desi_per_fiber_qa.py -t QSO -v kibo -o /global/cfs/cdirs/desi/users/rongpu/redshift_qa/per_fiber_qa/kibo

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tracer', required=True)
parser.add_argument('-v', '--version', default='iron', required=False)
parser.add_argument('-o', '--output', default='', required=False)
parser.add_argument('--min_nobs', default=50, type=int, required=False)
parser.add_argument('--fail_threshold', default=None, type=float, required=False)
args = parser.parse_args()

tracer = args.tracer.upper()
output_dir = args.output
version = args.version
min_nobs = args.min_nobs

stats_output_path = os.path.join(output_dir, tracer.lower()+'_fiber_stats.fits')

if not os.path.isdir(output_dir):
    try:
        os.makedirs(output_dir)
    except:
        pass

# The following target bits are the same in both main and SV3
target_bits = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS_ANY': 60}
target_bit = target_bits[tracer]

fn_dict = {'BGS_ANY': 'ztile-main-bright-cumulative.fits', 'LRG': 'ztile-main-dark-cumulative.fits', 'ELG': 'ztile-main-dark-cumulative.fits', 'QSO': 'ztile-main-dark-cumulative.fits'}
fn = os.path.join('/dvs_ro/cfs/cdirs/desi/spectro/redux/{}/zcatalog/v1'.format(version), fn_dict[tracer])

frac_fail_threshold = args.fail_threshold
if frac_fail_threshold is None:
    fail_threshold_dict = {'BGS_ANY': 0.05, 'LRG': 0.05}
    frac_fail_threshold = fail_threshold_dict[tracer]

columns = ['COADD_FIBERSTATUS', 'COADD_NUMEXP', 'COADD_NUMNIGHT', 'DELTACHI2', 'DESI_TARGET', 'FIBER', 'FIBERASSIGN_X', 'FIBERASSIGN_Y', 'FIRSTNIGHT', 'LASTNIGHT', 'MASKBITS', 'MAX_MJD', 'MEAN_MJD', 'MIN_MJD', 'SPECTYPE', 'SUBTYPE', 'TARGET_DEC', 'TARGET_RA', 'TARGETID', 'TILEID', 'TSNR2_BGS', 'TSNR2_LRG', 'Z', 'ZWARN']

cat_fn = '/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/{}_data/{}.fits'.format(version, tracer.lower())
if not os.path.isfile(cat_fn):
    if not os.path.isdir(os.path.dirname(cat_fn)):
        try:
            os.makedirs(os.path.dirname(cat_fn))
        except:
            pass
    cat = Table(fitsio.read(fn, columns=['DESI_TARGET']))
    idx = np.where(cat['DESI_TARGET'] & 2**target_bit > 0)[0]
    cat = Table(fitsio.read(fn, rows=idx, columns=columns))
    cat.write(cat_fn, overwrite=True)
else:
    cat = Table(fitsio.read(cat_fn))
print(len(cat))

if 'Z_not4clus' in cat.colnames:
    cat.rename_column('Z_not4clus', 'Z')

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

# Apply masks
if tracer=='LRG':
    tmp1 = Table(fitsio.read(os.path.join('/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_basic.fits'), columns=['TARGETID']))
    tmp2 = Table(fitsio.read(os.path.join('/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_lrgmask_v1.1.fits.gz')))
    lrgmask = hstack([tmp1, tmp2])
    lrgmask = lrgmask[lrgmask['lrg_mask']==0]
    mask = np.in1d(cat['TARGETID'], lrgmask['TARGETID'])
    print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]
elif tracer=='ELG':
    tmp1 = Table(fitsio.read(os.path.join('/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_basic.fits'), columns=['TARGETID']))
    tmp2 = Table(fitsio.read(os.path.join('/global/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_elgmask_v1.fits.gz')))
    elgmask = hstack([tmp1, tmp2])
    elgmask = elgmask[elgmask['elg_mask']==0]
    mask = np.in1d(cat['TARGETID'], elgmask['TARGETID'])
    print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

# # Remove duplicated objects
# print(len(cat), len(np.unique(cat['TARGETID'])))
# cat.sort('EFFTIME_LRG', reverse=True)
# _, idx_keep = np.unique(cat['TARGETID'], return_index=True)
# cat = cat[idx_keep]
# print(len(cat), len(np.unique(cat['TARGETID'])))

# Redshift quality cut
if tracer=='LRG':
    cat['q'] = cat['ZWARN']==0
    cat['q'] &= cat['Z']<1.45
    cat['q'] &= cat['DELTACHI2']>15
elif tracer=='BGS_ANY':
    cat['q'] = cat['ZWARN']==0
    cat['q'] &= cat['DELTACHI2']>40
else:
    raise ValueError(tracer+' redshift quality cut is not defined.')
print('Redshift quality', np.sum(~cat['q'])/len(cat))
mask_quality = cat['q'].copy()

cat['MEAN_X'], cat['MEAN_Y'] = 0., 0.
for fiber in np.unique(cat['FIBER']):
    mask = cat['FIBER']==fiber
    cat['MEAN_X'][mask] = np.mean(cat['FIBERASSIGN_X'][mask])
    cat['MEAN_Y'][mask] = np.mean(cat['FIBERASSIGN_Y'][mask])

# Fiber histogram
fig, ax = plt.subplots(10, 1, figsize=(16, 20))
for index in range(10):
    fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
    mask = (cat['FIBER']>fiber_min) & (cat['FIBER']<fiber_max)
    ax[index].hist(cat['FIBER'][mask], 500, range=(fiber_min, fiber_max), label='PETAL_LOC {}'.format(index))
    ax[index].hist(cat['FIBER'][mask & (~mask_quality)], 500, range=(fiber_min, fiber_max))
    ax[index].set_xlim(fiber_min-3, fiber_max+3)
    ax[index].legend(loc='upper left', markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, tracer.lower()+'_fiber_histogram.png'))
plt.close()


fiberstats = Table()
fiberstats['FIBER'], fiberstats['n_tot'] = np.unique(cat['FIBER'], return_counts=True)
fiberstats.sort('n_tot')
tt = Table()
tt['FIBER'], tt['n_fail'] = np.unique(cat['FIBER'][~mask_quality], return_counts=True)
fiberstats = join(fiberstats, tt, keys='FIBER', join_type='outer').filled(0)
fiberstats['frac_fail'] = fiberstats['n_fail']/fiberstats['n_tot']
error_floor = True
n, p = fiberstats['n_tot'].copy(), fiberstats['frac_fail'].copy()
if error_floor:
    p1 = np.maximum(p, 1/n)  # error floor
else:
    p1 = p
fiberstats['frac_fail_err'] = np.clip(np.sqrt(n * p * (1-p))/n, np.sqrt(n * p1 * (1-p1))/n, 1)

fiberstats.write(stats_output_path, overwrite=True)

fiberstats.sort('n_fail')

print('Bad fibers')
mask_threshold = fiberstats['frac_fail']>=frac_fail_threshold
print(np.sum(mask_threshold), np.sum(mask_threshold)/len(mask_threshold))
print(np.mean(fiberstats['frac_fail'][mask_threshold]), np.mean(fiberstats['frac_fail']), np.mean(fiberstats['frac_fail'][~mask_threshold]))
print(np.sum(fiberstats['n_fail'][mask_threshold])/np.sum(fiberstats['n_fail']))
bad_fibers = fiberstats['FIBER'][mask_threshold].copy()
print(len(bad_fibers))
print(list(bad_fibers))

# Failure rate vs fiber plot
ymax = 0.3
fig, ax = plt.subplots(10, 1, figsize=(16, 20))
for index in range(10):
    fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
    mask = (fiberstats['FIBER']>fiber_min) & (fiberstats['FIBER']<fiber_max)
    mask &= (fiberstats['n_tot']>=min_nobs)
    mask_good = mask & (~np.in1d(fiberstats['FIBER'], bad_fibers))
    mask_bad = mask & np.in1d(fiberstats['FIBER'], bad_fibers)
    mask_really_bad = mask & (fiberstats['frac_fail']>ymax)
    # plt.figure(figsize=(16, 2))
    ax[index].errorbar(fiberstats['FIBER'][mask_good], fiberstats['frac_fail'][mask_good],
                       yerr=(np.clip(fiberstats['frac_fail_err'][mask_good], None, fiberstats['frac_fail'][mask_good]), fiberstats['frac_fail_err'][mask_good]),
                       color='C0', fmt='.', ms=3, elinewidth=1)
    ax[index].errorbar(fiberstats['FIBER'][mask_bad], fiberstats['frac_fail'][mask_bad],
                   yerr=(np.clip(fiberstats['frac_fail_err'][mask_bad], None, fiberstats['frac_fail'][mask_bad]), fiberstats['frac_fail_err'][mask_bad]),
                   color='C3', fmt='.', ms=3, elinewidth=1)
    for bad_index in np.where(mask_really_bad)[0]:
        ax[index].arrow(fiberstats['FIBER'][bad_index], ymax*0.78, 0, ymax/10, head_width=3, head_length=ymax/10, fc='C3', ec='C3')
    ax[index].grid(alpha=0.5)
    ax[index].set_yticks([0., 0.1, 0.2, 0.3], minor=False)
    ax[index].set_ylim(-0.01, ymax)
    ax[index].set_xlim(fiber_min-3, fiber_max+3)
    ax[index].annotate('PETAL_LOC {}'.format(index), (fiber_min+3, ymax*0.6), fontsize='large')
    # ax[index].legend(loc='upper left', markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, tracer.lower()+'_failure_rate_vs_fiber.png'))
plt.close()

# Focal plane failure rate plot
fiberstats['MEAN_X'], fiberstats['MEAN_Y'] = 0., 0.
for index, fiber in enumerate(fiberstats['FIBER']):
    mask = cat['FIBER']==fiber
    fiberstats['MEAN_X'][index], fiberstats['MEAN_Y'][index] = cat['MEAN_X'][mask][0], cat['MEAN_Y'][mask][0]
mask = fiberstats['n_tot']>=min_nobs
plt.figure(figsize=(12, 11.5))
plt.scatter(fiberstats['MEAN_X'][mask], fiberstats['MEAN_Y'][mask], c=1-fiberstats['frac_fail'][mask],
            s=45, vmin=0.9, vmax=1., cmap='viridis')
plt.axis([-420, 420, -420, 420])
plt.colorbar(fraction=0.04, pad=0.04)
# plt.axis('off')
plt.savefig(os.path.join(output_dir, tracer.lower()+'_focal_plane_failure_rate.png'))
plt.close()

