# Use zcatalog instead of the LSS catalog
# Make 2-D density plots of redshift vs fiber
# Example:
# python plot_redshift_vs_fiber-zcatalog-compare_with_y1.py --tracer BGS_ANY -v jura --plot_dir /global/cfs/cdirs/desi/users/rongpu/redshift_qa/z_vs_fiber/Y3_jura_zcatalog_y1_subset
# python plot_redshift_vs_fiber-zcatalog-compare_with_y1.py --tracer LRG -v jura --plot_dir /global/cfs/cdirs/desi/users/rongpu/redshift_qa/z_vs_fiber/Y3_jura_zcatalog_y1_subset
# python plot_redshift_vs_fiber-zcatalog-compare_with_y1.py --tracer ELG -v jura --plot_dir /global/cfs/cdirs/desi/users/rongpu/redshift_qa/z_vs_fiber/Y3_jura_zcatalog_y1_subset
# python plot_redshift_vs_fiber-zcatalog-compare_with_y1.py --tracer QSO -v jura --plot_dir /global/cfs/cdirs/desi/users/rongpu/redshift_qa/z_vs_fiber/Y3_jura_zcatalog_y1_subset

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc, argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

params = {'figure.facecolor': 'w'}
plt.rcParams.update(params)

parser = argparse.ArgumentParser()
parser.add_argument("--tracer", help="tracer type (BGS_ANY, LRG, ELG, or QSO)", required=True)
parser.add_argument('-v', '--version', default='iron', required=False)
parser.add_argument("--fn", help="path of the catalog file", default=None)
parser.add_argument("--plot_dir", help="directory to save the plots", default=None)

args = parser.parse_args()
tracer = args.tracer.upper()
version = args.version
fn = args.fn
plot_dir = args.plot_dir

if not os.path.isdir(plot_dir):
    try:
        os.makedirs(plot_dir)
    except:
        pass

z_bins_dict = {'BGS_ANY': np.arange(-0.025, 0.7, 0.02), 'LRG': np.arange(-0.025, 1.5, 0.025), 'ELG': np.arange(0.6, 1.8, 0.025), 'QSO': np.arange(-0.025, 4.5, 0.1)}
vmax_dict = {'BGS_ANY': 1/3e4, 'LRG': 1/4e4, 'ELG': 1/6e4, 'QSO': 1/4e4}

# The following target bits are the same in both main and SV3
target_bits = {'LRG': 0, 'ELG': 1, 'QSO': 2, 'BGS_ANY': 60}
target_bit = target_bits[tracer]

if fn is None:
    fn_dict = {'BGS_ANY': 'ztile-main-bright-cumulative.fits', 'LRG': 'ztile-main-dark-cumulative.fits', 'ELG': 'ztile-main-dark-cumulative.fits', 'QSO': 'ztile-main-dark-cumulative.fits'}
    # fn = os.path.join('/dvs_ro/cfs/cdirs/desi/spectro/redux/iron/zcatalog', fn_dict[tracer])
    fn = os.path.join('/dvs_ro/cfs/cdirs/desi/spectro/redux/{}/zcatalog/v1'.format(version), fn_dict[tracer])
fn_y1 = fn.replace('jura', 'iron').replace('v1', 'v0')
print(fn_y1)

min_nobs = 100

########################################################################################################################

cat = Table(fitsio.read(fn_y1, columns=['DESI_TARGET']))
idx = np.where(cat['DESI_TARGET'] & 2**target_bit > 0)[0]
cat = Table(fitsio.read(fn_y1, rows=idx))
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

if tracer=='LRG':
    # Apply maskbits
    maskbits = [1, 8, 9, 11, 12, 13]
    mask = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask &= (cat['MASKBITS'] & 2**bit)==0
    print('MASKBITS  ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]
elif tracer=='ELG':
    # Apply maskbits
    maskbits = [1, 11, 12, 13]
    mask = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask &= (cat['MASKBITS'] & 2**bit)==0
    print('MASKBITS  ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

# # Remove duplicated objects
# print(len(cat), len(np.unique(cat['TARGETID'])))
# cat.sort('EFFTIME_LRG', reverse=True)
# _, idx_keep = np.unique(cat['TARGETID'], return_index=True)
# cat = cat[idx_keep]
# print(len(cat), len(np.unique(cat['TARGETID'])))

cat_y1 = cat.copy()

########################################################################################################################

cat = Table(fitsio.read(fn, columns=['DESI_TARGET']))
idx = np.where(cat['DESI_TARGET'] & 2**target_bit > 0)[0]
cat = Table(fitsio.read(fn, rows=idx))
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

if tracer=='LRG':
    # Apply maskbits
    maskbits = [1, 8, 9, 11, 12, 13]
    mask = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask &= (cat['MASKBITS'] & 2**bit)==0
    print('MASKBITS  ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]
elif tracer=='ELG':
    # Apply maskbits
    maskbits = [1, 11, 12, 13]
    mask = np.ones(len(cat), dtype=bool)
    for bit in maskbits:
        mask &= (cat['MASKBITS'] & 2**bit)==0
    print('MASKBITS  ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

# # Remove duplicated objects
# print(len(cat), len(np.unique(cat['TARGETID'])))
# cat.sort('EFFTIME_LRG', reverse=True)
# _, idx_keep = np.unique(cat['TARGETID'], return_index=True)
# cat = cat[idx_keep]
# print(len(cat), len(np.unique(cat['TARGETID'])))

########################################################################################################################
mask = np.in1d(cat['TARGETID'], cat_y1['TARGETID'])
mask &= np.in1d(cat['TILEID'], cat_y1['TILEID'])
cat = cat[mask]
print(np.sum(mask)/len(mask), len(cat), len(cat_y1))
########################################################################################################################

fiberstats = Table()
fiberstats['FIBER'], fiberstats['n_tot'] = np.unique(cat['FIBER'], return_counts=True)
fiberstats['weight'] = 1/fiberstats['n_tot']*np.median(fiberstats['n_tot'])
cat = join(cat, fiberstats[['FIBER', 'weight']], join_type='outer')
too_few_fibers = fiberstats['FIBER'][fiberstats['n_tot']<min_nobs]

fig, ax = plt.subplots(10, 1, figsize=(16, 20))
for index in range(10):
    fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
    mask = (cat['FIBER']>fiber_min) & (cat['FIBER']<fiber_max)
    xbins, ybins = np.linspace(fiber_min, fiber_max, 501), z_bins_dict[tracer]
    ybins = ybins - np.diff(ybins)[0]/2  # center one of the bins at z=0
    ax[index].hist2d(cat['FIBER'][mask], cat['Z'][mask], bins=[xbins, ybins], vmin=0, vmax=len(cat)*vmax_dict[tracer])
    ax[index].set_xlim(fiber_min, fiber_max)
plt.tight_layout()
if plot_dir is not None:
    plt.savefig(os.path.join(plot_dir, 'redshift_vs_fiber_{}.png'.format(tracer.lower())))
plt.show()

# Normalize by the number of objects in each fiber
fig, ax = plt.subplots(10, 1, figsize=(16, 20))
for index in range(10):
    fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
    mask = (cat['FIBER']>fiber_min) & (cat['FIBER']<fiber_max)
    mask &= ~np.in1d(cat['FIBER'], too_few_fibers)  # Do not plot fibers with too few objects
    xbins, ybins = np.linspace(fiber_min, fiber_max, 501), z_bins_dict[tracer]
    ybins = ybins - np.diff(ybins)[0]/2  # center one of the bins at z=0
    ax[index].hist2d(cat['FIBER'][mask], cat['Z'][mask], weights=cat['weight'][mask], bins=[xbins, ybins], vmin=0, vmax=len(cat)*vmax_dict[tracer])
    ax[index].set_xlim(fiber_min, fiber_max)
plt.tight_layout()
if plot_dir is not None:
    plt.savefig(os.path.join(plot_dir, 'redshift_vs_fiber_{}_norm.png'.format(tracer.lower())))
plt.show()

