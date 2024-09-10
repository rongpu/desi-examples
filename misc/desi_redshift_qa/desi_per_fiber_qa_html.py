# python desi_per_fiber_qa_html.py -o /global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/new/kibo

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio
from astropy.io import fits

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, help="output directory path", required=True)
args = parser.parse_args()

output_dir = args.output
html_dir = os.path.join(output_dir, 'html')

if not os.path.isdir(html_dir):
    os.makedirs(html_dir)

f = open(os.path.join(output_dir, 'fiber_directory.html'), "w")
f.write('<html>\n')
f.write('<table>\n')
for fiber in np.arange(5000):
    f.write('<td><a href=html/fiber_{}.html>FIBER_{}</a></td>\n'.format(fiber, fiber))
    f.write('</tr>\n')
f.write('</table>\n')
f.close()

for fiber in np.arange(5000):

    f = open(os.path.join(html_dir, 'fiber_{}.html'.format(fiber)), "w")
    f.write('<html>\n')
    f.write('<table>\n')
    for tracer in ['BGS_BRIGHT', 'LRG', 'ELG_LOP', 'QSO', 'ELG_VLO', 'BGS_FAINT']:

        f.write('<th></th>\n')
        f.write('<th>{}</th>\n'.format(tracer))
        f.write('<tr>\n')

        for apply_good_z_cut in [False, True]:

            if apply_good_z_cut:
                f.write('<td>Good z</td>\n')
                png_dir = '{}_goodz'.format(tracer.lower())
            else:
                f.write('<td>All</td>\n')
                png_dir = '{}_allz'.format(tracer.lower())

            image_fn = os.path.join(png_dir, 'FIBER_{}.png'.format(fiber))

            f.write('<td><a href=\'png/{}\'><img src=\'../png/{}\' width=\'1200\'></a></td>\n'.format(image_fn, image_fn))
            f.write('</tr>\n')

    f.write('</table>\n')
    f.close()
