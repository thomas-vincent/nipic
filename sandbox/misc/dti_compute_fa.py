#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute fractional anisotropy from raw DTI MRI.

Usage::

    dti_compute_fa DTI_VOL

See http://nipy.org/dipy/examples_built/quick_start.html#example-quick-start

TODO: utests
"""
import sys
import os.path as op

import logging
from optparse import OptionParser

import numpy as np

import nibabel

from dipy.io import read_bvec_file
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy

USAGE = 'usage: %%prog [options] MRI_DTI_VOL ...'
DESCRIPTION = 'Compute fractional anisotropy 3D volume from 4D raw DTI image.'
MIN_ARGS = 1
MAX_ARGS = 1

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('[MRI_snap]')

def main():
    parser = OptionParser(usage=USAGE, description=DESCRIPTION)
    
    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=0, help='Verbose level')

    (options, args) = parser.parse_args()

    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < MIN_ARGS or (MAX_ARGS >= 0 and nba > MAX_ARGS):
        parser.print_help()
        sys.exit(1)

    vol_fn = args[0]

    img = load_dti(vol_fn)
    data = img.get_data()
    logger.info('Computing DTI mask...')
    maskdata, mask = median_otsu(data, 3, 1, True, dilate=2) #TODO: improve this
    logger.info('Computing DTI gradient table...')
    gtab = gradient_table(img.extra['b_values'], img.extra['grad_table'])
    logger.info('Fitting DTI tensor model...')
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(maskdata)
    logger.info('Computing fractional anisotropy...')
    fa = fractional_anisotropy(tenfit.evals)
    fa[np.isnan(fa)] = 0

    img_fa = nibabel.Nifti1Image(fa.astype(np.float32), img.affine)
    out_fn = add_suffix(vol_fn, '_FA')
    logger.info('Saving fractional anisotropy to %s ...' % out_fn)
    img_fa.to_filename(out_fn)
    
def load_dti(mri_fn):
    bfn = split_ext_gz_safe(mri_fn)[0]
    img = nibabel.load(mri_fn)
    assert op.exists(bfn + '.bval') and op.exists(bfn + '.bvec')
    img.extra['grad_table'], img.extra['b_values'] = read_bvec_file(bfn)
    return img

def add_suffix(fn, suffix):
    """ Add a suffix before file extension.

    >>> add_suffix('./my_file.txt', '_my_suffix')
    './my_file_my_suffix.txt'
    """
    if suffix is None:
        return fn
    sfn = op.splitext(fn)
    if sfn[1] == '.gz':
        sfn = op.splitext(fn[:-3])
        sfn = (sfn[0], sfn[1] + '.gz')
    return sfn[0] + suffix + sfn[1]


def split_ext_gz_safe(fn):
    root, ext = op.splitext(fn)
    if ext == '.gz':
        root, n_ext = op.splitext(root)
        ext = n_ext + ext
    return root, ext

if __name__ == '__main__':
    main()
