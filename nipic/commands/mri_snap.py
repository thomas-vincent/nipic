#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate image of slices/mips from 3D or 4D MRI data


mri_snap VOL1 VOL2 VOL3 ... [--slice_axes <LIST_OF_AXES>
                             --mip_axes <LIST_OF_AXES>
                             --reductions <LIST_OF_REDUCTORS>]

TODO: black background + white text
TODO: use neurological convention
TODO: utests
"""
import sys
import os
import os.path as op
from io import StringIO
import tempfile
import shutil
import re

from subprocess import call

import logging
from optparse import OptionParser

import numpy as np
import matplotlib

import nibabel

import dipy.io
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

MRI_3D_AXES = ['sagittal', 'coronal', 'axial']
NIBABEL_SLICER_VIEWS = ['sagittal', 'coronal', 'axial']

MAX_SPLIT_SIZE = 5 

USAGE = 'usage: %%prog [options] MRI_VOL1 MRI_VOL2 ...'
DESCRIPTION = 'Generate image of mips and centered slices from 3D or 4D MRI data'
MIN_ARGS = 0
MAX_ARGS = -1

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('nipic')

from nipic.utils import mri_snap, REDUCTORS

def main():
    parser = OptionParser(usage=USAGE, description=DESCRIPTION)

    parser.add_option('-s', '--slice_axes', metavar='LIST_OF_STR',
                      type='str', default='axial,sagittal,coronal',
                      help='Comma-separated list of views along which ' \
                           'to generate centered slices. To disable slice '
                           'outputs, use "none". Default is all views.')

    parser.add_option('-m', '--mip_axes', metavar='LIST_OF_STR',
                      type='str', default='axial,sagittal,coronal',
                      help='Comma-separated list of views along which ' \
                           'to generate Maximum Intensity Projections ' \
                            '(MIPs). To disable MIP outputs, use "none". '
                            'Default is all views.')

    parser.add_option('-r', '--reductions', type='str',
                      metavar='LIST OF STR', default='default',
                      help='Reduce operations to apply to 4D MRI data ' \
                           'along the temporal axis. Choices are: '\
                           'split, mean, std, min, max, med. '\
                           'Default: split if nb vols <= 5 else use mean.')
    
    parser.add_option('-o', '--output-folder', type='str',
                      metavar='PATH',
                      help='Output folder to save images')
    
    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=0, help='Verbose level')

    (options, args) = parser.parse_args()

    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < MIN_ARGS or (MAX_ARGS >= 0 and nba > MAX_ARGS):
        parser.print_help()
        sys.exit(1)

    vol_fns = args
    for vol_fn in vol_fns:
        if options.output_folder is None:
            output_folder = op.dirname(vol_fn)
        else:
            output_folder = options.output_folder
        logger.info('Processing %s ...' % vol_fn)
        mri_snap(vol_fn, output_folder, get_axes_list(options.slice_axes),
                 get_axes_list(options.mip_axes),
                 get_reductors(options.reductions))

def get_axes_list(al):
    if al == 'none':
        return []
    
    allowed = set(MRI_3D_AXES)
    axes_list = al.split(',')
    for axis in axes_list:
        if axis not in allowed:
            raise Exception('Unknown axis: %s (allowed: %s)'
                            % (axis, ','.join(MRI_3D_AXES)))
    return axes_list

def get_reductors(rl):
    reductor_names = rl.split(',')
    for reductor_name in reductor_names:
        if reductor_name not in REDUCTORS:
            raise Exception('Unsupport reductors: %s (allowed: %s)'
                            % (reductor_name,
                               ','.join(REDUCTORS.keys())))
    return [REDUCTORS[rn] for rn in reductor_names]


