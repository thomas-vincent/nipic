#! /usr/bin/env python
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
class awrap:
    """
    Wrap calls of numpy reduced functions to always output arrays with same 
    number of dimensions as input and also encapsulate the output in a list.
    Purpose is to have homogeneous outputs between the split function which procudes 
    several arrays and other numpy ufuncs producing one single array. 
    """
    def __init__(self, f):
        self.func = f
        
    def __call__(self, a, axis=0, extra=None):
        if self.func == np.split:
            r = (self.func.__name__, np.split(a, a.shape[axis], axis=axis))
        else:
            r = (self.func.__name__, [self.func(a, axis=axis, keepdims=1)])
        return r

MAX_SPLIT_SIZE = 5    
def reduce_guess(a, axis=0, extra=None):
    if extra is not None and extra.has_key('grad_table') and extra.has_key('b_values'): #DTI
        # http://nipy.org/dipy/examples_built/quick_start.html#example-quick-start
        logger.info('Computing DTI mask...')
        maskdata, mask = median_otsu(a, 3, 1, True, dilate=2) #TODO: improve this
        logger.info('Computing DTI gradient table...')
        gtab = gradient_table(extra['b_values'], extra['grad_table'])
        logger.info('Fitting DTI tensor model...')
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(maskdata)
        logger.info('Computing fractional anisotropy...')
        fa = fractional_anisotropy(tenfit.evals)
        fa[np.isnan(fa)] = 0
        return ('FA', [fa])
    else:
        if a.shape[axis] <= MAX_SPLIT_SIZE:
            return awrap(np.split)(a, axis)
        else:
            return awrap(np.mean)(a, axis)
        
REDUCTORS = {'mean' : awrap(np.mean), 'std': awrap(np.std), 'min' : awrap(np.min),
             'max' : awrap(np.max), 'median' : awrap(np.median),
             'split' : awrap(np.split), 'default': reduce_guess}

NIBABEL_SLICER_VIEWS = ['sagittal', 'coronal', 'axial']

USAGE = 'usage: %%prog [options] MRI_VOL1 MRI_VOL2 ...'
DESCRIPTION = 'Generate image of mips and centered slices from 3D or 4D MRI data'
MIN_ARGS = 0
MAX_ARGS = -1

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('[MRI_snap]')

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
        logger.info('Processing %s ...' % vol_fn)
        mri_snap(vol_fn, get_axes_list(options.slice_axes),
                 get_axes_list(options.mip_axes),
                 get_reductors(options.reductions))

def mri_snap(mri_fn, slice_axes=MRI_3D_AXES, mip_axes=MRI_3D_AXES,
             reductors=(np.mean,)):
    """
    Input: 
        - mri_fn (str):
            path to MRI file from which to generate snapshot images
        - slice_axes (list of str):
            list of axes along which to generate centered-sliced snaps 
            (axial, coronal, sagittal).
        - mip_axes (list of str):
            list of axes along which to generate centered-sliced snaps
            (axial, coronal, sagittal).
        - reductors (list of callables):
            list of functions to apply to create 3D volumes from 4D MRI.
    """
    img = load_mri(mri_fn)
    base_fn = split_ext_gz_safe(mri_fn)[0]
    if len(img.shape) > 3:
        idata = img.get_data()
        for reductor in reductors:
            rname, rvols = reductor(idata, axis=3, extra=img.extra)
            for ivol, vol in enumerate(rvols):
                if len(rvols) == 1:
                    suffix = '_' + rname
                else:
                    suffix = '_vol%04d' % ivol 
                reduced_img = img.__class__(vol, img.affine,
                                            img.header, img.extra)
                plot_slices(reduced_img, slice_axes, base_fn + suffix)
                plot_mips(reduced_img, mip_axes, base_fn + suffix)
    else:
        plot_slices(img, slice_axes, base_fn)
        plot_mips(img, mip_axes, base_fn)

def load_mri(mri_fn):
    bfn = split_ext_gz_safe(mri_fn)[0]
    img = nibabel.load(mri_fn)
    if op.exists(bfn + '.bval') and op.exists(bfn + '.bvec'):
        # DTI data -> load gradient table data
        img.extra['grad_table'], img.extra['b_values'] = dipy.io.read_bvec_file(bfn)
    return img

def plot_mips(img, mip_axes, base_fn):

    for aname in mip_axes:
        ia = MRI_3D_AXES.index(aname)
        mshape = img.shape[:ia] + (1,) + img.shape[ia+1:]
        mip_data = img.get_data().max(axis=ia).reshape(*mshape)
        mip_img = img.__class__(mip_data, img.affine, img.header,
                                img.extra)
        slicer = mip_img.orthoview()
        center = [img.shape[o]/2 for o in slicer._order]
        slicer.set_position(*np.dot(slicer._affine, center+[1])[:3])

        fig = slicer.figs[0]
        fig_transf = fig.dpi_scale_trans.inverted()

        ax = slicer._axes[NIBABEL_SLICER_VIEWS.index(aname)]
        img_fn = base_fn + '_mip_' + aname + '.png'
        ext = ax.get_window_extent().transformed(fig_transf)
        fig.savefig(img_fn, bbox_inches=ext.expanded(1.1,1.1))

            
def plot_slices(img, axes, base_fn):
    slicer = img.orthoview()
    center = [img.shape[o]/2 for o in slicer._order]
    slicer.set_position(*np.dot(slicer._affine, center+[1])[:3])

    if img.extra.has_key('grad_table'): # DTI
        [a.findobj(matplotlib.image.AxesImage)[0].set_cmap('jet')
         for a in slicer._axes]

    fig = slicer.figs[0]
    fig_transf = fig.dpi_scale_trans.inverted()
    for aname in axes:
        ax = slicer._axes[NIBABEL_SLICER_VIEWS.index(aname)]
        img_fn = base_fn + '_center_slice_' + aname + '.png'
        ext = ax.get_window_extent().transformed(fig_transf)
        fig.savefig(img_fn, bbox_inches=ext.expanded(1.1,1.1))
    # auto_crop_img(img_fn)
    
def auto_crop_img(ifn, bgcolor='white'):
    logger.info('Auto-cropping image %s ...' % ifn)
    from PIL import Image, ImageChops
    image = Image.open(ifn)
    if image.mode != "RGB":
        image = image.convert("RGB")
    bg = Image.new("RGB", image.size, bgcolor)
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        image.crop(bbox).save(ifn)
        
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

def split_ext_gz_safe(fn):
    root, ext = op.splitext(fn)
    if ext == '.gz':
        root, n_ext = op.splitext(root)
        ext = n_ext + ext
    return root, ext

if __name__ == '__main__':
    main()
