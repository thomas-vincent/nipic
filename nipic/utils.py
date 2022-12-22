#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('nipic')

class awrap:
    """
    Wrap calls of numpy reduced functions to always output arrays with same 
    number of dimensions as input and also encapsulate the output in a list.
    Purpose is to have homogeneous outputs between the split function 
    which procudes several arrays and other numpy ufuncs producing 
    one single array. 
    """
    def __init__(self, f):
        self.func = f
        
    def __call__(self, a, axis=0, extra=None):
        if self.func == np.split:
            r = (self.func.__name__, np.split(a, a.shape[axis], axis=axis))
        else:
            r = (self.func.__name__, [self.func(a, axis=axis, keepdims=1)])
        return r



def reduce_guess(a, axis=0, extra=None):
    if (extra is not None and 'grad_table' in extra and
        'b_values' in extra): # DTI
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

REDUCTORS = {'mean' : awrap(np.mean), 'std': awrap(np.std), 
             'min' : awrap(np.min),
             'max' : awrap(np.max), 'median' : awrap(np.median),
             'split' : awrap(np.split), 'default': reduce_guess}

class UnsupportedNumberOfDims(Exception): pass

def mri_snap(mri_fn, output_folder, slice_axes=MRI_3D_AXES, mip_axes=MRI_3D_AXES,
             reductors=(np.mean,)):
    """
    Generate image of slices/mips from 3D or 4D MRI data

    TODO: black background + white text
    TODO: use neurological convention
    TODO: utests

    Input: 
        - mri_fn (str):
            path to MRI file from which to generate snapshot images
        - output_folder (str):
            path to store iamges
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
    base_fn = op.join(output_folder,
                      op.basename(split_ext_gz_safe(mri_fn)[0]))
    if len(img.shape) > 3:
        idata = img.get_data()
        if len(img.shape) == 5:
            idata = idata[:,:,:,:,0] # TODO reduction on more axes
        elif len(img.shape) > 5:
            raise UnsupportedNumberOfDims()
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
        img.extra['grad_table'], img.extra['b_values'] = \
            dipy.io.read_bvec_file(bfn)
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
    [a.findobj(matplotlib.image.AxesImage)[0].cmap.set_bad(color='red')
     for a in slicer._axes]
    if 'grad_table' in img.extra: # DTI
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

def split_ext_gz_safe(fn):
    root, ext = op.splitext(fn)
    if ext == '.gz':
        root, n_ext = op.splitext(root)
        ext = n_ext + ext
    return root, ext

