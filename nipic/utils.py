#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import os.path as op
from io import StringIO
import tempfile
import shutil
import re
import colorsys

from subprocess import call

import logging
from optparse import OptionParser

import numpy as np

import nibabel as nib
import dipy

MRI_3D_AXES = ['sagittal', 'coronal', 'axial']
NIBABEL_SLICER_VIEWS = ['sagittal', 'coronal', 'axial']

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

def load_mri(mri_fn):
    bfn = split_ext_gz_safe(mri_fn)[0]
    img = nib.load(mri_fn)
    if op.exists(bfn + '.bval') and op.exists(bfn + '.bvec'):
        # DTI data -> load gradient table data
        img.extra['grad_table'], img.extra['b_values'] = \
            dipy.io.read_bvec_file(bfn)
    return img

def save_img_with_new_dtype(data, image, out_fn):
    hd = image.header
    new_image = nib.Nifti2Image(data, image.affine, header=hd)
    nib.save(new_image, out_fn)

def change_color_lightness(color_rgba, lightness_ratio):
    """ 0-255 color coding to comply with freesurfer"""
    r, g, b, a = [c/255 for c in color_rgba]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    rgb = colorsys.hls_to_rgb(h,
                              max(0, min(lightness_ratio * l, 1)),
                              s)
    return [int(c * 255) for c in rgb] + [color_rgba[3]]

def color_average_rgba(color_1, color_2):
    r1, g1, b1, a1 = color_1
    r2, g2, b2, a2 = color_2
    return ( int(((r1**2 + r2**2) / 2)**.5),
             int(((g1**2 + g2**2) / 2)**.5),
             int(((b1**2 + b2**2) / 2)**.5),
             int((a1 + a2)/2) )
