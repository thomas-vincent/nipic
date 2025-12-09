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
import matplotlib as mpl
import matplotlib.pyplot as plt

from subprocess import call

import logging

import pandas as pd
import numpy as np

import nibabel as nib

MRI_3D_AXES = ['sagittal', 'coronal', 'axial']
NIBABEL_SLICER_VIEWS = ['sagittal', 'coronal', 'axial']

#logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('nipic')

def two_intervals_intersection(i_a, i_b):
    if i_a is None or i_b is None:
        raise ValueError("Intervals cannot be None")

    if len(i_a) != 2 or len(i_b) != 2:
        raise ValueError("Intervals must have two elements")

    if any(np.isnan(v) for v in i_a):
        raise ValueError("Interval i_a cannot contain NaNs")

    if any(np.isnan(v) for v in i_b):
        raise ValueError("Interval i_b cannot contain NaNs")

    a_s, a_e = i_a
    b_s, b_e = i_b

    if (b_s > a_e) or (a_s > b_e):
        return None
    else:
        o_s = max(a_s, b_s)
        o_e = min(a_e, b_e)

    return (o_s, o_e)

import unittest

class TestInterIntervals(unittest.TestCase):
    def setUp(self):
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)

    def test_non_overlapping(self):
        self.assertIsNone(two_intervals_intersection((-5, -2), (5, 10)))

    def test_null_interval(self):
        self.assertEqual(two_intervals_intersection((-5, 5), (0, 0)),
                         (0,0))

    def test_within(self):
        self.assertEqual(two_intervals_intersection((-15, 20), (5, 10)),
                          (5, 10))

    def test_overlap_right(self):
        self.assertEqual(two_intervals_intersection((-15, 20), (5, 25)),
                          (5, 20))

    def test_overlap_left(self):
        self.assertEqual(two_intervals_intersection((-15, 20), (-30, 5)),
                         (-15, 5))

    def test_inf_left(self):
        self.assertEqual(two_intervals_intersection((-15, 20), (-5, np.inf)),
                         (-5, 20))

    def test_inf_right(self):
        self.assertEqual(two_intervals_intersection((-15, 20), (-np.inf, 3)),
                         (-15, 3))

    def test_inf(self):
        self.assertEqual(two_intervals_intersection((-15, 20),
                                                    (-np.inf, np.inf)),
                         (-15, 20))


class awrap:
    """
    Wrap calls of numpy reduced functions to always output arrays with same
    number of dimensions as input and also encapsulate the output in a list.
    Purpose is to have homogeneous outputs between the numpy split function
    (procuding several arrays) and other numpy ufuncs (producing
    one single array).
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
        import dipy
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

def df_save_excel(dfs, fn, index=True):

    if isinstance(dfs, pd.DataFrame):
        dfs = {'sheet' : dfs}

    writer = pd.ExcelWriter(fn, engine='xlsxwriter')

    for sheet_name, df in dfs.items():
        assert(df.columns.is_unique)
        df.to_excel(writer, sheet_name=sheet_name, index=index)  # send df to writer
        worksheet = writer.sheets[sheet_name]  # pull worksheet object
        for idx, col in enumerate(df): # loop through all columns
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name))  # len of column name/header
            ))
            max_len = int(max_len * 1.2)  # adding a little extra space
            worksheet.set_column(idx, idx, max_len)  # set column width

    writer.close()

def max_neg_value(a):
    a = np.array(a)
    return a[np.where(a<=0, a, -np.nan).argmax()]

from IPython import embed
def min_pos_value(a):
    a = np.array(a)
    embed()
    return a[np.where(a>=0, a, np.inf).argmin()]

class TestMaxPosNegValue(unittest.TestCase):
    def setUp(self):
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)

    def test_no_pos(self):
        self.assertTrue(np.isnan(min_pos_value([-5, -10])))

    def test_no_neg(self):
        self.assertTrue(np.isnan(max_neg_value([5, 10])))

    def test_neg(self):
        self.assertEqual(max_neg_value([-7, -5, -10, -9]), -5)

    def test_pos(self):
        self.assertEqual(min_pos_value([7, 5, 10, 9]), 5)
