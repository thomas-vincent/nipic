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
from optparse import OptionParser

from .freesurfer import load_lut
from nilearn.plotting import plot_img

import pandas as pd
import numpy as np

import nibabel as nib

MRI_3D_AXES = ['sagittal', 'coronal', 'axial']
NIBABEL_SLICER_VIEWS = ['sagittal', 'coronal', 'axial']

#logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('nipic')

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

from IPython import embed
import subprocess
from pandas.util import hash_pandas_object
import hashlib
def hash_pandas_whole_object(obj):
    return hashlib.sha1(hash_pandas_object(obj).values).hexdigest()


brain_region_template_svg = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="223.52757mm"
   height="122.6767mm"
   viewBox="0 0 223.52757 122.6767"
   version="1.1"
   id="svg8"
   inkscape:version="0.92.5 (2060ec1f9f, 2020-04-08)"
   sodipodi:docname="region_plot_template.svg">
  <defs
     id="defs2" />
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="0.98994949"
     inkscape:cx="459.79204"
     inkscape:cy="286.96775"
     inkscape:document-units="mm"
     inkscape:current-layer="layer1"
     showgrid="false"
     inkscape:window-width="2560"
     inkscape:window-height="1354"
     inkscape:window-x="0"
     inkscape:window-y="29"
     inkscape:window-maximized="1"
     fit-margin-top="0"
     fit-margin-left="0"
     fit-margin-right="0"
     fit-margin-bottom="0" />
  <metadata
     id="metadata5">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
        <dc:title></dc:title>
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(253.85593,-123.54208)">
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r01_c00.png"
       y="179.62534"
       x="-253.8723"
       id="image867"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r01_c01.png"
       y="179.62534"
       x="-197.99229"
       id="image878"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r01_c02.png"
       y="179.62534"
       x="-142.11229"
       id="image889"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r01_c03.png"
       y="179.62534"
       x="-86.232285"
       id="image900"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r00_c00.png"
       y="116.1511"
       x="-253.83551"
       id="image823"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r00_c01.png"
       y="116.1511"
       x="-197.95551"
       id="image834"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r00_c02.png"
       y="116.1511"
       x="-142.0755"
       id="image845"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref=""
       xlink:href="brain_regions_r00_c03.png"
       y="116.1511"
       x="-86.195496"
       id="image856"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
  </g>
</svg>"""

def draw_vol_mapping(values_to_map, region_template_fn, t1_template_fn,
                     cache_dir=None, output_vol_fn=None, figure_fn=None,
                     vmin=None, vmax=None, color_map=None, colorbar_label=None,
                     ignore_regions_not_in_template=False):

    def _forge_volume():
        logger.info('Forge volume from values_to_map')
        # Forge volume with given values
        template_img = nib.load(region_template_fn)
        template_data = template_img.get_fdata()
        out_data = np.empty(template_data.shape, dtype=np.float64)
        # out_transparency = np.zeros(template_data.shape, dtype=np.float64)
        lut = load_lut(aseg_only=False)
        lut_name_to_idx = {e['name']:i for i,e in lut.items()}
        for region_label, value in values_to_map.items():
            region_idx = lut_name_to_idx[region_label]
            region_mask = np.where(template_data==region_idx)
            if len(region_mask[0]) == 0 and not ignore_regions_not_in_template:
                raise ValueError('No region %s (idx=%d) found in template %s' %
                                 (region_label, region_idx, region_template_fn))
            else:
                if pd.isna(value):
                    out_data[region_mask] = 0
                else:
                    out_data[region_mask] = value
        return  nib.Nifti1Image(out_data, template_img.affine)

    if cache_dir is not None:
        values_hash_str = hash_pandas_whole_object(values_to_map.sort_index())
        cached_vol_fn = op.join(cache_dir, values_hash_str + '.nii')
        if op.exists(cached_vol_fn):
            logger.info('Load cached volume of mapped values')
            out_img = nib.load(cached_vol_fn)
        else:
            out_img = _forge_volume()
            nib.save(out_img, cached_vol_fn)
    else:
        out_img = _forge_volume()

    if vmin is None:
        vmin = values_to_map.min()

    if vmax is None:
        vmax = values_to_map.max()

    if output_vol_fn is not None:
        nib.save(out_img, output_vol_fn)

    if color_map is None:
        if vmin < 0 and vmax > 0:
            color_map = mpl.colormaps['bwr']
        elif vmax <= 0:
            color_map = mpl.colormaps['winter']
        elif vmin >= 0:
            color_map = mpl.colormaps['autumn']
        else:
            color_map = mpl.colormaps['jet']

    # transparency_img = nib.Nifti1Image(out_transparency, template_img.affine)

    rows = 2
    cols = 4
    cuts = np.linspace(-93,63, rows*cols)

    icut = 0
    logger.info('Plot brain snippets (vmin=%1.3f, vmax=%1.3f)',
                vmin, vmax)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_svg_fn = op.join(tmp_dir, 'region_plot_template.svg')
        with open(tmp_svg_fn, 'w') as fout:
            fout.write(brain_region_template_svg)

        for irow in range(rows):
            for icol in range(cols):
                sub_figure_fn = op.join(tmp_dir,
                                        'brain_regions_r%02d_c%02d.png' %
                                        (irow, icol))
                plot_img(out_img, bg_img=nib.load(t1_template_fn),
                         display_mode='y',
                         cut_coords=[cuts[icut]], output_file=sub_figure_fn,
                         colorbar=False, threshold=0,
                         black_bg=True, vmin=vmin, vmax=vmax, cmap=color_map)
                icut += 1

        ms = plt.matshow(np.array([[vmin, vmax]]), cmap=color_map)
        fig, ax = plt.subplots(figsize=(2,3))
        if vmin < 0 and vmax > 0:
            ticks = [vmin, 0, vmax]
        else:
            ticks = [vmin, vmin + abs(vmin/2), vmax/2 - abs(vmax/2), vmax]
        class MyFormatter(mpl.ticker.Formatter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            def __call__(self, x, pos=None):
                abx = abs(x)
                if abx < 1:
                    fmt = '%1.2f' 
                elif abx < 10:
                    fmt = '%1.2f'
                elif abx < 100:
                    fmt = '%1.1f'
                else:
                    fmt = '%1.0f'
                if x >= 0:
                    fmt = '$\quad$' + fmt
                return fmt % x
        cbar = plt.colorbar(ms, ticks=ticks, ax=ax,
                            format=MyFormatter())
        # embed()
        for tick in cbar.ax.yaxis.get_major_ticks():
            tick.label.set_family('monospace')
            tick.label1.set_family('monospace')
            tick.label2.set_family('monospace')
        ax.remove()
        r,ext = op.splitext(figure_fn)
        plt.savefig(r + '_colorbar' + ext, bbox_inches='tight')

        cmd = ['inkscape', tmp_svg_fn, '--export-png=%s' % figure_fn]
        subprocess.check_output(cmd)
