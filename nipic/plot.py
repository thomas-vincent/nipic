import tempfile
import os.path as op

import pandas as pd

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict

import unittest
from numpy.testing import assert_allclose
import logging
logger = logging.getLogger('nipic')

from .utils import two_intervals_intersection

def two_thresh_cmap(vmin, vmax, col_vmin, col_vmax,
                    col_thresh_neg, col_thresh_pos,
                    vthresh_neg=0.0, vthresh_pos=0.0,
                    col_center=None, clip=False, N=256):

    assert(vthresh_neg is not None)
    assert(vthresh_pos is not None)
    assert(vthresh_pos >= 0)
    assert(vthresh_neg <= 0)

    if col_center is None:
        col_center = np.array([125, 125, 125, 255]) / 255 # gray

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)

    i_gray = norm(two_intervals_intersection((vmin, vmax),
                                             (vthresh_neg, vthresh_pos)))

    def add_col_stop(d, affix, color_before, color_after):
        for i,c in enumerate(('red', 'green', 'blue')):
            d[c].append((affix, color_before[i], color_after[i]))

    cdict = defaultdict(list)
    if i_gray is not None:
        if i_gray[0] == 0.0:
            add_col_stop(cdict, 0.0, col_center, col_center)
        else:
            add_col_stop(cdict, 0.0, col_vmin, col_vmin)
            add_col_stop(cdict, i_gray[0], col_thresh_neg, col_center)

        if i_gray[1] == 1.0:
            add_col_stop(cdict, 1.0, col_center, col_center)
        else:
            add_col_stop(cdict, i_gray[1], col_center, col_thresh_pos)
            add_col_stop(cdict, 1.0, col_vmax, col_vmax)
    else:
        add_col_stop(cdict, 0.0, col_vmin, col_vmin)
        add_col_stop(cdict, norm(0.0), col_thresh_neg, col_thresh_pos)
        add_col_stop(cdict, 1.0, col_vmax, col_vmax)

    cmap = mpl.colors.LinearSegmentedColormap("custom", cdict, N=N)
    return cmap, norm


class TestTwoThreshCMap(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        logger.setLevel(logging.DEBUG)

        self.col_vmin = np.array([0, 0, 255, 255]) / 255
        self.col_thresh_neg = np.array([125, 255, 255, 255]) / 255
        self.col_thresh_pos = np.array([255, 255, 125, 255]) / 255
        self.col_vmax = np.array([255, 0, 0, 255]) / 255
        self.col_center = np.array([100, 100, 100, 255]) / 255

    def _show_cmap(self, cmap, norm, th_neg=0.0, th_pos=0.0):
        np.random.seed(1234)
        data = np.random.uniform(norm.vmin-1, norm.vmax+1, (10, 10))
        data = np.ma.masked_where((data >= th_neg) & (data <= th_pos), data)
        fig, ax  = plt.subplots()
        img = ax.imshow(data, aspect="auto", origin="upper",
                        cmap=cmap, norm=norm)
        fig.colorbar(img, ax=ax)
        plt.show()
        plt.close(fig)

    def test_two_sides(self):
        vmin = -4
        vmax = 5
        vthresh_neg = -2
        vthresh_pos = 3
        N = 512
        cmap, norm = two_thresh_cmap(vmin, vmax, self.col_vmin, self.col_vmax,
                                     col_thresh_neg=self.col_thresh_neg,
                                     col_thresh_pos=self.col_thresh_pos,
                                     vthresh_neg=vthresh_neg,
                                     vthresh_pos=vthresh_pos,
                                     col_center=self.col_center, N=N)
        # self._show_cmap(cmap, norm, vthresh_neg, vthresh_pos)
        assert_allclose(cmap(norm(vmin)), self.col_vmin)
        assert_allclose(cmap(norm(vmax)), self.col_vmax)
        assert_allclose(cmap(norm(vthresh_pos)), self.col_thresh_pos, atol=0.005)
        assert_allclose(cmap(norm(vthresh_neg)), self.col_thresh_neg, atol=0.005)


    def test_only_pos_thresh(self):
        vmin = -4
        vmax = 5
        vthresh_pos = 3
        N = 512
        cmap, norm = two_thresh_cmap(vmin, vmax, self.col_vmin, self.col_vmax,
                                     col_thresh_neg=self.col_thresh_neg,
                                     col_thresh_pos=self.col_thresh_pos,
                                     vthresh_pos=vthresh_pos,
                                     col_center=self.col_center, N=N)
        # self._show_cmap(cmap, norm, vthresh_neg, vthresh_pos)
        assert_allclose(cmap(norm(vmin)), self.col_vmin)
        assert_allclose(cmap(norm(vmax)), self.col_vmax)
        assert_allclose(cmap(norm(0.0)), self.col_thresh_neg, atol=0.005)
        assert_allclose(cmap(norm(vthresh_pos)), self.col_thresh_pos, atol=0.005)

    def test_only_neg_thresh(self):
        vmin = -4
        vmax = 5
        vthresh_neg = -2
        N = 512
        cmap, norm = two_thresh_cmap(vmin, vmax, self.col_vmin, self.col_vmax,
                                     col_thresh_neg=self.col_thresh_neg,
                                     col_thresh_pos=self.col_thresh_pos,
                                     vthresh_neg=vthresh_neg,
                                     col_center=self.col_center, N=N)
        # self._show_cmap(cmap, norm, vthresh_neg)
        assert_allclose(cmap(norm(vmin)), self.col_vmin)
        assert_allclose(cmap(norm(vmax)), self.col_vmax)
        assert_allclose(cmap(norm(0.01)), self.col_thresh_pos, atol=0.005)
        assert_allclose(cmap(norm(vthresh_neg)), self.col_thresh_neg, atol=0.005)

    def test_no_thresh(self):
        vmin = -4
        vmax = 5
        N = 512
        cmap, norm = two_thresh_cmap(vmin, vmax, self.col_vmin, self.col_vmax,
                                     col_thresh_neg=self.col_thresh_neg,
                                     col_thresh_pos=self.col_thresh_pos,
                                     col_center=self.col_center, N=N)
        # self._show_cmap(cmap, norm)
        assert_allclose(cmap(norm(vmin)), self.col_vmin)
        assert_allclose(cmap(norm(vmax)), self.col_vmax)
        assert_allclose(cmap(norm(0.01)), self.col_thresh_pos, atol=0.005)
        assert_allclose(cmap(norm(-0.01)), self.col_thresh_neg, atol=0.005)


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

from .freesurfer import load_lut
from nilearn.plotting import plot_img
import nibabel as nib

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
                sub_figure_fn = op.join(tmp_dir, 'brain_regions_r%02d_c%02d.png' %(irow, icol))
                plot_img(out_img, bg_img=nib.load(t1_template_fn), display_mode='y',
                         cut_coords=[cuts[icut]], output_file=sub_figure_fn,
                         colorbar=False, threshold=0,
                         black_bg=True, vmin=vmin, vmax=vmax, cmap=color_map)
                icut += 1

        cmd = ['inkscape', tmp_svg_fn, '--export-png=%s' % figure_fn]
        subprocess.check_output(cmd)
