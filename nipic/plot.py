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

from .utils import two_intervals_intersection, if_none, if_nan, min_pos_value, max_neg_value

def two_thresh_cmap(vmin, vmax, col_vmin=None, col_vmax=None,
                    col_thresh_neg=None, col_thresh_pos=None,
                    vthresh_neg=0.0, vthresh_pos=0.0,
                    col_center=None, clip=False, N=256):

    assert(vthresh_neg is not None)
    assert(vthresh_pos is not None)
    if vthresh_pos < 0:
        raise ValueError('vthresh_pos not positive (%f)' % vthresh_pos)
    if vthresh_neg > 0:
        raise ValueError('vthresh_neg not negative (%f)' % vthresh_neg)

    col_vmin = if_none(col_vmin, np.array([0.0, 0.0, 1.0, 1.0]))
    col_thresh_neg = if_none(col_thresh_neg, np.array([0.5, 1.0, 1.0, 1.0]))
    col_thresh_pos = if_none(col_thresh_pos, np.array([1.0, 1.0, 0.5, 1.0]))
    col_vmax = if_none(col_vmax, np.array([1.0, 0, 0, 1.0]))
    col_center = if_none(col_center, np.array([0.5, 0.5, 0.5, 1.0]))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)

    logger.debug('Compute gray interval from vmin=%f, vmax=%f, vthresh_neg=%f, vthresh_pos=%f',
                 vmin, vmax, vthresh_neg, vthresh_pos)
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

brain_region_svg_templates = {}

brain_region_svg_templates[('2x4', 'with_colorbar')] = \
"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
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
   width="244.7162mm"
   height="122.6767mm"
   viewBox="0 0 244.7162 122.6767"
   version="1.1"
   id="svg8"
   inkscape:version="0.92.5 (2060ec1f9f, 2020-04-08)"
   sodipodi:docname="template_2x4_with_colorbar.svg">
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
     inkscape:cx="449.82763"
     inkscape:cy="61.299412"
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
     transform="translate(253.85593,-123.54207)">
    <rect
       style="fill:#000000;fill-opacity:1;stroke:none;stroke-width:0.35268956;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       id="rect903-3"
       width="244.7162"
       height="122.6767"
       x="-253.85593"
       y="123.54207"
       ry="0" />
    <image
       y="134.93954"
       x="-30.374321"
       id="image829"
       preserveAspectRatio="none"
       height="102.362"
       width="21.082001"
       xlink:href="colorbar.png"
       sodipodi:absref="colorbar.png" />
    <image
       sodipodi:absref="brain_regions_s04.png"
       xlink:href="brain_regions_s04.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image867"
       x="-253.8723"
       y="179.62534" />
    <image
       sodipodi:absref="brain_regions_s05.png"
       xlink:href="brain_regions_s05.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image878"
       x="-197.99229"
       y="179.62534" />
    <image
       sodipodi:absref="brain_regions_s06.png"
       xlink:href="brain_regions_s06.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image889"
       x="-142.11229"
       y="179.62534" />
    <image
       sodipodi:absref="brain_regions_s07.png"
       xlink:href="brain_regions_s07.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image900"
       x="-86.232285"
       y="179.62534" />
    <image
       sodipodi:absref="brain_regions_s00.png"
       xlink:href="brain_regions_s00.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image823"
       x="-253.83551"
       y="116.1511" />
    <image
       sodipodi:absref="brain_regions_s01.png"
       xlink:href="brain_regions_s01.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image834"
       x="-197.95551"
       y="116.1511" />
    <image
       sodipodi:absref="brain_regions_s02.png"
       xlink:href="brain_regions_s02.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image845"
       x="-142.0755"
       y="116.1511" />
    <image
       sodipodi:absref="brain_regions_s03.png"
       xlink:href="brain_regions_s03.png"
       width="55.880001"
       height="76.199997"
       preserveAspectRatio="none"
       id="image856"
       x="-86.195496"
       y="116.1511" />
  </g>
</svg>"""

brain_region_svg_templates[('2x4', 'without_colorbar')] = \
"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
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
   width="223.5229mm"
   height="122.6767mm"
   viewBox="0 0 223.5229 122.6767"
   version="1.1"
   id="svg8"
   inkscape:version="0.92.5 (2060ec1f9f, 2020-04-08)"
   sodipodi:docname="template_2x4_without_colorbar.svg">
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
     inkscape:cx="662.75905"
     inkscape:cy="270.29716"
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
     transform="translate(253.85593,-123.54207)">
    <rect
       style="fill:#000000;fill-opacity:1;stroke:none;stroke-width:0.35268956;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       id="rect903-3"
       width="223.5229"
       height="122.6767"
       x="-253.85593"
       y="123.54207"
       ry="0" />
    <image
       sodipodi:absref="brain_regions_s04.png"
       xlink:href="brain_regions_s04.png"
       y="179.62534"
       x="-253.8723"
       id="image867"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s05.png"
       xlink:href="brain_regions_s05.png"
       y="179.62534"
       x="-197.99229"
       id="image878"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s06.png"
       xlink:href="brain_regions_s06.png"
       y="179.62534"
       x="-142.11229"
       id="image889"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s07.png"
       xlink:href="brain_regions_s07.png"
       y="179.62534"
       x="-86.232285"
       id="image900"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s00.png"
       xlink:href="brain_regions_s00.png"
       y="116.1511"
       x="-253.83551"
       id="image823"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s01.png"
       xlink:href="brain_regions_s01.png"
       y="116.1511"
       x="-197.95551"
       id="image834"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s02.png"
       xlink:href="brain_regions_s02.png"
       y="116.1511"
       x="-142.0755"
       id="image845"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s03.png"
       xlink:href="brain_regions_s03.png"
       y="116.1511"
       x="-86.195496"
       id="image856"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
  </g>
</svg>
"""

brain_region_svg_templates[('1x8', 'with_colorbar')] = \
"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
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
   width="458.72604mm"
   height="54.609253mm"
   viewBox="0 0 458.72604 54.609254"
   version="1.1"
   id="svg8"
   inkscape:version="0.92.5 (2060ec1f9f, 2020-04-08)"
   sodipodi:docname="template_1x8_with_colorbar.svg">
  <defs
     id="defs2" />
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="2.8"
     inkscape:cx="1532.867"
     inkscape:cy="137.37458"
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
     transform="translate(357.02429,-128.36832)">
    <rect
       style="fill:#000000;fill-opacity:1;stroke:none;stroke-width:0.35268956;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       id="rect844"
       width="458.72604"
       height="54.609253"
       x="-357.02429"
       y="128.36832"
       ry="0.25793117" />
    <image
       y="127.97025"
       x="90.017021"
       id="image829"
       preserveAspectRatio="none"
       height="55.952267"
       width="11.523668"
       xlink:href="colorbar.png"
       sodipodi:absref="colorbar.png" />
    <image
       sodipodi:absref="brain_regions_s04.png"
       xlink:href="brain_regions_s04.png"
       y="116.1511"
       x="-133.50299"
       id="image867"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s05.png"
       xlink:href="brain_regions_s05.png"
       y="116.1511"
       x="-77.622986"
       id="image878"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s06.png"
       xlink:href="brain_regions_s06.png"
       y="116.1511"
       x="-21.742985"
       id="image889"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s07.png"
       xlink:href="brain_regions_s07.png"
       y="116.1511"
       x="34.13702"
       id="image900"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s00.png"
       xlink:href="brain_regions_s00.png"
       y="116.1511"
       x="-357.02301"
       id="image823"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s01.png"
       xlink:href="brain_regions_s01.png"
       y="116.1511"
       x="-301.14301"
       id="image834"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s02.png"
       xlink:href="brain_regions_s02.png"
       y="116.1511"
       x="-245.263"
       id="image845"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
    <image
       sodipodi:absref="brain_regions_s03.png"
       xlink:href="brain_regions_s03.png"
       y="116.1511"
       x="-189.383"
       id="image856"
       preserveAspectRatio="none"
       height="76.199997"
       width="55.880001" />
  </g>
</svg>
"""
from .freesurfer import load_lut
from nilearn.plotting import plot_img
import nibabel as nib

from IPython import embed
def draw_vol_mapping(values_to_map, region_template_fn, t1_template_fn,
                     cache_dir=None, output_vol_fn=None, figure_fn=None,
                     vmin=None, vmax=None,
                     col_vmin=None, col_vmax=None,
                     use_double_threshold=True,
                     vthresh_neg=None, vthresh_pos=None,
                     col_thresh_neg=None, col_thresh_pos=None,
                     colorbar=True, colorbar_label=None,
                     ignore_regions_not_in_template=False,
                     layout='2x4'):

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
        img = nib.Nifti1Image(out_data, template_img.affine)
        return img

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

    if output_vol_fn is not None:
        nib.save(out_img, output_vol_fn)

    def masked_fdata(self):
        fdata = self.get_fdata()
        return np.ma.masked_where(np.isnan(fdata), fdata)
    out_img.get_fdata = masked_fdata

    if vmin is None:
        vmin = values_to_map.min()

    if vmax is None:
        vmax = values_to_map.max()

    if use_double_threshold:
        if vthresh_neg is None:
            vthresh_neg = if_nan(max_neg_value(values_to_map), 0.0)
        if vthresh_pos is None:
            vthresh_pos = if_nan(min_pos_value(values_to_map), 0.0)
    else:
        vthresh_neg, vthresh_pos = 0.0, 0.0

    color_map, norm = two_thresh_cmap(vmin, vmax, col_vmin=col_vmin, col_vmax=col_vmax,
                                      col_thresh_neg=col_thresh_neg,
                                      col_thresh_pos=col_thresh_pos,
                                      vthresh_neg=vthresh_neg, vthresh_pos=vthresh_pos)

    # transparency_img = nib.Nifti1Image(out_transparency, template_img.affine)

    plt.style.use('dark_background')

    nb_cuts = np.prod([int(e) for e in layout.split('x')])
    assert(nb_cuts > 0)
    assert(nb_cuts < 50)

    cuts = np.linspace(-93,63, nb_cuts)

    colorbar_tag = ['without_colorbar', 'with_colorbar'][colorbar]
    icut = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_svg_fn = op.join(tmp_dir, 'region_plot_template.svg')
        with open(tmp_svg_fn, 'w') as fout:
            fout.write(brain_region_svg_templates[(layout, colorbar_tag)])

        if colorbar:
            logger.info('Plot colorbar')
            fig,ax = plt.subplots()
            cbar_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=cbar_norm, cmap=color_map), ax=ax)

            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)
            ax.remove()

            plt.savefig(op.join(tmp_dir, 'colorbar.png'), bbox_inches='tight')

        logger.info('Plot brain snippets (vmin=%1.3f, vmax=%1.3f)',
                    vmin, vmax)

        for icut, cut in enumerate(cuts):
            sub_figure_fn = op.join(tmp_dir, 'brain_regions_s%02d.png' %icut)
            plot_img(out_img, bg_img=nib.load(t1_template_fn), display_mode='y',
                     cut_coords=[cut], output_file=sub_figure_fn,
                     colorbar=False, threshold=0,
                     black_bg=True, vmin=vmin, vmax=vmax, cmap=color_map)

        cmd = ['inkscape', tmp_svg_fn, '--export-png=%s' % figure_fn]
        subprocess.check_output(cmd)
