import sys
import os
import os.path as op
import shutil
import tempfile
import unittest
import logging
import subprocess

import re

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pyvista as pv
pv.global_theme.transparent_background = True

from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
import nipic.csvd as svd

logging.basicConfig()
logger = logging.getLogger('nipic')

def norm01(arr):
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val)


def save_img_with_new_dtype(data, image, out_fn):
    hd = image.header
    new_image = nib.Nifti2Image(data, image.affine, header=hd)
    nib.save(new_image, out_fn)

CUBE_CORNERS = np.array([
    [0,0,0], [0,0,1], [0,1,0], [0,1,1],
    [1,0,0], [1,0,1], [1,1,0], [1,1,1]
    ], dtype=float)


MEASURE_RE = re.compile('^# Measure .*$', flags=re.MULTILINE)
STATS_TABLE_RE = re.compile(r'^(# ColHeaders.*\n(?:.+\n)+)', flags=re.MULTILINE)

import io
import pandas as pd

def get_subjects_dir(fs_home=None):
    home = (fs_home if fs_home is not None
            else os.getenv('FREESURFER_HOME'))
    if len(home) == 0:
        raise Exception('FREESURFER_HOME env variable not found. ' \
                        'Make sure to source SetUpFreeSurfer.sh or '\
                        'check freesurfer installation.')
    subjects_dir = os.getenv('SUBJECTS_DIR')
    if subjects_dir is None or len(subjects_dir) == 0:
        raise Exception('SUBJECTS_DIR env variable not defined. ' \
                        'Make sure to source SetUpFreeSurfer.sh or '\
                        'check freesurfer installation.')
    return subjects_dir

class Freesurfer:

    def __init__(self, fs_home=None, subject_dir=None):
        self.home = (fs_home if fs_home is not None
                        else os.getenv('FREESURFER_HOME'))
        if len(self.home) == 0:
            raise Exception('FREESURFER_HOME env variable not found. ' \
                            'Make sure to source SetUpFreeSurfer.sh or '\
                            'check freesurfer installation.')
        self.subjects_dir = (subject_dir if subject_dir is not None
                             else os.getenv('SUBJECTS_DIR'))
        if self.subjects_dir is None or len(self.subjects_dir) == '':
            raise Exception('SUBJECTS_DIR env variable not defined. ' \
                            'Make sure to source SetUpFreeSurfer.sh or '\
                            'check freesurfer installation.')

        logger.debug('FS home: %s', self.home)
        logger.debug('FS subjects dir: %s', self.subjects_dir)

        self.tmp_dir = tempfile.mkdtemp()


    def stat_seg_to_df(self, subject, segmentation_label='aseg', 
                       struct_names=None, struct_stats=None,
                       measure_labels=None):

        stat_fn = self.stats_fn(subject, segmentation_label + '.stats')
        if not op.exists(stat_fn):
            logger.error('No stat file found for %s: %s', subject,  stat_fn)
            return None
        with open(stat_fn) as fin:
            content = fin.read()

        def safe_unit_suffix(u):
            if u == 'unitless':
                return ''
            su = '_' + u.replace('^', '',).replace(' ', '')
            assert(su.isidentifier())
            return su

        measures = pd.DataFrame({'subject' : [subject]})
        for measure_line in MEASURE_RE.findall(content):
            # # Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1109112.000000, mm^3
            _, ml, ll, m, mu = [e.strip() for e in measure_line.replace('# Measure ', '').split(',')]
            if 'Number' in ll:
                m = int(m)
            else:
                m = float(m)
            if measures is None or ml in measure_labels:
                measures[ml + safe_unit_suffix(mu)] = m
                
            if ml == 'eTIV':
                eTIV = m
        measures.set_index('subject', inplace=True)

        stats_table_str = STATS_TABLE_RE.findall(content)[0].replace('# ColHeaders', '')
        lines = stats_table_str.split('\n')
        header = lines[0].strip().split(' ')
        stats = pd.read_csv(io.StringIO('\n'.join(lines[1:])), header=None,
                            delim_whitespace=True)
        stats.columns = header
        stats.drop(columns=['Index', 'SegId'], inplace=True)

        
        for col in stats.columns:
            if col != 'StructName':
                stats[col + '-to-eTIV'] = stats[col] / eTIV
        
        if struct_stats is not None:
            stats = stats[['StructName'] + struct_stats]

        if struct_names is not None:
            stats = stats[stats['StructName'].isin(struct_names)]

        stats = stats.set_index('StructName').stack().to_frame()
        stats.index = ['_'.join(col) for col in stats.index.values]
        
        
            
        
        stats = stats.T
        stats['subject'] = [subject]
        stats.set_index('subject', inplace=True)
   

        return pd.concat((measures, stats), axis=1)
            

    def stat_seg_measure_global(self, measure_label, subject_names=None, segmentation_label='aseg'):
        pass
        
    def stat_seg_measure_regional(self, measure_label, struct_names=None, subject_names=None, segmentation_label='aseg'):
        pass

    def auto_angio_lesions(self, subject_name, save_figures=False):
        # load WM mask
        aseg_img = nib.load(self.mri_fn(subject_name, 'aseg.mgz'))
        aseg = aseg_img.get_fdata().astype(int)
        logger.debug('Loaded segmentation of size %s, '\
                     'intensity range [%d - %d], dtype %s',
                     aseg.shape, aseg.min(), aseg.max(), aseg.dtype)

        gm = np.bitwise_or(aseg==3, aseg==42)
        gm_mask = np.where(gm)

        vtc = np.any((aseg==4, aseg==43, aseg==14, aseg==15),
                     axis=0)
        vtc_mask = np.where(vtc)

        bg_mask = np.where(aseg==0)
        # coarse WM mask: everything than is supratentorial, not in cortex
        # and not in ventricules
        wm = np.bitwise_not(np.bitwise_or(gm, vtc))
        wm[bg_mask] = False
        # remove infratentorial structures, brain stem, CSF
        # also remove plexus coroid
        not_cerebrum = np.any((aseg==46, aseg==47, aseg==7, aseg==8,
                               aseg==16, aseg==6, aseg==44, aseg==24,
                               aseg==31, aseg==63, aseg==5),
                              axis=0)
        wm[np.where(not_cerebrum)] = False
        wm_mask = np.where(wm)

        # load FLAIR
        flair_img = nib.load(self.mri_fn(subject_name, 'FLAIR.mgz'))
        flair = flair_img.get_fdata()
        logger.debug('Loaded FLAIR of size %s, '\
                     'intensity range [%d - %d], dtype %s',
                     flair.shape, flair.min(), flair.max(), flair.dtype)
        flair_wm = flair[wm_mask]
        # flair_gm = flair[gm_mask]
        flair_vtc = flair[vtc_mask]

        # load T1
        t1_img = nib.load(self.mri_fn(subject_name, 'T1.mgz'))
        t1 = t1_img.get_fdata()
        logger.debug('Loaded T1 of size %s, '\
                     'intensity range [%d - %d], dtype %s',
                     t1.shape, t1.min(), t1.max(), t1.dtype)
        t1_wm = t1[wm_mask]
        # t1_gm = t1[gm_mask]
        t1_vtc = t1[vtc_mask]

        ## Extract White Matter Hyperintensities

        t1_raw_img = nib.load(self.mri_fn(subject_name, 'orig.mgz'))
        t1_raw = t1_img.get_fdata()


        flair_raw_img = nib.load(self.mri_fn(subject_name, 'FLAIR_reg_to_orig.mgz'))
        flair_raw = flair_raw_img.get_fdata()
        wm_no_subcortical_gm = wm.copy()
        subcortical_gm = np.any((aseg==11, aseg==50, # caudate
                                 aseg==12, aseg==51, # putamen
                                 aseg==13, aseg==52, # pallidium
                                 aseg==10, aseg==49, # thalamus
                                 aseg==26, aseg==58, # Accumbens
                                 aseg==17, aseg==53, # Hypocampus
                                 aseg==18, aseg==54, # Amydgala
                                 aseg==28, aseg==60, # ventral DC
        ), axis=0)
        wm_no_subcortical_gm[np.where(subcortical_gm)] = 0
        wm_noscgm_fn = self.mri_fn(subject_name, 'wm_no_subcortical_gm.mgz')
        save_img_with_new_dtype(wm_no_subcortical_gm, aseg_img, wm_noscgm_fn)
        logger.info('Segmentation of white matter withou subcortical structures saved to %s',
                    wm_noscgm_fn)

        # Load SAMSEG's WMH PPM
        samseg_wmh_ppm_fn = self.samseg_fn(subject_name,
                                           op.join('posteriors',
                                                   'WM-hypointensities.mgz'))
        samseg_wmh_ppm_img = nib.load(samseg_wmh_ppm_fn)
        samseg_wmh_ppm = samseg_wmh_ppm_img.get_fdata()
        wmh = np.zeros_like(aseg)
        wmh[np.where(samseg_wmh_ppm >= svd.WMH_SAMSEG_PPM_THRESHOLD)] = svd.WMH
        from IPython import embed; embed()

        flair_raw_wm = flair_raw[wm_no_subcortical_gm]
        flair_wmh_thresh = np.median(flair_raw_wm) + np.std(flair_raw_wm)
        wmh[np.where(np.bitwise_and(wmh==svd.WMH,
                                    flair_raw <= flair_wmh_thresh))] = svd.WMH_SAMSEG_FP

        if 0:
            t1_raw_wm = t1_raw[wm_no_subcortical_gm]
            t1_wmh_thresh = np.median(t1_raw_wm) - 0.5 * np.std(flair_raw_wm)
            flair_wmh_thresh = np.median(flair_raw_wm) +  np.std(flair_raw_wm)
            # from IPython import embed; embed()

            missed_wmh = np.bitwise_and(wmh == 0,
                                        np.bitwise_and(t1_raw < t1_wmh_thresh,
                                                       flair_raw > flair_wmh_thresh))
            wmh[np.where(missed_wmh)] = 498

        wmh[np.where(np.bitwise_not(wm_no_subcortical_gm))] = 0

        if 0:
            # Clean up isolated voxels
            for c_lab, c_loc in enumerate(ndi.find_objects(wmh==svd.WMH)):
                m_cluster = np.where(wmh == c_lab+1)
                if len(m_cluster[0]) < svd.WMH_MIN_CLUSTER_SIZE:
                    wmh[m_cluster] = 0

        wmh_fn = self.mri_fn(subject_name, 'WMH.mgz')
        save_img_with_new_dtype(wmh, aseg_img, wmh_fn)

        ## Extract Lacunes and PVS
        flair_hypo_thresh_pct = 99
        flair_hypo_thresh = np.percentile(flair_vtc, flair_hypo_thresh_pct)
        print('flair_hypo_thresh=', flair_hypo_thresh)

        flair_hyper_thresh_pct = 99.5
        flair_hyper_thresh = np.percentile(flair_wm, flair_hyper_thresh_pct)
        flair_max = flair_wm.max()
        print('flair_hyper_thresh=', flair_hyper_thresh)

        t1_hypo_thresh_pct = 95
        t1_hypo_thresh = np.percentile(t1_vtc, t1_hypo_thresh_pct)

        if save_figures:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist2d(flair_wm, t1_wm, bins=100, norm=LogNorm())
            ax.set_xlabel('FLAIR WM')
            ax.set_ylabel('T1 WM')

            ax.add_patch(Rectangle((0, 0), flair_hypo_thresh, t1_hypo_thresh,
                                   edgecolor = 'red',
                                   fill=False,
                                   lw=2))
            ax.add_patch(Rectangle((flair_hyper_thresh, 0),
                                   flair_max-flair_hyper_thresh,
                                   t1_hypo_thresh,
                                   edgecolor = 'green',
                                   fill=False,
                                   lw=2))
            plt.savefig(self.image_fn(subject_name, 'hist_T1_x_FLAIR_WM.png'))

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist2d(flair_vtc, t1_vtc, bins=100, norm=LogNorm())
            ax.set_xlabel('FLAIR VTC')
            ax.set_ylabel('T1 VTC')

            ax.add_patch(Rectangle((0, 0), flair_hypo_thresh, t1_hypo_thresh,
                                   edgecolor = 'red',
                                   fill=False,
                                   lw=2))
            ax.add_patch(Rectangle((flair_hyper_thresh, 0),
                                   flair_max-flair_hyper_thresh,
                                   t1_hypo_thresh,
                                   edgecolor = 'green',
                                   fill=False,
                                   lw=2))
            plt.savefig(self.image_fn(subject_name, 'hist_T1_x_FLAIR_VTC.png'))

        lacunes_pvs = np.bitwise_and(wm, t1 < t1_hypo_thresh)
        lacunes_pvs = np.bitwise_and(lacunes_pvs,
                                     np.bitwise_or(flair <= flair_hypo_thresh,
                                                   flair >= flair_hyper_thresh))
        lacunes_pvs[bg_mask] = False
        lacunes_pvs_tmp_fn = self.mri_fn(subject_name, 'lacunes_pvs_unfiltered.mgz')
        save_img_with_new_dtype(lacunes_pvs, aseg_img, lacunes_pvs_tmp_fn)

        min_cluster_voxel_size = 4
        unit = 'mm'

        voxel_size = t1_img.header.get_zooms()

        logger.info('Clean lacune clusters...')
        labels, nlabels = ndi.label(lacunes_pvs)
        lacunes, pvs = np.zeros_like(lacunes_pvs), np.zeros_like(lacunes_pvs)
        for c_lab, c_loc in enumerate(ndi.find_objects(labels)):
            cluster = ndi.binary_fill_holes(labels[c_loc]==c_lab+1)
            m_cluster = np.where(labels == c_lab+1)
            if cluster.sum() >= min_cluster_voxel_size:
                cluster_surf, cluster_voxel_edges = \
                    smooth_object_from_label(cluster, voxel_size)

                tag = ''
                if (cluster_surf.max_diameter >= svd.LACUNE_MIN_LENGTH_MM and
                    cluster_surf.length <= svd.LACUNE_MAX_LENGTH_MM and
                    cluster_surf.isoperimeter_quotient > svd.LACUNE_MIN_ISOPQ):
                    lacunes[m_cluster] = True
                    tag = '_lacune_'
                elif (cluster_surf.isoperimeter_quotient >= svd.PVS_MIN_ISOPQ and
                      cluster_surf.isoperimeter_quotient < svd.PVS_MAX_ISOPQ):
                    pvs[m_cluster] = True
                    tag = '_pvs_'
                else:
                    tag = '_filtered_'
                    if cluster_surf.isoperimeter_quotient >= svd.LACUNE_MIN_ISOPQ:
                        if cluster_surf.max_diameter < svd.LACUNE_MIN_LENGTH_MM:
                            tag += ('_lacune_too_small_%1.2f%s' %
                                    (cluster_surf.max_diameter, unit))
                        elif cluster_surf.length > svd.LACUNE_MAX_LENGTH_MM:
                            tag += ('_lacune_too_big_%1.2f%s' %
                                    (cluster_surf.length, unit))
                    elif cluster_surf.isoperimeter_quotient < svd.PVS_MIN_ISOPQ:
                        tag += '_unfit_shape_'

                image_fn = self.image_fn(subject_name,
                                         'cavity_cluster_%d%s_ipq_%1.2f.png' %
                                         (c_lab+1, tag,
                                          cluster_surf.isoperimeter_quotient))
                if save_figures:
                    pl = pv.Plotter(off_screen=True)
                    pl.add_mesh(cluster_surf, show_edges=True,
                                scalars=cluster_surf.curvature(), cmap='autumn',
                                show_scalar_bar=False)
                    pl.add_mesh(cluster_voxel_edges,
                                show_scalar_bar=False, color='k',
                                line_width=2)
                    pl.show(screenshot=image_fn)

        lp_seg = np.zeros_like(aseg)
        lp_seg[np.where(lacunes)] = svd.LACUNE
        lp_seg[np.where(pvs)] = svd.PV_SPACE
        lp_seg_out_fn = self.mri_fn(subject_name, 'aseg_lp.mgz')
        save_img_with_new_dtype(lp_seg, aseg_img, lp_seg_out_fn)

        aseg[np.where(lacunes)] = svd.LACUNE
        aseg[np.where(pvs)] = svd.PV_SPACE
        aseg_out_fn = self.mri_fn(subject_name, 'aseg_angiol.mgz')
        save_img_with_new_dtype(aseg, aseg_img, aseg_out_fn)

        logger.info('Coarse segmentation of angiopathic lesions saved to %s',
                    aseg_out_fn)


    def coreg_other_mri(self, other_mri_fn, contrast_type):
        # TODO...
        mri_import_cmd_pat = ''
        mc = MRIConvert()
        mc.inputs.in_file = other_mri_fn
        imported_mri_fn = self.mri_fn(op.splitext(op.basename(other_mri_fn))[0] +
                                      '_raw.mgz' )
        mc.inputs.out_file = imported_mri_fn
        mc.inputs.out_type = 'mgz'
        mc.run()

        bbreg = BBRegister(subject_id='me', source_file='structural.nii',
                           init='header', contrast_type=contrast_type)

    def clean(self):
        shutil.rmtree(self.tmp_dir)

    def load_lut(self, add_csvd=True):
        lut_fn = op.join(self.home, 'FreeSurferColorLUT.txt')
        if not op.exists(lut_fn):
            raise Exception('Standard FS LUT file not found: %s' % lut_fn)

        lut = {}
        with open(lut_fn) as flut:
            for line in flut.readlines():
                line = line.strip('\n')
                if not line.startswith('#') and not len(line) == 0:
                    toks = line.split()
                    lut[int(toks[0])] = {'name':toks[1],
                                         'color':[int(toks[2]),
                                                  int(toks[3]),
                                                  int(toks[4]),
                                                  int(toks[5])]}

        if add_csvd:
            lut.update(svd.fs_csvd_lut)
        return lut

    def mri_fn(self, subject_name, volume_bfn):
        return op.join(self.subject_dir(subject_name), 'mri',
                       volume_bfn)

    def samseg_fn(self, subject_name, volume_bfn):
        return op.join(self.subject_dir(subject_name), 'mri', 'samseg',
                       volume_bfn)

    def tmp_fn(self, subject_name, tmp_bfn):
        return op.join(self.subject_dir(subject_name), 'tmp',
                       tmp_bfn)

    def surface_fn(self, subject_name, surface_bfn):
        return op.join(self.subject_dir(subject_name), 'surf',
                       surface_bfn)

    def image_fn(self, subject_name, image_bfn):
        return op.join(ensure_dir_exists(op.join(self.subject_dir(subject_name),
                                                 'image')),
                       image_bfn)

    def stats_fn(self, subject_name, stats_bfn):
        return op.join(ensure_dir_exists(op.join(self.subject_dir(subject_name),
                                                 'stats')),
                       stats_bfn)

    def subjects(self):
        subjects = []
        for subfolder in os.listdir(self.subjects_dir):
            if (subfolder != 'fsaverage' and 
                op.exists(op.join(self.subjects_dir, subfolder, 'surf', 'lh.pial'))):
                subjects.append(subfolder)
        return subjects

    def subject_dir(self, subject_name):
        return check_file_exists(op.join(self.subjects_dir, subject_name))

    def init_angio_lesions_graph(self, subject_name, fs_aseg_bfn,
                                 output_graph_bfn, force=False):
        output_graph_fn = self.mri_fn(subject_name, output_graph_bfn)
        if force or not op.exists(output_graph_fn):
            input_seg_fn = self.mri_fn(subject_name, fs_aseg_bfn)
            tmp_seg_fn = op.join(self.tmp_dir, 'vol_S16.nii')

            # TODO: 
            #       Apply multimodal intensity thresholds to segment lesions
            # detect_angio_lesions(input_seg_fn, output_lesion_seg_fn)
            # be sure to save as S16 so that conversion is no longer needed

            cmd = ['AimsFileConvert', '-i', input_seg_fn, '-o', tmp_seg_fn,
                   '-t', 'S16']
            logger.info('Save input segmentation as S16 volume')
            self.run_cmd(cmd)

            aseg_graph_fn = self.mri_fn(subject_name,
                                        op.splitext(fs_aseg_bfn)[0] + '.arg')
            cmd = ['AimsGraphConvert', '-i', tmp_seg_fn, '-o',
                   aseg_graph_fn, '--bucket']
            logger.info('Convert to ROI graph')
            self.run_cmd(cmd)

            aseg_agraph = aims.read(aseg_graph_fn)
            all_rois = aseg_agraph.vertices().list()

            logger.info('Fix roi names according to FS LUT')
            # TODO: only fix names (assume no missing or extra ROI)
            missing_roi_indexes = set(al.fs_angio_lut.keys())
            for roi in all_rois:
                if roi['roi_label'] not in al.fs_angio_lut:
                    aseg_agraph.removeVertex(roi)
                else:
                    fs_roi_def = fs_lut[roi['roi_label']]
                    roi['name'] = fs_roi_def['name']
                    missing_roi_indexes.remove(roi['roi_label'])

            if len(missing_roi_indexes) > 0:
                for angio_roi_idx in missing_roi_indexes:
                    roi_vertex = aseg_agraph.addVertex('roi')
                    roi_vertex['name'] = al.fs_angio_lut[angio_roi_idx]['name']
                    roi_vertex['roi_label'] = angio_roi_idx
                    logger.info('Add missing roi to aseg graph: %s',
                                roi_vertex['name'])

            logger.info('Save angio lesions graph')
            aims.write(aseg_agraph, output_graph_fn)
        else:
            logger.info('Aseg graph already exists')

    def run_cmd(self, cmd):
        if sys.version_info[0] < 3:
            result = Dummy()
            result.returncode = subprocess.call(cmd)
        else:
            result = subprocess.run(cmd)
        if result.returncode != 0:
            msg = 'Error running command %s' % ' '.join(cmd)
            raise RuntimeError(msg)


class TestSmoothing(unittest.TestCase):
    def setUp(self):
        logger.setLevel(logging.DEBUG)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def plot_cluster(self, surf, voxels, image_fn=None):
        pl = pv.Plotter(off_screen=image_fn is not None)
        pl.add_mesh(surf, show_edges=True,
                    show_scalar_bar=False)
        pl.add_mesh(voxels,
                    show_scalar_bar=False, color='k',
                    line_width=2)
        pl.show(screenshot=image_fn)
        
    def test_volume_constraint(self):
        cluster = np.array([[[1,1,0],
                             [0,1,1],
                             [0,0,1]]], dtype=int)

        vdims = (1,1,1)
        ref_vol = None
        logger.debug('Save images of smoothed cluster to: %s', self.tmp_dir) 
        for os_factor in (1, 2, 5, 10):
            cluster_surf, cluster_voxel_edges = \
                smooth_object_from_label(cluster, vdims, os_factor=os_factor)
            image_fn = op.join(self.tmp_dir, 'cluster_osf_%d.png' % os_factor)
            self.plot_cluster(cluster_surf, cluster_voxel_edges, image_fn)
            if ref_vol is None:
                ref_vol = cluster_surf.volume
            else:
                self.assert_equal(cluster_surf.volume, ref_vol)

class TestFreesurfer(unittest.TestCase):

    def setUp(self):
        logger.setLevel(logging.DEBUG)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_subject_dir(self):
        fs = Freesurfer()
        self.assertTrue('bert' in fs.subject_dir('bert'))

def check_file_exists(fn):
    if not op.exists(fn):
        raise FileNotFoundError(fn)
    return fn

def ensure_dir_exists(folder):
    if not op.exists(folder):
        os.makedirs(folder)
    return folder

def voxels_corners(bin_mask, voxel_size):
    positions = np.array(np.where(bin_mask))
    ndim, npos = positions.shape
    n_corners = CUBE_CORNERS.shape[0]
    all_corners = np.empty((npos * n_corners, ndim), dtype=float)
    for ipos, position in enumerate(positions.T):
        all_corners[ipos*n_corners:(ipos+1)*n_corners, :] = \
            CUBE_CORNERS + position
    return np.unique(all_corners, axis=0) * voxel_size


def measure_object(label, dims):
    label = np.lib.pad(label, ((1, 1), (1, 1), (1, 1)))
    zy_face_area = dims[1] * dims[2]
    xz_face_area = dims[0] * dims[2]
    xy_face_area = dims[0] * dims[1]
    area = 0.0
    bg = np.bitwise_not(label).astype(np.uint8)
    for i,j,k in np.array(np.where(label)).T:
        area += (zy_face_area * (bg[i-1, j, k] + bg[i+1, j, k]) +
                 xz_face_area * (bg[i, j-1, k] + bg[i, j+1, k]) +
                 xy_face_area * (bg[i, j, k-1] + bg[i, j, k+1]))
    volume = label.sum() * np.prod(dims)
    isoperimeter_quotient = 36 * np.pi * volume**2 / area**3
    length = np.max(pdist(voxels_corners(label, dims)))
    return area, volume, length, isoperimeter_quotient

import pyvista as pv

def smooth_object_from_label(label, dims, os_factor=3, tbs_n_iter=50,
                             tbs_pass_band=0.08,
                             tbs_normalize_coordinates=True):

    obj_size = label.sum()
    min_sample_size = 10
    if os_factor is not None and os_factor > 1:
        #os_factor = (os_factor if os_factor is not None else
        #             int(np.ceil(min_sample_size / obj_size)))
        label = (label
                 .repeat(os_factor, axis=0)
                 .repeat(os_factor, axis=1)
                 .repeat(os_factor, axis=2))
        dims = [d/os_factor for d in dims]

    grid = pv.UniformGrid()
    grid.dimensions = np.array(label.shape) + 1
    grid.spacing = dims
    grid.cell_data["values"] = label.flatten(order="F")

    vol = grid.threshold(0.1)
    surf = vol.extract_geometry()
    orig_edges = surf.extract_feature_edges()
    smooth_surf = surf.smooth_taubin(n_iter=tbs_n_iter, pass_band=tbs_pass_band,
                                     normalize_coordinates=tbs_normalize_coordinates)

    volume = smooth_surf.volume
    area = smooth_surf.area

    print(f'Original surface volume:   {surf.volume:.1f}')
    print(f'Original surface area:   {surf.area:.1f}')
    print(f'Taubin smoothed volume:    {smooth_surf.volume:.1f}')
    print(f'Taubin smoothed area:    {smooth_surf.area:.1f}')

    smooth_surf.isoperimeter_quotient = 36 * np.pi * volume**2 / area**3
    smooth_surf.max_diameter = np.max(pdist(smooth_surf.points))

    print(f'Isoperimeter_Quotient:   {smooth_surf.isoperimeter_quotient:.3f}')
    print(f'Length:    {smooth_surf.max_diameter:.1f}')

    return smooth_surf, orig_edges

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


import nipype.interfaces.io as nio
import nipype.pipeline as nppl
import nipype.interfaces.utility as nut
import nipype.interfaces.base as nifbase

class ReconAllInputSpec(nifbase.TraitedSpec):
    t1_fn = nifbase.File(desc="T1", exists=True,
                         mandatory=True, argstr='-i %s')
    flair_fn = nifbase.File(desc="FLAIR", exists=True,
                            mandatory=True, argstr='-FLAIR %s')
    fs_subject = nifbase.Str(desc="fs_subject", mandatory=True, 
                             argstr='-s %s')


recon_outputs = {"ASEG" : ('mri', 'aseg.mgz'),
"RIBBON": "{subject_id}/mri/ribbon.mgz",
"ANNOT_LH": "{subject_id}/label/lh.aparc.annot",
"ANNOT_RH": "{subject_id}/label/rh.aparc.annot",
"WHITE_LH": "{subject_id}/surf/lh.white",
"WHITE_RH": "{subject_id}/surf/rh.white",
"PIAL_LH": "{subject_id}/surf/lh.pial",
"PIAL_RH": "{subject_id}/surf/rh.pial",
"subject_id": "{subject_id}"}
class ReconAllOutputSpec(nifbase.TraitedSpec):
    aseg =  nifbase.File(desc='aseg', exists=True)  
    
class ReconAllInterface(nifbase.Interface):
    input_spec = ReconAllInputSpec
    output_spec = ReconAllOutputSpec
    _cmd = 'recon-all -all -parallel -FLAIRpial'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        fs = Freesurfer()
        for output_name in list(outputs.keys()):
            outputs['aseg'] = fs.image_fn(self.inputs.fs_subject, output_name)
        return outputs

    def _run_interface(self, runtime):
        super(ReconAllInterface, self)._run_interface(runtime)


