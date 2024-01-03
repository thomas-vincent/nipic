import os.path as op
from subprocess import run
from glob import glob

import numpy as np
import nibabel as nib

from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, BaseInterface, traits, TraitedSpec,
    CommandLineInputSpec, CommandLine, File, Directory)

import nipype.pipeline.engine as pe
import nipype.interfaces.freesurfer as fs 

from bullseye_pipeline.bullseye_pipeline import create_bullseye_pipeline

from nipic.utils import (save_img_with_new_dtype, change_color_lightness,
                         color_average_rgba)
from nipic.freesurfer import load_lut, read_lut_file, lut_to_str
from nipic.logging import logger

WMH = 15000
WMH_SAMSEG_FP = 15001
SMALL_INFARCT = 15002
LACUNE = 15003
MICRO_BLEED = 15004
PV_SPACE = 150005

fs_csvd_lut = {
    WMH : {
        'name' : 'WMH',
        'color' : [200, 70, 255, 0]
    },
    WMH_SAMSEG_FP : {
        'name' : 'WMH_samseg_FP',
        'color' : [255, 100, 100, 0]
    },
    SMALL_INFARCT : {
        'name' : 'small_infarct',
        'color' : [200, 70, 255, 0]
    },
    LACUNE : {
        'name' : 'lacune',
        'color' : [200, 70, 255, 0]
    },
    MICRO_BLEED : {
        'name' : 'micro_bleed',
        'color' : [200, 70, 255, 0]
    },
    PV_SPACE : {
        'name' : 'perivascular_space',
        'color' : [200, 70, 255, 0]
    }
}

WMH_SAMSEG_PPM_THRESHOLD = 0.1
WMH_MIN_CLUSTER_SIZE = 3

LACUNE_MIN_LENGTH_MM = 2 # lower than 3mm because of smoothing effect for small obj
LACUNE_MAX_LENGTH_MM = 20 # larger to be more conversative
LACUNE_MIN_ISOPQ = 0.5

PVS_MIN_ISOPQ = 0.3
PVS_MAX_ISOPQ = 0.5

def check_dependencies():
    # Freesurfer
    if run(['recon-all', '--version']).returncode != 0:
        url = 'https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall'
        raise Exception("Freesurfer not found. See %s" % url)

    if run(['samseg', '--version']).returncode != 0:
        raise Exception('SAMSEG not available. Maybe Freesurfer is too old?')

class SAMSEGInputSpec(CommandLineInputSpec):
    FLAIR_file = File(desc="FLAIR file", exists=True, mandatory=True,
                      argstr="-i %s")
    T1_file = File(desc="T1 file", exists=True, mandatory=True,
                   argstr="--input %s")

    output_dir = Directory(desc=('Directory for SAMSEG outputs, '
                                 'usually in mri/samseg'),
                           mandatory=True, argstr='--output %s',
                           hash_files=False) 

    pallidum_separate = traits.Bool(True, mandatory=False,
                                    argstr='--pallidum-separate',
                                    desc='Extract pallidum',
                                    usedefault=True)

    save_PPM = traits.Bool(True, mandatory=False, argstr='--save-posteriors',
                           desc='Save PPM files for all tissues',
                           usedefault=True)

    nb_threads = traits.Int(desc='Number of threads for parallelization',
                            argstr='--threads %d', mandatory=False, default=1)

class SAMSEGOutputSpec(TraitedSpec):
    WMH_PPM_file = File(desc="PPM File", exists=True)

class SAMSEG(CommandLine):
    input_spec = SAMSEGInputSpec
    output_spec = SAMSEGOutputSpec
    _cmd = 'run_samseg'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        
        outputs['WMH_PPM_file'] = glob(op.join(self.inputs.output_dir,
                                               'posteriors',
                                               'WM-hypointensities.*'))[0]
        return outputs

bullseye_lobes_lut = {
    11 : {
        'name' : 'fontral_left',
        'color' : [240,163,255, 0],
    },
    12 : {
        'name' : 'occipital_left',
        'color' : [0,117,220, 0],
    },
    13 : {
        'name' : 'temporal_left',
        'color' : [153,63,0, 0],
    },
    14 : {
        'name' : 'parietal_left',
        'color' : [0,92,49, 0],
    },
    21 : {
        'name' : 'frontal_right',
        'color' : [255,0,16, 0],
    },
    22 : {
        'name' : 'occiptal_right',
        'color' : [94,241,242, 0],
    },
    23 : {
        'name' : 'temporal_right',
        'color' : [0,153,143, 0],
    },
    24 : {
        'name' : 'parietal_right',
        'color' : [76,0,92, 0]
    },
}

bullseye_depth_lut = {
    1 : {
        'name' : 'inner',
        'color' : [255, 255, 128, 0],
    },
    2 : {
        'name' : 'mid_low',
        'color' : [255, 80, 5, 0],
    },
    3 : {
        'name' : 'mid_high',
        'color' : [255, 255, 0, 0],
    },
    4 : {
        'name' : 'outer',
        'color' : [153, 0, 0, 0],
    },
}

bullseye_lut = {}
for lobe_index, lobe_def in bullseye_lobes_lut.items():
    for depth_index, depth_def in bullseye_depth_lut.items():
        bullseye_lut[int(str(lobe_index) + str(depth_index))] = {
            'name' : lobe_def['name'] + '_' + depth_def['name'],
            'color' : color_average_rgba(lobe_def['color'],
                                         depth_def['color'])
        }

def create_csvd_workflow(fs_subjects_dir, fs_subject_id, bullseye_out_dir,
                         nb_threads=1, work_dir=None):
    check_dependencies()
    fs_subject_dir = op.join(fs_subjects_dir, fs_subject_id)

    csvd_work_dir = work_dir + '_csvd' if work_dir is not None else None
    csvd_workflow = pe.Workflow(name='CSVD_workflow',
                                base_dir=csvd_work_dir)
    
    be_work_dir = work_dir + '_bullseye' if work_dir is not None else None
    bullseye_workflow = create_bullseye_pipeline(fs_subjects_dir, be_work_dir,
                                                 bullseye_out_dir,
                                                 [fs_subject_id],
                                                 name='bullseye_parcellation')

    dump_bullseye_lut = pe.Node(BullseyeLUT(), name='bullseye_LUT')
    dump_bullseye_lut.inputs.out_bullseye_lut_file = \
        op.join(bullseye_out_dir, 'bullseye_wmparc_LUT.txt')
    dump_bullseye_lut.inputs.out_depth_lut_file = \
        op.join(bullseye_out_dir, 'shells_wmparc_LUT.txt')

    input_node = pe.Node(IdentityInterface(fields=['fs_subject_dir',
                                                   'nb_threads']),
                         name='inputspec')
    input_node.inputs.nb_threads = nb_threads
    input_node.inputs.fs_subject_dir = fs_subject_dir

    # Coregister raw FLAIR on original T1
    coreg = pe.Node(fs.MRICoreg(), name='coreg_FLAIR_on_T1')
    #mri_coreg --mov FLAIRraw.mgz --ref orig.mgz --reg FLAIR_to_T1.lta
    def _coreg_input(fs_subject_dir):
        from glob import glob
        import os.path as op
        fs_subject_mri_dir = op.join(fs_subject_dir, 'mri')
        orig_fn = glob(op.join(fs_subject_mri_dir, 'orig.*'))[0]
        flair_fn = glob(op.join(fs_subject_mri_dir, 'orig', 'FLAIRraw.*'))[0]
        return [flair_fn, orig_fn]

    coreg_input = pe.Node(Function(input_names=['fs_subject_dir'],
                                   output_names=['FLAIRraw', 'T1_orig'],
                                   function=_coreg_input),
                          name='coreg_input_FLAIR_to_T1')
    csvd_workflow.connect(input_node, 'fs_subject_dir',
                          coreg_input, 'fs_subject_dir')

    csvd_workflow.connect(coreg_input, 'FLAIRraw', coreg, 'source_file')
    csvd_workflow.connect(coreg_input, 'T1_orig', coreg, 'reference_file')

    apply_coreg = pe.Node(fs.ApplyVolTransform(), name="apply_coreg_FLAIR_on_T1")

    def _apply_coreg_output(fs_subject_dir):
        from glob import glob
        import os.path as op
        fs_subject_mri_dir = op.join(fs_subject_dir, 'mri')
        ext = op.splitext(glob(op.join(fs_subject_mri_dir, 'orig.*'))[0])[-1]
        return op.join(fs_subject_mri_dir, 'FLAIRraw_to_orig' + ext)

    apply_coreg_output = pe.Node(Function(input_names=['fs_subject_dir'],
                                          output_names=['FLAIRraw_reg_to_orig'],
                                          function=_apply_coreg_output),
                                 name='output_of_apply_coreg_FLAIR_to_T1')
    csvd_workflow.connect(input_node, 'fs_subject_dir',
                          apply_coreg_output, 'fs_subject_dir')

    csvd_workflow.connect(coreg_input, 'FLAIRraw', apply_coreg, 'source_file')
    csvd_workflow.connect(coreg_input, 'T1_orig', apply_coreg, 'target_file')
    
    csvd_workflow.connect(coreg, 'out_lta_file', apply_coreg, 'reg_file')
    csvd_workflow.connect(apply_coreg_output, 'FLAIRraw_reg_to_orig',
                          apply_coreg, 'transformed_file')

    # Run SAMSEG on FLAIR and T1
    # run_samseg --input 001.mgz FLAIR_reg.mgz --pallidum-separate --save-posteriors --threads 8 --output ../samseg/

    samseg = pe.Node(SAMSEG(), name='samseg_for_wmh')
    samseg.inputs.output_dir = op.join(fs_subject_dir, 'mri', 'samseg')
    samseg.inputs.nb_threads = nb_threads
    csvd_workflow.connect(apply_coreg, 'transformed_file', samseg, 'FLAIR_file')
    csvd_workflow.connect(coreg_input, 'T1_orig', samseg, 'T1_file')

    # Refine SAMSEG outputs
    filter_wmh = pe.Node(SAMSEGFilterWMH(), name='filter_wmh')

    aseg_fn = glob(op.join(fs_subject_dir, 'mri', 'aseg.???'))[0]
    ext = op.splitext(aseg_fn)[-1]
    filter_wmh.inputs.fs_aseg_file = aseg_fn
    filter_wmh.inputs.out_WM_no_GM_mask_file = \
        op.join(fs_subject_dir, 'mri', 'wm_no_subcortical_gm' + ext)
    filter_wmh.inputs.out_filtered_WMH_file = \
        op.join(fs_subject_dir, 'mri', 'WMH' + ext)

    csvd_workflow.connect(apply_coreg, 'transformed_file',
                          filter_wmh, 'fs_FLAIR_raw_file')
    csvd_workflow.connect(samseg, 'WMH_PPM_file',
                          filter_wmh, 'samseg_WMh_posterior_file')

    merge_csvd_seg = pe.Node(MergeCSVDSeg(), name='merge_csvd_seg')
    # TODO refactor parcellation split
    merge_csvd_seg.inputs.fs_aseg_file = aseg_fn
    merge_csvd_seg.inputs.out_merged_csvd_aseg_file = \
        op.join(fs_subject_dir, 'mri', 'aseg_csvd' + ext)
    merge_csvd_seg.inputs.out_merged_csvd_aseg_lut_file = \
        op.join(fs_subject_dir, 'mri', 'aseg_csvd_LUT.txt')
    csvd_workflow.connect(filter_wmh, 'filtered_WMH_file',
                          merge_csvd_seg, 'wmh_file')

    csvd_split_be = pe.Node(SplitCSVDSeg(), name='split_bullseye')
    csvd_split_be.inputs.out_seg_file = op.join(fs_subject_dir, 'mri',
                                                'aseg_csvd_bullseye' + ext)
    csvd_split_be.inputs.out_seg_lut_file = op.join(fs_subject_dir, 'mri',
                                                    'aseg_csvd_bullseye_LUT.txt')
    csvd_workflow.connect(bullseye_workflow, 'bullseye_wmparc.out_file',
                          csvd_split_be, 'parcellation_file')
    csvd_workflow.connect(dump_bullseye_lut, 'bullseye_lut_file',
                          csvd_split_be, 'parcellation_lut_file')

    csvd_split_shells = pe.Node(SplitCSVDSeg(), name='split_shells')
    csvd_split_shells.inputs.out_seg_file = op.join(fs_subject_dir, 'mri',
                                                    'aseg_csvd_shells' + ext)
    csvd_split_shells.inputs.out_seg_lut_file = \
        op.join(fs_subject_dir, 'mri', 'aseg_csvd_shells_LUT.txt')
    csvd_workflow.connect(bullseye_workflow, 'shells_wmparc.out_file',
                          csvd_split_shells, 'parcellation_file')
    csvd_workflow.connect(dump_bullseye_lut, 'shells_lut_file',
                          csvd_split_shells, 'parcellation_lut_file')
    
    # TODO aseg_CSVD inter arterial territories
    # TODO aseg_CSVD inter arterial bullseye

    if 0:
        # Intersection of bullseye and aseg
        # Produce a LUT <seg_id>_<BE_parc_id>
    
        # TODO seg stats 
        seg_stats = pe.Node(fs.SegStats(), name='seg_stats')
        seg_stats.inputs.subject = ...
        seg_stats.inputs.args = ('--seed 1234 --seg mri/aseg_csvd_bulleseye.mgz '
                                 '--pv mri/norm.mgz '
                                 '--empty --brainmask mri/brainmask.mgz '
                                 '--brain-vol-from-seg --excludeid 0 --excl-ctxgmwm '
                                 '--supratent --subcortgray --in mri/norm.mgz '
                                 '--in-intensity-name norm --in-intensity-units MR '
                                 '--etiv --surf-wm-vol --surf-ctx-vol --totalgray '
                                 '--euler --ctab ../nipic_fs_LUT.txt '
                                 '--subject 00006_T0 ')
        csvd_workflow.connect(merge_csvd_aseg, 'csvd_aseg_file',
                              seg_stats, 'segmentation_file')
    
        # TODO produce radial plots for all CSVD
        
    return csvd_workflow

class BullseyeLUTInputSpec(BaseInterfaceInputSpec):

    out_bullseye_lut_file = File(mandatory=True,
                                 desc='LUT file',
                                 hash_files=False)

    out_depth_lut_file = File(mandatory=True,
                              desc='LUT file',
                              hash_files=False)

class BullseyeLUTOutputSpec(TraitedSpec):
    bullseye_lut_file = File(desc='Bullseye LUT file')
    depth_lut_file = File(desc='Depth LUT file')

class BullseyeLUT(BaseInterface):
    input_spec = BullseyeLUTInputSpec
    output_spec = BullseyeLUTOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.out_bullseye_lut_file, 'w') as fout:
            fout.write(lut_to_str(bullseye_lut))
        with open(self.inputs.out_depth_lut_file, 'w') as fout:
            fout.write(lut_to_str(bullseye_depth_lut))

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bullseye_lut_file'] = self.inputs.out_bullseye_lut_file
        outputs['depth_lut_file'] = self.inputs.out_depth_lut_file
        return outputs


def filter_samseg_wmh(aseg_fn, raw_flair_fn, samseg_wmh_ppm_fn,
                      out_WM_no_GM_mask_fn, out_WMH_mask_fn):    
    # load WM mask
    aseg_img = nib.load(aseg_fn)
    aseg = aseg_img.get_fdata().astype(int)
    logger.debug('Loaded segmentation of size %s, '\
                 'intensity range [%d - %d], dtype %s',
                 aseg.shape, aseg.min(), aseg.max(), aseg.dtype)
    bg_mask = np.where(aseg==0)
    gm = np.bitwise_or(aseg==3, aseg==42)
    # gm_mask = np.where(gm)

    vtc = np.any((aseg==4, aseg==43, aseg==14, aseg==15),
                 axis=0)
    # vtc_mask = np.where(vtc)

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
    # wm_mask = np.where(wm)

    flair_raw_img = nib.load(raw_flair_fn)
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
    save_img_with_new_dtype(wm_no_subcortical_gm, aseg_img, out_WM_no_GM_mask_fn)
    logger.info('Segmentation of white matter withou subcortical structures saved to %s',
                out_WM_no_GM_mask_fn)

    samseg_wmh_ppm_img = nib.load(samseg_wmh_ppm_fn)
    samseg_wmh_ppm = samseg_wmh_ppm_img.get_fdata()
    wmh = np.zeros_like(aseg)
    wmh[np.where(samseg_wmh_ppm >= WMH_SAMSEG_PPM_THRESHOLD)] = WMH

    flair_raw_wm = flair_raw[wm_no_subcortical_gm]
    flair_wmh_thresh = np.median(flair_raw_wm) + np.std(flair_raw_wm)
    false_positive = np.where(np.bitwise_and(wmh==WMH,
                                             flair_raw <= flair_wmh_thresh))
    wmh[false_positive] = WMH_SAMSEG_FP

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

    save_img_with_new_dtype(wmh, aseg_img, out_WMH_mask_fn)
    return out_WMH_mask_fn

class MergeCSVDSegInputSpec(BaseInterfaceInputSpec):
    fs_aseg_file = File(exists=True, mandatory=True,
                        desc='Freesurfer aseg procuded by recon-all')
    wmh_file = File(exists=True, mandatory=True, desc='WMH segmentation')

    out_merged_csvd_aseg_file = File(mandatory=True,
                                     desc='Merged segmentation file',
                                     hash_files=False)

    out_merged_csvd_aseg_lut_file = File(mandatory=True,
                                         desc='LUT file for merged segmentation',
                                         hash_files=False)

class MergeCSVDSegOutputSpec(TraitedSpec):
    merged_csvd_aseg_file = File(desc='Merged aseg file')
    merged_csvd_aseg_lut_file = File(desc='Merged aseg LUT file')

class MergeCSVDSeg(BaseInterface):
    input_spec = MergeCSVDSegInputSpec
    output_spec = MergeCSVDSegOutputSpec

    def _run_interface(self, runtime):
        merge_csvd_aseg(
            self.inputs.fs_aseg_file,
            self.inputs.wmh_file,
            self.inputs.out_merged_csvd_aseg_file,
            self.inputs.out_merged_csvd_aseg_lut_file
        )
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['merged_csvd_aseg_file'] = self.inputs.out_merged_csvd_aseg_file
        outputs['merged_csvd_aseg_lut_file'] = \
            self.inputs.out_merged_csvd_aseg_lut_file
        return outputs

def merge_csvd_aseg(aseg_fn, wmh_fn, parcellation_fn, parcellation_lut_fn,
                    out_aseg_fn, out_aseg_lut_fn):
    aseg_img = nib.load(aseg_fn)
    aseg = aseg_img.get_fdata().astype(int)

    wmh_img = nib.load(wmh_fn)
    wmh = wmh_img.get_fdata().astype(int)
    wmh[np.where(wmh==WMH_SAMSEG_FP)] = 0

    wmh_mask = np.where(wmh != 0)
    
    aseg[wmh_mask] = wmh[wmh_mask]

    save_img_with_new_dtype(aseg, aseg_img, out_aseg_fn)
    with open(out_aseg_lut_fn, 'w') as fout:
        fout.write(lut_to_str(seg_lut))

    aseg_lut = load_lut(aseg_only=True)
    aseg_lut.update(fs_csvd_lut)

    save_img_with_new_dtype(aseg, aseg_img, out_aseg_fn)
    with open(out_aseg_lut_fn, 'w') as fout:
        fout.write(lut_to_str(aseg_lut))

class SplitCSVDSegInputSpec(BaseInterfaceInputSpec):
    seg_file = File(mandatory=True,
                    desc='Segmentation file in which to split CSVD clusters')

    parcellation_file = File(mandatory=True,
                             desc='Parcellation file to split CSVD clusters')

    parcellation_lut_file = File(mandatory=True,
                                 desc='LUT file for parcellation')

    out_seg_file = File(mandatory=True, desc='Output seg file',
                        hash_files=False)

    out_seg_lut_file = File(mandatory=True, desc='Output seg LUT file',
                            hash_files=False)

    
class SplitCSVDSegOutputSpec(TraitedSpec):
    split_seg_file = File(desc='Splitte seg file')
    split_seg_lut_file = File(desc='Splitted seg LUT file')

class SplitCSVDSeg(BaseInterface):
    input_spec = SplitCSVDSegInputSpec
    output_spec = SplitCSVDSegOutputSpec

    def _run_interface(self, runtime):
        split_csvd_seg(
            self.inputs.csvd_seg_file,
            self.inputs.csvd_seg_lut_file,
            self.inputs.parcellation_file,
            self.inputs.parcellation_lut_file,
            self.inputs.out_split_csvd_seg_file,
            self.inputs.out_split_csvd_seg_lut_file
        )
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['split_seg_file'] = self.inputs.out_seg_file
        outputs['split_seg_lut_file'] = self.inputs.out_seg_lut_file
        return outputs

def split_csvd_seg(csvd_seg_fn, csvd_seg_lut_fn, parcellation_fn,
                   parcellation_lut_fn, out_seg_fn, out_seg_lut_fn):
    parcellation_img = nib.load(parcellation_fn)
    parcellation = parcellation_img.get_fdata().astype(int)
    parcellation_lut = read_lut_file(parcellation_lut_fn)

    nb_parcels = len(parcellation_lut)
    lightness_ratios = [0.5 * (1 + i/(nb_parcels-1)) for i in range(nb_parcels)]

    seg_img = nib.load(csvd_seg_fn)
    seg = seg_img.get_fdata().astype(int)
    seg_lut = read_lut(csvd_seg_lut_fn)

    cluster_index = max(idx for idx in seg_lut) + 1
    for csvd_index, csvd_def in fs_csvd_lut.items():
        csvd_mask = (aseg == csvd_index)
        for (parcel_index, parcel_def), l_ratio in zip(parcellation_lut.items(),
                                                       lightness_ratios):
            m = np.where(np.bitwise_and(csvd_mask,
                                        parcellation==parcel_index))
            seg[m] = cluster_index
            seg_lut[cluster_index] = {
                'name' : csvd_def['name'] + '_' + parcel_def['name'],
                'color' : change_color_lightness(csvd_def['color'], l_ratio)
            }
            cluster_index += 1
        seg_lut.pop(csvd_def['name'], None)

    save_img_with_new_dtype(seg, seg_img, out_seg_fn)

    with open(out_seg_lut_fn, 'w') as fout:
        fout.write(lut_to_str(seg_lut))

class SAMSEGFilterWMHInputSpec(BaseInterfaceInputSpec):
    fs_aseg_file = File(exists=True, mandatory=True,
                        desc='Freesurfer aseg procuded by recon-all')
    fs_FLAIR_raw_file = File(exists=True, mandatory=True,
                             desc='raw FLAIR image used as input of SAMSEG')
    samseg_WMh_posterior_file = File(exists=True, mandatory=True,
                                     desc=('Posterior probability map of WMhypo'
                                           'produced by SAMSEG'))

    out_WM_no_GM_mask_file = File(mandatory=True,
                                  desc='WM Mask with no subcortical GM',
                                  hash_files=False)
    out_filtered_WMH_file = File(mandatory=True, desc='Filtered WM Hyperintensities',
                                 hash_files=False)

class SAMSEGFilterWMHOutputSpec(TraitedSpec):
    WM_no_GM_mask_file = File(desc='WM Mask with no subcortical GM')
    filtered_WMH_file = File(desc='WMH mask file')

class SAMSEGFilterWMH(BaseInterface):
    input_spec = SAMSEGFilterWMHInputSpec
    output_spec = SAMSEGFilterWMHOutputSpec

    def _run_interface(self, runtime):
        filter_samseg_wmh(
            self.inputs.fs_aseg_file,
            self.inputs.fs_FLAIR_raw_file,
            self.inputs.samseg_WMh_posterior_file,
            self.inputs.out_WM_no_GM_mask_file,
            self.inputs.out_filtered_WMH_file
        )
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['WM_no_GM_mask_file'] = self.inputs.out_WM_no_GM_mask_file
        outputs['filtered_WMH_file'] = self.inputs.out_filtered_WMH_file
        return outputs
