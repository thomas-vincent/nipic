from subprocess import run

from nipype.interfaces.base import isdefined
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
from nipype.interfaces.io import FreeSurferSource
from nipype.interfaces.freesurfer import MRICoreg, ApplyVolTransform

import nipic.freesurfer as fs

logging.basicConfig()
logger = logging.getLogger('nipic')


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
    if run(['freesurfer']).returncode != 0:
        url = 'https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall'
        raise Exception(f'Freesurfer not found. See {url}')

    if run(['samseg', '--help']).returncode != 0:
        raise Exception('SAMSEG not available. Maybe Freesurfer is too old?')


def create_csvd_workflow(subjects_dir=None):
    check_dependencies()

    fs_subjects_dir = (subjects_dir if subjects_dir is not None
                       else fs.get_subjects_dir())
    logger.info("Freesurfer's SUBJECTS_DIR: %s", fs_subjects_dir)

    workflow = ...
    input_node = ...

    # Coregister raw FLAIR on original T1 

    freesurfer = FreesurferOutput(subjects_dir=subjects_dir)
    workflow.connect(input_node, 'subject_id', freesurfer, 'subject_id')
    
    coreg = MRICoreg()
    #mri_coreg --mov FLAIRraw.mgz --ref orig.mgz --reg FLAIR_to_T1.lta

    def _coreg_input(subjects_dir, subject_id):
        subject_mri_dir = op.join(subjects_dir, subject_id, 'mri')
        orig_fn = glob(op.join(subject_mri_dir, 'orig.*'))[0]
        flair_fn = glob(op.join(subject_mri_dir, 'orig', 'FLAIRraw.*'))[0]
        return [orig_fn, flair_fn]

    coreg_input = pe.Node(Function(input_names=['subjects_dir', 'subject_id'],
                                   output_names=['FLAIRraw', 'T1_orig'],
                                   function=_coreg_input),
                          name='coreg_FLAIRraw_to_T1')
    workflow.connect(coreg_input, 'FLAIRraw', coreg, 'source_file')
    workflow.connect(coreg_input, 'T1_orig', coreg, 'reference_file')
    
    #mri_vol2vol --mov  FLAIRraw.mgz  --reg FLAIR_to_T1.lta  --o FLAIR_reg.mgz --targ orig.mgz
    apply_coreg = ApplyVolTransform()

    def _apply_coreg_output(subjects_dir, subject_id):
        ext = op.splitext(glob(op.join(subject_mri_dir, 'orig.*'))[0])[-1]
        return op.join(subjects_dir, subject_id, 'mri', 'FLAIRraw_to_orig' + ext)

    apply_coreg_output = pe.Node(Function(input_names=['subjects_dir', 'subject_id'],
                                          output_names=['FLAIRraw_reg_to_orig'],
                                          function=_apply_coreg_output),
                                 name='apply_coreg_FLAIRraw_to_T1')

    workflow.connect(coreg_input, 'FLAIRraw', apply_coreg, 'source_file')
    workflow.connect(coreg_input, 'T1_orig', apply_coreg, 'target_file')
    workflow.connect(coreg, 'out_reg_file', apply_coreg, 'reg_file')
    workflow.connect(apply_coreg_ouput, 'FLAIRraw_reg_to_orig', apply_coreg, 'transformed_file')


    # Run SAMSEG on FLAIR and T1 

    # run_samseg --input 001.mgz FLAIR_reg.mgz --pallidum-separate --save-posteriors --threads 8 --output ../samseg/

    class SAMSEGInputSpec(CommandLineInputSpec):
        FLAIR_file = File(desc="FLAIR file", exists=True, mandatory=True, argstr="--input %s")
        T1_file = File(desc="T1 file", exists=True, mandatory=True, argstr="--input %s")

        output_dir = traits.Directory(desc='Directory for SAMSEG outputs, usually in mri/samseg',
                                      mandatory=True, argstr='--output %s') 

        pallidum_separate = traits.Bool(mandatory=False, argstr='--pallidum-separate',
                                        desc='Extract pallidum', default=True)

        save_PPM = traits.Bool(mandatory=False, argstr='--save-posteriors',
                               desc='Save PPM files for all tissues',
                               default=True)

        nb_threads = traits.Int(desc='Number of threads for parallelization',
                                mandatory=False, default=1)

    class SAMSEGOutputSpec(TraitedSpec):
        WMH_PPM_file = File(desc="PPM File", exists=True)

    class GZipTask(CommandLine):
        input_spec = SAMSEGInputSpec
        output_spec = SAMSEGOutputSpec
        _cmd = 'samseg'

        def _list_outputs(self):
            outputs = self.output_spec().get()
            
            outputs['WMH_PPM_file'] = glob(op.join(self.inputs.output_dir,
                                                   'posteriors',
                                                   'WM-hypointensities.*'))[0]
            return outputs

    return workflow
