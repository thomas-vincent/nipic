import os
import os.path as op
from bids import BIDSLayout
import json
import sys
import re
import subprocess
import shutil
import numpy as np
import nibabel as nb

import logging

logger = logging.getLogger('nipic')
console_handler = logging.StreamHandler(stream=sys.stdout)
fmt = '%(name)s | %(asctime)s %(levelname)-8s  %(message)s'
dfmt = '%Y-%m-%d %H:%M:%S'
console_handler.setFormatter(logging.Formatter(fmt=fmt,
                                               datefmt=dfmt))
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

def insure_folder_exists(fn):
    folder = op.dirname(fn)
    if not op.exists(folder):
        os.makedirs(folder)
    return fn


def split_by_echo_time(fn):
    folder, bfn = op.dirname(fn), op.basename(fn)
    cfn = op.splitext(bfn)[0]
    with open(op.splitext(fn)[0] + '.json') as fin:
        multi_echo_json = json.load(fin)

    img = nb.load(fn)
    img_shape = img.header.get_data_shape()
    logger.debug('Unstack %d echo volumes from %s', img.shape[3], fn)

    sub, ses, acq, run, part, suff = cfn.split('_')
    for iecho, vol in enumerate(nb.four_to_three(img)):
        echo_cfn = '_'.join([sub, ses, acq, run, part, 
                             'echo-%d' % (iecho+1), suff])
        nb.save(vol, op.join(folder, echo_cfn + '.nii'))
        
        echo_json = multi_echo_json.copy()
        echo_json.pop('EchoTimes_ms')
        echo_json['EchoTime'] = multi_echo_json['EchoTimes_ms'][iecho]
        echo_json_fn = op.join(folder, echo_cfn + '.json')
        with open(echo_json_fn, 'w') as fout:
            json.dump(echo_json, fout)

    os.remove(fn)

def fix_nans(fn):
    img = nb.load(fn)    
    data = img.get_fdata()
    new_img = nb.Nifti2Image(np.nan_to_num(data), 
                             img.affine, header=img.header)
    nb.save(new_img, fn)


def safe_nii_average(fns):
    
    imgs = [nb.load(fn) for fn in fns]
    

ACQ_APPEND_RE = re.compile('(?P<pre>.*_acq-[a-zA-Z0-9]+)(?P<post>_.*)')
UNCOMBINED_SWI_ACQ_TAG = 'Axialt2SWI3daxialOPT1'
ECHO_REPL_RE = re.compile('(?P<pre>.*)_echo-\d+(?P<post>_.*)')

mri_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/'
output_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/derivatives'

mri_db = BIDSLayout(mri_dir)

subjects = ['%05d' % i for i in range(12,15)]
for subject in subjects:
    uncombined_swi_phase_fns = \
         [fn for fn in mri_db.get(subject=subject,
                                  acquisition=UNCOMBINED_SWI_ACQ_TAG,
                                  part='phase', return_type='filename')
                                  if not fn.endswith('.json')]
    uncombined_swi_mag_fns = \
         [fn for fn in mri_db.get(subject=subject,
                                  acquisition=UNCOMBINED_SWI_ACQ_TAG,
                                  part='mag', return_type='filename')
                                  if not fn.endswith('.json')]
    logger.info('subject %s: %d results', subject, len(uncombined_swi_phase_fns))
    if len(uncombined_swi_phase_fns) == 0:
        continue
    if len(uncombined_swi_phase_fns) == 1:
        uncombined_swi_phase_fn = uncombined_swi_phase_fns[0]
        uncombined_swi_mag_fn = uncombined_swi_mag_fns[0]
        json_fn = op.splitext(uncombined_swi_phase_fn)[0] + '.json'
        with open(json_fn) as fin:
            swi_json = json.load(fin)
        echo_times_str =  '[%s]' % ','.join('%1.2f' % e 
                                            for e in swi_json['EchoTimes_ms'])
        dum_fn = insure_folder_exists(op.join(output_dir, 
                                              op.relpath(uncombined_swi_mag_fn, 
                                                         mri_dir)))
        anat_out_dir = op.dirname(dum_fn)
        cmd =  ['romeo', '-p', uncombined_swi_phase_fn, 
                '-m', uncombined_swi_mag_fn, '-B', '-t', echo_times_str, 
                '-o', anat_out_dir, '-v']
        cmd_str = ' '.join(cmd)  
        logger.debug('Run: %s', ' '.join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        cmd_output = result.stdout.decode('utf-8')
        if result.returncode != 0:                
            logger.debug('romeo output:\n%s', cmd_output)
            logger.error('Command failed: %s', cmd_str)
        else:
            anat_orig_dir = op.dirname(uncombined_swi_phase_fn)
            uncombined_swi_phase_bfn = op.basename(uncombined_swi_phase_fn)
            uncombined_swi_phase_cfn = op.splitext(uncombined_swi_phase_bfn)[0]
            sub, ses, acq, run, _, suff = uncombined_swi_phase_cfn.split('_')
            for part in ('mag', 'phase'):
                orig_cfn = '_'.join([sub, ses, acq, run, 'part-' + part, suff])
                fixed_cfn = '_'.join([sub, ses, acq + 'CombinedRomeo', 
                                      run, 'part-' + part, suff])
                src = op.join(anat_out_dir, 'combined_%s.nii' % part)
                fix_nans(src)
                dest = op.join(anat_out_dir, fixed_cfn + '.nii')
                logger.debug('Fix romeo output filename: %s -> %s',
                             src, dest)
                os.rename(src, dest)
            
                shutil.copy(op.join(anat_orig_dir, orig_cfn + '.json'),
                            op.join(anat_out_dir, fixed_cfn + '.json'))
                conbined_fns[part] = split_by_echo_time(dest)
            
            fixed_cfn = '_'.join([sub, ses, acq + 'CombinedRomeo', 
                                  run, 'mask'])
            src = op.join(anat_out_dir, 'mask.nii')
            combined_mask_fn = op.join(anat_out_dir, fixed_cfn + '.nii')
            dest = combined_mask_fn
            logger.debug('Fix romeo output filename: %s -> %s',
                         src, dest)
            os.rename(src, dest)
            
            qsm_fns = []
            tgv_suffix = '_qsm_recon'
            for echo_phase_fn, echo_time_ms in combined_fns['phase']:
                cmd = ['tgv_qsm', '-p', echo_phase_fn, '-m', combined_mask_fn,
                       '-t', '%1.5f' % (echo_time_ms/1000), '-i', '1000',
                       '-e', 5, '-o', tgv_suffix, '--no-resampling']
                qsm_fns.append(op.splitext(echo_phase_fn)[0] + tgv_suffix +
                               '_000.nii.gz')
                  
            qsm_avg_cfn = '_'.join([sub, ses, acq + 'QSMEchoAvg', run, 
                                    'part-phase', 'Chimap'])  
            
            safe_nii_average(qsm_fns, op.join(anat_out_dir, qsm_avg_cfn))
  
sys.exit(0)

import nipype.interfaces.io as nio
import nipype.pipeline as nppl
import nipype.interfaces.utility as nut
import nipype.interfaces.base as nifbase 
# romeo -p 008_sub-ACR-0001-00006_T0_Axial_t2_SWI_3d_axial_OPT1_MEGRE_ph.nii -m 005_sub-ACR-0001-00006_T0_Axial_t2_SWI_3d_axial_OPT1_MEGRE.nii -B -t "[6.92,13.45,19.98,26.50]" -v
#cmd =  ['romeo', '-p', phase_data_fn, '-m', mag_data_fn, '-B', '-t', echo_times_str, '-v']

class RomeoInputSpec(nifbase.TraitedSpec):
    phase = nifbase.File(desc="phase", exists=True,
                 mandatory=True, argstr='-p %s')
    mag = nifbase.File(desc="magnitude", exists=True,
                 mandatory=True, argstr='-m %s')
    echo_times_ms = nifbase.traits.ListFloat(desc='Echo times [msec]', 
                                             mandatory=True, argstr="-t [%s]")

class RomeoOutputSpec(nifbase.TraitedSpec):
    B0 = nifbase.File('B0.nii', usedefault=True)
    unwrapped_phase = nifbase.File('unwrapped.nii', usedefault=True)

class Romeo(nifbase.Interface):
    input_spec = RomeoInputSpec
    output_spec = RomeoOutputSpec
    _cmd = "romeo -B --no-rescale -v"

    def _run_interface(self, runtime):
        save_multi_echo(self.inputs.phase, "multi-echo-phase.nii")
        save_multi_echo(self.inputs.mag, "multi-echo-mag.nii")
        super(RomeoInterface, self)._run_interface(runtime)


# run_2_qsm.py /home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/ /home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/derivatives/qsmxt --nextqsm_unwrapping_algorithm romeob0







query = {
    'swi-uncombined-phase' : {
        'acquisition' : UNCOMBINED_SWI_ACQ_TAG,
        'part' : 'phase'
    },
    'swi-uncombined-mag' : {
        'acquisition' : UNCOMBINED_SWI_ACQ_TAG,
        'part' : 'mag'
    }
}
bids_grab = nppl.Node(
    interface=nio.BIDSDataGrabber(base_dir=mri_dir, subject='00006',
                                  output_query=query),
    name='grab_SWI')
                                         
                                         
romeo = nppl.Node(interface=Romeo(), name='unwrap_romeo')
wf = Workflow(name="preproc")
wf.connect([
   (bids_grab, romeo, [("swi-uncombined-phase", "phase"),
                       ("swi-uncombined-mag", "magnitude")]),
   (echo_times, romeo, [('EchoTimesMs')])])
wf.run()
