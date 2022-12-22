import os
import os.path as op
import shutil
import logging
import sys

from pydicom import dcmread

import logging
logging.basicConfig()
logger = logging.getLogger('lesca proc')
logger.setLevel(logging.INFO)

def read_dcm_header(fn, required_fields, defer_size='1 KB'):
    logger.debug('Read header from %s', fn)
    dcm = dcmread(fn, stop_before_pixels=True, defer_size=defer_size)
    #from IPython import embed; embed()
    h = {}
    for a in required_fields:
        v = dcm.__getattr__(a)
        if a == 'InstanceNumber':
            v = '%05d' % v
        if a == 'SeriesNumber':
            v = '%03d' % v
        h[a] = v
    return h

def insure_folder_exists(fn):
    folder = op.dirname(fn)
    if not op.exists(folder):
        os.makedirs(folder)
    return fn

def fix_full_head_coil(fn):
    fixed_fn = fn.replace('_coil-Head_32', 'AllCoils')
    if fixed_fn != fn:
        print(fn, '->', fixed_fn)
        os.rename(fn, fixed_fn)
    return fixed_fn

root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/'
for subject_sfolder in os.listdir(root):
    subject_folder = op.join(root, subject_sfolder)
    if op.isdir(subject_folder):
        for visit in os.listdir(subject_folder):
            anat_folder = op.join(subject_folder, visit, 'anat')
            if op.exists(anat_folder):
                for bfn in os.listdir(anat_folder):
                    fn = op.join(anat_folder, bfn)
                    if 'SWI' in fn and 'part-phase' in fn and 'run-1' not in fn:
                        phase_fn = fix_full_head_coil(fn)
                        phase_cfn = op.splitext(phase_fn)[0]
                        toks = phase_cfn.split('_')
                        fixed_phase_cfn = '_'.join(toks[:-2] + ['run-1', toks[-2], toks[-1]])
                        for ext in ('.nii', '.json'):
                            src = phase_cfn + ext
                            dest = fixed_phase_cfn + ext
                            print(src, '->', dest)
                            os.rename(src, dest)
                        mag_fn = fix_full_head_coil(fn.replace('_part-phase', ''))
                        mag_cfn = op.splitext(mag_fn)[0]
                        toks = mag_cfn.split('_')
                        fixed_mag_cfn = '_'.join(toks[:-1] + ['run-1', toks[-1]])
                        for ext in ('.nii', '.json'):
                            src = mag_cfn + ext
                            dest = fixed_mag_cfn + ext
                            print(src, '->', dest)
                            os.rename(src, dest)
                        print()       
                         
