import os
import os.path as op
import shutil
import logging
import sys
import subprocess

import logging
logging.basicConfig()
logger = logging.getLogger('lesca proc')
logger.setLevel(logging.INFO)

acq_root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/ACR-0001-00006/ACTIONCARDIORISK_T0_20190830'
output_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_proc/dcm_header'
if not op.exists(output_dir):
    os.makedirs(output_dir)

for acq_subdir in os.listdir(acq_root):
    acq_dir = op.join(acq_root, acq_subdir)
    if 'SWI' in acq_dir:
        for dcm_fn in list(sorted(os.listdir(acq_dir)))[:400]:
            dcm_info = subprocess.check_output(['dcmdump', op.join(acq_dir, dcm_fn)])
            dcm_info_fn = op.join(output_dir, op.splitext(dcm_fn)[0] + '.txt')
            with open(dcm_info_fn, 'w') as fout:
                fout.write(dcm_info.decode('utf-8'))
