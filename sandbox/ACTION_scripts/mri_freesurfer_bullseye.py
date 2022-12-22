import sys
import os
import os.path as op
from glob import glob
import subprocess
from collections import defaultdict

from bids import BIDSLayout

import logging
logging.basicConfig()
logger = logging.getLogger('lesca proc')
logger.setLevel(logging.INFO)


fs_subject_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/derivatives/freesurfer'
be_workdir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/derivatives/bullseye/workdir'
be_output_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/derivatives/bullseye'

subjects = [s for s in os.listdir(fs_subject_dir) if s != 'fsaverage']

cmd = ['run_bullseye_pipeline', '-b', '-p', '8', '-s', fs_subject_dir, 
       '--subjects'] + subjects + ['-w', be_workdir, '-o', be_output_dir]
logger.info('Run: %s', ' '.join(cmd))
subprocess.run(cmd)

