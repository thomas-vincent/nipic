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

mri_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/'
fs_subject_dir = '/home/lesca/DataServer/freesurfer_data/'

# recon-all -all -i /home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/ACR-0001-00006_ACR-0001-00006/IRM_RECHERCHE_CLAUDINE_GAUTHIER_20190830_094238_195000/SAG_MPRAGE_0002/ACR-0001-00006.MR.IRM_RECHERCHE_CLAUDINE_GAUTHIER.0002.0002.2019.08.30.11.07.35.128417.124131428.IMA -FLAIR /home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/ACR-0001-00006_ACR-0001-00006/IRM_RECHERCHE_CLAUDINE_GAUTHIER_20190830_094238_195000/SAG_FLAIR_3D_T2_SPACE_0004/ACR-0001-00006.MR.IRM_RECHERCHE_CLAUDINE_GAUTHIER.0004.0002.2019.08.30.11.07.35.128417.124137576.IMA -s ACR-0001-00006_T0 -parallel -FLAIRpial

FS_RECON_CMD = ['recon-all', '-all', '-parallel', '-FLAIRpial']

def main():
    mri_db = BIDSLayout(mri_dir)
    subjects = mri_db.get_subjects()
    for subject in subjects:
        time_points = defaultdict(dict)
        
        for suffix, acq in (('T1w', 'SagMPRAGE'),
                            ('T2w', 'Axialt2tsetra512'),
                            ('FLAIR', 'SagFlair3dt2space')):
            sel = mri_db.get(subject=subject, suffix=suffix,
                             acquisition=acq, extension='nii')
            for bidsf in sel:
                time_point = bidsf.get_entities()['session'][-2:]
                logger.info('%s %s %s: %s', suffix, acq, time_point, bidsf.path)
                time_points[time_point][suffix] = bidsf.path        
        
        for time_point, imgs in time_points.items():
            logger.info('Processing %s / %s', subject, time_point)
            abort = False
            if 'T1w' not in imgs:
                logger.error('Missing T1w')
                abort = True
            if 'FLAIR' not in imgs:
                logger.error('Missing FLAIR')
                abort = True
            if abort:
                continue
            cmd = FS_RECON_CMD + ['-i', imgs['T1w'],
                                  '-FLAIR', imgs['FLAIR'],
                                  '-s', subject + '_' + time_point]
            logger.info('Run: %s', ' '.join(cmd))
            subprocess.run(cmd)

def check_fn_exists(fn):
    if fn is None or not op.exists(fn):
        logger.error('File not found: %s', fn)
        return False
    return True

if __name__ == '__main__':
    main()
