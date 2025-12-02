import sys
import os.path as op
import logging
from optparse import OptionParser
import subprocess
import shutil
from nipic.freesurfer import Freesurfer

logger = logging.getLogger('nipic')

def main():

    min_args = 1
    max_args = 1

    usage = 'usage: %prog [options] SUBJECT_LABEL'
    description = ('Apply fixed FLAIR coregistration. Expect it to be saved as mri/transforms/FLAIRraw.manual.lta')

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=logging.INFO, help='Verbose level')

    (options, args) = parser.parse_args()

    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        sys.exit(1)

    fs = Freesurfer()
    subject_fs_id = args[0]

    manual_lta = fs.transform_fn(subject_fs_id, 'FLAIRraw.manual.lta')
    main_lta = fs.transform_fn(subject_fs_id, 'FLAIRraw.lta')
    logger.info('Copy %s to %s', manual_lta, main_lta)
    shutil.copyfile(manual_lta, main_lta)

    logger.info('Apply coregistration to FLAIR')
    cmd = ['mri_convert', '-odt', 'float', '-at',
           main_lta, '-rl',
           fs.mri_fn(subject_fs_id, 'orig.mgz'),
           fs.mri_orig_fn(subject_fs_id, 'FLAIRraw.mgz'),
           fs.mri_fn(subject_fs_id, 'FLAIR.prenorm.mgz')]
    logger.debug('Command: %s', cmd)
    subprocess.run(cmd, check=True)

    logger.info('Normalize FLAIR')
    cmd = ['mri_normalize', '-seed', '1234', '-sigma', '0.5',
           '-nonmax_suppress', '0', '-min_dist', '1', '-aseg',
           fs.mri_fn(subject_fs_id, 'aseg.presurf.mgz'),
           '-surface', fs.surface_fn(subject_fs_id, 'lh.white'), 'identity.nofile',
           '-surface', fs.surface_fn(subject_fs_id, 'rh.white'), 'identity.nofile',
           fs.mri_fn(subject_fs_id, 'FLAIR.prenorm.mgz'),
           fs.mri_fn(subject_fs_id, 'FLAIR.norm.mgz')]
    logger.debug('Command: %s', cmd)
    subprocess.run(cmd, check=True)

    logger.info('Apply finalsurfs mask to FLAIR')
    cmd = ['mri_mask', '-transfer', '255', '-keep_mask_deletion_edits',
           fs.mri_fn(subject_fs_id, 'FLAIR.norm.mgz'),
           fs.mri_fn(subject_fs_id, 'brain.finalsurfs.mgz'),
           fs.mri_fn(subject_fs_id, 'FLAIR.mgz')]
    logger.debug('Command: %s', cmd)
    subprocess.run(cmd, check=True)

    logger.info('Manual FLAIR coregistration applied. Run autorecon again using:')
    logger.info('recon-all -autorecon3  -parallel -FLAIRpial -3T -s %s', subject_fs_id)
