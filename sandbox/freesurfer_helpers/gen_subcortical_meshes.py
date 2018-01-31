#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Produce surfaces of subcortical structures (hippocampus, thalamus, cerrebelum, ...) as well as annotation files and mapped data (volume ...)

Usage::

    gen_subcortical_meshes SUBJECT_LIST [--merge --labels <LABEL_IDS> 
                       --subject_dir=<path_to_freesurfer_subject_dir>]
"""
import sys
import os
import os.path as op
import tempfile

from subprocess import call

import logging
from optparse import OptionParser

import numpy as np

import nibabel

DEFAULT_LABELS = '7,8,10,11,12,13,16,17,18,24,26,28,31,43,44,46,47,49,'\
                 '50,51,52,53,54,58,60,63,251,252,253,254,255'

fs_home = os.getenv('FREESURFER_HOME')
fs_lut_fn = op.join(fs_home, 'FreeSurferColorLUT.txt')

USAGE = 'usage: %%prog [options] SUBJECT_LIST'
DESCRIPTION = 'Generate subcortical meshes from freesurfer volumic segmentation. '\
              'SUBJECT_LIST is comma-separated list of freesurfer subject ids ' \
              '(no space).'

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('[FS subcortical extraction]')

MIN_ARGS = 1
MAX_ARGS = 1

def main():
    parser = OptionParser(usage=USAGE, description=DESCRIPTION)

    parser.add_option('-s', '--subject_dir', metavar='PATH',
                      type='str', default=os.getenv('SUBJECTS_DIR'),
                      help='Freesurfer subject dir, default is ' \
                           'env. variable SUBJECTS_DIR')

    parser.add_option('-l', '--labels', metavar='LIST_OF_INT',
                      type='str', default=DEFAULT_LABELS,
                      help='Comma-separated list of labels (int) of subcortical '\
                      'structures to extract (no space). ' 
                      'See http://freesurfer.net/fswiki/SubcorticalSegmentation ' \
                      'and %s' % fs_lut_fn)

    parser.add_option('-x', '--split', action='store_true', default=False,
                      help='Do not merge all meshes into one')
    
    parser.add_option('-f', '--force', action='store_true', default=False,
                      help='Force recomputation even if mesh already exists')

    parser.add_option('-v', '--verbose', dest='verbose', metavar='VERBOSELEVEL',
                  type='int', default=0, help='Verbose level')

    (options, args) = parser.parse_args()

    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < MIN_ARGS or (MAX_ARGS >= 0 and nba > MAX_ARGS):
        parser.print_help()
        sys.exit(1)

    set_fs_subject_dir(options.subject_dir)
        
    subjects = args[0].split(',')
    labels = get_labels_from_opt(options.labels,
                                 authorized=[int(l)
                                             for l in DEFAULT_LABELS.split(',')])
    logger.info('Using structure labels: %s', ','.join(str(l) for l in labels))

    tmp_dir = tempfile.mkdtemp(prefix='nipic_')
    
    
    for subject in subjects:
        tmp_files = []
        all_mesh_fns = []
            
        subject_dir = op.join(options.subject_dir, subject)
        logger.info('Processing subject %s...' % subject)
        if not op.exists(subject_dir):
            raise Exception('Subject %s not found in %s. ' %
                            (subject, options.subject_dir))

        merged_mesh_fn = op.join(subject_dir, 'surf', 'aseg.gii') #TODO add suffix

        if options.force or \
           (not options.split and
            not op.exists(add_suffix(merged_mesh_fn, '.surf'))): 
            for label in labels:
                logger.info('Processing label %d of subject %s...',
                            label, subject)
                slabel = '%03d' % label
    
                if options.split:
                    target_lab_mesh_fn = op.join(subject_dir, 'surf',
                                                 'aseg_%s.gii' % slabel)
                else: #temporary mesh file
                    target_lab_mesh_fn = op.join(tmp_dir, 'aseg_%s.gii'
                                                 % slabel)
                    tmp_files.append(target_lab_mesh_fn)
                    
                if options.force or not op.exists(target_lab_mesh_fn):
                    logger.info('Pre-tessellating segmentation for label %d ' \
                                'of subject %s...', label, subject)
                    tmp_aseg_filled_fn = op.join(tmp_dir,
                                                 'aseg_%s_filled.mgz' % slabel)
                    tmp_files.append(tmp_aseg_filled_fn)
                    fs_cmd = [op.join(fs_home, 'bin', 'mri_pretess'),
                              op.join(subject_dir, 'mri', 'aseg.mgz'),
                              slabel,
                              op.join(subject_dir, 'mri', 'norm.mgz'),
                              tmp_aseg_filled_fn]
                    logger.debug('Freesurfer command: %s', ' '.join(fs_cmd))
                    assert call(fs_cmd)==0
                    
                    logger.info('Tessellating segmentation for label %d ' \
                                'of subject %s...', label, subject)
                    tmp_aseg_filled_tess_fn = op.join(tmp_dir,
                                                      'aseg_%s_filled_tess' \
                                                      % slabel)
                    tmp_files.append(tmp_aseg_filled_tess_fn)
                    fs_cmd = [op.join(fs_home, 'bin', 'mri_tessellate'),
                              tmp_aseg_filled_fn, slabel,
                              tmp_aseg_filled_tess_fn]
                    logger.debug('Freesurfer command: %s', ' '.join(fs_cmd))
                    assert call(fs_cmd)==0
    
                    logger.info('Smoothing tessellation for label %d of ' \
                                'subject %s...', label, subject)
                    tmp_aseg_filled_smoothed_tess_fn = \
                        op.join(tmp_dir,'aseg_%s_filled_tess_smoothed' % slabel)
                    tmp_files.append(tmp_aseg_filled_smoothed_tess_fn)
                    fs_cmd = [op.join(fs_home, 'bin', 'mris_smooth'),
                              tmp_aseg_filled_tess_fn,
                              tmp_aseg_filled_smoothed_tess_fn]
                    logger.debug('Freesurfer command: %s', ' '.join(fs_cmd))
                    assert call(fs_cmd)==0
                    
                    logger.info('Convert tessellation to gifti for label %d ' \
                                'of subject %s...', label, subject)
                    
                    fs_cmd = [op.join(fs_home, 'bin', 'mris_convert'),
                              tmp_aseg_filled_smoothed_tess_fn,
                              target_lab_mesh_fn]
                    logger.debug('Freesurfer command: %s', ' '.join(fs_cmd))
                    assert call(fs_cmd)==0
    
                    all_mesh_fns.append(target_lab_mesh_fn)
            if not options.split:
                # merge surface gifti files
                logger.info('Merging meshes for all labels of ' \
                            'subject %s to %s...', subject, merged_mesh_fn)
    
                label_mask = merge_surfaces(all_mesh_fns, merged_mesh_fn,
                                            labels)
        else:
            logger.info('Skipping subject %s (Mesh of subcortical ' \
                        'structures already exists: %s).',
                        subject, add_suffix(merged_mesh_fn, '.surf'))

        [os.remove(fn) for fn in tmp_files]

def merge_surfaces(mesh_fns, merged_mesh_fn, labels=None):
    
    if labels is None:
        labels = range(len(mesh_fns))
        
    meshes = [nibabel.load(mesh_fn) for mesh_fn in mesh_fns]

    all_coords = np.concatenate(tuple(m.darrays[0].data for m in meshes))
    all_coords = nibabel.gifti.GiftiDataArray(data=all_coords,
                                              intent=meshes[0].darrays[0].intent,
                                              datatype=meshes[0].darrays[0].datatype,
                                              encoding=meshes[0].darrays[0].encoding,
                                              coordsys=meshes[0].darrays[0].coordsys)
    shifts = np.concatenate(([0], np.cumsum([len(m.darrays[0].data)
                                             for m in meshes])))
    all_faces = np.concatenate(tuple(m.darrays[1].data + s
                                     for m,s in zip(meshes, shifts)))
    all_faces = nibabel.gifti.GiftiDataArray(data=all_faces,
                                              intent=meshes[0].darrays[1].intent,
                                              datatype=meshes[0].darrays[1].datatype,
                                              encoding=meshes[0].darrays[1].encoding)
    merged_mesh = nibabel.gifti.GiftiImage(meshes[0].header, meshes[0].extra,
                                           darrays=[all_coords, all_faces])
    merged_mesh.to_filename(add_suffix(merged_mesh_fn, '.surf'))

    label_mask = np.concatenate(tuple(np.zeros(len(m.darrays[0].data), dtype=int) + l 
                                      for l,m in zip(labels,meshes)))
    label_mask = nibabel.gifti.GiftiDataArray(data=label_mask,
                                              intent='NIFTI_INTENT_LABEL',
                                              datatype='NIFTI_TYPE_INT32')

    labels_mask_img = nibabel.gifti.GiftiImage(darrays=[label_mask])
    labels_mask_img.to_filename(add_suffix(merged_mesh_fn, '.label'))
        
    return label_mask

def get_labels_from_opt(labels_opt, authorized=None):
    labels = [int(l) for l in labels_opt.split(',')]
    slabels = set(labels)
    if authorized is not None and not slabels.issubset(authorized):
        raise Exception('Invalid labels: %r' % slabels.diff(authorized))
        
    return labels

def set_fs_subject_dir(path):
    if not op.exists(path):
        raise Exception('Freesurfer subject dir does exist (%s). ' \
                        'Consider option --subject_dir or define env ' \
                        'variable SUBJECTS_DIR '% path)
    logger.info('Freesurfer SUBJECT_DIR is %s', path)
    os.environ['SUBJECT_DIR'] = path

def add_suffix(fn, suffix):
    """ Add a suffix before file extension (gz safe).

    >>> add_suffix('./my_file.txt', '_my_suffix')
    './my_file_my_suffix.txt'
    """
    if suffix is None:
        return fn
    sfn = op.splitext(fn)
    if sfn[1] == '.gz':
        sfn = op.splitext(fn[:-3])
        sfn = (sfn[0], sfn[1] + '.gz')
    return sfn[0] + suffix + sfn[1]    
    
if __name__ == '__main__':
    main()
