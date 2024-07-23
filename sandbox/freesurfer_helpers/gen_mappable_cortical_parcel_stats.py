#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import os.path as op
from io import StringIO
import tempfile
import shutil
import re

from subprocess import call

import logging
from optparse import OptionParser

import numpy as np

import nibabel
from nibabel import gifti

fs_home = os.getenv('FREESURFER_HOME')

USAGE = 'usage: %%prog [options] SUBJECT_LIST'
DESCRIPTION = 'Generate parcelwise mappable morphometrics from freesurfer '\
              'results of recon-all, for all available parcellations ' \
              '(Desikan-Killiany, DKT and Destrieux). ' \
              'SUBJECT_LIST is comma-separated list of freesurfer ' \
              'subject ids (no space).'
MIN_ARGS = 1
MAX_ARGS = 1

PARCELLATION_TAGS = {'destrieux' : '.a2009s',
                     'dk' : '', 'dkt' : '.DKTatlas'}

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger('[FS cortical parcel stats extraction]')

def main():
    parser = OptionParser(usage=USAGE, description=DESCRIPTION)

    parser.add_option('-s', '--subject_dir', metavar='PATH',
                      type='str', default=os.getenv('SUBJECTS_DIR'),
                      help='Freesurfer subject dir, default is ' \
                           'env. variable SUBJECTS_DIR')

    parser.add_option('-p', '--parcellations', type='str',
                      metavar='LIST OF STR',
                      default='dk,dkt,destrieux',
                      help='Parcellation(s) for which to '
                      'produce mappable metrics, as a comma-separated list ' \
                      '(no space). Choices: dk, dkt, destrieux. '\
                      'See https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation.')

    parser.add_option('-m', '--metrics', type='str',
                      metavar='LIST OF STR', default=None,
                      help='List of metrics to map, as a comma-separated '\
                      'list (no space). Eg.: NumVert SurfArea GrayVol '\
                      'ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd.')

    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=0, help='Verbose level')

    (options, args) = parser.parse_args()

    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < MIN_ARGS or (MAX_ARGS >= 0 and nba > MAX_ARGS):
        parser.print_help()
        sys.exit(1)

    set_fs_subject_dir(options.subject_dir)

    subjects = args[0].split(',')
    parcellation_tags = get_parcellations_from_opt(options.parcellations)

    metrics = None
    if options.metrics is not None:
        metrics = options.metrics.split(',')
    # tmp_dir = tempfile.mkdtemp(prefix='nipic_')

    for subject in subjects:
        subject_dir = op.join(options.subject_dir, subject)
        logger.info('Processing subject %s...' % subject)

        for parcellation in parcellation_tags:
            for hemi in ['l','r']:
                annotation_fn = op.join(subject_dir, 'label',
                                        '%sh.aparc%s.annot' %
                                        (hemi, parcellation))
                logger.info('Reading annotation file %s ...' % annotation_fn)
                annotation = nibabel.freesurfer.read_annot(annotation_fn)
                label_mask = annotation[0]
                parcel_ids = dict( (an, i)
                                   for i,an in  enumerate(annotation[2]) )

                stats_fn = op.join(subject_dir, 'stats',
                                   '%sh.aparc%s.stats'%
                                   (hemi, parcellation))
                logger.info('Reading stats file %s ...' % stats_fn)
                stats = read_stats(stats_fn)

                stat_names = [n for n in stats.dtype.fields.keys()
                              if n != 'StructName']
                if metrics is not None:
                    stat_names_set = set(stat_names)
                    assert all([m in stat_names_set for m in metrics])
                else:
                    metrics = stat_names

                mapped_stats = dict( (m, np.zeros(label_mask.shape))
                                     for m in metrics )
                for iparcel in range(stats.shape[0]):
                    parcel_id = parcel_ids[stats['StructName'][iparcel]]
                    parcel_mask = np.where(label_mask==parcel_id)
                    for metric in metrics:
                        mapped_stats[metric][parcel_mask] = \
                                                    stats[metric][iparcel]

                for metric in metrics:
                    mapped_stats_fn = op.join(subject_dir, 'surf',
                                              '%sh.aparc%s.%s'%
                                              (hemi, parcellation, metric))
                    logger.info('Writing mapped stats file %s ...' %
                                mapped_stats_fn)
                    nibabel.freesurfer.write_morph_data(mapped_stats_fn,
                                                        mapped_stats[metric])

def get_parcellations_from_opt(parcellations_opt):
    parcellations = parcellations_opt.split(',')
    assert set(parcellations).issubset(PARCELLATION_TAGS.keys())
    return [PARCELLATION_TAGS[p] for p in parcellations]

def read_stats(stats_fn):
    # Remove comment lines, fix column header and standardize delimiter
    with open(stats_fn) as fstat:
        content = re.sub('(?m)^#(?! *ColHeaders *).*\n?', '', fstat.read())
        content = re.sub('# *ColHeaders *', '', content)
        content = re.sub(' +', ' ', content)

    stats = np.genfromtxt(StringIO(unicode(content)), delimiter=' ',
                          names=True, dtype=None)
    return stats

def set_fs_subject_dir(path):
    if not op.exists(path):
        raise Exception('Freesurfer subject dir does exist (%s). ' \
                        'Consider option --subject_dir or define env ' \
                        'variable SUBJECTS_DIR '% path)
    logger.info('Freesurfer SUBJECT_DIR is %s', path)
    os.environ['SUBJECT_DIR'] = path


if __name__ == '__main__':
    main()
