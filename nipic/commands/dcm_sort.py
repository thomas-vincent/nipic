#!/usr/bin/env python3
import os
import os.path as op
import sys
import shutil
import re

import logging
from optparse import OptionParser

from tqdm import tqdm

from pydicom import dcmread

logger = logging.getLogger('lesca proc')
console_handler = logging.StreamHandler(stream=sys.stdout)
fmt = '%(name)s | %(asctime)s %(levelname)-8s  %(message)s'
dfmt = '%Y-%m-%d %H:%M:%S'
console_handler.setFormatter(logging.Formatter(fmt=fmt,
                                               datefmt=dfmt))
logger.addHandler(console_handler)

def read_dcm_header(fn, required_fields, defer_size='1 KB', allow_missing_field=False):
    logger.debug('Read header from %s', fn)
    dcm = dcmread(fn, stop_before_pixels=True, defer_size=defer_size)
    h = {}
    for a in required_fields:
        try:
            v = dcm.__getattr__(a)
        except AttributeError:

            if allow_missing_field:
                logger.debug('Field %s not found in header of %s', a, fn)
                continue
            else:
                logger.error('Field %s not found in header of %s', a, fn)
                raise
        if a == 'InstanceNumber':
            v = '%05d' % v
        if a == 'SeriesNumber':
            v = '%03d' % v
        h[a] = v

    return h


def main():
    min_args = 2
    max_args = 2

    usage = 'usage: %prog [options] INPUT_DICOM_DIR OUPUT_DICOM_DIR'
    description = 'Copy and sort Dicom files (subject_id/study_date/series_description/data.dcm).'

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-v', '--verbos', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=20,
                      help='Amount of verbose: '\
                           '0 (NOTSET: quiet), '\
                           '50 (CRITICAL), ' \
                           '40 (ERROR), ' \
                           '30 (WARNING), '\
                           '20 (INFO, default), '\
                           '10 (DEBUG)')

    parser.add_option('-c', '--copy-dir', dest='copy_dir',
                      metavar='PATH',
                      type='str',
                      help='Where to copy source files after importation')


    parser.add_option('-m', '--move-dir', dest='move_dir',
                      metavar='PATH',
                      type='str',
                      help='Where to move source files after importation')

    parser.add_option('-r', '--recurse', action='store_true',
                      help='Recurse into source subfolders')

    parser.add_option('-p', '--patient-id-regexp', default='[\s\S]*',
                      help='Regexp for patient ID to filter series')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)


    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    src_dir, dest_dir = args
    if not op.exists(src_dir):
        logger.error('input directory does not exist %s', src_dir)
        raise FileNotFoundError(src_dir)

    if not op.exists(dest_dir):
        os.makedirs(dest_dir)

    logger.info('Parse input directory %s', src_dir)
    src_rfns = []
    for path, subdirs, bfns in os.walk(src_dir):
        for bfn in bfns:
            src_rfns.append(op.relpath(op.join(path, bfn), src_dir))
        if not options.recurse:
            break

    patient_id_re = re.compile(options.patient_id_regexp)

    imported_rfns = []
    logger.info('Parsing %d files from %s. Try to export to %s', len(src_rfns), src_dir, dest_dir)
    for src_rfn in tqdm(src_rfns):
        src_fn = op.join(src_dir, src_rfn)

        try:
            dh = read_dcm_header(src_fn, required_fields=['SeriesInstanceUID', 'StudyDate',
                                                          'SeriesDescription', 'PatientID'])
        except:
            logger.error('Could not read DCM file %s', src_fn)
            continue

        if patient_id_re.match(dh['PatientID']) is None:
            logger.debug('Ignore participant %s with wrong format', dh['PatientID'])
            continue

        series_subdir = dh['SeriesDescription'] + '_' + dh['SeriesInstanceUID']
        if len(series_subdir) > 255:
            raise IOError('Name of series subdir too long %s', series_subdir)
        fn_dest_dir = op.join(dest_dir, dh['PatientID'], dh['StudyDate'], series_subdir)
        if not op.exists(fn_dest_dir):
            os.makedirs(fn_dest_dir)

        dest_fn = op.join(fn_dest_dir, op.basename(src_fn))
        if not op.exists(dest_fn):
            shutil.copy2(src_fn, dest_fn)
        else:
            logger.debug('Skipped existing destination %s -> %s', src_fn, dest_fn)

        imported_rfns.append(src_rfn)

    if options.copy_dir is not None:

        logger.info('Copy %d files from %s to %s', len(imported_rfns), src_dir, options.copy_dir)
        for src_rfn in tqdm(imported_rfns):
            dest_fn = op.join(options.copy_dir, src_rfn)
            dest_dir = op.dirname(dest_fn)
            if not op.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy2(op.join(src_dir, src_rfn), dest_fn)

    if options.move_dir is not None:

        logger.info('Move %d files from %s to %s', len(imported_rfns), src_dir, options.move_dir)
        for src_rfn in tqdm(imported_rfns):
            dest_fn = op.join(options.move_dir, src_rfn)
            dest_dir = op.dirname(dest_fn)
            if not op.exists(dest_dir):
               os.makedirs(dest_dir)
            shutil.move(op.join(src_dir, src_rfn), dest_fn)

if __name__ == '__main__':
    main()
