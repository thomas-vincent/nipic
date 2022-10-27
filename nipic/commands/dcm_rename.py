# -*- coding: utf-8 -*-
import sys
import os.path as op
import logging
from optparse import OptionParser
import unittest
import tempfile
import shutil
import os

logger = logging.getLogger('nipic')
console_handler = logging.StreamHandler(stream=sys.stdout)
fmt = '%(name)s | %(asctime)s %(levelname)-8s  %(message)s'
dfmt = '%Y-%m-%d %H:%M:%S'
console_handler.setFormatter(logging.Formatter(fmt=fmt,
                                               datefmt=dfmt))
logger.addHandler(console_handler)

from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.errors import InvalidDicomError

def main():
    min_args = 2
    max_args = 2

    usage = 'usage: %prog [options] OUTPUT_FOLDER [DICOM_FOLDER|DICOM_FILE]'
    description = 'Copy/move Dicom files using file names composed from header data.'

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=0,
                      help='Amount of verbose: '\
                           '0 (NOTSET: quiet, default), '\
                           '50 (CRITICAL), ' \
                           '40 (ERROR), ' \
                           '30 (WARNING), '\
                           '20 (INFO), '\
                           '10 (DEBUG)')

    parser.add_option('-m', '--move', action='store_true', default=False,
                      help='Move files instead of copying')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    output_dcm_path = args[0]
    input_dcm_path = args[1]
    dcm_tagged_export(input_dcm_path, output_dcm_path, output_pattern=BIDS_PATTERN,
                      file_operation_func=shutil.copy)
    
def walk_files(root):
    for path, subdirs, fns in os.walk(root):
        for fn in fns:
            yield op.join(path, fn)

class TargetFileExistsError(Exception): pass
class ChildOutputFolderError(Exception): pass

from tqdm import tqdm
from pathlib import Path

#ACR-000-XXX/IRM_recherche.../Axial_T2_FLAIR/  par exemple.
BIDS_PATTERN = op.join('{PatientID}', '{StudyID}',
                       '{SeriesDescription}_{SeriesNumber}',
                       '{PatientID}_{SeriesDescription}_{InstanceNumber}.dcm')

def dcm_tagged_export(dcm_path, output_folder, output_pattern=BIDS_PATTERN,
                      file_operation_func=shutil.copy):
    if op.isfile(dcm_path):
        dcm_fns = [dcm_path]
    else:
        if Path(dcm_path) in Path(output_folder).parents:
            raise ChildOutputFolderError()
        dcm_fns = walk_files(dcm_path)

    file_operations = dict()
    for dcm_fn in dcm_fns:
        try:
            output_fn = dcm_tagged_fn(dcm_fn, output_folder)
            if output_fn in file_operations:
                msg = ('source: %s, previous source %s, destination: %s' %
                       (dcm_fn, file_operations[output_fn], output_fn))
                raise TargetFileExistsError(msg)
            file_operations[output_fn] = dcm_fn
        except InvalidDicomError:
            logger.warning('Ignored file (invalid Dicom): %s', dcm_fn)

    for dest_fn, src_fn in tqdm(file_operations.items()):
        logger.debug('%s %s -> %s)', file_operation_func.__name__, src_fn,
                     dest_fn)
        if not op.exists(dest_fn):
            file_operation_func(src_fn, insure_folder_exists(dest_fn))
        else:
            logger.warning('Skip creation of (already exists): %s', dest_fn)


def insure_folder_exists(fn):
    folder = op.dirname(fn)
    if not op.exists(folder):
        os.makedirs(folder)
    return fn

def dcm_tagged_fn(dcm_fn, output_folder,
                  output_pattern=BIDS_PATTERN):
    dcm_header = read_dcm_header(dcm_fn)
    return op.join(output_folder, output_pattern.format_map(dcm_header))

def read_dcm_header(fn):
    dcm = dcmread(fn)
    h = {}
    for a in dcm.dir():
        v = dcm.__getattr__(a)
        if a == 'InstanceNumber':
            v = '%05d' % v
        if len(str(v)) < 300:
            h[a] = v
    # from IPython import embed; embed()
    return h

class TestDcmRenaming(unittest.TestCase):

    def setUp(self):
        self.tmp_data_dir = tempfile.mkdtemp()
        self.tmp_output_dir = tempfile.mkdtemp()
        logger.setLevel(logging.DEBUG)

        dcm_fn = get_testdata_file("MR_small.dcm")
        ds = dcmread(dcm_fn)
        self.subject_id = 'test_subject'
        self.protocol = 'test_procotol'
        self.acq_type = 'FLAIR'
        self.volume_index = 2
        self.series_index = '4'

        ds.PatientID = self.subject_id
        ds.StudyID = self.protocol
        ds.SeriesDescription = self.acq_type
        ds.InstanceNumber = self.volume_index
        ds.SeriesNumber = self.series_index

        self.dcm_fn_s1 = op.join(self.tmp_data_dir, 'MR_FLAIR_s1.dcm')
        ds.save_as(self.dcm_fn_s1)
        
        self.dcm_fn_s2_flair_1 = op.join(self.tmp_data_dir, 'MR_FLAIR_s2_1.dcm')
        ds.PatientID = 'S2'
        ds.StudyID = self.protocol
        ds.SeriesDescription = self.acq_type
        ds.InstanceNumber = 1
        ds.SeriesNumber = '1'
        ds.save_as(self.dcm_fn_s2_flair_1)

        self.dcm_fn_s2_flair_2 = op.join(self.tmp_data_dir, 'MR_FLAIR_s2_2.dcm')
        ds.PatientID = 'S2'
        ds.StudyID = self.protocol
        ds.SeriesDescription = self.acq_type
        ds.InstanceNumber = 2
        ds.SeriesNumber = '1'
        ds.save_as(self.dcm_fn_s2_flair_2)

        self.dcm_fn_s2_flair_2bis = op.join(self.tmp_data_dir,
                                            'MR_FLAIR_s2_2b.dcm')
        ds.save_as(self.dcm_fn_s2_flair_2bis)

    def tearDown(self):
        shutil.rmtree(self.tmp_data_dir)
        shutil.rmtree(self.tmp_output_dir)
 
    def test_tagged_fn(self):
        expected_fn = op.join(self.tmp_data_dir, self.subject_id, self.protocol,
                              '%s_%s' % (self.acq_type, self.series_index),
                              '%s_%s_%05d.dcm' % (self.subject_id,
                                                self.acq_type,
                                                self.volume_index))
        output_fn = dcm_tagged_fn(self.dcm_fn_s1, self.tmp_data_dir)
        self.assertEqual(output_fn, expected_fn)


    def test_export_copy_single_file(self):
        dcm_tagged_export(self.dcm_fn_s1, self.tmp_output_dir)
        expected_fn = op.join(self.tmp_output_dir, self.subject_id,
                              self.protocol,
                              '%s_%s' % (self.acq_type, self.series_index),
                              '%s_%s_%05d.dcm' % (self.subject_id,
                                                self.acq_type,
                                                self.volume_index))
        self.assertTrue(op.exists(expected_fn))
        self.assertTrue(op.exists(self.dcm_fn_s1))

    def test_export_move_single_file(self):
        dcm_tagged_export(self.dcm_fn_s1, self.tmp_output_dir,
                          file_operation_func=shutil.move)
        expected_fn = op.join(self.tmp_output_dir, self.subject_id,
                              self.protocol,
                              '%s_%s' % (self.acq_type, self.series_index),
                              '%s_%s_%05d.dcm' % (self.subject_id,
                                                self.acq_type,
                                                self.volume_index))
        self.assertTrue(op.exists(expected_fn))
        self.assertFalse(op.exists(self.dcm_fn_s1))
        

    def test_duplicate_target(self):
        self.assertRaises(TargetFileExistsError, dcm_tagged_export,
                          self.tmp_data_dir, self.tmp_output_dir)

    def test_nested_dest_folder(self):
        dest_dir = op.join(self.tmp_data_dir, 'export')
        os.makedirs(dest_dir)
        self.assertRaises(ChildOutputFolderError, dcm_tagged_export,
                          self.tmp_data_dir, dest_dir)

    def test_export_from_archive(self):
        raise NotImplementedError

    def test_export_from_archive_no_move(self):
        raise NotImplementedError
