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

    usage = 'usage: %prog [options] [DICOM_FOLDERS|DICOM_FILES] OUTPUT_FOLDER'
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

    parser.add_option('-t', '--time-point',
                      help='Time point tag to add in destination file names')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    output_dcm_path = args[-1]
    input_dcm_pathes = args[:-1]
    dcm_tagged_export(input_dcm_pathes, output_dcm_path, output_pattern=BIDS_PATTERN,
                      file_operation_func=shutil.copy, time_point=options.time_point)
    
def walk_files(root):
    for path, subdirs, fns in os.walk(root):
        for fn in fns:
            yield op.join(path, fn)
            
def list_files(root, all_fns=None):
    all_fns = [] if all_fns is None else all_fns
    for path, subdirs, fns in os.walk(root):
        all_fns.extend(op.join(path, fn) for fn in fns)
    return all_fns
            
def list_leaf_dirs(root, all_dirs=None):
    all_dirs = [] if all_dirs is None else all_dirs
    for path, subdirs, fns in os.walk(root):
        if len(subdirs) == 0:
            all_dirs.append(path)
    return all_dirs

class TargetFileExistsError(Exception): pass
class ChildOutputFolderError(Exception): pass

from tqdm import tqdm
from pathlib import Path
import traceback

from string import Formatter
#ACR-000-XXX/IRM_recherche.../Axial_T2_FLAIR/  par exemple.
BIDS_PATTERN = op.join('{PatientID}', '{StudyID}{_TimePoint}_{StudyDate}',
                       '{SeriesNumber}_{SeriesDescription}',
                       '{PatientID}{_TimePoint}_{SeriesDescription}_{InstanceNumber}_{ContentTime}.dcm')


DCMNIIX_BASE_PATTERN = 'sub-{patient_ID}/ses-{study_ID}{_TimePoint}_{StudyDate}/{data_type}/{series_number}_sub-{patient_ID}{_TimePoint}_{description}'

DCMNIIX_MAIN_PATTERN = DCMNIIX_BASE_PATTERN + '_{bids_suffix}.nii'

DCMNIIX_ME_MC_PATTERN = DCMNIIX_BASE_PATTERN + '_echo-{echo_number}_coil-{coil}_{bids_suffix}.nii' 

export_rules = {
    '(.*field_map.*)' : { 
        'bids_type' : 'fmap',
        'dcmniix_pattern' : DCMNIIX_MAIN_PATTERN
        },
    '(.*MPRAGE.*)|(.*FLAIR.*)|(.*SWI.*)|(.*_T2_.*)|(.*_vessels_.*)|(.*_tof_.*)|(.*_pd_.*)' : {
        'bids_type' : 'anat',
        'dcmniix_pattern' : DCMNIIX_MAIN_PATTERN
    },
    '(.*SWI.*OPT1.*)' : {
        'bids_type' : 'anat',
        'dcmniix_pattern' : DCMNIIX_ME_MC_PATTERN
    },
    '(.*BOLD.*)' : {
        'bids_type' : 'func',
        'dcmniix_pattern' : DCMNIIX_MAIN_PATTERN
    },
    '(.*DIFF.*)' : {
        'bids_type' : 'dwi',
        'dcmniix_pattern' : DCMNIIX_MAIN_PATTERN
    },   
    '(.*pcasl.*)' : {
        'bids_type' : 'perf',
        'dcmniix_pattern' : DCMNIIX_MAIN_PATTERN
    },
}

bids_suffix = {
    '(.*SWI.*OPT1.*)' : 'MEGRE',
    '(.*SWI.*)' : 'SWI',
    '(.*_T2_.*)' : 'T2w',
    '(.*FLAIR.*)' : 'FLAIR',
    '(.*MPRAGE.*)' : 'T1w',
    '(.*BOLD.*)' : 'T2starw',
    '(.*_diff_.*)' : 'DWI',
    '.*T1w.*' : 'T1w',
    '.*_pd_.*' : 'PDw',
    '.*MT.*' : 'MTR',
    '(.*vessels.*)|(.*tof.*)' : 'angio'
}

dcmniix_tags = {
    'coil' : '%a',
    'basename' : '%b',
    'comments' : '%c',
    'description' : '%d',
    'echo_number' : '%e',
    'folder_name' : '%f',
    'patient_ID' : '%i',
    'series_instance_UID' : '%j',
    'study_instance_UID' : '%k',
    'manufacturer' : '%m',
    'patient_name' : '%n',
    'protocol' : '%p',
    'instance_number' : '%r',
    'series_number' : '%s',
    'time' : '%t',
    'acquisition_number' : '%u',
    'vendor' : '%v',
    'study_ID' : '%x',
    'sequence_name' : '%z',
}

def dcm_tagged_export(dcm_pathes, output_folder, time_point=None):

    logger.info('List files...')
    dcm_dirs = []
    for dcm_path in dcm_pathes:
        list_leaf_dirs(dcm_path, dcm_dirs=dcm_dirs)

    file_operations = dict()
    logger.info('Resolve file operations...')
    for dcm_fn in tqdm(dcm_fns):
        if dcm_fn.endswith('DICOMDIR'):
            logger.info('Ignore %s', dcm_fn)
            continue
        try:
            output_fn = dcm_tagged_fn(dcm_fn, output_folder, time_point=time_point)
            if output_fn in file_operations:
                msg = ('source: %s, previous source %s, destination: %s' %
                       (dcm_fn, file_operations[output_fn], output_fn))
                raise TargetFileExistsError(msg)
            file_operations[output_fn] = dcm_fn
        except InvalidDicomError:
            logger.warning('Ignored file (invalid Dicom): %s', dcm_fn)
        except:
            logger.error('Error getting target file name for %s:\n%s', dcm_fn, 
                         traceback.format_exc())
            raise

    logger.info('Apply file operations...')
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

def dcm_tagged_fn(dcm_fn, output_folder, output_pattern=BIDS_PATTERN, time_point=None):
    dcm_fields = set(e[1] for e in Formatter().parse(output_pattern)  
                    if e[1] is not None and e[1] != '_TimePoint')
    dcm_header = read_dcm_header(dcm_fn, required_fields=dcm_fields)
    dcm_header['_TimePoint'] = '_%s' % time_point if time_point is not None else ''
    return op.join(output_folder, safe_fn(output_pattern.format_map(dcm_header)))

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

def safe_fn(fn):
    fn = ''.join(c for c in fn 
                 if c.isalpha() or c.isdigit() or c in '- _.' or c==op.sep).rstrip()
    fn = fn.replace(' ', '_').replace('_--_', '_')
    return fn

class TestDcmRenaming(unittest.TestCase):

    def setUp(self):
        self.tmp_data_dir = tempfile.mkdtemp()
        self.tmp_output_dir = tempfile.mkdtemp()
        logger.setLevel(logging.DEBUG)

        dcm_fn = get_testdata_file("MR_small.dcm")
        ds = dcmread(dcm_fn)
        self.subject_id = 'test_subject'
        self.protocol = 'test_procotol'
        self.acq_type = 'FLAIR [A]'
        self.volume_index = 2
        self.series_index = 4
        self.volume_time = '1234'
        self.acq_date = '20221014'

        ds.PatientID = self.subject_id
        ds.StudyID = self.protocol
        ds.SeriesDescription = self.acq_type
        ds.InstanceNumber = self.volume_index
        ds.SeriesNumber = self.series_index
        ds.ContentTime = self.volume_time
        ds.StudyDate = self.acq_date

        self.dcm_fn_s1 = op.join(self.tmp_data_dir, 'MR_FLAIR_s1.dcm')
        ds.save_as(self.dcm_fn_s1)
        
        self.dcm_fn_s2_flair_1 = op.join(self.tmp_data_dir, 'MR_FLAIR_s2_1.dcm')
        ds.PatientID = 'S2'
        ds.StudyID = self.protocol
        ds.SeriesDescription = self.acq_type
        ds.InstanceNumber = 1
        ds.SeriesNumber = '1'
        ds.ContentTime = '1234'
        ds.StudyDate = '20210107'
        ds.save_as(self.dcm_fn_s2_flair_1)

        self.dcm_fn_s2_flair_2 = op.join(self.tmp_data_dir, 'MR_FLAIR_s2_2.dcm')
        ds.PatientID = 'S2'
        ds.StudyID = self.protocol
        ds.SeriesDescription = self.acq_type
        ds.InstanceNumber = 2
        ds.SeriesNumber = '1'
        ds.ContentTime = '1235'
        ds.StudyDate = '20210107'
        ds.save_as(self.dcm_fn_s2_flair_2)

        self.dcm_fn_s2_flair_2bis = op.join(self.tmp_data_dir,
                                            'MR_FLAIR_s	2_2b.dcm')
        ds.save_as(self.dcm_fn_s2_flair_2bis)

    def tearDown(self):
        shutil.rmtree(self.tmp_data_dir)
        shutil.rmtree(self.tmp_output_dir)
 
    def test_tagged_fn(self):
        expected_fn = op.join(self.tmp_data_dir, self.subject_id, 
                              '%s_%s' % (self.protocol, self.acq_date),
                              '%03d_%s' % (self.series_index, 'FLAIR_A'),
                              '%s_%s_%05d_%s.dcm' % (self.subject_id,
                                                'FLAIR_A',
                                                self.volume_index,
                                                self.volume_time))
        output_fn = dcm_tagged_fn(self.dcm_fn_s1, self.tmp_data_dir)
        self.assertEqual(output_fn, expected_fn)

    def test_tagged_fn_with_time_point(self):
        expected_fn = op.join(self.tmp_data_dir, self.subject_id, 
                              '%s_T0_%s' % (self.protocol, self.acq_date),
                              '%03d_%s' % (self.series_index, 'FLAIR_A'),
                              '%s_T0_%s_%05d_%s.dcm' % (self.subject_id,
                                                     'FLAIR_A',
                                                     self.volume_index,
                                                     self.volume_time))
        output_fn = dcm_tagged_fn(self.dcm_fn_s1, self.tmp_data_dir, time_point='T0')
        self.assertEqual(output_fn, expected_fn)

    def test_export_copy_single_file(self):
        dcm_tagged_export(self.dcm_fn_s1, self.tmp_output_dir)
        expected_fn = op.join(self.tmp_output_dir, self.subject_id,
                              '%s_%s' % (self.protocol, self.acq_date),
                              '%03d_%s' % (self.series_index, 'FLAIR_A'),
                              '%s_%s_%05d_%s.dcm' % (self.subject_id,
                                                     'FLAIR_A',
                                                     self.volume_index,
                                                     self.volume_time))
        self.assertTrue(op.exists(expected_fn))
        self.assertTrue(op.exists(self.dcm_fn_s1))

    def test_export_move_single_file(self):
        dcm_tagged_export(self.dcm_fn_s1, self.tmp_output_dir,
                          file_operation_func=shutil.move)
        expected_fn = op.join(self.tmp_output_dir, self.subject_id,
                              '%s_%s' % (self.protocol, self.acq_date),
                              '%03d_%s' % (self.series_index, 'FLAIR_A'),
                              '%s_%s_%05d_%s.dcm' % (self.subject_id,
                                                'FLAIR_A',
                                                self.volume_index,
                                                self.volume_time))
        self.assertTrue(op.exists(expected_fn))
        self.assertFalse(op.exists(self.dcm_fn_s1))
        

    def test_duplicate_target(self):
        self.assertRaises(TargetFileExistsError, dcm_tagged_export,
                          self.tmp_data_dir, self.tmp_output_dir)

    def test_export_folder(self):
        raise NotImplementedError()

    def test_nested_dest_folder(self):
        dest_dir = op.join(self.tmp_data_dir, 'export')
        os.makedirs(dest_dir)
        self.assertRaises(ChildOutputFolderError, dcm_tagged_export,
                          self.tmp_data_dir, dest_dir)


    def test_export_from_archive(self):
        raise NotImplementedError()

    def test_export_from_archive_no_move(self):
        raise NotImplementedError()
