# -*- coding: utf-8 -*-
import sys
import os.path as op
import logging
from optparse import OptionParser
import unittest
import tempfile
import shutil
import os

import re

logger = logging.getLogger('nipic')
console_handler = logging.StreamHandler(stream=sys.stdout)
fmt = '%(name)s | %(asctime)s %(levelname)-8s  %(message)s'
dfmt = '%Y-%m-%d %H:%M:%S'
console_handler.setFormatter(logging.Formatter(fmt=fmt,
                                               datefmt=dfmt))
logger.addHandler(console_handler)

import subprocess

from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.errors import InvalidDicomError

def main():
    min_args = 2
    max_args = 2

    usage = 'usage: %prog [options] DICOM_FOLDERS OUTPUT_FOLDER'
    description = 'Convert Dicom files to nifti files organised in BIDS folders.'

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

    parser.add_option('-t', '--time-point',
                      help='Time point tag to add in destination file names')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    output_bids_path = args[-1]
    input_dcm_pathes = args[:-1]
    dcm_export_to_nii(input_dcm_pathes, output_bids_path, time_point=options.time_point)

            
def list_leaf_dirs(root, all_dirs=None):
    all_dirs = [] if all_dirs is None else all_dirs
    for path, subdirs, fns in os.walk(root):
        if len(subdirs) == 0:
            all_dirs.append(path)
    return all_dirs

class ChildOutputFolderError(Exception): pass

from tqdm import tqdm
from pathlib import Path
import traceback

from string import Formatter

BIDS_PATTERN = op.join('{PatientID}', '{StudyID}{_TimePoint}_{StudyDate}',
                       '{SeriesNumber}_{SeriesDescription}',
                       '{PatientID}{_TimePoint}_{SeriesDescription}_{InstanceNumber}_{ContentTime}.dcm')


DCMNIIX_BASE_PATTERN = 'sub-{patient_ID}/ses-{study_ID}{_TimePoint}_{StudyDate}/{bids_type}/{SeriesNumber}_sub-{patient_ID}{_TimePoint}_{description}'

DCMNIIX_MAIN_PATTERN = DCMNIIX_BASE_PATTERN + '_{bids_suffix}.nii'

DCMNIIX_ME_PATTERN = DCMNIIX_BASE_PATTERN + '_echo-{echo_number}_{bids_suffix}.nii' 
DCMNIIX_ME_MC_PATTERN = DCMNIIX_BASE_PATTERN + '_echo-{echo_number}_coil-{coil}_{bids_suffix}.nii' 

export_rules = {
    '.*t2_gre_3d_field_map.*': { 
        'bids_type' : 'fmap',
        'dcm2niix_pattern' : DCMNIIX_ME_PATTERN
    },
    '.*field_map.*' : { 
        'bids_type' : 'fmap',
        'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
        },
    '(.*SWI.*OPT1.*)' : {
        'bids_type' : 'anat',
        'dcm2niix_pattern' : DCMNIIX_ME_MC_PATTERN
    },
    '(.*MPRAGE.*)|(.*FLAIR.*)|(.*SWI.*)|(.*_T2_.*)|(.*_vessels_.*)|(.*_tof_.*)|(.*_pd_.*)|(.*MT.*)' : {
        'bids_type' : 'anat',
        'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    },
    '(.*BOLD.*)' : {
        'bids_type' : 'func',
        'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    },
    '(.*DIFF.*)' : {
        'bids_type' : 'dwi',
        'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    },   
    '(.*pcasl.*)' : {
        'bids_type' : 'perf',
        'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    },
}
export_rules = {re.compile(regexp, re.IGNORECASE):v for regexp, v in export_rules.items()}

bids_suffix_rules = {
    '(.*SWI.*OPT1.*)' : 'MEGRE',
    '(.*SWI.*)' : 'SWI',
    '(.*FLAIR.*)' : 'FLAIR',
    '(.*_T2_.*)' : 'T2w',
    '(.*MPRAGE.*)' : 'T1w',
    '(.*BOLD.*)' : 'T2starw',
    '(.*_diff_.*)' : 'DWI',
    '.*T1w.*' : 'T1w',
    '.*_pd_.*' : 'PDw',
    '.*MT.*' : 'MTR',
    '(.*vessels.*)|(.*tof.*)' : 'angio'
}
bids_suffix_rules = {re.compile(regexp, re.IGNORECASE):v 
                     for regexp, v in bids_suffix_rules.items()}

dcm2niix_tags = {
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


DCM2NIIX_OUTPUT_FN_RE = re.compile('Convert \d+ DICOM as (.*?) \(')


def dcm_export_to_nii(dcm_pathes, output_folder, time_point=None):

    logger.info('List directories...')
    dcm_dirs = []
    for dcm_path in dcm_pathes:
        list_leaf_dirs(dcm_path, all_dirs=dcm_dirs)

    file_operations = dict()
    logger.info('Resolve file operations...')
    for dcm_dir in tqdm(dcm_dirs):
        if dcm_dir.endswith('DICOMDIR'):
            logger.info('Ignore %s', dcm_dir)
            continue

        dcm2niix_file_pattern = make_dcm2niix_file_pattern(dcm_dir, time_point=time_point)
        if dcm2niix_file_pattern is not None:
            cmd = ['dcm2niix', '-f', dcm2niix_file_pattern, '-o', output_folder, dcm_dir]
            cmd_str = ' '.join(cmd)
            logger.debug('Running: %s', cmd_str)
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            cmd_output = result.stdout.decode('utf-8')
            if result.returncode != 0:
                logger.error('Command failed: %s', cmd_str)
            else:
                for output_fn in DCM2NIIX_OUTPUT_FN_RE.findall(cmd_output):
                     output_fn = fix_single_coil_fn(output_fn)
                     
                        if ('coil' in dcm2niix_file_pattern and 
                            'echo' in dcm2niix_file_pattern):
                    '.*echo-(?P<echo_number>%d+)'
                    print('todo merge_nii(tmp_dir)') 
                    from IPython import embed; embed()
                    sys.exit(1) 
def make_dcm2niix_file_pattern(dcm_acq_path, time_point=None):
    dcm_fn = op.join(dcm_acq_path, os.listdir(dcm_acq_path)[0])
    tags = read_dcm_header(dcm_fn, required_fields=['StudyDate', 'SeriesDescription', 'SeriesNumber'])
    tags.update(dcm2niix_tags)
    tags['_TimePoint'] = '_%s' % time_point if time_point is not None else ''

    dcm2niix_pat, bids_type = None, None
    for fn_re, rule in export_rules.items():
        if fn_re.match(tags['SeriesDescription']):
            bids_type = rule['bids_type']
            dcm2niix_pat = rule['dcm2niix_pattern']
            break
    if bids_type is None:
        logger.error('No export rule matching %s (dcm_path: %s)', tags['SeriesDescription'], dcm_acq_path)
        return None
    tags['bids_type'] = bids_type

    bids_suffix = None
    for fn_re, suffix in bids_suffix_rules.items():
        if fn_re.match(tags['SeriesDescription']):
            bids_suffix = suffix
            break
    if bids_suffix is None:
        logger.error('No BIDS suffix defined for %s (dcm_path: %s)', tags['SeriesDescription'], dcm_acq_path)
        return None 
    tags['bids_suffix'] = bids_suffix

    logger.debug('Tags for dcm2niix output filename: %s', tags)
    dcm2niix_output_fn = dcm2niix_pat.format(**tags)
    logger.debug('dcm2niix ouput filename: %s', dcm2niix_output_fn)
    return dcm2niix_output_fn

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
