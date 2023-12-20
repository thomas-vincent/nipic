# -*- coding: utf-8 -*-
import sys
import os.path as op
from pathlib import Path
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
    dcm_export_to_nii(input_dcm_pathes, output_bids_path)


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

import pandas as pd

BIDS_PATTERN = op.join('{PatientID}', '{StudyID}{TimePoint}_{StudyDate}',
                       '{SeriesNumber}_{SeriesDescription}',
                       '{PatientID}_{TimePoint}_{SeriesDescription}_{InstanceNumber}_{ContentTime}.dcm')

DCMNIIX_BASE_PATTERN = 'sub-{PatientID}/ses-{StudyDate}{TimePoint}/{bids_type}/sub-{PatientID}_ses-{StudyDate}{TimePoint}_acq-{bids_series_description}'

DCMNIIX_MAIN_PATTERN = DCMNIIX_BASE_PATTERN + '_{bids_suffix}'

DCMNIIX_ME_PATTERN = DCMNIIX_BASE_PATTERN + '_echo-{echo_number}_{bids_suffix}' 
DCMNIIX_ME_MC_PATTERN = DCMNIIX_BASE_PATTERN + '_run-1_echo-{echo_number}_coil-{coil}_{bids_suffix}' 

export_rules = {
    # '(.*t2_gre_3d_field_map.*)|(.*B1Map.*)': { 
        # 'bids_type' : 'fmap',
        # 'dcm2niix_pattern' : DCMNIIX_ME_PATTERN
    # },
    # '.*field_map.*' : { 
        # 'bids_type' : 'fmap',
        # 'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
        # },
    # '(.*SWI.*OPT1.*)' : {
        # 'bids_type' : 'anat',
        # 'dcm2niix_pattern' : DCMNIIX_ME_MC_PATTERN
    # },
    '(.*MPRAGE.*)|(.*FLAIR.*)' : {
        'bids_type' : 'anat',
        'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    },
    # '(.*MPRAGE.*)|(.*FLAIR.*)|(.*SWI.*)|(.*_T2_.*)|(.*_vessels_.*)|(.*_tof_.*)|(.*_pd_.*)|(.*MT.*)|(.*T1w.*)|(.*tof.*)' : {
        # 'bids_type' : 'anat',
        # 'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    # },
    # '(.*BOLD.*)' : {
        # 'bids_type' : 'func',
        # 'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    # },
    # '(.*DIFF.*)' : {
        # 'bids_type' : 'dwi',
        # 'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    # },   
    # '(.*pcasl.*)' : {
        # 'bids_type' : 'perf',
        # 'dcm2niix_pattern' : DCMNIIX_MAIN_PATTERN
    # },
    'Localizer_Brain.*' : None,
    'MoCoSeries' : None,
    'PhoenixZIPReport' : None,
    't2_space_dark-fluid_sag_caipi4' : None
}
export_rules = {re.compile(regexp, re.IGNORECASE):v for regexp, v in export_rules.items()}

bids_suffix_rules = {
    '(.*t2_gre_3d_field_map.*)|(.*B1Map.*)': 'fieldmap', 
    '(.*SWI.*)|(.*BOLD.*)' : 'T2starw',
    '(.*FLAIR.*)' : 'FLAIR',
    '(.*_T2_.*)' : 'T2w',
    '(.*MPRAGE.*)' : 'T1w',
    '(.*_diff_.*)' : 'DWI',
    '.*T1w.*' : 'T1w',
    '.*_pd_.*' : 'PDw',
    '.*MT.*' : 'MTR',
    '.*pcasl.*M0.*' : 'm0scan',
    '.*pcasl.*' : 'asl',
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


DCM2NIIX_OUTPUT_FN_RE = re.compile(r'Convert \d+ DICOM as (.*?) \(')
ECHO_RE = re.compile(r'.*echo-(?P<echo_number>\d+).*')
ECHO_REPL_RE = re.compile('(?P<pre>.*)_echo-\d+(?P<post>_.*)')
COIL_RE = re.compile(r'.*coil-H(?P<coil>(?:\d+)|(?:Head_\d+)|(?:[^_]+)).*')
COIL_REPL_RE = re.compile('(?P<pre>.*)_coil-H(?P<coil>(?:\d+)|(?:ead_\d+)|(?:[^_]+?))(?P<post>_.*)')

#SINGLE_COIL_REPL_RE = re.compile('(.*)_coil-HE\d+(_.*)')
HEAD_COIL_REPL_RE = re.compile('(?P<pre>.*)_coil-H(?P<coil>(?:EA;HEP)|(?:ead_\d+))(?P<post>_.*)')
ACQ_APPEND_RE = re.compile('(?P<pre>.*_acq-[a-zA-Z0-9]+)(?P<post>_.*)')
#MULTI_COIL_REPL_RE = re.compile('(.*)_coil-Head_\d+(_.*)')
#COIL_REPL_RE = re.compile('(.*)_coil-H\d+(_.*)')

import json
import nibabel as nb
from collections import defaultdict

BIDSVersion = "1.0.2"

def get_first_fn(folder, criterion=None):
    if criterion is None:
        criterion = lambda fn: True
    for bfn in os.listdir(folder):
        fn = op.join(folder, bfn)
        if criterion(fn):
            return fn
    return None


class ExportInfo:

    def __init__(self, data_fn):
        self.data_fn = data_fn
        self.reload_table()

    def already_done(self, dcm_dir):
        listing = os.listdir(dcm_dir)
        if len(listing) == 0:
            return True
        dcm_fn = op.join(dcm_dir, listing[0])
        dh = read_dcm_header(dcm_fn, 
                             required_fields=['SeriesInstanceUID', 'StudyDate', 
                                              'SeriesDescription', 'PatientID'])
        series_id = dh['SeriesInstanceUID']
        target_fns = self.export_table.get(series_id, None)
        if target_fns is not None:
            if len(target_fns) == 0:
                self.export_table.pop(series_id)
                self.save_table()
            for fn in target_fns:
                if not op.exists(fn):
                    logger.info('Export %s for series %s does not exist', fn, series_id)
                    for fn in target_fns:
                        if op.exists(fn):
                            logger.info('Remove %s (incomplete export of series %s)', fn, series_id)
                            os.remove(fn)
                        json_fn = op.splitext(fn)[0] + '.json'
                        if op.exists(json_fn):
                            os.remove(json_fn) 
                    self.export_table.pop(series_id)
                    self.save_table()
                    break
        done = series_id in self.export_table
        if done:
            logger.info(('Series already exported: {SeriesInstanceUID} '
                         '({PatientID}, {StudyDate}, {SeriesDescription})')
                        .format(**dh))
        return done

    def register_dcm_export(self, dcm_dir, exported_fns):
        dcm_fn = op.join(dcm_dir, os.listdir(dcm_dir)[0])
        dh = read_dcm_header(dcm_fn, required_fields=['SeriesInstanceUID'])
        series_id = dh['SeriesInstanceUID']
        logger.debug('Register %s export files for series %s', 
                     len(exported_fns), series_id) 
        self.export_table[series_id] = tuple(exported_fns)
        self.save_table()

    def save_table(self, export_table=None):
        export_table = (export_table if export_table is not None 
                        else self.export_table)

        with open(self.data_fn, 'w') as fout:
            logger.debug('Save export table with %s entries to %s',
                         len(export_table), self.data_fn) 
            json.dump(export_table, fout)
    
    def reload_table(self):
        if not op.exists(self.data_fn):
            self.save_table({})
        with open(self.data_fn) as fin:
            self.export_table = json.load(fin)

def check_export(dcm_dir, export_info):
    """ Insure that files listed in export info exist """
    
    return export_info[dh['SeriesInstanceUID']]


def fix_mag_tag(fn):
   toks = fn.split('_')
   phase_fn = '_'.join(toks[:-1] + ['part-phase', toks[-1]])
   if op.exists(phase_fn):
       fixed_mag_fn = '_'.join(toks[:-1] + ['part-mag', toks[-1]])
       fixed_mag_cfn = op.splitext(fixed_mag_fn)[0]
       cfn = op.splitext(fn)[0]
       for ext in ('.json', '.nii'):
           src = cfn + ext
           dest = fixed_mag_cfn + ext
           logger.debug('Fix mag tag: %s -> %s', src, dest)
           os.rename(src, dest)
       return  fixed_mag_cfn + '.nii'
   return fn




def get_series_dirs(ds, series_link_root_dir):
    series_pathes = []
    series_link_root = Path(series_link_root_dir)
    root_dir = Path(ds.filename).resolve().parent
    for patient in ds.patient_records:
        print(
            f"PATIENT: PatientID={patient.PatientID}, "
            f"PatientName={patient.PatientName}"
        )

        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            study_descr = study.StudyDescription or "(no value available)"
            print(
                f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
                f"StudyDate={study.StudyDate}, StudyDescription={study_descr}"
            )

            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]
            for series in all_series:
                series_descr = getattr(series, "SeriesDescription", "NA")

                images = [ii for ii in series.children
                          if ii.DirectoryRecordType == "IMAGE"]

                print(f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                      f"Modality={series.Modality}, SeriesDescription={series_descr} - "
                      f"{len(images)} SOP Instance(s)")
               
                if series.Modality == 'MR':
                    series_dir = series_link_root / series.SeriesInstanceUID
                    series_dir.mkdir()
                    # Get the absolute file path to each instance
                    #   Each IMAGE contains a relative file path to the root directory
                    elems = [ii["ReferencedFileID"] for ii in images]
                    # Make sure the relative file path is always a list of str
                    paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
                    paths = [Path(*p) for p in paths]
                    for p in paths:
                        (series_dir / p.name).symlink_to(Path(root_dir) / p)
                    series_pathes.append(series_dir)                    
    return series_pathes


from datetime import datetime

def get_studies_info(ds):
    studies_info = []
    dfmt = '%Y%m%d%H%M%S'
    root_dir = Path(ds.filename).resolve().parent
    for patient in ds.patient_records:
        print(
            f"PATIENT: PatientID={patient.PatientID}, "
            f"PatientName={patient.PatientName}"
        )

        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            study_descr = study.StudyDescription or "(no value available)"
            print(
                f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
                f"StudyDate={study.StudyDate}, StudyDescription={study_descr}"
            )

            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]
            ts_start = datetime.now()
            ts_end = datetime.fromtimestamp(0)
            for series in all_series:
                series_descr = getattr(
                                series, "SeriesDescription", "(no value available)"
                            )
                
                images = [
                    ii for ii in series.children
                    if ii.DirectoryRecordType == "IMAGE"
                ]

                print(
                    f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                    f"Modality={series.Modality}, SeriesDescription={series_descr} - "
                    f"{len(images)} SOP Instance(s)"
                )
                # Get the absolute file path to each instance
                #   Each IMAGE contains a relative file path to the root directory
                elems = [ii["ReferencedFileID"] for ii in images]
                # Make sure the relative file path is always a list of str
                paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
                paths = [Path(*p) for p in paths]

                # List the instance file paths
                for p in paths:
                    

                    # Optionally read the corresponding SOP Instance
                    h = dcmread(Path(root_dir) / p, stop_before_pixels=True, defer_size='1 Kb')
                    # print(f"{'  ' * 3}IMAGE: Path={os.fspath(p)} - {h.AcquisitionDateTime}")
                    try:
                        ts = datetime.strptime(h.AcquisitionDateTime.split('.')[0], dfmt)
                        ts_start = min(ts_start, ts)
                        ts_end = max(ts_end, ts)
                    except AttributeError:
                        logger.error('Could not get AcquisitionDateTime for %s', os.fspath(p))

                    
            print(
                f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
                f"StudyDate={study.StudyDate}, StudyDescription={study_descr} - "
                f"StartDate={ts_start}, EndDate={ts_end} - "
                f"Duration={ts_end - ts_start}"
            )   
            studies_info.append((patient.PatientID, study.StudyDate, study.StudyTime, {'start_time' : ts_start, 'end_time' : ts_end}))
    return studies_info



def add_series(root, series_fns):
    series_info = set()
    for path, subdirs, rfns in os.walk(root):
        logger.info('Resolve series in %s', op.join(path))
        for rfn in rfns:
            fn = op.join(path, rfn)
            try:
                h = read_dcm_header(fn, 
                                    required_fields=['StudyID', 'StudyDate',
                                                     'SeriesInstanceUID', 'PatientID'],
                                    allow_missing_field=True)
            except InvalidDicomError:
                logger.info('Could not read DCM file %s', rfn)
                continue
            if 'SeriesInstanceUID' in h:
                series_fns[h['SeriesInstanceUID']].add(fn)
                series_info.add((h['StudyDate'], h['PatientID']))
    logger.info('Resolved visits: %s', '\n'.join('%s %s' % e for e in sorted(series_info, key=lambda e: e[0])))

def group_series(series_fn, series_link_root_dir):
    series_link_root = Path(series_link_root_dir)
    series_dirs = []
    for suid, fns in series_fn.items():
        series_dir = op.join(series_link_root_dir, suid)
        if not op.exists(series_dir):
            os.makedirs(series_dir)
            series_dirs.append(series_dir)
        logger.info('Create %d DCM links in %s', len(fns), series_dir)
        for fn in fns:
            bfn = op.basename(fn)
            src_fn = os.path.abspath(fn)
            dest_fn = op.join(series_dir, bfn)
            logger.debug('Create sym link %s -> %s ', src_fn, dest_fn)
            try:
                Path(dest_fn).symlink_to(Path(src_fn))
            except:
                logger.error('Sym link failed. Series ignored')
                series_dirs.pop()
                break
    return series_dirs

def parse_series_info(series_dcm_path):
    series_dcm_path = Path(series_dcm_path)
    logger.info('Parse series info in %s', series_dcm_path)
    dcm_fns = os.listdir(series_dcm_path)
    if len(dcm_fns) == 0:
        return None
    ts_start = datetime.now()
    ts_end = datetime.fromtimestamp(0)
    date_fmt = '%Y%m%d%H%M%S'
    sizes = set()
 
    for dcm_fn in dcm_fns:
        h = read_dcm_header(series_dcm_path / dcm_fn, 
                            required_fields=['StudyID', 'StudyDate',
                                             'SeriesInstanceUID', 
                                             'SeriesDescription', 
                                             'PatientID', 'AcquisitionDate',
                                             'AcquisitionDateTime', 'RepetitionTime',
                                             'AcquisitionTime', 'Rows', 'Columns', 
                                             'NumberOfFrames'],
                            allow_missing_field=True)
        if 'AcquisitionDateTime' in h:
            adate = h['AcquisitionDateTime'].split('.')[0]
        elif 'AcquisitionDate' in h and 'AcquisitionTime' in h:
            adate = h['AcquisitionDate']+h['AcquisitionTime'].split('.')[0]
        else:
            logger.warning('Could not retrieve acquisition time from %s in series %s', dcm_fn, h['SeriesDescription'])
            adate = None
        
        if adate is not None:
            ts = datetime.strptime(adate, date_fmt)
            ts_start = min(ts_start, ts)
            ts_end = max(ts_end, ts)
        if 'Rows' in h and 'Columns' in h:
            if 'NumberOfFrames' in h:
                sizes.add(f'{h["NumberOfFrames"]}fx{h["Rows"]}x{h["Columns"]}')
            else:
                sizes.add(f'?fx{h["Rows"]}x{h["Columns"]}')
        else:
            logger.warning('Could not retrieve Rows / Columns from %s in series %s', dcm_fn, h['SeriesDescription'])

    if ts_start == ts_end and 'RepetitionTime' in h:
        ts_end = ts_start + pd.Timedelta(h['RepetitionTime'] * len(dcm_fns), unit="ms")

    return {
        **h, 
        **{
            'Timestamp_First_Image' : ts_start,
            'Timestamp_Last_Image' : ts_end,
            'Nb_Instances' : len(dcm_fns),
            'Instance_Size' : ', '.join(sizes)
        }
    }



def get_series_uid(series_dcm_path):
    series_dcm_path = Path(series_dcm_path)
    dcm_fns = os.listdir(series_dcm_path)
    if len(dcm_fns) == 0:
        return None
    h = read_dcm_header(series_dcm_path / dcm_fns[0], 
                        required_fields=['SeriesInstanceUID'])
    return h['SeriesInstanceUID']

manifest_columns = ['SeriesInstanceUID', 'PatientID', 
                    'StudyDate', 'SeriesDescription', 
                    'Timestamp_First_Image', 'Timestamp_Last_Image',
                    'Nb_Instances', 'Instance_Size']
def update_manifest(series_dirs, manifest_fn):
    if not op.exists(manifest_fn):
        manifest_df = pd.DataFrame(columns=manifest_columns)
        manifest_df.to_excel(manifest_fn)
        
    manifest_df = pd.read_excel(manifest_fn, engine='openpyxl')
    manifest_df.set_index('SeriesInstanceUID', inplace=True)
       
    series_recorded = set(manifest_df.index)
    to_concat = [manifest_df]
    for series_path in series_dirs:
        series_uid = get_series_uid(series_path)
        if series_uid is not None and series_uid not in series_recorded:
            try:
                series_info = parse_series_info(series_path)
            except InvalidDicomError:
                logger.error('Error reading DICOM in %s (skipped)', series_path)
                continue
            if series_info is not None:
                to_concat.append(pd.DataFrame.from_records([series_info])[manifest_columns]
                                .set_index('SeriesInstanceUID'))
    manifest_df = pd.concat(to_concat)
    manifest_df.reset_index().to_excel(manifest_fn, index=False)

def dcm_export_to_nii(dcm_pathes, output_folder, time_point=None, 
                      project_name=None, force=False):

    logger.info('List directories...')
    series_fns = defaultdict(set)
    info_studies = []
    series_tmp_dir = tempfile.mkdtemp()
    logger.debug('Temporary series dir: %s', series_tmp_dir)
    for dcm_path in dcm_pathes:
        add_series(dcm_path, series_fns)
    series_dirs = group_series(series_fns, series_tmp_dir)

    update_manifest(series_dirs, op.join(output_folder, 'series_info.xlsx'))

    dataset_description_fn = op.join(output_folder, 'dataset_description.json')
    if not op.exists(dataset_description_fn):
        if project_name is None:
             dcm_fn = get_first_fn(dcm_dirs[0], 
                                   lambda fn: fn.lower().endswith('dcm'))
             h = read_dcm_header(dcm_fn, required_fields=['StudyID'])
             project_name = h['StudyID']

        with open(dataset_description_fn, 'w') as fout:
                  json.dump({'Name' : project_name,
                             'BIDSVersion' : BIDSVersion}, fout)

    export_info = ExportInfo(op.join(output_folder, 'export_info.json'))
    file_operations = dict()
    logger.info('Resolve file operations...')
    for dcm_dir in tqdm(series_dirs):
        dcm2niix_file_pattern = make_dcm2niix_file_pattern(dcm_dir)
        if dcm2niix_file_pattern is not None:
            cmd = ['dcm2niix', '-v', '0',
                   '-f', dcm2niix_file_pattern, '-o',
                   output_folder, dcm_dir]
            cmd_str = ' '.join(cmd)
            logger.debug('Running: %s', cmd_str)
            result = subprocess.run(cmd, stdout=subprocess.PIPE)
            cmd_output = result.stdout.decode('utf-8')
            if result.returncode != 0:
                logger.debug('dcm2niix output:\n%s', cmd_output)
                logger.error('Command failed: %s', cmd_str)
            else:
                try:
                    dcm_output_fns = check_dcm_outputs(cmd_output)
                except Dcm2niixMissingImageError:
                    dcm_fn = op.join(dcm_dir, os.listdir(dcm_dir)[0])
                    dh = read_dcm_header(dcm_fn,
                                         required_fields=['StudyDate',
                                                          'SeriesInstanceUID',
                                                          'SeriesDescription',
                                                          'PatientID'])
                    logger.error(('Missing image(s) for {PatientID} / '
                                  '{StudyDate} / {SeriesDescription} / '
                                  '{SeriesInstanceUID}').format(**dh))
                    continue
                def merge_headers_echo_times(hs):
                    ch = hs[0].copy()
                    ch.pop('EchoTime')
                    ch['EchoTimes_ms'] = [h['EchoTime']*1000 for h in hs]
                    return ch
                try:
                    output_fns = concat_nii_series(dcm_output_fns, ECHO_RE, ECHO_REPL_RE,
                                                   'echo times',
                                                   lambda m: int(m['echo_number']),
                                                   merge_headers_echo_times)
                except Exception as e:
                    dcm_fn = op.join(dcm_dir, os.listdir(dcm_dir)[0])
                    dh = read_dcm_header(dcm_fn,
                                         required_fields=['StudyDate',
                                                          'SeriesInstanceUID',
                                                          'SeriesDescription',
                                                          'PatientID'])
                    dh['exception'] = repr(e)
                    logger.error(('Error while concatenating over echo times for {PatientID} / '
                                  '{StudyDate} / {SeriesDescription} / '
                                  '{SeriesInstanceUID}\n{exception}').format(**dh))
                    continue
                def get_coil_index(coil_match):
                    c = coil_match['coil']
                    if c.isdigit():
                        c = int(c)
                    return c
                def merge_headers_coils(hs):
                    ch = hs[0].copy()
                    ch['CoilString'] = 'Head%d' % len(hs)
                    return ch
                try:
                    output_fns = concat_nii_series(output_fns, COIL_RE, COIL_REPL_RE,
                                                   'coils', get_coil_index,
                                                   merge_headers_coils)
                except Exception as e:
                    dcm_fn = op.join(dcm_dir, os.listdir(dcm_dir)[0])
                    dh = read_dcm_header(dcm_fn, 
                                         required_fields=['StudyDate',
                                                          'SeriesInstanceUID', 
                                                          'SeriesDescription', 
                                                          'PatientID'])
                    dh['exception'] = repr(e)
                    logger.error(('Error while concatenating over echo times for {PatientID} / '
                                  '{StudyDate} / {SeriesDescription} / '
                                  '{SeriesInstanceUID}\n{exception}').format(**dh))
                    continue

                output_fns = [fix_mag_tag(fn) for fn in output_fns]
                export_info.register_dcm_export(dcm_dir, output_fns)
    shutil.rmtree(series_tmp_dir)

from string import ascii_lowercase
from heapq import heappush
from pprint import pformat

class NiiConcatenationError(Exception): pass

def concat_nii_series(fns, match_re, repl_re, info_str,
    get_vol_index_from_match, merge_headers):
    to_concat = defaultdict(list)
    out_fns = []
    for fn in fns:
        m = match_re.match(fn)
        if m is not None:
            cfn = repl_re.sub(r'\g<pre>\g<post>', fn)
            ivol = get_vol_index_from_match(m)
            heap = to_concat[cfn]
            heappush(heap, (ivol, fn))
        else:
            out_fns.append(fn)

    to_remove = []
    for cfn, heap_vols in to_concat.items():
        vol_fns = [e[1] for e in heap_vols]
        if len(vol_fns) > 1:
            logger.info('Concat %d volumes over %s in %s...',
                        len(vol_fns), info_str, cfn)
            if op.exists(cfn):
                logger.error('Target already exists %s', cfn)
            else:
                try:    
                    nb.save(nb.concat_images(vol_fns), cfn)
                except:
                    [os.remove(fn) for fn in vol_fns]
                    raise NiiConcatenationError()
                out_fns.append(cfn)
                headers = []
                for vol_fn in vol_fns:
                    to_remove.append(vol_fn)
                    json_fn = op.splitext(vol_fn)[0] + '.json'
                    with open(json_fn) as fin:
                        headers.append(json.load(fin))
                    to_remove.append(json_fn)
                
                json_cfn = op.splitext(cfn)[0] + '.json'       
                with open(json_cfn, 'w') as fout:
                    json.dump(merge_headers(headers), fout)
            
        else:
            out_fns.append(vol_fns[0])   
    [os.remove(f) for f in to_remove]
    return out_fns

def fix_echo_number(output_cfns):
    fixed_outputs = []
    processed = set()
    for output_cfn in output_cfns:
        # logger.debug('Check output %s for echo number issue', output_cfn)
        duplicate_cfns = [output_cfn]
        if '_echo-1' in output_cfn:
             for dup_suffix in ascii_lowercase:
                dup_cfn = '%s%s' % (output_cfn, dup_suffix)
                if op.exists(dup_cfn + '.json'):
                    duplicate_cfns.append(dup_cfn)
                else:
                    break
        if len(duplicate_cfns) > 1:
            echo_vols = []
            for dup_cfn in duplicate_cfns:
                with open(dup_cfn + '.json') as fin:
                    heappush(echo_vols, 
                             (float(json.load(fin)['EchoTime']), dup_cfn))
            tmp_dir = tempfile.mkdtemp()
            move_operations = []
            for iecho, (echo_time, dup_cfn) in enumerate(echo_vols):
                echo_number = (iecho+1)
                fix_cfn = ECHO_REPL_RE.sub(r'\g<pre>_echo-%d\g<post>' % echo_number, 
                                           output_cfn)       

                src_nii_fn = dup_cfn + '.nii'
                dest_nii_fn = fix_cfn + '.nii'
                logger.debug('Fixed echo number: %s -> %s', 
                             src_nii_fn, dest_nii_fn)
                src_tmp_nii_fn = op.join(tmp_dir, op.basename(src_nii_fn))
                shutil.move(src_nii_fn, src_tmp_nii_fn)
                move_operations.append((src_tmp_nii_fn, dest_nii_fn))
    
                src_json_fn = dup_cfn + '.json'
                dest_json_fn = fix_cfn + '.json'
                logger.debug('Fixed echo number: %s -> %s', 
                             src_json_fn, dest_json_fn)
                src_tmp_json_fn = op.join(tmp_dir, op.basename(src_json_fn))
                with open(src_json_fn) as fin:
                    src_json = json.load(fin)
                src_json['EchoNumber'] = echo_number
                with open(src_tmp_json_fn, 'w') as fout:
                    json.dump(src_json, fout)
                os.remove(src_json_fn)
                move_operations.append((src_tmp_json_fn, dest_json_fn))

                processed.add(dup_cfn)

            for src, dest in move_operations:
                if op.exists(dest):
                    logger.error('Target already exists while ' \
                                 'fixing echo number: %s -> %s', src, dest)
                    os.remove(src)
                else:
                    shutil.move(src, dest)
                    if dest.endswith('.nii'):
                        fixed_outputs.append(dest)
            os.rmdir(tmp_dir)   
        elif output_cfn not in processed:
            #logger.debug('No echo number issue')
            fixed_outputs.append(output_cfn + '.nii')    
    return fixed_outputs
    
def fix_stacked_coils(output_fns):
    fixed_outputs = []
    for output_fn in output_fns:
        if '_coil-' in output_fn:
            img = nb.load(output_fn)
            img_shape = img.header.get_data_shape()
            if len(img_shape)  >= 5 and img_shape[4] != 1:
                logger.warning('Unhandled >5D volume %s', output_fn)
                fixed_outputs.append(output_fn) 
                continue
            elif len(img_shape) >= 4 and img_shape[3] != 1:
                logger.debug('Unstack %d coil volumes from %s',
                             img.shape[3], output_fn)
                multicoil_json_fn = op.splitext(output_fn)[0] + '.json'
                with open(multicoil_json_fn) as fin:
                    multicoil_json = json.load(fin)

                for icoil, vol in enumerate(nb.four_to_three(img)):
                    coil_str = 'H%d' % (icoil+1)
                    coil_fn = COIL_REPL_RE.sub(r'\g<pre>_coil-%s\g<post>' % coil_str, 
                                               output_fn)
                    nb.save(vol, coil_fn)
                    
                    coil_json = multicoil_json.copy()
                    coil_json['CoilString'] = coil_str
                    coil_json_fn = op.splitext(coil_fn)[0] + '.json'
                    with open(coil_json_fn, 'w') as fout:
                        json.dump(coil_json, fout)

                    fixed_outputs.append(coil_fn)
                os.remove(output_fn)
                os.remove(multicoil_json_fn)
            else: # not more than 3 dims
                fixed_hc_fn = HEAD_COIL_REPL_RE.sub(r'\g<pre>\g<post>', output_fn)
                if fixed_hc_fn != output_fn:
                    fixed_hc_fn = ACQ_APPEND_RE.sub(r'\g<pre>AllCoils\g<post>', fixed_hc_fn)
                    logger.debug('Fix coil tag: %s -> %s', 
                                 output_fn, fixed_hc_fn)
                    shutil.move(output_fn, fixed_hc_fn)
                    json_fn = op.splitext(output_fn)[0] + '.json'
                    fixed_json_fn = op.splitext(fixed_hc_fn)[0] + '.json'
                    shutil.move(json_fn, fixed_json_fn)
                    fixed_outputs.append(fixed_hc_fn)
                else:
                    fixed_outputs.append(output_fn)
        else:
            fixed_outputs.append(output_fn)
    return fixed_outputs

class Dcm2niixMissingImageError(Exception): pass

def check_dcm_outputs(dcm2niix_output):
    output_fns = DCM2NIIX_OUTPUT_FN_RE.findall(dcm2niix_output)
    if 'Missing image' in dcm2niix_output:
        for fn in output_fns:
            os.remove(fn + '.nii' if not fn.endswith('nii') else fn )
            os.remove((op.splitext(fn)[0] if fn.endswith('nii') else fn) + '.json')
        raise Dcm2niixMissingImageError()
    logger.debug('dcm2niix output:\n%s', dcm2niix_output)
    fixed_outputs = fix_echo_number(output_fns)
    fixed_outputs = fix_phase_suffix(fixed_outputs)
    return fix_stacked_coils(fixed_outputs)

def fix_phase_suffix(output_fns):
    fixed_fns = []
    for fn in output_fns:
        cfn = op.splitext(fn)[0]
        if cfn.endswith('_ph'):
            toks = cfn.split('_')
            fixed_cfn = '_'.join(toks[:-2] + ['part-phase', toks[-2]])
            for ext in ('.nii', '.json'):
                shutil.move(cfn + ext, fixed_cfn + ext)
            fixed_fns.append(fixed_cfn + '.nii')
            logger.debug('Fix phase tag: %s -> %s', cfn, fixed_cfn)
            
        else:
            fixed_fns.append(fn)
    return fixed_fns

# romeo 008_sub-ACR-0001-00006_T0_Axial_t2_SWI_3d_axial_OPT1_MEGRE.nii -m 005_sub-ACR-0001-00006_T0_Axial_t2_SWI_3d_axial_OPT1_MEGRE.nii -B -t "[6.92,13.45,19.98,26.50]" -o outputdir

def fix_single_coil_fn(fn):
    fixed_fn = SINGLE_COIL_REPL_RE.sub(r'\g<pre>\g<post>', fn)
    if fixed_fn != fn:
        logger.debug('Remove single coil tag: %s -> %s', fn, fixed_fn)
        shutil.move(fn, fixed_fn)
        json_fn = op.splitext(fn)[0] + '.json'
        fixed_json_fn = op.splitext(fixed_fn)[0] + '.json'
        shutil.move(json_fn, fixed_json_fn)
    fixed_fn = HEAD_COIL_REPL_RE.sub(r'\g<pre>_coil-HF\g<post>', fn)
    if fixed_fn != fn:
        logger.debug('Fix head coil tag: %s -> %s', fn, fixed_fn)
        shutil.move(fn, fixed_fn)
        json_fn = op.splitext(fn)[0] + '.json'
        fixed_json_fn = op.splitext(fixed_fn)[0] + '.json'
        shutil.move(json_fn, fixed_json_fn)
    return fixed_fn

# Largest suffix matching a BIDS subject tag
BIDS_SUBJECT_TAG_RE = re.compile('^.*?(?P<pid>[a-zA-Z0-9]+)$')

def make_dcm2niix_file_pattern(dcm_acq_path):
    dcm_fn = op.join(dcm_acq_path, os.listdir(dcm_acq_path)[0])
    tags = read_dcm_header(dcm_fn, 
                          required_fields=['StudyDate', 'SeriesDescription',
                                           'SeriesNumber', 'PatientID'])

    m_pid = BIDS_SUBJECT_TAG_RE.match(tags['PatientID'])
    if m_pid is not None:
        logger.info('Fix participant ID %s as %s', 
                    tags['PatientID'], m_pid['pid'])
        tags['PatientID'] = m_pid['pid']
        
        
    tags['bids_series_description'] = (tags['SeriesDescription']
                                       .replace('_', '').replace(' ', '')
                                       .replace('-', ''))

    tags.update(dcm2niix_tags)

    dcm2niix_pat, bids_type = None, None
    for fn_re, rule in export_rules.items():
        if fn_re.match(tags['SeriesDescription']):
            if rule is not None:
                bids_type = rule['bids_type']
                dcm2niix_pat = rule['dcm2niix_pattern']
            else:
                logger.info('Skip %s (dcm_path: %s)', 
                            tags['SeriesDescription'], dcm_acq_path)
                return None
            break
    if bids_type is None:
        logger.error('No export rule matching %s (dcm_path: %s)',
                     tags['SeriesDescription'], dcm_acq_path)
        return None

    tags['bids_type'] = bids_type

    bids_suffix = None
    for fn_re, suffix in bids_suffix_rules.items():
        if fn_re.match(tags['SeriesDescription']):
            bids_suffix = suffix
            break
    if bids_suffix is None:
        logger.error('No BIDS suffix defined for %s (dcm_path: %s)',
                     tags['SeriesDescription'], dcm_acq_path)
        sys.exit(1)

    tags['bids_suffix'] = bids_suffix

    logger.debug('Tags for dcm2niix output filename: %s', pformat(tags))
    dcm2niix_output_fn = dcm2niix_pat.format(**tags)
    logger.debug('dcm2niix ouput filename: %s', dcm2niix_output_fn)
    return dcm2niix_output_fn

def read_dcm_header(fn, required_fields, defer_size='1 KB', allow_missing_field=False):
    # logger.debug('Read header from %s', fn)
    dcm = dcmread(fn, stop_before_pixels=True, defer_size=defer_size)
    #from IPython import embed; embed()
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
