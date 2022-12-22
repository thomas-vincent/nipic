import os
import os.path as op
import shutil
import logging
import sys

from pydicom import dcmread

import logging
logging.basicConfig()
logger = logging.getLogger('lesca proc')
logger.setLevel(logging.INFO)

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

def insure_folder_exists(fn):
    folder = op.dirname(fn)
    if not op.exists(folder):
        os.makedirs(folder)
    return fn

time_point = 'T0'
root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/'
# root = '/media/lesca/Elements/ACTIONcardioRisk_BIDS'
for subject in os.listdir(root):
    mixed_dir = op.join(root, subject, 'ACTIONCARDIORISK_%s' % time_point)
    if op.exists(mixed_dir):
        for acq_subdir in os.listdir(mixed_dir):
            acq_dir = op.join(mixed_dir, acq_subdir)
            vdate = read_dcm_header(op.join(acq_dir, os.listdir(acq_dir)[0]), ['StudyDate'])['StudyDate']
            visit_dir_new = op.join(root, subject, 'ACTIONCARDIORISK_%s_%s' % (time_point, vdate))
            for fn in os.listdir(acq_dir):
                src = op.join(acq_dir, fn)
                dest = op.join(visit_dir_new, acq_subdir, fn)
                print('%s\n->\n%s' % (src, dest))
                print()
                shutil.move(src, insure_folder_exists(dest))
            if len(os.listdir(acq_dir)) == 0:
                os.rmdir(acq_dir)
            else:
                print('WARNING %s not empty' % acq_dir)            
        if len(os.listdir(mixed_dir)) == 0:
            os.rmdir(mixed_dir)
        else:
            print('WARNING %s not empty' % mixed_dir)
        
            
