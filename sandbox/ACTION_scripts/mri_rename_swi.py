import os
import os.path as op
import shutil
import logging
import sys
import subprocess
from collections import defaultdict
from pydicom import dcmread

import logging
logging.basicConfig()
logger = logging.getLogger('lesca proc')
logger.setLevel(logging.DEBUG)


def read_dcm_header(fn, defer_size='1 KB'):
    logger.debug('Read header from %s', fn)
    return dcmread(fn, stop_before_pixels=True, defer_size=defer_size)

acq_root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/ACR-0001-00006/ACTIONCARDIORISK_T0_20190830/05_Axial_t2_SWI_3d_axial_OPT1'

tmp_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/test/test_swi'
dcm_dispatch = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for dcm_fn in os.listdir(acq_root):
    h = read_dcm_header(op.join(acq_root, dcm_fn))
    echo_time, slice_pos, coil = h.EchoTime, h.SliceLocation, h[('0051', '100f')].value
    dcm_dispatch[echo_time][coil][slice_pos].append(dcm_fn)

for et in sorted(dcm_dispatch.keys()):
    for sl in sorted(dcm_dispatch[et].keys()):
        for c in sorted(dcm_dispatch[et][sl].keys()):
            print('%s %s %s: %d' % (et, sl, c, len(dcm_dispatch[et][sl][c])))

for fn in list(sorted(fn[0] for fn in dcm_dispatch[6.92]['H9'].values())):
    shutil.copy(op.join(acq_root, fn), op.join(tmp_dir, fn))
