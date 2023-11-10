import sys
import logging
from optparse import OptionParser
import subprocess
from itertools import chain
from nipic.freesurfer import Freesurfer

logger = logging.getLogger('nipic')

# ('aseg.mgz', 'colormap=lut', 'opacity=0.2')
VOLUMES = [
    {
        'fn' : '{root}/mri/T1.mgz',
        'p_0' : ['name={subject}_T1.mgz'],
        'p_1' : ['name={subject}_T1.mgz'],
    },
    {
        'fn' : '{root}/mri/FLAIR.mgz',
        'p_0' : ['name={subject}_FLAIR.mgz'],
        'p_1' : ['name={subject}_FLAIR.mgz'],
    },
    {
        'fn' : '{root}/mri/wm.mgz',
        'p_0' : ['name={subject}_wm.mgz'],
        'p_1' : ['name={subject}_wm.mgz'],
    },
    {
        'fn' : '{root}/mri/brainmask.mgz',
        'p_0' : ['name={subject}_brainmask.mgz'],
        'p_1' : ['name={subject}_brainmask.mgz'],
    }
]

SURFACES = [
    {
        'fn' : '{root}/surf/lh.white',
        'p_0' : ['edgecolor=blue', 'name={subject}_lh.white'],
        'p_1' : ['edgecolor=0,255,255', 'name={subject}_lh.white'],
    },
    {
        'fn' : '{root}/surf/lh.pial',
        'p_0' : ['edgecolor=red', 'overlay={root}/surf/lh.thickness',
                 'overlay_color=colorwheel,inverse',
                 'overlay_threshold=0.75,5', 'name={subject}_lh.pial'],
        'p_1' : ['edgecolor=255,126,0', 'overlay={root}/surf/lh.thickness',
                 'overlay_color=colorwheel,inverse',
                 'overlay_threshold=0.75,5', 'name={subject}_lh.pial']
    },
    {
        'fn': '{root}/surf/rh.white',
        'p_0' : ['edgecolor=blue', 'name={subject}_rh.white'],
        'p_1' : ['edgecolor=0,255,255', 'name={subject}_rh.white'],
    },
    {
        'fn': '{root}/surf/rh.pial',
        'p_0' : ['edgecolor=red', 'overlay={root}/surf/rh.thickness',
                 'overlay_color=colorwheel,inverse',
                 'overlay_threshold=0.75,5', 'name={subject}_rh.pial'],
        'p_1' : ['edgecolor=255,126,0', 'overlay={root}/surf/rh.thickness',
                 'overlay_color=colorwheel,inverse',
                 'overlay_threshold=0.75,5', 'name={subject}_rh.pial'],
    }
]

def main():

    min_args = 1
    max_args = 2

    usage = 'usage: %prog [options] SUBJECT_LABEL'
    description = ('Produce segmentation stats table')

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=0, help='Verbose level')

    (options, args) = parser.parse_args()

    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        sys.exit(1)

    freesurfer = Freesurfer()
    subjects = [(arg, freesurfer.subject_dir(arg)) for arg in args[:2]]
    cmd = (['freeview'] +
           ['-v'] + list(chain(*[[':'.join([v['fn'].format(root=sroot)] +
                                           [s.format(root=sroot, subject=subj) for s in
                                            v.get('p_%d' % isubj, [])])
                                  for v in VOLUMES]
                                 for isubj, (subj, sroot) in enumerate(subjects)])) +
           ['-f'] + list(chain(*[[':'.join([v['fn'].format(root=sroot)] +
                                           [s.format(root=sroot, subject=subj) for s in
                                            v.get('p_%d' % isubj, [])])
                                  for v in SURFACES]
                                 for isubj, (subj, sroot) in enumerate(subjects)]))
    )
    logger.debug('Command: %s', cmd)
    subprocess.run(cmd)
