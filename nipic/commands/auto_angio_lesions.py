import sys
import logging
from optparse import OptionParser

from nipic.freesurfer import Freesurfer
from nipic.angio_lesions import fs_angio_lut

logger = logging.getLogger('nipic')

def main():
    min_args = 1
    max_args = 1

    usage = 'usage: %prog [options] SUBJECT_NAME'
    description = ('Produce a coarse segmentation of angiopathic lesions from '\
                   'T1, T2, FLAIR, DWI and T2-GRE')

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

    parser.add_option('-g', '--save-figures', action='store_true', default=False,
                      help='Save histograms')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    subject_name = args[0]

    freesurfer = Freesurfer()
    freesurfer.auto_angio_lesions(subject_name,
                                  save_figures=options.save_figures)

    
