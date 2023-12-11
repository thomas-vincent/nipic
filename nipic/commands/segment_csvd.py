import sys
import logging
from optparse import OptionParser

from nipic.freesurfer import Freesurfer
from nipic.csvd import create_csvd_workflow

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

    parser.add_option('-s', '--fs_subjects_dir',
                      help="Path to Freesurfer's subject directory")

    parser.add_option('-g', '--save-figures', action='store_true', default=False,
                      help='Save histograms')

    parser.add_option('-n', '--nb_threads', default=1, type='int',
                      help='Number of threads for parallel processing')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    subject_id = args[0]

    csvd_workflow = create_csvd_workflow(options.fs_subjects_dir)
    csvd_workflow.input_node.subject_id = subject_id
    csvd_workflow.input_node.save_figure = options.save_figures
    csvd_workflow.input_node.nb_threads = options.nb_threads
    csvd_workflow.run()

