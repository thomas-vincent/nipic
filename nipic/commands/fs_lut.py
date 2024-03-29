from pprint import pprint
import logging
from optparse import OptionParser

import nipic.freesurfer as fs
import nipic.csvd as svd

logger = logging.getLogger('nipic')

def main():
    min_args = 0
    max_args = 0

    usage = 'usage: %prog [options] SUBJECT_NAME'
    description = "Print freesurfer's LUT to stdout"

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

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    lut = fs.load_lut(aseg_only=True)
    lut.update(svd.fs_csvd_lut)
    print(fs.lut_to_str(lut))
