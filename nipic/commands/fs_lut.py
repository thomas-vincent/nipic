from pprint import pprint
import logging
from optparse import OptionParser

from nipic.freesurfer import Freesurfer

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

    lines = []
    for idx, tissue in Freesurfer().load_lut(aseg_only=True, add_csvd=True).items():
        name = tissue['name']
        r, g, b, a = tissue['color']
        lines.append(f'{idx}\t{name}\t{r}\t{g}\t{b}\t{a}')
    print('\n'.join(lines))
