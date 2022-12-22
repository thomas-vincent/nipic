import sys
import logging
from optparse import OptionParser
import pandas as pd

from nipic.freesurfer import Freesurfer

logger = logging.getLogger('nipic')

def main():

    min_args = 0
    max_args = 0

    usage = 'usage: %prog [options]'
    description = ('Produce segmentation stats table')

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-s', '--subjects', metavar='LIST_OF_STR',
                      type='str', 
                      help='Comma-separated list of subjects.')
                      
    parser.add_option('-t', '--stats', metavar='LIST_OF_STR',
                      type='str', 
                      help='Comma-separated list of stats.')

    parser.add_option('-r', '--region-names', metavar='LIST_OF_STR',
                      type='str',
                      help='Comma-separated list of region Ids')

    parser.add_option('-m', '--measures', metavar='LIST_OF_STR',
                      type='str',
                      help='Comma-separated list of measure labels')
    
    parser.add_option('-o', '--output-file', type='str',
                      metavar='PATH',
                      help='Output files to save table')
    
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
    if options.subjects is not None:
        subjects = options.subjects
    else:
        subjects = freesurfer.subjects()
    logger.info('Subjects: %s', subjects)  
    
    def safe_split(s, split_on):
        if s is None:
            return None
        else:
            return [e for e in s.split(split_on) if len(e)>0]
    to_concat = [freesurfer.stat_seg_to_df(s, struct_names=safe_split(options.region_names, ','),
                                           struct_stats=safe_split(options.stats, ','),
                                           measure_labels=safe_split(options.measures, ','))
                 for s in subjects]
                 
    stats = pd.concat(to_concat, axis=0, join='outer')
    # from IPython import embed; embed()
    
    if options.output_file is not None:
        stats.sort_index().reset_index().to_excel(options.output_file, index=False)
        
    
                 
