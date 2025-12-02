from pprint import pprint
import logging
from optparse import OptionParser
import json
import pandas as pd

logger = logging.getLogger('nipic')


def df_delta(df_orig, df_new, label_orig='original', label_new='new'):

    delta = {}
    delta['missing_rows'] = df_orig.index.difference(df_new.index)
    delta['new_rows'] = df_new.index.difference(df_orig.index)

    common_index = df_orig.index.intersection(df_new.index)

    delta['missing_cols'] = list(sorted(set(df_orig.columns)
                                        .difference(df_new.columns)))
    delta['new_cols'] = list(sorted(set(df_new.columns)
                                    .difference(df_orig.columns)))

    common_cols = list(sorted(set(df_new.columns)
                              .intersection(df_orig.columns)))
    # import pdb; pdb.set_trace()
    delta['diff'] = (df_orig.loc[common_index, common_cols].round(4)
                     .compare(df_new.loc[common_index, common_cols].round(4),
                              result_names=(label_orig, label_new)))
    return delta

def report_table_difference(table_orig_fn, table_orig_sheet_name,
                            table_new_fn, table_new_sheet_name,
                            label_orig, label_new, report_fn,
                            index_cols=diff_def['common_index']):

    delta = df_delta((pd.read_excel(table_orig_fn, engine='openpyxl', sheet_name=table_orig_sheet_name)
                      .set_index(index_cols)),
                     (pd.read_excel(table_new_fn, engine='openpyxl', sheet_name=table_new_sheet_name)
                      .set_index(index_cols)),
                     label_orig=label_orig, label_new=label_new)

    if report_fn is None:
        report_fn = op.join(op.dirname(table_new_fn),
                            f'{label_orig}_to_{label_orig}.txt')

    report = []
    report.append('Differences between:\nORIG: %s\nNEW: %s' %
                  (table_orig_fn, table_new_fn))
    report.append('')
    if len(delta['missing_rows']) > 0:
        report.append('WARNING: rows in ORIG but not in NEW:')
        report.append(', '.join(delta['missing_rows']))
        report.append('')

    if len(delta['missing_cols']) > 0:
        report.append('WARNING: columns in ORIG but not in NEW:')
        report.append(', '.join(delta['missing_cols']))
        report.append('')

    if delta['diff'].shape[0] > 0:
        diff_fn =  op.splitext(report_fn)[0] + '.xlsx'
        report.append('WARNING: values changed from ORIG to NEW:')
        report.append('see %s' % diff_fn)
        delta['diff'].reset_index().to_excel(diff_fn)
        report.append('')

    if len(delta['new_rows']) > 0:
        report.append('New rows in NEW:')
        report.append(', '.join(delta['new_rows']))
        report.append('')

    if len(delta['new_cols']) > 0:
        report.append('New columns in NEW:')
        report.append(', '.join(delta['new_cols']))
        report.append('')

    with open(report_fn, 'w') as fout:
        fout.write('\n'.join(report))


def main():
    min_args = 1
    max_args = 1

    usage = 'usage: %prog [options] DIFF_DEF_JSON'
    description = "Difference between two excel sheets"

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

    with open(options.args[0]) as fin:
        diff_def = json.load(fin)


    report_table_difference(diff_def['first_item']['data_file'],
                            diff_def['first_item'].get('sheet_name', 0),
                            table_new_fn, table_new_sheet_name,
                            label_orig, label_new, report_fn,o
                            index_cols=diff_def['common_index'])

    df1 = pd.read_excel(,
                        sheet_name=)
    df1 = df1.set_index()
