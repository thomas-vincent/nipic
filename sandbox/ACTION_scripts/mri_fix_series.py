import os
import os.path as op
import shutil


time_point = 'T0'
root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/'
# root = '/media/lesca/Elements/ACTIONcardioRisk_BIDS'
for subject in os.listdir(root):
    src = op.join(root, subject, 'ACTIONCARDIORISK')
    dest = op.join(root, subject, 'ACTIONCARDIORISK_%s' % time_point)
    if op.exists(src):
        print('%s\n->\n%s' % (src, dest))
        print()
        if op.exists(dest):
            raise IOError('Destination exists')
        os.rename(src, dest)
    session_dir = dest

    if 1:
        for acq_subdir in os.listdir(session_dir):
            tokens = acq_subdir.split('_')
            if op.isdir(op.join(session_dir, acq_subdir)) and not tokens[0].isdigit():
                series_number = tokens[-1]
                src = op.join(session_dir, acq_subdir)
                dest = op.join(session_dir, '_'.join(['%02d' % int(tokens[-1])] + tokens[:-1]))
                print('%s\n->\n%s' % (src, dest))
                print()
                if op.exists(dest):
                    raise IOError('Destination exists')
                os.rename(src, dest)
    if 1:
        for acq_subdir in os.listdir(session_dir):
            acq_dir = op.join(session_dir, acq_subdir)
            for bfn in os.listdir(acq_dir):
                src = op.join(acq_dir, bfn)
                dest = op.join(acq_dir, bfn.replace(subject, '%s_%s' % (subject, time_point)))
                print('%s\n->\n%s' % (src, dest))
                print()
                if op.exists(dest):
                    raise IOError('Destination exists')
                os.rename(src, dest)
