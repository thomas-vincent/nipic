import os
import os.path as op
import shutil

def safe_fn(fn):
    fn = ''.join(c for c in fn 
                 if c.isalpha() or c.isdigit() or c in '- _.' or c==op.sep).rstrip()
    fn = fn.replace(' ', '_').replace('_--_', '_')
    return fn


def insure_folder_exists(fn):
    folder = op.dirname(fn)
    if not op.exists(folder):
        os.makedirs(folder)
    return fn

src_root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/'
dest_root = '/media/lesca/Elements/ACTIONcardioRisk_BIDS'
for path, subdirs, fns in os.walk(src_root):
    for fn in fns:
        src = op.join(path, fn)
        dest = op.join(dest_root, op.relpath(path, src_root), fn)
        if not op.exists(dest):
            print('%s\n->\n%s' % (src, dest))
            print()
            shutil.copy(src, insure_folder_exists(dest))
