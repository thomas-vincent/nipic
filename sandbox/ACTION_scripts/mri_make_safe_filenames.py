import os
import os.path as op
import shutil

def safe_fn(fn):
    fn = ''.join(c for c in fn 
                 if c.isalpha() or c.isdigit() or c in '- _.' or c==op.sep).rstrip()
    fn = fn.replace(' ', '_').replace('_--_', '_')
    return fn

# root = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI/'
root = '/media/lesca/Elements/ACTIONcardioRisk_BIDS'
for path, subdirs, fns in os.walk(root):
    for subdir in subdirs:
        fixed_dir = safe_fn(subdir)
        src = op.join(path, subdir)
        dest = op.join(path, fixed_dir)
        if fixed_dir != subdir:
            print('%s\n->\n%s' % (src, dest))
            print()
            if op.exists(dest):
                raise IOError('Destination exists')
            os.rename(src, dest)

for path, subdirs, fns in os.walk(root):
    for fn in fns:
        fixed_fn = safe_fn(fn)
        src = op.join(path, fn)
        dest = op.join(path, fixed_fn)
        if fixed_fn != fn:
            print('%s\n->\n%s' % (src, dest))
            print()
            if op.exists(dest):
                raise IOError('Destination exists')
            os.rename(src, dest)
