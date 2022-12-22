import os.path as op

def rebase(folder, root, new_root):
    return op.join(new_root, op.relpath(folder, root))

def bids_split(path):
    print('!!!!!!!!!path:', path)
    if not op.isdir(path):
        path = op.dirname(path)
    toks = path.split(op.sep)
    root = toks[0] if toks[0] != '' else op.sep
    for itok, f in enumerate(toks[1:]):
        root = op.join(root, f)
        if op.exists(op.join(root, 'dataset_description.json')):
            return root, op.join(*toks[(itok+2):])    
    raise Exception('BIDS root not found from %s', path)

import nipype.interfaces.io as nio    
class UnpackSingleBIDSDataGrabber(nio.BIDSDataGrabber):
    def _list_outputs(self):
         outputs = super(UnpackSingleBIDSDataGrabber, self)._list_outputs()
         new_outputs = {}
         for k,v in outputs.items():
             if len(v) == 1:
                 new_outputs[k] = v[0]
             else:
                 new_outputs[k] = v  
         return new_outputs
