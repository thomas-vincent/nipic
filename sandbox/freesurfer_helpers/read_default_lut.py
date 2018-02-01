"""
Read the standard LookUp Table of freesurfer from the file FreeSurferColorLUT.txt.

See https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#ColorLookupTablefile

"""
import os
import os.path as op

def read_default_lut():
    """
    TODO: doc, test
    """
    fs_home = os.getenv('FREESURFER_HOME')
    if len(fs_home) == 0:
        raise Exception('FREESURFER_HOME env variable not found. ' \
                        'Check freesurfer installation.')
    
    lut_fn = op.join(fs_home, 'FreeSurferColorLUT.txt')
    if not op.exists(lut_fn):
        raise Exception('Standard FS LUT file not found: %s' % lut_fn) 
    
    lut = {}
    with open(lut_fn) as flut:
        for line in flut.readlines():
            line = line.strip('\n')
            if not line.startswith('#') and not len(line) == 0:
                toks = line.split()
                lut[int(toks[0])] = {'name':toks[1],
                                     'Color':[int(toks[2]),
                                              int(toks[3]),
                                              int(toks[4])]}
                
    return lut

