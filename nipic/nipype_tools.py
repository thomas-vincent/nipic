from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe

def get_first(**kwargs):
    return next(iter(kwargs.values()))[0]

def first_of(node, field, wf):
    first = pe.Node(Function(input_names=[field],
                             output_names=[field],
                             function=get_first),
                    name='grab_single_%s' % field)
    wf.connect(node, field, first, field)
    return first
