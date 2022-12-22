import os
import nipype.interfaces.io as nio
import nipype.pipeline as nppl
import nipype.interfaces.utility as nut
import nipype.interfaces.base as nifbase 



def bids_rebase_to_derivatives(bids_in_file):
    from nipic.bids import bids_split
    import os
    bids_root, sub_path = bids_split(bids_in_file)
    rebased_path = os.path.join(bids_root, 'derivatives', sub_path)
    if not os.path.exists(rebased_path):
        os.makedirs(rebased_path)
    return rebased_path

bids_rebase_derivs = nut.Function(input_names=['bids_in_file'],          
                                  output_names=['out_file'],
                                  function=bids_rebase_to_derivatives)

class SnapInputSpec(nifbase.TraitedSpec):
    mri_fn = nifbase.File(desc="mri_fn", exists=True,
                          mandatory=True, argstr='%s')
    output_folder = nifbase.Directory(desc="output_folder", exists=True,
                                      mandatory=True, argstr='-o %s')

class SnapOutputSpec(nifbase.TraitedSpec):
    img_fn = nifbase.File(desc='snapshot_image')

class SnapInterface(nifbase.CommandLine):
    input_spec = SnapInputSpec
    output_spec = SnapOutputSpec
    _cmd = "mri_snap --slice_axes axial --mip none"

    def _run_interface(self, runtime):
        super(SnapInterface, self)._run_interface(runtime)

    def _list_outputs(self):
        import os.path as op
        outputs = self.output_spec().get()
        output_img_bfn = op.splitext(op.basename(self.inputs.mri_fn))[0]+'.png'
        outputs['img_fn'] = op.join(self.inputs.output_folder,
                                    output_img_bfn)
        return outputs
        
        

