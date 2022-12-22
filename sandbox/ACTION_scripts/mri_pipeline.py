# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
- Uncombined SWI -> Romeo combined -> QSM
- T1 & FLAIR -> fs_results
- fs_results -> bullseye_wmparc
- Diff MRI -> DTI_ADC
- fs_results & DTI_ADC & QSM & bullseye_wmparc -> aseg_angiol
- aseg_angiol -> morphometrics table
"""
import os
import nipype.interfaces.io as nio
import nipype.pipeline as nppl
import nipype.interfaces.utility as nut
import nipype.interfaces.base as nifbase 

# TODO put these nipic.workflows
from nipic.bids import UnpackSingleBIDSDataGrabber
from nipic.freesurfer import ReconAllInterface

from nipic.worflows import SnapInterface
from nipic.workflows import bids_rebase_derivs


os.environ["SUBJECTS_DIR"] = ("/home/lesca/DataServer/Project/ACTIONcardioRisk/"
                              "MRI_BIDS/derivatives/freesurfer")

## Data definitions
mri_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/'
deriv_dir = '/home/lesca/DataServer/Project/ACTIONcardioRisk/MRI_BIDS/derivative'

UNCOMBINED_SWI_ACQ_TAG = 'Axialt2SWI3daxialOPT1'
query = {
    't1' : {
        'suffix' : 'T1w',
        'acquisition' : 'SagMPRAGE',
        "extension": ["nii", ".nii.gz"] 
    },
    'FLAIR' : {
        'suffix' : 'FLAIR',
        'acquisition' : 'SagFlair3dt2space',
        "extension": ["nii", ".nii.gz"]
    },
    'swi-uncombined-phase' : {
        'acquisition' : UNCOMBINED_SWI_ACQ_TAG,
        'part' : 'phase',
        "extension": ["nii", ".nii.gz"]
    },
    'swi-uncombined-mag' : {
        'acquisition' : UNCOMBINED_SWI_ACQ_TAG,
        'part' : 'mag',
        "extension": ["nii", ".nii.gz"]
    }
}


itf = UnpackSingleBIDSDataGrabber(base_dir=mri_dir,
                                  subject='00006',
                                  output_query=query)
bids_grab = nppl.Node(interface=itf, name='grab_data')

wf = nppl.Workflow(name="main_workflow")
for img_label in ("t1", 'FLAIR'):
    rebase = nppl.Node(interface=bids_rebase_derivs, 
                       name='rebase_derivatives_' + img_label)
    snap_node = nppl.Node(interface=SnapInterface(), 
                          name='snapshot_%s' % img_label)
    wf.connect(bids_grab, img_label, rebase, "bids_in_file")
    wf.connect(rebase, "out_file", snap_node, "output_folder")
    wf.connect(bids_grab, img_label, snap_node, "mri_fn")

wf.run()
