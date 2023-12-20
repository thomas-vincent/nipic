import os.path as op

import numpy as np
import matplotlib
import nibabel as nib

from .utils import awrap, MRI_3D_AXES, NIBABEL_SLICER_VIEWS, split_ext_gz_safe

MAX_SPLIT_SIZE = 5

def reduce_guess(a, axis=0):
    if a.shape[axis] <= MAX_SPLIT_SIZE:
        return awrap(np.split)(a, axis)
    else:
        return awrap(np.mean)(a, axis)

REDUCTORS = {'mean' : awrap(np.mean), 'std': awrap(np.std), 
             'min' : awrap(np.min),
             'max' : awrap(np.max), 'median' : awrap(np.median),
             'split' : awrap(np.split), 'default': reduce_guess}

class UnsupportedNumberOfDims(Exception): pass

def mri_snap(mri_fn, output_folder, slice_axes=MRI_3D_AXES, mip_axes=MRI_3D_AXES,
             reductors=None):
    """
    Generate image of slices/mips from 3D or 4D MRI data

    TODO: black background + white text
    TODO: use neurological convention
    TODO: utests

    Input: 
        - mri_fn (str):
            path to MRI file from which to generate snapshot images
        - output_folder (str):
            path to store iamges
        - slice_axes (list of str):
            list of axes along which to generate centered-sliced snaps 
            (axial, coronal, sagittal).
        - mip_axes (list of str):
            list of axes along which to generate centered-sliced snaps
            (axial, coronal, sagittal).
        - reductors (list of callables):
            list of functions to apply to create 3D volumes from 4D MRI.
    """
    img = load_mri(mri_fn)
    base_fn = op.join(output_folder,
                      op.basename(split_ext_gz_safe(mri_fn)[0]))
    reductors = (reductors if reductors is not None
                 else [REDUCTORS['mean']])

    if len(img.shape) > 3:
        idata = img.get_data()
        if len(img.shape) == 5:
            idata = idata[:,:,:,:,0] # TODO reduction on more axes
        elif len(img.shape) > 5:
            raise UnsupportedNumberOfDims()
        for reductor in reductors:
            if (img.extra is not None and 'grad_table' in img.extra and
                'b_values' in img.extra): # DTI
                rname, rvols = reduce_dti(idata, extra=img.extra)
            else:
                rname, rvols = reductor(idata, axis=3, extra=img.extra)
            for ivol, vol in enumerate(rvols):
                if len(rvols) == 1:
                    suffix = '_' + rname
                else:
                    suffix = '_vol%04d' % ivol
                reduced_img = img.__class__(vol, img.affine,
                                            img.header, img.extra)
                plot_slices(reduced_img, slice_axes, base_fn + suffix)
                plot_mips(reduced_img, mip_axes, base_fn + suffix)
    else:
        plot_slices(img, slice_axes, base_fn)
        plot_mips(img, mip_axes, base_fn)

def plot_mips(img, mip_axes, base_fn):

    for aname in mip_axes:
        ia = MRI_3D_AXES.index(aname)
        mshape = img.shape[:ia] + (1,) + img.shape[ia+1:]
        mip_data = img.get_data().max(axis=ia).reshape(*mshape)
        mip_img = img.__class__(mip_data, img.affine, img.header,
                                img.extra)
        slicer = mip_img.orthoview()
        center = [img.shape[o]/2 for o in slicer._order]
        slicer.set_position(*np.dot(slicer._affine, center+[1])[:3])

        fig = slicer.figs[0]
        fig_transf = fig.dpi_scale_trans.inverted()

        ax = slicer._axes[NIBABEL_SLICER_VIEWS.index(aname)]
        img_fn = base_fn + '_mip_' + aname + '.png'
        ext = ax.get_window_extent().transformed(fig_transf)
        fig.savefig(img_fn, bbox_inches=ext.expanded(1.1,1.1))

            
def plot_slices(img, axes, base_fn):
    slicer = img.orthoview()
    center = [img.shape[o]/2 for o in slicer._order]
    slicer.set_position(*np.dot(slicer._affine, center+[1])[:3])
    [a.findobj(matplotlib.image.AxesImage)[0].cmap.set_bad(color='red')
     for a in slicer._axes]
    if 'grad_table' in img.extra: # DTI
        [a.findobj(matplotlib.image.AxesImage)[0].set_cmap('jet')
         for a in slicer._axes]
    
    fig = slicer.figs[0]
    fig_transf = fig.dpi_scale_trans.inverted()
    for aname in axes:
        ax = slicer._axes[NIBABEL_SLICER_VIEWS.index(aname)]
        img_fn = base_fn + '_center_slice_' + aname + '.png'
        ext = ax.get_window_extent().transformed(fig_transf)
        fig.savefig(img_fn, bbox_inches=ext.expanded(1.1,1.1))
    # auto_crop_img(img_fn)
