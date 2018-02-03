"""
Generate screenshots of cortical and sub-cortical surfaces with morphometrics overlays. 

Usage::

    plot_morphometrics SUBJECT_LIST
                       [
                        --subject_dir==<path_to_freesurfer_subject_dir>
                        --verbose <VERBOSE_LEVEL>]
"""
import os
import os.path as op
from subprocess import call

# view_options = '--layout 1 --viewport 3d --colorscale --zoom 1.5 --viewsize 1000 1000 --nocursor --hide-3d-slices'
# view_orientations = {'left': '', 'right': '--cam azimuth 180'}
# surf_overlay_params = {'ThickAvg' : ':overlay_color=colorwheel,inverse:overlay_threshold=1.5,3.2'}
# parcellation_tags = '.a2009s'

# for subject in subjects:
#     subject_dir = op.join()


cmd_pats = [
    ['freeview', '-f', '{subject}/surf/lh.pial:overlay={subject}/surf/lh.aparc.a2009s.ThickAvg:overlay_color=colorwheel,inverse:overlay_threshold=1.5,3.2', '-f', '{subject}/surf/rh.pial:overlay={subject}/surf/rh.aparc.a2009s.ThickAvg:overlay_color=colorwheel,inverse:overlay_threshold=1.4,3.5', '--layout', '1', '--viewport', '3d', '--colorscale', '--zoom', '1.5', '--viewsize', '1000', '1000', '--nocursor', '--hide-3d-slices', '--screenshot', '{subject}/image/cortical_thickness_left.png'],
    ['freeview', '-f', '{subject}/surf/lh.pial:overlay={subject}/surf/lh.aparc.a2009s.ThickAvg:overlay_color=colorwheel,inverse:overlay_threshold=1.5,3.2', '-f', '{subject}/surf/rh.pial:overlay={subject}/surf/rh.aparc.a2009s.ThickAvg:overlay_color=colorwheel,inverse:overlay_threshold=1.4,3.5', '--layout', '1', '--viewport', '3d', '--colorscale', '--zoom', '1.5', '--viewsize', '1000', '1000', '--nocursor', '--hide-3d-slices', '-cam', 'azimuth', '180', '--screenshot', '{subject}/image/cortical_thickness_right.png']]

os.environ['LD_LIBRARY_PATH'] = '/home/tom/Projects/Research/Software/Brainvisa/bin_pack/lib/'

for subject in ['SYN_MTL_001_T0', 'SYN_MTL_002_T0', 'SYN_MTL_999_T0']:
    img_dir = op.join(subject, 'image')
    if not op.exists(img_dir):
        os.makedirs(img_dir)
    for cmd_pat in cmd_pats:
        cmd = [cp.format(subject=subject) for cp in cmd_pat]
        assert call(cmd)==0
