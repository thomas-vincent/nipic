import sys
import logging
from optparse import OptionParser

import anatomist.api as anatomist

from nipic.freesurfer import Freesurfer
from nipic.angio_lesions import fs_angio_lut

logger = logging.getLogger('nipic')

def main():
    min_args = 1
    max_args = 1

    usage = 'usage: %prog [options] SUBJECT_NAME'
    description = 'Use anatomist to tag angiopathic lesions for given subject.'

    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-v', '--verbose', dest='verbose',
                      metavar='VERBOSELEVEL',
                      type='int', default=0,
                      help='Amount of verbose: '\
                           '0 (NOTSET: quiet, default), '\
                           '50 (CRITICAL), ' \
                           '40 (ERROR), ' \
                           '30 (WARNING), '\
                           '20 (INFO), '\
                           '10 (DEBUG)')

    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)

    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1

    subject_name = args[0]

    freesurfer = Freesurfer()
    # convert aseg.img to aseg.arg (volume image to set of drawable rois)
    aseg_graph_fn = freesurfer.seg_to_aims_graph(subject_name, 'aseg.mgz',
                                                 'aseg.arg')


    # Load freesurfer's LUT
    fs_lut = freesurfer.load_lut()

    # Load aseg in anatomist: fix ROI names and colors according to freesurfer's LUT
    anatomist_app = anatomist.Anatomist()
    aseg_aobj = anatomist_app.loadObject(aseg_graph_fn)
    aseg_agraph = aseg_aobj.toAimsObject()

    rois = aseg_aobj.vertices().list()
    missing_roi_indexes = (set(fs_angio_lut.keys())
                           .difference({r['roi_label'] for r in rois}))
    if len(missing_roi_indexes):
        logger.info('Fix missing rois for angiopathic lesions in aseg graph')
        for angio_roi_idx in missing_roi_indexes:
            roi_vertex = aseg_agraph.addVertex('roi')
            roi_vertex['name'] = fs_angio_lut[angio_roi_idx]['name']
            roi_vertex['roi_label'] = angio_roi_idx
            logger.info('Add missing roi to aseg graph: %s', roi_vertex['name'])
        logger.info('Save fixed aseg graph')
        aseg_aobj.setChanged()
        aseg_aobj.notifyObservers()

    for roi in rois:
        fs_roi_def = fs_lut[roi['roi_label']]
        roi['name'] = fs_roi_def['name']
        aroi = roi['ana_object']
        roi_material = aroi.GetMaterial()
        roi_material.set({'diffuse':fs_roi_def['color_rgb']})
        aroi.SetMaterial(roi_material)
        aroi.setChanged()
        aroi.notifyObservers()
