


fs_angio_lut = {
    15000 : {
        'name' : 'DWMH',
        'color' : [200, 70, 255, 0]
    },
    15001 : {
        'name' : 'VWMH',
        'color' : [200, 70, 255, 0]
    }, 
    15002 : {
        'name' : 'small_infarct',
        'color' : [200, 70, 255, 0]
    },
    15003 : {
        'name' : 'lacune',
        'color' : [200, 70, 255, 0]
    },
    15004 : {
        'name' : 'micro_bleed',
        'color' : [200, 70, 255, 0]
    },
    15005 : {
        'name' : 'perivascular_space',
        'color' : [200, 70, 255, 0]
    }
}
import anatomist.api as anatomist


# convert aseg.img to aseg.arg (volume image to set of drawable rois)

# Load freesurfer's LUT

# Load aseg in anatomist: fix ROI names and colors according to freesurfer's LUT
anatomist_app = anatomist.Anatomist()
aseg_agraph = anatomist_app.loadObject('aseg.arg').toAimsObject()
rois = gsa.vertices().list()
fs_lut = {}
for angio_roi_idx in (set(fs_angio_lut.keys())
                      .difference({r['roi_label'] for r in rois})):
    roi_vertex = gsa.addVertex('roi')
    roi_vertex['name'] = fs_angio_lut[angio_roi_idx]['name']
    roi_vertex['roi_label'] = angio_roi_idx
# TODO save and reload

for roi in rois:
    fs_roi_def = fs_lut[roi['roi_label']]
    roi['name'] = fs_roi_de['name']
    aroi = roi['ana_object']
    roi_material = aroir.GetMaterial()
    roi_material.set({'diffuse':fs_roi_def['color_rgb']})
    aroi.SetMaterial(roi_material)
    aroi.notifyObservers()
