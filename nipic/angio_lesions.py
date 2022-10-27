
DWMH = 15000
VWMH = 15001
SMALL_INFARCT = 15002
LACUNE = 15003
MICRO_BLEED = 15004
PV_SPACE = 150005

fs_angio_lut = {
    DWMH : {
        'name' : 'DWMH',
        'color' : [200, 70, 255, 0]
    },
    VWMH : {
        'name' : 'VWMH',
        'color' : [200, 70, 255, 0]
    }, 
    SMALL_INFARCT : {
        'name' : 'small_infarct',
        'color' : [200, 70, 255, 0]
    },
    LACUNE : {
        'name' : 'lacune',
        'color' : [200, 70, 255, 0]
    },
    MICRO_BLEED : {
        'name' : 'micro_bleed',
        'color' : [200, 70, 255, 0]
    },
    PV_SPACE : {
        'name' : 'perivascular_space',
        'color' : [200, 70, 255, 0]
    }
}

LACUNE_MIN_LENGTH_MM = 2 # lower than 3mm because of smoothing effect for small obj
LACUNE_MAX_LENGTH_MM = 20 # larger to be more convervative
LACUNE_MIN_ISOPQ = 0.5

PVS_MIN_ISOPQ = 0.3
PVS_MAX_ISOPQ = 0.5
