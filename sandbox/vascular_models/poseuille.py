#! /usr/bin/env python3

import numpy as np

def cpt_flow_poseuille(p1, p2, radius, length, viscosity):
    """
    p1, p2 : mmHg
    radius : cm
    length : cm
    viscosity : Pa.s

    Result: flow in ml.s-1
    """
    return (pressure_mmHg_to_Pa(p1) - pressure_mmHg_to_Pa(p2)) * \
        np.pi * radius**4 / (8 * viscosity * length) * 1000

def cpt_pressure_drop_poseuille(flow, radius, length, viscosity):
    """
    flow : ml.s-1
    radius : cm
    length : cm
    viscosity : Pa.s

    Result:  delta pressure in mmHg
    """
    return pressure_Pa_to_mmHg(flow * (8 * viscosity * length) /   \
                               (np.pi * radius**4 * 1000))


def pressure_Pa_to_mmHg(p):
    return p * 133.3

def pressure_mmHg_to_Pa(p):
    return p / 133.3
    
def main():
    viscosity_blood = 0.005 # Pa.s
    blood_flow_aorta_in = 83 # ml.s-1
    # Ascending portion of the aorta:
    radius_aorta = 1 # cm
    length_aorta = 5 # cm
    p_aorta_in = 120 # mmHg, systolic peak
    p_aorta_out = 119.3 # mmHg, systolic peak

    flow_aorta = cpt_flow_poseuille(p_aorta_in, p_aorta_out, radius_aorta,
                                    length_aorta, viscosity_blood)
    print('Flow aorta = {} ml.s-1'.format(flow_aorta))
    pdrop_aorta = cpt_pressure_drop_poseuille(blood_flow_aorta_in, radius_aorta,
                                              length_aorta, viscosity_blood)
    print('Pressure drop aorta = {} mmHg'.format(pdrop_aorta))

    # Arteriolar:
    radius_arteriolar = 0.0015 # cm
    length_arteriolar = 0.5 # cm
    p_arteriolar_in = 100 # mmHg
    p_arteriolar_out = 50 # mmHg

    flow_arteriolar = cpt_flow_poseuille(p_arteriolar_in, p_arteriolar_out,
                                    radius_arteriolar, length_arteriolar,
                                    viscosity_blood)
    print('Flow arteriolar = {} ml.s-1'.format(flow_arteriolar))


if __name__ == '__main__':
    main()
