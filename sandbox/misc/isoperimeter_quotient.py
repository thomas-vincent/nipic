import numpy as np

def ipq(surf):
    return 36 * np.pi * surf.volume**2 / surf.area**3

import pyvista as pv
pv.global_theme.transparent_background = True

sphere = pv.Sphere()
pl = pv.Plotter(off_screen=True)
pl.add_mesh(sphere, show_edges=True,
            show_scalar_bar=False)
print('Sphere IPQ =', ipq(sphere))
pl.show(screenshot='sphere.png')

pl = pv.Plotter(off_screen=True)
cone = pv.Cone()
print('Cone IPQ =', ipq(cone))
pl.add_mesh(cone, show_edges=True,
            show_scalar_bar=False)
pl.show(screenshot='cone.png')

cylinder = pv.Cylinder(center=[1, 2, 3], direction=[1, 1, 1],
                       radius=1, height=2)
pl = pv.Plotter(off_screen=True)
print('Cylinder h2 IPQ =', ipq(cylinder))
pl.add_mesh(cylinder, show_edges=True,
            show_scalar_bar=False)
pl.show(screenshot='cylinder_h2.png', cpos='xy')

cylinder = pv.Cylinder(center=[1, 2, 3], direction=[1, 1, 1],
                       radius=1, height=10)
print('Cylinder h10 IPQ =', ipq(cylinder))
pl = pv.Plotter(off_screen=True)
pl.add_mesh(cylinder, show_edges=True,
            show_scalar_bar=False)
pl.show(screenshot='cylinder_h10.png', cpos='xy')

from pyvista import examples
teapot = examples.download_teapot()
print('teapot IPQ =', ipq(teapot))
pl = pv.Plotter(off_screen=True)
pl.add_mesh(teapot, show_edges=True,
            show_scalar_bar=False)
pl.show(screenshot='teapot.png', cpos='xy')


