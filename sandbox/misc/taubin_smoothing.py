
"""
So: 1) Gaussian Filter applied to the 3D array (scipy.filters) 2) Marching Cubes algorithm to mesh it (scikit-image tool) 3) Sum up the areas of triangles (scikit-image tool)
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np


from scipy.spatial.distance import pdist
from nipic.freesurfer import voxels_corners, set_axes_equal

def measure_object(label, dims):
    label = np.lib.pad(label, ((1, 1), (1, 1), (1, 1)))
    zy_face_area = dims[1] * dims[2]
    xz_face_area = dims[0] * dims[2]
    xy_face_area = dims[0] * dims[1]
    area = 0.0
    bg = np.bitwise_not(label).astype(np.uint8)
    for i,j,k in np.array(np.where(label)).T:
        area += (zy_face_area * (bg[i-1, j, k] + bg[i+1, j, k]) +
                 xz_face_area * (bg[i, j-1, k] + bg[i, j+1, k]) +
                 xy_face_area * (bg[i, j, k-1] + bg[i, j, k+1]))
    volume = label.sum() * np.prod(dims)
    isoperimeter_quotient = 36 * np.pi * volume**2 / area**3
    length = np.max(pdist(voxels_corners(label, dims)))
    return area, volume, length, isoperimeter_quotient

import pyvista as pv

def measure_smooth_object(label, dims):
    # label = np.lib.pad(label, ((2, 2), (2, 2), (2, 2)))

    grid = pv.UniformGrid()
    grid.dimensions = np.array(label.shape) + 1
    grid.spacing = dims
    grid.cell_data["values"] = label.flatten(order="F")
    grid.plot(show_edges=True)

    volume = np.abs(mesh.volume)
    area = mesh.area

    isoperimeter_quotient = 36 * np.pi * volume**2 / area**3
    length = np.max(pdist(v))

    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.voxels(label)
        # #ax.scatter(*cluster_vertices.T)
        mesh = Poly3DCollection(v[f])
        mesh.set_edgecolor('k')
        mesh.set_facecolor('r')
        ax.add_collection3d(mesh)
        ax.set_box_aspect((1, 1, 1))
        set_axes_equal(ax)
        plt.show()

    return area, volume, length, isoperimeter_quotient


def rastered_sphere(radius):
    size = radius*2 + 1
    x0, y0, z0 = (radius, radius, radius)
    x, y, z = np.mgrid[0:size:1, 0:size:1, 0:size:1]
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    r[r > radius] = 0
    return r.astype(bool)


cluster = rastered_sphere(3)

dims = (1,1,1)

import scipy.ndimage as ndi

os_factor = 2
cluster_os = (cluster
              .repeat(os_factor, axis=0)
              .repeat(os_factor, axis=1)
              .repeat(os_factor, axis=2))
dims_os = [d/os_factor for d in dims]

grid = pv.UniformGrid()
grid.dimensions = np.array(cluster_os.shape) + 1
grid.spacing = dims_os
grid.cell_data["values"] = cluster_os.flatten(order="F")

vol = grid.threshold(0.1)
surf = vol.extract_geometry()
orig_edges = surf.extract_feature_edges()
smooth_w_taubin = surf.smooth_taubin()

pl = pv.Plotter()
pl.add_mesh(smooth_w_taubin, show_edges=True, show_scalar_bar=False)
pl.add_mesh(orig_edges, show_scalar_bar=False, color='k', line_width=2)
pl.show()

# output the volumes of the original and smoothed meshes
print(f'Original surface volume:   {surf.volume:.1f}')
print(f'Taubin smoothed volume:    {smooth_w_taubin.volume:.1f}')

voxel_size = (1,1,1)
area, volume, length, isopq = measure_object(cluster, voxel_size)
print('area=%1.3f' % area)
print('volume=%1.3f' % volume)
print('isopq=%1.3f' % isopq)
print('length=%1.3f' % length)


print('---------')
print('Smooth version')
print('---------')
area, volume, length, isopq = measure_smooth_object(cluster, voxel_size)
print('area=%1.3f' % area)
print('volume=%1.3f' % volume)
print('isopq=%1.3f' % isopq)
print('length=%1.3f' % length)


plt.show()
