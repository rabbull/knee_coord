from typing import cast, AnyStr, Dict

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
from scipy.interpolate import griddata
from trimesh import Trimesh

from utils import *

trimesh.util.attach_to_log()


class BoneCoordination:
    def __init__(self, medial_point, lateral_point, proximal_point, distal_point):
        self._t = Transformation3D()

        self._origin = (lateral_point + medial_point) / 2
        self._t.set_translation(self._origin)

        raw_x = lateral_point - medial_point
        self._unit_x = normalize(raw_x)
        raw_z = proximal_point - distal_point
        raw_z_proj_x = np.dot(raw_z, self._unit_x) * self._unit_x
        fixed_z = raw_z - raw_z_proj_x
        self._unit_z = normalize(fixed_z)
        raw_y = np.cross(self._unit_z, self._unit_x)
        self._unit_y = normalize(raw_y)
        self._t.set_rotation(Rotation.from_matrix(
            np.column_stack((self._unit_x, self._unit_y, self._unit_z))))

    def transform(self, x):
        return self._t.transform(x)

    @property
    def unit_x(self):
        return self._unit_x

    @property
    def unit_y(self):
        return self._unit_y

    @property
    def unit_z(self):
        return self._unit_z

    @property
    def origin(self):
        return self._origin


def main():
    femur_stl_path = 'acc_task/Femur.stl'
    femur_cart_stl_path = 'acc_task/Femur_Cart_Smooth.stl'
    tibia_stl_path = 'acc_task/Tibia.stl'
    tibia_cart_stl_path = 'acc_task/Tibia_Cart_Smooth.stl'

    femur_mesh = cast(Trimesh, trimesh.load_mesh(femur_stl_path))
    femur_cart_mesh = cast(Trimesh, trimesh.load_mesh(femur_cart_stl_path))
    tibia_mesh = cast(Trimesh, trimesh.load_mesh(tibia_stl_path))
    tibia_cart_mesh = cast(Trimesh, trimesh.load_mesh(tibia_cart_stl_path))
    all_raw_meshes = [femur_mesh, femur_cart_mesh, tibia_mesh, tibia_cart_mesh]
    for m in all_raw_meshes:
        if not m.is_watertight:
            raise ValueError(f'{m} is not watertight')

    coord_points_path = 'acc_task/Coordination_Pt.txt'
    coord_points = {}
    with open(coord_points_path, 'r') as f:
        for line in f.readlines():
            key, val_s = tuple(line.split())
            x, y, z = tuple(val_s.split(','))
            coord_points[key] = np.array([float(x), float(y), float(z)])
    femur_medial_point = coord_points['Femur_Medial']
    femur_lateral_point = coord_points['Femur_Lateral']
    femur_distal_point = coord_points['Femur_Distal']
    femur_proximal_point = coord_points['Femur_Proximal']
    tibia_medial_point = coord_points['Tibia_Medial']
    tibia_lateral_point = coord_points['Tibia_Lateral']
    tibia_distal_point = coord_points['Tibia_Distal']
    tibia_proximal_point = coord_points['Tibia_Proximal']
    femur_coord = BoneCoordination(femur_medial_point, 
                                   femur_lateral_point, 
                                   femur_distal_point, 
                                   femur_proximal_point)
    tibia_coord = BoneCoordination(tibia_medial_point, 
                                   tibia_lateral_point, 
                                   tibia_distal_point, 
                                   tibia_proximal_point)

    coord = tibia_coord

    fm = remove_bubbles(
        femur_mesh.union(femur_cart_mesh))  # femur mesh, with cartilage
    tm = remove_bubbles(
        tibia_mesh.union(tibia_cart_mesh))  # tibia mesh, with cartilage
    for m in [fm, tm]:
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        if not m.is_watertight:
            raise ValueError(f'{m} is not watertight')

    # use intersections between whole bones instead of cartilages to prevent piercing
    contact_area = fm.intersection(tm)

    components = contact_area.split()
    assert len(components) == 2
    components.sort(key=lambda c: coord.transform(c.centroid)[0])
    components = dict(zip(['Medial', 'Lateral'], components))

    fig, ax = plt.subplots(figsize=(10, 8))
    all_data = {}
    for name, component in components.items():
        origins = component.vertices[np.dot(
            component.vertex_normals, coord.unit_z) < 0]
        eps = 0.001
        direction = coord.unit_z
        origins += eps * direction
        directions = np.tile(direction, (len(origins), 1))
        locations, ray_indices, _ = component.ray.intersects_location(
            origins, directions)

        origins = origins[ray_indices]
        distances = np.linalg.norm(locations - origins, axis=1)

        points_2d = (coord.transform(origins))[:, :2]
        x = points_2d[:, 0]
        y = points_2d[:, 1]

        grid_x, grid_y = np.mgrid[min(x):max(x):200j, min(y):max(y):200j]
        grid_z = griddata(points_2d, distances,
                          (grid_x, grid_y), method='linear')

        all_data[name] = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'x': x,
            'y': y,
            'distances': distances,
            'max_point': points_2d[np.argmax(distances)],
            'max_distance': max(distances)
        }

    # 绘制合并图像
    for name, data in all_data.items():
        ax.imshow(data['grid_z'].T, extent=(min(data['x']), max(data['x']), min(
            data['y']), max(data['y'])), origin='lower', cmap='viridis')
        ax.scatter(*data['max_point'], marker='x',
                   color='red', label=f'{name} Max Thickness')

    fig.colorbar(ax.images[0], ax=ax, label='Thickness', shrink=0.5)
    ax.set_title('Combined Thickness of Contact Areas')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    plt.tight_layout()
    plt.savefig('./output/test.png', dpi=300)
    plt.close()


def remove_bubbles(mesh: Trimesh) -> Trimesh:
    components = mesh.split()
    largest = None
    for component in components:
        if largest is None or component.volume > largest.volume:
            largest = component
    return largest


if __name__ == '__main__':
    main()
