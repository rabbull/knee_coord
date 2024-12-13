import copy
import logging
from math import floor
from typing import Self, cast, List, Tuple
import csv
from datetime import datetime

import numpy as np
import tqdm
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from sklearn.cluster import KMeans

import config
from utils import *

trimesh.util.attach_to_log(level=logging.INFO)

WORLD_AXIS = trimesh.creation.axis(axis_length=100, axis_radius=2)

depth_cmap = LinearSegmentedColormap.from_list("depth_map", ['blue', 'green', 'yellow', 'red', ])


def get_animation_frame_path(index: int):
    return f'output/animation_frame_{index}.png'


def get_contact_frame_path(index: int):
    return f'output/contact_frame_{index}.png'


def get_animation_path():
    return 'output/animation.gif'


def get_contact_animation_path():
    return 'output/contact.gif'


class BoneCoordination:
    def __init__(self):
        self._t = Transformation3D()
        self._extra = {}

    @classmethod
    def from_feature_points(cls, medial_point, lateral_point, proximal_point, distal_point, extra=None) -> Self:
        self = cls()

        self._origin = (lateral_point + medial_point) / 2
        self._t.set_translation(self._origin)

        raw_x = lateral_point - medial_point
        unit_x = normalize(raw_x)
        raw_z = proximal_point - distal_point
        raw_z_proj_x = np.dot(raw_z, unit_x) * unit_x
        fixed_z = raw_z - raw_z_proj_x
        unit_z = normalize(fixed_z)
        raw_y = np.cross(unit_z, unit_x)
        unit_y = normalize(raw_y)
        self._t.set_rotation(Rotation.from_matrix(
            np.column_stack((unit_x, unit_y, unit_z))))

        self._extra = extra if extra else {}
        return self

    @classmethod
    def from_translation_and_quat(cls, translation, quat, extra=None) -> Self:
        self = cls()

        self._t = Transformation3D()
        self._t.set_translation(translation)
        self._t.set_rotation(Rotation.from_quat(quat))

        self._extra = extra if extra else {}
        return self

    @property
    def t(self) -> Transformation3D:
        return self._t

    def project(self, x):
        return self.t.inverse().transform(x)

    def copy(self):
        return copy.deepcopy(self)


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

    fm, fcm = femur_mesh, femur_cart_mesh
    efcm = remove_bubbles(fm.union(fcm))
    tm, tcm = tibia_mesh, tibia_cart_mesh
    etcm = remove_bubbles(tm.union(tcm))

    if any(not m.is_watertight for m in [efcm, etcm]):
        raise ValueError(f'model is not watertight: {[m.is_watertight for m in [efcm, etcm]]}')

    ori_feature_point_path = 'acc_task/Coordination_Pt.txt'
    ori_femur_coord, ori_tibia_coord = \
        load_coord_from_file(ori_feature_point_path)

    # calculate extent of the plotting
    proj_tm = etcm.copy()
    proj_tm.vertices = ori_tibia_coord.project(proj_tm.vertices)
    r, t = np.max(proj_tm.vertices[:, :2], axis=0)
    l, b = np.min(proj_tm.vertices[:, :2], axis=0)
    padding = 5
    extent = [l - padding, r + padding, b - padding, t + padding]

    if config.Y_ROTATE_EXP:
        rotate_y(efcm, etcm, ori_tibia_coord)

    coord_series_csv_path = 'Bill_Kinematic.csv'
    transformations = read_bone_transformation_series(coord_series_csv_path)
    plot_data_list = []
    for i, (ft, tt) in enumerate(tqdm.tqdm(transformations)):
        plot_data = {
            'femur_transform': ft,
            'tibia_transform': tt,
        }
        plot_data_list.append(plot_data)

        coord = ori_tibia_coord.copy()
        coord.t.apply_transformation(tt)
        plot_data['coordination'] = coord

        if config.GENERATE_ANIMATION or config.GENERATE_DEPTH_MAP or config.GENERATE_DEPTH_CURVE:
            tufm = efcm.copy().apply_transform(ft.mat_homo)
            tfm = fm.copy().apply_transform(ft.mat_homo)
            tfcm = fcm.copy().apply_transform(ft.mat_homo)
            tutm = etcm.copy().apply_transform(tt.mat_homo)
            ttm = tm.copy().apply_transform(tt.mat_homo)
            ttcm = tcm.copy().apply_transform(tt.mat_homo)
            plot_data['transformed_femur_mesh'] = tufm
            plot_data['transformed_tibia_mesh'] = tutm

        if config.GENERATE_ANIMATION:
            tfcm.visual.vertex_colors = (200, 200, 100)
            ttcm.visual.vertex_colors = (200, 100, 200)
            img_path = get_animation_frame_path(i)
            img_data = trimesh.Scene([tfm, tfcm, ttm, ttcm]).save_image(visible=False)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            img = Image.open(img_path)
            plot_data['animation_frame'] = img

        if config.GENERATE_DEPTH_MAP:
            origins, distances = calc_bone_distance_map(tfcm, ttcm, coord)
            plot_data['bone_distance_map'] = {
                'origins': origins,
                'distances': distances,
            }

        if config.GENERATE_DEPTH_CURVE or config.GENERATE_DEPTH_MAP:
            contact_area = tufm.intersection(tutm)
            components = contact_area.split()

            contact_components = []
            for component in components:
                if component is None:
                    continue
                origins, depths = calc_contact_depth_map(component, coord)
                contact_components.append({
                    'component': component,
                    'origins': origins,
                    'depths': depths,
                })
            plot_data['components'] = contact_components

    if config.GENERATE_ANIMATION:
        gen_animation(
            [d['animation_frame'] for d in plot_data_list], get_animation_path()
        )
    if config.GENERATE_DEPTH_CURVE:
        plot_max_depth_curve(plot_data_list)
    if config.GENERATE_DEPTH_MAP:
        plot_contact_depth_maps(extent, plot_data_list)
    if config.GENERATE_DOF_CURVES:
        plot_dof_curves(ori_femur_coord, ori_tibia_coord, plot_data_list)


def plot_dof_curves(ofc, otc, plot_data_list):
    x = []
    y_tx, y_ty, y_tz = [], [], []
    y_rx, y_ry, y_rz = [], [], []

    for index, pd in enumerate(plot_data_list):
        x.append(index + 1)
        ft, tt = pd['femur_transform'], pd['tibia_transform']
        fc = ofc.copy()
        tc = otc.copy()
        fc.t.apply_transformation(ft)
        tc.t.apply_transformation(tt)
        r = fc.t.relative_to(tc.t)
        tx, ty, tz = r.mat_t
        y_tx.append(tx), y_ty.append(ty), y_tz.append(tz)
        rot = Rotation.from_matrix(r.mat_r)
        rx, ry, rz = rot.as_euler('xyz', degrees=True)
        y_rx.append(rx), y_ry.append(ry), y_rz.append(rz)

    fig, ax = plt.subplots()
    ax.plot(x, y_tx, 'x-')
    ax.set_title('Translation X')
    fig.savefig('output/dof_curve_tx.png')
    fig, ax = plt.subplots()
    ax.plot(x, y_ty, 'x-')
    ax.set_title('Translation Y')
    fig.savefig('output/dof_curve_ty.png')
    fig, ax = plt.subplots()
    ax.plot(x, y_tz, 'x-')
    ax.set_title('Translation Z')
    fig.savefig('output/dof_curve_tz.png')

    fig, ax = plt.subplots()
    ax.plot(x, y_tz, 'x-')
    ax.set_title('Translation')
    fig.savefig('output/dof_curve_translation.png')

    fig, ax = plt.subplots()
    ax.plot(x, y_rx, 'x-')
    ax.set_title('Euler Angle (x-y-z) X')
    fig.savefig('output/dof_curve_rx.png')
    fig, ax = plt.subplots()
    ax.plot(x, y_ry, 'x-')
    ax.set_title('Euler Angle (x-y-z) Y')
    fig.savefig('output/dof_curve_ry.png')
    fig, ax = plt.subplots()
    ax.plot(x, y_rz, 'x-')
    ax.set_title('Euler Angle (x-y-z) Z')
    fig.savefig('output/dof_curve_rz.png')


def rotate_y(efcm, etcm, tibia_coord):
    x = np.arange(-10, 10.1, 5)
    ax = trimesh.creation.axis(axis_length=20, axis_radius=0.1)
    ym = []
    yl = []
    scenes_y = []
    scenes_z = []
    for deg in x:
        arc = deg / 180 * np.pi
        t = etcm.copy()
        t.vertices = tibia_coord.project(t.vertices)
        f = efcm.copy()
        f.vertices = tibia_coord.project(f.vertices)
        f = f.apply_transform(Transformation3D().apply_rotation(Rotation.from_euler('y', arc)).mat_homo)

        cs = f.intersection(t).split()
        scene = trimesh.Scene([ax] + cs)
        scene.set_camera(center=(0, 0, 0), distance=96)
        img_data = scene.save_image(visible=False)
        img_path = f'output/rotate_{deg:.1f}_deg_z.png'
        with open(img_path, 'wb') as f:
            f.write(img_data)
        scenes_z.append(Image.open(img_path))

        scene = trimesh.Scene([ax] + cs)
        scene.set_camera(center=(0, 0, 0), distance=96, angles=(np.pi / 2, 0, 0))
        img_data = scene.save_image(visible=False)
        img_path = f'output/rotate_{deg:.1f}_deg_y.png'
        with open(img_path, 'wb') as f:
            f.write(img_data)
        scenes_y.append(Image.open(img_path))

        mds = [0]
        lds = [0]
        for c in cs:
            _, depths = calc_contact_depth_map(c, tibia_coord)
            if c.centroid[0] < 0:
                mds += list(depths)
            else:
                lds += list(depths)
        ym.append(max(mds))
        yl.append(max(lds))

        print(deg, max(mds), max(lds))

    plt.plot(x, ym, 'x-', label='Medial')
    plt.plot(x, yl, 'x-', label='Lateral')
    plt.xlabel('Degree')
    plt.ylabel('Max Depth')
    plt.legend()
    plt.show()

    gen_animation(scenes_z, 'output/rotation_animation_z.gif')
    gen_animation(scenes_y, 'output/rotation_animation_y.gif')


def plot_max_depth_curve(plot_data_list):
    n = len(plot_data_list)
    mdms = []
    mdls = []
    mds = []
    for pd in plot_data_list:
        mdm, mdl = 0, 0
        for c in pd['components']:
            if pd['coordination'].project(c['component'].centroid)[0] < 0:
                mdm = max(mdm, np.max(c['depths']))
            else:
                mdl = max(mdl, np.max(c['depths']))
        mdms.append(mdm)
        mdls.append(mdl)
        mds.append(max(mdm, mdl))

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mds)
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mdms, label='Medial')
    ax.plot(np.arange(n), mdls, label='Lateral')
    ax.legend()
    fig.show()


def plot_contact_depth_maps(extent, plot_data_list):
    def frame_path(index):
        return f'output/depth_map_frame_{index}.jpg'

    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:500j, extent[2]:extent[3]:500j]

    distance_threshold = 10
    vmin, vmax = 1e9, -1e9
    for pd in plot_data_list:
        distances = pd['bone_distance_map']['distances']
        distances = distances[(~np.isnan(distances)) & (distances < distance_threshold)]
        g_depth = np.concatenate([c['depths'] for c in pd['components']])
        g_depth = g_depth[~np.isnan(g_depth)]
        all_data = np.concatenate([-distances, g_depth])
        vmax = max(np.max(all_data), vmax)
        vmin = min(np.min(all_data), vmin)

    for frame_index, pd in enumerate(plot_data_list):
        coord = pd['coordination']
        origins = pd['bone_distance_map']['origins'].astype(Real)
        distances = pd['bone_distance_map']['distances']
        mask = distances < distance_threshold
        origins = origins[mask]
        depths = -distances[mask]
        deepest = []

        for c in pd['components']:
            c_origins = c['origins'].astype(Real)
            c_depth = c['depths']
            c_mesh = c['component']
            vertices = c_mesh.vertices.astype(Real)

            deepest.append(c_origins[np.argmax(c_depth)])

            s_origins = (np.round(origins, decimals=3) * 1e4).astype(np.int64)
            s_vertices = (np.round(vertices, decimals=3) * 1e4).astype(np.int64)
            intersect = np.intersect1d(s_origins, s_vertices)
            keep = np.all(~np.isin(s_origins, intersect), axis=1)
            origins = origins[keep]
            depths = depths[keep]

            origins = np.vstack([origins, c_origins])
            depths = np.concatenate([depths, c_depth])

        labels = KMeans(n_clusters=2, random_state=42).fit_predict(origins)
        groups = [labels == j for j in range(2)]

        g_origins = [origins[grp_mask] for grp_mask in groups]
        g_origins_2d = [coord.project(origins)[:, :2] for origins in g_origins]
        g_depth = [depths[grp_mask] for grp_mask in groups]
        g_z = [griddata(g_origins_2d[i], g_depth[i], (grid_x, grid_y), method='linear') for i in range(2)]
        z = np.where(np.isnan(g_z[0]), g_z[1], g_z[0])
        z = np.where(np.isnan(z), g_z[0], z)

        # depth map
        fig, ax = plt.subplots()
        fig.suptitle(f'Depth Map - Frame {frame_index}')
        im = ax.contourf(grid_x, grid_y, z, levels=int(floor((vmax - vmin))), cmap=depth_cmap)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Depth')
        if len(deepest) > 0:
            deepest = np.array(deepest)
            deepest_2d = coord.project(deepest)[:, :2]
            ax.scatter(deepest_2d[:, 0], deepest_2d[:, 1], marker='+', s=100, color='turquoise')
        fig.savefig(frame_path(frame_index))

        if frame_index == 1:
            mesh = pd['transformed_tibia_mesh']
            mesh: trimesh.Trimesh
            vertices = mesh.vertices
            coord.project(vertices)

            r_vertices = (np.round(vertices, decimals=3) * 1e4).astype(np.int64)
            r_origins = (np.round(origins, decimals=3) * 1e4).astype(np.int64)
            intersection = np.intersect1d(r_origins, r_vertices)
            matched = np.all(np.isin(r_vertices, intersection), axis=1)


    # generate gif
    gen_animation([Image.open(frame_path(i)) for i in range(len(plot_data_list))], 'output/depth_map_animation.gif')


def gen_animation(frames, output_path, fps: float = 5):
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )


def load_coord_from_file(path):
    coord_points = {}
    with open(path, 'r') as f:
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
    femur_coord = BoneCoordination.from_feature_points(femur_medial_point,
                                                       femur_lateral_point,
                                                       femur_distal_point,
                                                       femur_proximal_point)
    tibia_coord = BoneCoordination.from_feature_points(tibia_medial_point,
                                                       tibia_lateral_point,
                                                       tibia_distal_point,
                                                       tibia_proximal_point)
    return femur_coord, tibia_coord


def calc_contact_depth_map(contact_component, coord):
    origins, directions = prepare_rays_from_model(contact_component, -coord.t.unit_z, True)
    locations, ray_indices, _ = \
        contact_component.ray.intersects_location(origins, directions)
    if len(ray_indices) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def calc_bone_distance_map(fm, tm, coord):
    origins, directions = prepare_rays_from_model(tm, -coord.t.unit_z)
    locations, ray_indices, _ = \
        fm.ray.intersects_location(origins, directions, multiple_hits=False)
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def prepare_rays_from_model(model, direction, reverse=False):
    ud = normalize(direction)
    mask = np.dot(model.vertex_normals, ud) > 0
    origins = model.vertices[mask]
    eps = 1e-4
    if reverse:
        ud = -ud
    origins += ud * eps
    directions = np.tile(ud, (len(origins), 1))
    return origins, directions


def calc_and_draw_contact(fm, tm, tc: BoneCoordination, output_path):
    contact_area = fm.intersection(tm)
    components = contact_area.split()
    assert len(components) <= 2

    d = {'Medial': None, 'Lateral': None}
    for component in components:
        if tc.t.transform(component.centroid)[0] < 0:
            d['Medial'] = component
        else:
            d['Lateral'] = component
    components = {f'{index}': component for index,
    component in enumerate(components)}

    fig, ax = plt.subplots(figsize=(10, 8))
    all_data = {}
    for name, component in components.items():
        if component is None:
            continue
        origins, distances = calc_contact_depth_map(component, tc)
        points_2d = (tc.t.transform(origins))[:, :2]
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

    for name, data in all_data.items():
        ax.imshow(data['grid_z'].T, extent=(min(data['x']), max(data['x']), min(
            data['y']), max(data['y'])), origin='lower', cmap='viridis')
        ax.scatter(*data['max_point'], marker='x',
                   color='red', label=f'{name} Max Depth')

    fig.colorbar(ax.images[0], ax=ax, label='Depth', shrink=0.4)
    ax.set_title('Medial/Lateral Compartment Cartilage Contact')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def read_bone_transformation_series(path) -> List[Tuple[Transformation3D, Transformation3D]]:
    rows = {}

    def get_row_id(r):
        return int(r['measurement_id']), int(r['anatomy_id'])

    def get_row_key(r):
        fmt = '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(r['update_timestamp'], fmt)

    with open(path, 'r') as fp:
        for row in csv.DictReader(fp):
            row_id = get_row_id(row)
            if row_id not in rows or get_row_key(rows[row_id]) < get_row_key(row):
                rows[row_id] = row
    rows = sorted(rows.values(), key=get_row_id)
    assert len(rows) % 2 == 0, f'Invalid Data: {rows}'

    res = []
    for i in range(len(rows) // 2):
        femur = rows[i * 2]
        tibia = rows[i * 2 + 1]
        res.append(tuple(
            BoneCoordination.from_translation_and_quat(
                translation=get_real_numbers_from_dict(r, ['tx', 'ty', 'tz']),
                quat=get_real_numbers_from_dict(r, ['r0', 'r1', 'r2', 'r3']),
                extra={
                    'original_data': r,
                },
            ).t for r in [femur, tibia]
        ))
    return res


if __name__ == '__main__':
    main()
