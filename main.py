import copy
import json
import logging
import os
from collections import defaultdict
from math import floor
from typing import Self, cast, List, Tuple
import csv
from datetime import datetime

import numpy as np
import pyrender
import scipy.interpolate
import tqdm
import trimesh
import pyrender as pyr
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata, CubicSpline, Akima1DInterpolator, PchipInterpolator
from sklearn.cluster import KMeans

import config
import task
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
    ctx = task.Context()
    femur_stl_path = 'acc_task/Femur.stl'
    femur_cart_stl_path = 'acc_task/Femur_Cart_Smooth.stl'
    tibia_stl_path = 'acc_task/Tibia.stl'
    tibia_cart_stl_path = 'acc_task/Tibia_Cart_Smooth.stl'

    task_femur_mesh = ctx.add_task('femur_mesh', lambda: trimesh.load_mesh(femur_stl_path))
    task_femur_cart_mesh = ctx.add_task('femur_cart_mesh', lambda: trimesh.load_mesh(femur_cart_stl_path))
    task_tibia_mesh = ctx.add_task('tibia_mesh', lambda: trimesh.load_mesh(tibia_stl_path))
    task_tibia_cart_mesh = ctx.add_task('tibia_cart_mesh', lambda: trimesh.load_mesh(tibia_cart_stl_path))
    task_extended_femur_mesh = ctx.add_task('extended_femur_mesh', lambda fm, fcm: remove_bubbles(fm.union(fcm)),
                                            deps=[task_femur_mesh, task_femur_cart_mesh])
    task_extended_tibia_mesh = ctx.add_task('extended_tibia_mesh', lambda tm, tcm: remove_bubbles(tm.union(tcm)),
                                            deps=[task_tibia_mesh, task_tibia_cart_mesh])

    task_all_extended_meshes = ctx.add_task('all_extended_meshes', lambda f, t: [f, t],
                                            deps=[task_extended_femur_mesh, task_extended_tibia_mesh])
    task_watertight_test_extended_meshes = ctx.add_task('watertight_test_extended_meshes',
                                                        lambda meshes: not any(not m.is_watertight for m in meshes),
                                                        deps=[task_all_extended_meshes])
    if not task_watertight_test_extended_meshes():
        raise ValueError(f'model is not watertight: {[m.is_watertight for m in task_all_extended_meshes()]}')

    ori_feature_point_path = 'acc_task/Coordination_Pt.txt'
    task_original_coordinates = ctx.add_task('original_coordinates',
                                             lambda: load_coord_from_file(ori_feature_point_path))
    task_original_coordinates_femur = ctx.add_task('original_coordinates_femur', lambda pair: pair[0],
                                                   deps=[task_original_coordinates])
    task_original_coordinates_tibia = ctx.add_task('original_coordinates_tibia', lambda pair: pair[1],
                                                   deps=[task_original_coordinates])

    task_extent = ctx.add_task('extent', calc_extent, deps=[task_extended_tibia_mesh, task_original_coordinates_tibia])
    task_exp_y_rotation = ctx.add_task('exp_y_rotation', exp_y_rotation,
                                       deps=[task_extended_femur_mesh, task_extended_tibia_mesh,
                                             task_original_coordinates_tibia])
    task_frame_bone_coordinates = ctx.add_task('frame_bone_coordinates', load_frame_bone_coordinates)
    task_frame_bone_transformations = ctx.add_task('frame_bone_transformations',
                                                   lambda bcs: [tuple(e.t for e in bc) for bc in bcs],
                                                   deps=[task_frame_bone_coordinates])
    task_frame_bone_transformations_femur = ctx.add_task('frame_bone_transformations_femur',
                                                         lambda pairs: [pair[0] for pair in pairs],
                                                         deps=[task_frame_bone_transformations])
    task_frame_bone_transformations_tibia = ctx.add_task('frame_bone_transformations_tibia',
                                                         lambda pairs: [pair[1] for pair in pairs],
                                                         deps=[task_frame_bone_transformations])
    task_frame_coordinates = ctx.add_task('frame_coordinates', calc_frame_coordinates,
                                          deps=[task_original_coordinates_tibia, task_frame_bone_transformations_tibia])
    task_frame_femur_meshes = ctx.add_task('frame_femur_meshes',
                                           lambda m, fts: [m.copy().apply_transform(ft.mat_homo) for ft in fts],
                                           deps=[task_femur_mesh, task_frame_bone_transformations_femur])
    task_frame_tibia_meshes = ctx.add_task('frame_tibia_meshes',
                                           lambda m, tts: [m.copy().apply_transform(tt.mat_homo) for tt in tts],
                                           deps=[task_tibia_mesh, task_frame_bone_transformations_tibia])
    task_frame_femur_cart_meshes = ctx.add_task('frame_femur_cart_meshes',
                                                lambda m, fts: [m.copy().apply_transform(ft.mat_homo) for ft in fts],
                                                deps=[task_femur_cart_mesh, task_frame_bone_transformations_femur])
    task_frame_tibia_cart_meshes = ctx.add_task('frame_tibia_cart_meshes',
                                                lambda m, tts: [m.copy().apply_transform(tt.mat_homo) for tt in tts],
                                                deps=[task_tibia_cart_mesh, task_frame_bone_transformations_tibia])
    task_frame_extended_femur_meshes = \
        ctx.add_task('frame_extended_femur_meshes',
                     lambda m, fts: [m.copy().apply_transform(ft.mat_homo) for ft in fts],
                     deps=[task_extended_femur_mesh, task_frame_bone_transformations_femur])
    task_frame_extended_tibia_meshes = \
        ctx.add_task('frame_extended_tibia_meshes',
                     lambda m, tts: [m.copy().apply_transform(tt.mat_homo) for tt in tts],
                     deps=[task_extended_tibia_mesh, task_frame_bone_transformations_tibia])

    task_bone_animation_frames = ctx.add_task('bone_animation_frames', gen_bone_animation_frames, deps=[
        task_frame_femur_meshes, task_frame_femur_cart_meshes, task_frame_tibia_meshes, task_frame_tibia_cart_meshes,
    ])
    task_bone_animation = ctx.add_task('bone_animation', gen_bone_animation, deps=[task_bone_animation_frames])

    def foo(fms, tms, coords):
        res = []
        for fm, tm, coord in zip(fms, tms, coords):
            print(fm, tm, coord)
            res.append(calc_bone_distance_map(fm, tm, coord))
        return res

    task_frame_bone_distance_maps = \
        ctx.add_task('frame_bone_distance_maps',
                     foo,
                     deps=[task_frame_extended_femur_meshes,
                           task_frame_extended_tibia_meshes,
                           task_frame_coordinates])
    task_frame_bone_distance_map_origins = ctx.add_task('frame_bone_distance_map_origins',
                                                        lambda pairs: [pair[1] for pair in pairs],
                                                        deps=[task_frame_bone_distance_maps])
    task_frame_bone_distance_map_distances = ctx.add_task('frame_bone_distance_map_distances',
                                                          lambda pairs: [pair[1] for pair in pairs],
                                                          deps=[task_frame_bone_distance_maps])

    task_frame_contact_areas = ctx.add_task('frame_contact_areas',
                                            lambda fms, tms: list(map(lambda fm, tm: fm.intersection(tm), fms, tms)),
                                            deps=[task_frame_extended_femur_meshes, task_frame_extended_tibia_meshes])
    task_frame_contact_components = ctx.add_task('frame_contact_components',
                                                 lambda cas: list(map(lambda ca: ca.split(), cas)),
                                                 deps=[task_frame_contact_areas])
    task_frame_contact_component_depth_maps = \
        ctx.add_task('frame_contact_component_depth_maps',
                     lambda coord, fcs: [[calc_contact_depth_map(c, coord) for c in cs] for cs in fcs],
                     deps=[task_frame_coordinates, task_frame_contact_components])
    task_frame_contact_component_depth_map_origins = \
        ctx.add_task('frame_frame_contact_component_depth_map_origins',
                     lambda dms: [[pair[0] for pair in dm] for dm in dms],
                     deps=[task_frame_contact_component_depth_maps])
    task_frame_contact_component_depth_map_depths = \
        ctx.add_task('frame_frame_contact_component_depth_map_depths',
                     lambda dms: [[pair[1] for pair in dm] for dm in dms],
                     deps=[task_frame_contact_component_depth_maps])

    task_max_depth_curve = ctx.add_task('max_depth_curve', plot_max_depth_curve,
                                        deps=[
                                            task_frame_contact_components,
                                            task_frame_contact_component_depth_maps,
                                            task_frame_coordinates
                                        ])
    task_dof_data = ctx.add_task('dof_data', calc_dof,
                                 deps=[
                                     task_original_coordinates_femur,
                                     task_original_coordinates_tibia,
                                     task_frame_bone_transformations_femur,
                                     task_frame_bone_transformations_tibia,
                                 ])
    task_plot_dof_curves = ctx.add_task('plot_dof_curves', plot_dof_curves, deps=[task_dof_data])
    task_interpolate_dof = ctx.add_task('interpolate_dof', interpolate_dof, deps=[task_dof_data])

    task_contact_depth_map_extent = \
        ctx.add_task('contact_depth_map_extent', gen_contact_depth_map_extent, deps=[task_extent])
    task_contact_depth_map_background = ctx.add_task('contact_depth_map_background', gen_contact_depth_map_background,
                                                     deps=[task_contact_depth_map_extent,
                                                           task_tibia_cart_mesh,
                                                           task_original_coordinates_tibia])
    task_contact_depth_map_frames = ctx.add_task('frame_contact_depth_map_frames', plot_contact_depth_maps,
                                                 deps=[
                                                     task_contact_depth_map_extent,
                                                     task_contact_depth_map_background,
                                                     task_frame_coordinates,
                                                     task_frame_bone_distance_map_origins,
                                                     task_frame_bone_distance_map_distances,
                                                     task_frame_contact_components,
                                                     task_frame_contact_component_depth_map_origins,
                                                     task_frame_contact_component_depth_map_depths,
                                                 ])
    task_contact_depth_map_animation = \
        ctx.add_task('frame_contact_depth_map_animation',
                     lambda frames: gen_animation(frames, 'output/depth_map_animation.gif'),
                     deps=[task_contact_depth_map_frames])
    if config.Y_ROTATE_EXP:
        task_exp_y_rotation()
    if config.GENERATE_ANIMATION:
        task_bone_animation()
    if config.GENERATE_DEPTH_CURVE:
        task_max_depth_curve()
    if config.GENERATE_DEPTH_MAP:
        task_contact_depth_map_animation()
    if config.GENERATE_DOF_CURVES:
        task_plot_dof_curves()
    if config.INTERPOLATE_DOF:
        task_interpolate_dof()


def gen_contact_depth_map_extent(extent):
    largest_side = np.max(np.abs(extent))
    extent = (-largest_side, largest_side, -largest_side, largest_side)
    return extent


def get_contact_depth_map_resolution():
    return 500, 500


def gen_contact_depth_map_background(contact_depth_map_extent, tibia_cart_mesh, tibia_coord):
    extent = contact_depth_map_extent
    largest_side = extent[1]
    res = get_contact_depth_map_resolution()
    bg = gen_orthographic_photo(tibia_cart_mesh.copy(), tibia_coord, res, largest_side, largest_side)
    bg = bg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return bg


def plot_contact_depth_maps(extent, background, frame_coordinates,
                            frame_bone_distance_map_origins, frame_bone_distance_map_distances,
                            task_frame_contact_components,
                            task_frame_contact_component_depth_map_origins,
                            task_frame_contact_component_depth_map_depths):
    def frame_path(index):
        return f'output/depth_map_frame_{index}.jpg'

    n = len(frame_bone_distance_map_distances)

    res = get_contact_depth_map_resolution()
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:res[0] * 1j, extent[2]:extent[3]:res[1] * 1j]
    distance_threshold = 10
    vmin, vmax = 1e9, -1e9
    for frame_index in range(n):
        distances = frame_bone_distance_map_distances[frame_index]
        distances = distances[(~np.isnan(distances)) & (distances < distance_threshold)]
        frame_contact_depths = task_frame_contact_component_depth_map_depths[frame_index]
        if len(frame_contact_depths) > 0:
            g_depth = frame_contact_depths
            g_depth = g_depth[~np.isnan(g_depth)]
            all_data = np.concatenate([-distances, g_depth])
        else:
            all_data = -distances
        vmax = max(np.max(all_data), vmax)
        vmin = min(np.min(all_data), vmin)

    frames = []
    for frame_index in range(n):
        coord = frame_coordinates[frame_index]
        origins = frame_bone_distance_map_origins[frame_index].astype(Real)
        distances = frame_bone_distance_map_distances[frame_index]
        mask = distances < distance_threshold
        origins = origins[mask]
        depths = -distances[mask]
        deepest = []

        for c_mesh, c_origins, c_depth in zip(
                task_frame_contact_components[frame_index],
                task_frame_contact_component_depth_map_origins[frame_index],
                task_frame_contact_component_depth_map_depths[frame_index]
        ):
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
        ax.imshow(background, extent=extent, interpolation='none', aspect='equal')
        im = ax.contourf(
            grid_x, grid_y, z,
            cmap=depth_cmap,
            alpha=0.5,
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Depth')
        if len(deepest) > 0:
            deepest = np.array(deepest)
            deepest_2d = coord.project(deepest)[:, :2]
            ax.scatter(deepest_2d[:, 0], deepest_2d[:, 1], marker='+', s=100, color='turquoise')
        image_path = frame_path(frame_index)
        fig.savefig(image_path)
        frames.append(Image.open(image_path))
    return frames


def gen_bone_animation(bone_animation_frames):
    gen_animation(bone_animation_frames, get_animation_path())


def gen_bone_animation_frames(frame_femur_meshes, frame_femur_cart_meshes, frame_tibia_meshes, frame_tibia_cart_meshes):
    assert (len(frame_femur_meshes) == len(frame_femur_cart_meshes)
            == len(frame_tibia_meshes) == len(frame_tibia_cart_meshes))
    num_frames = len(frame_femur_meshes)

    images = []
    for i in range(num_frames):
        tfm = frame_femur_meshes[i]
        tfcm = frame_femur_cart_meshes[i].copy()
        ttm = frame_tibia_meshes[i]
        ttcm = frame_tibia_cart_meshes[i].copy()
        tfcm.visual.vertex_colors = (200, 200, 100)
        ttcm.visual.vertex_colors = (200, 100, 200)
        scene = trimesh.Scene([tfm, tfcm, ttm, ttcm])

        image_data = scene.save_image(visible=False)
        image_path = get_animation_frame_path(i)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        image = Image.open(image_path)
        images.append(image)
    return images


def calc_frame_coordinates(original_coordinates_tibia, frame_transformations_tibia):
    res = []
    for tt in frame_transformations_tibia:
        coord = original_coordinates_tibia.copy()
        coord.t.apply_transformation(tt)
        res.append(coord)
    return res


def load_frame_bone_coordinates() -> list[tuple[BoneCoordination, BoneCoordination]]:
    with open(config.FRAME_RELATIVE_TRANSFORM_FILE, 'r') as fp:
        lines = list(filter(lambda l: l.startswith('['), fp.readlines()))
    frame_transformations = []
    for line_index, line in enumerate(lines[:2]):
        data = np.array(json.loads(line), dtype=Real)
        if line_index % 2 == 0:
            femur_transformation = BoneCoordination.from_translation_and_quat(data[:3], data[3:])
        else:
            tibia_transformation = BoneCoordination.from_translation_and_quat(data[:3], data[3:])
            frame_transformations.append((femur_transformation, tibia_transformation))
    return frame_transformations


def calc_extent(extended_tibia_mesh, tibia_coord, padding=5):
    proj_tm = extended_tibia_mesh.copy()
    proj_tm.vertices = tibia_coord.project(proj_tm.vertices)
    r, t = np.max(proj_tm.vertices[:, :2], axis=0)
    l, b = np.min(proj_tm.vertices[:, :2], axis=0)
    return [l - padding, r + padding, b - padding, t + padding]


def gen_orthographic_photo(mesh: trimesh.Trimesh, coord: BoneCoordination, res: Tuple[int, int], xmag: float,
                           ymag: float) -> Image.Image:
    mesh.vertices = coord.project(mesh.vertices)
    pyr_tibia_cart_mesh = pyr.Mesh.from_trimesh(mesh)
    pyr_scene = pyr.Scene()
    pyr_scene.add(pyr_tibia_cart_mesh)
    pyr_camera = pyrender.OrthographicCamera(xmag, ymag, znear=0.1, zfar=1e5)
    pyr_camera_pose = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, -500],
        [0, 0, 0, 1],
    ])
    pyr_scene.add(pyr_camera, pose=pyr_camera_pose)
    pyr_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    pyr_scene.add(pyr_light, pose=pyr_camera_pose)
    renderer = pyr.OffscreenRenderer(viewport_width=res[0], viewport_height=res[1])
    color, _ = renderer.render(pyr_scene)
    img = Image.fromarray(color)
    return img


def calc_dof(original_coordinates_femur, original_coordinates_tibia,
             frame_transformations_femur, frame_transformations_tibia):
    y_tx, y_ty, y_tz = [], [], []
    y_rx, y_ry, y_rz = [], [], []

    for index, (ft, tt) in enumerate(zip(frame_transformations_femur, frame_transformations_tibia)):
        fc = original_coordinates_femur.copy()
        tc = original_coordinates_tibia.copy()
        fc.t.apply_transformation(ft)
        tc.t.apply_transformation(tt)
        r = fc.t.relative_to(tc.t)
        tx, ty, tz = r.mat_t
        y_tx.append(tx), y_ty.append(ty), y_tz.append(tz)
        rot = Rotation.from_matrix(r.mat_r)
        rx, ry, rz = rot.as_euler('xyz', degrees=True)
        y_rx.append(rx), y_ry.append(ry), y_rz.append(rz)

    return np.array(y_tx), np.array(y_ty), np.array(y_tz), np.array(y_rx), np.array(y_ry), np.array(y_rz)


def plot_dof_curves(dof_data):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data
    x = np.arange(len(y_tx)) + 1

    fig, ax = plt.subplots()
    ax.plot(x, y_tx, 'x-')
    ax.set_title('Translation X')
    fig.savefig('output/dof_curve_tx.png')
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(x, y_ty, 'x-')
    ax.set_title('Translation Y')
    fig.savefig('output/dof_curve_ty.png')
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(x, y_tz, 'x-')
    ax.set_title('Translation Z')
    fig.savefig('output/dof_curve_tz.png')
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(x, y_rx, 'x-')
    ax.set_title('Euler Angle (x-y-z) X')
    fig.savefig('output/dof_curve_rx.png')
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(x, y_ry, 'x-')
    ax.set_title('Euler Angle (x-y-z) Y')
    fig.savefig('output/dof_curve_ry.png')
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(x, y_rz, 'x-')
    ax.set_title('Euler Angle (x-y-z) Z')
    fig.savefig('output/dof_curve_rz.png')
    fig.show()


def interpolate_dof(dof_data):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data
    n = len(y_tx)
    x = np.arange(n) + 1
    degrees_of_freedom = {
        'Translation X': y_tx,
        'Translation Y': y_ty,
        'Translation Z': y_tz,
        'Euler Angle (x-y-z) X': y_rx,
        'Euler Angle (x-y-z) Y': y_ry,
        'Euler Angle (x-y-z) Z': y_rz,
    }

    methods = {
        'Akima': Akima1DInterpolator,
        'Cubic Spline': CubicSpline,
        'Pchip': PchipInterpolator,
    }

    nx = 100
    start = 1
    stop = n
    step = (stop - start) / nx
    xi = np.arange(start, stop + 1e-6, step)

    dof_interpolate_data = {
        'x': xi,
    }
    for dof_name, y in degrees_of_freedom.items():
        fig, ax = plt.subplots()
        ax.plot(x, y, 'x-', label='Original')
        ax.set_title(dof_name)
        for method_name, cls in methods.items():
            interpolate = cls(x, y)
            yi = interpolate(xi)
            dof_interpolate_data[' - '.join([dof_name, method_name])] = yi
            ax.plot(xi, yi, label=method_name)
        ax.legend()
        fig.savefig('output/dof_curve_interpolated_{}.png'.format(dof_name))
        fig.show()

    scipy.io.savemat('output/dof_curve_interpolated.mat', dof_interpolate_data)


def exp_y_rotation(efcm, etcm, tibia_coord):
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


def plot_max_depth_curve(frame_contact_components, contact_component_depth_maps, frame_coordinates):
    n = len(frame_coordinates)
    mdms = []
    mdls = []
    mds = []
    for i in range(n):
        mdm, mdl = 0, 0
        coordinate = frame_coordinates[i]
        components = frame_contact_components[i]
        depth_maps = contact_component_depth_maps[i]
        for c, (_, depths) in zip(components, depth_maps):
            if coordinate.project(c.centroid)[0] < 0:
                mdm = max(mdm, np.max(depths))
            else:
                mdl = max(mdl, np.max(depths))
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
