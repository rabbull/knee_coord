import copy
import json
import logging
import os
from typing import Self, List, Tuple, Optional, Sequence
import csv
from datetime import datetime

import numpy as np
import scipy.interpolate
import trimesh
import pyrender as pyr
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata, CubicSpline, Akima1DInterpolator, PchipInterpolator
from scipy.linalg import svd
from sklearn.cluster import KMeans

import config
import task
from utils import *

trimesh.util.attach_to_log(level=logging.INFO)

WORLD_AXIS = my_axis(axis_length=200, axis_radius=2)

depth_cmap = LinearSegmentedColormap.from_list(
    "depth_map", ['blue', 'green', 'yellow', 'red', ])


def get_frame_output_directory(index):
    return os.path.join(config.OUTPUT_DIRECTORY, f'frame_{index}')


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

    task_femur_mesh = ctx.add_task(
        'femur_mesh', lambda: load_mesh(config.FEMUR_MODEL_FILE))
    task_tibia_mesh = ctx.add_task(
        'tibia_mesh', lambda: load_mesh(config.TIBIA_MODEL_FILE))
    task_femur_cart_mesh = ctx.add_task(
        'femur_cart_mesh', lambda: load_mesh(config.FEMUR_CARTILAGE_MODEL_FILE))
    task_tibia_cart_mesh = ctx.add_task(
        'tibia_cart_mesh', lambda: load_mesh(config.TIBIA_CARTILAGE_MODEL_FILE))

    task_extended_femur_mesh = \
        ctx.add_task('extended_femur_mesh', mesh_union, deps=[
                     task_femur_mesh, task_femur_cart_mesh])
    task_extended_tibia_mesh = \
        ctx.add_task('extended_tibia_mesh', mesh_union, deps=[
                     task_tibia_mesh, task_tibia_cart_mesh])

    # check waterproof
    task_all_extended_meshes = ctx.add_task('all_extended_meshes', lambda f, t: [f, t],
                                            deps=[task_extended_femur_mesh, task_extended_tibia_mesh])
    task_watertight_test_extended_meshes = ctx.add_task('watertight_test_extended_meshes',
                                                        lambda meshes: not any(
                                                            not m.is_watertight for m in meshes),
                                                        deps=[task_all_extended_meshes])
    if not task_watertight_test_extended_meshes():
        raise ValueError(
            f'model is not watertight: {[m.is_watertight for m in task_all_extended_meshes()]}')

    task_original_coordinates = ctx.add_task('original_coordinates',
                                             lambda: load_coord_from_file(config.FEATURE_POINT_FILE))
    task_original_coordinates_femur = \
        ctx.add_task('original_coordinates_femur', take_kth(0),
                     deps=[task_original_coordinates])
    task_original_coordinates_tibia = \
        ctx.add_task('original_coordinates_tibia', take_kth(1),
                     deps=[task_original_coordinates])

    task_extent = ctx.add_task('extent', calc_extent, deps=[
                               task_extended_tibia_mesh, task_original_coordinates_tibia])
    task_contact_depth_map_extent = \
        ctx.add_task('contact_depth_map_extent',
                     gen_contact_depth_map_extent, deps=[task_extent])
    task_contact_depth_map_background_femur = ctx.add_task('contact_depth_map_background_femur',
                                                           gen_contact_depth_map_background,
                                                           deps=[
                                                               task_contact_depth_map_extent,
                                                               task_femur_mesh,
                                                               task_original_coordinates_femur,
                                                               task_femur_cart_mesh,
                                                           ])
    task_contact_depth_map_background_tibia = ctx.add_task('contact_depth_map_background_tibia',
                                                           gen_contact_depth_map_background,
                                                           deps=[
                                                               task_contact_depth_map_extent,
                                                               task_tibia_mesh,
                                                               task_original_coordinates_tibia,
                                                               task_tibia_cart_mesh,
                                                           ])

    task_exp_y_rotation = ctx.add_task('exp_y_rotation', exp_y_rotation,
                                       deps=[task_extended_femur_mesh, task_extended_tibia_mesh,
                                             task_original_coordinates_tibia])

    if config.MOVEMENT_DATA_FORMAT == config.MomentDataFormat.CSV:
        task_frame_bone_coordinates = ctx.add_task(
            'frame_bone_coordinates', load_frame_bone_coordinates_csv)
    elif config.MOVEMENT_DATA_FORMAT == config.MomentDataFormat.JSON:
        task_frame_bone_coordinates = ctx.add_task(
            'frame_bone_coordinates', load_frame_bone_coordinates_raw)
    else:
        raise NotImplementedError(
            f'Unknown MOVEMENT_DATA_FORMAT: {config.MOVEMENT_DATA_FORMAT}')

    task_frame_bone_transformations = \
        ctx.add_task('frame_bone_transformations',
                     lambda bcs: [tuple(e.t for e in bc) for bc in bcs],
                     deps=[task_frame_bone_coordinates])
    task_frame_bone_transformations_femur = \
        ctx.add_task('frame_bone_transformations_femur', list_take_kth(
            0), deps=[task_frame_bone_transformations])
    task_frame_bone_transformations_tibia = \
        ctx.add_task('frame_bone_transformations_tibia', list_take_kth(
            1), deps=[task_frame_bone_transformations])

    task_frame_femur_coordinates = ctx.add_task('frame_femur_coordinates', calc_frame_coordinates,
                                                deps=[task_original_coordinates_femur, task_frame_bone_transformations_femur])
    task_frame_tibia_coordinates = ctx.add_task('frame_tibia_coordinates', calc_frame_coordinates,
                                                deps=[task_original_coordinates_tibia, task_frame_bone_transformations_tibia])
    task_frame_femur_meshes = ctx.add_task('frame_femur_meshes', transform_frame_mesh,
                                           deps=[task_femur_mesh, task_frame_bone_transformations_femur])
    task_frame_tibia_meshes = ctx.add_task('frame_tibia_meshes', transform_frame_mesh,
                                           deps=[task_tibia_mesh, task_frame_bone_transformations_tibia])
    task_frame_femur_cart_meshes = ctx.add_task('frame_femur_cart_meshes', transform_frame_mesh,
                                                deps=[task_femur_cart_mesh, task_frame_bone_transformations_femur])
    task_frame_tibia_cart_meshes = ctx.add_task('frame_tibia_cart_meshes', transform_frame_mesh,
                                                deps=[task_tibia_cart_mesh, task_frame_bone_transformations_tibia])
    task_frame_extended_femur_meshes = \
        ctx.add_task('frame_extended_femur_meshes', transform_frame_mesh,
                     deps=[task_extended_femur_mesh, task_frame_bone_transformations_femur])
    task_frame_extended_tibia_meshes = \
        ctx.add_task('frame_extended_tibia_meshes', transform_frame_mesh,
                     deps=[task_extended_tibia_mesh, task_frame_bone_transformations_tibia])

    task_bone_animation_frames = ctx.add_task('bone_animation_frames', gen_bone_animation_frames, deps=[
        task_frame_femur_meshes, task_frame_tibia_meshes,
        task_frame_femur_coordinates, task_frame_tibia_coordinates,
        task_frame_femur_cart_meshes, task_frame_tibia_cart_meshes,
    ])
    task_bone_animation = ctx.add_task('bone_animation', gen_bone_animation, deps=[
                                       task_bone_animation_frames])

    task_frame_contact_areas = ctx.add_task('frame_contact_areas', get_contact_area,
                                            deps=[
                                                task_watertight_test_extended_meshes,
                                                task_frame_extended_femur_meshes,
                                                task_frame_extended_tibia_meshes,
                                            ])

    if config.DEPTH_DIRECTION == config.DepthDirection.Z_AXIS:
        def job(_, coords): return [coord.t.unit_z for coord in coords]
    elif config.DEPTH_DIRECTION == config.DepthDirection.CONTACT_PLANE:
        job = calc_frame_contact_plane_normal_vectors
    elif config.DEPTH_DIRECTION == config.DepthDirection.VERTEX_NORMAL:
        def job(contact_areas, _): return [
            contact_area.vertex_normal for contact_area in contact_areas]
    else:
        raise NotImplementedError(
            f'Unknown DEPTH_DIRECTION: {config.DEPTH_DIRECTION}')
    task_frame_ray_directions = ctx.add_task(
        'frame_frame_ray_directions', job,
        deps=[task_frame_contact_areas, task_frame_tibia_coordinates],
    )

    task_frame_femur_cart_thickness = ctx.add_task('frame_femur_cart_thickness', calc_frame_cart_thickness,
                                                   deps=[task_frame_ray_directions, task_frame_femur_cart_meshes])
    task_frame_tibia_cart_thickness = ctx.add_task('frame_tibia_cart_thickness', calc_frame_cart_thickness,
                                                   deps=[task_frame_ray_directions, task_frame_tibia_cart_meshes])
    task_frame_femur_cart_thickness_origins = \
        ctx.add_task('frame_femur_cart_thickness_origins', list_take_kth(
            0), deps=[task_frame_femur_cart_thickness])
    task_frame_femur_cart_thickness_map = \
        ctx.add_task('frame_femur_cart_thickness_map', list_take_kth(
            1), deps=[task_frame_femur_cart_thickness])
    task_frame_tibia_cart_thickness_origins = \
        ctx.add_task('frame_tibia_cart_thickness_origins', list_take_kth(
            0), deps=[task_frame_tibia_cart_thickness])
    task_frame_tibia_cart_thickness_map = \
        ctx.add_task('frame_tibia_cart_thickness_map', list_take_kth(
            1), deps=[task_frame_tibia_cart_thickness])

    task_frame_femur_cart_thickness_origins_projected = ctx.add_task(
        'frame_femur_cart_thickness_origins_projected', project_cart_thickness_origins,
        deps=[task_frame_femur_cart_thickness_origins,
              task_frame_tibia_coordinates],
    )
    task_frame_tibia_cart_thickness_origins_projected = ctx.add_task(
        'frame_tibia_cart_thickness_origins_projected', project_cart_thickness_origins,
        deps=[task_frame_tibia_cart_thickness_origins,
              task_frame_tibia_coordinates],
    )
    task_plot_femur_cart_thickness = ctx.add_task(
        'plot_femur_cart_thickness',
        lambda a, b, c, d: plot_frame_cart_thickness_heatmap(
            a, b, c, d, 'femur', 1),
        deps=[task_contact_depth_map_extent,
              task_contact_depth_map_background_femur,
              task_frame_femur_cart_thickness_map,
              task_frame_femur_cart_thickness_origins_projected])
    task_plot_tibia_cart_thickness = ctx.add_task(
        'plot_tibia_cart_thickness',
        lambda a, b, c, d: plot_frame_cart_thickness_heatmap(
            a, b, c, d, 'tibia', 2),
        deps=[task_contact_depth_map_extent,
              task_contact_depth_map_background_tibia,
              task_frame_tibia_cart_thickness_map,
              task_frame_tibia_cart_thickness_origins_projected])

    task_frame_bone_distance_maps = \
        ctx.add_task('frame_bone_distance_maps', calc_bone_distance_map,
                     deps=[task_frame_femur_cart_meshes,
                           task_frame_tibia_cart_meshes,
                           task_frame_femur_meshes,
                           task_frame_tibia_meshes,
                           task_frame_ray_directions])
    task_frame_bone_distance_map_origins = \
        ctx.add_task('frame_bone_distance_map_origins', list_take_kth(
            0), deps=[task_frame_bone_distance_maps])
    task_frame_bone_distance_map_distances = \
        ctx.add_task('frame_bone_distance_map_distances',
                     list_take_kth(1), deps=[task_frame_bone_distance_maps])

    task_frame_contact_components = \
        ctx.add_task('frame_contact_components',
                     get_contact_components, deps=[task_frame_contact_areas])
    task_frame_contact_component_depth_maps = \
        ctx.add_task('frame_contact_component_depth_maps', calc_contact_depth_map,
                     deps=[task_frame_ray_directions, task_frame_contact_components])
    task_frame_contact_component_depth_map_origins = \
        ctx.add_task('frame_frame_contact_component_depth_map_origins',
                     lambda dms: [[pair[0] for pair in dm]
                                  for dm in dms] if dms else None,
                     deps=[task_frame_contact_component_depth_maps])
    task_frame_contact_component_depth_map_depths = \
        ctx.add_task('frame_frame_contact_component_depth_map_depths',
                     lambda dms: [[pair[1] for pair in dm]
                                  for dm in dms] if dms else None,
                     deps=[task_frame_contact_component_depth_maps])

    task_max_depth_curve = ctx.add_task('max_depth_curve', plot_max_depth_curve,
                                        deps=[
                                            task_frame_contact_components,
                                            task_frame_contact_component_depth_maps,
                                            task_frame_tibia_coordinates
                                        ])
    task_dof_data = ctx.add_task('dof_data', calc_dof,
                                 deps=[
                                     task_original_coordinates_femur,
                                     task_original_coordinates_tibia,
                                     task_frame_bone_transformations_femur,
                                     task_frame_bone_transformations_tibia,
                                 ])
    task_plot_dof_curves = ctx.add_task(
        'plot_dof_curves', plot_dof_curves, deps=[task_dof_data])
    task_interpolate_dof = ctx.add_task(
        'interpolate_dof', interpolate_dof, deps=[task_dof_data])

    task_contact_depth_map_frames = ctx.add_task('frame_contact_depth_map_frames', plot_contact_depth_maps,
                                                 deps=[
                                                     task_contact_depth_map_extent,
                                                     task_contact_depth_map_background_tibia,
                                                     task_frame_tibia_coordinates,
                                                     task_frame_bone_distance_map_origins,
                                                     task_frame_bone_distance_map_distances,
                                                     task_frame_contact_components,
                                                     task_frame_contact_component_depth_map_origins,
                                                     task_frame_contact_component_depth_map_depths,
                                                 ])
    task_contact_depth_map_animation = \
        ctx.add_task('frame_contact_depth_map_animation',
                     lambda fs: gen_animation(fs, os.path.join(
                         config.OUTPUT_DIRECTORY, 'depth_map_animation.gif')),
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
    if config.GENERATE_FEMUR_CARTILAGE_THICKNESS_MAP:
        task_plot_femur_cart_thickness()
    if config.GENERATE_TIBIA_CARTILAGE_THICKNESS_MAP:
        task_plot_tibia_cart_thickness()


def load_mesh(path: Optional[str]):
    if path:
        return trimesh.load_mesh(path)
    return None


def mesh_union(mesh1: trimesh.Trimesh, mesh2: Optional[trimesh.Trimesh]):
    if mesh2:
        return remove_bubbles(mesh1.union(mesh2))
    return remove_bubbles(mesh1)


def take_kth(k: int):
    def res(s: Sequence):
        return s[k]

    return res


def list_take_kth(k: int):
    def res(s: list[Sequence]):
        return [e[k] if e is not None else None for e in s] if s is not None else None

    return res


# lambda m, fts: [m.copy().apply_transform(ft.mat_homo) for ft in fts]
def transform_frame_mesh(mesh: Optional[trimesh.Trimesh], transformations: list[Transformation3D]):
    if mesh:
        return [mesh.copy().apply_transform(transformation.mat_homo) for transformation in transformations]
    return [None for _ in transformations]


# lambda flag, fms, tms: list(map(
#         lambda fm, tm: fm.intersection(tm),
#         fms, tms,
#     )) if flag else None,
def get_contact_area(check_watertight: bool, extended_femur_meshes, extended_tibia_meshes):
    if not check_watertight:
        return None
    return [fm.intersection(tm) for fm, tm in zip(extended_femur_meshes, extended_tibia_meshes)]


# lambda cas: list(map(lambda ca: ca.split(), cas)) if cas else None
def get_contact_components(contact_areas):
    if contact_areas is None:
        return None
    res = []
    for contact_area in contact_areas:
        res.append(contact_area.split())
    return res


def project_cart_thickness_origins(cart_thickness_origins, coordinates):
    if cart_thickness_origins is None or any(origins is None for origins in cart_thickness_origins):
        return None
    projected = []
    for coordinate, origin in zip(coordinates, cart_thickness_origins):
        origin_2d = coordinate.project(origin)[:, :2]
        projected.append(origin_2d)
    return projected


def gen_contact_depth_map_extent(extent):
    largest_side = np.max(np.abs(extent))
    extent = (-largest_side, largest_side, -largest_side, largest_side)
    return extent


def gen_contact_depth_map_background(contact_depth_map_extent, tibia_mesh, tibia_coord, tibia_cart_mesh=None):
    extent = contact_depth_map_extent
    largest_side = extent[1]
    res = config.DEPTH_MAP_RESOLUTION
    tm = tibia_mesh.copy()
    tm.visual.vertex_colors = hex_to_rgba1(config.DEPTH_MAP_BONE_COLOR_TIBIA)
    meshes = [tm]
    if tibia_cart_mesh:
        tcm = tibia_cart_mesh.copy()
        tcm.visual.vertex_colors = hex_to_rgba1(
            config.DEPTH_MAP_CARTILAGE_COLOR_TIBIA)
        meshes.append(tcm)
    bg = gen_orthographic_photo(
        meshes, tibia_coord, res, largest_side, largest_side, np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, -1000],
            [0, 0, 0, 1],
        ]), config.DEPTH_MAP_LIGHT_INTENSITY)
    bg = bg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return bg


def plot_contact_depth_maps(extent,
                            background,
                            frame_coordinates,
                            frame_bone_distance_map_origins,
                            frame_bone_distance_map_distances,
                            frame_contact_components,
                            frame_contact_component_depth_map_origins,
                            frame_contact_component_depth_map_depths):
    n = len(frame_bone_distance_map_distances)
    res = config.DEPTH_MAP_RESOLUTION
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:res[0] * 1j, extent[2]:extent[3]:res[1] * 1j]
    distance_threshold = 10
    vmin, vmax = 1e9, -1e9
    exclude_frames = set()
    for frame_index in range(n):
        distances = frame_bone_distance_map_distances[frame_index]
        distances = distances[(~np.isnan(distances)) &
                              (distances < distance_threshold)]
        depths = frame_contact_component_depth_map_depths
        frame_contact_depths = depths[frame_index] if depths else []
        if len(frame_contact_depths) > 0:
            g_depth = np.concatenate(frame_contact_depths)
            g_depth = g_depth[~np.isnan(g_depth)]
            all_data = np.concatenate([-distances, g_depth])
        else:
            all_data = -distances
        if len(all_data) == 0:
            exclude_frames.add(frame_index)
            continue
        vmax = max(np.max(all_data), vmax)
        vmin = min(np.min(all_data), vmin)
    if all(i in exclude_frames for i in range(n)):
        return

    frames = []
    for frame_index in range(n):
        if frame_index in exclude_frames:
            continue
        coord = frame_coordinates[frame_index]
        origins = frame_bone_distance_map_origins[frame_index].astype(Real)
        distances = frame_bone_distance_map_distances[frame_index]
        mask = distances < distance_threshold
        origins = origins[mask]
        depths = -distances[mask]
        deepest = []

        if (frame_contact_components and
                frame_contact_component_depth_map_origins and
                frame_contact_component_depth_map_depths):
            for c_mesh, c_origins, c_depth in zip(
                    frame_contact_components[frame_index],
                    frame_contact_component_depth_map_origins[frame_index],
                    frame_contact_component_depth_map_depths[frame_index]
            ):
                c_vertices = c_mesh.vertices.astype(Real)
                deepest.append(c_origins[np.argmax(c_depth)])
                s_origins = (np.round(origins, decimals=3)
                             * 1e4).astype(np.int64)
                s_vertices = (np.round(c_vertices, decimals=3)
                              * 1e4).astype(np.int64)
                intersect = np.intersect1d(s_origins, s_vertices)
                keep = np.all(~np.isin(s_origins, intersect), axis=1)
                origins = origins[keep]
                depths = depths[keep]
                origins = np.vstack([origins, c_origins])
                depths = np.concatenate([depths, c_depth])

        # labels = KMeans(n_clusters=2, random_state=42).fit_predict(origins)
        # groups = [labels == j for j in range(2)]

        groups = [coord.project(origins)[:, 0] >= 0,
                  coord.project(origins)[:, 0] < 0]

        g_origins = [origins[grp_mask] for grp_mask in groups]
        g_origins_2d = [coord.project(origins)[:, :2] for origins in g_origins]
        g_depth = [depths[grp_mask] for grp_mask in groups]
        g_z = []
        for i in range(2):
            if (g_origins_2d[i] is not None and len(g_origins_2d[i]) > 0
                    and g_depth[i] is not None and len(g_depth[i]) > 0):
                g_z.append(
                    griddata(g_origins_2d[i], g_depth[i], (grid_x, grid_y), method='linear'))
        if len(g_z) == 2:
            z = np.where(np.isnan(g_z[0]), g_z[1], g_z[0])
            z = np.where(np.isnan(z), g_z[0], z)
        else:
            z = g_z[0]

        # depth map
        fig, ax = plt.subplots()
        fig.suptitle(f'Depth Map - Frame {frame_index}')
        ax.imshow(background, extent=extent,
                  interpolation='none', aspect='equal')
        im = ax.contourf(
            grid_x, grid_y, z,
            levels=np.arange(vmin, vmax, 1),
            cmap=depth_cmap,
            alpha=0.5,
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Depth')
        if len(deepest) > 0:
            deepest = np.array(deepest)
            deepest_2d = coord.project(deepest)[:, :2]
            ax.scatter(deepest_2d[:, 0], deepest_2d[:, 1],
                       marker='+', s=100, color='turquoise')
        ax.invert_xaxis()

        image_path = os.path.join(get_frame_output_directory(
            frame_index), f'depth_map_frame_{frame_index}.jpg')
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(image_path)
        frames.append(Image.open(image_path))
    return frames


def gen_bone_animation(bone_animation_frames):
    gen_animation(bone_animation_frames, os.path.join(
        config.OUTPUT_DIRECTORY, 'animation.gif'))


def gen_bone_animation_frames(frame_femur_meshes, frame_tibia_meshes,
                              frame_femur_coords, frame_tibia_coords,
                              frame_femur_cart_meshes=None, frame_tibia_cart_meshes=None):
    num_frames = len(frame_femur_meshes)
    assert num_frames == len(frame_tibia_meshes)
    if frame_femur_cart_meshes and frame_tibia_cart_meshes:
        assert num_frames == len(frame_femur_cart_meshes)
        assert num_frames == len(frame_tibia_cart_meshes)

    images = []
    for i in range(num_frames):
        fm = frame_femur_meshes[i].copy()
        tm = frame_tibia_meshes[i].copy()
        fm.visual.vertex_colors = hex_to_rgba1(
            config.ANIMATION_BONE_COLOR_FEMUR)
        tm.visual.vertex_colors = hex_to_rgba1(
            config.ANIMATION_BONE_COLOR_TIBIA)
        meshes = [fm, tm]

        if (frame_femur_cart_meshes and frame_tibia_cart_meshes and
                frame_femur_cart_meshes[i] and frame_tibia_cart_meshes[i]):
            fcm = frame_femur_cart_meshes[i].copy()
            tcm = frame_tibia_cart_meshes[i].copy()
            fcm.visual.vertex_colors = hex_to_rgba1(
                config.ANIMATION_CARTILAGE_COLOR_FEMUR)
            tcm.visual.vertex_colors = hex_to_rgba1(
                config.ANIMATION_CARTILAGE_COLOR_TIBIA)
            meshes.extend([fcm, tcm])

        if config.ANIMATION_SHOW_BONE_COORDINATE:
            for component in WORLD_AXIS:
                for coord in [
                    frame_femur_coords[i],
                    frame_tibia_coords[i],
                ]:
                    c = component.copy()
                    c.apply_transform(coord.t.mat_homo)
                    meshes.append(c)

        rot_90 = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rot_90_rev = np.linalg.inv(rot_90)
        pose_x = np.array([
            [0, 0, -1, -1000],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ]) @ rot_90
        pose_x_rev = np.array([
            [0, 0, 1, 1000],
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ]) @ rot_90_rev
        pose_front = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 1000],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]) @ rot_90 @ rot_90

        if config.ANIMATION_DIRECTION == config.AnimationCameraDirection.FIX_TIBIA_FRONT:
            camera_pose = pose_front
        elif config.ANIMATION_DIRECTION == config.AnimationCameraDirection.FIX_TIBIA_L2M:
            camera_pose = pose_x
        elif config.ANIMATION_DIRECTION == config.AnimationCameraDirection.FIX_TIBIA_M2L:
            camera_pose = pose_x_rev
        else:
            raise NotImplementedError(
                f'Unknown Animation Camera Direction: {config.ANIMATION_DIRECTION}')

        image: Image.Image = gen_orthographic_photo(
            meshes, frame_tibia_coords[i], config.ANIMATION_RESOLUTION, 128, 128, camera_pose, config.ANIMATION_LIGHT_INTENSITY)

        image_path = os.path.join(
            get_frame_output_directory(i), f'animation_frame_{i}.png')
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(image_path)

        images.append(image)
    return images


def calc_frame_coordinates(original_coordinates_tibia, frame_transformations_tibia):
    res = []
    for tt in frame_transformations_tibia:
        coord = original_coordinates_tibia.copy()
        coord.t.apply_transformation(tt)
        res.append(coord)
    return res


def load_frame_bone_coordinates_raw() -> list[tuple[BoneCoordination, BoneCoordination]]:
    with open(config.MOVEMENT_DATA_FILE, 'r') as fp:
        lines = list(filter(lambda l: l.startswith('['), fp.readlines()))
    frame_transformations = []
    for line_index, line in enumerate(lines):
        data = np.array(json.loads(line), dtype=Real)
        if line_index % 2 == 0:
            femur_transformation = BoneCoordination.from_translation_and_quat(
                data[:3], data[3:])
        else:
            tibia_transformation = BoneCoordination.from_translation_and_quat(
                data[:3], data[3:])
            frame_transformations.append(
                (femur_transformation, tibia_transformation))
    return frame_transformations


def load_frame_bone_coordinates_csv() -> list[tuple[BoneCoordination, BoneCoordination]]:
    femur_coords = {}
    tibia_coords = {}
    with (open(config.MOVEMENT_DATA_FILE, 'r') as f):
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            tx, ty, tz = float(row['tx']), float(row['ty']), float(row['tz'])
            r0, r1, r2, r3 = float(row['r0']), float(
                row['r1']), float(row['r2']), float(row['r3'])
            coord = BoneCoordination.from_translation_and_quat(
                translation=np.array([tx, ty, tz]),
                quat=np.array([r0, r1, r2, r3]),
                extra={
                    'original_data': row,
                }
            )
            if row['anatomy_id'] == '1':
                femur_coords[int(row['measurement_id'])] = coord
            else:
                tibia_coords[int(row['measurement_id'])] = coord
    return list(zip(
        [femur_coords[key] for key in sorted(femur_coords.keys())],
        [tibia_coords[key] for key in sorted(tibia_coords.keys())],
    ))


def calc_extent(extended_tibia_mesh, tibia_coord, padding=5):
    proj_tm = extended_tibia_mesh.copy()
    proj_tm.vertices = tibia_coord.project(proj_tm.vertices)
    r, t = np.max(proj_tm.vertices[:, :2], axis=0)
    l, b = np.min(proj_tm.vertices[:, :2], axis=0)
    return [l - padding, r + padding, b - padding, t + padding]


def gen_orthographic_photo(meshes: list[trimesh.Trimesh], coord: BoneCoordination, res: Tuple[int, int], xmag: float,
                           ymag: float, camera_pose: np.array, light_intensity=3.0) -> Image.Image:
    pyr_scene = pyr.Scene()
    for mesh in meshes:
        mesh = mesh.copy()
        mesh.vertices = coord.project(mesh.vertices)
        pyr_mesh = pyr.Mesh.from_trimesh(mesh)
        pyr_scene.add(pyr_mesh)
    pyr_camera = pyr.OrthographicCamera(xmag, ymag, znear=0.1, zfar=1e5)
    pyr_scene.add(pyr_camera, pose=camera_pose)
    pyr_light = pyr.DirectionalLight(color=np.ones(3), intensity=3.0)
    pyr_scene.add(pyr_light, pose=camera_pose)
    renderer = pyr.OffscreenRenderer(
        viewport_width=res[0], viewport_height=res[1])
    color, _ = renderer.render(pyr_scene)
    img = Image.fromarray(color)
    return img


def calc_dof(original_coordinates_femur, original_coordinates_tibia,
             frame_transformations_femur, frame_transformations_tibia):
    y_tx, y_ty, y_tz = [], [], []
    y_rx, y_ry, y_rz = [], [], []

    for _, (ft, tt) in enumerate(zip(frame_transformations_femur, frame_transformations_tibia)):
        fc = original_coordinates_femur.copy()
        tc = original_coordinates_tibia.copy()
        fc.t.apply_transformation(ft)
        tc.t.apply_transformation(tt)
        r = fc.t.relative_to(tc.t)
        tx, ty, tz = r.mat_t
        y_tx.append(tx), y_ty.append(ty), y_tz.append(tz)
        if config.DOF_ROTATION_METHOD.value.startswith('euler'):
            rot = Rotation.from_matrix(r.mat_r)
            rx, ry, rz = rot.as_euler(
                config.DOF_ROTATION_METHOD.value[-3:], degrees=True)
        elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.PROJECTION:
            rx, ry, rz = extract_rotation_projection(
                fc.t.mat_homo, tc.t.mat_homo)
        else:
            raise NotImplementedError(
                f'unkown rotation method: {config.DOF_ROTATION_METHOD}')
        y_rx.append(rx), y_ry.append(ry), y_rz.append(rz)

    return np.array(y_tx), np.array(y_ty), np.array(y_tz), np.array(y_rx), np.array(y_ry), np.array(y_rz)


def plot_dof_curves(dof_data):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data
    x = np.arange(len(y_tx)) + 1

    fig, ax = plt.subplots()
    ax.plot(x, y_tx, 'x-')
    ax.set_title('Translation X')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_tx.png'))
    fig, ax = plt.subplots()
    ax.plot(x, y_ty, 'x-')
    ax.set_title('Translation Y')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_ty.png'))
    fig, ax = plt.subplots()
    ax.plot(x, y_tz, 'x-')
    ax.set_title('Translation Z')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_tz.png'))

    if config.DOF_ROTATION_METHOD.value.startswith('euler'):
        method = 'Euler ' + \
            '-'.join(list(config.DOF_ROTATION_METHOD.value[-3:]))
    elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.PROJECTION:
        method = 'Projection'
    else:
        raise NotImplementedError(
            f'unkown rotation method: {config.DOF_ROTATION_METHOD}')
    fig, ax = plt.subplots()
    ax.plot(x, y_rx, 'x-')
    ax.set_title(f'X Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_rx.png'))
    fig, ax = plt.subplots()
    ax.plot(x, y_ry, 'x-')
    ax.set_title(f'Y Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_ry.png'))
    fig, ax = plt.subplots()
    ax.plot(x, y_rz, 'x-')
    ax.set_title(f'Z Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_rz.png'))


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
        path = os.path.join(config.OUTPUT_DIRECTORY,
                            f'dof_curve_interpolated_{dof_name}.png')
        fig.savefig(path)

    path = os.path.join(config.OUTPUT_DIRECTORY, 'dof_curve_interpolated.mat')
    scipy.io.savemat(path, dof_interpolate_data)


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
        f = f.apply_transform(Transformation3D().apply_rotation(
            Rotation.from_euler('y', arc)).mat_homo)

        cs = f.intersection(t).split()
        scene = trimesh.Scene([ax] + cs)
        scene.set_camera(center=(0, 0, 0), distance=96)
        img_data = scene.save_image(visible=True)
        img_path = os.path.join(config.OUTPUT_DIRECTORY,
                                f'rotate_{deg:.1f}_deg_z.png')
        with open(img_path, 'wb') as f:
            f.write(img_data)
        scenes_z.append(Image.open(img_path))

        scene = trimesh.Scene([ax] + cs)
        scene.set_camera(center=(0, 0, 0), distance=96,
                         angles=(np.pi / 2, 0, 0))
        img_data = scene.save_image(visible=True)
        img_path = os.path.join(config.OUTPUT_DIRECTORY,
                                f'rotate_{deg:.1f}_deg_y.png')
        with open(img_path, 'wb') as f:
            f.write(img_data)
        scenes_y.append(Image.open(img_path))

        mds = [0]
        lds = [0]
        for c in cs:
            _, depths = do_calc_contact_depth_map(c, tibia_coord)
            if c.centroid[0] < 0:
                mds += list(depths)
            else:
                lds += list(depths)
        ym.append(max(mds))
        yl.append(max(lds))

    plt.plot(x, ym, 'x-', label='Medial')
    plt.plot(x, yl, 'x-', label='Lateral')
    plt.xlabel('Degree')
    plt.ylabel('Max Depth')
    plt.legend()

    gen_animation(scenes_z, os.path.join(
        config.OUTPUT_DIRECTORY, 'rotation_animation_z.gif'))
    gen_animation(scenes_y, os.path.join(
        config.OUTPUT_DIRECTORY, 'rotation_animation_y.gif'))


def plot_max_depth_curve(frame_contact_components, contact_component_depth_maps, frame_coordinates):
    if not (frame_contact_components and contact_component_depth_maps):
        return
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
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, 'max_depth_curve.png'))

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mdms, label='Medial')
    ax.plot(np.arange(n), mdls, label='Lateral')
    ax.legend()
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY,
                'max_depth_curve_split.png'))


def gen_animation(frames, output_path, fps: float = 2.5):
    if not frames or len(frames) == 0:
        print('No frames to generate: {}', output_path)
        return
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


# ctx.add_task('frame_contact_component_depth_maps',
#              lambda vs, fcs: [[calc_contact_depth_map(c, v) for c in cs] for v, cs in
#                               zip(vs, fcs)] if fcs else None,
#              deps=[task_frame_ray_direction, task_frame_contact_components])
def calc_contact_depth_map(frame_ray_directions, frame_contact_components):
    res = []
    for direction, components in zip(frame_ray_directions, frame_contact_components):
        depth_maps = []
        for component in components:
            depth_maps.append(do_calc_contact_depth_map(component, direction))
        res.append(depth_maps)
    return res


def do_calc_contact_depth_map(contact_component, v):
    origins, directions = prepare_rays_from_model(contact_component, -v, True)
    locations, ray_indices, _ = \
        contact_component.ray.intersects_location(origins, directions)
    if len(ray_indices) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def calc_bone_distance_map(fcs, tcs, fs, ts, vs):
    if fcs is None or tcs is None or any(fc is None for fc in fcs) or any(tc is None for tc in tcs):
        fcs, tcs = fs, ts  # use bone instead
    return [do_calc_bone_distance_map(f, t, v) for f, t, v in zip(fcs, tcs, vs)]


def do_calc_bone_distance_map(fm, tm, v):
    origins, directions = prepare_rays_from_model(tm, v)
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


def calc_frame_contact_plane_normal_vectors(frame_contact_areas: list[trimesh.Trimesh],
                                            coords: list[BoneCoordination]) -> list[np.ndarray]:
    normal_vectors = []
    for idx, contact_area in enumerate(frame_contact_areas):
        uz = normalize(coords[idx].t.unit_z)
        if contact_area.is_empty:
            normal_vectors.append(uz)
            continue
        vertices = contact_area.vertices
        centroid = contact_area.centroid
        adj_vertices = vertices - centroid
        u, s, vh = svd(adj_vertices)
        normal = normalize(vh[-1, :])
        if normal.dot(uz) > normal.dot(-uz):
            normal = -normal
        normal_vectors.append(normal)
    return normal_vectors


def calc_frame_cart_thickness(frame_directions, frame_cart_meshes):
    if frame_cart_meshes is None or any(mesh is None for mesh in frame_cart_meshes):
        return None
    frame_cart_thickness_map = []
    for direction, cart_mesh in zip(frame_directions, frame_cart_meshes):
        origins, thickness = do_calc_contact_depth_map(cart_mesh, direction)
        frame_cart_thickness_map.append((origins, thickness))
    return frame_cart_thickness_map


def plot_frame_cart_thickness_heatmap(extent, background, frame_cart_thickness_maps,
                                      frame_cart_thickness_origins_projected, bone_name, num_components):
    if frame_cart_thickness_maps is None or frame_cart_thickness_origins_projected is None:
        return None

    n = len(frame_cart_thickness_maps)
    res = config.DEPTH_MAP_RESOLUTION
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:res[0] * 1j, extent[2]:extent[3]:res[1] * 1j]

    vmin, vmax = 1e9, -1e9
    for frame_index in range(n):
        thickness_map = frame_cart_thickness_maps[frame_index]
        vmin = min(vmin, thickness_map.min())
        vmax = max(vmax, thickness_map.max())

    frames = []
    for frame_index in range(n):
        fig, ax = plt.subplots()
        fig.suptitle(
            f'Cartilage Thickness Map - Frame {frame_index} - {bone_name.capitalize()}')
        origins = frame_cart_thickness_origins_projected[frame_index]
        thickness_map = frame_cart_thickness_maps[frame_index]
        labels = KMeans(num_components).fit_predict(origins)

        ims = []
        for i in range(num_components):
            component_origins = origins[labels == i]
            component_thickness_map = thickness_map[labels == i]
            component_z = griddata(
                component_origins, component_thickness_map, (grid_x, grid_y), method='linear')
            im = ax.contourf(
                grid_x, grid_y, component_z,
                levels=np.arange(vmin, vmax, 1),
                cmap=depth_cmap,
                alpha=0.5,
            )
            ims.append(im)

        ax.imshow(background, extent=extent,
                  interpolation='none', aspect='equal')
        cb = fig.colorbar(ims[0], ax=ax)
        cb.set_label('Thickness')
        image_path = os.path.join(
            get_frame_output_directory(frame_index),
            f'cartilage_thickness_map_{bone_name}_{frame_index}.jpg')
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(image_path)
        frames.append(Image.open(image_path))
    return frames


def extract_rotation_projection(H_F, H_T):
    """
    按“投影到与目标轴垂直的平面”并区分内外旋的方法,
    针对单帧 (H_T, H_F) 计算:
      - 绕 X 轴的角度 (投影到 YZ 平面, 使用 F 的 Y 轴)
      - 绕 Y 轴的角度 (投影到 XZ 平面, 使用 F 的 Z 轴)
      - 绕 Z 轴的角度 (投影到 XY 平面, 使用 F 的 X 轴)
    坐标系定义:
      - T, F 分别是两个骨头的局部坐标系
      - H_T, H_F: 4x4 齐次变换(单帧)
    返回:
      (angle_x, angle_y, angle_z): 3 个 float, 单位: 度, 含正负号
    """

    # ========== 1) 提取旋转矩阵 & 构造 R_{TF} = R_T^T * R_F ==========
    R_T = H_T[:3, :3]   # T 在世界系的 3x3
    R_F = H_F[:3, :3]   # F 在世界系的 3x3

    R_T_trans = R_T.T   # R_T^T
    R_TF = R_T_trans @ R_F  # 3x3

    # ========== 2) 小工具: 投影到与某轴垂直平面, 计算带符号夹角 ==========
    def project_and_signed_angle(u_in_T, ref_in_T, axis_idx):
        """
        单帧版:
          - u_in_T, ref_in_T: 各是 3D 向量(已经在 T坐标系 中表达).
          - axis_idx: 0表示绕X轴 => 投影YZ平面, 1->绕Y=>投影XZ, 2->绕Z=>投影XY.

        过程:
          1) 把 u_in_T, ref_in_T 在该平面上投影 => 对 axis_idx 分量置 0
          2) 归一化, 然后算无符号夹角 = atan2(||u×v||, u·v)
          3) 用叉乘在 axis_idx 方向分量判断正负, 实现带符号角度 => [-180°, +180°].
        返回:
          angle_deg (float)
        """
        # 复制
        u_proj = u_in_T.copy()
        r_proj = ref_in_T.copy()

        # 投影: 把 axis_idx 分量清 0
        u_proj[axis_idx] = 0.0
        r_proj[axis_idx] = 0.0

        # 若模长过小, 避免除0
        eps = 1e-12
        norm_u = np.linalg.norm(u_proj)
        norm_r = np.linalg.norm(r_proj)
        if norm_u < eps or norm_r < eps:
            return 0.0

        u_hat = u_proj / norm_u
        r_hat = r_proj / norm_r

        dot_ = np.dot(u_hat, r_hat)
        cross_ = np.cross(u_hat, r_hat)
        cross_len = np.linalg.norm(cross_)

        # 无符号幅度
        angle_rad = np.arctan2(cross_len, dot_)

        # 符号: 叉乘在 axis_idx 分量判断
        sign_ = np.sign(cross_[axis_idx])
        angle_rad *= sign_

        return np.degrees(angle_rad)

    # ========== 3) 依次计算绕 X/Y/Z 轴 ==========
    #  定义 T 系的基向量 (在 T系, x=(1,0,0), y=(0,1,0), z=(0,0,1))
    eT_x = np.array([1, 0, 0], dtype=float)
    eT_y = np.array([0, 1, 0], dtype=float)
    eT_z = np.array([0, 0, 1], dtype=float)

    #  绕 X => 用 F系的 Y=(0,1,0), 先转到 T系 => 投影YZ => 和 eT_y 投影做夹角
    vF_y = np.array([0, 1, 0], dtype=float)
    vT_y = R_TF @ vF_y  # F 的 Y 向量在 T 系下的表达
    angle_x = project_and_signed_angle(vT_y, eT_y, axis_idx=0)

    #  绕 Y => 用 F系的 Z=(0,0,1), => 投影XZ => 和 eT_z 做夹角
    vF_z = np.array([0, 0, 1], dtype=float)
    vT_z = R_TF @ vF_z
    angle_y = project_and_signed_angle(vT_z, eT_z, axis_idx=1)

    #  绕 Z => 用 F系的 X=(1,0,0), => 投影XY => 和 eT_x 做夹角
    vF_x = np.array([1, 0, 0], dtype=float)
    vT_x = R_TF @ vF_x
    angle_z = project_and_signed_angle(vT_x, eT_x, axis_idx=2)

    return angle_x, angle_y, angle_z


if __name__ == "__main__":
    main()
