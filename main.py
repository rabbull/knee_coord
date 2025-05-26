import copy
import functools
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Self, List, Tuple
import csv
from datetime import datetime

import scipy.interpolate
import pyrender as pyr
import tqdm
import trimesh
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.linalg import svd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import av
import pandas as pd
from trimesh.creation import icosphere, cylinder

import config
import task
from utils import *

trimesh.util.attach_to_log(level=logging.INFO)

WORLD_AXIS = my_axis(axis_length=1000, axis_radius=2)

depth_cmap = LinearSegmentedColormap.from_list("depth_map", ['blue', 'green', 'yellow', 'red'])


def get_frame_output_directory(index):
    return os.path.join(config.OUTPUT_DIRECTORY, f'frame_{index}')


class BoneCoordination:
    def __init__(self):
        self._t = Transformation3D()
        self._extra = {}

    @classmethod
    def from_feature_points(cls, side, medial_point, lateral_point, proximal_point, distal_point, extra=None) -> Self:
        self = cls()

        match side:
            case config.KneeSide.LEFT:
                left_point, right_point = lateral_point, medial_point
            case config.KneeSide.RIGHT:
                left_point, right_point = medial_point, lateral_point
            case _:
                raise NotImplementedError
        del medial_point, lateral_point

        self._origin = (left_point + right_point) / 2
        self._t.set_translation(self._origin)

        raw_x = right_point - left_point
        unit_x = normalize(raw_x)
        raw_z = distal_point - proximal_point
        raw_z_proj_x = np.dot(raw_z, unit_x) * unit_x
        fixed_z = raw_z - raw_z_proj_x
        unit_z = normalize(fixed_z)
        raw_y = np.cross(unit_z, unit_x)
        unit_y = normalize(raw_y)
        mat_r = np.column_stack((unit_x, unit_y, unit_z))
        self._t.set_rotation(Rotation.from_matrix(mat_r))

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

    if config.IGNORE_CARTILAGE:
        config.FEMUR_CARTILAGE_MODEL_FILE = None
        config.TIBIA_CARTILAGE_MODEL_FILE = None
    task_femur_mesh = ctx.add_task('femur_mesh', lambda: load_mesh(config.FEMUR_MODEL_FILE))
    task_tibia_mesh = ctx.add_task('tibia_mesh', lambda: load_mesh(config.TIBIA_MODEL_FILE))
    task_femur_cart_mesh = ctx.add_task('femur_cart_mesh', lambda: load_mesh(config.FEMUR_CARTILAGE_MODEL_FILE))
    task_tibia_cart_mesh = ctx.add_task('tibia_cart_mesh', lambda: load_mesh(config.TIBIA_CARTILAGE_MODEL_FILE))

    task_extended_femur_mesh = \
        ctx.add_task('extended_femur_mesh', mesh_union, deps=[task_femur_mesh, task_femur_cart_mesh])
    task_extended_tibia_mesh = \
        ctx.add_task('extended_tibia_mesh', mesh_union, deps=[task_tibia_mesh, task_tibia_cart_mesh])

    task_all_extended_meshes = ctx.add_task('all_extended_meshes', lambda f, t: [f, t],
                                            deps=[task_extended_femur_mesh, task_extended_tibia_mesh])
    task_watertight_test_extended_meshes = \
        ctx.add_task('watertight_test_extended_meshes',
                     lambda meshes: not any(not m.is_watertight for m in meshes),
                     deps=[task_all_extended_meshes])

    task_original_coordinates = \
        ctx.add_task('original_coordinates', functools.partial(load_coord_from_file, config.FEATURE_POINT_FILE))
    task_original_coordinates_femur = \
        ctx.add_task('original_coordinates_femur', take_kth(0), deps=[task_original_coordinates])
    task_original_coordinates_tibia = \
        ctx.add_task('original_coordinates_tibia', take_kth(1), deps=[task_original_coordinates])

    task_depth_map_base_meshes = ctx.add_task('depth_map_base_meshes',
                                              lambda fm, tm: {config.BaseBone.FEMUR: fm, config.BaseBone.TIBIA: tm},
                                              deps=[task_femur_mesh, task_tibia_mesh])
    task_depth_map_base_coords = ctx.add_task('depth_map_base_coords',
                                              lambda fm, tm: {config.BaseBone.FEMUR: fm, config.BaseBone.TIBIA: tm},
                                              deps=[task_original_coordinates_femur, task_original_coordinates_tibia])
    task_depth_map_base_cart_meshes = ctx.add_task('depth_map_base_cart_meshes',
                                                   lambda fm, tm: {config.BaseBone.FEMUR: fm,
                                                                   config.BaseBone.TIBIA: tm},
                                                   deps=[task_femur_cart_mesh, task_tibia_cart_mesh])

    task_extents = ctx.add_task('extents', calc_extents, deps=[task_depth_map_base_meshes, task_depth_map_base_coords])
    task_depth_map_extent = ctx.add_task('contact_depth_map_extent', gen_contact_depth_map_extent, deps=[task_extents])
    task_depth_map_background = ctx.add_task('depth_map_background', gen_contact_depth_map_background, deps=[
        task_depth_map_extent, task_depth_map_base_meshes, task_depth_map_base_coords, task_depth_map_base_cart_meshes,
    ])

    match config.MOVEMENT_DATA_FORMAT:
        case config.MomentDataFormat.CSV:
            task_frame_bone_coordinates_raw = ctx.add_task('frame_bone_coordinates', load_frame_bone_coordinates_csv)
        case config.MomentDataFormat.JSON:
            task_frame_bone_coordinates_raw = ctx.add_task('frame_bone_coordinates', load_frame_bone_coordinates_raw)
        case _:
            raise NotImplementedError(f'Unknown MOVEMENT_DATA_FORMAT: {config.MOVEMENT_DATA_FORMAT}')

    task_frame_bone_transformations_raw = \
        ctx.add_task('frame_bone_transformations_raw',
                     lambda bcs: [tuple(e.t for e in bc) for bc in bcs],
                     deps=[task_frame_bone_coordinates_raw])
    task_frame_bone_transformations_femur_raw = \
        ctx.add_task('frame_bone_transformations_femur_raw',
                     list_take_kth(0),
                     deps=[task_frame_bone_transformations_raw])
    task_frame_bone_transformations_tibia_raw = \
        ctx.add_task('frame_bone_transformations_tibia_raw',
                     list_take_kth(1),
                     deps=[task_frame_bone_transformations_raw])

    task_frame_bone_transformations_femur_smoothed = \
        ctx.add_task('frame_bone_transformations_femur_smoothed',
                     smooth_transformations,
                     deps=[task_frame_bone_transformations_femur_raw])
    task_frame_bone_transformations_tibia_smoothed = \
        ctx.add_task('frame_bone_transformations_tibia_smoothed',
                     smooth_transformations,
                     deps=[task_frame_bone_transformations_tibia_raw])

    task_frame_bone_transformations_femur = task_frame_bone_transformations_femur_smoothed if config.MOVEMENT_SMOOTH else task_frame_bone_transformations_femur_raw
    task_frame_bone_transformations_tibia = task_frame_bone_transformations_tibia_smoothed if config.MOVEMENT_SMOOTH else task_frame_bone_transformations_tibia_raw
    task_frame_bone_transformations = ctx.add_task('frame_bone_transformations',
                                                   lambda fm, tm: {config.BaseBone.FEMUR: fm,
                                                                   config.BaseBone.TIBIA: tm},
                                                   deps=[task_frame_bone_transformations_femur,
                                                         task_frame_bone_transformations_tibia])

    # task_frame_femur_coordinates = \
    #     ctx.add_task('frame_femur_coordinates', do_calc_frame_coordinates, deps=[
    #         task_original_coordinates_femur,
    #         task_frame_bone_transformations_femur,
    #     ])
    # task_frame_tibia_coordinates = \
    #     ctx.add_task('frame_tibia_coordinates', do_calc_frame_coordinates, deps=[
    #         task_original_coordinates_tibia,
    #         task_frame_bone_transformations_tibia,
    #     ])
    task_frame_coordinates = ctx.add_task('frame_coordinates', calc_frame_coordinates,
                                          deps=[task_depth_map_base_coords, task_frame_bone_transformations])
    task_frame_femur_coordinates = ctx.add_task('frame_femur_coordinates', take_kth(config.BaseBone.FEMUR),
                                                deps=[task_frame_coordinates])
    task_frame_tibia_coordinates = ctx.add_task('frame_tibia_coordinates', take_kth(config.BaseBone.TIBIA),
                                                deps=[task_frame_coordinates])

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
    task_movement_animation = \
        ctx.add_task('movement_animation', gen_movement_animation, deps=[task_bone_animation_frames])

    task_frame_contact_areas = ctx.add_task('frame_contact_areas', get_contact_area,
                                            deps=[
                                                task_watertight_test_extended_meshes,
                                                task_frame_extended_femur_meshes,
                                                task_frame_extended_tibia_meshes,
                                            ])

    if config.DEPTH_DIRECTION == config.DepthDirection.Z_AXIS_TIBIA or config.DEPTH_DIRECTION == config.DepthDirection.Z_AXIS_FEMUR:
        base = config.BaseBone.TIBIA if config.DEPTH_DIRECTION == config.DepthDirection.Z_AXIS_TIBIA else config.BaseBone.FEMUR

        def job(_, frame_coordinates):
            return [coord.t.unit_z for coord in frame_coordinates[base]]
    elif config.DEPTH_DIRECTION == config.DepthDirection.CONTACT_PLANE:
        job = calc_frame_contact_plane_normal_vectors
    elif config.DEPTH_DIRECTION == config.DepthDirection.VERTEX_NORMAL:
        def job(contact_areas, _):
            return [contact_area.vertex_normal for contact_area in contact_areas]
    else:
        raise NotImplementedError(
            f'Unknown DEPTH_DIRECTION: {config.DEPTH_DIRECTION}')
    task_frame_ray_directions = ctx.add_task('frame_frame_ray_directions', job, deps=[
        task_frame_contact_areas, task_frame_coordinates
    ])

    task_frame_bone_distance_maps = \
        ctx.add_task('frame_bone_distance_maps', calc_bone_distance_map,
                     deps=[task_frame_femur_cart_meshes,
                           task_frame_tibia_cart_meshes,
                           task_frame_femur_meshes,
                           task_frame_tibia_meshes,
                           task_frame_ray_directions])
    task_frame_bone_distance_map_origins = \
        ctx.add_task('frame_bone_distance_map_origins', list_take_kth(0), deps=[task_frame_bone_distance_maps])
    task_frame_bone_distance_map_distances = \
        ctx.add_task('frame_bone_distance_map_distances', list_take_kth(1), deps=[task_frame_bone_distance_maps])

    task_frame_contact_components = \
        ctx.add_task('frame_contact_components', get_contact_components, deps=[task_frame_contact_areas])
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

    task_max_depth_curve = ctx.add_task('max_depth_curve', plot_max_depth_curve, deps=[
        task_frame_contact_components,
        task_frame_contact_component_depth_maps,
        task_frame_coordinates,
    ])
    task_min_distance_curve = ctx.add_task('min_distance_curve', plot_min_distance_curve, deps=[
        task_frame_bone_distance_map_origins,
        task_frame_bone_distance_map_distances,
        task_frame_coordinates,
    ])
    task_dof_data_smoothed = ctx.add_task(
        'dof_data_smoothed',
        lambda *args: calc_dof(*args) if config.MOVEMENT_SMOOTH else None,
        deps=[
            task_original_coordinates_femur,
            task_original_coordinates_tibia,
            task_frame_bone_transformations_femur_smoothed,
            task_frame_bone_transformations_tibia_smoothed,
        ])
    task_dof_data_raw = ctx.add_task('dof_data_raw', calc_dof, deps=[
        task_original_coordinates_femur,
        task_original_coordinates_tibia,
        task_frame_bone_transformations_femur_raw,
        task_frame_bone_transformations_tibia_raw])
    task_dump_dof = ctx.add_task('dump_dof', dump_dof,
                                 deps=[task_dof_data_raw, task_dof_data_smoothed])
    task_plot_dof_curves = ctx.add_task('plot_dof_curves', plot_dof_curves,
                                        deps=[task_dof_data_raw, task_dof_data_smoothed])

    task_frame_deepest_points = ctx.add_task('frame_deepest_points', calc_contact_deepest_points, deps=[
        task_frame_coordinates,
        task_frame_contact_component_depth_map_origins,
        task_frame_contact_component_depth_map_depths,
    ])

    task_contact_depth_map_frames = ctx.add_task('frame_contact_depth_map_frames', plot_contact_depth_maps,
                                                 deps=[
                                                     task_depth_map_extent,
                                                     task_depth_map_background,
                                                     task_frame_coordinates,
                                                     task_frame_bone_distance_map_origins,
                                                     task_frame_bone_distance_map_distances,
                                                     task_frame_contact_components,
                                                     task_frame_contact_component_depth_map_origins,
                                                     task_frame_contact_component_depth_map_depths,
                                                     task_frame_deepest_points,
                                                 ])
    task_contact_depth_map_animation = \
        ctx.add_task('frame_contact_depth_map_animation',
                     gen_depth_map_animation,
                     deps=[task_contact_depth_map_frames])

    task_frame_femur_cart_thickness_curve = \
        ctx.add_task('frame_femur_cart_thickness',
                     functools.partial(plot_cartilage_thickness_curve, bone_name='femur'),
                     deps=[
                         task_frame_femur_cart_meshes,
                         task_frame_deepest_points,
                         task_frame_ray_directions
                     ])
    task_frame_tibia_cart_thickness_curve = \
        ctx.add_task('frame_tibia_cart_thickness',
                     functools.partial(plot_cartilage_thickness_curve, bone_name='tibia', first2=True),
                     deps=[
                         task_frame_tibia_cart_meshes,
                         task_frame_deepest_points,
                         task_frame_ray_directions
                     ])

    task_femur_normed_max_depth_curve = ctx.add_task(
        'femur_normed_max_depth_curve',
        functools.partial(plot_normed_max_depth_curve, name='femur'),
        deps=[
            task_frame_femur_cart_thickness_curve,
            task_max_depth_curve,
        ])
    task_tibia_normed_max_depth_curve = ctx.add_task(
        'tibia_normed_max_depth_curve',
        functools.partial(plot_normed_max_depth_curve, name='tibia'),
        deps=[
            task_frame_tibia_cart_thickness_curve,
            task_max_depth_curve,
        ])

    task_area_curve = ctx.add_task('area_curve', plot_contact_area_curve, deps=[
        task_frame_contact_components, task_frame_coordinates, task_frame_ray_directions,
    ])

    task_dump_all_data = ctx.add_task('dump_all_data', dump_all_data, deps=[
        task_dof_data_raw, task_dof_data_smoothed, task_area_curve, task_frame_deepest_points, task_max_depth_curve,
    ])

    if config.GENERATE_CART_THICKNESS_CURVE:
        task_frame_femur_cart_thickness_curve()
        task_frame_tibia_cart_thickness_curve()
    if config.GENERATE_ANIMATION:
        task_movement_animation()
    if config.GENERATE_DEPTH_CURVE:
        task_max_depth_curve()
    if config.GENERATE_DEPTH_MAP:
        task_contact_depth_map_animation()
    if config.GENERATE_DOF_CURVES:
        task_plot_dof_curves()
        task_dump_dof()
    if config.GENERATE_NORM_DEPTH_CURVE:
        task_tibia_normed_max_depth_curve()
        task_femur_normed_max_depth_curve()
    if config.GENERATE_AREA_CURVE:
        task_area_curve()
    if config.DUMP_ALL_DATA:
        task_dump_all_data()
    if config.GENERATE_DISTANCE_CURVE:
        task_min_distance_curve()


def gen_movement_animation(images: dict[config.AnimationCameraDirection, list[Image.Image]]):
    for direction, frames in images.items():
        gen_animation(frames, name=f'animation_{direction.value}', duration=config.ANIMATION_DURATION)


def gen_depth_map_animation(images: dict[config.BaseBone, list[Image.Image]]):
    for direction, frames in images.items():
        gen_animation(frames, name=f'depth_map_{direction.value}', duration=config.DEPTH_MAP_DURATION)


def load_mesh(path: Optional[str]):
    if path:
        return trimesh.load_mesh(path)
    return None


def mesh_union(mesh1: trimesh.Trimesh, mesh2: Optional[trimesh.Trimesh]):
    if mesh2:
        return remove_bubbles(mesh1.union(mesh2))
    return remove_bubbles(mesh1)


def transform_frame_mesh(mesh: Optional[trimesh.Trimesh], transformations: list[Transformation3D]):
    if mesh:
        return [mesh.copy().apply_transform(transformation.mat_homo) for transformation in transformations]
    return [None for _ in transformations]


def get_contact_area(check_watertight: bool, extended_femur_meshes, extended_tibia_meshes):
    if not check_watertight:
        return None
    return [fm.intersection(tm) for fm, tm in tqdm.tqdm(zip(extended_femur_meshes, extended_tibia_meshes))]


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


def gen_contact_depth_map_extent(extents):
    depth_map_extents = {}
    for base, extent in extents.items():
        largest_side = np.max(np.abs(extent))
        extent = (-largest_side, largest_side, -largest_side, largest_side)
        depth_map_extents[base] = extent
    return depth_map_extents


def gen_contact_depth_map_background(extents, bone_meshes, coords, cart_meshes=None):
    res = config.DEPTH_MAP_RESOLUTION
    intensity = config.DEPTH_MAP_LIGHT_INTENSITY
    backgrounds = {}
    for base, extent in extents.items():
        bone_mesh = bone_meshes[base].copy()
        coord = coords[base]
        cart_mesh = cart_meshes[base].copy() if cart_meshes is not None and cart_meshes[base] is not None else None
        bone_mesh.visual.vertex_colors = hex_to_rgba1(config.DEPTH_MAP_BONE_COLOR_TIBIA)
        meshes = [bone_mesh]
        if cart_mesh is not None:
            cart_mesh.visual.vertex_colors = hex_to_rgba1(config.DEPTH_MAP_CARTILAGE_COLOR_TIBIA)
            meshes.append(cart_mesh)
        match base:
            case config.BaseBone.TIBIA:
                pose = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1000],
                    [0, 0, 0, 1],
                ])
            case config.BaseBone.FEMUR:
                pose = np.array([
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, -1000],
                    [0, 0, 0, 1],
                ])
            case _:
                raise ValueError(f'Unknown bone: {base}')
        background = gen_orthographic_photo(meshes, coord, res, extent[1], extent[1], pose, intensity)
        if base == config.BaseBone.FEMUR:
            background = background.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        backgrounds[base] = background

    return backgrounds


def calc_contact_deepest_points(
        frame_coordinates,
        frame_contact_component_depth_map_origins,
        frame_contact_component_depth_map_depths,
):
    deepest_points = {}
    for base, coords in frame_coordinates.items():
        deepest_points[base] = \
            do_calc_contact_deepest_points(coords, frame_contact_component_depth_map_origins,
                                           frame_contact_component_depth_map_depths)
    return deepest_points


def do_calc_contact_deepest_points(
        frame_coordinates,
        frame_contact_component_depth_map_origins,
        frame_contact_component_depth_map_depths,
):
    n = len(frame_contact_component_depth_map_origins)
    frame_deepest_points = []
    for frame_index in range(n):
        coord = frame_coordinates[frame_index]
        deepest = []
        if (frame_contact_component_depth_map_origins and
                frame_contact_component_depth_map_depths):
            for c_origins, c_depth in zip(
                    frame_contact_component_depth_map_origins[frame_index],
                    frame_contact_component_depth_map_depths[frame_index]
            ):
                if len(c_origins) == 0 or len(c_depth) == 0:
                    continue
                idx = np.argmax(c_depth)
                deepest.append((c_origins[idx], c_depth[idx]))

        left, right = (None, -1e9), (None, -1e9)
        for origin, depth in deepest:
            if coord.project(origin)[0] < 0 and left[1] < depth:
                left = (origin, depth)
            if coord.project(origin)[0] > 0 and right[1] < depth:
                right = (origin, depth)

        frame_deepest_points.append((left[0], right[0]))

    return frame_deepest_points


def plot_contact_depth_maps(extents,
                            backgrounds,
                            frame_coordinates,
                            frame_bone_distance_map_origins,
                            frame_bone_distance_map_distances,
                            frame_contact_components,
                            frame_contact_component_depth_map_origins,
                            frame_contact_component_depth_map_depths,
                            frame_deepest_points):
    frames = {}
    for base, extent in extents.items():
        background = backgrounds[base]
        deepest_points = frame_deepest_points[base]
        coords = frame_coordinates[base]
        frames[base] = do_plot_contact_depth_maps(
            base.value,
            extent,
            background,
            coords,
            frame_bone_distance_map_origins,
            frame_bone_distance_map_distances,
            frame_contact_components,
            frame_contact_component_depth_map_origins,
            frame_contact_component_depth_map_depths,
            deepest_points,
        )
    return frames


def do_plot_contact_depth_maps(name,
                               extent,
                               background,
                               frame_coordinates,
                               frame_bone_distance_map_origins,
                               frame_bone_distance_map_distances,
                               frame_contact_components,
                               frame_contact_component_depth_map_origins,
                               frame_contact_component_depth_map_depths,
                               frame_deepest_points):
    n = len(frame_bone_distance_map_distances)
    res = config.DEPTH_MAP_RESOLUTION
    grid_x, grid_y = np.mgrid[extent[0]:extent[1]:res[0] * 1j, extent[2]:extent[3]:res[1] * 1j]
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    distance_threshold = config.DEPTH_MAP_DEPTH_THRESHOLD
    vmin, vmax = 1e9, -1e9
    exclude_frames = set()
    for frame_index in range(n):
        distances = frame_bone_distance_map_distances[frame_index]
        distances = distances[(~np.isnan(distances)) & (distances < distance_threshold)]
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
        return None

    frames = []
    for frame_index in tqdm.tqdm(range(n)):
        # if frame_index in exclude_frames:
        #     continue
        coord = frame_coordinates[frame_index]
        origins = frame_bone_distance_map_origins[frame_index].astype(Real)
        distances = frame_bone_distance_map_distances[frame_index]
        mask = (~np.isnan(distances)) & (distances < distance_threshold)
        origins = origins[mask]
        depths = -distances[mask]

        if (frame_contact_components and
                frame_contact_component_depth_map_origins and
                frame_contact_component_depth_map_depths):
            for c_mesh, c_origins, c_depth in zip(
                    frame_contact_components[frame_index],
                    frame_contact_component_depth_map_origins[frame_index],
                    frame_contact_component_depth_map_depths[frame_index]
            ):
                if len(c_origins) == 0 or len(c_depth) == 0:
                    continue
                c_vertices = c_mesh.vertices.astype(Real)
                s_origins = (np.round(origins, decimals=3) * 1e4).astype(np.int64)
                s_vertices = (np.round(c_vertices, decimals=3) * 1e4).astype(np.int64)
                intersect = np.intersect1d(s_origins, s_vertices)
                # o_view = s_origins.view([('', s_origins.dtype)] * 2)
                # v_view = s_vertices.view([('', s_vertices.dtype)] * 2)
                # intersect = np.intersect1d(o_view, v_view)
                keep = np.all(~np.isin(s_origins, intersect), axis=1)
                origins = origins[keep]
                depths = depths[keep]
                origins = np.vstack([origins, c_origins])
                depths = np.concatenate([depths, c_depth])

        origins_projected = coord.project(origins)
        origins_projected_2d = origins_projected[:, :2]

        origins_projected_x = origins_projected[:, 0]
        groups = [origins_projected_x >= 0, origins_projected_x < 0]

        g_origins_2d = [origins_projected_2d[grp_mask] for grp_mask in groups]
        g_depth = [depths[grp_mask] for grp_mask in groups]

        g_z = []
        for i in range(2):
            if (g_origins_2d[i] is not None and len(g_origins_2d[i]) > 1
                    and g_depth[i] is not None and len(g_depth[i]) > 1):
                o, d = g_origins_2d[i], g_depth[i]
                tree = cKDTree(o)
                z = griddata(o, d, (grid_x, grid_y), method='linear')
                dists, _ = tree.query(grid_points)
                z.ravel()[dists > 2] = np.nan
                g_z.append(z)

        if len(g_z) == 2:
            z = np.where(np.isnan(g_z[0]), g_z[1], g_z[0])
            z = np.where(np.isnan(z), g_z[0], z)
        elif len(g_z) == 1:
            z = g_z[0]
        else:
            z = np.full(grid_x.shape, np.nan)

        # depth map
        fig, ax = plt.subplots()
        fig.suptitle(f'Depth Map {float(frame_index) / float(n) * 100:.1f}%')
        ax.imshow(background, extent=extent, interpolation='none', aspect='equal')
        im = ax.contourf(
            grid_x, grid_y, z,
            levels=np.arange(vmin, vmax, 1),
            cmap=depth_cmap,
            alpha=0.5,
            extend='both',
        )
        cb = fig.colorbar(im, ax=ax, extend='both')
        cb.set_label('Depth')

        left_deepest_point, right_deepest_point = frame_deepest_points[frame_index]
        if left_deepest_point is not None or right_deepest_point is not None:
            deepest_points = []
            if left_deepest_point is not None: deepest_points.append(left_deepest_point)
            if right_deepest_point is not None: deepest_points.append(right_deepest_point)
            deepest_points = np.array(deepest_points, dtype=Real)
            deepest_points_2d = coord.project(deepest_points)[:, :2]
            ax.scatter(deepest_points_2d[:, 0], deepest_points_2d[:, 1], marker='+', s=100, color='turquoise')

        for label in ax.get_yticklabels():
            if label.get_text() == '0':
                label.set_text('')
        for label in ax.get_xticklabels():
            if label.get_text() == '0':
                x, y = label.get_position()
                label.set_position((x + 1, y))
        ax.invert_xaxis()
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        image_path = os.path.join(get_frame_output_directory(frame_index), f'depth_map_{name}_frame_{frame_index}.jpg')
        directory = os.path.dirname(image_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(image_path)
        frames.append(Image.open(image_path))
        plt.close(fig)
    return frames


def gen_animation(frames: list[Image.Image], name: str, duration: float):
    gif_path = os.path.join(config.OUTPUT_DIRECTORY, f'{name}.gif')
    mp4_path = os.path.join(config.OUTPUT_DIRECTORY, f'{name}.mp4')

    if not frames or len(frames) == 0:
        print('No frames to generate: {}', gif_path)
        return

    # gif
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration / len(frames) * 1000,
        loop=0
    )

    # mp4
    fps = 24
    total_frames = int(fps * duration)
    n_src = len(frames)
    if n_src == total_frames:
        sel = frames
    else:
        sel = []
        step = (n_src - 1) / (total_frames - 1) if total_frames > 1 else 0
        for i in range(total_frames):
            idx = int(round(i * step))
            sel.append(frames[idx])

    with av.open(mp4_path, mode='w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = sel[0].size
        stream.pix_fmt = 'yuv420p'
        stream.options = {
            'crf': '0',
            'preset': 'veryslow',
        }
        for img in tqdm.tqdm(sel):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def gen_bone_animation_frames(frame_femur_meshes, frame_tibia_meshes,
                              frame_femur_coords, frame_tibia_coords,
                              frame_femur_cart_meshes=None, frame_tibia_cart_meshes=None):
    num_frames = len(frame_femur_meshes)
    assert num_frames == len(frame_tibia_meshes)
    if frame_femur_cart_meshes and frame_tibia_cart_meshes:
        assert num_frames == len(frame_femur_cart_meshes)
        assert num_frames == len(frame_tibia_cart_meshes)

    images = defaultdict(list)
    for i in tqdm.tqdm(range(num_frames)):
        fm = frame_femur_meshes[i].copy()
        tm = frame_tibia_meshes[i].copy()
        fm.visual.vertex_colors = hex_to_rgba1(config.ANIMATION_BONE_COLOR_FEMUR)
        tm.visual.vertex_colors = hex_to_rgba1(config.ANIMATION_BONE_COLOR_TIBIA)
        meshes = [fm, tm]

        if (frame_femur_cart_meshes and frame_tibia_cart_meshes and
                frame_femur_cart_meshes[i] and frame_tibia_cart_meshes[i]):
            fcm = frame_femur_cart_meshes[i].copy()
            tcm = frame_tibia_cart_meshes[i].copy()
            fcm.visual.vertex_colors = hex_to_rgba1(config.ANIMATION_CARTILAGE_COLOR_FEMUR)
            tcm.visual.vertex_colors = hex_to_rgba1(config.ANIMATION_CARTILAGE_COLOR_TIBIA)
            meshes.extend([fcm, tcm])

        if config.ANIMATION_SHOW_BONE_COORDINATE:
            for anatomy_id, coord in enumerate([
                frame_femur_coords[i], frame_tibia_coords[i],
            ]):
                meshes.extend(my_axis(axis_length=200, axis_radius=2, transform=coord.t.mat_homo))

        pose_x = np.array([
            [0, 0, 1, 500.],
            [1, 0, 0, 0.],
            [0, 1, 0, 0.],
            [0, 0, 0, 1.],
        ], dtype=Real)
        pose_xi = np.array([
            [0, 0, -1, -500],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=Real)
        pose_y = np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 500],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=Real)

        camera_poses = {
            config.KneeSide.LEFT: {
                config.AnimationCameraDirection.FIX_TIBIA_FRONT: pose_y,
                config.AnimationCameraDirection.FIX_TIBIA_L2M: pose_xi,
                config.AnimationCameraDirection.FIX_TIBIA_M2L: pose_x,
            },
            config.KneeSide.RIGHT: {
                config.AnimationCameraDirection.FIX_TIBIA_FRONT: pose_y,
                config.AnimationCameraDirection.FIX_TIBIA_L2M: pose_x,
                config.AnimationCameraDirection.FIX_TIBIA_M2L: pose_xi,
            }
        }

        for direction in config.ANIMATION_CAMERA_DIRECTION:
            camera_pose = camera_poses[config.KNEE_SIDE][direction]
            image = gen_orthographic_photo(meshes, frame_tibia_coords[i], config.ANIMATION_RESOLUTION,
                                           128, 128, camera_pose, config.ANIMATION_LIGHT_INTENSITY)
            image_path = os.path.join(get_frame_output_directory(i), f'animation_{direction.value}_frame_{i}.png')
            directory = os.path.dirname(image_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            image.save(image_path)
            images[direction].append(image)

    return images


def calc_frame_coordinates(base_coordinates, base_transformations):
    res = {}
    for base, coordinates in base_coordinates.items():
        transformations = base_transformations[base]
        res[base] = do_calc_frame_coordinates(coordinates, transformations)
    return res


def do_calc_frame_coordinates(original_coordinates, frame_transformations):
    res = []
    for transformation in tqdm.tqdm(frame_transformations):
        coordinate = original_coordinates.copy()
        coordinate.t.apply_transformation(transformation)
        res.append(coordinate)
    return res


def load_frame_bone_coordinates_raw() -> list[tuple[BoneCoordination, BoneCoordination]]:
    with open(config.MOVEMENT_DATA_FILE, 'r') as fp:
        lines = list(filter(lambda l: l.startswith('['), fp.readlines()))
    frame_transformations = []
    for line_index, line in enumerate(lines):
        data = np.array(json.loads(line), dtype=Real)
        if line_index % 2 == 0:
            femur_transformation = BoneCoordination.from_translation_and_quat(data[:3], data[3:])
        else:
            tibia_transformation = BoneCoordination.from_translation_and_quat(data[:3], data[3:])
            frame_transformations.append((femur_transformation, tibia_transformation))
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

    coords = list(zip(
        [femur_coords[key] for key in sorted(femur_coords.keys())],
        [tibia_coords[key] for key in sorted(tibia_coords.keys())],
    ))
    if config.MOVEMENT_PICK_FRAMES is not None:
        res = []
        for index in config.MOVEMENT_PICK_FRAMES:
            res.append(coords[index])
        coords = res
    return coords


def calc_extents(meshes: dict[config.BaseBone, trimesh.Trimesh], coords: dict[config.BaseBone, BoneCoordination],
                 padding=5):
    bases = set(meshes.keys()).intersection(coords.keys())
    extents = {}
    for base in bases:
        mesh = meshes[base]
        coord = coords[base]
        extents[base] = calc_extent(mesh, coord, padding)
    return extents


def calc_extent(extended_tibia_mesh, tibia_coord, padding=5):
    proj_tm = extended_tibia_mesh.copy()
    proj_tm.vertices = tibia_coord.project(proj_tm.vertices)
    r, t = np.max(proj_tm.vertices[:, :2], axis=0)
    l, b = np.min(proj_tm.vertices[:, :2], axis=0)
    return [l - padding, r + padding, b - padding, t + padding]


def gen_orthographic_photo(meshes: list[trimesh.Trimesh], coord: BoneCoordination, res: Tuple[int, int],
                           xmag: float, ymag: float, camera_pose: np.array, light_intensity=3.0) -> Image.Image:
    pyr_scene = pyr.Scene()
    # scene = trimesh.Scene()
    for mesh in meshes:
        mesh = mesh.copy()
        mesh.vertices = coord.project(mesh.vertices)
        # scene.add_geometry(mesh)
        pyr_mesh = pyr.Mesh.from_trimesh(mesh)
        pyr_scene.add(pyr_mesh)
    pyr_camera = pyr.OrthographicCamera(xmag, ymag, znear=0.1, zfar=1e5)
    pyr_scene.add(pyr_camera, pose=camera_pose)
    # scene.add_geometry(my_axis(transform=camera_pose, axis_length=50, axis_radius=4))
    # scene.show()
    pyr_light = pyr.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    pyr_scene.add(pyr_light, pose=camera_pose)
    renderer = pyr.OffscreenRenderer(
        viewport_width=res[0], viewport_height=res[1])
    color, _ = renderer.render(pyr_scene)
    img = Image.fromarray(color)
    return img


def to_jcs(homo, side: config.KneeSide = config.KneeSide.LEFT):
    r = homo[:3, :3]
    p = homo[:3, 3]
    e3_t = np.array([0., 0., 1.])
    e3_f = r @ e3_t
    e1_f = np.array([1., 0., 0.])
    e2_f = np.cross(e3_f, e1_f)
    norm_e2 = np.linalg.norm(e2_f)
    if norm_e2 < 1e-9:
        e2_f = np.zeros(3)
    else:
        e2_f /= norm_e2
    dot13 = np.dot(e1_f, e3_f)
    dot13 = np.clip(dot13, -1.0, 1.0)
    beta = np.arccos(dot13)
    if side == config.KneeSide.RIGHT:
        adduction = beta - np.pi / 2
    else:
        adduction = np.pi / 2 - beta
    fwd_femur = np.array([0., 1., 0.])
    cos_alpha = np.dot(e2_f, fwd_femur)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha_raw = np.arccos(cos_alpha)
    cross_dir = np.cross(fwd_femur, e2_f)
    sign_test = np.dot(cross_dir, e1_f)
    alpha = alpha_raw if sign_test >= 0 else -alpha_raw
    j_t_in_f = r @ np.array([0., 1., 0.])
    cos_gamma = np.dot(j_t_in_f, e2_f)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma_raw = np.arccos(cos_gamma)
    cross_test = np.cross(e2_f, j_t_in_f)
    sign_test2 = np.dot(cross_test, e3_f)
    gamma = gamma_raw if sign_test2 >= 0 else -gamma_raw
    p_f = - r.T @ p
    q1 = np.dot(p_f, e1_f)
    q2 = np.dot(p_f, e2_f)
    q3 = - np.dot(p_f, e3_f)

    return {
        'adduction': adduction,
        'flexion': alpha,
        'tibial_rotation': gamma,
        'q1': q1,
        'q2': q2,
        'q3': q3
    }


def calc_dof(original_coordinates_femur, original_coordinates_tibia,
             frame_transformations_femur, frame_transformations_tibia):
    y_tx, y_ty, y_tz = [], [], []
    y_rx, y_ry, y_rz = [], [], []

    for _, (ft, tt) in enumerate(zip(frame_transformations_femur, frame_transformations_tibia)):
        fc = original_coordinates_femur.copy()
        tc = original_coordinates_tibia.copy()
        fc.t.apply_transformation(ft)
        tc.t.apply_transformation(tt)

        if config.DOF_BASE_BONE == config.BaseBone.FEMUR:
            r = tc.t.relative_to(fc.t)
        elif config.DOF_BASE_BONE == config.BaseBone.TIBIA:
            r = fc.t.relative_to(tc.t)
        else:
            raise NotImplementedError(f'unknown base bone: {config.DOF_BASE_BONE}')

        if config.DOF_ROTATION_METHOD == config.DofRotationMethod.JCS:
            transform: dict[str, Any] = to_jcs(r.mat_homo, side='left')
            tx, ty, tz = transform['q1'], transform['q2'], transform['q3']
            ry = transform['adduction'] / np.pi * 180
            rx = transform['flexion'] / np.pi * 180
            rz = transform['tibial_rotation'] / np.pi * 180
        else:
            tx, ty, tz = r.mat_t
            if config.DOF_ROTATION_METHOD.value.startswith('euler'):
                rot = Rotation.from_matrix(r.mat_r)
                rx, ry, rz = rot.as_euler(config.DOF_ROTATION_METHOD.value[-3:], degrees=True)
            elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.PROJECTION:
                rx, ry, rz = extract_rotation_projection(fc.t.mat_homo, tc.t.mat_homo)
            else:
                raise NotImplementedError(f'unknown rotation method: {config.DOF_ROTATION_METHOD}')

        y_tx.append(tx), y_ty.append(ty), y_tz.append(tz)
        y_rx.append(rx), y_ry.append(ry), y_rz.append(rz)

    return np.array(y_tx), np.array(y_ty), np.array(y_tz), np.array(y_rx), np.array(y_ry), np.array(y_rz)


def dump_dof(raw, smoothed):
    do_dump_dof(raw, "dof_raw.csv")
    if smoothed is not None:
        do_dump_dof(smoothed, "dof_smoothed.csv")


def do_dump_dof(dof_data, filename):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data
    num_frames = len(y_tx)
    x = np.arange(1, num_frames + 1)

    csv_path = os.path.join(config.OUTPUT_DIRECTORY, filename)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Translation X', 'Translation Y', 'Translation Z',
                         'Rotation X', 'Rotation Y', 'Rotation Z'])

        for i in range(num_frames):
            writer.writerow([
                x[i],
                y_tx[i],
                y_ty[i],
                y_tz[i],
                y_rx[i],
                y_ry[i],
                y_rz[i]
            ])


def plot_dof_curves(dof_data_raw, dof_data_smoothed):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data_raw
    if dof_data_smoothed is not None:
        ys_tx, ys_ty, ys_tz, ys_rx, ys_ry, ys_rz = dof_data_smoothed
    raw_line = 'x-' if dof_data_smoothed is None else 'x'
    smoothed_line = '-'

    x = np.arange(len(y_tx)) + 1
    if dof_data_smoothed is not None:
        xs = np.arange(len(ys_tx)) + 1
        x = np.linspace(xs[0], xs[-1], len(x))

    fig, ax = plt.subplots()
    ax.plot(x, y_tx, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_tx, smoothed_line)
    ax.set_title('Translation X')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_tx.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_ty, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_ty, smoothed_line)
    ax.set_title('Translation Y')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_ty.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_tz, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_tz, smoothed_line)
    ax.set_title('Translation Z')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_tz.png'))
    plt.close(fig)

    if config.DOF_ROTATION_METHOD.value.startswith('euler'):
        method = 'Euler ' + \
                 '-'.join(list(config.DOF_ROTATION_METHOD.value[-3:]))
    elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.PROJECTION:
        method = 'Projection'
    elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.JCS:
        method = 'JCS'
    else:
        raise NotImplementedError(
            f'unkown rotation method: {config.DOF_ROTATION_METHOD}')

    fig, ax = plt.subplots()
    ax.plot(x, y_rx, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_rx, smoothed_line)
    ax.set_title(f'X Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_rx.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_ry, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_ry, smoothed_line)
    ax.set_title(f'Y Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_ry.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_rz, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_rz, smoothed_line)
    ax.set_title(f'Z Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_rz.png'))
    plt.close(fig)


def smooth_dof(dof_data):
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
        method_name, cls = config.MOVEMENT_INTERPOLATE_METHOD.value
        interpolate = cls(x, y)
        yi = interpolate(xi)
        dof_interpolate_data[dof_name] = yi
        ax.plot(xi, yi, label='Interpolated')
        ax.legend()
        path = os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_interpolated_{dof_name}.png')
        fig.savefig(path)
        plt.close(fig)

    path = os.path.join(config.OUTPUT_DIRECTORY, 'dof_curve_interpolated.mat')
    scipy.io.savemat(path, dof_interpolate_data)

    return dof_interpolate_data


def plot_max_depth_curve(frame_contact_components, contact_component_depth_maps, frame_coordinates):
    res = {}
    for base, coords in frame_coordinates.items():
        res[base] = do_plot_max_depth_curve(base.value, frame_contact_components, contact_component_depth_maps, coords)
    return res


def do_plot_max_depth_curve(name, frame_contact_components, contact_component_depth_maps, frame_coordinates):
    if not (frame_contact_components and contact_component_depth_maps):
        return None
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
            if depths is None or depths.shape[0] == 0:
                continue
            if ((cx := coordinate.project(c.centroid)[0]) >= 0 and config.KNEE_SIDE == config.KneeSide.LEFT) or \
                    (cx < 0 and config.KNEE_SIDE == config.KneeSide.RIGHT):
                mdm = max(mdm, np.max(depths))
            else:
                mdl = max(mdl, np.max(depths))
        mdms.append(mdm)
        mdls.append(mdl)
        mds.append(max(mdm, mdl))

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mds)
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'max_depth_curve_{name}.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mdms, label='Medial')
    ax.plot(np.arange(n), mdls, label='Lateral')
    ax.legend()
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'max_depth_curve_{name}_split.png'))
    plt.close(fig)

    df = pd.DataFrame({
        'index': np.arange(n),
        'max_depth': np.array(mds, dtype=Real),
        'max_depth_medial': np.array(mdms, dtype=Real),
        'max_depth_lateral': np.array(mdls, dtype=Real),
    })
    df.to_csv(os.path.join(config.OUTPUT_DIRECTORY, f'max_depth_curve_{name}.csv'), index=False)

    return mds, mdms, mdls


def plot_min_distance_curve(frame_bone_distance_origins, frame_bone_distances, frame_coordinates):
    res = {}
    for base, coords in frame_coordinates.items():
        res[base] = do_plot_min_distance_curve(base.value, frame_bone_distance_origins, frame_bone_distances, coords)
    return res


def do_plot_min_distance_curve(name, frame_bone_distance_origins, frame_bone_distances, frame_coordinates):
    if frame_bone_distance_origins is None or frame_bone_distances is None or frame_coordinates is None:
        return None
    n = len(frame_coordinates)
    mdms = []
    mdls = []
    mds = []
    for i in range(n):
        mdm, mdl = 0, 0
        coordination = frame_coordinates[i]
        origins = frame_bone_distance_origins[i]
        distances = frame_bone_distances[i]

        origins_x = coordination.project(origins)[:, 0]
        distances_left = distances[origins_x < 0]
        distances_right = distances[origins_x > 0]
        min_distance_left = np.min(distances_left) if len(distances_left) > 0 else 0
        min_distance_right = np.min(distances_right) if len(distances_right) > 0 else 0
        if config.KNEE_SIDE == config.KneeSide.LEFT:
            mdm, mdl = min_distance_right, min_distance_left
        else:
            mdm, mdl = min_distance_left, min_distance_right
        mdms.append(mdm)
        mdls.append(mdl)
        mds.append(min(mdm, mdl))

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mds)
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'min_distance_curve_{name}.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.arange(n), mdms, label='Medial')
    ax.plot(np.arange(n), mdls, label='Lateral')
    ax.legend()
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'min_distance_curve_{name}_split.png'))
    plt.close(fig)

    df = pd.DataFrame({
        'index': np.arange(n),
        'min_distance': np.array(mds, dtype=Real),
        'min_distance_medial': np.array(mdms, dtype=Real),
        'min_distance_lateral': np.array(mdls, dtype=Real),
    })
    df.to_csv(os.path.join(config.OUTPUT_DIRECTORY, f'min_distance_curve_{name}.csv'), index=False)

    return mds, mdms, mdls


def load_coord_from_file(path):
    coord_points = {}
    with (open(path, 'r') as f):
        for line in f.readlines():
            if line.isspace():
                continue
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
    femur_coord = BoneCoordination.from_feature_points(config.KNEE_SIDE,
                                                       femur_medial_point,
                                                       femur_lateral_point,
                                                       femur_distal_point,
                                                       femur_proximal_point)
    tibia_coord = BoneCoordination.from_feature_points(config.KNEE_SIDE,
                                                       tibia_medial_point,
                                                       tibia_lateral_point,
                                                       tibia_distal_point,
                                                       tibia_proximal_point)
    return femur_coord, tibia_coord


def calc_contact_depth_map(frame_ray_directions, frame_contact_components):
    res = []
    for direction, components in tqdm.tqdm(zip(frame_ray_directions, frame_contact_components)):
        if config.DEPTH_BASE_BONE == config.BaseBone.FEMUR:
            direction = -direction
        depth_maps = []
        for component in components:
            depth_map = do_calc_contact_depth_map(component, direction)
            depth_maps.append(depth_map)
        res.append(depth_maps)
    return res


def do_calc_contact_depth_map(contact_component, v):
    origins, directions = prepare_rays_from_model(contact_component, v, True)

    # scene = trimesh.Scene([contact_component])
    # for origin, direction in zip(origins, directions):
    #     sphere = icosphere(radius=0.01)
    #     sphere.apply_translation(origin)
    #     scene.add_geometry(sphere)
    #
    #     ud = direction / np.linalg.norm(direction)
    #     end = origin + ud * 0.1
    #     segment = np.stack([origin, end], axis=0)
    #     arrow = cylinder(radius=0.005, segment=segment)
    #     scene.add_geometry(arrow)
    # scene.show()

    locations, ray_indices, _ = \
        contact_component.ray.intersects_location(origins, directions, multiple_hits=False)
    if len(ray_indices) == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def calc_bone_distance_map(fcs, tcs, fs, ts, vs):
    if fcs is None or tcs is None or any(fc is None for fc in fcs) or any(tc is None for tc in tcs):
        fcs, tcs = fs, ts  # use bone instead
    if config.DEPTH_BASE_BONE == config.BaseBone.FEMUR:
        return [do_calc_bone_distance_map(t, f, -v) for f, t, v in tqdm.tqdm(zip(fcs, tcs, vs))]
    elif config.DEPTH_BASE_BONE == config.BaseBone.TIBIA:
        return [do_calc_bone_distance_map(f, t, v) for f, t, v in tqdm.tqdm(zip(fcs, tcs, vs))]
    raise NotImplementedError(f'Unknown base bone: {config.DEPTH_BASE_BONE}')


def do_calc_bone_distance_map(target, base, v):
    origins, directions = prepare_rays_from_model(base, v)
    locations, ray_indices, _ = \
        target.ray.intersects_location(origins, directions, multiple_hits=False)
    origins = origins[ray_indices]
    depths = np.linalg.norm(locations - origins, axis=1)
    return origins, depths


def prepare_rays_from_model(model, direction, inward: bool = False, eps: float = 1e-4):
    ud = normalize(direction)
    mask = np.dot(model.vertex_normals, ud) > 0
    origins = model.vertices[mask]

    if inward:
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
            component_z = griddata(component_origins, component_thickness_map, (grid_x, grid_y), method='linear')
            im = ax.contourf(
                grid_x, grid_y, component_z,
                levels=np.arange(vmin, vmax, 1),
                cmap=depth_cmap,
                alpha=0.5,
            )
            ims.append(im)

        ax.imshow(background, extent=extent, interpolation='none', aspect='equal')
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
        plt.close(fig)
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
    R_T = H_T[:3, :3]  # T 在世界系的 3x3
    R_F = H_F[:3, :3]  # F 在世界系的 3x3

    R_T_trans = R_T.T  # R_T^T
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

    #  绕 X => 用 F 系的 Y=(0,1,0), 先转到 T系 => 投影YZ => 和 eT_y 投影做夹角
    vF_y = np.array([0, 1, 0], dtype=float)
    vT_y = R_TF @ vF_y  # F 的 Y 向量在 T 系下的表达
    angle_x = project_and_signed_angle(vT_y, eT_y, axis_idx=0)

    #  绕 Y => 用 F 系的 Z=(0,0,1), => 投影 XZ => 和 eT_z 做夹角
    vF_z = np.array([0, 0, 1], dtype=float)
    vT_z = R_TF @ vF_z
    angle_y = project_and_signed_angle(vT_z, eT_z, axis_idx=1)

    #  绕 Z => 用 F系的 X=(1,0,0), => 投影XY => 和 eT_x 做夹角
    vF_x = np.array([1, 0, 0], dtype=float)
    vT_x = R_TF @ vF_x
    angle_z = project_and_signed_angle(vT_x, eT_x, axis_idx=2)

    return angle_x, angle_y, angle_z


def smooth_transformations(raw: List[Transformation3D]) -> List[Transformation3D]:
    n = len(raw)
    x = np.arange(n)
    nx = 100
    xi = np.linspace(0, n - 1, nx)
    y_rx, y_ry, y_rz = np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real)
    y_tx, y_ty, y_tz = np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real)
    for i, t in enumerate(raw):
        tx, ty, tz = t.mat_t
        rx, ry, rz = Rotation.from_matrix(t.mat_r).as_euler('xyz')
        y_tx[i] = tx
        y_ty[i] = ty
        y_tz[i] = tz
        y_rx[i] = rx
        y_ry[i] = ry
        y_rz[i] = rz
    _, cls_interpolator = config.MOVEMENT_INTERPOLATE_METHOD.value

    dofi = []
    for y in [y_tx, y_ty, y_tz, y_rx, y_ry, y_rz]:
        interpolator = cls_interpolator(x, y)
        yi = interpolator(xi)
        dofi.append(yi)
    yi_tx, yi_ty, yi_tz, yi_rx, yi_ry, yi_rz = dofi

    smoothed = []
    for i in range(len(xi)):
        tx = yi_tx[i]
        ty = yi_ty[i]
        tz = yi_tz[i]
        rx = yi_rx[i]
        ry = yi_ry[i]
        rz = yi_rz[i]

        t = Transformation3D()
        t.set_translation(np.array([tx, ty, tz]))
        t.set_rotation(Rotation.from_euler(seq='xyz', angles=[rx, ry, rz]))
        smoothed.append(t)
    return smoothed


def plot_cartilage_thickness_curve(frame_cart_meshes, frame_deepest_points, frame_ray_direction, bone_name: str = '',
                                   first2: bool = False):
    res = {}
    for base, deepest_points in frame_deepest_points.items():
        medial, lateral = do_plot_cartilage_thickness_curve(base.value, frame_cart_meshes, deepest_points,
                                                            frame_ray_direction, bone_name, first2)
        res[base] = (medial, lateral)
    return res


def do_plot_cartilage_thickness_curve(
        base_name, frame_cart_meshes, frame_deepest_points, frame_ray_direction, bone_name: str = '',
        first2: bool = False):
    if frame_cart_meshes is None or len(frame_cart_meshes) == 0 or config.IGNORE_CARTILAGE:
        print("Cartilages ignored.")
        return None

    n = len(frame_cart_meshes)
    left_x, right_x = np.zeros((n,), dtype=Real), np.zeros((n,), dtype=Real)
    for frame_index in range(n):
        cart_mesh = frame_cart_meshes[frame_index]
        ray_direction = frame_ray_direction[frame_index]
        ray_direction = ray_direction.reshape(1, 3)

        thickness = [0, 0]
        for point_index, point in enumerate(frame_deepest_points[frame_index]):
            origin = np.array([point], dtype=Real) - ray_direction * 1e6
            hits, ray_indices, _ = \
                cart_mesh.ray.intersects_location(origin, ray_direction, multiple_hits=True)
            if len(ray_indices) >= 2:
                distances = np.linalg.norm(hits - origin, axis=1)
                indices = np.argsort(distances)
                if first2:
                    indices = indices[:2]
                else:
                    indices = indices[-2:]
                thickness[point_index] = np.linalg.norm(hits[indices[0]] - hits[indices[1]])
        left_x[frame_index], right_x[frame_index] = thickness
    fig, ax = plt.subplots()

    medial, lateral = right_x, left_x
    if config.KneeSide == config.KneeSide.RIGHT:
        medial, lateral = lateral, medial
    ax.plot(medial, label='Medial')
    ax.plot(lateral, label='Lateral')
    ax.legend()
    ax.set_title(f'Cartilage Thickness Curve - {bone_name.capitalize()} - Base {base_name}')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'{bone_name}_cartilage_thickness_base_{base_name}.jpg'))
    plt.close(fig)

    pd.DataFrame({
        'Frame': np.arange(n),
        "Medial": medial,
        'Lateral': lateral,
    }).to_csv(os.path.join(config.OUTPUT_DIRECTORY, f'{bone_name}_cartilage_thickness_base_{base_name}.csv'),
              index=False)

    return medial, lateral


def plot_normed_max_depth_curve(thickness_curve, max_depth_curve, name: str):
    res = {}
    for base in thickness_curve:
        res[base] = do_plot_normed_max_depth_curve(base.value, thickness_curve[base], max_depth_curve[base], name)
    return res


def do_plot_normed_max_depth_curve(base_name: str, thickness_curve, max_depth_curve, name: str):
    if thickness_curve is None or max_depth_curve is None:
        print("normed max depth curve no data", file=sys.stderr)
        return None
    thickness_medial, thickness_lateral = thickness_curve
    _, max_depth_medial, max_depth_lateral = max_depth_curve
    n = min(len(thickness_medial), len(thickness_lateral), len(max_depth_medial), len(max_depth_lateral))
    medial = np.zeros((n,), dtype=Real)
    lateral = np.zeros((n,), dtype=Real)
    for frame_index, (thickness_medial_i, thickness_lateral_i, max_depth_medial_i, max_depth_lateral_i) in \
            enumerate(zip(thickness_medial, thickness_lateral, max_depth_medial, max_depth_lateral)):
        medial[frame_index] = max_depth_medial_i / thickness_medial_i if thickness_medial_i != 0 else 0
        lateral[frame_index] = max_depth_lateral_i / thickness_lateral_i if thickness_lateral_i != 0 else 0
    fig, ax = plt.subplots()
    ax.plot(medial, label='Medial')
    ax.plot(lateral, label='Lateral')
    ax.legend()
    ax.set_title(f'Max Depth Curve - Normed by {name.capitalize()} - Base {base_name}')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'{name}_normed_max_depth_{base_name}.jpg'))
    plt.close(fig)
    return medial, lateral


def plot_contact_area_curve(
        frame_contact_components: list[list[trimesh.Trimesh]],
        frame_coordinates: dict[config.BaseBone, list[BoneCoordination]],
        frame_ray_directions: list[np.ndarray],
):
    res = {}
    for base in frame_coordinates:
        res[base] = do_plot_contact_area_curve(base.value, frame_contact_components, frame_coordinates[base],
                                               frame_ray_directions)
    return res


def do_plot_contact_area_curve(
        name: str,
        frame_contact_components: list[list[trimesh.Trimesh]],
        frame_coordinates: list[BoneCoordination],
        frame_ray_directions: list[np.ndarray],
):
    medial, lateral = [], []
    for frame_index, (contact_components, coordination, ray_direction) \
            in enumerate(zip(frame_contact_components, frame_coordinates, frame_ray_directions)):
        if len(contact_components) == 0:
            medial.append(0)
            lateral.append(0)
            continue

        medial_area, lateral_area = 0, 0
        for contact_component in contact_components:
            dot = contact_component.face_normals @ ray_direction
            visible_faces = np.where(dot < 0)[0]
            if len(visible_faces) == 0:
                continue
            visible_triangles = contact_component.triangles[visible_faces]
            projected_triangles = coordination.project(visible_triangles)

            def area_of_tri(tri):
                a, b = tri[1] - tri[0], tri[2] - tri[0]
                return 0.5 * np.linalg.norm(np.cross(a, b))

            total_area = np.sum(np.array([area_of_tri(tri) for tri in projected_triangles]))

            component_pos = coordination.project(contact_component.centroid)[0]
            if (component_pos > 0 and config.KNEE_SIDE == config.KneeSide.LEFT) \
                    or (component_pos < 0 and config.KNEE_SIDE == config.KneeSide.RIGHT):
                medial_area += total_area
            else:
                lateral_area += total_area
        medial.append(medial_area)
        lateral.append(lateral_area)

    fig, ax = plt.subplots()
    ax.plot(medial, label='Medial')
    ax.plot(lateral, label='Lateral')
    ax.legend()
    ax.set_title(f'Contact Area Curve - Base {name.capitalize()}')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'contact_area_base_{name}.jpg'))
    plt.close(fig)

    return medial, lateral


def dump_all_data(
        task_dof_data_raw, task_dof_data_smoothed, task_area_curve, task_frame_deepest_points, task_max_depth_curve,
):
    dof = task_dof_data_raw
    if config.MOVEMENT_SMOOTH:
        dof = task_dof_data_smoothed
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof
    n = len(y_tx)
    x = np.arange(1, n + 1)
    for base in [config.BaseBone.FEMUR, config.BaseBone.TIBIA]:
        area_medial, area_lateral = task_area_curve[base]
        print(len(area_medial), area_medial)
        deepest_points = task_frame_deepest_points[base]
        max_depth, max_depth_medial, max_depth_lateral = task_max_depth_curve[base]
        csv_path = os.path.join(config.OUTPUT_DIRECTORY, f'{base.value}_all_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\', )
            writer.writerow(['Frame', 'Translation X', 'Translation Y', 'Translation Z',
                             'Rotation X', 'Rotation Y', 'Rotation Z', 'Area Medial', 'Area Lateral',
                             'Deepest Point Medial X', 'Deepest Point Medial X', 'Deepest Point Medial Z',
                             'Depth', 'Depth Medial', 'Depth Lateral'])
            for i in range(n):
                writer.writerow([
                    x[i],
                    y_tx[i],
                    y_ty[i],
                    y_tz[i],
                    y_rx[i],
                    y_ry[i],
                    y_rz[i],
                    area_medial[i] if area_medial[i] is not None else 0,
                    area_lateral[i] if area_lateral[i] is not None else 0,
                    deepest_points[i][0][0] if deepest_points[i] and deepest_points[i][0] is not None else 'None',
                    deepest_points[i][0][1] if deepest_points[i] and deepest_points[i][0] is not None else 'None',
                    deepest_points[i][0][2] if deepest_points[i] and deepest_points[i][0] is not None else 'None',
                    deepest_points[i][1][0] if deepest_points[i] and deepest_points[i][1] is not None else 'None',
                    deepest_points[i][1][1] if deepest_points[i] and deepest_points[i][1] is not None else 'None',
                    deepest_points[i][1][2] if deepest_points[i] and deepest_points[i][1] is not None else 'None',
                    max_depth[i] if max_depth[i] is not None else 0,
                    max_depth_medial[i] if max_depth_medial[i] is not None else 0,
                    max_depth_lateral[i] if max_depth_lateral[i] is not None else 0,
                ])


if __name__ == "__main__":
    main()
