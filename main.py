import functools
import itertools
import logging
import csv
import os
import platform

import numpy as np
import trimesh

import config
import task
from animation import transform_frame_mesh, gen_bone_animation_frames, gen_movement_animation, \
    transform_frame_fixed_points
from area import plot_contact_area_curve
from bone import calc_frame_coordinates
from contact import get_contact_area, get_contact_components
from depth_map import gen_contact_depth_map_extent, gen_contact_depth_map_background, gen_contact_depth_map_mask, \
    calc_frame_contact_plane_normal_vectors, calc_bone_distance_map, calc_contact_depth_map, plot_max_depth_curve, \
    plot_min_distance_curve, calc_contact_deepest_points, plot_contact_depth_maps, gen_depth_map_animation, \
    do_plot_deepest_points, plot_deepest_points, plot_fixed_points
from dof import calc_dof, dump_dof, plot_dof_curves
from myio import load_mesh, load_coord_from_file, load_frame_bone_coordinates_csv, load_frame_bone_coordinates_raw, \
    load_fixed_points
from render import calc_extents
from smooth import smooth_transformations
from thickness import plot_cartilage_thickness_curve, plot_cartilage_thickness_curve_sum, plot_deformity_curve
from utils import my_axis, mesh_union, take_kth, list_take_kth

# force off-screen renderer if a Linux system is used
if platform.system() == "Linux":
    os.environ["PYOPENGL_PLATFORM"] = "egl"

trimesh.util.attach_to_log(level=logging.INFO)

WORLD_AXIS = my_axis(axis_length=1000, axis_radius=2)


def main():
    check_config()

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

    task_fixed_points = ctx.add_task('fixed_points', load_fixed_points, deps=[])

    task_original_coordinates = \
        ctx.add_task('original_coordinates', functools.partial(load_coord_from_file, config.FEATURE_POINT_FILE))
    task_original_coordinates_femur = \
        ctx.add_task('original_coordinates_femur', take_kth(0), deps=[task_original_coordinates])
    task_original_coordinates_tibia = \
        ctx.add_task('original_coordinates_tibia', take_kth(1), deps=[task_original_coordinates])

    task_depth_map_base_meshes = ctx.add_task('depth_map_base_meshes',
                                              lambda fm, tm: {config.BoneType.FEMUR: fm, config.BoneType.TIBIA: tm},
                                              deps=[task_femur_mesh, task_tibia_mesh])
    task_depth_map_base_coords = ctx.add_task('depth_map_base_coords',
                                              lambda fm, tm: {config.BoneType.FEMUR: fm, config.BoneType.TIBIA: tm},
                                              deps=[task_original_coordinates_femur, task_original_coordinates_tibia])
    task_depth_map_base_cart_meshes = ctx.add_task('depth_map_base_cart_meshes',
                                                   lambda fm, tm: {config.BoneType.FEMUR: fm,
                                                                   config.BoneType.TIBIA: tm},
                                                   deps=[task_femur_cart_mesh, task_tibia_cart_mesh])

    task_extents = ctx.add_task('extents', calc_extents, deps=[task_depth_map_base_meshes, task_depth_map_base_coords])
    task_depth_map_extent = ctx.add_task('contact_depth_map_extent', gen_contact_depth_map_extent, deps=[task_extents])
    task_depth_map_background = ctx.add_task('depth_map_background', gen_contact_depth_map_background, deps=[
        task_depth_map_extent, task_depth_map_base_meshes, task_depth_map_base_coords, task_depth_map_base_cart_meshes,
    ])
    task_depth_map_heatmap_mask = ctx.add_task('depth_map_heatmap_mask', gen_contact_depth_map_mask, deps=[
        task_depth_map_extent, task_depth_map_base_meshes, task_depth_map_base_coords, task_depth_map_base_cart_meshes,
    ])

    if config.MOVEMENT_DATA_FORMAT == config.MomentDataFormat.CSV:
        task_frame_bone_coordinates_raw = ctx.add_task('frame_bone_coordinates', load_frame_bone_coordinates_csv)
    elif config.MOVEMENT_DATA_FORMAT == config.MomentDataFormat.JSON:
        task_frame_bone_coordinates_raw = ctx.add_task('frame_bone_coordinates', load_frame_bone_coordinates_raw)
    else:
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
                                                   lambda fm, tm: {config.BoneType.FEMUR: fm,
                                                                   config.BoneType.TIBIA: tm},
                                                   deps=[task_frame_bone_transformations_femur,
                                                         task_frame_bone_transformations_tibia])

    task_frame_coordinates = ctx.add_task('frame_coordinates', calc_frame_coordinates,
                                          deps=[task_depth_map_base_coords, task_frame_bone_transformations])
    task_frame_femur_coordinates = ctx.add_task('frame_femur_coordinates', take_kth(config.BoneType.FEMUR),
                                                deps=[task_frame_coordinates])
    task_frame_tibia_coordinates = ctx.add_task('frame_tibia_coordinates', take_kth(config.BoneType.TIBIA),
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
    task_frame_fixed_points = ctx.add_task('frame_fixed_points', transform_frame_fixed_points,
                                           deps=[task_fixed_points, task_frame_bone_transformations])

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

    z_axis_base_map = {config.DepthDirection.Z_AXIS_TIBIA: config.BoneType.TIBIA,
                       config.DepthDirection.Z_AXIS_FEMUR: config.BoneType.FEMUR}
    if config.DEPTH_DIRECTION in z_axis_base_map:
        def job(_, frame_coordinates):
            return [coord.t.unit_z for coord in frame_coordinates[z_axis_base_map[config.DEPTH_DIRECTION]]]
    elif config.DEPTH_DIRECTION == config.DepthDirection.CONTACT_PLANE:
        job = calc_frame_contact_plane_normal_vectors
    elif config.DEPTH_DIRECTION == config.DepthDirection.VERTEX_NORMAL:
        raise NotImplementedError("VERTEX_NORMAL is no longer supported")
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
                                                     task_depth_map_heatmap_mask,
                                                     task_frame_coordinates,
                                                     task_frame_bone_distance_map_origins,
                                                     task_frame_bone_distance_map_distances,
                                                     task_frame_contact_components,
                                                     task_frame_contact_component_depth_map_origins,
                                                     task_frame_contact_component_depth_map_depths,
                                                     task_frame_deepest_points,
                                                 ])

    task_plot_deepest_points = ctx.add_task('plot_deepest_points', plot_deepest_points, deps=[
        task_depth_map_background, task_depth_map_extent, task_frame_deepest_points, task_frame_coordinates,
    ])
    task_plot_fixed_points = ctx.add_task('plot_fixed_points', plot_fixed_points, deps=[
        task_depth_map_background, task_depth_map_extent, task_frame_fixed_points, task_frame_coordinates,
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
    task_frame_cart_thickness_curve = \
        ctx.add_task('frame_cart_thickness', plot_cartilage_thickness_curve_sum,
                     deps=[task_frame_femur_cart_thickness_curve, task_frame_tibia_cart_thickness_curve])

    task_femur_deformity_curve = ctx.add_task(
        'femur_deformity_curve',
        functools.partial(plot_deformity_curve, name='femur'),
        deps=[
            task_frame_femur_cart_thickness_curve,
            task_max_depth_curve,
        ])
    task_tibia_deformity_curve = ctx.add_task(
        'tibia_deformity_curve',
        functools.partial(plot_deformity_curve, name='tibia'),
        deps=[
            task_frame_tibia_cart_thickness_curve,
            task_max_depth_curve,
        ])
    task_deformity_curve = ctx.add_task(
        'deformity_curve',
        functools.partial(plot_deformity_curve, name='sum'),
        deps=[
            task_frame_cart_thickness_curve,
            task_max_depth_curve,
        ])

    task_area_curve = ctx.add_task('area_curve', plot_contact_area_curve, deps=[
        task_frame_contact_components, task_frame_coordinates,
    ])

    task_dump_all_data = ctx.add_task('dump_all_data', dump_all_data, deps=[
        task_dof_data_raw, task_dof_data_smoothed, task_area_curve, task_frame_deepest_points, task_max_depth_curve,
        task_femur_deformity_curve, task_tibia_deformity_curve, task_deformity_curve,
    ])

    if config.GENERATE_CART_THICKNESS_CURVE:
        task_frame_femur_cart_thickness_curve()
        task_frame_tibia_cart_thickness_curve()
    if config.GENERATE_ANIMATION:
        task_movement_animation()
    if config.GENERATE_DEPTH_CURVE:
        task_max_depth_curve()
    if config.GENERATE_DEPTH_MAP:
        task_plot_deepest_points()
        task_contact_depth_map_animation()
    if config.GENERATE_DOF_CURVES:
        task_plot_dof_curves()
        task_dump_dof()
    if config.GENERATE_NORM_DEPTH_CURVE:
        task_tibia_deformity_curve()
        task_femur_deformity_curve()
        task_deformity_curve()
    if config.GENERATE_AREA_CURVE:
        task_area_curve()
    if config.DUMP_ALL_DATA:
        task_dump_all_data()
    if config.GENERATE_DISTANCE_CURVE:
        task_min_distance_curve()
    if config.GENERATE_FIXED_POINT_PLOT:
        task_plot_fixed_points()


def dump_all_data(
        task_dof_data_raw, task_dof_data_smoothed, task_area_curve, task_frame_deepest_points, task_max_depth_curve,
        task_femur_deformity_curve, task_tibia_deformity_curve, task_deformity_curve):
    dof = task_dof_data_raw
    if config.MOVEMENT_SMOOTH:
        dof = task_dof_data_smoothed
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof
    n = len(y_tx)
    x = np.arange(1, n + 1)
    for base in [config.BoneType.FEMUR, config.BoneType.TIBIA]:
        area_medial, area_lateral = task_area_curve[base]
        deepest_points = task_frame_deepest_points[base]
        max_depth, max_depth_medial, max_depth_lateral = task_max_depth_curve[base]
        femur_deformity_medial, femur_deformity_lateral = task_femur_deformity_curve[base] if task_femur_deformity_curve is not None else (None, None)
        tibia_deformity_medial, tibia_deformity_lateral = task_tibia_deformity_curve[base] if task_tibia_deformity_curve is not None else (None, None)
        deformity_medial, deformity_lateral = task_deformity_curve[base] if task_deformity_curve is not None else (None, None)
        csv_path = os.path.join(config.OUTPUT_DIRECTORY, f'{base.value}_all_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\', )
            writer.writerow([
                'Frame', 'Translation X', 'Translation Y', 'Translation Z',
                'Rotation X', 'Rotation Y', 'Rotation Z', 'Area Medial', 'Area Lateral',
                'Deepest Point Medial X', 'Deepest Point Medial Y', 'Deepest Point Medial Z',
                'Deepest Point Lateral X', 'Deepest Point Lateral Y', 'Deepest Point Lateral Z',
                'Depth', 'Depth Medial', 'Depth Lateral',
                'Femur Deformity Medial', 'Femur Deformity Lateral',
                'Tibia Deformity Medial', 'Tibia Deformity Lateral',
                'Deformity Medial', 'Deformity Lateral',
            ])
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
                    femur_deformity_medial[i] if femur_deformity_medial is not None and femur_deformity_medial[i] is not None else 0,
                    femur_deformity_lateral[i] if femur_deformity_lateral is not None and femur_deformity_lateral[i] is not None else 0,
                    tibia_deformity_medial[i] if tibia_deformity_medial is not None and tibia_deformity_medial[i] is not None else 0,
                    tibia_deformity_lateral[i] if tibia_deformity_lateral is not None and tibia_deformity_lateral[i] is not None else 0,
                    deformity_medial[i] if deformity_medial is not None and deformity_medial[i] is not None else 0,
                    deformity_lateral[i] if deformity_lateral is not None and deformity_lateral[i] is not None else 0,
                ])


def check_config():
    if config.DEPTH_DIRECTION == config.DepthDirection.CONTACT_PLANE and config.IGNORE_CARTILAGE:
        raise ValueError("CONTACT_PLANE must not be used with IGNORE_CARTILAGE")
    if config.DEPTH_DIRECTION == config.DepthDirection.VERTEX_NORMAL:
        raise ValueError("VERTEX_NORMAL is no longer supported")


if __name__ == "__main__":
    main()
