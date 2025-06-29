import os
from collections import defaultdict
from PIL import Image

import tqdm
import numpy as np
import trimesh

import config
from render import gen_orthographic_photo, gen_animation
from utils import hex_to_rgba1, my_axis, Real, get_frame_output_directory, Transformation3D


def transform_frame_mesh(mesh: None | trimesh.Trimesh, transformations: list[Transformation3D]):
    if mesh:
        return [mesh.copy().apply_transform(transformation.mat_homo) for transformation in transformations]
    return [None for _ in transformations]


def transform_frame_fixed_points(
        fixed_points: dict[config.BoneType, dict[str, list[float]]],
        transformations: dict[config.BoneType, list[Transformation3D]],
) -> dict[config.BoneType, dict[str, list[list[float]]]]:
    transformed = {}
    for bone_type, bone_fixed_points in fixed_points.items():
        transformed[bone_type] = defaultdict(list)
        for name, point in bone_fixed_points.items():
            for transformation in transformations[bone_type]:
                transformed[bone_type][name].append(transformation.transform(point))
    return transformed


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


def gen_movement_animation(images: dict[config.AnimationCameraDirection, list[Image.Image]]):
    for direction, frames in images.items():
        gen_animation(frames, name=f'animation_{direction.value}', duration=config.ANIMATION_DURATION)
