import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import config
from utils import safe_div, Real


def plot_cartilage_thickness_curve(frame_cart_meshes, frame_deepest_points, frame_ray_direction, bone_name: str = '',
                                   first2: bool = False):
    if frame_cart_meshes is None or len(frame_cart_meshes) == 0 or config.IGNORE_CARTILAGE:
        print("Cartilages ignored.")
        return None
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

    medial, lateral = right_x, left_x
    if config.KneeSide == config.KneeSide.RIGHT:
        medial, lateral = lateral, medial

    fig, ax = plt.subplots()
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


def plot_cartilage_thickness_curve_sum(tibia_cart_thickness, femur_cart_thickness):
    res = {}
    for base in tibia_cart_thickness:
        tibia_medial, tibia_lateral = tibia_cart_thickness[base]
        femur_medial, femur_lateral = femur_cart_thickness[base]
        medial = tibia_medial + femur_medial
        lateral = tibia_lateral + femur_lateral
        fig, ax = plt.subplots()
        ax.plot(medial, label='Medial')
        ax.plot(lateral, label='Lateral')
        ax.legend()
        ax.set_title(f'Cartilage Thickness Curve - Sum - Base {base}')
        fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'cartilage_thickness_base_{base}_sum.jpg'))
        plt.close(fig)
        res[base] = (medial, lateral)
    return res


def plot_deformity_curve(thickness_curve, max_depth_curve, name: str):
    if thickness_curve is None or max_depth_curve is None:
        return None
    res = {}
    for base in thickness_curve:
        res[base] = do_plot_deformity_curve(base.value, thickness_curve[base], max_depth_curve[base], name)
    return res


def do_plot_deformity_curve(base_name: str, thickness_curve, max_depth_curve, name: str):
    if thickness_curve is None or max_depth_curve is None:
        print("normed max depth curve no archive", file=sys.stderr)
        return None
    thickness_medial, thickness_lateral = thickness_curve
    _, max_depth_medial, max_depth_lateral = max_depth_curve
    n = min(len(thickness_medial), len(thickness_lateral), len(max_depth_medial), len(max_depth_lateral))
    medial = np.zeros((n,), dtype=Real)
    lateral = np.zeros((n,), dtype=Real)
    for frame_index, (thickness_medial_i, thickness_lateral_i, max_depth_medial_i, max_depth_lateral_i) in \
            enumerate(zip(thickness_medial, thickness_lateral, max_depth_medial, max_depth_lateral)):
        thickness = thickness_medial_i + thickness_lateral_i
        medial[frame_index] = safe_div(max_depth_medial_i, thickness)
        lateral[frame_index] = safe_div(max_depth_lateral_i, thickness)
    fig, ax = plt.subplots()
    ax.plot(medial, label='Medial')
    ax.plot(lateral, label='Lateral')
    ax.legend()
    ax.set_title(f'Deformity Curve - Normed by {name.capitalize()} - Base {base_name}')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'{name}_normed_max_depth_{base_name}.jpg'))
    plt.close(fig)
    return medial, lateral
