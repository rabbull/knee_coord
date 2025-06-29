import csv
import json

import numpy as np
import trimesh

import config
from bone import BoneCoordination
from utils import Real


def load_mesh(path: None | str):
    if path:
        return trimesh.load_mesh(path)
    return None


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

def load_coord_from_file(path) -> tuple[BoneCoordination, BoneCoordination]:
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
                                                       femur_proximal_point,
                                                       femur_distal_point,
                                                       config.BoneType.FEMUR)
    tibia_coord = BoneCoordination.from_feature_points(config.KNEE_SIDE,
                                                       tibia_medial_point,
                                                       tibia_lateral_point,
                                                       tibia_proximal_point,
                                                       tibia_distal_point,
                                                       config.BoneType.TIBIA)
    return femur_coord, tibia_coord


def load_fixed_points():
    return config.FIXED_POINTS