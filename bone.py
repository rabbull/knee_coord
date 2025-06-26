import copy
from typing import Self

import numpy as np
import tqdm
from scipy.spatial.transform import Rotation

from utils import Transformation3D, normalize
import config


class BoneCoordination:
    def __init__(self):
        self._t = Transformation3D()
        self._extra = {}

    @classmethod
    def from_feature_points(cls, side: config.KneeSide, medial_point: np.ndarray, lateral_point: np.ndarray,
                            proximal_point: np.ndarray, distal_point: np.ndarray, bone_type: config.BoneType,
                            extra=None) -> Self:
        self = cls()

        if side == config.KneeSide.LEFT:
            left_point, right_point = lateral_point, medial_point
        elif side == config.KneeSide.RIGHT:
            left_point, right_point = medial_point, lateral_point
        else:
            raise NotImplementedError
        del medial_point, lateral_point

        self._origin = (left_point + right_point) / 2
        self._t.set_translation(self._origin)

        if bone_type == config.BoneType.FEMUR:
            f = self.__set_rotation_femur
        elif bone_type == config.BoneType.TIBIA:
            f = self.__set_rotation_tibia
        else:
            raise ValueError
        f(left_point, right_point, proximal_point, distal_point)

        self._extra = extra if extra else {}
        return self

    def __set_rotation_femur(self, left, right, proximal, distal):
        raw_x = right - left
        unit_x = normalize(raw_x)
        raw_z = proximal - distal
        raw_y = np.cross(raw_z, unit_x)
        unit_y = normalize(raw_y)
        unit_z = normalize(np.cross(unit_x, unit_y))

        mat_r = np.column_stack((unit_x, unit_y, unit_z))
        self._t.set_rotation(Rotation.from_matrix(mat_r))

    def __set_rotation_tibia(self, left, right, proximal, distal):
        raw_z = proximal - distal
        unit_z = normalize(raw_z)
        raw_x = right - left
        raw_y = np.cross(unit_z, raw_x)
        unit_y = normalize(raw_y)
        unit_x = normalize(np.cross(unit_y, unit_z))

        mat_r = np.column_stack((unit_x, unit_y, unit_z))
        self._t.set_rotation(Rotation.from_matrix(mat_r))

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
