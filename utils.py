from typing import Union

import numpy as np
import trimesh
from nptyping import Shape
from nptyping.ndarray import NDArray
from scipy.spatial.transform import Rotation

Real: type = np.float64
RotationMatrix: type = NDArray[Shape['3, 3'], Real]
TranslationMatrix: type = NDArray[Shape['3'], Real]
HomogeneousMatrix: type = NDArray[Shape['4, 4'], Real]
Quaternion: type = NDArray[Shape['4'], Real]
EulerAngles: type = NDArray[Shape['3'], Real]
Point3D: type = NDArray[Shape['3'], Real]
PointList3D: type = NDArray[Shape['*, 3'], Real]

X = np.array([1, 0, 0], dtype=np.float64)
Y = np.array([0, 1, 0], dtype=np.float64)
Z = np.array([0, 0, 1], dtype=np.float64)


def _translate(axis, dist):
    t = np.eye(4)
    t[axis, -1] = dist
    return t


AXIS_X = trimesh.creation.box(extents=[10, 1, 1], transform=_translate(0, 5.))
AXIS_Y = trimesh.creation.box(extents=[1, 15, 1], transform=_translate(1, 7.5))
AXIS_Z = trimesh.creation.box(extents=[1, 1, 20], transform=_translate(2, 10.))
AXES = [AXIS_X, AXIS_Y, AXIS_Z]


def normalize(x: NDArray[Shape['*, ...'], Real]):
    return x / np.linalg.norm(x)


class Transformation3D(object):
    def __init__(self, h: HomogeneousMatrix = None):
        if h is None:
            h = np.eye(4)
        self._h = h

    def set_translation(self, translation: TranslationMatrix) -> 'Transformation3D':
        if isinstance(translation, TranslationMatrix):  # noqa
            self._h[:3, 3] = translation
        else:
            raise ValueError
        return self

    def set_rotation(self, rotation: Union[Rotation, RotationMatrix]) -> 'Transformation3D':
        if isinstance(rotation, Rotation):
            r = rotation.as_matrix()
        elif isinstance(rotation, RotationMatrix):  # noqa
            r = rotation
        else:
            raise ValueError
        self._h[:3, :3] = r
        return self

    @property
    def mat_r(self) -> RotationMatrix:
        return self._h[:3, :3]

    @property
    def mat_t(self) -> TranslationMatrix:
        return self._h[:3, 3]

    @property
    def quaternion(self) -> Quaternion:
        return Rotation.from_matrix(self.mat_r).as_quat()

    @property
    def euler_angles(self, seq='zyx') -> EulerAngles:
        return Rotation.from_matrix(self.mat_r).as_euler(seq, degrees=False)

    def apply_rotation(self, rotation: Union[Rotation, RotationMatrix]) -> 'Transformation3D':
        return self.apply_transformation(Transformation3D().set_rotation(rotation))

    def apply_translation(self, translation: TranslationMatrix) -> 'Transformation3D':
        return self.apply_transformation(Transformation3D().set_translation(translation))

    def apply_transformation(self, rhs: 'Transformation3D') -> 'Transformation3D':
        self._h = self._h @ rhs._h
        return self

    def transform(self, x: Union[PointList3D, Point3D]) -> Union[PointList3D, Point3D]:
        if isinstance(x, Point3D):  # noqa
            x_hom = np.hstack([x, 1])
            x_transformed = self._h @ x_hom
            return x_transformed[:3]

        elif isinstance(x, PointList3D):  # noqa
            x_hom = np.hstack([x, np.ones((x.shape[0], 1))])
            x_transformed = (self._h @ x_hom.T).T
            return x_transformed[:, :3]

        else:
            raise ValueError

    def inverse(self) -> 'Transformation3D':
        return Transformation3D(np.linalg.inv(self._h))

    def relative_to(self, rhs: 'Transformation3D') -> 'Transformation3D':
        return Transformation3D(np.linalg.inv(rhs._h) @ self._h)
