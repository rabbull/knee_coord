import csv
from typing import IO, Union, Dict, AnyStr, Self

import numpy as np
from scipy.spatial.transform import Rotation

from utils import *

class KneePose:
    femoral_coord: Transformation3D
    tibial_coord: Transformation3D

    def __init__(self):
        self.femoral_coord = Transformation3D()
        self.tibial_coord = Transformation3D()

    def copy(self):
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def _read_from_file(fp: IO) -> Dict[AnyStr, Real]:
        points = dict()
        for row in csv.DictReader(fp, delimiter='\t'):
            for label in ['X', 'Y', 'Z', '']:
                if label not in row:
                    raise ValueError
            points[row['']] = np.array([row['X'], row['Y'], row['Z']], dtype='double')
        return points

    def update(self, fp: IO):
        points = self._read_from_file(fp)
        self.femoral_coord = self.from_feature(
            points['Femoral Lateral tendon end point'],
            points['Femoral Medial tendon end point'],
            points['Femoral Upper point of Long bone axis'],
            points['Femoral Lower point of Long bone axis'],
        )
        self.tibial_coord = self.from_feature(
            points['Tibial Lateral tendon end point'],
            points['Tibial Medial tendon end point'],
            points['Tibial Upper point of Long bone axis'],
            points['Tibial Lower point of Long bone axis'],
        )

    def relative_to(self, rhs: 'KneePose'):
        relative_femoral = self.femoral_coord.relative_to(rhs.femoral_coord)
        relative_tibial = self.tibial_coord.relative_to(rhs.tibial_coord)
        return relative_femoral, relative_tibial

    @classmethod
    def from_feature(cls: Self, medial: Point3D, lateral: Point3D,
                     up: Point3D, down: Point3D) -> Transformation3D:
        transformation = Transformation3D()

        origin = (medial + lateral) / 2
        transformation.set_translation(origin)

        x = medial - lateral
        unit_x = normalize(x)

        z_raw = up - down
        z_raw_y_proj = np.dot(z_raw, unit_x) * unit_x
        z = z_raw - z_raw_y_proj
        unit_z = normalize(z)

        y = np.cross(unit_z, unit_x)
        unit_y = normalize(y)  # re-normalize in case of float precision error

        transformation.set_rotation(np.concatenate([
            unit_x.reshape(3, 1),
            unit_y.reshape(3, 1),
            unit_z.reshape(3, 1),
        ], axis=1))

        return transformation


def main():
    original_knee = KneePose()
    transformed_knee = KneePose()

    with open('entry_task/PointList.txt', 'r') as f:
        original_knee.update(f)
    with open('RT_Transform/PointList.txt', 'r') as f:
        transformed_knee.update(f)

    print(original_knee.femoral_coord._h)
    print(transformed_knee.femoral_coord._h)
    print()

    femoral_transform, tibial_transform = transformed_knee.relative_to(original_knee)
    print(femoral_transform.mat_t)
    print(femoral_transform.euler_angles / np.pi * 180)


def main2():
    key = 'Femoral Lower point of Long bone axis'
    with open('entry_task/PointList.txt', 'r') as fp:
        points = KneePose._read_from_file(fp)
    orig = points[key]
    with open('RT_Transform/PointList.txt', 'r') as fp:
        points = KneePose._read_from_file(fp)
    target = points[key]
    print(orig)
    print(target)

    t = Transformation3D()
    t.apply_rotation(Rotation.from_euler('y', -40, degrees=True))
    t.apply_translation(np.array([0., -2., 0.]))
    t.apply_rotation(Rotation.from_euler('y', -20, degrees=True))
    t.apply_translation(np.array([-5., 0., 0.]))
    t.apply_rotation(Rotation.from_euler('x', 45, degrees=True))
    transformed = t.transform(orig)
    print(transformed)


if __name__ == '__main__':
    main2()
