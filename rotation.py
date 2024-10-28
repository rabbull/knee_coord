import csv
import numpy as np
from typing import IO

from mpl_toolkits.mplot3d.proj3d import transform


class Transformation(object):
    def __init__(self):
        self.transformation_matrix = np.eye(4)

    def set_translation(self, mat_t: np.ndarray):
        self.transformation_matrix[:3, 3] = mat_t

    def set_rotation(self, mat_r: np.ndarray):
        self.transformation_matrix[:3, :3] = mat_r

    @property
    def mat_r(self):
        return self.transformation_matrix[:3, :3]

    @property
    def mat_t(self):
        return self.transformation_matrix[:3, 3].reshape([3, 1])

    @property
    def quaternion(self):
        mat_r = self.mat_r
        trace = np.trace(mat_r)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat_r[2, 1] - mat_r[1, 2]) * s
            y = (mat_r[0, 2] - mat_r[2, 0]) * s
            z = (mat_r[1, 0] - mat_r[0, 1]) * s
        else:
            if mat_r[0, 0] > mat_r[1, 1] and mat_r[0, 0] > mat_r[2, 2]:
                s = 2.0 * np.sqrt(1.0 + mat_r[0, 0] - mat_r[1, 1] - mat_r[2, 2])
                w = (mat_r[2, 1] - mat_r[1, 2]) / s
                x = 0.25 * s
                y = (mat_r[0, 1] + mat_r[1, 0]) / s
                z = (mat_r[0, 2] + mat_r[2, 0]) / s
            elif mat_r[1, 1] > mat_r[2, 2]:
                s = 2.0 * np.sqrt(1.0 + mat_r[1, 1] - mat_r[0, 0] - mat_r[2, 2])
                w = (mat_r[0, 2] - mat_r[2, 0]) / s
                x = (mat_r[0, 1] + mat_r[1, 0]) / s
                y = 0.25 * s
                z = (mat_r[1, 2] + mat_r[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + mat_r[2, 2] - mat_r[0, 0] - mat_r[1, 1])
                w = (mat_r[1, 0] - mat_r[0, 1]) / s
                x = (mat_r[0, 2] + mat_r[2, 0]) / s
                y = (mat_r[1, 2] + mat_r[2, 1]) / s
                z = 0.25 * s
        return np.array([x, y, z, w])

    @property
    def euler_angles(self, sequence='zyx'):
        r = self.mat_r

        if sequence == 'zyx':
            sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(r[2, 1], r[2, 2])
                y = np.arctan2(-r[2, 0], sy)
                z = np.arctan2(r[1, 0], r[0, 0])
            else:
                x = np.arctan2(-r[1, 2], r[1, 1])
                y = np.arctan2(-r[2, 0], sy)
                z = 0
            return np.array([z, y, x])

        elif sequence == 'xyz':
            sy = np.sqrt(r[2, 2] ** 2 + r[1, 2] ** 2)
            singular = sy < 1e-6
            if not singular:
                z = np.arctan2(r[1, 0], r[0, 0])
                y = np.arctan2(-r[2, 0], sy)
                x = np.arctan2(r[2, 1], r[2, 2])
            else:
                z = np.arctan2(-r[0, 1], r[0, 0])
                y = np.arctan2(-r[2, 0], sy)
                x = 0
            return np.array([x, y, z])

        elif sequence == 'zxy':
            sz = np.sqrt(r[1, 1] ** 2 + r[2, 1] ** 2)
            singular = sz < 1e-6
            if not singular:
                x = np.arctan2(-r[2, 1], r[1, 1])
                z = np.arctan2(-r[0, 2], r[0, 0])
                y = np.arctan2(r[0, 1], sz)
            else:
                x = np.arctan2(-r[0, 1], r[1, 1])
                z = np.arctan2(-r[0, 2], sz)
                y = 0
            return np.array([z, x, y])

        elif sequence == 'yxz':
            sx = np.sqrt(r[2, 2] ** 2 + r[0, 2] ** 2)
            singular = sx < 1e-6
            if not singular:
                y = np.arctan2(r[0, 2], r[2, 2])
                x = np.arctan2(-r[1, 2], sx)
                z = np.arctan2(r[1, 0], r[1, 1])
            else:
                y = np.arctan2(-r[0, 1], r[0, 0])
                x = np.arctan2(-r[1, 2], sx)
                z = 0
            return np.array([y, x, z])

        else:
            raise ValueError(f"Unsupported rotation sequence: {sequence}")

    def apply_rotation(self, mat_r: np.ndarray):
        self.transformation_matrix[:3, :3] = mat_r @ self.transformation_matrix[:3, :3]

    def apply_translation(self, mat_t: np.ndarray):
        self.transformation_matrix[:3, 3] += mat_t.reshape(3)

    def apply_transformation(self, rhs: 'Transformation'):
        self.transformation_matrix = rhs.transformation_matrix @ self.transformation_matrix

    def apply_quaternion(self, quaternion):
        x, y, z, w = quaternion
        rotation_matrix = np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])

        self.transformation_matrix[:3, :3] = rotation_matrix @ self.transformation_matrix[:3, :3]

    def apply_euler_angles(self, euler_angles, sequence='zyx'):
        rx, ry, rz = euler_angles
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        rotation_matrix = np.eye(3)
        for axis in sequence.lower():
            if axis == 'x':
                rotation_matrix = rotation_matrix @ rot_x
            elif axis == 'y':
                rotation_matrix = rotation_matrix @ rot_y
            elif axis == 'z':
                rotation_matrix = rotation_matrix @ rot_z
            else:
                raise ValueError(f"Unsupported axis in rotation sequence: {axis}")

        self.transformation_matrix[:3, :3] = rotation_matrix @ self.transformation_matrix[:3, :3]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_hom = np.append(x, 1).reshape(4, 1)
        x_hom_new = self.transformation_matrix @ x_hom
        return x_hom_new[:3]

    def relative_to(self, rhs: 'Transformation'):
        other_inv = np.linalg.inv(rhs.transformation_matrix)
        relative_transformation = other_inv @ self.transformation_matrix
        result = Transformation()
        result.transformation_matrix = relative_transformation
        return result


class KneePose:
    femoral_coord: Transformation
    tibial_coord: Transformation

    def __init__(self):
        self.femoral_coord = Transformation()
        self.tibial_coord = Transformation()

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def update(self, fp: IO):
        points = dict()
        for row in csv.DictReader(fp, delimiter='\t'):
            for label in ['X', 'Y', 'Z', '']:
                if label not in row:
                    exit(1)
            points[row['']] = np.array([row['X'], row['Y'], row['Z']], dtype='double')
        self.femoral_coord = calculate_transformation(
            points['Femoral Lateral tendon end point'],
            points['Femoral Medial tendon end point'],
            points['Femoral Upper point of Long bone axis'],
            points['Femoral Lower point of Long bone axis'],
        )
        self.tibial_coord = calculate_transformation(
            points['Tibial Lateral tendon end point'],
            points['Tibial Medial tendon end point'],
            points['Tibial Upper point of Long bone axis'],
            points['Tibial Lower point of Long bone axis'],
        )

    def relative_to(self, rhs: 'KneePose'):
        relative_femoral = self.femoral_coord.relative_to(rhs.femoral_coord)
        relative_tibial = self.tibial_coord.relative_to(rhs.tibial_coord)
        return relative_femoral, relative_tibial


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def calculate_transformation(left: np.ndarray, right: np.ndarray, up: np.ndarray, down: np.ndarray) -> Transformation:
    transformation = Transformation()

    origin = (left + right) / 2
    transformation.set_translation(origin)

    z = left - right
    unit_z = normalize(z)

    y_raw = up - down
    y_raw_z_proj = np.dot(y_raw, unit_z) * unit_z
    y = y_raw - y_raw_z_proj
    unit_y = normalize(y)

    x = np.cross(unit_y, unit_z)
    unit_x = normalize(x)  # re-normalize in case of float precision error

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

    print(original_knee.femoral_coord.transformation_matrix)
    print(transformed_knee.femoral_coord.transformation_matrix)
    print()

    femoral_transform, tibial_transform = transformed_knee.relative_to(original_knee)
    print(femoral_transform.mat_t)
    print(femoral_transform.euler_angles / np.pi * 180)


if __name__ == '__main__':
    main()
