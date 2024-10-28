import csv
import numpy as np


class Coordination(object):
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

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_hom = np.append(x, 1).reshape(4, 1)
        x_hom_new = self.transformation_matrix @ x_hom
        return x_hom_new[:3]

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
    def euler_angles(self):
        mat_r = self.mat_r
        sy = np.sqrt(mat_r[0, 0] ** 2 + mat_r[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(mat_r[2, 1], mat_r[2, 2])
            y = np.arctan2(-mat_r[2, 0], sy)
            z = np.arctan2(mat_r[1, 0], mat_r[0, 0])
        else:
            x = np.arctan2(-mat_r[1, 2], mat_r[1, 1])
            y = np.arctan2(-mat_r[2, 0], sy)
            z = 0

        return np.array([x, y, z])


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def build_coord(left: np.ndarray, right: np.ndarray, up: np.ndarray, down: np.ndarray) -> Coordination:
    coord = Coordination()

    origin = (left + right) / 2
    coord.set_translation(origin)

    z = left - right
    unit_z = normalize(z)

    y_raw = up - down
    y_raw_z_proj = np.dot(y_raw, unit_z) * unit_z
    y = y_raw - y_raw_z_proj
    unit_y = normalize(y)

    x = np.cross(unit_y, unit_z)
    unit_x = normalize(x)  # re-normalize in case of float precision error

    coord.set_rotation(np.concatenate([
        unit_x.reshape(3, 1),
        unit_y.reshape(3, 1),
        unit_z.reshape(3, 1),
    ], axis=1))

    return coord


def main():
    point_list_file = open('PointList.txt', 'r')
    point_list_reader = csv.DictReader(point_list_file, delimiter='\t')

    points = dict()
    for row in point_list_reader:
        for label in ['X', 'Y', 'Z', '']:
            if label not in row:
                exit(1)
        points[row['']] = np.array([row['X'], row['Y'], row['Z']], dtype='double')

    """
    Femoral Coordination:
    """

    femoral_coordination = build_coord(
        points['Femoral Lateral tendon end point'],
        points['Femoral Medial tendon end point'],
        points['Femoral Upper point of Long bone axis'],
        points['Femoral Lower point of Long bone axis'],
    )

    with np.printoptions(formatter={'float': '{:.2f}'.format}):
        print(femoral_coordination.mat_r)
        print(femoral_coordination.mat_t)

    """
    Tibial Coordination:
    """

    tibial_coordination = build_coord(
        points['Tibial Lateral tendon end point'],
        points['Tibial Medial tendon end point'],
        points['Tibial Upper point of Long bone axis'],
        points['Tibial Lower point of Long bone axis'],
    )
    with np.printoptions(formatter={'float': '{:.2f}'.format}):
        print(tibial_coordination.mat_r)
        print(tibial_coordination.mat_t)


if __name__ == '__main__':
    main()
