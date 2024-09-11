import csv
import numpy as np


class Coordination(object):
    def __init__(self):
        self.transformation_matrix = np.eye(4)

    def set_translation(self, T: np.ndarray):
        self.transformation_matrix[:3, 3] = T

    def set_rotation(self, R: np.ndarray):
        self.transformation_matrix[:3, :3] = R

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
