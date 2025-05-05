import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

from utils import Real


def main():
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 500],
        [0, 0, 0, 1],
    ], dtype=Real)

    ux = [1, 0, 0]
    uy = [0, 1, 0]
    uz = [0, 0, 1]

    ry = rotation_matrix(np.pi, uy)
    rx = rotation_matrix(-np.pi / 2, ux)
    camera_pose = ry @ rx @ camera_pose
    print(camera_pose.astype(np.int64))

    print("Camera Position:", camera_pose[:3, 3])
    print("Camera Forward (Z):", camera_pose[:3, 2])
    print("Camera View Direction:", -camera_pose[:3, 2])


if __name__ == '__main__':
    main()
