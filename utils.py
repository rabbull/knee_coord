from typing import Optional, Sequence, Mapping, Any

import numpy as np
import trimesh
from trimesh.typed import Number
from trimesh.creation import cylinder, icosphere
import trimesh.transformations as tf
from numpy.typing import ArrayLike
from trimesh import Trimesh
from scipy.spatial.transform import Rotation
import trimesh.scene
import trimesh.scene.transforms

Real: type = np.float64

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


def normalize(x):
    return x / np.linalg.norm(x)


class Transformation3D(object):
    def __init__(self, h = None):
        if h is None:
            h = np.eye(4)
        self._h = h

    def set_translation(self, translation) -> 'Transformation3D':
        self._h[:3, 3] = translation
        return self

    def set_rotation(self, rotation) -> 'Transformation3D':
        if isinstance(rotation, Rotation):
            r = rotation.as_matrix()
        else:
            r = rotation
        self._h[:3, :3] = r
        return self

    @property
    def mat_r(self):
        return self._h[:3, :3]

    @property
    def mat_t(self):
        return self._h[:3, 3]

    @property
    def mat_homo(self):
        return self._h

    @property
    def unit_x(self):
        return self.mat_r[:3, 0]

    @property
    def unit_y(self):
        return self.mat_r[:3, 1]

    @property
    def unit_z(self):
        return self.mat_r[:3, 2]

    @property
    def quaternion(self):
        return Rotation.from_matrix(self.mat_r).as_quat()

    @property
    def euler_angles(self, seq='zyx'):
        return Rotation.from_matrix(self.mat_r).as_euler(seq, degrees=False)

    def apply_rotation(self, rotation) -> 'Transformation3D':
        return self.apply_transformation(Transformation3D().set_rotation(rotation))

    def apply_translation(self, translation) -> 'Transformation3D':
        return self.apply_transformation(Transformation3D().set_translation(translation))

    def apply_transformation(self, rhs: 'Transformation3D') -> 'Transformation3D':
        self._h = rhs._h @ self._h
        return self

    def transform(self, x):
        x = np.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3, got {x.shape[-1]}")
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, 3)
        x_hom = np.hstack([x_flat, np.ones((x_flat.shape[0], 1))])
        x_transformed = (self._h @ x_hom.T).T
        return x_transformed[:, :3].reshape(*orig_shape, 3)

    def inverse(self) -> 'Transformation3D':
        return Transformation3D(np.linalg.inv(self._h))

    def relative_to(self, rhs: 'Transformation3D') -> 'Transformation3D':
        return Transformation3D(np.linalg.inv(rhs._h) @ self._h)


def remove_bubbles(mesh: Trimesh) -> Trimesh:
    components = mesh.split()
    largest = None
    for component in components:
        if largest is None or component.volume > largest.volume:
            largest = component
    return largest


def get_real_numbers_from_dict(d, keys):
    return np.array([float(d[k]) for k in keys], dtype=Real)


def show_model(path, *meshes):
    scene = trimesh.Scene(list(meshes))
    with open(path, 'wb') as f:
        f.write(scene.save_image())


# def show_model(_, *meshes):
#     scene = trimesh.Scene(list(meshes))
#     scene.show()


def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # Compute forward, right, and up vectors
    forward = center - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    # Create rotation matrix
    rotation = np.stack([right, up, -forward], axis=1)  # Combine basis vectors

    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -np.dot(rotation, eye)

    return transform


def hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb1(h: str) -> tuple:
    return tuple(c / 255.0 for c in hex_to_rgb(h))


def hex_to_rgba1(h: str) -> tuple:
    return tuple(list(hex_to_rgb1(h)) + [1.0])


def explode_axis(axis):
    meshes = []
    for name, geom in axis.geometry.items():
        if geom.visual.face_colors is not None:
            face_colors = geom.visual.face_colors[:, :3]
        else:
            face_colors = np.ones((len(geom.faces), 3), dtype=np.uint8) * 255
        vertex_colors = np.zeros((len(geom.vertices), 4), dtype=np.uint8)
        for i, face in enumerate(geom.faces):
            vertex_colors[face] = np.append(face_colors[i], 255)
        geom.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        meshes.append(geom)
    return meshes


def my_axis(
    origin_size: Number = 0.04,
    transform: Optional[ArrayLike] = None,
    origin_color: Optional[ArrayLike] = None,
    axis_radius: Optional[Number] = None,
    axis_length: Optional[Number] = None,
) -> list[trimesh.Trimesh]:
    """
    Return an XYZ axis marker as a  Trimesh, which represents position
    and orientation. If you set the origin size the other parameters
    will be set relative to it.

    Parameters
    ----------
    origin_size : float
      Radius of sphere that represents the origin
    transform : (4, 4) float
      Transformation matrix
    origin_color : (3,) float or int, uint8 or float
      Color of the origin
    axis_radius : float
      Radius of cylinder that represents x, y, z axis
    axis_length: float
      Length of cylinder that represents x, y, z axis

    Returns
    -------
    marker : trimesh.Trimesh
      Mesh geometry of axis indicators
    """
    # the size of the ball representing the origin
    origin_size = float(origin_size)

    # set the transform and use origin-relative
    # sized for other parameters if not specified
    if transform is None:
        transform = np.eye(4)
    if origin_color is None:
        origin_color = [255, 255, 255, 255]
    if axis_radius is None:
        axis_radius = origin_size / 5.0
    if axis_length is None:
        axis_length = origin_size * 10.0

    # generate a ball for the origin
    axis_origin = icosphere(radius=origin_size)
    axis_origin.apply_transform(transform)

    # apply color to the origin ball
    axis_origin.visual.vertex_colors = origin_color

    # create the cylinder for the z-axis
    translation = tf.translation_matrix([0, 0, axis_length / 2])
    z_axis = cylinder(
        radius=axis_radius, height=axis_length, transform=transform.dot(translation)
    )
    # XYZ->RGB, Z is blue
    z_axis.visual.vertex_colors = [0, 0, 255]

    # create the cylinder for the y-axis
    translation = tf.translation_matrix([0, 0, axis_length / 2])
    rotation = tf.rotation_matrix(np.radians(-90), [1, 0, 0])
    y_axis = cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(rotation).dot(translation),
    )
    # XYZ->RGB, Y is green
    y_axis.visual.vertex_colors = [0, 255, 0]

    # create the cylinder for the x-axis
    translation = tf.translation_matrix([0, 0, axis_length / 2])
    rotation = tf.rotation_matrix(np.radians(90), [0, 1, 0])
    x_axis = cylinder(
        radius=axis_radius,
        height=axis_length,
        transform=transform.dot(rotation).dot(translation),
    )
    # XYZ->RGB, X is red
    x_axis.visual.vertex_colors = [255, 0, 0]

    # append the sphere and three cylinders
    return [axis_origin, x_axis, y_axis, z_axis]

def take_kth(k: Any):
    def res(s: Mapping[Any, Any]):
        return s[k]

    return res


def list_take_kth(k: int):
    def res(s: list[Sequence]):
        return [e[k] if e is not None else None for e in s] if s is not None else None

    return res
