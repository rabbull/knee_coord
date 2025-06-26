import numpy as np
from scipy.spatial.transform import Rotation

import config
from utils import Transformation3D, Real


def smooth_transformations(raw: list[Transformation3D]) -> list[Transformation3D]:
    n = len(raw)
    x = np.arange(n)
    nx = 100
    xi = np.linspace(0, n - 1, nx)
    y_rx, y_ry, y_rz = np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real)
    y_tx, y_ty, y_tz = np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real), np.zeros_like(x, dtype=Real)
    for i, t in enumerate(raw):
        tx, ty, tz = t.mat_t
        rx, ry, rz = Rotation.from_matrix(t.mat_r).as_euler('xyz')
        y_tx[i] = tx
        y_ty[i] = ty
        y_tz[i] = tz
        y_rx[i] = rx
        y_ry[i] = ry
        y_rz[i] = rz
    _, cls_interpolator = config.MOVEMENT_INTERPOLATE_METHOD.value

    dofi = []
    for y in [y_tx, y_ty, y_tz, y_rx, y_ry, y_rz]:
        interpolator = cls_interpolator(x, y)
        yi = interpolator(xi)
        dofi.append(yi)
    yi_tx, yi_ty, yi_tz, yi_rx, yi_ry, yi_rz = dofi

    smoothed = []
    for i in range(len(xi)):
        tx = yi_tx[i]
        ty = yi_ty[i]
        tz = yi_tz[i]
        rx = yi_rx[i]
        ry = yi_ry[i]
        rz = yi_rz[i]

        t = Transformation3D()
        t.set_translation(np.array([tx, ty, tz]))
        t.set_rotation(Rotation.from_euler(seq='xyz', angles=[rx, ry, rz]))
        smoothed.append(t)
    return smoothed
