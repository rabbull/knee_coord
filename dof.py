import csv
import os
from typing import Any

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

import config


def to_jcs(homo, side: config.KneeSide = config.KneeSide.LEFT) -> dict[str, float]:
    r = homo[:3, :3]
    p = homo[:3, 3]
    e3_t = np.array([0., 0., 1.])
    e3_f = r @ e3_t
    e1_f = np.array([1., 0., 0.])
    e2_f = np.cross(e3_f, e1_f)
    norm_e2 = np.linalg.norm(e2_f)
    if norm_e2 < 1e-9:
        e2_f = np.zeros(3)
    else:
        e2_f /= norm_e2
    dot13 = np.dot(e1_f, e3_f)
    dot13 = np.clip(dot13, -1.0, 1.0)
    beta = np.arccos(dot13)
    if side == config.KneeSide.RIGHT:
        adduction = beta - np.pi / 2
    else:
        adduction = np.pi / 2 - beta
    fwd_femur = np.array([0., 1., 0.])
    cos_alpha = np.dot(e2_f, fwd_femur)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha_raw = np.arccos(cos_alpha)
    cross_dir = np.cross(fwd_femur, e2_f)
    sign_test = np.dot(cross_dir, e1_f)
    alpha = alpha_raw if sign_test >= 0 else -alpha_raw
    j_t_in_f = r @ np.array([0., 1., 0.])
    cos_gamma = np.dot(j_t_in_f, e2_f)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma_raw = np.arccos(cos_gamma)
    cross_test = np.cross(e2_f, j_t_in_f)
    sign_test2 = np.dot(cross_test, e3_f)
    gamma = gamma_raw if sign_test2 >= 0 else -gamma_raw
    p_f = - r.T @ p
    q1 = np.dot(p_f, e1_f)
    q2 = np.dot(p_f, e2_f)
    q3 = - np.dot(p_f, e3_f)

    return {
        'adduction': adduction,
        'flexion': alpha,
        'tibial_rotation': gamma,
        'q1': q1,
        'q2': q2,
        'q3': q3
    }


def calc_dof(original_coordinates_femur, original_coordinates_tibia,
             frame_transformations_femur, frame_transformations_tibia):
    y_tx, y_ty, y_tz = [], [], []
    y_rx, y_ry, y_rz = [], [], []

    for _, (ft, tt) in enumerate(zip(frame_transformations_femur, frame_transformations_tibia)):
        fc = original_coordinates_femur.copy()
        tc = original_coordinates_tibia.copy()
        fc.t.apply_transformation(ft)
        tc.t.apply_transformation(tt)

        if config.DOF_BASE_BONE == config.BoneType.FEMUR:
            r = tc.t.relative_to(fc.t)
        elif config.DOF_BASE_BONE == config.BoneType.TIBIA:
            r = fc.t.relative_to(tc.t)
        else:
            raise NotImplementedError(f'unknown base bone: {config.DOF_BASE_BONE}')

        tx, ty, tz = r.mat_t
        if config.DOF_ROTATION_METHOD in {
            config.DofRotationMethod.JCS,
            config.DofRotationMethod.JCS_ROT
        }:
            transform = to_jcs(r.mat_homo, side=config.KNEE_SIDE)
            ry = transform['adduction'] / np.pi * 180
            rx = transform['flexion'] / np.pi * 180
            rz = transform['tibial_rotation'] / np.pi * 180
            if config.DOF_ROTATION_METHOD == config.DofRotationMethod.JCS:
                tx, ty, tz = transform['q1'], transform['q2'], transform['q3']
        else:
            if config.DOF_ROTATION_METHOD.value.startswith('euler'):
                rot = Rotation.from_matrix(r.mat_r)
                rx, ry, rz = rot.as_euler(config.DOF_ROTATION_METHOD.value[-3:], degrees=True)
            elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.PROJECTION:
                rx, ry, rz = extract_rotation_projection(fc.t.mat_homo, tc.t.mat_homo)
            else:
                raise NotImplementedError(f'unknown rotation method: {config.DOF_ROTATION_METHOD}')

        y_tx.append(tx), y_ty.append(ty), y_tz.append(tz)
        y_rx.append(rx), y_ry.append(ry), y_rz.append(rz)

    return np.array(y_tx), np.array(y_ty), np.array(y_tz), np.array(y_rx), np.array(y_ry), np.array(y_rz)


def dump_dof(raw, smoothed):
    do_dump_dof(raw, "dof_raw.csv")
    if smoothed is not None:
        do_dump_dof(smoothed, "dof_smoothed.csv")


def do_dump_dof(dof_data, filename):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data
    num_frames = len(y_tx)
    x = np.arange(1, num_frames + 1)

    csv_path = os.path.join(config.OUTPUT_DIRECTORY, filename)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Translation X', 'Translation Y', 'Translation Z',
                         'Rotation X', 'Rotation Y', 'Rotation Z'])

        for i in range(num_frames):
            writer.writerow([
                x[i],
                y_tx[i],
                y_ty[i],
                y_tz[i],
                y_rx[i],
                y_ry[i],
                y_rz[i]
            ])


def plot_dof_curves(dof_data_raw, dof_data_smoothed):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data_raw
    if dof_data_smoothed is not None:
        ys_tx, ys_ty, ys_tz, ys_rx, ys_ry, ys_rz = dof_data_smoothed
    raw_line = 'x-' if dof_data_smoothed is None else 'x'
    smoothed_line = '-'

    x = np.arange(len(y_tx)) + 1
    if dof_data_smoothed is not None:
        xs = np.arange(len(ys_tx)) + 1
        x = np.linspace(xs[0], xs[-1], len(x))

    fig, ax = plt.subplots()
    ax.plot(x, y_tx, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_tx, smoothed_line)
    ax.set_title('Translation X')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_tx.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_ty, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_ty, smoothed_line)
    ax.set_title('Translation Y')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_ty.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_tz, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_tz, smoothed_line)
    ax.set_title('Translation Z')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_tz.png'))
    plt.close(fig)

    if config.DOF_ROTATION_METHOD.value.startswith('euler'):
        method = 'Euler ' + \
                 '-'.join(list(config.DOF_ROTATION_METHOD.value[-3:]))
    elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.PROJECTION:
        method = 'Projection'
    elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.JCS:
        method = 'JCS'
    elif config.DOF_ROTATION_METHOD == config.DofRotationMethod.JCS_ROT:
        method = 'JCS Rotation'
    else:
        raise NotImplementedError(
            f'unknown rotation method: {config.DOF_ROTATION_METHOD}')

    fig, ax = plt.subplots()
    ax.plot(x, y_rx, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_rx, smoothed_line)
    ax.set_title(f'X Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_rx.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_ry, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_ry, smoothed_line)
    ax.set_title(f'Y Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_ry.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(x, y_rz, raw_line)
    if dof_data_smoothed is not None:
        ax.plot(xs, ys_rz, smoothed_line)
    ax.set_title(f'Z Rotation ({method})')
    fig.savefig(os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_rz.png'))
    plt.close(fig)


def smooth_dof(dof_data):
    y_tx, y_ty, y_tz, y_rx, y_ry, y_rz = dof_data
    n = len(y_tx)
    x = np.arange(n) + 1
    degrees_of_freedom = {
        'Translation X': y_tx,
        'Translation Y': y_ty,
        'Translation Z': y_tz,
        'Euler Angle (x-y-z) X': y_rx,
        'Euler Angle (x-y-z) Y': y_ry,
        'Euler Angle (x-y-z) Z': y_rz,
    }

    nx = 100
    start = 1
    stop = n
    step = (stop - start) / nx
    xi = np.arange(start, stop + 1e-6, step)

    dof_interpolate_data = {
        'x': xi,
    }
    for dof_name, y in degrees_of_freedom.items():
        fig, ax = plt.subplots()
        ax.plot(x, y, 'x-', label='Original')
        ax.set_title(dof_name)
        method_name, cls = config.MOVEMENT_INTERPOLATE_METHOD.value
        interpolate = cls(x, y)
        yi = interpolate(xi)
        dof_interpolate_data[dof_name] = yi
        ax.plot(xi, yi, label='Interpolated')
        ax.legend()
        path = os.path.join(config.OUTPUT_DIRECTORY, f'dof_curve_interpolated_{dof_name}.png')
        fig.savefig(path)
        plt.close(fig)

    path = os.path.join(config.OUTPUT_DIRECTORY, 'dof_curve_interpolated.mat')
    scipy.io.savemat(path, dof_interpolate_data)

    return dof_interpolate_data


def extract_rotation_projection(H_F, H_T):
    """
    按“投影到与目标轴垂直的平面”并区分内外旋的方法,
    针对单帧 (H_T, H_F) 计算:
      - 绕 X 轴的角度 (投影到 YZ 平面, 使用 F 的 Y 轴)
      - 绕 Y 轴的角度 (投影到 XZ 平面, 使用 F 的 Z 轴)
      - 绕 Z 轴的角度 (投影到 XY 平面, 使用 F 的 X 轴)
    坐标系定义:
      - T, F 分别是两个骨头的局部坐标系
      - H_T, H_F: 4x4 齐次变换(单帧)
    返回:
      (angle_x, angle_y, angle_z): 3 个 float, 单位: 度, 含正负号
    """

    # ========== 1) 提取旋转矩阵 & 构造 R_{TF} = R_T^T * R_F ==========
    R_T = H_T[:3, :3]  # T 在世界系的 3x3
    R_F = H_F[:3, :3]  # F 在世界系的 3x3

    R_T_trans = R_T.T  # R_T^T
    R_TF = R_T_trans @ R_F  # 3x3

    # ========== 2) 小工具: 投影到与某轴垂直平面, 计算带符号夹角 ==========
    def project_and_signed_angle(u_in_T, ref_in_T, axis_idx):
        """
        单帧版:
          - u_in_T, ref_in_T: 各是 3D 向量(已经在 T坐标系 中表达).
          - axis_idx: 0表示绕X轴 => 投影YZ平面, 1->绕Y=>投影XZ, 2->绕Z=>投影XY.

        过程:
          1) 把 u_in_T, ref_in_T 在该平面上投影 => 对 axis_idx 分量置 0
          2) 归一化, 然后算无符号夹角 = atan2(||u×v||, u·v)
          3) 用叉乘在 axis_idx 方向分量判断正负, 实现带符号角度 => [-180°, +180°].
        返回:
          angle_deg (float)
        """
        # 复制
        u_proj = u_in_T.copy()
        r_proj = ref_in_T.copy()

        # 投影: 把 axis_idx 分量清 0
        u_proj[axis_idx] = 0.0
        r_proj[axis_idx] = 0.0

        # 若模长过小, 避免除0
        eps = 1e-12
        norm_u = np.linalg.norm(u_proj)
        norm_r = np.linalg.norm(r_proj)
        if norm_u < eps or norm_r < eps:
            return 0.0

        u_hat = u_proj / norm_u
        r_hat = r_proj / norm_r

        dot_ = np.dot(u_hat, r_hat)
        cross_ = np.cross(u_hat, r_hat)
        cross_len = np.linalg.norm(cross_)

        # 无符号幅度
        angle_rad = np.arctan2(cross_len, dot_)

        # 符号: 叉乘在 axis_idx 分量判断
        sign_ = np.sign(cross_[axis_idx])
        angle_rad *= sign_

        return np.degrees(angle_rad)

    # ========== 3) 依次计算绕 X/Y/Z 轴 ==========
    #  定义 T 系的基向量 (在 T系, x=(1,0,0), y=(0,1,0), z=(0,0,1))
    eT_x = np.array([1, 0, 0], dtype=float)
    eT_y = np.array([0, 1, 0], dtype=float)
    eT_z = np.array([0, 0, 1], dtype=float)

    #  绕 X => 用 F 系的 Y=(0,1,0), 先转到 T系 => 投影YZ => 和 eT_y 投影做夹角
    vF_y = np.array([0, 1, 0], dtype=float)
    vT_y = R_TF @ vF_y  # F 的 Y 向量在 T 系下的表达
    angle_x = project_and_signed_angle(vT_y, eT_y, axis_idx=0)

    #  绕 Y => 用 F 系的 Z=(0,0,1), => 投影 XZ => 和 eT_z 做夹角
    vF_z = np.array([0, 0, 1], dtype=float)
    vT_z = R_TF @ vF_z
    angle_y = project_and_signed_angle(vT_z, eT_z, axis_idx=1)

    #  绕 Z => 用 F系的 X=(1,0,0), => 投影XY => 和 eT_x 做夹角
    vF_x = np.array([1, 0, 0], dtype=float)
    vT_x = R_TF @ vF_x
    angle_z = project_and_signed_angle(vT_x, eT_x, axis_idx=2)

    return angle_x, angle_y, angle_z
