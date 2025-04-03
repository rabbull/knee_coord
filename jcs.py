import numpy as np


def to_jcs(homo, side='left'):
    R = homo[:3, :3]
    p = homo[:3, 3]
    e3_t = np.array([0., 0., 1.])
    e3_in_femur = R @ e3_t
    e1_f = np.array([1., 0., 0.])
    e2_in_femur = np.cross(e3_in_femur, e1_f)
    norm_e2 = np.linalg.norm(e2_in_femur)
    if norm_e2 < 1e-9:
        e2_in_femur = np.zeros(3)
    else:
        e2_in_femur /= norm_e2
    dot13 = np.dot(e1_f, e3_in_femur)
    dot13 = np.clip(dot13, -1.0, 1.0)
    beta = np.arccos(dot13)
    if side.lower() == 'right':
        adduction = beta - np.pi / 2
    else:
        adduction = np.pi / 2 - beta
    fwd_femur = np.array([0., 1., 0.])
    cos_alpha = np.dot(e2_in_femur, fwd_femur)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha_raw = np.arccos(cos_alpha)
    cross_dir = np.cross(fwd_femur, e2_in_femur)
    sign_test = np.dot(cross_dir, e1_f)
    alpha = alpha_raw if sign_test >= 0 else -alpha_raw
    j_t_in_f = R @ np.array([0., 1., 0.])
    cos_gamma = np.dot(j_t_in_f, e2_in_femur)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma_raw = np.arccos(cos_gamma)
    cross_test = np.cross(e2_in_femur, j_t_in_f)
    sign_test2 = np.dot(cross_test, e3_in_femur)
    gamma = gamma_raw if sign_test2 >= 0 else -gamma_raw
    R_T = R.T
    p_in_femur = - R_T @ p
    q1 = np.dot(p_in_femur, e1_f)
    q2 = np.dot(p_in_femur, e2_in_femur)
    q3 = - np.dot(p_in_femur, e3_in_femur)

    return {
        'adduction': adduction,
        'flexion': alpha,
        'tibial_rotation': gamma,
        'q1': q1,
        'q2': q2,
        'q3': q3
    }


def from_jcs(jcs, side='left'):
    adduction = jcs['adduction']
    flexion = jcs['flexion']
    gamma = jcs['tibial_rotation']
    q1 = jcs['q1']
    q2 = jcs['q2']
    q3 = jcs['q3']

    if side.lower() == 'right':
        beta = adduction + np.pi / 2
    else:
        beta = np.pi / 2 - adduction

    e3 = np.array([
        np.cos(beta),
        - np.sin(beta) * np.sin(flexion),
        np.sin(beta) * np.cos(flexion)
    ])

    e1_f = np.array([1., 0., 0.])

    e2 = np.cross(e3, e1_f)
    norm_e2 = np.linalg.norm(e2)
    if norm_e2 < 1e-9:
        e2 = np.zeros(3)
    else:
        e2 = e2 / norm_e2

    e1 = np.cross(e2, e3)

    R_base = np.column_stack((e1, e2, e3))

    c = np.cos(gamma)
    s = np.sin(gamma)
    Rz_gamma = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    R = R_base @ Rz_gamma

    p_in_femur = np.array([q1, q2, -q3])
    p = - R @ p_in_femur

    homo = np.eye(4)
    homo[:3, :3] = R
    homo[:3, 3] = p
    return homo


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("测试 1：单位矩阵")
    homo1 = np.eye(4)
    jcs1 = to_jcs(homo1, side='left')
    homo1_rec = from_jcs(jcs1, side='left')
    print("原始齐次矩阵：\n", homo1)
    print("重构齐次矩阵：\n", homo1_rec)
    print("-" * 50)

    print("测试 2：设定角度和平移")
    # 设定角度（单位：弧度）
    adduction_true = 0.2  # 左侧时 adduction = 0.2
    flexion_true = 0.3
    tibial_rot_true = -0.1
    # 平移量（femur 坐标下的参数 q1, q2, q3）
    q1_true = 10.0
    q2_true = -5.0
    q3_true = 3.0
    # 构造 JCS 参数字典
    jcs2 = {
        'adduction': adduction_true,
        'flexion': flexion_true,
        'tibial_rotation': tibial_rot_true,
        'q1': q1_true,
        'q2': q2_true,
        'q3': q3_true
    }
    homo2 = from_jcs(jcs2, side='left')
    jcs2_rec = to_jcs(homo2, side='left')
    print("输入 JCS 参数：")
    for k, v in jcs2.items():
        print(f"  {k}: {v:.4f}")
    print("重构 JCS 参数：")
    for k, v in jcs2_rec.items():
        print(f"  {k}: {v:.4f}")
    print("重构齐次矩阵：\n", homo2)
    print("-" * 50)

    print("测试 3：基于随机角度和平移")
    # 随机设定角度和平移
    adduction_rand = 0.1
    flexion_rand = -0.2
    tibial_rot_rand = 0.15
    q1_rand = 5.0
    q2_rand = 2.5
    q3_rand = -1.0
    jcs3 = {
        'adduction': adduction_rand,
        'flexion': flexion_rand,
        'tibial_rotation': tibial_rot_rand,
        'q1': q1_rand,
        'q2': q2_rand,
        'q3': q3_rand
    }
    homo3 = from_jcs(jcs3, side='left')
    jcs3_rec = to_jcs(homo3, side='left')
    print("输入 JCS 参数：")
    for k, v in jcs3.items():
        print(f"  {k}: {v:.4f}")
    print("重构 JCS 参数：")
    for k, v in jcs3_rec.items():
        print(f"  {k}: {v:.4f}")
    print("重构齐次矩阵：\n", homo3)
