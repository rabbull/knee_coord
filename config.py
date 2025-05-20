from enum import Enum
from typing import Optional

from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator


class AnimationCameraDirection(Enum):
    AUTO = 'auto'  # TODO
    FIX_TIBIA_FRONT = 'front'
    FIX_TIBIA_L2M = 'l2m'
    FIX_TIBIA_M2L = 'm2l'


class DepthDirection(Enum):
    Z_AXIS_FEMUR = 'z_axis_femur'
    Z_AXIS_TIBIA = 'z_axis_tibia'
    CONTACT_PLANE = 'contact_plane'
    VERTEX_NORMAL = 'vertex_normal'  # TODO


class MomentDataFormat(Enum):
    CSV = 'csv'
    JSON = 'json'


class DofRotationMethod(Enum):
    EULER_XYZ = 'euler_xyz'
    EULER_YZX = 'euler_yzx'
    EULER_ZXY = 'euler_zxy'
    EULER_XZY = 'euler_xzy'
    EULER_ZYX = 'euler_zyx'
    EULER_YXZ = 'euler_yxz'
    PROJECTION = 'projection'  # 以绕 X 轴旋转为例，先选取股骨的 Y 轴基向量，再投影到胫骨坐标系 YZ 平面上，再计算相对胫骨的 Y 基向量旋转了多少
    JCS = 'jcs'


class InterpolateMethod(Enum):
    CubicSpline = 'Cubic Spline', CubicSpline
    Akima = 'Akima', Akima1DInterpolator
    Pchip = 'Pchip', PchipInterpolator


class BaseBone(Enum):
    FEMUR = 'femur'
    TIBIA = 'tibia'


class KneeSide(Enum):
    LEFT = 'left'
    RIGHT = 'right'


OUTPUT_DIRECTORY = 'output'

# tasks
GENERATE_ANIMATION = True
GENERATE_DEPTH_CURVE = True
GENERATE_DEPTH_MAP = True
GENERATE_DOF_CURVES = True
GENERATE_CART_THICKNESS_CURVE = True
GENERATE_NORM_DEPTH_CURVE = True
# 除以软骨厚度之和

KNEE_SIDE = KneeSide.LEFT
FEMUR_MODEL_FILE = 'acc_task/Femur.stl'
FEMUR_CARTILAGE_MODEL_FILE = 'acc_task/Femur_Cart_Smooth.stl'
TIBIA_MODEL_FILE = 'acc_task/Tibia.stl'
TIBIA_CARTILAGE_MODEL_FILE = 'acc_task/Tibia_Cart_Smooth.stl'
FEATURE_POINT_FILE = 'acc_task/Coordination_Pt.txt'
IGNORE_CARTILAGE = False

MOVEMENT_DATA_FORMAT = MomentDataFormat.JSON
MOVEMENT_DATA_FILE = 'interpolate/C_G_1st.txt'
MOVEMENT_SMOOTH = True
MOVEMENT_PICK_FRAMES: Optional[list[int]] = None
MOVEMENT_INTERPOLATE_METHOD: InterpolateMethod = InterpolateMethod.CubicSpline

# color picker: https://g.co/kgs/Te8C3VZ
ANIMATION_BONE_COLOR_FEMUR = '#ffffff'
ANIMATION_BONE_COLOR_TIBIA = '#ffffff'
ANIMATION_CARTILAGE_COLOR_FEMUR = '#00e5ff'
ANIMATION_CARTILAGE_COLOR_TIBIA = '#8800ff'
ANIMATION_RESOLUTION = (1000, 1000)
ANIMATION_CAMERA_DIRECTION: list[AnimationCameraDirection] = [AnimationCameraDirection.FIX_TIBIA_FRONT,
                                                              AnimationCameraDirection.FIX_TIBIA_M2L]
ANIMATION_LIGHT_INTENSITY = 3.0
ANIMATION_SHOW_BONE_COORDINATE = True  # RED: x-axis; GREEN: y-axis; BLUE: z-axis
ANIMATION_DURATION = 10  # bone movement animation length, in seconds

DEPTH_MAP_BONE_COLOR_FEMUR = '#ffffff'
DEPTH_MAP_BONE_COLOR_TIBIA = '#ffffff'
DEPTH_MAP_CARTILAGE_COLOR_FEMUR = '#1d16a1'
DEPTH_MAP_CARTILAGE_COLOR_TIBIA = '#1d16a1'
DEPTH_MAP_RESOLUTION = (1000, 1000)
DEPTH_MAP_LIGHT_INTENSITY = 3.0
DEPTH_DIRECTION: DepthDirection = DepthDirection.Z_AXIS_FEMUR
DEPTH_BASE_BONE: BaseBone = BaseBone.TIBIA
DEPTH_MAP_MARK_MAX = True
DEPTH_MAP_DEPTH_THRESHOLD = 20
DEPTH_MAP_DURATION = ANIMATION_DURATION  # depth map animation length, in seconds

DOF_ROTATION_METHOD: DofRotationMethod = DofRotationMethod.JCS
DOF_BASE_BONE: BaseBone = BaseBone.FEMUR

# WIP TASKS, DO NOT ENABLE
Y_ROTATE_EXP = False
GENERATE_FEMUR_CARTILAGE_THICKNESS_MAP = False
GENERATE_TIBIA_CARTILAGE_THICKNESS_MAP = False
