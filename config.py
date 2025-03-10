from enum import Enum


class AnimationCameraDirection(Enum):
    AUTO = 'auto'  # TODO
    FIX_TIBIA_FRONT = 'front'
    FIX_TIBIA_L2M = 'medial'
    FIX_TIBIA_M2L = 'lateral'


class DepthDirection(Enum):
    Z_AXIS = 'z_axis'
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


class BaseBone(Enum):
    FEMUR = 'femur'
    TIBIA = 'tibia'

OUTPUT_DIRECTORY = 'output'

# tasks
GENERATE_ANIMATION = True
GENERATE_DEPTH_CURVE = True
GENERATE_DEPTH_MAP = True
GENERATE_DOF_CURVES = True
INTERPOLATE_DOF = True

FEMUR_MODEL_FILE = 'acc_task/Femur.stl'
FEMUR_CARTILAGE_MODEL_FILE = 'acc_task/Femur_Cart_Smooth.stl'
TIBIA_MODEL_FILE = 'acc_task/Tibia.stl'
TIBIA_CARTILAGE_MODEL_FILE = 'acc_task/Tibia_Cart_Smooth.stl'
FEATURE_POINT_FILE = 'acc_task/Coordination_Pt.txt'

MOVEMENT_DATA_FORMAT = MomentDataFormat.CSV
MOVEMENT_DATA_FILE = 'model_0220/Bill_Kinematic_RK.csv'

# color picker: https://g.co/kgs/Te8C3VZ
ANIMATION_BONE_COLOR_FEMUR = '#ffffff'
ANIMATION_BONE_COLOR_TIBIA = '#ffffff'
ANIMATION_CARTILAGE_COLOR_FEMUR = '#00e5ff'
ANIMATION_CARTILAGE_COLOR_TIBIA = '#8800ff'
ANIMATION_RESOLUTION = (1000, 1000)
ANIMATION_DIRECTION: AnimationCameraDirection = AnimationCameraDirection.FIX_TIBIA_FRONT
ANIMATION_LIGHT_INTENSITY = 3.0
ANIMATION_SHOW_BONE_COORDINATE = True  # RED: x-axis; GREEN: y-axis; BLUE: z-axis

DEPTH_MAP_BONE_COLOR_FEMUR = '#ffffff'
DEPTH_MAP_BONE_COLOR_TIBIA = '#ffffff'
DEPTH_MAP_CARTILAGE_COLOR_FEMUR = '#1d16a1'
DEPTH_MAP_CARTILAGE_COLOR_TIBIA = '#1d16a1'
DEPTH_MAP_RESOLUTION = (1000, 1000)
DEPTH_MAP_LIGHT_INTENSITY = 3.0
DEPTH_DIRECTION: DepthDirection = DepthDirection.CONTACT_PLANE
DEPTH_BASE_BONE: BaseBone = BaseBone.FEMUR
DEPTH_MAP_MARK_MAX = True

DOF_ROTATION_METHOD: DofRotationMethod = DofRotationMethod.PROJECTION

# WIP TASKS, DO NOT ENABLE
Y_ROTATE_EXP = False
GENERATE_FEMUR_CARTILAGE_THICKNESS_MAP = False
GENERATE_TIBIA_CARTILAGE_THICKNESS_MAP = False
