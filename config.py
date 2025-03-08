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
ANIMATION_DIRECTION: AnimationCameraDirection = AnimationCameraDirection.FIX_TIBIA_M2L
ANIMATION_LIGHT_INTENSITY = 3.0
ANIMATION_SHOW_BONE_COORDINATE = True  # RED: x-axis; GREEN: y-axis; BLUE: z-axis

DEPTH_MAP_BONE_COLOR_FEMUR = '#ffffff'
DEPTH_MAP_BONE_COLOR_TIBIA = '#ffffff'
DEPTH_MAP_CARTILAGE_COLOR_FEMUR = '#1d16a1'
DEPTH_MAP_CARTILAGE_COLOR_TIBIA = '#1d16a1'
DEPTH_MAP_LIGHT_INTENSITY = 3.0

DEPTH_DIRECTION = DepthDirection.CONTACT_PLANE


# WIP TASKS, DO NOT ENABLE
Y_ROTATE_EXP = False
GENERATE_FEMUR_CARTILAGE_THICKNESS_MAP = False
GENERATE_TIBIA_CARTILAGE_THICKNESS_MAP = False
