from enum import Enum


class DepthDirection(Enum):
    Z_AXIS = 'z_axis'
    CONTACT_PLANE = 'contact_plane'
    VERTEX_NORMAL = 'vertex_normal'  # TODO: buggy

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

# WIP
Y_ROTATE_EXP = False
GENERATE_FEMUR_CARTILAGE_THICKNESS_MAP = False
GENERATE_TIBIA_CARTILAGE_THICKNESS_MAP = False

FEMUR_MODEL_FILE = 'acc_task/Femur.stl'
FEMUR_CARTILAGE_MODEL_FILE = 'acc_task/Femur_Cart_Smooth.stl'
TIBIA_MODEL_FILE = 'acc_task/Tibia.stl'
TIBIA_CARTILAGE_MODEL_FILE = 'acc_task/Tibia_Cart_Smooth.stl'
FEATURE_POINT_FILE = 'acc_task/Coordination_Pt.txt'

MOVEMENT_DATA_FORMAT = MomentDataFormat.CSV
MOVEMENT_DATA_FILE = 'model_0220/Bill_Kinematic_RK.csv'

# color picker: https://g.co/kgs/Te8C3VZ
ANIMATION_BONE_COLOR_FEMUR = '#e6e2d5'
ANIMATION_BONE_COLOR_TIBIA = '#e6e2d5'
ANIMATION_CARTILAGE_COLOR_FEMUR = '#00e5ff'
ANIMATION_CARTILAGE_COLOR_TIBIA = '#8800ff'
DEPTH_MAP_BONE_COLOR_FEMUR = '#e6e2d5'
DEPTH_MAP_BONE_COLOR_TIBIA = '#e6e2d5'
DEPTH_MAP_CARTILAGE_COLOR_FEMUR = '#111111'
DEPTH_MAP_CARTILAGE_COLOR_TIBIA = '#111111'

DEPTH_DIRECTION = DepthDirection.CONTACT_PLANE