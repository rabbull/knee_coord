GENERATE_ANIMATION = True
GENERATE_DEPTH_CURVE = False
GENERATE_DEPTH_MAP = True
GENERATE_DOF_CURVES = False
Y_ROTATE_EXP = False
INTERPOLATE_DOF = False

# FRAME_RELATIVE_TRANSFORM_FILE = 'model_0207/cleaned_trial_2_rs.csv'
FRAME_RELATIVE_TRANSFORM_FILE = 'interpolate/RM_G_1st.txt'

import numpy as np
offset_femur = np.array((-221.31, -227.81, -1223.9))
offset_tibia = np.array((-97.5, -97.5, -172.5))
offset_femur = np.zeros(3)
offset_tibia = np.zeros(3)