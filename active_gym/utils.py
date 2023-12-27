import numpy as np

def euler_to_rotation_matrix(pyr: np.ndarray):
    pitch, yaw, roll = pyr[0], pyr[1], pyr[2]
    # Pitch (X-axis rotation)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Yaw (Y-axis rotation)
    R_y = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # Roll (Z-axis rotation)
    R_z = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_z, np.dot(R_y, R_x))

    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R
    R_4x4[:3, 3] = np.array([0, 0, 0])
    
    return R_4x4