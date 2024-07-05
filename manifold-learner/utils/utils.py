import math 
import numpy as np
from ahrs import Quaternion

def quat2rpy(quat) -> np.ndarray:

    # Normalise the quaternion
    norm = np.linalg.norm(quat, 2)
    quat_norm = quat/norm
    
    # Convert from quaternion to RPY angles
    euler = Quaternion(quat_norm).to_angles()
    roll, pitch, yaw = np.round(rad2deg(euler), 1)
    
    # Correct the orientation for the offset
    rpy = fix_orientation(np.array([[roll, pitch, yaw]]))
    
    # Return the RPY values
    return rpy

def rad2deg(vals):
    return vals*(180/math.pi)

def fix_orientation(euler) -> np.ndarray:
    
    euler[0,0] = euler[0,0] - 180
    return euler
