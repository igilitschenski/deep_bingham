import math
import numpy as np
import quaternion

def convert_euler_to_quaternion(roll, yaw, pitch):
    """Converts roll, yaw, pitch to a quaternion.
    """


    cy = math.cos(math.radians(roll) * 0.5)
    sy = math.sin(math.radians(roll) * 0.5)

    cp = math.cos(math.radians(yaw) * 0.5)
    sp = math.sin(math.radians(yaw) * 0.5)

    cr = math.cos(math.radians(pitch) * 0.5)
    sr = math.sin(math.radians(pitch) * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    return quat


def get_relative_rotation(src_orientation, dest_orientation):
    a = quaternion.quaternion(src_orientation[0], src_orientation[1],
                              src_orientation[2], src_orientation[3])
    b = quaternion.quaternion(dest_orientation[0], dest_orientation[1],
                              dest_orientation[2], dest_orientation[3])

    relative_rotation = quaternion.as_float_array(b * a.inverse())

    return relative_rotation


def get_frame_index(name, frame):
    for idx in range(len(frame)):
        if frame.iloc[idx, 0] == name:
            return idx
    raise Exception(
        "Could not find image {} in data frame, unsuccessful in finding frame index".format(
            name))


def convert_euler_to_quaternion_idiap(yaw, pitch, roll):
    yaw = math.radians(yaw)
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    quat = np.array([w, x, y, z])
    quat = quat / np.linalg.norm(quat)
    return quat



def quaternion_to_euler(w, x, y, z):
    sinr_cosp = +2.0 * (w * x + y * z)
    cosr_cosp = +1.0 - 2.0 * (x * x + y * y)
    pitch = math.atan2(sinr_cosp, cosr_cosp)

    sinp = +2.0 * (w * y - z * x)
    sinp = +1.0 if sinp > +1.0 else sinp
    sinp = -1.0 if sinp < -1.0 else sinp
    yaw = math.asin(sinp)

    siny_cosp = +2.0 * (w * z + x * y)
    cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
    roll = math.atan2(siny_cosp, cosy_cosp)

    return roll, yaw, pitch
