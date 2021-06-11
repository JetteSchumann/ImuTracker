import math
import numpy as np


class DCM(object):

    # result of R_z(yaw).dot(R_y(pitch)).dot(R_x(roll))
    @staticmethod
    def from_euler_angles(roll, pitch, yaw):
        return np.array([[math.cos(pitch) * math.cos(yaw),
                          math.sin(roll) * math.sin(pitch) * math.cos(yaw) - math.cos(roll) * math.sin(yaw),
                          math.cos(roll) * math.sin(pitch) * math.cos(yaw) + math.sin(roll) * math.sin(yaw)],
                         [math.cos(pitch) * math.sin(yaw),
                          math.sin(roll) * math.sin(pitch) * math.sin(yaw) + math.cos(roll) * math.cos(yaw),
                          math.cos(roll) * math.sin(pitch) * math.sin(yaw) - math.sin(roll) * math.cos(yaw)],
                         [-math.sin(pitch),
                          math.sin(roll) * math.cos(pitch),
                          math.cos(roll) * math.cos(pitch)]])


