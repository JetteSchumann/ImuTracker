import math
import numpy as np
from numpy import linalg as LA


class Quaternion(object):
    # TODO w,x,y,z?
    def __init__(self, omega=1.0, x=0.0, y=0.0, z=0.0):
        self._q = np.array([omega, x, y, z])

    def __repr__(self):
        return "%r(_q=%r)" % (self.__class__.__name__, self._q)

    def __str__(self):
        return "[omega=%f, x=%f, y=%f, z=%f]" % (self._q[0], self._q[1], self._q[2], self._q[3])

    def __add__(self, q2):

        omega = self._q[0] + q2[0]
        x = self._q[1] + q2[1]
        y = self._q[2] + q2[2]
        z = self._q[3] + q2[3]
        return Quaternion(omega, x, y, z)

    def __sub__(self, q2):
        omega = self._q[0] - q2[0]
        x = self._q[1] - q2[1]
        y = self._q[2] - q2[2]
        z = self._q[3] - q2[3]
        return Quaternion(omega, x, y, z)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self._q[0] * other, self._q[1] * other, self._q[2] * other, self._q[3] * other)

        elif isinstance(other, Quaternion):
            q2 = other
            omega = self._q[0] * q2[0] - self._q[1] * q2[1] - self._q[2] * q2[2] - self._q[3] * q2[3]
            x = self._q[0] * q2[1] + self._q[1] * q2[0] + self._q[2] * q2[3] - self._q[3] * q2[2]
            y = self._q[0] * q2[2] - self._q[1] * q2[3] + self._q[2] * q2[0] + self._q[3] * q2[1]
            z = self._q[0] * q2[3] + self._q[1] * q2[2] - self._q[2] * q2[1] + self._q[3] * q2[0]
            return Quaternion(omega, x, y, z)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Quaternion(self._q[0] / other, self._q[1] / other, self._q[2] / other, self._q[3] / other)
        else:
            return self * other.get_inverse()

    def __getitem__(self, index):
        return self._q[index]

    # pre: angles in rad
    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        q = [0, 0, 0, 0]

        c1 = math.cos(yaw / 2.0)
        c2 = math.cos(pitch / 2.0)
        c3 = math.cos(roll / 2.0)
        s1 = math.sin(yaw / 2.0)
        s2 = math.sin(pitch / 2.0)
        s3 = math.sin(roll / 2.0)

        q[0] = c1 * c2 * c3 - s1 * s2 * s3
        q[1] = s1 * s2 * c3 + c1 * c2 * s3
        q[2] = s1 * c2 * c3 + c1 * s2 * s3
        q[3] = c1 * s2 * c3 - s1 * c2 * s3

        q /= np.linalg.norm(q)

        return Quaternion(q[0], q[1], q[2], q[3])

    @staticmethod
    def quaternion_from_angle_and_axis(theta, axis):
        axis /= LA.norm(axis)
        theta = math.radians(theta)
        C = math.cos(theta/2)
        S = math.sin(theta/2)

        return Quaternion(C, axis[0]*S, axis[1]*S, axis[2]*S)

    # TODO check https://www.lythaniel.fr/index.php/tag/madgwick/
    # pre: quaternion needs to be normalized
    def to_euler(self):

        omega = self._q[0]
        x = self._q[1]
        y = self._q[2]
        z = self._q[3]

        # print(x*y+z*omega)
        if x * y + z * omega > 0.499:
            # singularity at north pole
            yaw = 2 * math.atan2(x, omega)
            pitch = math.pi / 2
            roll = 0
        elif x * y + z * omega < -0.499:
            # singularity at south pole
            yaw = -2 * math.atan2(x, omega)
            pitch = - math.pi / 2
            roll = 0
        else:
            yaw = math.atan2(2 * y * omega - 2 * x * z, 1 - 2 * math.pow(y, 2) - 2 * math.pow(z, 2))
            pitch = math.asin(2 * x * y + 2 * z * omega)
            roll = math.atan2(2 * x * omega - 2 * y * z, 1 - 2 * math.pow(x, 2) - 2 * math.pow(z, 2))

        return [roll, pitch, yaw]

    def to_euler2(self):
        R11 = 2 * (self._q[0] ** 2) - 1 + 2 * (self._q[1] ** 2)
        R21 = 2 * (self._q[1] * self._q[2] - self._q[0] * self._q[3])
        R31 = 2 * (self._q[1] * self._q[3] + self._q[0] * self._q[2])
        R32 = 2 * (self._q[2] * self._q[3] - self._q[0] * self._q[1])
        R33 = 2 * (self._q[0] ** 2) - 1 + 2 * (self._q[3] ** 2)

        roll = math.atan2(R32, R33)
        pitch = -math.atan(R31 / (math.sqrt(1 - (R31 ** 2))))
        yaw = math.atan2(R21, R11)

        return [roll, pitch, yaw]

    def to_euler3(self):
        w = self._q[0]
        x = self._q[1]
        y = self._q[2]
        z = self._q[3]
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = 1 if t2 > 1 else t2
        t2 = -1 if t2 < -1 else t2
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.atan2(t3, t4)

        return [X, Y, Z]

    def Q2Eul(self):
        '''Calculates the Euler Angles from Quaternion
        a is the real part
        b, c, d are the complex elements'''
        # Source: Buchholz, J. J. (2013). Vorlesungsmanuskript Regelungstechnik und Flugregler.
        # GRIN Verlag. Retrieved from http://www.grin.com/de/e-book/82818/regelungstechnik-und-flugregler
        self.normalize()

        a, b, c, d = self._q

        yaw = np.arctan2(2.0 * (b * c + a * d), (a ** 2 + b ** 2 - c ** 2 - d ** 2)) * 180.0 / np.pi
        pitch = np.arcsin(2.0 * (a * c - b * d)) * 180.0 / np.pi
        roll = -np.arctan2(2.0 * (c * d + a * b), -(a ** 2 - b ** 2 - c ** 2 + d ** 2)) * 180.0 / np.pi

        return np.array([roll, pitch, yaw])

    def normalize(self):
        # TODO check
        self._q /= LA.norm(self._q)

    def get_array(self):
        # return deep copy
        return np.array([self._q[0], self._q[1], self._q[2], self._q[3]])

    def get_conjugate(self):

        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])

    def get_norm(self):

        return LA.norm(self._q)

    def get_inverse(self):

        return self.get_conjugate() / self.get_norm()

    # Lust2001
    def get_rot_mat(self):

        self.normalize()

        return np.array([[1 - 2 * (self._q[2] ** 2 + self._q[3] ** 2),
                          2 * (self._q[1] * self._q[2] - self._q[0] * self._q[3]),
                          2 * (self._q[1] * self._q[3] + self._q[0] * self._q[2])],
                         [2 * (self._q[1] * self._q[2] + self._q[0] * self._q[3]),
                          1 - 2 * (self._q[1] ** 2 + self._q[3] ** 2),
                         2 * (self._q[2] * self._q[3] - self._q[0] * self._q[1])],
                        [2 * (self._q[1] * self._q[3] - self._q[0] * self._q[2]),
                         2 * (self._q[2] * self._q[3] + self._q[0] * self._q[1]),
                         1 - 2 * (self._q[1] ** 2 + self._q[2] ** 2)]])

    # transform quaternion to DCM matrix as in Zhang2016
    def get_DCM(self):
        self.normalize()
        a, b, c, d = self._q
        R11 = (a ** 2 + b ** 2 - c ** 2 - d ** 2)
        R12 = 2.0 * (b * c - a * d)
        R13 = 2.0 * (b * d + a * c)

        R21 = 2.0 * (b * c + a * d)
        R22 = a ** 2 - b ** 2 + c ** 2 - d ** 2
        R23 = 2.0 * (c * d - a * b)

        R31 = 2.0 * (b * d - a * c)
        R32 = 2.0 * (c * d + a * b)
        R33 = a ** 2 - b ** 2 - c ** 2 + d ** 2

        return np.matrix([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    def rotate_v(self, v):

        self.normalize()
        #print('normaluzed self: ', self)
        if(len(v)==2):
            q2 = Quaternion(0, v[0], v[1], 0)
        else:
            q2 = Quaternion(0, v[0], v[1], v[2])
        #print('q2: ', q2)
        #print('self.conj: ', self.get_conjugate())
        result = self * q2 * self.get_conjugate()
        #print('self*q2= ', self*q2)
        #print('self*q2*self.get_conjugate= ', self * q2 * self.get_conjugate())
        return result._q[1:]

    def get_ENU_quat(self):

        return Quaternion(self._q[1], -self._q[2], -self._q[3], self._q[0])
