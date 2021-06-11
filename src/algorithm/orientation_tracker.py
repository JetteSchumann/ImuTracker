import abc
from src.algorithm.quaternion import *
import numpy as np
import copy
import logging


class OrientationTracker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._euler_angles_rad = []
        self._euler_angles_deg = []
        self._euler_angles_unwrapped_rad = []
        self._euler_angles_unwrapped_deg = []
        self._logger = logging.getLogger(__name__)

        self._ROTATION_TYPE= 'abstract'

    @abc.abstractmethod
    def calc_orientation_step(self, current_orientation, acc, mag, gyro, i):
        """Calculates orientation for one time step. Needs to be implemented in derived class.

        Parameters
        ----------
        current_orientation : current orientation of type self._ROTATION_TYPE (derived class)
        acc : 3D np.array with acceleration data
        mag : 3D np.array with magnetometer data
        gyro : 3D np.array with gyroscope data
        i : step


        Returns
        -------
        new orientation of type self._ROTATION_TYPE (derived class)
        """
        return

    def get_rotation_type(self):
        return self._ROTATION_TYPE

class MadgwickFilter(OrientationTracker):

    def __init__(self, start_beta=2.5):

        super().__init__()

        # high beta at beginning to speed up convergence, after 1 second set to 0.041
        self._beta = start_beta
        self._ROTATION_TYPE = 'quaternion'

        self._logger.info('Madgwick Filter initialized')


    # Must be called for consecutive i
    def calc_orientation_step(self, current_orientation, acc_orig, gyro_orig, mag_orig, i):

        # parameters are passed by reference!!! create deep copy so that parameters are not changed
        acc = copy.deepcopy(acc_orig)
        mag = copy.deepcopy(mag_orig)
        gyro = copy.deepcopy(gyro_orig)

        q = current_orientation

        # check acc
        if LA.norm(acc) == 0:
            print('warning: norm(acc)=0')
            return

        # Normalize acc measurements
        acc /= LA.norm(acc)

        # check mag
        if LA.norm(mag) == 0:
            print('warning: norm(mag)=0')
            return

        # Normalize mag measurements
        mag /= LA.norm(mag)

        # Reference direction of Earth's magnetic field
        h = q.rotate_v(mag)

        # coord system is in NED! TODO check https://stackoverflow.com/questions/26416738/madgwick-imu-algorithm-simulate-on-iphone
        #b = np.array([0, 0, LA.norm(h[0:2]), h[2]]) # for ENU because y points to the North
        b = np.array([0, LA.norm(h[0:2]), 0, h[2]])  # for NED because x points to the North

        # Gradient decent algorithm
        F = np.array([2 * (q[1] * q[3] - q[0] * q[2]) - acc[0],
                      2 * (q[0] * q[1] + q[2] * q[3]) - acc[1],
                      2 * (0.5 - q[1] ** 2 - q[2] ** 2) - acc[2],
                      2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - mag[0],
                      2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - mag[1],
                      2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - mag[2]])

        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0],
            [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0],
             -4 * b[1] * q[3] + 2 * b[3] * q[1]],
            [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[0],
             2 * b[1] * q[1] + 2 * b[3] * q[3], -2 * b[1] * q[0] + 2 * b[3] * q[2]],
            [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2],
             2 * b[1] * q[1]]])

        # print('F: ', F)
        # print('J: ', J)

        step = J.T.dot(F)
        # print('step: ', step)

        # Normalize
        step /= LA.norm(step)
        # print('step normalized: ', step)

        # Calculate rate of change of q
        # multi = (q * Quaternion(0, gyro[0], gyro[1], gyro[2])) * 0.5
        # print('STEP ', step.T)

        # reset beta after 1 second of convergence
        # TODO make dependent on sample_rate
        if i < 101:
            self._beta = 2.5
        else:
            self._beta = 0.041

        q_dot = ((q * Quaternion(0, gyro[0], gyro[1], gyro[2])) * 0.5) - self._beta * step.T
        q_next = q + (q_dot * (1 / 100))
        q_next.normalize()

        # print('tmp: ', tmp)
        # print('q_next: ', q_next)

        # print('Madgwick q : ', q_next)

        return q_next

    def calc_orientation_step_without_mag(self, current_orientation, acc_orig, gyro_orig, i):

        acc = copy.deepcopy(acc_orig)
        gyro = copy.deepcopy(gyro_orig)
        q = current_orientation

        # check if acc has changed
        if LA.norm(acc) == 0:
            print('warning: norm(acc)=0')
            return

        # Normalize acc measurements
        acc /= LA.norm(acc)

        # Gradient decent algorithm
        F = np.array([2 * (q[1] * q[3] - q[0] * q[2]) - acc[0],
                      2 * (q[0] * q[1] + q[2] * q[3]) - acc[1],
                      2 * (0.5 - q[1] ** 2 - q[2] ** 2) - acc[2]])

        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0]])

        # print('F: ', F)
        # print('J: ', J)

        step = J.T.dot(F)
        # print('step: ', step)

        # Normalize
        step /= LA.norm(step)
        # print('step normalized: ', step)

        if i < 101:
            self._beta = 2.5
        else:
            self._beta = 0.041

        # Calculate rate of change of q
        q_dot = ((q * Quaternion(0, gyro[0], gyro[1], gyro[2])) * 0.5) - self._beta * step.T
        q_next = q + (q_dot * (1 / 100))
        q_next.normalize()

        # print('tmp: ', tmp)
        # print('q_next: ', q_next)

        return q_next

class DcmTracker(OrientationTracker):

    def __init__(self):
        self._ROTATION_TYPE='matrix'


class KalmanFilter(OrientationTracker):

    # based on Yun2006

    def __init__(self, start_beta=2.5):

        super().__init__()

        # high beta at beginning to speed up convergence, after 1 second set to 0.041
        self._beta = start_beta
        self._ROTATION_TYPE = 'quaternion'

    # Must be called for consecutive i
    def calc_orientation_step(self, current_orientation, acc_orig, gyro_orig, mag_orig, delta_t):

        # parameters are passed by reference!!! create deep copy so that parameters are not changed
        acc = copy.deepcopy(acc_orig)
        mag = copy.deepcopy(mag_orig)
        gyro = copy.deepcopy(gyro_orig)

        q = current_orientation

        tau = [0.5, 0.5, 0.5]

        # Init
        x_k = np.zeros(7)
        z_k = np.zeros(7)
        z_k_pior = np.zeros(7)
        P_k = np.matrix(np.eye(7))

        Phi_k = np.matrix(np.zeros((7, 7)))  # discrete state transition matrix Phi_k
        for ii in range(3):
            Phi_k[ii, ii] = np.exp(-delta_t / tau[ii])

        H_k = np.eye(7)  # Identity matrix

        Q_k = np.zeros((7, 7))  # process noise matrix Q_k
        D = np.r_[0.4, 0.4, 0.4]  # [rad^2/sec^2]; from Yun, 2006

        for ii in range(3):
            Q_k[ii, ii] = D[ii] / (2 * tau[ii]) * (1 - np.exp(-2 * delta_t / tau[ii]))

        # Evaluate measurement noise covariance matrix R_k
        R_k = np.zeros((7, 7))
        r_angvel = 0.01;  # [rad**2/sec**2]; from Yun, 2006
        r_quats = 0.0001;  # from Yun, 2006
        for ii in range(7):
            if ii < 3:
                R_k[ii, ii] = r_angvel
            else:
                R_k[ii, ii] = r_quats

        # Reference direction of Earth's magnetic field
        h = q.rotate_v(mag)

        # coord system is in NED! TODO check https://stackoverflow.com/questions/26416738/madgwick-imu-algorithm-simulate-on-iphone
        #b = np.array([0, 0, LA.norm(h[0:2]), h[2]]) # for ENU because y points to the North
        b = np.array([0, LA.norm(h[0:2]), 0, h[2]])  # for NED because x points to the North

        # Gradient decent algorithm
        F = np.array([2 * (q[1] * q[3] - q[0] * q[2]) - acc[0],
                      2 * (q[0] * q[1] + q[2] * q[3]) - acc[1],
                      2 * (0.5 - q[1] ** 2 - q[2] ** 2) - acc[2],
                      2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - mag[0],
                      2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - mag[1],
                      2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - mag[2]])

        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0],
            [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0],
             -4 * b[1] * q[3] + 2 * b[3] * q[1]],
            [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[0],
             2 * b[1] * q[1] + 2 * b[3] * q[3], -2 * b[1] * q[0] + 2 * b[3] * q[2]],
            [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2],
             2 * b[1] * q[1]]])

        # print('F: ', F)
        # print('J: ', J)

        step = J.T.dot(F)
        # print('step: ', step)

        # Normalize
        step /= LA.norm(step)
        # print('step normalized: ', step)

        # Calculate rate of change of q
        # multi = (q * Quaternion(0, gyro[0], gyro[1], gyro[2])) * 0.5
        # print('STEP ', step.T)

        # reset beta after 1 second of convergence
        # TODO make dependent on sample_rate
        if i < 101:
            self._beta = 2.5
        else:
            self._beta = 0.041

        q_dot = ((q * Quaternion(0, gyro[0], gyro[1], gyro[2])) * 0.5) - self._beta * step.T
        q_next = q + (q_dot * (1 / 100))
        q_next.normalize()

        # print('tmp: ', tmp)
        # print('q_next: ', q_next)

        return q_next


class MahonyFilter(OrientationTracker):

    def __init__(self, delta_t, Kp=1, Ki=0):

        super().__init__()

        # high beta at beginning to speed up convergence, after 1 second set to 0.041
        self._Kp = Kp
        self._Ki = Ki
        self._e_integral = np.array([0, 0, 0])
        self._delta_t = delta_t
        self._ROTATION_TYPE = 'quaternion'

        self._logger.info('Mahony Filter initialized')

    def calc_orientation_step(self, current_orientation, acc_orig, gyro_orig, mag_orig, i):

        # parameters are passed by reference!!! create deep copy so that parameters are not changed
        acc = copy.deepcopy(acc_orig)
        mag = copy.deepcopy(mag_orig)
        gyro = copy.deepcopy(gyro_orig)

        q = current_orientation

        # check acc
        if LA.norm(acc) == 0:
            print('warning: norm(acc)=0')
            return

        # Normalize acc measurements
        acc /= LA.norm(acc)

        # check mag
        if LA.norm(mag) == 0:
            print('warning: norm(mag)=0')
            return

        # Normalize mag measurements
        mag /= LA.norm(mag)

        # Reference direction of Earth's magnetic field
        h = q.rotate_v(mag)

        # coord system is in NED! TODO check https://stackoverflow.com/questions/26416738/madgwick-imu-algorithm-simulate-on-iphone
        # b = np.array([0, 0, LA.norm(h[0:2]), h[2]]) # for ENU because y points to the North
        b = np.array([0, LA.norm(h[0:2]), 0, h[2]])  # for NED because x points to the North

        # Estimated direction of gravity and magnetic field
        v = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]),
            2 * (q[0] * q[1] + q[2] * q[3]),
            q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2])

        w = np.array([
            2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]),
            2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]),
            2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2)])

        # Error is sum of cross product between estimated direction and measured direction of fields
        e = np.cross(acc, v) + np.cross(mag, w)

        if self._Ki > 0:
            self._e_integral += e * self._delta_t
        else:
            self._e_integral = np.array([0, 0, 0])

        # Apply feedback terms
        gyro += self._Kp * e + self._Ki * self._e_integral;

        # Compute rate of change of quaternion
        q_dot = ((q * Quaternion(0, gyro[0], gyro[1], gyro[2])) * 0.5)

        # Integrate to yield quaternion
        q_next = q + (q_dot * self._delta_t)
        q_next.normalize()

        # print('Mahony q: ', q_next)

        return q_next
