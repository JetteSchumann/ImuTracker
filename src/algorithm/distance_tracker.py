from src.data.imu_data import *

import scipy.linalg

import logging


class DistanceTracker(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._TYPE = 'abstract'
        self._logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def calc_positions(self, start_velocity, start_pos, tracking_data):
        return


class DoubleIntegration(DistanceTracker):

    def __init__(self):
        super().__init__()
        self._TYPE = 'DoubleIntegration'

    def calc_positions(self, start_velocity, start_pos, tracking_data):
        # Initialize data container for calculation of trajectories
        num_tracking_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index
        v = np.zeros((num_tracking_steps, 3))
        s = np.zeros((num_tracking_steps, 3))
        v[0] = start_velocity
        s[0] = start_pos

        delta_t = 1 / tracking_data.imu_data.sample_rate

        acc_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index

        for i in range(1, num_tracking_steps):
            v_next = v[i - 1] + delta_t / 2 * (
                    tracking_data.acc_global_lin[acc_offset + i - 1] +
                    tracking_data.acc_global_lin[acc_offset + i])
            s_next = s[i - 1] + delta_t / 2 * (v[i - 1] + v_next)

            v[i] = v_next
            s[i] = s_next

        self._logger.info('Done')

        return [s, v]


class DoubleIntegrationCorrection(DistanceTracker):

    def __init__(self):
        super().__init__()
        self._TYPE = 'DoubleIntegrationCorrection'

    def calc_positions(self, start_velocity, start_pos, tracking_data, surrounding_traj, v_range):

        self._logger.info('Started position calculation: ' + self._TYPE)
        # Initialize data container for calculation of trajectories
        num_tracking_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index
        v = np.zeros((num_tracking_steps, 3))
        s = np.zeros((num_tracking_steps, 3))
        v[0] = start_velocity
        s[0] = start_pos

        delta_t = 1 / tracking_data.imu_data.sample_rate

        num_out_of_range = 0

        i = 1

        acc_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index
        corrected_samples_tmp = []

        while i < num_tracking_steps:

            v_next = v[i - 1] + delta_t / 2 * (
                    tracking_data.acc_global_lin[acc_offset + i - 1] +
                    tracking_data.acc_global_lin[acc_offset + i])

            camera_frame = tracking_data.imu_to_camera[tracking_data.start_tracking_index + i]

            if surrounding_traj.frames[0] < camera_frame < surrounding_traj.frames[-1]:
                traj_index = surrounding_traj.get_index_from_frame(camera_frame)
                v_surrounding = surrounding_traj.v[traj_index]

                # calc norm of diff for xy only!
                diff = v_surrounding - v_next
                v_norm_diff = LA.norm(diff[0:2])

                if v_norm_diff > v_range:
                    num_out_of_range += 1
                    corrected_samples_tmp.append(i)
                    v_next = v_surrounding

            s_next = s[i - 1] + delta_t / 2 * (v[i - 1] + v_next)

            # store data
            v[i] = v_next
            s[i] = s_next
            i += 1

        self._logger.info('Correction done: #values %i,  #num_out_of_range %i, v_range: %f', num_tracking_steps,
                          num_out_of_range, v_range)

        print('corrected samples: ', tracking_data.acc_adaption_corrected_samples)

        tracking_data.acc_adaption_corrected_samples = np.array(corrected_samples_tmp)
        tracking_data.acc_adaption_corrected_samples += tracking_data.start_tracking_index

        return [s, v]


class DoubleIntegrationAdaptiveCorrection(DistanceTracker):

    def __init__(self):
        super().__init__()
        self._TYPE = 'DoubleIntegrationAdaptiveCorrection'

    def calc_positions(self, start_velocity, start_pos, tracking_data, surrounding_traj, v_range):
        self._logger.info('started position calculation: ' + self._TYPE)
        # Initialize data container for calculation of trajectories
        num_tracking_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index
        v = np.zeros((num_tracking_steps, 3))
        s = np.zeros((num_tracking_steps, 3))
        v[0] = start_velocity
        s[0] = start_pos

        delta_t = 1 / tracking_data.imu_data.sample_rate
        num_out_of_range = 0

        i = 1
        last_corrected_i = 0

        traj_for_adaption = surrounding_traj
        adapt_acc = True

        acc_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index
        corrected_samples_tmp = []

        # for analysis only
        tracking_data.acc_before_adaption = np.copy(tracking_data.acc_global_lin)
        tracking_data.v_before_adaption = np.zeros((num_tracking_steps, 3))

        while i < num_tracking_steps:

            v_next = v[i - 1] + delta_t / 2 * (
                    tracking_data.acc_global_lin[acc_offset + i - 1] +
                    tracking_data.acc_global_lin[acc_offset + i])

            if traj_for_adaption is not None:
                # correct v with surrounding trajectory information

                camera_frame = tracking_data.imu_to_camera[tracking_data.start_tracking_index + i]

                if traj_for_adaption.frames[0] < camera_frame < traj_for_adaption.frames[-1]:
                    traj_index = traj_for_adaption.get_index_from_frame(camera_frame)
                    v_surrounding = traj_for_adaption.v[traj_index]

                    # calc norm of diff for xy only!
                    diff = v_surrounding - v_next
                    v_norm_diff = LA.norm(diff[0:2])

                    if v_norm_diff > v_range:
                        num_out_of_range += 1

                    if v_norm_diff > v_range and adapt_acc and i > last_corrected_i:
                        corrected_samples_tmp.append(i)

                        # estimate change of acc that was caused by change of velocity. derivative: acc = change in v / (1/samplerate * num_samples)
                        num_uncorrected_consecutive = i - last_corrected_i
                        acc_diff = diff / delta_t / num_uncorrected_consecutive
                        print("acc diff: ", acc_diff)

                        tracking_data.acc_before_adaption[acc_offset + last_corrected_i + 1:,
                        :] = tracking_data.acc_global_lin[acc_offset + last_corrected_i + 1:, :]
                        tracking_data.v_before_adaption[last_corrected_i:i, :] = v[last_corrected_i:i, :]

                        tracking_data.acc_global_lin[acc_offset + last_corrected_i + 1:, :] += acc_diff

                        # save current i as last corrected
                        tmp = last_corrected_i
                        last_corrected_i = i
                        i = tmp + 1
                        continue

            s_next = s[i - 1] + delta_t / 2 * (v[i - 1] + v_next)

            # store data
            v[i] = v_next
            s[i] = s_next
            i += 1

        self._logger.info('Correction done: #values %i,  #num_out_of_range %i, v_range: %f', num_tracking_steps,
                          num_out_of_range, v_range)

        print('corrected samples: ', tracking_data.acc_adaption_corrected_samples)

        tracking_data.acc_adaption_corrected_samples = np.array(corrected_samples_tmp)
        tracking_data.acc_adaption_corrected_samples += tracking_data.start_tracking_index

        # for analysis only
        tracking_data.v_after_adaption = np.copy(v)

        return [s, v]


class MAUKF2D(DistanceTracker):

    def __init__(self):

        super().__init__()
        self._TYPE = 'MAUKF2D'

    def calc_positions(self, start_velocity, start_pos, tracking_data):

        self._logger.info('started MAUUKF calculation in 2D with IMU data of 2 sensors')

        num_steps = tracking_data[0].end_tracking_index - tracking_data[0].start_tracking_index

        if tracking_data[0].end_tracking_index != tracking_data[1].end_tracking_index or tracking_data[
            0].start_tracking_index != tracking_data[1].start_tracking_index:
            self._logger.error('Different tracking ranges for IMU1 and IMU2: %d-%d, %d-%d',
                               tracking_data[0].start_tracking_index, tracking_data[0].end_tracking_index,
                               tracking_data[1].start_tracking_index, tracking_data[1].end_tracking_index)

        print()

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data[0].imu_data._sample_rate

        # dist between tracked 2D positions should be 0
        dist = 0

        print('START VELOCITY ', start_velocity)

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1, pos_S2, velocity_S2, acc_S2 (global frame) (dim=12)
        x_k = np.matrix(np.concatenate((start_pos[0:2], start_velocity[0:2], [0, 0], start_pos[0:2],
                                        start_velocity[0:2], [0, 0]), axis=0)).transpose()
        y_k = np.matrix(np.zeros((5, 1)))  # measurement vector: acc_S1, acc_S2, dist

        q_offset = tracking_data[0].start_tracking_index - tracking_data[0].start_q_calc_index

        # C: to get acc out of x
        C1 = np.concatenate((np.zeros((2, 4)), np.eye(2), np.zeros((2, 6))), axis=1)
        C2 = np.concatenate((np.zeros((2, 6)), np.zeros((2, 4)), np.eye(2)), axis=1)
        C = np.matrix(np.concatenate((C1, C2), axis=0))

        P_k = np.matrix(np.eye(12)) / 100  # error covariance matrix P_k

        Q = np.matrix(np.eye(12))  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        acc_rms = 0.004 * ImuData.gravity_constant
        R = np.matrix(np.eye(5)) * (acc_rms ** 2)  # * 100
        R[4, 4] = 0

        # define process matrix
        a1 = np.concatenate((np.eye(2), np.eye(2) * dt, np.eye(2) * (dt ** 2) / 2), axis=1)
        a2 = np.concatenate((np.zeros((2, 2)), np.eye(2), np.matrix(np.eye(2)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((2, 4)), np.eye(2)), axis=1)
        A_tmp = np.matrix(np.concatenate((a1, a2, a3), axis=0))

        A1 = np.concatenate((A_tmp, np.zeros((6, 6))), axis=1)
        A2 = np.concatenate((np.zeros((6, 6)), A_tmp), axis=1)
        A_k = np.matrix(np.concatenate((A1, A2), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # VARIABLE FOR UKF, for parameter settings see Teixeira2009
        x_dim = len(x_k)
        num_sigma_points = x_dim * 2 + 1
        alpha = 1  # 0.003
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for i in range(1, num_sigma_points):
            weights_m[i] = 1 / (2 * (x_dim + lambd))
            weights_c[i] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        for i in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, i + 1] = x_k + tmp[:, i]
            # fill n+1 ... 2n
            sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

        print('weights ', np.shape(weights_m))

        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # constraining Matrix, calculates dist^2 between posS1 und posS2
        G1 = np.concatenate((np.eye(2), np.zeros((2, 4)), -1 * np.matrix(np.eye(2)), np.zeros((2, 4))), axis=1)
        G2 = np.zeros((4, 12))
        G3 = np.concatenate((-1 * np.matrix(np.eye(2)), np.zeros((2, 4)), np.eye(2), np.zeros((2, 4))), axis=1)
        G4 = G2
        G = np.matrix(np.concatenate((G1, G2, G3, G4), axis=0))

        print('G ', np.shape(G))

        # for state storage
        x_out = np.zeros((num_steps, 12))

        num_corrected = 0
        corrected_samples = []

        x_out[0] = x_k.copy().ravel()

        for k in range(1, num_steps):

            # print('k=', k)
            # PREDICTION

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for dim in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[dim, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[dim, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, i] - new_mean_x
                new_covariance_x += weights_c[i] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C to sigma points
            transformed_sigma_points_y1 = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))

            transformed_sigma_points_y2 = np.zeros((1, num_sigma_points))
            for i in range(num_sigma_points):
                transformed_sigma_points_y2[0, i] = sigma_mat[:, i].transpose() * G * sigma_mat[:, i]

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            dist_tmp = sigma_mat[0:2, 1] - sigma_mat[6:8, 1]
            # print('self calc norm: ', LA.norm(dist_tmp), LA.norm(dist_tmp) ** 2)

            transformed_sigma_points_y = np.matrix(
                np.concatenate((transformed_sigma_points_y1, transformed_sigma_points_y2), axis=0))

            new_mean_y = np.zeros((5, 1))
            for i in range(5):
                new_mean_y[i, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[i, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((5, 5))

            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, i] - new_mean_y
                new_covariance_y += weights_m[i] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 5)))

            for i in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, i] - new_mean_x
                diff_y = transformed_sigma_points_y[:, i] - new_mean_y
                T += weights_c[i] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION
            y_k[0:2, 0] = np.matrix(tracking_data[0].acc_global_lin[q_offset + k][0:2]).transpose()
            y_k[2:4, 0] = np.matrix(tracking_data[1].acc_global_lin[q_offset + k][0:2]).transpose()
            y_k[4] = np.matrix([[0]])
            # R = np.matrix(np.eye(2)) * 0.004 * ImuData.gravity_constant

            # print('acc global index: ', q_offset + k, ', start_q: ', self._start_q_calc_index)

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for i in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, i + 1] = x_k + tmp[:, i]
                # fill n+1 ... 2n
                sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        print('num steps: ', num_steps, ', num corrected: ', num_corrected)

        return x_out


class MAUKF2DCorrection(DistanceTracker):

    def __init__(self):

        super().__init__()
        self._TYPE = 'MAUKF2DCorrection'

    # cond: data sets of IMU1 and IMU2 have equal length, same sample rate, same run offset!!
    # x_k=A_k-1*x_k-1 + B_k-1*u_k-1 + G_k-1*w_k-1
    # y_k=C_k*x_k+v_k
    # x_k: state vector
    # u_k input --> acc
    # y_k: measurement vector
    # A_k: state process matrix
    # B_k: relates the input to the state
    # C: relates state to measurement
    # v_k: measurement noise
    # w_k: process noise
    # R_k: measurement covariance matrix
    # Q_k: process covariance matrix
    # P_k: error covariance matrix
    def calc_positions(self, start_velocity, start_pos, tracking_data, surrounding_traj, v_range):

        self._logger.info('started MAUUKF calculation in 2D with IMU data of 2 sensors')

        num_steps = tracking_data[0].end_tracking_index - tracking_data[0].start_tracking_index

        if tracking_data[0].end_tracking_index != tracking_data[1].end_tracking_index or tracking_data[
            0].start_tracking_index != tracking_data[1].start_tracking_index:
            self._logger.error('Different tracking ranges for IMU1 and IMU2: %d-%d, %d-%d',
                               tracking_data[0].start_tracking_index, tracking_data[0].end_tracking_index,
                               tracking_data[1].start_tracking_index, tracking_data[1].end_tracking_index)

        print()

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data[0].imu_data._sample_rate

        # dist between tracked 2D positions should be 0
        dist = 0

        print('START VELOCITY ', start_velocity)

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1, pos_S2, velocity_S2, acc_S2 (global frame) (dim=12)
        x_k = np.matrix(np.concatenate((start_pos[0:2], start_velocity[0:2], [0, 0], start_pos[0:2],
                                        start_velocity[0:2], [0, 0]), axis=0)).transpose()
        y_k = np.matrix(np.zeros((5, 1)))  # measurement vector: acc_S1, acc_S2, dist

        q_offset = tracking_data[0].start_tracking_index - tracking_data[0].start_q_calc_index

        # C: to get acc out of x
        C1 = np.concatenate((np.zeros((2, 4)), np.eye(2), np.zeros((2, 6))), axis=1)
        C2 = np.concatenate((np.zeros((2, 6)), np.zeros((2, 4)), np.eye(2)), axis=1)
        C = np.matrix(np.concatenate((C1, C2), axis=0))

        P_k = np.matrix(np.eye(12)) / 100  # error covariance matrix P_k

        Q = np.matrix(np.eye(12))  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        acc_rms = 0.004 * ImuData.gravity_constant
        R = np.matrix(np.eye(5)) * (acc_rms ** 2)  # * 100
        R[4, 4] = 0

        # define process matrix
        a1 = np.concatenate((np.eye(2), np.eye(2) * dt, np.eye(2) * (dt ** 2) / 2), axis=1)
        a2 = np.concatenate((np.zeros((2, 2)), np.eye(2), np.matrix(np.eye(2)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((2, 4)), np.eye(2)), axis=1)
        A_tmp = np.matrix(np.concatenate((a1, a2, a3), axis=0))

        A1 = np.concatenate((A_tmp, np.zeros((6, 6))), axis=1)
        A2 = np.concatenate((np.zeros((6, 6)), A_tmp), axis=1)
        A_k = np.matrix(np.concatenate((A1, A2), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # VARIABLE FOR UKF, for parameter settings see Teixeira2009
        x_dim = len(x_k)
        num_sigma_points = x_dim * 2 + 1
        alpha = 1  # 0.003
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for i in range(1, num_sigma_points):
            weights_m[i] = 1 / (2 * (x_dim + lambd))
            weights_c[i] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        for i in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, i + 1] = x_k + tmp[:, i]
            # fill n+1 ... 2n
            sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

        print('weights ', np.shape(weights_m))

        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # constraining Matrix, calculates dist^2 between posS1 und posS2
        G1 = np.concatenate((np.eye(2), np.zeros((2, 4)), -1 * np.matrix(np.eye(2)), np.zeros((2, 4))), axis=1)
        G2 = np.zeros((4, 12))
        G3 = np.concatenate((-1 * np.matrix(np.eye(2)), np.zeros((2, 4)), np.eye(2), np.zeros((2, 4))), axis=1)
        G4 = G2
        G = np.matrix(np.concatenate((G1, G2, G3, G4), axis=0))

        print('G ', np.shape(G))

        # for state storage
        x_out = np.zeros((num_steps, 12))

        num_corrected = 0
        corrected_samples = []

        x_out[0] = x_k.copy().ravel()

        for k in range(1, num_steps):

            print('k=', k)
            # PREDICTION

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for dim in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[dim, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[dim, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, i] - new_mean_x
                new_covariance_x += weights_c[i] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C to sigma points
            transformed_sigma_points_y1 = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))

            transformed_sigma_points_y2 = np.zeros((1, num_sigma_points))
            for i in range(num_sigma_points):
                transformed_sigma_points_y2[0, i] = sigma_mat[:, i].transpose() * G * sigma_mat[:, i]

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            dist_tmp = sigma_mat[0:2, 1] - sigma_mat[6:8, 1]
            # print('self calc norm: ', LA.norm(dist_tmp), LA.norm(dist_tmp) ** 2)

            transformed_sigma_points_y = np.matrix(
                np.concatenate((transformed_sigma_points_y1, transformed_sigma_points_y2), axis=0))

            new_mean_y = np.zeros((5, 1))
            for i in range(5):
                new_mean_y[i, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[i, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((5, 5))

            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, i] - new_mean_y
                new_covariance_y += weights_m[i] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 5)))

            for i in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, i] - new_mean_x
                diff_y = transformed_sigma_points_y[:, i] - new_mean_y
                T += weights_c[i] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # CORRECTION

            camera_frame = tracking_data[0].imu_to_camera[tracking_data[0].start_tracking_index + k]
            traj_index = surrounding_traj.get_index_from_frame(camera_frame)
            traj_v = surrounding_traj.v[traj_index][0:2]

            current_v1 = x_k[2:4].flatten()
            current_v2 = x_k[8:10].flatten()
            diff1 = LA.norm(traj_v - current_v1)
            diff2 = LA.norm(traj_v - current_v2)

            print('diff: ', diff1, diff2)

            if diff1 > v_range or diff2 > v_range:
                print('corrected acc')
                num_corrected += 1
                x_k[2] = traj_v[0]
                x_k[3] = traj_v[1]
                x_k[8] = traj_v[0]
                x_k[9] = traj_v[1]

                y_k[0:2, 0] = np.matrix(surrounding_traj.acc[traj_index][0:2]).transpose()
                y_k[2:4, 0] = np.matrix(surrounding_traj.acc[traj_index][0:2]).transpose()
                y_k[4] = np.matrix([[0]])
                corrected_samples.append(k)
            else:
                y_k[0:2, 0] = np.matrix(tracking_data[0].acc_global_lin[q_offset + k][0:2]).transpose()
                y_k[2:4, 0] = np.matrix(tracking_data[1].acc_global_lin[q_offset + k][0:2]).transpose()
                y_k[4] = np.matrix([[0]])

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for i in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, i + 1] = x_k + tmp[:, i]
                # fill n+1 ... 2n
                sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

        for i in range(len(tracking_data)):
            tracking_data[i].acc_adaption_corrected_samples = np.array(corrected_samples)
            tracking_data[i].acc_adaption_corrected_samples += tracking_data[
                i].start_tracking_index
            print('corrected samples: ', corrected_samples)

        print('calc finished, x_dim:  ', x_out.shape)
        print('num steps: ', num_steps, ', num corrected: ', num_corrected)

        return x_out


class MAUKF2DAdaptiveCorrection(DistanceTracker):

    def __init__(self):

        super().__init__()
        self._TYPE = 'MAUKF2DAdaptiveCorrection'

    def calc_positions(self, start_velocity, start_pos, tracking_data, surrounding_traj, v_range):

        self._logger.info('started MAUUKF calculation in 2D with IMU data of 2 sensors')

        num_steps = tracking_data[0].end_tracking_index - tracking_data[0].start_tracking_index

        if tracking_data[0].end_tracking_index != tracking_data[1].end_tracking_index or tracking_data[
            0].start_tracking_index != tracking_data[1].start_tracking_index:
            self._logger.error('Different tracking ranges for IMU1 and IMU2: %d-%d, %d-%d',
                               tracking_data[0].start_tracking_index, tracking_data[0].end_tracking_index,
                               tracking_data[1].start_tracking_index, tracking_data[1].end_tracking_index)

        print()

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data[0].imu_data._sample_rate

        # dist between tracked 2D positions should be 0
        dist = 0

        print('START VELOCITY ', start_velocity)

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1, pos_S2, velocity_S2, acc_S2 (global frame) (dim=12)
        x_k = np.matrix(np.concatenate((start_pos[0:2], start_velocity[0:2], [0, 0], start_pos[0:2],
                                        start_velocity[0:2], [0, 0]), axis=0)).transpose()
        y_k = np.matrix(np.zeros((5, 1)))  # measurement vector: acc_S1, acc_S2, dist

        q_offset = tracking_data[0].start_tracking_index - tracking_data[0].start_q_calc_index

        # C: to get acc out of x
        C1 = np.concatenate((np.zeros((2, 4)), np.eye(2), np.zeros((2, 6))), axis=1)
        C2 = np.concatenate((np.zeros((2, 6)), np.zeros((2, 4)), np.eye(2)), axis=1)
        C = np.matrix(np.concatenate((C1, C2), axis=0))

        P_k = np.matrix(np.eye(12)) / 100  # error covariance matrix P_k

        Q = np.matrix(np.eye(12))  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        acc_rms = 0.004 * ImuData.gravity_constant
        R = np.matrix(np.eye(5)) * (acc_rms ** 2)  # * 100
        R[4, 4] = 0

        # define process matrix
        a1 = np.concatenate((np.eye(2), np.eye(2) * dt, np.eye(2) * (dt ** 2) / 2), axis=1)
        a2 = np.concatenate((np.zeros((2, 2)), np.eye(2), np.matrix(np.eye(2)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((2, 4)), np.eye(2)), axis=1)
        A_tmp = np.matrix(np.concatenate((a1, a2, a3), axis=0))

        A1 = np.concatenate((A_tmp, np.zeros((6, 6))), axis=1)
        A2 = np.concatenate((np.zeros((6, 6)), A_tmp), axis=1)
        A_k = np.matrix(np.concatenate((A1, A2), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # VARIABLE FOR UKF, for parameter settings see Teixeira2009
        x_dim = len(x_k)
        num_sigma_points = x_dim * 2 + 1
        alpha = 1  # 0.003
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for i in range(1, num_sigma_points):
            weights_m[i] = 1 / (2 * (x_dim + lambd))
            weights_c[i] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        for i in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, i + 1] = x_k + tmp[:, i]
            # fill n+1 ... 2n
            sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

        print('weights ', np.shape(weights_m))

        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # constraining Matrix, calculates dist^2 between posS1 und posS2
        G1 = np.concatenate((np.eye(2), np.zeros((2, 4)), -1 * np.matrix(np.eye(2)), np.zeros((2, 4))), axis=1)
        G2 = np.zeros((4, 12))
        G3 = np.concatenate((-1 * np.matrix(np.eye(2)), np.zeros((2, 4)), np.eye(2), np.zeros((2, 4))), axis=1)
        G4 = G2
        G = np.matrix(np.concatenate((G1, G2, G3, G4), axis=0))

        print('G ', np.shape(G))

        # for state storage
        x_out = np.zeros((num_steps, 12))

        num_corrected = 0
        corrected_samples = []

        x_out[0] = x_k.copy().ravel()

        last_corrected_k = 0
        last_corrected_sigma_mat = sigma_mat

        k = 1
        start_again = False

        while k < num_steps:

            print('k=', k)
            # PREDICTION

            if start_again:
                sigma_mat = last_corrected_sigma_mat
                start_again = False
                print('use last corrected sigma mat')

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for dim in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[dim, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[dim, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, i] - new_mean_x
                new_covariance_x += weights_c[i] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C to sigma points
            transformed_sigma_points_y1 = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))

            transformed_sigma_points_y2 = np.zeros((1, num_sigma_points))
            for i in range(num_sigma_points):
                transformed_sigma_points_y2[0, i] = sigma_mat[:, i].transpose() * G * sigma_mat[:, i]

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            transformed_sigma_points_y = np.matrix(
                np.concatenate((transformed_sigma_points_y1, transformed_sigma_points_y2), axis=0))

            new_mean_y = np.zeros((5, 1))
            for i in range(5):
                new_mean_y[i, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[i, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((5, 5))

            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, i] - new_mean_y
                new_covariance_y += weights_m[i] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 5)))

            for i in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, i] - new_mean_x
                diff_y = transformed_sigma_points_y[:, i] - new_mean_y
                T += weights_c[i] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # CORRECTION

            camera_frame = tracking_data[0].imu_to_camera[tracking_data[0].start_tracking_index + k]
            traj_index = surrounding_traj.get_index_from_frame(camera_frame)
            traj_v = surrounding_traj.v[traj_index][0:2]

            current_v1 = x_k[2:4].flatten()
            current_v2 = x_k[8:10].flatten()
            diff1 = traj_v - current_v1
            diff1_norm = LA.norm(diff1)
            diff2 = traj_v - current_v2
            diff2_norm = LA.norm(diff2)

            if (diff1_norm > v_range or diff2_norm > v_range) and k > last_corrected_k:
                acc_diff1 = diff1 / dt / (k - last_corrected_k)
                num_rows = np.shape(tracking_data[0].acc_global_lin[q_offset + last_corrected_k + 1:])[0]
                acc_diff1_x = np.ones(num_rows) * acc_diff1[0]
                acc_diff1_y = np.ones(num_rows) * acc_diff1[1]

                tracking_data[0].acc_global_lin[q_offset + last_corrected_k + 1:][:, 0] += acc_diff1_x
                tracking_data[0].acc_global_lin[q_offset + last_corrected_k + 1:][:, 1] += acc_diff1_y

                acc_diff2 = diff2 / dt / (k - last_corrected_k)
                acc_diff2_x = np.ones(num_rows) * acc_diff2[0]
                acc_diff2_y = np.ones(num_rows) * acc_diff2[1]
                tracking_data[1].acc_global_lin[q_offset + last_corrected_k + 1:][:, 0] += acc_diff2_x
                tracking_data[1].acc_global_lin[q_offset + last_corrected_k + 1:][:, 1] += acc_diff2_y

                corrected_samples.append(k)

                tmp = last_corrected_k
                last_corrected_k = k
                k = tmp + 1
                print('start again at: ', k, ', last corrected: ', last_corrected_k)
                start_again = True

                continue

            else:
                y_k[0:2, 0] = np.matrix(tracking_data[0].acc_global_lin[q_offset + k][0:2]).transpose()
                y_k[2:4, 0] = np.matrix(tracking_data[1].acc_global_lin[q_offset + k][0:2]).transpose()
                y_k[4] = np.matrix([[0]])

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for i in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, i + 1] = x_k + tmp[:, i]
                # fill n+1 ... 2n
                sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

            if k == last_corrected_k:
                last_corrected_sigma_mat = sigma_mat

            k = k + 1

        for i in range(len(tracking_data)):
            tracking_data[i].acc_adaption_corrected_samples = np.array(corrected_samples)
            tracking_data[i].acc_adaption_corrected_samples += tracking_data[
                i].start_tracking_index
            print('corrected samples: ', corrected_samples)

        print('calc finished, x_dim:  ', x_out.shape)
        print('num steps: ', num_steps, ', num corrected: ', num_corrected)

        return x_out


class KF3D_DCM(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'KF3D_DCM'

    def calc_positions(self, start_velocity, start_pos, tracking_data):
        self._logger.info('Started position calculation: ' + self._TYPE)

        num_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data.sample_rate

        # Initializations
        # state vector: pos, velocity, acc (global frame) (dim=9)
        x_k = np.matrix(np.concatenate((start_pos, start_velocity, [0, 0, 0]), axis=0)).transpose()
        y_k = np.matrix(np.zeros(3)).transpose()  # measurement vector: acc (local frame)

        q_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index

        C = np.matrix(np.concatenate((np.zeros((3, 6)), np.eye(3)), axis=1))

        P_k = np.matrix(np.eye(9))  # error covariance matrix P_k

        Q = np.matrix(np.eye(9)) * 0.1  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(3, 3)) * 0.004 * ImuData.gravity_constant

        # define fixed rows for process matrix
        a1 = np.concatenate((np.matrix(np.eye(3)), np.matrix(np.eye(3)) * dt), axis=1)
        a2 = np.concatenate((np.matrix(np.zeros((3, 3))), np.matrix(np.eye(3))), axis=1)
        a3 = np.matrix(np.zeros((3, 6)))
        A_draft = np.concatenate((a1, a2, a3), axis=0)

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A_draft: ', A_draft)

        # for state storage
        x_out = np.zeros((num_steps, 9))

        for k in range(num_steps):
            # PREDICTION

            # get current rotation matrix for rotating local acc into global space
            DCM_s1 = tracking_data.quaternions_global[q_offset + k].get_DCM()
            # add this rotation to process matrix
            last_columns = np.concatenate((DCM_s1 * (dt ** 2) / 2, DCM_s1 * dt, np.matrix(np.eye(3))), axis=0)
            A_k = np.concatenate((A_draft, last_columns), axis=1)

            # apply state process matrix
            x_k = A_k * x_k
            # calc covariance error matrix
            P_k = A_k * P_k * A_k.transpose() + Q

            # CORRECTION by taking measurement values into account

            # Update measurement vector z_k
            local_gravity = tracking_data.quaternions_global[q_offset + k].get_inverse().rotate_v(
                [0, 0, 9.81])
            y_k = np.matrix(tracking_data.imu_data.acc_local_filtered[
                                tracking_data.start_tracking_index + k] - local_gravity).transpose()

            # Calculate Kalman Gain
            K_k = P_k * C.transpose() * np.linalg.inv(C * P_k * C.transpose() + R)

            print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
                  '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION
            x_k = x_k + K_k * (y_k - C * x_k)

            print('new x shape: ', x_k.shape)

            # Update error covariance matrix
            P_k = P_k - K_k * (C * P_k * C.transpose() + R) * K_k.transpose()

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        return x_out


class KF3D_global_acc(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'KF3D_global_acc'

    def calc_positions(self, start_velocity, start_pos, tracking_data):
        self._logger.info('Started position calculation: ' + self._TYPE)

        num_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data.sample_rate

        # Initializations
        x_k = np.matrix(
            np.zeros(9)).transpose()  # state vector: pos, velocity, acc (global frame) TODO set start values
        y_k = np.matrix(np.zeros(3)).transpose()  # measurement vector: acc (local frame)

        q_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index

        C = np.matrix(np.concatenate((np.zeros((3, 6)), np.eye(3)), axis=1))

        P_k = np.matrix(np.eye(9))  # error covariance matrix P_k

        Q = np.matrix(np.eye(9)) * 0.1  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(3, 3)) * 0.004 * ImuData.gravity_constant

        # define process matrix
        a1 = np.concatenate((np.eye(3), np.eye(3) * dt, np.matrix(np.eye(3)) * (dt ** 2) / 2.0), axis=1)
        a2 = np.concatenate((np.zeros((3, 3)), np.eye(3), np.matrix(np.eye(3)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((3, 6)), np.matrix(np.eye(3))), axis=1)
        A_k = np.matrix(np.concatenate((a1, a2, a3), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # for state storage
        x_out = np.zeros((num_steps, 9))

        gravity_acc = np.matrix(np.zeros(9)).transpose()
        gravity_acc[8, 0] = 9.81
        print('GRAVITY: ', gravity_acc)
        # TODO subtract gravity from local acc

        num_consecutive = 10

        x_tmp = np.array(np.zeros((9, num_consecutive)))

        for k in range(num_steps):
            # PREDICTION

            # apply state process matrix
            x_k = A_k * x_k
            # calc covariance error matrix
            P_k = A_k * P_k * A_k.transpose() + Q

            # CORRECTION by taking measurement values into account

            # Update measurement vector z_k
            y_k = np.matrix(tracking_data.acc_global_lin[q_offset + k]).transpose()
            # Calculate Kalman Gain
            K_k = P_k * C.transpose() * np.linalg.inv(C * P_k * C.transpose() + R)

            print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
                  '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION
            x_k = x_k + K_k * (y_k - C * x_k)

            print('new x shape: ', x_k.shape)

            # Update error covariance matrix
            P_k = P_k - K_k * (C * P_k * C.transpose() + R) * K_k.transpose()

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        return x_out


class KF2D_global_acc(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'KF2D_global_acc'

    def calc_positions(self, start_velocity, start_pos, tracking_data):
        self._logger.info('Started position calculation: ' + self._TYPE)

        num_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data.sample_rate

        # Initializations
        x_k = np.matrix(np.concatenate((start_pos[0:2], start_velocity[0:2], [0, 0]), axis=0)).transpose()
        y_k = np.matrix(np.zeros(2)).transpose()  # measurement vector: acc (local frame)

        q_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index

        C = np.matrix(np.concatenate((np.zeros((2, 4)), np.eye(2)), axis=1))

        P_k = np.matrix(np.eye(6))  # error covariance matrix P_k

        # V = np.matrix([[1 / 2 * (dt ** 2), 1 / 2 * (dt ** 2), dt, dt, 1, 1]])
        # V = np.matrix(np.concatenate((v, v), axis=1))
        # Q = V.transpose() * V * 0.004 * ImuData.gravity_constant #*100000
        Q = np.matrix(np.eye(6, 6)) / 100  # * 100

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(2, 2)) * 100  # * 100 * ImuData.gravity_constant

        # define process matrix
        a1 = np.concatenate((np.eye(2), np.eye(2) * dt, np.matrix(np.eye(2)) * (dt ** 2) / 2.0), axis=1)
        a2 = np.concatenate((np.zeros((2, 2)), np.eye(2), np.matrix(np.eye(2)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((2, 4)), np.matrix(np.eye(2))), axis=1)
        A_k = np.matrix(np.concatenate((a1, a2, a3), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # for state storage
        x_out = np.zeros((num_steps, 6))

        for k in range(num_steps):
            # PREDICTION

            # apply state process matrix
            x_k = A_k * x_k
            # calc covariance error matrix
            P_k = A_k * P_k * A_k.transpose() + Q

            # CORRECTION by taking measurement values into account

            # Update measurement vector z_k
            y_k = np.matrix(tracking_data.acc_global_lin[q_offset + k][0:2]).transpose()
            # Calculate Kalman Gain
            K_k = P_k * C.transpose() * np.linalg.inv(C * P_k * C.transpose() + R)

            print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
                  '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION
            x_k = x_k + K_k * (y_k - C * x_k)

            print('new x shape: ', x_k.shape)

            # Update error covariance matrix
            P_k = P_k - K_k * (C * P_k * C.transpose() + R) * K_k.transpose()

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        return x_out


class UKF2D_1IMU(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'UKF2D_1IMU'

    # cond: data sets of IMU1 and IMU2 have equal length, same sample rate
    # x_k=A_k-1*x_k-1 + B_k-1*u_k-1 + G_k-1*w_k-1
    # y_k=C_k*x_k+v_k
    # x_k: state vector
    # u_k input --> acc
    # y_k: measurement vector
    # A_k: state process matrix
    # B_k: relates the input to the state
    # C: relates state to measurement
    # v_k: measurement noise
    # w_k: process noise
    # R_k: measurement covariance matrix
    # Q_k: process covariance matrix
    # P_k: error covariance matrix
    def calc_positions(self, start_velocity, start_pos, tracking_data, surrounding_traj=None, v_range=None):

        self._logger.info('started UKF calculation in 2D with IMU data of one sensor only')

        num_steps = tracking_data[0].end_tracking_index - tracking_data[0].start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data[0]._sample_rate

        print('START VELOCITY ', start_velocity)

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1 (global frame) (dim=6)
        x_k = np.matrix(np.concatenate((start_pos[0:2], start_velocity[0:2], [0, 0]), axis=0)).transpose()
        y_k = np.matrix(np.zeros((2, 1)))  # measurement vector: acc_S1

        q_offset = tracking_data[0].start_tracking_index - tracking_data[0].start_q_calc_index

        # C: to get acc out of x
        C = np.matrix(np.concatenate((np.zeros((2, 4)), np.eye(2)), axis=1))

        P_k = np.matrix(np.eye(6))  # error covariance matrix P_k

        # calc Q based on process formulas
        V = np.matrix([[1 / 2 * (dt ** 2), 1 / 2 * (dt ** 2), dt, dt, 1, 1]])
        # V = np.matrix(np.concatenate((v, v), axis=1))
        # Q = V.transpose() * V *10000 #* 0.004 * ImuData.gravity_constant
        Q = np.matrix(np.eye(6))  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(2)) * 0.004 * ImuData.gravity_constant
        # R = np.matrix(np.eye(7)) * 10e-9

        # define process matrix
        a1 = np.concatenate((np.eye(2), np.eye(2) * dt, np.eye(2) * (dt ** 2) / 2), axis=1)
        a2 = np.concatenate((np.zeros((2, 2)), np.eye(2), np.matrix(np.eye(2)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((2, 4)), np.eye(2)), axis=1)
        A_k = np.matrix(np.concatenate((a1, a2, a3), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # VARIABLE FOR UKF, for parameter settings see Teixeira2009
        x_dim = len(x_k)
        num_sigma_points = x_dim * 2 + 1
        alpha = 1  # 0.003
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for i in range(1, num_sigma_points):
            weights_m[i] = 1 / (2 * (x_dim + lambd))
            weights_c[i] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        for i in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, i + 1] = x_k + tmp[:, i]
            # fill n+1 ... 2n
            sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

        print('weights ', np.shape(weights_m))

        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # for state storage
        x_out = np.zeros((num_steps, 6))

        num_corrected = 0
        corrected_samples = []

        x_out[0] = x_k.copy().ravel()

        for k in range(1, num_steps):

            print('k=', k)
            # PREDICTION

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for dim in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[dim, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[dim, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, i] - new_mean_x
                new_covariance_x += weights_c[i] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q  # TODO check *dt?

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C to sigma points
            transformed_sigma_points_y = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            new_mean_y = np.zeros((2, 1))
            for i in range(2):
                new_mean_y[i, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[i, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((2, 2))

            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, i] - new_mean_y
                new_covariance_y += weights_m[i] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 2)))

            for i in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, i] - new_mean_x
                diff_y = transformed_sigma_points_y[:, i] - new_mean_y
                T += weights_c[i] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # print('len imu_to_camera: ', len(self._tracking_data[sensor_num].imu_to_camera))
            # print('index: ', self._tracking_data[sensor_num].start_tracking_index + k )

            camera_frame = tracking_data[0].imu_to_camera[
                tracking_data[0].start_tracking_index + k]

            traj_index = surrounding_traj.get_index_from_frame(camera_frame)
            traj_v = surrounding_traj.v[traj_index][0:2]

            if surrounding_traj is not None:
                current_v = x_k[2:4]
                diff = abs(LA.norm(traj_v) - LA.norm(current_v))

                # print('camera frame: ', camera_frame, ', traj_index: ', traj_index,', traj_v: ', traj_v, ', imu_v: ', current_v, 'diff: ', diff)

                if diff > v_range:
                    print('corrected')
                    num_corrected += 1
                    x_k[2] = traj_v[0]
                    x_k[3] = traj_v[1]
                    y_k[0:2, 0] = np.matrix(surrounding_traj.acc[traj_index][0:2]).transpose()
                    corrected_samples.append(k)

                else:
                    # Update state vector x_k with measurement Input DATA ASSIMILATION
                    y_k[0:2, 0] = np.matrix(tracking_data[0].acc_global_lin[q_offset + k][0:2]).transpose()
            else:
                # Update state vector x_k with measurement Input DATA ASSIMILATION
                y_k[0:2, 0] = np.matrix(tracking_data[0].acc_global_lin[q_offset + k][0:2]).transpose()

            # print('acc global index: ', q_offset + k, ', start_q: ', self._start_q_calc_index)

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for i in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, i + 1] = x_k + tmp[:, i]
                # fill n+1 ... 2n
                sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        print('num steps: ', num_steps, ', num corrected: ', num_corrected)

        if surrounding_traj is not None:
            for i in range(2):
                tracking_data[i].acc_adaption_corrected_samples = np.array(corrected_samples)
                tracking_data[i].acc_adaption_corrected_samples += tracking_data[
                    i].start_tracking_index
                print('corrected samples: ', corrected_samples)

        return x_out


class UKF3D_2IMU_DCM(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'UKF2D_2IMU_DCM'

    # cond: data sets of IMU1 and IMU2 have equal length, same sample rate
    # x_k=A_k-1*x_k-1 + B_k-1*u_k-1 + G_k-1*w_k-1
    # y_k=C_k*x_k+v_k
    # with LOCAL acc data
    def calc_positions(self, start_velocity, start_pos, tracking_data):

        self._logger.info('started UKF calculation in 2D with IMU data of one sensor only')

        num_steps = tracking_data.end_tracking_index - tracking_data.start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data._sample_rate

        print('START VELOCITY ', start_velocity)

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1, pos_S2, velocity_S2, acc_S2  (dim=18)
        x_k = np.matrix(np.concatenate(
            (start_pos, start_velocity, [0, 0, 0], start_pos, start_velocity, [0, 0, 0]),
            axis=0)).transpose()
        y_k = np.matrix(np.zeros((7, 1)))  # measurement vector: acc_S1, acc_S2, D^2 (dist between pos_S1 xy, pos_S2 xy)

        C1 = np.concatenate((np.zeros((3, 6)), np.eye(3), np.zeros((3, 9))), axis=1)
        C2 = np.concatenate((np.zeros((3, 9)), np.zeros((3, 6)), np.eye(3)), axis=1)
        C = np.matrix(np.concatenate((C1, C2), axis=0))

        P_k = np.matrix(np.eye(18))  # error covariance matrix P_k

        # calc Q based on process formulas
        # v = [[1/2*(dt**2), 1/2*(dt**2), 1/2*(dt**2), dt, dt, dt, 0, 0,0]]
        # V = np.matrix(np.concatenate((v, v), axis=1))
        # Q = V.transpose() * V * 0.004 * ImuData.gravity_constant
        Q = np.matrix(np.eye(18))  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(7)) * 0.004 * ImuData.gravity_constant

        # define process matrix, last columns is set in loop
        a1 = np.concatenate((np.eye(3), np.eye(3) * dt), axis=1)
        a2 = np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=1)
        a3 = np.zeros((3, 6))
        A_draft = np.concatenate((a1, a2, a3), axis=0)

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A_draft: ', A_draft)

        # VARIABLE FOR UKF, for parameter settings see Teixeira2009
        x_dim = len(x_k)
        num_sigma_points = x_dim * 2 + 1
        alpha = 1  # 0.003
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for i in range(1, num_sigma_points):
            weights_m[i] = 1 / (2 * (x_dim + lambd))
            weights_c[i] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        for i in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, i + 1] = x_k + tmp[:, i]
            # fill n+1 ... 2n
            sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

        print('weights ', np.shape(weights_m))

        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # constraining Matrix, calculates dist^2 between posS1 und posS2, X Y ONLY
        eye_2D = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        G1 = np.concatenate((eye_2D, np.zeros((3, 6)), -1 * eye_2D, np.zeros((3, 6))), axis=1)
        G2 = np.zeros((6, 18))
        G3 = np.concatenate((-1 * eye_2D, np.zeros((3, 6)), eye_2D, np.zeros((3, 6))), axis=1)
        G4 = G2
        G = np.matrix(np.concatenate((G1, G2, G3, G4), axis=0))

        print('G ', np.shape(G))

        # for state storage
        x_out = np.zeros((num_steps, 18))

        num_corrected = 0
        q_offset = tracking_data.start_tracking_index - tracking_data.start_q_calc_index
        imu_offset = tracking_data.start_tracking_index

        x_out[0] = x_k.copy().ravel()

        for k in range(1, num_steps):

            print('k=', k)
            # PREDICTION

            # get current rotation matrix for rotating local acc into global space
            DCM_s1 = tracking_data[0].quaternions_global[q_offset + k].get_DCM()
            DCM_s2 = tracking_data[1].quaternions_global[q_offset + k].get_DCM()
            print('DCM shape: ', np.shape(DCM_s1))
            # add this rotation to process matrix
            last_column1 = np.concatenate((DCM_s1 * (dt ** 2) / 2, DCM_s1 * dt, np.matrix(np.eye(3))), axis=0)
            last_column2 = np.concatenate((DCM_s2 * (dt ** 2) / 2, DCM_s2 * dt, np.matrix(np.eye(3))), axis=0)
            A1 = np.concatenate((A_draft, last_column1, np.zeros((9, 9))), axis=1)
            A2 = np.concatenate((np.zeros((9, 9)), A_draft, last_column2), axis=1)
            A_k = np.matrix(np.concatenate((A1, A2), axis=0))

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for dim in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[dim, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[dim, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, i] - new_mean_x
                new_covariance_x += weights_c[i] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q  # TODO check *dt?

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C to sigma points
            transformed_sigma_points_y1 = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))

            transformed_sigma_points_y2 = np.zeros((1, num_sigma_points))
            for i in range(num_sigma_points):
                transformed_sigma_points_y2[0, i] = sigma_mat[:, i].transpose() * G * sigma_mat[:, i]

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            dist_tmp = sigma_mat[0:2, 1] - sigma_mat[9:11, 1]
            print('self calc norm: ', LA.norm(dist_tmp), LA.norm(dist_tmp) ** 2)

            transformed_sigma_points_y = np.matrix(
                np.concatenate((transformed_sigma_points_y1, transformed_sigma_points_y2), axis=0))

            new_mean_y = np.zeros((7, 1))
            for i in range(7):
                new_mean_y[i, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[i, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((7, 7))

            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, i] - new_mean_y
                new_covariance_y += weights_m[i] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 7)))

            for i in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, i] - new_mean_x
                diff_y = transformed_sigma_points_y[:, i] - new_mean_y
                T += weights_c[i] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION

            y_k[0:3, 0] = np.matrix(tracking_data[0].imu_data.acc_local_filtered[imu_offset + k]).transpose()
            y_k[3:6, 0] = np.matrix(tracking_data[1].imu_data.acc_local_filtered[imu_offset + k]).transpose()
            y_k[4] = np.matrix([[0]])

            # R = np.matrix(np.eye(2)) * 0.004 * ImuData.gravity_constant

            # print('acc global index: ', q_offset + k, ', start_q: ', self._start_q_calc_index)

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for i in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, i + 1] = x_k + tmp[:, i]
                # fill n+1 ... 2n
                sigma_mat[:, i + 1 + x_dim] = x_k - tmp[:, i]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        print('num steps: ', num_steps, ', num corrected: ', num_corrected)
        return x_out


class UKF3D_pos_constrained(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'UKF3D_pos_constrained'

    # cond: data sets of IMU1 and IMU2 have equal length, same sample rate
    def calc_positions(self, start_velocity, start_pos, tracking_data, dist):

        print('applying UKF position constrained')

        num_steps = tracking_data[0].end_tracking_index - tracking_data[0].start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data[0]._sample_rate

        print('START VELOCITY ', start_velocity)

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1, pos_S2, velocity_S2, acc_S2  (global frame) (dim=18)
        x_k = np.matrix(np.concatenate(
            (start_pos, start_velocity, [0, 0, 0], start_pos, start_velocity, [0, 0, 0]),
            axis=0)).transpose()
        y_k = np.matrix(np.zeros((7, 1)))  # measurement vector: acc_S1, acc_S2, D^2 (dist between pos_S1, pos_S2)

        q_offset = tracking_data[0].start_tracking_index - tracking_data[0].start_q_calc_index

        C1 = np.concatenate((np.zeros((3, 6)), np.eye(3), np.zeros((3, 9))), axis=1)
        C2 = np.concatenate((np.zeros((3, 9)), np.zeros((3, 6)), np.eye(3)), axis=1)
        C = np.matrix(np.concatenate((C1, C2), axis=0))

        P_k = np.matrix(np.eye(18))  # error covariance matrix P_k

        # calc Q based on process formulas
        # v = [[1/2*(dt**2), 1/2*(dt**2), 1/2*(dt**2), dt, dt, dt, 0, 0,0]]
        # V = np.matrix(np.concatenate((v, v), axis=1))
        # Q = V.transpose() * V * 0.004 * ImuData.gravity_constant
        Q = np.matrix(np.eye(18)) * 0.1  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(7)) * 0.004 * ImuData.gravity_constant
        # R = np.matrix(np.eye(7)) * 10e-9
        R[6, 6] = 0.001
        # R = np.matrix(np.eye(7)) * math.pow(10,-16)

        # define process matrix
        a1 = np.concatenate((np.eye(3), np.eye(3) * dt, np.eye(3) * (dt ** 2) / 2), axis=1)
        a2 = np.concatenate((np.zeros((3, 3)), np.eye(3), np.matrix(np.eye(3)) * dt), axis=1)
        a3 = np.concatenate((np.zeros((3, 6)), np.eye(3)), axis=1)
        A_tmp = np.concatenate((a1, a2, a3), axis=0)

        A1 = np.concatenate((A_tmp, np.zeros((9, 9))), axis=1)
        A2 = np.concatenate((np.zeros((9, 9)), A_tmp), axis=1)
        A_k = np.matrix(np.concatenate((A1, A2), axis=0))

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_k)

        # TODO check init data

        # VARIABLE FOR UKF, for parameter settings see Teixira2009
        num_sigma_points = len(x_k) * 2 + 1
        x_dim = len(x_k)
        alpha = 1
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for sensor_num in range(1, num_sigma_points):
            weights_m[sensor_num] = 1 / (2 * (x_dim + lambd))
            weights_c[sensor_num] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        print('tmp ', np.shape(tmp), tmp)

        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        for dim in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, dim + 1] = x_k + tmp[:, dim]
            # fill n+1 ... 2n
            sigma_mat[:, dim + 1 + x_dim] = x_k - tmp[:, dim]

        print('weights ', np.shape(weights_m))
        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # constraining Matrix
        G1 = np.concatenate((np.eye(3), np.zeros((3, 6)), -1 * np.matrix(np.eye(3)), np.zeros((3, 6))), axis=1)
        G2 = np.zeros((6, 18))
        G3 = np.concatenate((-1 * np.matrix(np.eye(3)), np.zeros((3, 6)), np.eye(3), np.zeros((3, 6))), axis=1)
        G4 = G2
        G = np.matrix(np.concatenate((G1, G2, G3, G4), axis=0))

        print('G ', np.shape(G))

        # for state storage
        x_out = np.zeros((num_steps, 18))

        # TODO: try MAD algorithm as in Wang2010, switch for x,y,z for 2 sensors

        x_out[0] = x_k.copy().ravel()

        for k in range(1, num_steps):

            print('k=', k)
            # PREDICTION

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for dim in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[dim, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[dim, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for i in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, i] - new_mean_x
                new_covariance_x += weights_c[i] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q  # TODO check *dt?

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C and G to sigma points
            transformed_sigma_points_y1 = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))
            # calc dist for each sigma points
            transformed_sigma_points_y2 = np.zeros((1, num_sigma_points))
            for i in range(num_sigma_points):
                transformed_sigma_points_y2[0, i] = sigma_mat[:, i].transpose() * G * sigma_mat[:, i]

            # transformed_sigma_points_y2 = sigma_mat.transpose() * G * sigma_mat

            # print('sigma_mat[:,1] ', sigma_mat[:,1], ', dist: ', transformed_sigma_points_y2[0,1])
            dist_tmp = sigma_mat[0:3, 1] - sigma_mat[9:12, 1]
            # print('self calc norm: ', LA.norm(dist_tmp), LA.norm(dist_tmp)**2)

            transformed_sigma_points_y = np.matrix(
                np.concatenate((transformed_sigma_points_y1, transformed_sigma_points_y2), axis=0))

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            new_mean_y = np.zeros((7, 1))
            for i in range(7):
                new_mean_y[i, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[i, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((7, 7))

            # print('new dist: ', new_mean_y[6])
            # print('dist for all sigma points: ', transformed_sigma_points_y[6,:])

            # for each sigma point
            for sensor_num in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, sensor_num] - new_mean_y
                new_covariance_y += weights_m[sensor_num] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 7)))

            for sensor_num in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, sensor_num] - new_mean_x
                diff_y = transformed_sigma_points_y[:, sensor_num] - new_mean_y
                T += weights_c[sensor_num] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION
            y_k[0:3, 0] = np.matrix(tracking_data[0].acc_global_lin[q_offset + k]).transpose()
            y_k[3:6, 0] = np.matrix(tracking_data[1].acc_global_lin[q_offset + k]).transpose()
            y_k[6, 0] = np.matrix([[dist ** 2]])
            # print('dist soll: ', y_k[6,0])

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for sensor_num in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, sensor_num + 1] = x_k + tmp[:, sensor_num]
                # fill n+1 ... 2n
                sigma_mat[:, sensor_num + 1 + x_dim] = x_k - tmp[:, sensor_num]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

        print('calc finished, x_dim:  ', x_out.shape)
        return x_out


class UKF_DCM_pos_constrained(DistanceTracker):

    def __init__(self):
        super().__init__()
        self.TYPE = 'UKF_DCM_pos_constrained'

    # cond: data sets of IMU1 and IMU2 have equal length, same sample rate
    def calc_positions(self, start_velocity, start_pos, tracking_data, dist):

        print('applying UKF position constrained')

        num_steps = tracking_data[0].end_tracking_index - tracking_data[0].start_tracking_index

        # Set parameters for Kalman Filter
        dt = 1. / tracking_data.imu_data[0]._sample_rate

        # Initializations
        # state vector: pos_S1, velocity_S1, acc_S1, pos_S2, velocity_S2, acc_S2  (global frame) (dim=18)
        x_k = np.matrix(np.concatenate(
            (start_pos, start_velocity, [0, 0, 0], start_pos, start_velocity, [0, 0, 0]),
            axis=0)).transpose()
        y_k = np.matrix(np.zeros((7, 1)))  # measurement vector: acc_S1, acc_S2, D^2 (dist between pos_S1, pos_S2)

        q_offset = tracking_data[0].start_tracking_index - tracking_data[0].start_q_calc_index

        C1 = np.concatenate((np.zeros((3, 6)), np.eye(3), np.zeros((3, 9))), axis=1)
        C2 = np.concatenate((np.zeros((3, 9)), np.zeros((3, 6)), np.eye(3)), axis=1)
        C = np.matrix(np.concatenate((C1, C2), axis=0))

        P_k = np.matrix(np.eye(18))  # error covariance matrix P_k

        # calc Q based on process formulas
        # v = [[1/2*(dt**2), 1/2*(dt**2), 1/2*(dt**2), dt, dt, dt, 0, 0,0]]
        # V = np.matrix(np.concatenate((v, v), axis=1))
        # Q = V.transpose() * V * 0.004 * ImuData.gravity_constant
        Q = np.matrix(np.eye(18)) * 0.1  # process noise matrix Q_k

        # Evaluate measurement noise covariance matrix R
        # acc rms noise: 4 mg -> 0.004 g
        R = np.matrix(np.eye(7)) * 0.004 * ImuData.gravity_constant
        # R = np.matrix(np.eye(7)) * 10e-9
        R[6, 6] = 0.001
        # R = np.matrix(np.eye(7)) * math.pow(10,-16)

        # define process matrix
        a1 = np.concatenate((np.eye(3), np.eye(3) * dt), axis=1)
        a2 = np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=1)
        a3 = np.zeros((3, 6))
        A_draft = np.concatenate((a1, a2, a3), axis=0)

        print('INITIALIZATION \n x_k: ', x_k, '\n y_k: ', y_k, '\n C: ', C, '\n P_k: ', P_k, '\n Q: ', Q, '\n R: ', R,
              '\n A: ', A_draft)

        # VARIABLE FOR UKF, for parameter settings see Teixira2009
        num_sigma_points = len(x_k) * 2 + 1
        x_dim = len(x_k)
        alpha = 1
        beta = 2
        kappa = 0
        lambd = alpha ** 2 * (x_dim + kappa) - x_dim

        # init weights
        weights_m = np.zeros(num_sigma_points)
        weights_m[0] = lambd / (x_dim + lambd)

        weights_c = np.zeros(num_sigma_points)
        weights_c[0] = lambd / (x_dim + lambd) + (1 - alpha ** 2 + beta)

        for sensor_num in range(1, num_sigma_points):
            weights_m[sensor_num] = 1 / (2 * (x_dim + lambd))
            weights_c[sensor_num] = 1 / (2 * (x_dim + lambd))

        # calc sigma matrix
        sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
        tmp = (x_dim + lambd) * P_k
        tmp = np.matrix(scipy.linalg.sqrtm(tmp))

        print('tmp ', np.shape(tmp))

        sigma_mat[:, 0] = x_k

        print('shape sigma ', np.shape(sigma_mat[:, 0]))

        # fill columns of sigma matrix
        for sensor_num in range(x_dim):
            # fill 1 ... n
            sigma_mat[:, sensor_num + 1] = x_k + tmp[:, sensor_num]
            # fill n+1 ... 2n
            sigma_mat[:, sensor_num + 1 + x_dim] = x_k - tmp[:, sensor_num]

        print('weights ', np.shape(weights_m))
        print('x_k ', np.shape(x_k))
        print('SIGMA mat ', np.shape(sigma_mat))

        # constraining Matrix
        G1 = np.concatenate((np.eye(3), np.zeros((3, 6)), -1 * np.matrix(np.eye(3)), np.zeros((3, 6))), axis=1)
        G2 = np.zeros((6, 18))
        G3 = np.concatenate((-1 * np.matrix(np.eye(3)), np.zeros((3, 6)), np.eye(3), np.zeros((3, 6))), axis=1)
        G4 = G2
        G = np.matrix(np.concatenate((G1, G2, G3, G4), axis=0))

        print('G ', np.shape(G))

        # for state storage
        x_out = np.zeros((num_steps, 18))

        # MAD algorithm as in Wang2010, switch for x,y,z for 2 sensors
        acc_switch = np.array([[False, False, False], [False, False, False]])
        T_acc = np.zeros((2, 3))
        K_acc = 2  # TODO check
        T_v = np.zeros((2, 3))
        K_v = 0.5

        for sensor_num in range(2):
            start = tracking_data.imu_data[sensor_num].stationary_phase[0]
            end = tracking_data.imu_data[sensor_num].stationary_phase[1]
            for dim in range(3):
                T_acc[sensor_num, dim] = K_acc * np.amax(np.absolute(tracking_data[sensor_num].acc_global_lin[
                                                                     start - tracking_data[
                                                                         sensor_num].start_q_calc_index:end -
                                                                                                        tracking_data[
                                                                                                            sensor_num].start_q_calc_index,
                                                                     dim]))
        print('T_ACC calculated: ', T_acc)

        gravity = np.matrix(np.zeros((18, 1)))
        gravity[8, 0] = 9.81
        gravity[15, 0] = 9.81

        for k in range(num_steps):

            print('k=', k)
            # PREDICTION

            # get current rotation matrix for rotating local acc into global space
            DCM_s1 = tracking_data[0].quaternions_global[q_offset + k].get_DCM()
            DCM_s2 = tracking_data[1].quaternions_global[q_offset + k].get_DCM()
            # add this rotation to process matrix
            last_column1 = np.concatenate((DCM_s1 * (dt ** 2) / 2, DCM_s1 * dt, np.matrix(np.eye(3))), axis=0)
            last_column2 = np.concatenate((DCM_s2 * (dt ** 2) / 2, DCM_s2 * dt, np.matrix(np.eye(3))), axis=0)
            A1 = np.concatenate((A_draft, last_column1, np.zeros((9, 9))), axis=1)
            A2 = np.concatenate((np.zeros((9, 9)), A_draft, last_column2), axis=1)
            A_k = np.matrix(np.concatenate((A1, A2), axis=0))

            # apply state process matrix to all sigma points
            transformed_sigma_points_x = A_k * sigma_mat - gravity
            new_mean_x = np.zeros((x_dim, 1))

            # print('transformed_sigma_points_x ', np.shape(transformed_sigma_points_x))

            # calc new mean for each state variable
            for sensor_num in range(x_dim):
                # sum of the weighted values for each sigma point
                new_mean_x[sensor_num, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_x[sensor_num, j] for j in range(num_sigma_points)))

            new_covariance_x = np.zeros((x_dim, x_dim))
            # for each sigma point
            for sensor_num in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_x[:, sensor_num] - new_mean_x
                new_covariance_x += weights_c[sensor_num] * np.dot(diff, diff.T)

            # calc covariance error matrix
            new_covariance_x += dt * Q  # TODO check *dt?

            x_k = new_mean_x
            P_k = new_covariance_x

            # CORRECTION by taking measurement values into account

            # apply C and G to sigma points
            transformed_sigma_points_y1 = C * sigma_mat
            # print('transformed_sigma_points_y1 ', np.shape(transformed_sigma_points_y1))
            # calc dist for each sigma points
            transformed_sigma_points_y2 = np.zeros((1, num_sigma_points))
            for sensor_num in range(num_sigma_points):
                transformed_sigma_points_y2[0, sensor_num] = sigma_mat[:, sensor_num].transpose() * G * sigma_mat[:,
                                                                                                        sensor_num]

            transformed_sigma_points_y = np.matrix(
                np.concatenate((transformed_sigma_points_y1, transformed_sigma_points_y2), axis=0))

            # print(' C ', np.shape(C), ', G ', np.shape(G), ', sigma_mat ', np.shape(sigma_mat))
            # print('transformed_sigma_points_y ', np.shape(transformed_sigma_points_y))

            new_mean_y = np.zeros((7, 1))
            for sensor_num in range(7):
                new_mean_y[sensor_num, 0] = sum(
                    (weights_m[j] * transformed_sigma_points_y[sensor_num, j] for j in range(num_sigma_points)))

            new_covariance_y = np.zeros((7, 7))

            # for each sigma point
            for sensor_num in range(num_sigma_points):
                # take the distance from the mean
                # make it a covariance by multiplying by the transpose
                # weight it using the calculated weighting factor
                # and sum
                diff = transformed_sigma_points_y[:, sensor_num] - new_mean_y
                new_covariance_y += weights_m[sensor_num] * np.dot(diff, diff.T)
                # print('new_covariance: ', new_covariance_y)

            new_covariance_y += R

            # Calculate Kalman Gain
            T = np.matrix(np.zeros((x_dim, 7)))

            for sensor_num in range(num_sigma_points):
                diff_x = transformed_sigma_points_x[:, sensor_num] - new_mean_x
                diff_y = transformed_sigma_points_y[:, sensor_num] - new_mean_y
                T += weights_c[sensor_num] * np.dot(diff_x, diff_y.T)

            # print('T ', np.shape(T), ', new_covariance_y ', np.shape(new_covariance_y))

            # print('INVERT \n', new_covariance_y)
            K_k = T * np.linalg.inv(new_covariance_y)

            # print('K=', k, '\n x_k: ', x_k.shape, '\n y_k: ', y_k.shape, '\n P_k: ', P_k.shape, '\n K_k: ', K_k.shape,
            #      '\n A_k: ', A_k.shape)

            # Update state vector x_k with measurement Input DATA ASSIMILATION

            local_gravity0 = tracking_data[0].quaternions_global[q_offset + k].get_inverse().rotate_v(
                [0, 0, 9.81])
            local_gravity1 = tracking_data[1].quaternions_global[q_offset + k].get_inverse().rotate_v(
                [0, 0, 9.81])

            y_k[0:3, 0] = np.matrix(
                tracking_data.imu_data[0].acc_local_filtered[q_offset + k] - local_gravity0).transpose()
            y_k[3:6, 0] = np.matrix(
                tracking_data.imu_data[1].acc_local_filtered[q_offset + k] - local_gravity1).transpose()
            y_k[6, 0] = np.matrix([[dist ** 2]])

            x_k = x_k + K_k * (y_k - new_mean_y)
            P_k = P_k - K_k * new_covariance_y * K_k.transpose()

            # PREDICTION AND CORRECTION OF UKF DONE

            # calc new SIGMA POINTS
            # calc sigma matrix
            sigma_mat = np.matrix(np.zeros((x_dim, num_sigma_points)))
            tmp = (x_dim + lambd) * P_k
            tmp = np.matrix(scipy.linalg.sqrtm(tmp))

            sigma_mat[:, 0] = x_k

            # fill columns of sigma matrix
            for sensor_num in range(x_dim):
                # fill 1 ... n
                sigma_mat[:, sensor_num + 1] = x_k + tmp[:, sensor_num]
                # fill n+1 ... 2n
                sigma_mat[:, sensor_num + 1 + x_dim] = x_k - tmp[:, sensor_num]

            # print('SIGMA mat ', np.shape(sigma_mat))

            # store states for later
            x_out[k] = x_k.copy().ravel()

            # APPLY MAD ALGORITHM Wang2010
            # for both sensors
            for sensor_num in range(2):

                acc = np.absolute(x_k[sensor_num * 9 + 6: sensor_num * 9 + 9])
                v = np.absolute(x_k[sensor_num * 9 + 3: sensor_num * 9 + 6])

                # check if calculated acc exceed threshold
                for dim in range(3):
                    # turn switch on if acc exceeds range
                    if acc[dim] > T_acc[sensor_num, dim]:
                        acc_switch[sensor_num, dim] = True
                    else:
                        print('k:', k, ' - sensor ', sensor_num, ', dim ', dim, ', acc 0!')
                        # x_out[k, sensor_num * 9 + 6 + dim] = 0
                        # keep prior v
                        # x_out[k, sensor_num * 9 + 3 + dim] = x_out[k - 1, sensor_num * 9 + 3 + dim]

                    # check if switch is on
                    if acc_switch[sensor_num, dim]:
                        # calc new velocity threshold
                        T_v[sensor_num, dim] = K_v * max(v[dim], T_v[sensor_num, dim])
                        # print('max: ', max(v[dim], T_v[sensor_num, dim]), v[dim], T_v[sensor_num, dim])

                        # check velocity is decreasing and acc indicates standstill
                        if v[dim] < T_v[sensor_num, dim] and acc[dim] < T_acc[sensor_num, dim]:
                            print('k:', k, ' - sensor ', sensor_num, ', dim ', dim, ', velocity reset!')
                            # x_out[k, sensor_num * 9 + 3 + dim] = 0
                            acc_switch[sensor_num, dim] = False

        print('calc finished, x_dim:  ', x_out.shape)
        return x_out
