import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.pyplot import text
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import seaborn as sns
import os
import matplotlib.animation as animation
from scipy.fftpack import fft

from matplotlib import rc
from src.util.data_access_manager import *
import copy

from src.algorithm.quaternion import *


class Plotter:

    def __init__(self, output_dir, run, title='', save_figs=False, format='pdf'):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self._output_file = output_dir + '/' + run + '/'
        # sns.set_style("darkgrid")
        sns.set_style("whitegrid")
        sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'text.usetex': True})
        rc('font', **{'size': 12})
        self._figure_counter = 0
        self._title = run + '_' + title
        self._save_figs = save_figs
        self._format = format

        self._x_color = sns.xkcd_rgb["denim blue"]
        self._y_color = 'green'  # sns.xkcd_rgb["medium green"]
        self._z_color = 'red'  # sns.xkcd_rgb["pale red"]

    def __del__(self):
        plt.close('all')

    def show_plots(self):
        plt.show()

    # Description for imu data information: raw/calibrated with filter XY
    def add_overview_imu_data(self, imu_data, filtered=False):
        print("figure #", self._figure_counter)

        plt.figure('Overview IMU data [' + imu_data.description + ']', figsize=(18, 12))
        self._figure_counter += 1

        print('plot imu_values #', imu_data.num_samples)

        samples = np.arange(imu_data.num_samples)

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel(r'Acceleration /$\mathrm{ms^{-2}}$')
        plt.plot(samples, imu_data.acc_local[:, 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.acc_local[:, 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.acc_local[:, 2], label='z', color=self._z_color)
        if filtered:
            plt.plot(samples, imu_data.acc_local_filtered[:, 0], ':', label='acc_local_x_filtered')
            plt.plot(samples, imu_data.acc_local_filtered[:, 1], ':', label='acc_local_y_filtered')
            plt.plot(samples, imu_data.acc_local_filtered[:, 2], ':', label='acc_local_z_filtered')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 2, sharex=ax1)
        # plt.xlabel('Time [ms]')
        plt.ylabel(r'Angular velocity /$\mathrm{\degree s{^-1}}$')
        plt.plot(samples, imu_data.gyro_deg[:, 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.gyro_deg[:, 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.gyro_deg[:, 2], label='z', color=self._z_color)
        if filtered:
            plt.plot(samples, imu_data.gyro_deg_filtered[:, 0], ':', label='gyro_deg_x_filtered')
            plt.plot(samples, imu_data.gyro_deg[:, 1], ':', label='gyro_deg_y_filtered')
            plt.plot(samples, imu_data.gyro_deg[:, 2], ':', label='gyro_deg_z_filtered')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel(r'Magnetic field /mGs')
        plt.plot(samples, imu_data.magnetic_field[:, 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.magnetic_field[:, 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.magnetic_field[:, 2], label='z', color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_imu_raw_data' + '.' + self._format, format=self._format)
        else:
            plt.draw()

        return

        # Description for imu data information: raw/calibrated with filter XY

    def add_overview_imu_data_with_marker(self, imu_tracker, xmarker=0.0, sensor_num=0):
        print("figure #", self._figure_counter)

        imu_data = imu_tracker.tracking_data[sensor_num].imu_data

        # mark sample when xmarker is reached
        gt = imu_tracker.ground_truth_trajectory
        frame_of_reach = 0

        for i in range(len(gt.pos[:, 0])):
            if gt.pos[i, 0] <= xmarker:
                frame_of_reach = gt.frames[i]
                break;

        sample_of_reach = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_of_reach]

        plt.figure('Overview IMU data with pos marker [' + imu_data.description + ']', figsize=(18, 12))
        self._figure_counter += 1

        print('plot imu_values #', imu_data.num_samples)

        samples = np.arange(imu_data.num_samples)

        ax1 = plt.subplot(3, 1, 1)
        plt.title('Accelerometer, gyroscope and magnetometer data')
        plt.ylabel(r'Acceleration / $\mathrm{m/s^2}$')
        plt.plot(samples, imu_data.acc_local[:, 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.acc_local[:, 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.acc_local[:, 2], label='z', color=self._z_color)
        plt.axvline(sample_of_reach, color='black')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 2, sharex=ax1)
        # plt.xlabel('Time [ms]')
        plt.ylabel(r'Angular velocity / $\mathrm{\degree/s}$')
        plt.plot(samples, imu_data.gyro_deg[:, 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.gyro_deg[:, 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.gyro_deg[:, 2], label='z', color=self._z_color)
        plt.axvline(sample_of_reach, color='black')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel(r'Magnetic field / mGs')
        plt.plot(samples, imu_data.magnetic_field[:, 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.magnetic_field[:, 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.magnetic_field[:, 2], label='z', color=self._z_color)
        plt.axvline(sample_of_reach, color='black')
        text(sample_of_reach, 5, "x=%d" % xmarker, rotation=90, verticalalignment='center')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_imu_data_with_x=' + str(xmarker) + '.' + self._format,
                        format=self._format)
        else:
            plt.draw()

        return

    def plot_acc_data(self, imu_data, range=None):
        print("figure #", self._figure_counter)

        plt.figure('Overview acc data [' + imu_data.description + ']', figsize=(10, 6))
        self._figure_counter += 1

        if range is not None:
            samples = np.arange(range[0], range[1])
        else:
            samples = np.arange(imu_data.num_samples)

        plt.ylabel(r'Local acceleration /$\mathrm{ms^{-2}}$')
        plt.xlabel('Sample')
        plt.plot(samples, imu_data.acc_local[range[0]:range[1], 0], label='x', color=self._x_color)
        plt.plot(samples, imu_data.acc_local[range[0]:range[1], 1], label='y', color=self._y_color)
        plt.plot(samples, imu_data.acc_local[range[0]:range[1], 2], label='z', color=self._z_color)

        # plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_imu_acc_data' + '.' + self._format, format=self._format)
        else:
            plt.draw()

        return


    # Description for imu data information: raw/calibrated with filter XY
    def plot_acc_filter_data(self, imu_tracker, sensor_num=0):
        print("figure #", self._figure_counter)

        plt.figure('Overview filtered acc data [' + imu_tracker.description + ']')
        self._figure_counter += 1

        start_tracking = imu_tracker.tracking_data[sensor_num].start_tracking_index
        end_tracking = imu_tracker.tracking_data[sensor_num].end_tracking_index
        samples = np.arange(start_tracking, end_tracking)

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel('Global acc [m/s**2]')

        print('start: ', start_tracking, ', end: ', end_tracking, ', samples: ', samples)
        print('#filtered: ', len(imu_tracker._filtered_values[:, 6]))
        print('#acc_global: ', len(imu_tracker.tracking_data[sensor_num].acc_global[:, 0]))
        print('q_calc offset: ', imu_tracker.tracking_data[sensor_num].start_q_calc_index)

        acc_offset = start_tracking - imu_tracker.tracking_data[sensor_num].start_q_calc_index

        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[acc_offset:, 0], label='acc_global_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[acc_offset:, 1], label='acc_global_y',
                 color=self._y_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[acc_offset:, 2], label='acc_global_z',
                 color=self._z_color)

        if sensor_num == 0:
            plt.plot(samples, imu_tracker._filtered_values[:, 6], label='acc_filtered_x', color=self._x_color,
                     linestyle='dashed')
            plt.plot(samples, imu_tracker._filtered_values[:, 7], label='acc_filtered_y', color=self._y_color,
                     linestyle='dashed')
            plt.plot(samples, imu_tracker._filtered_values[:, 8], label='acc_filtered_z', color=self._z_color,
                     linestyle='dashed')
        elif sensor_num == 1:
            plt.plot(samples, imu_tracker._filtered_values[:, 15], label='acc_filtered_x', color=self._x_color,
                     linestyle='dashed')
            plt.plot(samples, imu_tracker._filtered_values[:, 16], label='acc_filtered_y', color=self._y_color,
                     linestyle='dashed')
            plt.plot(samples, imu_tracker._filtered_values[:, 17], label='acc_filtered_z', color=self._z_color,
                     linestyle='dashed')

        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        sample_to_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera
        camera_frame_start = sample_to_frame[imu_tracker.tracking_data[sensor_num].start_tracking_index]
        camera_frame_end = sample_to_frame[imu_tracker.tracking_data[sensor_num].end_tracking_index]

        gt_start_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame_start)
        gt_end_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame_end)  # +1

        camera_frames = []
        for i in samples:
            frame = sample_to_frame[i]
            if frame not in camera_frames:
                camera_frames.append(sample_to_frame[i])

        print('LEN ', len(camera_frames))

        plt.subplot(3, 1, 2, sharex=ax1)

        print('dim acc: ', np.shape(imu_tracker.ground_truth_trajectory.acc))
        print('dim v: ', np.shape(imu_tracker.ground_truth_trajectory.v))
        plt.plot(camera_frames, imu_tracker.ground_truth_trajectory.acc[gt_start_index:gt_end_index, 0],
                 label='camera_acc_x', color=self._x_color)
        plt.plot(camera_frames, imu_tracker.ground_truth_trajectory.acc[gt_start_index:gt_end_index, 1],
                 label='camera_acc_y', color=self._y_color)
        # plt.plot(samples, imu_tracker.ground_truth_trajectory.acc[gt_start_index:gt_end_index,2], label='camera_acc_z', color=self._z_color)

        if sensor_num == 0:
            plt.plot(samples, imu_tracker._filtered_values[:, 6], label='acc_filtered_x', color=self._x_color,
                     linestyle='dashed')
            plt.plot(samples, imu_tracker._filtered_values[:, 7], label='acc_filtered_y', color=self._y_color,
                     linestyle='dashed')
            # plt.plot(samples, imu_tracker._filtered_values[:, 8], label='acc_filtered_z', color=self._z_color, linestyle='dashed')
        elif sensor_num == 1:
            plt.plot(samples, imu_tracker._filtered_values[:, 15], label='acc_filtered_x', color=self._x_color,
                     linestyle='dashed')
            plt.plot(samples, imu_tracker._filtered_values[:, 16], label='acc_filtered_y', color=self._y_color,
                     linestyle='dashed')
            # plt.plot(samples, imu_tracker._filtered_values[:, 17], label='acc_filtered_z', color=self._z_color, linestyle='dashed')

        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)

        plt.plot(camera_frames,
                 [LA.norm(x) for x in imu_tracker.ground_truth_trajectory.acc[gt_start_index:gt_end_index, 0:3]],
                 label='camera_norm_acc', color=self._x_color)

        if sensor_num == 0:
            plt.plot(samples, [LA.norm(x) for x in imu_tracker._filtered_values[:, 6:8]], label='imu_norm_acc',
                     color=self._x_color, linestyle='dashed')
        elif sensor_num == 1:
            plt.plot(samples, [LA.norm(x) for x in imu_tracker._filtered_values[:, 15:17]], label='imu_norm_acc',
                     color=self._x_color, linestyle='dashed')

        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_acc_filtered_imu_tracker' + '.' + self._format,
                        format=self._format)

        return

    def add_fft_acc(self, imu_data, range):
        print("figure #", self._figure_counter)

        plt.figure('Overview IMU data [' + imu_data.description + ']')
        self._figure_counter += 1

        print('plot imu_values #', imu_data.num_samples)

        samples = np.arange(range[0], range[1])

        ax1 = plt.subplot(2, 1, 1)
        plt.title('Accelerometer data and FFT')
        plt.ylabel('Acceleration [m/s**2]')
        plt.plot(samples, imu_data.acc_local[range[0]:range[1], 0], label='acc_local_x')
        plt.plot(samples, imu_data.acc_local[range[0]:range[1], 1], label='acc_local_y')
        plt.plot(samples, imu_data.acc_local[range[0]:range[1], 2], label='acc_local_z')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')
        plt.xlabel('Sample')

        N = len(samples)
        T = 1 / imu_data.sample_rate
        x = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))

        yr_x = fft(imu_data.acc_local[range[0]:range[1], 2])
        y_x = 2 / N * np.abs(yr_x[0:np.int(N / 2)])

        plt.subplot(2, 1, 2)
        plt.xlabel('Freq [Hz]')
        plt.plot(x, y_x)

    def add_imu_trajectory_3D(self, imu_data):
        print("figure #", self._figure_counter)
        fig = plt.figure('Trajectories[' + imu_data.description + ']')
        self._figure_counter += 1

        ax = fig.add_subplot(211, projection='3d')
        ax.set_title('Caclulated trajectories ')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        trajectory = imu_data.trajectory
        minimum = min(min(trajectory.pos[:, 0]), min(trajectory.pos[:, 1]))
        maximum = max(max(trajectory.pos[:, 0]), max(trajectory.pos[:, 1]))
        ax.set_xlim(minimum, maximum)
        ax.set_ylim(minimum, maximum)

        # TODO switch X and Y for Andro Sensor?
        cm = plt.cm.get_cmap('rainbow')
        colors = np.arange(trajectory.t[0], trajectory.t[-1] + 1, 1 / imu_data.sample_rate * 1000)
        print('START ', trajectory.t[0])
        traj_plot = ax.scatter(trajectory.pos[:, 0], trajectory.pos[:, 1], trajectory.pos[:, 2], c=colors,
                               edgecolors='none', label='pos', cmap=cm)

        plt.subplot(212)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')

        plt.scatter(trajectory.pos[:, 0], trajectory.pos[:, 1], c=colors, label='pos', edgecolors='none', cmap=cm)
        plt.colorbar()

    # access to:
    # quaternions = np.array([Quaternions()])
    # euler_angles = np.array([[roll, pitch, yaw]]
    # description, t_ms
    def plot_angles(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        description = imu_tracker.description

        plt.figure('Quaternions and euler angles sensor (' + str(sensor_num) + ')')
        self._figure_counter += 1

        quaternions = imu_tracker.tracking_data[sensor_num].quaternions
        euler_angles = imu_tracker.tracking_data[sensor_num].euler_angles_unwrapped_deg

        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_q_calc_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        ax1 = plt.subplot(2, 1, 1)
        plt.title('Calculated quaternions and euler angles')
        plt.ylabel('Imaginary part')
        plt.ylim([-1.1, 1.1])
        x = np.array([q[1] for q in quaternions])
        y = np.array([q[2] for q in quaternions])
        z = np.array([q[3] for q in quaternions])

        plt.plot(samples, x, label="q1")
        plt.plot(samples, y, label="q2")
        plt.plot(samples, z, label="q3")
        plt.legend(loc='upper left')

        # Plot euler angles

        plt.subplot(2, 1, 2, sharex=ax1)
        plt.xlabel('Time [ms]')
        plt.ylabel('Euler angle [deg]')
        plt.plot(samples, euler_angles[:, 0], label='roll_x')
        plt.plot(samples, euler_angles[:, 1], label='pitch_y')
        plt.plot(samples, euler_angles[:, 2], label='yaw_z')
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_angles' + '.' + self._format, format=self._format)

    def plot_overview_acc_standstill_data(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview acc standstill data ' + self._title, figsize=(10, 10))
        self._figure_counter += 1

        start = 300  # imu_tracker.imu_data[sensor_num].steady_phase[0]
        end = 500  # imu_tracker.imu_data[sensor_num].steady_phase[1]
        samples = np.arange(start, end)

        ax1 = plt.subplot(2, 1, 1)
        plt.ylabel(r'Local linear acc / $m/s^2$')
        plt.ylim(-10, 10)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].acc_local_filtered[start:end, 0], label='acc_local_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].acc_local_filtered[start:end, 1], label='acc_local_y',
                 color=self._y_color)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].acc_local_filtered[start:end, 2], label='acc_local_z',
                 color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper right')

        print('local  abs: ', LA.norm(imu_tracker.imu_data[sensor_num].acc_local_filtered[start]))

        plt.subplot(2, 1, 2, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel(r'Global linear acc / $m/s^2$')
        plt.ylim(-0.1, 0.1)
        start_tmp = start - imu_tracker.tracking_data[sensor_num].start_q_calc_index
        end_tmp = end - imu_tracker.tracking_data[sensor_num].start_q_calc_index
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_tmp:end_tmp, 0],
                 label='acc_global_lin_x', color=self._x_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_tmp:end_tmp, 1],
                 label='acc_global_lin_y', color=self._y_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_tmp:end_tmp, 2],
                 label='acc_global_lin_z', color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper right')

        # plt.subplot(3, 1, 3)
        # plt.ylabel(r'Rotation rate / $\deg/s$')
        # plt.ylim(-10,10)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].gyro_deg_filtered[start:end, 0], label='gyro_x',
        #          color=self._x_color)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].gyro_deg_filtered[start:end, 1], label='gyro_y',
        #          color=self._y_color)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].gyro_deg_filtered[start:end, 2], label='gyro_z',
        #          color=self._z_color)
        # plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(
                self._output_file + self._title + '_overview_starting_phase' + str(sensor_num) + '.' + self._format,
                format=self._format)

    def plot_overview_imu_run_data(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview acc data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 10))
        self._figure_counter += 1

        start = imu_tracker.tracking_data[sensor_num].start_tracking_index  # - imu_tracker.start_q_calc_index
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index  # - imu_tracker.start_q_calc_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))
        # print('SECONDS: ', seconds)

        start_q = start - imu_tracker.tracking_data[sensor_num].start_q_calc_index
        end_q = end - imu_tracker.tracking_data[sensor_num].start_q_calc_index

        enter_bott = imu_tracker.tracking_data[sensor_num].camera_to_imu[397] / float(
            imu_tracker.imu_data[sensor_num].sample_rate)  # run0505D5
        leave_bott = imu_tracker.tracking_data[sensor_num].camera_to_imu[452] / float(
            imu_tracker.imu_data[sensor_num].sample_rate)
        # print('enter bott: ', enter_bott)
        # print('leave bott: ', leave_bott)

        ax1 = plt.subplot(2, 1, 1)
        plt.ylabel(r'Local linear acceleration /$\mathrm{ms^{-2}}$')
        # plt.ylabel(r'Beschleunigung / m/s^2')
        plt.ylim(-1.5, 1.5)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_local_lin[start_q:end_q, 0], label='x',
                 color=self._x_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_local_lin[start_q:end_q, 1], label='y',
                 color=self._y_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_local_lin[start_q:end_q, 2], label='z',
                 color=self._z_color)
        plt.axvline(enter_bott, color='black', linestyle='--')
        plt.axvline(leave_bott, color='black', linestyle='--')
        # plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        # ax1=plt.subplot(2, 1, 1)
        # plt.ylabel(r'Global linear acceleration / $\mathrm{m/s^2}$')
        # #plt.ylabel(r'Beschleunigung / m/s^2')
        # #plt.ylim(-1.5, 1.5)
        # plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 0], label='x', color=self._x_color)
        # plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 1], label='y', color=self._y_color)
        # plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 2], label='z', color=self._z_color)
        # plt.axvline(enter_bott, color='black', linestyle='--')
        # plt.axvline(leave_bott, color='black', linestyle='--')
        # #plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper left')

        plt.subplot(2, 1, 2, sharex=ax1)
        plt.ylabel(r'Angular velocity /$\mathrm{\degree s^{-1}}$')
        # plt.ylabel(r'Drehrate / \degree/s')
        # plt.ylim(-1, 1)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 0], label='x', color=self._x_color)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 1], label='y', color=self._y_color)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 2], label='z', color=self._z_color)
        plt.axvline(enter_bott, color='black', linestyle='--')
        plt.axvline(leave_bott, color='black', linestyle='--')
        # plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        # plt.subplot(3, 1, 3, sharex=ax1)
        #
        # plt.ylabel(r'Magnetic field / mGs')
        # plt.ylabel(r'Magnetfeld / mGs')
        # #plt.ylim(-1, 1)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].magnetic_field[start:end, 0], label='x', color=self._x_color)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].magnetic_field[start:end, 1], label='y', color=self._y_color)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].magnetic_field[start:end, 2], label='z', color=self._z_color)

        # plt.axvline(enter_bott, color='black', linestyle='--')
        # plt.axvline(leave_bott, color='black', linestyle='--')

        # plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper left')

        plt.xlabel(r'Time /s')
        # plt.xlabel(r'Zeit / $s$')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_imu_run_data_sensor' + str(
                sensor_num) + '.' + self._format, format=self._format)

        plt.close()

    def plot_overview_raw_imu_run_data(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview of raw imu data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 12))
        self._figure_counter += 1

        start = imu_tracker.tracking_data[sensor_num].start_tracking_index  # - imu_tracker.start_q_calc_index
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index  # - imu_tracker.start_q_calc_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))
        # print('SECONDS: ', seconds)

        start_q = start - imu_tracker.tracking_data[sensor_num].start_q_calc_index
        end_q = end - imu_tracker.tracking_data[sensor_num].start_q_calc_index

        enter_bott = imu_tracker.tracking_data[sensor_num].camera_to_imu[397] / float(
            imu_tracker.imu_data[sensor_num].sample_rate)  # run0505D5
        leave_bott = imu_tracker.tracking_data[sensor_num].camera_to_imu[452] / float(
            imu_tracker.imu_data[sensor_num].sample_rate)
        # print('enter bott: ', enter_bott)
        # print('leave bott: ', leave_bott)

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel(r'Global acceleration /$\mathrm{ms^{-2}}$')
        # plt.ylim(-1.5, 1.5)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global[start_q:end_q, 0], label='x',
                 color=self._x_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global[start_q:end_q, 1], label='y',
                 color=self._y_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global[start_q:end_q, 2], label='z',
                 color=self._z_color)
        plt.axvline(enter_bott, color='black', linestyle='--')
        plt.axvline(leave_bott, color='black', linestyle='--')
        # plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        ax1 = plt.subplot(3, 1, 2)
        plt.ylabel(r'Global linear acceleration /$\mathrm{ms^{-2}}$')
        # plt.ylim(-1.5, 1.5)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 0], label='x',
                 color=self._x_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 1], label='y',
                 color=self._y_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 2], label='z',
                 color=self._z_color)
        plt.axvline(enter_bott, color='black', linestyle='--')
        plt.axvline(leave_bott, color='black', linestyle='--')
        # plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        # plt.subplot(3, 1, 2, sharex=ax1)
        # plt.ylabel(r'Angular velocity /$\mathrm{\degree s^{-1}}$')
        # # plt.ylabel(r'Drehrate / \degree/s')
        # # plt.ylim(-1, 1)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 0], label='x', color=self._x_color)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 1], label='y', color=self._y_color)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 2], label='z', color=self._z_color)
        # plt.axvline(enter_bott, color='black', linestyle='--')
        # plt.axvline(leave_bott, color='black', linestyle='--')
        # # plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.ylabel(r'Magnetic field /mGs')
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].magnetic_field[start:end, 0], label='x', color=self._x_color)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].magnetic_field[start:end, 1], label='y', color=self._y_color)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].magnetic_field[start:end, 2], label='z', color=self._z_color)

        plt.axvline(enter_bott, color='black', linestyle='--')
        plt.axvline(leave_bott, color='black', linestyle='--')
        plt.legend(loc='upper left')

        # plt.margins(x=0.05, y=0.1)
        plt.xlabel(r'Time /s')
        # plt.xlabel(r'Zeit / $s$')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_raw_imu_run_data_sensor' + str(
                sensor_num) + '.' + self._format, format=self._format)

        plt.close()

    def plot_global_linear_acc(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview acc data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 6))
        self._figure_counter += 1

        start = imu_tracker.tracking_data[sensor_num].start_tracking_index  # - imu_tracker.start_q_calc_index
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index  # - imu_tracker.start_q_calc_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))
        print('SECONDS: ', seconds)

        start_q = start - imu_tracker.tracking_data[sensor_num].start_q_calc_index
        end_q = end - imu_tracker.tracking_data[sensor_num].start_q_calc_index

        ax1 = plt.subplot(111)
        plt.ylabel(r'Global linear acceleration /$\mathrm{ms^{-2}}$')
        # plt.ylabel(r'Beschleunigung / m/s^2')
        # plt.ylim(-1.5, 1.5)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 0], label='x',
                 color=self._x_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 1], label='y',
                 color=self._y_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 2], label='z',
                 color=self._z_color)
        # plt.axvline(enter_bott, color='black', linestyle='--')
        # plt.axvline(leave_bott, color='black', linestyle='--')
        # plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        # plt.subplot(212, sharex=ax1)
        # plt.ylabel(r'Angular velocity /$\mathrm{\degree s^{-1}}$')
        # # plt.ylabel(r'Drehrate / \degree/s')
        # # plt.ylim(-1, 1)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 0], label='x', color=self._x_color)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 1], label='y', color=self._y_color)
        # plt.plot(seconds, imu_tracker.imu_data[sensor_num].gyro_deg[start:end, 2], label='z', color=self._z_color)
        # # plt.axvline(enter_bott, color='black', linestyle='--')
        # # plt.axvline(leave_bott, color='black', linestyle='--')
        # # plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper left')

        # plt.ylim([-3,3])

        plt.xlabel(r'Time /s')
        # plt.xlabel(r'Zeit / $s$')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_global_linear_acc_data_sensor' + str(
                sensor_num) + '_new.' + self._format, format=self._format)

        plt.close()

    def plot_overview_acc_run_data(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview acc data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 6))
        self._figure_counter += 1

        start = imu_tracker.tracking_data[sensor_num].start_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_q_calc_index
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_q_calc_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        # ax1=plt.subplot(2, 1, 1)
        # plt.ylabel(r'Local linear acc / $m/s^2$')
        # plt.ylim(-0.7, 0.7)
        # plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_local_lin[start:end, 0], label='acc_local_lin_x', color=self._x_color)
        # plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_local_lin[start:end, 1], label='acc_local_lin_y', color=self._y_color)
        # plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_local_lin[start:end, 2], label='acc_local_lin_z', color=self._z_color)
        # plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper right')

        # plt.subplot(2, 1, 2, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel(r'Global linear acc /$m/s^2$')
        plt.ylim(-1, 1)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[start:end, 0], label='acc_global_lin_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[start:end, 1], label='acc_global_lin_y',
                 color=self._y_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[start:end, 2], label='acc_global_lin_z',
                 color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_acc_run_data_sensor' + str(
                sensor_num) + '.' + self._format, format=self._format)

    def plot_overview_velocity_run_data(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview velocity data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 6))
        self._figure_counter += 1

        start = 0
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        plt.xlabel('Sample')
        plt.ylabel(r'Velocity /$m/s$')
        # plt.ylim(-1, 1)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.v[start:end, 0], label='v_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.v[start:end, 1], label='v_y',
                 color=self._y_color)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.v[start:end, 2], label='v_z',
        #          color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(
                self._output_file + self._title + '_overview_velocity_run_data_sensor' + str(sensor_num) + self._format,
                format=self._format)

    def plot_overview_pos_run_data(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview position data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 6))
        self._figure_counter += 1

        start = 0
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index - imu_tracker.tracking_data[
            sensor_num].start_tracking_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        plt.xlabel('Sample')
        plt.ylabel(r'Travelled distance /$m/s$')
        # plt.ylim(-1, 1)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.pos[start:end, 0], label='s_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.pos[start:end, 1], label='s_y',
                 color=self._y_color)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.pos[start:end, 2], label='s_z',
        #          color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(
                self._output_file + self._title + '_overview_distance_run_data_sensor' + str(sensor_num) + self._format,
                format=self._format)

    def add_overview_imu_tracker(self, imu_tracker, sensor_num=0):

        print("figure #", self._figure_counter)
        plt.figure('Overview IMU tracker sensor ' + '(' + str(sensor_num) + ') ', figsize=(16, 10))
        self._figure_counter += 1

        samples = np.arange(len(imu_tracker.tracking_data[sensor_num].quaternions)) + imu_tracker.tracking_data[
            sensor_num].start_q_calc_index

        plt.title('Calculated global data')

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel('Global acc /$m/s^2$')
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global[:, 0], label='acc_global_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global[:, 1], label='acc_global_y',
                 color=self._y_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global[:, 2], label='acc_global_z',
                 color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.ylabel('Global linear acc /$m/s^2$')
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[:, 0], label='acc_global_lin_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[:, 1], label='acc_global_lin_y',
                 color=self._y_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[:, 2], label='acc_global_lin_z',
                 color=self._z_color)
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel('Euler angle /°')
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].euler_angles_deg[:, 0], label='roll_x',
                 color=self._x_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].euler_angles_deg[:, 1], label='pitch_y',
                 color=self._y_color)
        plt.plot(samples, imu_tracker.tracking_data[sensor_num].euler_angles_deg[:, 2], label='yaw_z',
                 color=self._z_color)
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_imu_tracker' + '.' + self._format,
                        format=self._format)

    def add_overview_pos_tracking(self, imu_tracker, sensor_num=0):

        # Plot quaternions and euler angles
        # self.plot_angles(imu_tracker)

        # Plot acc_global and acc_local_lin in another figure
        print("figure #", self._figure_counter)

        plt.figure('Overview position tracking data [' + imu_tracker.description + ']', figsize=(10, 12))
        self._figure_counter += 1

        start_tracking = imu_tracker.tracking_data[sensor_num].start_tracking_index
        end_tracking = imu_tracker.tracking_data[sensor_num].end_tracking_index
        samples = np.arange(start_tracking, end_tracking)
        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))

        print('test')

        plt.title('Position tracking data')

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel(r'Global linear acceleration /$\mathrm{ms^{-2}}$')

        acc_offset = start_tracking - imu_tracker.tracking_data[sensor_num].start_q_calc_index

        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[acc_offset:, 0], label='x',
                 color=self._x_color)
        plt.plot(seconds, imu_tracker.tracking_data[sensor_num].acc_global_lin[acc_offset:, 1], label='y',
                 linestyle='--', color=self._y_color)
        # plt.plot(samples, imu_tracker.tracking_data[sensor_num].acc_global_lin[acc_offset:, 2], label='acc_global_lin_z')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.ylabel(r'Velocity /$\mathrm{ms^{-1}}$')
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].trajectory.v[:, 0], label='x', color=self._x_color)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].trajectory.v[:, 1], label='y', linestyle='--',
                 color=self._y_color)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.v[:, 2], label='v_z')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.xlabel('Time /s')
        plt.ylabel('Position /m')
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].trajectory.pos[:, 0], label='x', color=self._x_color)
        plt.plot(seconds, imu_tracker.imu_data[sensor_num].trajectory.pos[:, 1], label='y', linestyle='--',
                 color=self._y_color)
        # plt.plot(samples, imu_tracker.imu_data[sensor_num].trajectory.pos[:, 2], label='s_z')
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_pos_tracking' + '.' + self._format,
                        format=self._format)

    def add_overview_pos_tracking_filtered2D(self, imu_tracker, sensor_num=0):

        # Plot quaternions and euler angles
        # self.plot_angles(imu_tracker)

        # Plot acc_global and acc_local_lin in another figure
        print("figure #", self._figure_counter)

        plt.figure('Overview filtered position tracking data sensor ' + '(' + str(sensor_num) + ') ', figsize=(10, 15))
        self._figure_counter += 1

        start_tracking = imu_tracker.tracking_data[sensor_num].start_tracking_index
        end_tracking = imu_tracker.tracking_data[sensor_num].end_tracking_index
        samples = np.arange(start_tracking, end_tracking)

        plt.title('Position tracking data for sensor #' + str(sensor_num))

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel(r'Global linear acc /$ms^{-2}$')

        sensor_offset = sensor_num * 6

        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 4], label='acc_global_lin_x')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 5], label='acc_global_lin_y')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.ylabel(r'Velocity /$ms^{-1}$]')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 2], label='v_x')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 3], label='v_y')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel(r'Distance /$m$')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 0], label='s_x')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 1], label='s_y')
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_pos_tracking_filtered2D_' + str(
                sensor_num) + '.' + self._format, format=self._format)

    def add_overview_pos_tracking_filtered3D(self, imu_tracker, sensor_num=0):

        # Plot quaternions and euler angles
        # self.plot_angles(imu_tracker)

        # Plot acc_global and acc_local_lin in another figure
        print("figure #", self._figure_counter)

        plt.figure('Overview filtered position tracking data sensor ' + '(' + str(sensor_num) + ') ')
        self._figure_counter += 1

        start_tracking = imu_tracker.tracking_data[sensor_num].start_tracking_index
        end_tracking = imu_tracker.tracking_data[sensor_num].end_tracking_index
        samples = np.arange(start_tracking, end_tracking)

        plt.title('Position tracking data for sensor #' + str(sensor_num))

        ax1 = plt.subplot(3, 1, 1)
        plt.ylabel('Global lin acc [m/s**2]')

        acc_offset = start_tracking - imu_tracker.tracking_data[sensor_num].start_q_calc_index
        sensor_offset = sensor_num * 9

        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 6], label='acc_global_lin_x')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 7], label='acc_global_lin_y')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 8], label='acc_global_lin_z')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.ylabel('Velocity [m/s]')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 3], label='v_x')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 4], label='v_y')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 5], label='v_z')
        plt.margins(x=0.05, y=0.1)
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.xlabel('Sample')
        plt.ylabel('Pos [m]')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 0], label='s_x')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 1], label='s_y')
        plt.plot(samples, imu_tracker._filtered_values[:, sensor_offset + 2], label='s_z')
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_overview_pos_tracking_filtered' + '.' + self._format,
                        format=self._format)

    def plot_rotation_validation_difference(self, imu_tracker, angle_diff, file_ext='', xmarker=None, sensor_num=0):
        print("figure #", self._figure_counter)
        plt.figure('Rotation validation for ' + imu_tracker.imu_data[sensor_num].description + file_ext,
                   figsize=(12, 5))
        self._figure_counter += 1

        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))

        plt.plot(seconds, angle_diff)
        print('Outputfile: ', self._output_file)

        if self._output_file == '../private/Output/OptitrackExperiments/Session1/':
            runs = [4.0, 12.28, 19.84, 26.32, 32.72]
            middle = [9.04, 16.64, 23.4, 29.76, 36.52]

            for i in range(len(runs)):
                plt.axvline(runs[i], color='black')
                text(runs[i] + 0.3, -19, "Run %d" % (i + 1), rotation=90, ha='left', va='bottom')

            for i in range(len(middle)):
                plt.axvline(middle[i], color='black', linestyle='--')
                text(middle[i] + 0.3, -19, "Middle", rotation=90, ha='left', va='bottom')

            plt.ylim(-20, 20)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session2/':
            turns = [9.5, 11.5, 16.9, 19.34, 26.9, 29.34, 35.42, 38.34, 45.5, 49.5, 54.94, 57.86, 64.54, 67.26, 72.06,
                     74.78, 81.34, 84.26, 88.9, 93.46]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(4.1, color='black')
            text(4.1 + 0.3, -19, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.3, -19, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-20, 20)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session3/':
            turns = [7.7, 10.22, 13.66, 17.86, 22.1, 24.66, 27.7, 30.66, 35.3, 38.14, 42.02, 45.7, 49.3, 52.5, 55.46,
                     58.58, 63.02, 65.62, 69.06, 72.98]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(3.7, color='black')
            text(3.7 + 0.3, -19, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.3, -19, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-20, 20)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session6/':
            turns = [32.91, 38.91, 44.67, 50.51, 56.51, 61.03, 65.87, 70.35, 75.79, 79.91,
                     84.27, 89.51, 94.87, 100.63, 106.35, 111.99]  # 10.71, 16.0, 22.19, 27.55,
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(3.75, color='black')
            text(3.75 + 0.5, 25, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.5, 25, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-40, 40)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session7/':
            turns = [15.54, 29.3, 34.7, 47.66, 52.66, 72.3, 78.06, 88.5, 94.1, 105.94, 112.9, 124.1, 129.58, 141.7,
                     147.7, 155.98]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(9.5, color='black')
            text(9.5 + 0.5, 25, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.5, 25, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-40, 40)

        if xmarker is not None:
            gt = imu_tracker.ground_truth_trajectory
            for i in range(len(gt.pos[:, 0])):
                if gt.pos[i, 0] <= xmarker:
                    frame_of_reach = gt.frames[i]
                    time_of_reach = frame_of_reach / gt.fps
                    break;
            plt.axvline(time_of_reach, color='black')
            text(time_of_reach, 5, "x=%d" % xmarker, rotation=90, verticalalignment='top')

        # plt.ylim(-25,20)

        plt.xlabel('Time /s')
        plt.ylabel('Difference between angles /°')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_rotation_validation' + file_ext + '.' + self._format,
                        format=self._format)

    def plot_rotation_validation_comparison(self, imu_tracker, estiamted_heading, gt_heading, file_ext='', xmarker=None,
                                            sensor_num=0):
        print("figure #", self._figure_counter)
        plt.figure('Rotation validation for ' + imu_tracker.imu_data[sensor_num].description + file_ext,
                   figsize=(12, 5))
        self._figure_counter += 1

        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))

        plt.plot(seconds, estiamted_heading, label="Calculated heading")
        plt.plot(seconds, gt_heading, label="GT heading")

        print('Outputfile: ', self._output_file)

        if self._output_file == '../private/Output/OptitrackExperiments/Session1/':
            runs = [4.0, 12.28, 19.84, 26.32, 32.72]
            middle = [9.04, 16.64, 23.4, 29.76, 36.52]

            for i in range(len(runs)):
                plt.axvline(runs[i], color='black')
                text(runs[i] + 0.3, -19, "Run %d" % (i + 1), rotation=90, ha='left', va='bottom')

            for i in range(len(middle)):
                plt.axvline(middle[i], color='black', linestyle='--')
                text(middle[i] + 0.3, -19, "Middle", rotation=90, ha='left', va='bottom')

            # plt.ylim(-20, 20)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session2/':
            turns = [9.5, 11.5, 16.9, 19.34, 26.9, 29.34, 35.42, 38.34, 45.5, 49.5, 54.94, 57.86, 64.54, 67.26, 72.06,
                     74.78, 81.34, 84.26, 88.9, 93.46]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(4.1, color='black')
            text(4.1 + 0.3, -19, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.3, -19, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            # plt.ylim(-20, 20)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session3/':
            turns = [7.7, 10.22, 13.66, 17.86, 22.1, 24.66, 27.7, 30.66, 35.3, 38.14, 42.02, 45.7, 49.3, 52.5, 55.46,
                     58.58, 63.02, 65.62, 69.06, 72.98]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(3.7, color='black')
            text(3.7 + 0.3, -19, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.3, -19, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            # plt.ylim(-20, 20)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session6/':
            turns = [32.91, 38.91, 44.67, 50.51, 56.51, 61.03, 65.87, 70.35, 75.79, 79.91,
                     84.27, 89.51, 94.87, 100.63, 106.35, 111.99]  # 10.71, 16.0, 22.19, 27.55,
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(3.75, color='black')
            text(3.75 + 0.5, 25, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.5, 25, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            # plt.ylim(-40, 40)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session7/':
            turns = [15.54, 29.3, 34.7, 47.66, 52.66, 72.3, 78.06, 88.5, 94.1, 105.94, 112.9, 124.1, 129.58, 141.7,
                     147.7, 155.98]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(9.5, color='black')
            text(9.5 + 0.5, 25, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.5, 25, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            # plt.ylim(-40, 40)

        if xmarker is not None:
            gt = imu_tracker.ground_truth_trajectory
            for i in range(len(gt.pos[:, 0])):
                if gt.pos[i, 0] <= xmarker:
                    frame_of_reach = gt.frames[i]
                    time_of_reach = frame_of_reach / gt.fps
                    break;
            plt.axvline(time_of_reach, color='black')
            text(time_of_reach, 5, "x=%d" % xmarker, rotation=90, verticalalignment='top')

        plt.xlabel('Time /s')
        plt.ylabel('Angle /°')
        plt.legend(loc='lower left')

        if self._save_figs:
            plt.savefig(
                self._output_file + self._title + '_rotation_validation_comparison' + file_ext + '.' + self._format,
                format=self._format)

    def plot_lifted_angles_validation(self, imu_tracker, estiamted_heading, gt_heading, offset, file_ext='',
                                      xmarker=None, sensor_num=0):
        print("figure #", self._figure_counter)
        plt.figure('Rotation validation for ' + imu_tracker.imu_data[sensor_num].description + file_ext,
                   figsize=(12, 5))
        self._figure_counter += 1

        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))

        plt.plot(seconds, gt_heading, label=r"$\alpha$")
        plt.plot(seconds, estiamted_heading, label=r"$\beta$")

        print('Outputfile: ', self._output_file)

        if self._output_file == '../private/Output/OptitrackExperiments/Session1/':
            runs = [4.0, 12.28, 19.84, 26.32, 32.72]
            middle = [9.04, 16.64, 23.4, 29.76, 36.52]

            for i in range(len(runs)):
                plt.axvline(runs[i], color='black')
                text(runs[i] + 0.3, 125, "Run %d" % (i + 1), rotation=90, ha='left', va='bottom')

            for i in range(len(middle)):
                plt.axvline(middle[i], color='black', linestyle='--')
                text(middle[i] + 0.3, 125, "Middle", rotation=90, ha='left', va='bottom')

            plt.ylim(100, 400)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session2/':
            turns = [9.5, 11.5, 16.9, 19.34, 26.9, 29.34, 35.42, 38.34, 45.5, 49.5, 54.94, 57.86, 64.54, 67.26, 72.06,
                     74.78, 81.34, 84.26, 88.9, 93.46]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(4.1, color='black')
            text(4.1 + 0.3, -75, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.3, -75, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-100, 500)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session3/':
            turns = [7.7, 10.22, 13.66, 17.86, 22.1, 24.66, 27.7, 30.66, 35.3, 38.14, 42.02, 45.7, 49.3, 52.5, 55.46,
                     58.58, 63.02, 65.62, 69.06, 72.98]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(3.7, color='black')
            text(3.7 + 0.3, -75, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.3, -75, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-100, 1050)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session6/':
            turns = [10.71, 16.0, 22.19, 27.55, 32.91, 38.91, 44.67, 50.51, 56.51, 61.03, 65.87, 70.35, 75.79, 79.91,
                     84.27, 89.51, 94.87, 100.63, 106.35, 111.99]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i, turns[i], turns[i + 1])

            plt.axvline(3.75, color='black')
            text(3.75 + 0.5, 175, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.5, 175, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(150, 550)

        elif self._output_file == '../private/Output/OptitrackExperiments/Session7/':
            turns = [15.54, 29.3, 34.7, 47.66, 52.66, 72.3, 78.06, 88.5, 94.1, 105.94, 112.9, 124.1, 129.58, 141.7,
                     147.7, 155.98]
            for i in range(0, len(turns), 2):
                plt.axvspan(turns[i], turns[i + 1], alpha=0.2, color='grey')
                print('i: ', i)

            plt.axvline(9.5, color='black')
            text(9.5 + 0.5, -225, "Run 1", rotation=90, ha='left', va='bottom')

            counter = 2
            for i in range(3, len(turns) - 1, 4):
                plt.axvline(turns[i], color='black')
                text(turns[i] + 0.5, -225, "Run %d" % counter, rotation=90, ha='left', va='bottom')
                counter += 1

            plt.ylim(-250, 300)

        if xmarker is not None:
            gt = imu_tracker.ground_truth_trajectory
            for i in range(len(gt.pos[:, 0])):
                if gt.pos[i, 0] <= xmarker:
                    frame_of_reach = gt.frames[i]
                    time_of_reach = frame_of_reach / gt.fps
                    break;
            plt.axvline(time_of_reach, color='black')
            text(time_of_reach, 5, "x=%d" % xmarker, rotation=90, verticalalignment='top')

        plt.xlabel('Time /s')
        plt.ylabel('Angle /°')
        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_lifted_angles' + file_ext + '.' + self._format,
                        format=self._format)



    def plot_trajectory(self, traj, axis_limits, range=None, geometry=None):
        print("figure #", self._figure_counter)
        fig = plt.figure('Trajectory plot ' + str(traj.identifier), figsize=(10, 6))
        self._figure_counter += 1

        if range is None:
            start = 0
            end = -1
        else:
            start = range[0]
            end = range[1]

        plt.plot(traj.pos[start:end:4, 0], traj.pos[start:end:4, 1], label='GT trajectory')

        if geometry == 'SiME':
            visuWalls = np.array((
                [-3.0, 1.65],
                [0.00, 1.65],
                [0.00, 1.65],
                [0.00, 0.45],
                [0.00, 0.45],
                [2.40, 0.45],
                [-3.0, -1.65],
                [0.00, -1.65],
                [0.00, -1.65],
                [0.00, -0.45],
                [0.00, -0.45],
                [2.40, -0.45],
            ))
            for i in range(0, visuWalls.shape[0] - 1,
                           2):  # step = 2 to plot lines
                plt.plot([visuWalls[i, 0], visuWalls[i + 1, 0]],
                         [visuWalls[i, 1], visuWalls[i + 1, 1]],
                         linewidth=2,
                         color='black',
                         zorder=70
                         )
        elif geometry == 'Rotunda40':
            visuWalls = np.array((
                [-3.0, 1.65],
                [0.00, 1.65],
                [0.00, 1.65],
                [0.00, 0.45],
                [0.00, 0.45],
                [2.40, 0.45],
                [-3.0, -1.65],
                [0.00, -1.65],
                [0.00, -1.65],
                [0.00, -0.45],
                [0.00, -0.45],
                [2.40, -0.45],
            ))
            for i in range(0, visuWalls.shape[0] - 1,
                           2):  # step = 2 to plot lines
                plt.plot([visuWalls[i, 0], visuWalls[i + 1, 0]],
                         [visuWalls[i, 1], visuWalls[i + 1, 1]],
                         linewidth=2,
                         color='black',
                         zorder=70
                         )

        plt.xlabel(r'x /m')
        plt.ylabel(r'y /m')
        plt.xlim(axis_limits[0], axis_limits[1])
        plt.ylim(axis_limits[2], axis_limits[3])

        plt.axis('equal')
        # plt.tight_layout()

        # ax = fig.gca()
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1.00))
        # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_traj_' + str(traj.identifier) + '_surr.' + self._format,
                        format=self._format)

    def plot_resulting_trajectory(self, imu_tracker, withAdaption=False, sensor_num=0, withGeometry=False,
                                  surrounding_traj=False):
        print("figure #", self._figure_counter)
        fig = plt.figure('Calculated trajectory for ' + imu_tracker.imu_data[sensor_num].description, figsize=(10, 6))
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        sample_to_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera
        camera_frame_start = sample_to_frame[imu_tracker.tracking_data[sensor_num].start_tracking_index]
        camera_frame_end = sample_to_frame[imu_tracker.tracking_data[sensor_num].end_tracking_index]

        print('start frame: ', camera_frame_start, ', end: ', camera_frame_end)

        gt_start_index = gt.get_index_from_frame(camera_frame_start)
        gt_end_index = gt.get_index_from_frame(camera_frame_end)
        # pusher_start_index = pusher.get_index_from_frame(camera_frame_start)
        # pusher_end_index = pusher.get_index_from_frame(camera_frame_end)

        if surrounding_traj:
            surrounding_traj = imu_tracker.surrounding_trajectory
            plt.scatter(surrounding_traj.pos[:, 0], surrounding_traj.pos[:, 1],
                        label='Pusher trajectory', s=2, color='darkgray')

            plt.scatter(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                        label='Wheelchair trajectory', s=4)
        else:
            plt.scatter(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                        label='Camera trajectory', s=4)

        plt.scatter(imu_tracker.imu_data[sensor_num].trajectory.pos[:, 0],
                    imu_tracker.imu_data[sensor_num].trajectory.pos[:, 1], label='IMU trajectory', s=4)

        if withAdaption:
            x_corrected = []
            y_corrected = []
            for sample in imu_tracker.tracking_data[sensor_num].acc_adaption_corrected_samples:
                x_corrected.append(
                    imu_tracker.imu_data[sensor_num].trajectory.pos[
                        sample - imu_tracker.tracking_data[sensor_num].start_tracking_index][0])
                y_corrected.append(
                    imu_tracker.imu_data[sensor_num].trajectory.pos[
                        sample - imu_tracker.tracking_data[sensor_num].start_tracking_index][1])

            plt.scatter(x_corrected, y_corrected, label='Adaption point', color='black', marker="x", s=30)

        # plot alignment points

        camera_pos_start = gt.get_position(
            imu_tracker.tracking_data[sensor_num].imu_to_camera[
                imu_tracker._tracking_data[sensor_num].start_alignment_sample])[0:2]
        camera_pos_end = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[
                                             imu_tracker._tracking_data[sensor_num].end_alignment_sample])[
                         0:2]

        # if withAdaption:
        #    plt.scatter([camera_pos_start[0],camera_pos_end[0]], [camera_pos_start[1],camera_pos_end[1]], marker="x", label='alignment points', color=self._z_color, s=30)

        if withGeometry:
            visuWalls = np.array((
                [-2.5, 1.65],
                [0.00, 1.65],
                [0.00, 1.65],
                [0.00, 0.45],
                [0.00, 0.45],
                [2.40, 0.45],
                [-2.5, -1.65],
                [0.00, -1.65],
                [0.00, -1.65],
                [0.00, -0.45],
                [0.00, -0.45],
                [2.40, -0.45],
            ))
            for i in range(0, visuWalls.shape[0] - 1,
                           2):  # step = 2 to plot lines
                plt.plot([visuWalls[i, 0], visuWalls[i + 1, 0]],
                         [visuWalls[i, 1], visuWalls[i + 1, 1]],
                         linewidth=2,
                         color='black',
                         zorder=70
                         )

        plt.xlabel(r'x /m')
        plt.ylabel(r'y /m')
        plt.xlim(-3, 4)
        plt.ylim(-2, 2)
        # ax.set_xlim(-4, 4)
        # ax.set_ylim(-2, 2)

        plt.axis('equal')
        # plt.tight_layout()

        # ax = fig.gca()
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(1.00))
        # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

        plt.legend(loc='upper right')

        if self._save_figs:
            plt.savefig(
                self._output_file + self._title + '_resulting_trajectory_' + str(sensor_num) + '.' + self._format,
                format=self._format)

    def plot_trajectories(self, trajectories, labels, colors, index_range=None):
        print("figure #", self._figure_counter)
        plt.figure('Calculated trajectories')
        self._figure_counter += 1

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')

        for i in range(len(trajectories)):
            if index_range == None:
                start = 0
                end = -1
            elif len(trajectories) == 1:
                start = index_range[0]
                end = index_range[1]
            else:
                start = index_range[i][0]
                end = index_range[i][1]
            plt.scatter(trajectories[i].pos[start:end, 0], trajectories[i].pos[start:end, 1], color=colors[i],
                        edgecolors='none',
                        label=labels[i])

        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_trajectories' + '.' + self._format, format=self._format)

    def plot_trajectories3D(self, trajectories, labels, colors, index_range=None):
        print("figure #", self._figure_counter)
        fig = plt.figure('Calculated 3D trajectories')
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        for i in range(len(trajectories)):
            if index_range == None:
                start = 0
                end = -1
            elif len(trajectories) == 1:
                start = index_range[0]
                end = index_range[1]
            else:
                start = index_range[i][0]
                end = index_range[i][1]
            plt.plot(trajectories[i].pos[start:end, 0], trajectories[i].pos[start:end, 1],
                     trajectories[i].pos[start:end, 2], color=colors[i],
                     label=labels[i])

        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_trajectories3D' + '.' + self._format, format=self._format)

    def plot_acc_compare(self, imu_tracker, sensor_num=0):
        print("figure #", self._figure_counter)
        self._figure_counter += 1
        plt.figure('Acceleration of IMU and gt', figsize=(20, 20))

        plt.xlabel('Frame')
        plt.ylabel('Acceleration')

        sample_start = imu_tracker.tracking_data[sensor_num].start_tracking_index
        sample_end = imu_tracker.tracking_data[sensor_num].end_tracking_index

        frame_start = imu_tracker.tracking_data[sensor_num].imu_to_camera[sample_start]
        frame_end = imu_tracker.tracking_data[sensor_num].imu_to_camera[sample_end]
        gt = imu_tracker.ground_truth_trajectory
        start_index = gt.get_index_from_frame(frame_start)
        end_index = gt.get_index_from_frame(frame_end)

        plt.plot(gt.frames[start_index:end_index], gt.acc[start_index:end_index, 0], label='gt_acc_x',
                 color=self._x_color)
        plt.plot(gt.frames[start_index:end_index], gt.acc[start_index:end_index, 1], label='gt_acc_y',
                 color=self._y_color)

        imu_acc_x = []
        imu_acc_y = []
        for frame in range(frame_start, frame_end):
            index = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame]
            imu_acc_x.append(imu_tracker.tracking_data[sensor_num].acc_global_lin[index, 0])
            imu_acc_y.append(imu_tracker.tracking_data[sensor_num].acc_global_lin[index, 1])

        plt.plot(gt.frames[start_index:end_index], imu_acc_x, label='imu_acc_x', color=self._z_color)
        plt.plot(gt.frames[start_index:end_index], imu_acc_y, label='imu_acc_y', color='black')

        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_acc_compare' + '.' + self._format, format=self._format)

    def plot_v_compare(self, wheeler_traj, pusher_traj):
        print("figure #", self._figure_counter)
        self._figure_counter += 1
        plt.figure('Velocity of PeTrack Trajectories', figsize=(20, 20))

        plt.xlabel('Frame')
        plt.ylabel('Velocity')

        offset = wheeler_traj.fps

        plt.plot(wheeler_traj.frames[offset:-offset], wheeler_traj.v[offset:-offset, 0], label='wheeler_x',
                 color=self._z_color)
        plt.plot(wheeler_traj.frames[offset:-offset], wheeler_traj.v[offset:-offset, 1], label='wheeler_y',
                 color=self._y_color)
        plt.plot(wheeler_traj.frames[offset:-offset], [LA.norm(x) for x in wheeler_traj.v[offset:-offset, :]],
                 label='wheeler_abs', color=self._x_color)
        plt.plot(pusher_traj.frames[offset:-offset], pusher_traj.v[offset:-offset, 0], label='pusher_x',
                 color=self._z_color, ls='--')
        plt.plot(pusher_traj.frames[offset:-offset], pusher_traj.v[offset:-offset, 1], label='pusher_y',
                 color=self._y_color, ls='--')
        plt.plot(pusher_traj.frames[offset:-offset], [LA.norm(x) for x in pusher_traj.v[offset:-offset, :]],
                 label='pusher_abs', color=self._x_color, ls='--')
        plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_v_compare' + '.' + self._format, format=self._format)

    def plot_v_imu_tracker(self, imu_tracker):
        print("figure #", self._figure_counter)
        plt.figure('Calculated v for PeTrack trajectory')
        self._figure_counter += 1

        start_index_imu = imu_tracker.calculated_trajectory.start_sample
        imu_num_samples = imu_tracker.calculated_trajectory.num_samples
        end_index_imu = start_index_imu + imu_num_samples

        imu_samples = range(start_index_imu, end_index_imu)
        camera_frames = np.array([imu_tracker.imu_sample_to_frame(x) for x in imu_samples])

        print('start end imu: ', start_index_imu, end_index_imu)
        print('camera_frames ', camera_frames)

        surround_traj_shift = imu_tracker.surrounding_trajectory.frames[0]
        print('surounding_shift: ', surround_traj_shift)

        camera_frames_shifted = camera_frames - surround_traj_shift

        print(camera_frames_shifted)

        plt.xlabel('IMU samples')
        plt.ylabel('Velocity (m/s)')

        imu_v_norm = [LA.norm(x) for x in imu_tracker.calculated_trajectory.v]

        print('LEN imu samples: ', len(imu_samples))
        print(' LEN IMU_v: ', len(imu_tracker.calculated_trajectory.v))
        print('LEN surrounding_v: ', len(imu_tracker.surrounding_trajectory.v))

        surrounding_v_norm = []
        camera_samples_surrounding = []
        surrounding_v = None
        for index in camera_frames_shifted:
            index = int(index)
            if 0 <= index < len(imu_tracker.surrounding_trajectory.v):
                surrounding_v_norm.append(LA.norm(imu_tracker.surrounding_trajectory.v[index]))
                if surrounding_v is None:
                    surrounding_v = np.array([imu_tracker.surrounding_trajectory.v[index]])
                else:
                    surrounding_v = np.append(surrounding_v, [imu_tracker.surrounding_trajectory.v[index]], axis=0)
                camera_samples_surrounding.append(imu_tracker.frame_to_imu_sample(index + surround_traj_shift))

        # plot v ground truth
        gt_traj_shift = imu_tracker.ground_truth_trajectory.frames[0]
        print('gt_shift: ', gt_traj_shift)

        camera_frames_shifted = camera_frames - gt_traj_shift
        print(camera_frames_shifted)

        gt_v_norm = []
        camera_samples_gt = []
        gt_v = None
        for index in camera_frames_shifted:
            index = int(index)
            if 0 <= index < len(imu_tracker.ground_truth_trajectory.v):
                gt_v_norm.append(LA.norm(imu_tracker.ground_truth_trajectory.v[index]))
                if gt_v is None:
                    gt_v = np.array([imu_tracker.ground_truth_trajectory.v[index]])
                else:
                    gt_v = np.append(gt_v, [imu_tracker.ground_truth_trajectory.v[index]], axis=0)

                camera_samples_gt.append(imu_tracker.frame_to_imu_sample(index + gt_traj_shift))

        print('SURROUNDING V ', surrounding_v)
        print('V ', imu_tracker.surrounding_trajectory.v[0])
        print(imu_tracker.calculated_trajectory.v)

        ax1 = plt.subplot(211)
        plt.plot(imu_samples, imu_v_norm, label='norm(v_imu)')
        plt.plot(camera_samples_surrounding, surrounding_v_norm, label='norm(v_pusher)')
        plt.plot(camera_samples_gt, gt_v_norm, label='norm(v_wheeler)')
        plt.legend(loc='upper left')

        plt.subplot(212, sharex=ax1)
        plt.plot(imu_samples, imu_tracker.calculated_trajectory.v[:, 0], label='imu_v_x', marker='o')
        plt.plot(imu_samples, imu_tracker.calculated_trajectory.v[:, 1], label='imu_v_y', marker='o')

        plt.plot(camera_samples_surrounding, surrounding_v[:, 0], label='pusher_v_x', marker='o')
        plt.plot(camera_samples_surrounding, surrounding_v[:, 1], label='pusher_v_y', marker='o')

        plt.plot(camera_samples_gt, gt_v[:, 0], label='wheeler_v_x', marker='o')
        plt.plot(camera_samples_gt, gt_v[:, 1], label='wheeler_v_y', marker='o')

        plt.legend(loc='upper left')
        #
        # t_s=[t / 1000 for t in petrack_trajectory.t_ms[petrack_start:]]
        # plt.plot(t_s, petrack_trajectory.v[petrack_start:,0], label='v_x_pusher', color=self._y_color)
        # plt.plot(t_s, petrack_trajectory.v[petrack_start:, 1], label='v_y_pusher', color='cornflowerblue')
        #
        # print('t_ms: ',len(imu_trajectory.t_ms))
        # print('v: ', len(imu_trajectory.v))
        # t_s = [t / 1000 for t in imu_trajectory.t_ms]
        #
        # plt.plot(t_s, imu_trajectory.v[:, 0], label='v_x_imu', color=self._z_color)
        # plt.plot(t_s, imu_trajectory.v[:, 1], label='v_y_imu', color='purple')
        # #plt.plot(petrack_trajectory.t, petrack_trajectory.v[:, 2], label='v_z')
        # plt.margins(x=0.05, y=0.1)
        #
        # plt.xlim(75, 78.5)
        # plt.ylim(-0.3,0.2)
        # plt.legend(loc='upper left')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_v_imu_tracker' + '.' + self._format, format=self._format)

    def plot_v_correction(self, corrected_v_counter, v_range):
        print("figure #", self._figure_counter)
        plt.figure('Number of corrected v values')
        self._figure_counter += 1
        print('Summe: ', np.sum(corrected_v_counter))

        plt.xlabel('Time (s)')
        plt.ylabel('Number of corrections')
        seconds = np.arange(1, len(corrected_v_counter) + 1)
        plt.bar(seconds, corrected_v_counter)
        # plt.xticks(x + .5, ['a', 'b', 'c']);

        if self._save_figs:
            plt.savefig('./../output/TGF_plots/Num_v_corrections' + str(v_range) + self._format, format=self._format,
                        dpi=300)

    def animate_signal(i, imu_tracker):
        print('_animate_signal()')
        print('animation_index: ', i)

        start = imu_tracker.tracking_data[0].start_tracking_index
        start_q = start - imu_tracker.tracking_data[0].start_q_calc_index

        samples = np.divide(np.arange(start, i), 100.0)
        # print('samples: ', samples)

        x_color = sns.xkcd_rgb["denim blue"]
        y_color = 'green'  # sns.xkcd_rgb["medium green"]
        z_color = 'red'  # sns.xkcd_rgb["pale red"]

        imu_data = imu_tracker.imu_data[0]

        ax1 = plt.subplot(311)
        ax1.set_xlim(i / 100.0 - 3, i / 100.0 + 0.05)
        ax1.set_ylim(-1.1, 1.1)

        # plt.title('Acc, gyro and mag data')# for ', ImuData.sample_to_timecode(i, imu_data.sample_rate))
        plt.ylabel(r'Beschleunigung / $m/s^2$')
        plt.plot(samples,
                 imu_tracker.tracking_data[0].acc_local_lin[start_q:i - imu_tracker.tracking_data[0].start_q_calc_index,
                 0],
                 color=x_color)  # , label='acc_local_x')
        plt.plot(samples,
                 imu_tracker.tracking_data[0].acc_local_lin[start_q:i - imu_tracker.tracking_data[0].start_q_calc_index,
                 1],
                 color=y_color)  # , label='acc_local_y')
        plt.plot(samples,
                 imu_tracker.tracking_data[0].acc_local_lin[start_q:i - imu_tracker.tracking_data[0].start_q_calc_index,
                 2],
                 color=z_color)  # , label='acc_local_z')
        # plt.margins(x=0.05, y=0.1)
        # plt.legend(loc='upper right')

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_ylim(-20, 45)
        plt.ylabel(r'Drehrate / $\degree/s$')
        plt.plot(samples, imu_data.gyro_deg[start:i, 0], color=x_color)  # , label='gyro_deg_x')
        plt.plot(samples, imu_data.gyro_deg[start:i, 1], color=y_color)  # , label='gyro_deg_y')
        plt.plot(samples, imu_data.gyro_deg[start:i, 2], color=z_color)  # , label='gyro_deg_z')
        # plt.legend(loc='upper right')

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_ylim(-1010, 1010)
        plt.xlabel(r'Zeit / $s$')
        plt.ylabel(r'Magnetfeld / $mGs$')
        plt.plot(samples, imu_data.magnetic_field[start:i, 0], color=x_color)  # , label='magnetic_field_x')
        plt.plot(samples, imu_data.magnetic_field[start:i, 1], color=y_color)  # , label='magnetic_field_y')
        plt.plot(samples, imu_data.magnetic_field[start:i, 2], color=z_color)  # , label='magnetic_field_z')
        # plt.legend(loc='upper right')

    def plot_raw_signal_animation(self, imu_tracker):
        fig = plt.figure('Imu data animation', figsize=(10, 8))

        self._figure_counter += 1
        start = imu_tracker.tracking_data[0].start_tracking_index
        end = imu_tracker.tracking_data[0].end_tracking_index
        animation_range = range(start, end, 4)
        print('animation range: ', animation_range)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200)

        # fig: The figure object that is used to get draw, resize, and any other needed events.
        # func: The function to call at each frame. The first argument will be the next value in frames. Any additional positional arguments can be supplied via the fargs parameter.
        # frames: Source of data to pass func and each frame of the animation
        # interval: Delay between frames in milliseconds. Defaults to 200.
        ani = animation.FuncAnimation(fig, Plotter.animate_signal, animation_range, fargs=[imu_tracker], interval=40,
                                      repeat=False)

        ani.save(self._output_file + self._title + '_raw_signal_animation_' + str(start) + '-' + str(end) + '.mp4',
                 writer=writer)

        # plt.show()

    @staticmethod
    def _animate_imu_tracker_data(i, imu_tracker):
        print('animation_index: ', i)

        imu_start_index = 3006
        imu_range = imu_tracker.get_imu_fusion_range()
        offset = imu_start_index - imu_range[0]
        end = i - imu_start_index
        # samples = np.arange(imu_range[1]-imu_range[0])
        samples = np.arange(imu_start_index, i)

        # print('FUSION RANGE ', imu_range, ', offset: ', offset)

        # print('#SAMPLES: ', len(samples), samples)

        # print('#ACC_GLOBAL: ', len(imu_tracker.acc_global[offset:end, 0]))

        # print('#EULER: ', len(imu_tracker.euler_angles_deg[imu_range[0]+offset:i, 2]))

        ax1 = plt.subplot(311)
        ax1.set_xlim(i - 1000, i + 50)
        ax1.set_ylim(-3, 3.5)
        plt.ylabel('Global acceleration [m/s**2]')
        plt.plot(samples, imu_tracker.acc_global[offset:end, 0], color=self._x_color)
        plt.plot(samples, imu_tracker.acc_global[offset:end, 1], color=self._y_color)
        plt.plot(samples, imu_tracker.acc_global[offset:end, 2], color=self._z_color)

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_ylim(-1, 1)
        plt.ylabel('Velocity [m/s]')
        plt.plot(samples, imu_tracker.v[offset:end, 0], color=self._x_color)
        plt.plot(samples, imu_tracker.v[offset:end, 1], color=self._y_color)
        plt.plot(samples, imu_tracker.v[offset:end, 2], color=self._z_color)

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_ylim(-10, 70)
        plt.ylabel('Euler angle [deg]')
        plt.plot(samples, imu_tracker.euler_angles_deg[imu_range[0] + offset:i, 0], color=self._x_color)
        plt.plot(samples, imu_tracker.euler_angles_deg[imu_range[0] + offset:i, 1], color=self._y_color)
        plt.plot(samples, imu_tracker.euler_angles_deg[imu_range[0] + offset:i, 2], color=self._z_color)

        # plt.tight_layout()

    def plot_imu_tracker_animation(self, imu_tracker):

        fig = plt.figure('test')

        self._figure_counter += 1

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200)

        # fig: The figure object that is used to get draw, resize, and any other needed events.
        # func: The function to call at each frame. The first argument will be the next value in frames. Any additional positional arguments can be supplied via the fargs parameter.
        # frames: Source of data to pass func and each frame of the animation
        # interval: Delay between frames in milliseconds. Defaults to 200.len(imu_tracker.v[:,0])
        ani = animation.FuncAnimation(fig, Plotter._animate_imu_tracker_data, np.arange(3007, 4511, 4),
                                      fargs=[imu_tracker],
                                      interval=40)
        ani.save(self._output_file + 'imu_tracker_animation_3007-4511.mp4', writer=writer)

        # plt.show()

    def create_trajectory_animation(self, trajectory, title, tail_size=0):

        fig = plt.figure('trajectory animation')
        ax = fig.add_subplot(111)

        self._figure_counter += 1
        x_min = min(trajectory.pos[:, 0]) - 1
        x_max = max(trajectory.pos[:, 0]) + 1
        y_min = min(trajectory.pos[:, 1]) - 1
        y_max = max(trajectory.pos[:, 1]) + 1

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x /m')
        ax.set_ylabel('y /m')
        ax.set_aspect('equal')

        def animate_trajectory(i):

            if i % 200 == 0:
                print('animate trajectory, i=', i)
            plt.title('frame #' + str(i + int(trajectory.frames[0])) + ', ' + DataAccessManager.sample_to_timecode(
                i + int(trajectory.frames[0]), trajectory.fps))

            if i < tail_size or tail_size == 0:
                plt.plot(trajectory.pos[:i, 0], trajectory.pos[:i, 1], linewidth=3, color='black')
            else:
                plt.plot(trajectory.pos[i - tail_size:i, 0], trajectory.pos[i - tail_size:i, 1], linewidth=3,
                         color='black')

        Writer = animation.writers['ffmpeg']
        movie_fps = 25
        writer = Writer(fps=movie_fps, metadata=dict(artist='JS'), bitrate=7200)

        # had plotting issues, see: https://stackoverflow.com/questions/20694016/matplotlib-animation-funcanimation-frames-argument
        test = list(np.arange(0, 1500, int(trajectory.fps / movie_fps)))  # len(trajectory.pos[:, 0])
        ani = animation.FuncAnimation(fig, animate_trajectory, test, interval=int(1000 / trajectory.fps), repeat=False)
        print(self._output_file + self._title + '_trajectory_animation' + title + '.mp4')
        ani.save(self._output_file + self._title + '_trajectory_animation' + title + '.mp4', writer=writer)

    def create_trajectory_animation_with_gt(self, imu_tracker):

        fig = plt.figure('trajectory animation with gt', figsize=(10, 10))
        ax = fig.add_subplot(111)

        trajectory = imu_tracker.ground_truth_trajectory
        imu_trajectory = imu_tracker.imu_data[0].trajectory
        self._figure_counter += 1

        ax.set_xlim(-3, 4)
        ax.set_ylim(-3, 3)
        ax.set_xlabel(r'x / $m$')
        ax.set_ylabel(r'y / $m$')
        ax.set_aspect('equal')

        start = imu_tracker.tracking_data[0].start_tracking_index
        end = imu_tracker.tracking_data[0].end_tracking_index
        start_gt = imu_tracker.ground_truth_trajectory.get_index_from_frame(
            imu_tracker.tracking_data[0].imu_to_camera[start])

        if True:
            visuWalls = np.array((
                [-2.5, 1.65],
                [0.00, 1.65],
                [0.00, 1.65],
                [0.00, 0.45],
                [0.00, 0.45],
                [2.40, 0.45],
                [-2.5, -1.65],
                [0.00, -1.65],
                [0.00, -1.65],
                [0.00, -0.45],
                [0.00, -0.45],
                [2.40, -0.45],
            ))
            for i in range(0, visuWalls.shape[0] - 1,
                           2):  # step = 2 to plot lines
                plt.plot([visuWalls[i, 0], visuWalls[i + 1, 0]],
                         [visuWalls[i, 1], visuWalls[i + 1, 1]],
                         linewidth=2,
                         color='black',
                         zorder=70
                         )

        def animate_trajectory_with_gt(i):

            if i % 200 == 0:
                print('animate trajectory, i=', i)

            camera_frame = imu_tracker.tracking_data[0].imu_to_camera[i]
            gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)

            # plt.title('frame #' + str(i+int(trajectory.frames[0]))+', '+DataAccessManager.sample_to_timecode(i+int(trajectory.frames[0]), trajectory.fps))
            plt.plot(trajectory.pos[start_gt:gt_index, 0], trajectory.pos[start_gt:gt_index, 1], linewidth=2,
                     color='black')
            plt.plot(imu_trajectory.pos[0:i - start, 0], imu_trajectory.pos[0:i - start, 1], linewidth=2,
                     color=self._z_color)

            black_line = mlines.Line2D([], [], color='black', label='Kamera Trajektorie')
            blue_line = mlines.Line2D([], [], color=self._z_color, label='IMU Trajektorie')
            plt.legend(handles=[black_line, blue_line])
            plt.title('Zeit: {:2.2f}  s'.format(i / 100.0))

        Writer = animation.writers['ffmpeg']
        movie_fps = 25
        writer = Writer(fps=movie_fps, metadata=dict(artist='JS'), bitrate=7200)

        # had plotting issues, see: https://stackoverflow.com/questions/20694016/matplotlib-animation-funcanimation-frames-argument
        animation_range = range(start, end, 4)
        ani = animation.FuncAnimation(fig, animate_trajectory_with_gt, animation_range, interval=40, repeat=False)
        ani.save(self._output_file + self._title + '_trajectory_animation_with_gt.mp4', writer=writer)

    def plot_trajectory_rotation_animation_optitrack(self, imu_tracker, gt_heading, gt_left, gt_right, sample_range,
                                                     sensor_num=0):

        fig = plt.figure('trajectory rotation animation', figsize=(20, 9))
        # gs = gridspec.GridSpec(1,2, height_ratios=[1,1])
        self._figure_counter += 1

        def traj_animation(i):
            print('animate plot index: ', i)
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[i]
            future_range = 200
            gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)
            gt_left_index = gt_left.get_index_from_frame(camera_frame)
            gt_right_index = gt_right.get_index_from_frame(camera_frame)

            ax1 = plt.subplot(121)
            ax1.set_aspect(1)

            ax1.cla()
            ax1.set_xlim(-3, 4)
            ax1.set_ylim(-3, 3)
            ax1.set_aspect('equal')

            ax1.set_xlabel('x')
            ax1.set_ylabel('y')

            # plot gt
            ax1.plot(imu_tracker.ground_truth_trajectory.pos[gt_index:gt_index + future_range, 0],
                     imu_tracker.ground_truth_trajectory.pos[gt_index:gt_index + future_range, 1], color='black',
                     label='gt')

            # plot left and right gt
            ax1.plot(gt_left.pos[gt_left_index:gt_left_index + future_range, 0],
                     gt_left.pos[gt_left_index:gt_left_index + future_range, 1], color='grey',
                     label='gt_wheels')
            ax1.plot(gt_right.pos[gt_right_index:gt_right_index + future_range, 0],
                     gt_right.pos[gt_right_index:gt_right_index + future_range, 1], color='grey',
                     label='gt_wheels')

            # plot connecting line left right
            ax1.plot([gt_left.pos[gt_left_index, 0], gt_right.pos[gt_right_index, 0]],
                     [gt_left.pos[gt_left_index, 1], gt_right.pos[gt_right_index, 1]], color='grey', linestyle='-.')

            # plot local imu z-axis
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                origin = imu_tracker.ground_truth_trajectory.pos[gt_index, 0:2]

                q_index = i - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                # long line is drawn first, than tip (left/right)
                colors = ['r', 'r', 'r']

                # calc gt heading vector
                angle = math.radians(gt_heading[i - imu_tracker.tracking_data[sensor_num].start_tracking_index])
                gt_x = math.cos(angle)
                gt_y = math.sin(angle)

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax1.quiver(origin[0], origin[1], gt_x, gt_y, label=['gt'],
                           scale=5, scale_units='width', headwidth='5', color=['g', 'g', 'g'])
                ax1.quiver(origin[0], origin[1], current_local_frame[3][0], current_local_frame[3][1], label=['z'],
                           scale=10, scale_units='width', headwidth='5', color=colors)

            black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
            grey_line = mlines.Line2D([], [], color='grey', label='wheels')
            red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
            green_line = mlines.Line2D([], [], color=self._y_color, label='gt heading')
            plt.legend(handles=[black_line, grey_line, red_line, green_line])
            plt.title('sample #' + str(i))

            ax2 = plt.subplot(122)

            # plot animated angle diffs
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index, i)
                ax2.plot(samples,
                         imu_tracker._angle_diff[:i - imu_tracker.tracking_data[sensor_num].start_tracking_index],
                         color=self._x_color)
                ax2.set_xlabel('sample')
                ax2.set_ylabel('angle error in °')
                ax2.set_xlim(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                             imu_tracker.tracking_data[sensor_num].end_tracking_index)
                ax2.set_ylim(-30, 30)

                asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
                ax2.set_aspect(asp)

        tmp = np.arange(sample_range[0], sample_range[1], 4)
        ani = animation.FuncAnimation(fig, traj_animation, tmp, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200000)
        ani.save(self._output_file + self._title + '_traj_rotation_animation_25fps.mp4', writer=writer)

        print('animation created: ', self._output_file + 'local_frame_animation.mp4')

    def plot_trajectory_rotation_animation_petrack(self, imu_tracker, angle_diff, gt_heading, sample_range=None,
                                                   sensor_num=0):

        fig = plt.figure('trajectory rotation animation', figsize=(20, 9))
        # gs = gridspec.GridSpec(1,2, height_ratios=[1,1])
        self._figure_counter += 1

        if sample_range is None:
            sample_range = []
            sample_range.append(imu_tracker.tracking_data[sensor_num].start_tracking_index)
            # end=sample_range[-1]+100
            # sample_range.append(end)
            sample_range.append(imu_tracker.tracking_data[sensor_num].end_tracking_index)

        def traj_animation(i):
            if i % 200 == 0:
                print('animate trajectory, i=', i)
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[i]
            future_range = 50
            gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)

            ax1 = plt.subplot(121)
            ax1.set_aspect(1)

            ax1.cla()
            ax1.set_xlim(-3, 3)
            ax1.set_ylim(-3, 3)
            ax1.set_aspect('equal')

            ax1.set_xlabel('x')
            ax1.set_ylabel('y')

            # plot gt
            ax1.plot(imu_tracker.ground_truth_trajectory.pos[gt_index:gt_index + future_range, 0],
                     imu_tracker.ground_truth_trajectory.pos[gt_index:gt_index + future_range, 1], color='black',
                     label='gt')

            # plot local imu z-axis
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                origin = imu_tracker.ground_truth_trajectory.pos[gt_index, 0:2]

                q_index = i - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                # long line is drawn first, than tip (left/right)
                colors = ['r', 'r', 'r']

                # calc gt heading vector
                angle = math.radians(gt_heading[i - imu_tracker.tracking_data[sensor_num].start_tracking_index])
                gt_x = math.cos(angle)
                gt_y = math.sin(angle)

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax1.quiver(origin[0], origin[1], gt_x, gt_y, label=['gt'],
                           scale=5, scale_units='width', headwidth='5', color=['g', 'g', 'g'])
                ax1.quiver(origin[0], origin[1], current_local_frame[3][0], current_local_frame[3][1], label=['z'],
                           scale=10, scale_units='width', headwidth='5', color=colors)

            black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
            grey_line = mlines.Line2D([], [], color='grey', label='wheels')
            red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
            green_line = mlines.Line2D([], [], color=self._y_color, label='gt heading')
            plt.legend(handles=[black_line, grey_line, red_line, green_line])
            plt.title('sample #' + str(i))

            ax2 = plt.subplot(122)

            # plot animated angle diffs
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index, i)
                ax2.plot(samples, angle_diff[:i - imu_tracker.tracking_data[sensor_num].start_tracking_index],
                         color=self._x_color)
                ax2.set_xlabel('sample')
                ax2.set_ylabel('angle error in °')
                ax2.set_xlim(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                             imu_tracker.tracking_data[sensor_num].end_tracking_index)
                ax2.set_ylim(-90, 90)

                asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
                ax2.set_aspect(asp)

        tmp = list(np.arange(sample_range[0], sample_range[1]))
        ani = animation.FuncAnimation(fig, traj_animation, tmp, interval=40, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200000)
        ani.save(self._output_file + self._title + '_traj_rotation_animation_petrack40.mp4', writer=writer)

        print('animation created: ', self._output_file + 'local_frame_animation.mp4')

    def plot_heading_animation_petrack(self, imu_tracker, gt_heading, sample_range=None, sensor_num=0):

        fig = plt.figure('heading animation', figsize=(10, 10))
        # gs = gridspec.GridSpec(1,2, height_ratios=[1,1])
        self._figure_counter += 1

        if sample_range is None:
            sample_range = []
            sample_range.append(imu_tracker.tracking_data[sensor_num].start_tracking_index)
            # end=sample_range[-1]+100
            # sample_range.append(end)
            sample_range.append(imu_tracker.tracking_data[sensor_num].end_tracking_index)

        def heading_animation(i):
            if i % 200 == 0:
                print('animate heading, i=', i)
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[i]
            future_range = 25
            gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)
            gt_index_2 = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame - 12)

            ax1 = plt.subplot(111)
            ax1.set_aspect(1)

            ax1.cla()
            ax1.set_xlim(-3, 4)
            ax1.set_ylim(-3, 3)
            ax1.set_aspect('equal')

            if True:
                visuWalls = np.array((
                    [-3, 1.65],
                    [0.00, 1.65],
                    [0.00, 1.65],
                    [0.00, 0.45],
                    [0.00, 0.45],
                    [2.40, 0.45],
                    [-3, -1.65],
                    [0.00, -1.65],
                    [0.00, -1.65],
                    [0.00, -0.45],
                    [0.00, -0.45],
                    [2.40, -0.45],
                ))
                for j in range(0, visuWalls.shape[0] - 1,
                               2):  # step = 2 to plot lines
                    plt.plot([visuWalls[j, 0], visuWalls[j + 1, 0]],
                             [visuWalls[j, 1], visuWalls[j + 1, 1]],
                             linewidth=2,
                             color='black',
                             zorder=70
                             )

            ax1.set_xlabel(r'x / $m$')
            ax1.set_ylabel(r'y / $m$')

            if gt_index - future_range < 0:
                ax1.plot(imu_tracker.ground_truth_trajectory.pos[0: gt_index + future_range, 0],
                         imu_tracker.ground_truth_trajectory.pos[0:gt_index + future_range, 1],
                         color='black')
            else:
                ax1.plot(imu_tracker.ground_truth_trajectory.pos[gt_index - future_range: gt_index + future_range, 0],
                         imu_tracker.ground_truth_trajectory.pos[gt_index - future_range:gt_index + future_range, 1],
                         color='black')

            # plot local imu z-axis
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                # plot gt

                origin = imu_tracker.ground_truth_trajectory.pos[gt_index, 0:2]

                q_index = i - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                # long line is drawn first, than tip (left/right)
                colors = ['r', 'r', 'r']

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax1.quiver(origin[0], origin[1], current_local_frame[3][0], current_local_frame[3][1], label=['z'],
                           scale=10, scale_units='width', headwidth='5', color=colors)
                ax1.quiver(origin[0], origin[1], current_local_frame[2][0], current_local_frame[2][1], label=['y'],
                           scale=10, scale_units='width', headwidth='5', color=['g', 'g', 'g'])

            black_line = mlines.Line2D([], [], color='black', label='Kamera Trajektorie')
            red_line = mlines.Line2D([], [], color=self._z_color, label='Sensor z-Achse')
            green_line = mlines.Line2D([], [], color=self._y_color, label='Sensor y-Achse')
            plt.legend(handles=[black_line, red_line, green_line])
            plt.title('Zeit: {:2.2f}  s'.format(i / 100.0))

        animation_range = range(sample_range[0], sample_range[1], 4)
        ani = animation.FuncAnimation(fig, heading_animation, animation_range, interval=40, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200000)
        ani.save(self._output_file + self._title + '_heading_animation_petrack40.mp4', writer=writer)

        print('animation created: ', self._output_file + 'local_frame_animation.mp4')

    def plot_heading_animation_optitrack_without_alignment(self, imu_tracker, gt_left, gt_right, angle_offset,
                                                           sample_range=None, sensor_num=0):

        fig = plt.figure('heading animation', figsize=(10, 10))
        self._figure_counter += 1

        if sample_range is None:
            sample_range = []
            sample_range.append(imu_tracker.tracking_data[sensor_num].start_tracking_index)
            sample_range.append(imu_tracker.tracking_data[sensor_num].end_tracking_index)

        q_align = Quaternion.quaternion_from_angle_and_axis(-angle_offset, [0, 0, 1])

        def heading_animation(i):
            # if i % 200 == 0:
            # print('animate heading, i=', i)
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[i]
            gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)

            ax1 = plt.subplot(111)
            ax1.set_aspect(1)

            ax1.cla()
            # Session1
            # ax1.set_xlim(-1, 1)
            # ax1.set_ylim(-1, 1)
            # Session2
            ax1.set_xlim(-2, 2.5)
            ax1.set_ylim(-2, 2)
            ax1.set_aspect('equal')

            ax1.set_xlabel(r'x / $m$')
            ax1.set_ylabel(r'y / $m$')

            ax1.plot(imu_tracker.ground_truth_trajectory.pos[gt_index, 0],
                     imu_tracker.ground_truth_trajectory.pos[gt_index, 1], color='black')

            ax1.plot([gt_left.pos[gt_index, 0], gt_right.pos[gt_index, 0]],
                     [gt_left.pos[gt_index, 1], gt_right.pos[gt_index, 1]], color='black', linestyle='--')
            ax1.plot(gt_left.pos[gt_index, 0], gt_left.pos[gt_index, 1], marker='x', color='black')
            ax1.plot(gt_right.pos[gt_index, 0], gt_right.pos[gt_index, 1], marker='x', color='black')

            body_axis = gt_left.pos[gt_index, 0:2] - gt_right.pos[gt_index, 0:2]

            heading_angle = math.degrees(math.atan(body_axis[1] / body_axis[0]))

            # -90 < atan(y/x) < 90, convert to 360 deg
            if body_axis[0] < 0:
                heading_angle += 180
            elif body_axis[1] < 0:
                heading_angle += 360

            # plot local imu y-axis
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                # plot gt

                origin = imu_tracker.ground_truth_trajectory.pos[gt_index, 0:2]

                # rotate local axes with q_align
                q_index = i - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].global_imu_coordinate_axes[q_index]

                aligned_axes = []

                for axis in current_local_frame:
                    rotated = q_align.rotate_v(axis)
                    aligned_axes.append([rotated[0], rotated[1], rotated[2]])

                # long line is drawn first, than tip (left/right)
                colors = ['r', 'r', 'r']

                y_normalized = aligned_axes[2][0:2] / LA.norm(aligned_axes[2][0:2])
                z_normalized = aligned_axes[3][0:2] / LA.norm(aligned_axes[3][0:2])

                estimated_body_axis = math.degrees(math.atan(y_normalized[1] / y_normalized[0]))

                if y_normalized[1] < 0:
                    estimated_body_axis += 180
                elif y_normalized[0] < 0:
                    estimated_body_axis += 360

                diff = estimated_body_axis - heading_angle

                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax1.quiver(origin[0], origin[1], z_normalized[0], z_normalized[1], label=['z'],
                           scale=10, scale_units='width', headwidth='5', color=colors)
                ax1.quiver(origin[0], origin[1], y_normalized[0], y_normalized[1], label=['y'],
                           scale=10, scale_units='width', headwidth='5', color=['g', 'g', 'g'])

            # black_line = mlines.Line2D([], [], color='black', label='Kamera Trajektorie')
            red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
            green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
            plt.legend(handles=[red_line, green_line])
            plt.title('Zeit: {:2.2f}  s'.format(i / 100.0))

        animation_range = range(sample_range[0], sample_range[1], 4)
        ani = animation.FuncAnimation(fig, heading_animation, animation_range, interval=40, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200000)
        ani.save(self._output_file + self._title + '_heading_animation_optitrack_without_alignment.mp4', writer=writer)

        print('animation created: ', self._output_file + 'heading_animation_optitrack_without_alignment.mp4')

    def plot_heading_animation_optitrack(self, imu_tracker, gt_left, gt_right, sample_range=None, sensor_num=0):

        fig = plt.figure('heading animation', figsize=(10, 10))
        self._figure_counter += 1

        if sample_range is None:
            sample_range = []
            sample_range.append(imu_tracker.tracking_data[sensor_num].start_tracking_index)
            sample_range.append(imu_tracker.tracking_data[sensor_num].end_tracking_index)

        def heading_animation(i):
            if i % 200 == 0:
                print('animate heading, i=', i)
            camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[i]
            gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)

            ax1 = plt.subplot(111)
            ax1.set_aspect(1)

            ax1.cla()
            # Session1
            # ax1.set_xlim(-1, 1)
            # ax1.set_ylim(-1, 1)
            # other sessions
            ax1.set_xlim(-2, 2.5)
            ax1.set_ylim(-2, 2)
            ax1.set_aspect('equal')

            ax1.set_xlabel(r'x / $m$')
            ax1.set_ylabel(r'y / $m$')

            ax1.plot(imu_tracker.ground_truth_trajectory.pos[gt_index, 0],
                     imu_tracker.ground_truth_trajectory.pos[gt_index, 1], color='black')

            ax1.plot([gt_left.pos[gt_index, 0], gt_right.pos[gt_index, 0]],
                     [gt_left.pos[gt_index, 1], gt_right.pos[gt_index, 1]], color='black', linestyle='--')
            ax1.plot(gt_left.pos[gt_index, 0], gt_left.pos[gt_index, 1], marker='x', color='black')
            ax1.plot(gt_right.pos[gt_index, 0], gt_right.pos[gt_index, 1], marker='x', color='black')

            body_axis = gt_left.pos[gt_index, 0:2] - gt_right.pos[gt_index, 0:2]

            heading_angle = math.degrees(math.atan(body_axis[1] / body_axis[0]))

            # -90 < atan(y/x) < 90, convert to 360 deg
            if body_axis[0] < 0:
                heading_angle += 180
            elif body_axis[1] < 0:
                heading_angle += 360

            # plot local imu y-axis
            # if i is in tracking range
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= i >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                # plot gt

                origin = imu_tracker.ground_truth_trajectory.pos[gt_index, 0:2]

                # rotate local axes with q_align
                q_index = i - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                # long line is drawn first, than tip (left/right)
                colors = ['r', 'r', 'r']

                y_normalized = current_local_frame[2][0:2] / LA.norm(current_local_frame[2][0:2])
                z_normalized = current_local_frame[3][0:2] / LA.norm(current_local_frame[3][0:2])

                estimated_body_axis = math.degrees(math.atan(y_normalized[1] / y_normalized[0]))

                if y_normalized[1] < 0:
                    estimated_body_axis += 180
                elif y_normalized[0] < 0:
                    estimated_body_axis += 360

                diff = estimated_body_axis - heading_angle

                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax1.quiver(origin[0], origin[1], z_normalized[0], z_normalized[1], label=['z'],
                           scale=10, scale_units='width', headwidth='5', color=colors)
                ax1.quiver(origin[0], origin[1], y_normalized[0], y_normalized[1], label=['y'],
                           scale=10, scale_units='width', headwidth='5', color=['g', 'g', 'g'])

            # black_line = mlines.Line2D([], [], color='black', label='Kamera Trajektorie')
            red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
            green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
            plt.legend(handles=[red_line, green_line])
            plt.title('Time: {:2.2f}  s'.format(i / 100.0))

        animation_range = range(sample_range[0], sample_range[1], 4)
        ani = animation.FuncAnimation(fig, heading_animation, animation_range, interval=40, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='JS'), bitrate=7200000)
        ani.save(self._output_file + self._title + '_heading_animation_optitrack.mp4', writer=writer)

        print('animation created: ', self._output_file + 'heading_animation_optitrack.mp4')

    def plot_heading_validation(self, imu_tracker, gt_heading, axis_limits, stepsize=50, sample_range=None,
                                sensor_num=0):

        fig = plt.figure('trajectory heading validation sensor ' + self._title, figsize=(10, 6))
        self._figure_counter += 1

        if sample_range is None:
            sample_range = []
            sample_range.append(imu_tracker.tracking_data[sensor_num].start_tracking_index)
            # end=sample_range[-1]+100
            # sample_range.append(end)
            sample_range.append(imu_tracker.tracking_data[sensor_num].end_tracking_index)

        camera_frame_start = imu_tracker.tracking_data[sensor_num].imu_to_camera[sample_range[0]]
        camera_frame_end = imu_tracker.tracking_data[sensor_num].imu_to_camera[sample_range[1]]
        gt_index_start = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame_start)
        gt_index_end = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame_end)

        ax1 = plt.subplot(111)
        ax1.set_aspect(1)

        ax1.set_xlim(axis_limits[0], axis_limits[1])
        ax1.set_ylim(axis_limits[2], axis_limits[3])

        ax1.set_aspect('equal')

        ax1.set_xlabel(r'x / m')
        ax1.set_ylabel(r'y / m')

        # plot gt
        ax1.plot(imu_tracker.ground_truth_trajectory.pos[gt_index_start:gt_index_end, 0],
                 imu_tracker.ground_truth_trajectory.pos[gt_index_start:gt_index_end, 1], color='black')

        # plot local imu z-axis
        # if i is in tracking range
        for sample in range(sample_range[0], sample_range[1], stepsize):
            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                camera_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera[sample]
                gt_index = imu_tracker.ground_truth_trajectory.get_index_from_frame(camera_frame)
                origin = imu_tracker.ground_truth_trajectory.pos[gt_index, 0:2]

                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index][
                                          3][0:2]
                clf_norm = LA.norm(current_local_frame)

                # calc gt heading vector
                angle = math.radians(gt_heading[sample - imu_tracker.tracking_data[sensor_num].start_tracking_index])
                gt_x = math.cos(angle)
                gt_y = math.sin(angle)
                gt_norm = LA.norm([gt_x, gt_y])

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax1.quiver(origin[0], origin[1], gt_x / gt_norm, gt_y / gt_norm, label=['gt'], units='xy', scale=2.,
                           width=0.03, headwidth=3., headlength=4., color=['g', 'g', 'g'])
                ax1.quiver(origin[0], origin[1], current_local_frame[0] / clf_norm, current_local_frame[1] / clf_norm,
                           label=['z'], units='xy', scale=1., width=0.03, headwidth=3., headlength=4.,
                           linestyle='dashed', color=['r', 'r', 'r'])

        black_line = mlines.Line2D([], [], color='black', label='GT tajectory')
        red_line = mlines.Line2D([], [], color=self._z_color, label='IMU orientation')
        green_line = mlines.Line2D([], [], color=self._y_color, label='GT velocity')
        plt.legend(handles=[black_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_heading_validation' + str(sensor_num) + '.' + self._format,
                        format=self._format)

    def plot_vector(self, imu_tracker):

        print("figure #", self._figure_counter)
        plt.figure('Velocity vector')
        self._figure_counter += 1

        start_index_imu = imu_tracker.calculated_trajectory.start_sample
        imu_num_samples = imu_tracker.calculated_trajectory.num_samples
        end_index_imu = start_index_imu + imu_num_samples

        imu_samples = range(start_index_imu, end_index_imu)
        imu_v_norm = [LA.norm(x) for x in imu_tracker.calculated_trajectory.v]

        plt.plot(imu_samples, imu_v_norm)

    def plot_coord_system(self, coord_list, sample):

        print('figure #', self._figure_counter)
        fig = plt.figure('Coordinate system axes ')
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # long line is drawn first, than tip (left/right)
        colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

        # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
        ax.quiver(coord_list[0][0], coord_list[0][1], coord_list[0][2], coord_list[1:][:, 0],
                  coord_list[1:][:, 1], coord_list[1:][:, 2], label=['x', 'y', 'z'], color=colors)

        # plot global coordinate system axes
        ax.plot([0, 1], [0, 0], [0, 0], label='$X_camera$', linestyle="dashed", color='black')
        ax.plot([0, 0], [0, 1], [0, 0], label='$Y_camera$', linestyle="dashed", color='black')
        ax.plot([0, 0], [0, 0], [0, 1], label='$Z_camera$', linestyle="dashed", color='black')

        black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='local x-axis')
        green_line = mlines.Line2D([], [], color=self._y_color, label='local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
        plt.legend(handles=[black_line, blue_line, green_line, red_line])

        plt.title('Axes for sample ' + str(sample))

    def create_local_frames_animation(self, imu_tracker, range, stepsize=1, title_ext='_local_frame_animation'):
        print('figure #', self._figure_counter)
        fig = plt.figure('Local coord animation ', figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        def quiver_animation(i, coord_list):
            ax.cla()

            if i % 100 == 0:
                print('creating animation, i=', i)

            # to make local frame better visible. depends on inital alignment (Madgwick) --> location of data capturing
            ax.view_init(25, 40)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            # long line is drawn first, than tip (left/right)
            colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

            # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
            ax.quiver(coord_list[i][0][0], coord_list[i][0][1], coord_list[i][0][2], coord_list[i][1:][:, 0],
                      coord_list[i][1:][:, 1], coord_list[i][1:][:, 2], label=['x', 'y', 'z'], color=colors)

            # plot global coordinate system axes
            # ax.plot([0, 1], [0, 0], [0, 0], label='$X_camera$', linestyle="dashed", color='black')
            # ax.plot([0, 0], [0, 1], [0, 0], label='$Y_camera$', linestyle="dashed", color='black')
            # ax.plot([0, 0], [0, 0], [0, 1], label='$Z_camera$', linestyle="dashed", color='black')

            # black_line = mlines.Line2D([], [], color='black', label='global frame')
            blue_line = mlines.Line2D([], [], color=self._x_color, label='local x-axis')
            green_line = mlines.Line2D([], [], color=self._y_color, label='local y-axis')
            red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
            plt.legend(handles=[blue_line, green_line, red_line], loc='upper left')

            plt.title('sample ' + str(i))

        ani = animation.FuncAnimation(fig, quiver_animation,
                                      np.arange(range[0] - imu_tracker.tracking_data[0].start_q_calc_index,
                                                range[1] - imu_tracker.tracking_data[0].start_q_calc_index, stepsize),
                                      interval=1,
                                      fargs=[imu_tracker.tracking_data[0].global_imu_coordinate_axes])

        # Writer = animation.writers['ffmpeg']
        writer = animation.FFMpegFileWriter(fps=100, metadata=dict(artist='JS'), bitrate=7200)
        ani.save(self._output_file + self._title + title_ext + '.mp4', writer=writer)

        print('animation created: ', self._output_file + title_ext)

    def create_vector_rotation_animation(self, imu_tracker, vector_to_rotate, range, stepsize,
                                         title_ext='_vector_animation'):
        print('figure #', self._figure_counter)
        fig = plt.figure('Vector rot animation ')
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        def quiver_animation(i, coord_list):
            ax.cla()

            # to make local frame better visible. depends on inital alignment (Madgwick) --> location of data capturing
            ax.view_init(30, 140)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            q = imu_tracker.quaternions[i]
            rotated_vector = q.rotate_v(vector_to_rotate)
            rotated_vector = rotated_vector / LA.norm(rotated_vector)

            # long line is drawn first, than tip (left/right)
            colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

            # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
            ax.quiver(coord_list[i][0][0], coord_list[i][0][1], coord_list[i][0][2], coord_list[i][1:][:, 0],
                      coord_list[i][1:][:, 1], coord_list[i][1:][:, 2], label=['x', 'y', 'z'], color=colors)

            ax.quiver(coord_list[i][0][0], coord_list[i][0][1], coord_list[i][0][2], rotated_vector[0],
                      rotated_vector[1], rotated_vector[2], color='cyan')

            # plot global coordinate system axes
            ax.plot([0, 1], [0, 0], [0, 0], label='$X_camera$', linestyle="dashed", color='black')
            ax.plot([0, 0], [0, 1], [0, 0], label='$Y_camera$', linestyle="dashed", color='black')
            ax.plot([0, 0], [0, 0], [0, 1], label='$Z_camera$', linestyle="dashed", color='black')

            black_line = mlines.Line2D([], [], color='black', label='global frame')
            blue_line = mlines.Line2D([], [], color=self._x_color, label='local x-axis')
            green_line = mlines.Line2D([], [], color=self._y_color, label='local y-axis')
            red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
            plt.legend(handles=[black_line, blue_line, green_line, red_line], loc='upper left')

            plt.title('sample ' + str(i))

        ani = animation.FuncAnimation(fig, quiver_animation,
                                      np.arange(range[0] - imu_tracker.tracking_data[0].start_q_calc_index,
                                                range[1] - imu_tracker.tracking_data[0].start_q_calc_index, stepsize),
                                      interval=1,
                                      fargs=[imu_tracker._global_imu_coordinate_axes])

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=100, metadata=dict(artist='JS'), bitrate=7200)
        ani.save(self._output_file + self._title + title_ext + '.mp4', writer=writer)

        print('animation created: ', self._output_file + title_ext)

    def plot_gt_with_local_acc_data3D(self, imu_tracker, frame_range, step_size, sensor_num=0, additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with local acc data ')
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                 gt.pos[gt_start_index:gt_end_index, 2], color='black', label='gt')

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1],
                         traj.pos[start_index:end_index, 2], color='grey', label='shoulder')

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         [additional_traj[0].pos[index_first, 2], additional_traj[1].pos[index_second, 2]],
                         color='grey', linestyle='-.')

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end: ', end_imu)

        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                # print('len acc:' ,len(imu_tracker.imu_data[sensor_num].acc_local_filtered), ', len q: ', len(imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes))
                # print('q_index: ', q_index)
                current_local_frame = copy.deepcopy(
                    imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index])
                acc_local = imu_tracker.tracking_data[sensor_num].acc_local_lin[q_index]
                acc_global = imu_tracker.tracking_data[sensor_num].acc_global_lin[q_index]

                # print('q ', imu_tracker.tracking_data[sensor_num].quaternions[q_index])
                # print('acc_local: ', acc_local)
                # print('acc_global: ', acc_global)

                # acc_local /= LA.norm(acc_local)
                # acc_global /= LA.norm(acc_global)

                # print('acc_local normalized: ', acc_local)
                # print('acc_global normalized: ', acc_global)

                # scaling of unity coord systems
                current_local_frame[1] *= acc_local[0]
                current_local_frame[2] *= acc_local[1]
                current_local_frame[3] *= acc_local[2]

                # long line is drawn first, than tip (left/right)
                colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                scaling = 1  # 0.2
                scaling_low = 0.2

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], origin[2], current_local_frame[1:][:, 0] * scaling,
                          current_local_frame[1:][:, 1] * scaling, current_local_frame[1:][:, 2] * scaling,
                          label=['x', 'y', 'z'], color=colors)
                # ax.quiver(origin[0], origin[1], origin[2], acc_global[0]*scaling_low, acc_global[1]*scaling_low, acc_global[2]*scaling_low, color='cyan')

        black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='scaled acc_x')
        green_line = mlines.Line2D([], [], color=self._y_color, label='scaled acc_y')
        red_line = mlines.Line2D([], [], color=self._z_color, label='scaled acc_z')
        plt.legend(handles=[black_line, blue_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_acc' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_global_acc(self, imu_tracker, frame_range, step_size, additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with normalized partitioning of global acc')
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                 gt.pos[gt_start_index:gt_end_index, 2], color='black', label='gt')

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1],
                         traj.pos[start_index:end_index, 2], color='grey', label='shoulder')

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         [additional_traj[0].pos[index_first, 2], additional_traj[1].pos[index_second, 2]],
                         color='grey', linestyle='-.')

        start_imu = imu_tracker.camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.camera_to_imu[frame_range[1]]

        global_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.imu_end_tracking_index >= sample >= imu_tracker.imu_start_tracking_index:
                acc_index = sample - imu_tracker.tracking_data[0].start_q_calc_index
                current_global_acc = imu_tracker.acc_global_lin[acc_index]

                acc_normalized = current_global_acc / LA.norm(current_global_acc)

                # long line is drawn first, than tip (left/right)
                colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

                origin = gt.get_position(imu_tracker.imu_to_camera[sample])

                scaling = 0.2

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], origin[2], global_axes[:][0] * scaling * acc_normalized[0],
                          global_axes[:][1] * scaling * acc_normalized[1],
                          global_axes[:][2] * scaling * acc_normalized[2], label=['x', 'y', 'z'], color=colors)

        black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='global acc_x')
        green_line = mlines.Line2D([], [], color=self._y_color, label='global acc_y')
        red_line = mlines.Line2D([], [], color=self._z_color, label='global acc_z')
        plt.legend(handles=[black_line, blue_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_global_acc' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_local_frames(self, imu_tracker, frame_range, step_size, sensor_num=0):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with local frames for sensor #' + str(sensor_num))
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        sample_to_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera
        camera_frame_start = sample_to_frame[imu_tracker.tracking_data[sensor_num].start_tracking_index]
        camera_frame_end = sample_to_frame[imu_tracker.tracking_data[sensor_num].end_tracking_index]

        print('start frame: ', camera_frame_start, ', end: ', camera_frame_end)

        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                 gt.pos[gt_start_index:gt_end_index, 2], color='black', label='gt')

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        gravity_global = [0, 0, 9.81]

        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].global_imu_coordinate_axes[q_index]

                # long line is drawn first, than tip (left/right)
                colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                scaling = 0.2

                # local_gravity = imu_tracker.tracking_data[sensor_num].quaternions[q_index].get_inverse().rotate_v(
                #     gravity_global)
                # print('local gravity: ', local_gravity, ', local acc: ',
                #       imu_tracker.imu_data[sensor_num].acc_local_filtered[q_index])
                # global_acc = imu_tracker.tracking_data[sensor_num].quaternions[q_index].rotate_v(
                #     imu_tracker.imu_data[sensor_num].acc_local_filtered[sample] - local_gravity)
                # print('global lin acc: ', global_acc)

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], origin[2], current_local_frame[1:][:, 0] * scaling,
                          current_local_frame[1:][:, 1] * scaling, current_local_frame[1:][:, 2] * scaling,
                          label=['x', 'y', 'z'], color=colors)
            # ax.quiver(origin[0], origin[1], origin[2], global_acc[0] * scaling, global_acc[1] * scaling,
            #          global_acc[2] * scaling, color='cyan')

        black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='local x-axis')
        green_line = mlines.Line2D([], [], color=self._y_color, label='local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
        # cyan_line = mlines.Line2D([], [], color='cyan', label='globa_lin_acc')
        plt.legend(handles=[black_line, blue_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_global_frames' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_aligned_local_frames3D(self, imu_tracker, frame_range, limits, step_size, sensor_num=0,
                                            additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with aligned local frames for sensor #' + str(sensor_num), figsize=(10, 20))
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        # set elevation and azimuth angle
        ax.view_init(30, -24)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_zlim(limits[4], limits[5])
        ax.set_xlabel('x /m')
        ax.set_ylabel('y /m')
        ax.set_zlabel('z /m')

        # for Fig fig::05_3d_wheelchair_data
        # sns.set_style("whitegrid")#, {'axes.grid': False})
        # ax.grid(False)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                 gt.pos[gt_start_index:gt_end_index, 2], color='black', label='gt', zorder=10)

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1],
                         traj.pos[start_index:end_index, 2], color='grey', label='shoulder', zorder=5)

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         [additional_traj[0].pos[index_first, 2], additional_traj[1].pos[index_second, 2]],
                         color='grey', linestyle='-.', zorder=10)

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end_imu: ', end_imu)

        # WATCH OUT: only works for same step sizes / frame rate
        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                # long line is drawn first, than tip (left/right)
                colors = [self._x_color, self._y_color, self._z_color, self._x_color, self._x_color, self._y_color,
                          self._y_color, self._z_color, self._z_color]

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                scaling = 0.2

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], origin[2], current_local_frame[1:][:, 0] * scaling,
                          current_local_frame[1:][:, 1] * scaling, current_local_frame[1:][:, 2] * scaling,
                          label=['x', 'y', 'z'], color=colors, zorder=5)

        black_line = mlines.Line2D([], [], color='black', label='GT trajectory')
        grey_line = mlines.Line2D([], [], color='grey', label='Handles')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='Local x-axis')
        green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
        plt.legend(handles=[black_line, grey_line, blue_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_frames' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_aligned_local_frames3D_with_offset(self, imu_tracker, frame_range, offset_angle, limits, step_size,
                                                        sensor_num=0,
                                                        additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with aligned local frames for sensor #' + str(sensor_num), figsize=(10, 20))
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        # set elevation and azimuth angle
        ax.view_init(30, -24)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_zlim(limits[4], limits[5])
        ax.set_xlabel('x /m')
        ax.set_ylabel('y /m')
        ax.set_zlabel('z /m')

        q_align = Quaternion.quaternion_from_angle_and_axis(-offset_angle, [0, 0, 1])

        # for Fig fig::05_3d_wheelchair_data
        # sns.set_style("whitegrid")#, {'axes.grid': False})
        # ax.grid(False)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.zaxis.set_major_locator(MaxNLocator(integer=True))

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                 gt.pos[gt_start_index:gt_end_index, 2], color='black', label='gt', zorder=10)

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1],
                         traj.pos[start_index:end_index, 2], color='grey', label='shoulder', zorder=5)

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         [additional_traj[0].pos[index_first, 2], additional_traj[1].pos[index_second, 2]],
                         color='grey', linestyle='-.', zorder=10)

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end_imu: ', end_imu)

        # WATCH OUT: only works for same step sizes / frame rate
        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index

                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]
                aligned_axes = np.array([[0, 0, 0]])

                for axis in current_local_frame:
                    rotated = q_align.rotate_v(axis)
                    aligned_axes = np.append(aligned_axes, [[rotated[0], rotated[1], rotated[2]]], axis=0)

                # y_normalized = aligned_axes[2][0:2] / LA.norm(aligned_axes[2][0:2])
                # z_normalized = aligned_axes[3][0:2] / LA.norm(aligned_axes[3][0:2])

                # long line is drawn first, than tip (left/right)
                colors = [self._x_color, self._y_color, self._z_color, self._x_color, self._x_color, self._y_color,
                          self._y_color, self._z_color, self._z_color]

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                scaling = 0.2

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], origin[2], aligned_axes[1:][:, 0] * scaling,
                          aligned_axes[1:][:, 1] * scaling, aligned_axes[1:][:, 2] * scaling,
                          label=['x', 'y', 'z'], color=colors, zorder=5)

        black_line = mlines.Line2D([], [], color='black', label='GT trajectory')
        grey_line = mlines.Line2D([], [], color='grey', label='Handles')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='Local x-axis')
        green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
        plt.legend(handles=[black_line, grey_line, blue_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_frames_aligned3D' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_aligned_local_frames2D(self, imu_tracker, frame_range, limits, step_size, sensor_num=0,
                                            additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with aligned local frames 2D for sensor #' + str(sensor_num), figsize=(10, 6))
        ax = fig.add_subplot(111)
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        # set elevation and azimuth angle
        # ax.view_init(30, -24)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_xlabel('x /m')
        ax.set_ylabel('y /m')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1], color='black',
                 label='gt')

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1], color='grey',
                         label='shoulder')

            diff_between_surrounding_points = []

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         color='grey', linestyle='-.')

                diff = LA.norm(
                    np.subtract([additional_traj[0].pos[index_first, 0], additional_traj[0].pos[index_second, 1]],
                                [additional_traj[1].pos[index_first, 0], additional_traj[1].pos[index_second, 1]]))
                diff_between_surrounding_points.append(diff)

            print('Max and min diff between surrounding points: ', max(diff_between_surrounding_points),
                  min(diff_between_surrounding_points))

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end_imu: ', end_imu)

        # WATCH OUT: only works for same step sizes / frame rate
        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index

                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    q_index]

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                ax.quiver(origin[0], origin[1], current_local_frame[1:][1, 0], current_local_frame[1:][1, 1],
                          units='xy', scale=2, width=0.03, headwidth=3., headlength=4.,
                          color=['g', 'g', 'g'])
                ax.quiver(origin[0], origin[1], current_local_frame[1:][2, 0],
                          current_local_frame[1:][2, 1], units='xy', scale=2, width=0.03, headwidth=3., headlength=4.,
                          color=['r', 'r', 'r'])

        black_line = mlines.Line2D([], [], color='black', label='GT trajectory')
        grey_line = mlines.Line2D([], [], color='grey', label='Shoulders')
        green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
        plt.legend(handles=[black_line, grey_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_frames2D' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_aligned_local_frames2D_with_offset(self, imu_tracker, frame_range, offset_angle, limits, step_size,
                                                        sensor_num=0,
                                                        additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with aligned local frames 2D for sensor #' + str(sensor_num), figsize=(6, 6))
        ax = fig.add_subplot(111)
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        q_align = Quaternion.quaternion_from_angle_and_axis(-offset_angle, [0, 0, 1])

        ax.set_aspect('equal')
        # set elevation and azimuth angle
        # ax.view_init(30, -24)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_xlabel('x /m')
        ax.set_ylabel('y /m')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1], color='black',
                 label='gt')

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1], color='grey',
                         label='shoulder')

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         color='grey', linestyle='-.')

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end_imu: ', end_imu)

        # WATCH OUT: only works for same step sizes / frame rate
        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:

                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index

                local_frames = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]
                aligned_axes = []

                for axis in local_frames:
                    rotated = q_align.rotate_v(axis)
                    aligned_axes.append([rotated[0], rotated[1], rotated[2]])

                y_normalized = aligned_axes[2][0:2] / LA.norm(aligned_axes[2][0:2])
                z_normalized = aligned_axes[3][0:2] / LA.norm(aligned_axes[3][0:2])

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                ax.quiver(origin[0], origin[1], z_normalized[0], z_normalized[1], label=['z'],
                          scale=3, scale_units='width', headwidth='5', color=['r', 'r', 'r'])
                ax.quiver(origin[0], origin[1], y_normalized[0], y_normalized[1], label=['y'],
                          scale=3, scale_units='width', headwidth='5', color=['g', 'g', 'g'])

        # black_line = mlines.Line2D([], [], color='black', label='GT trajectory')
        grey_line = mlines.Line2D([], [], color='grey', label='Shoulder axis', linestyle='-.')
        green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
        plt.legend(handles=[grey_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_frames2D' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_aligned_local_frames2D_xz(self, imu_tracker, frame_range, limits, step_size, sensor_num=0,
                                               additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with aligned local xz frames 2D for sensor #' + str(sensor_num), figsize=(10, 6))
        ax = fig.add_subplot(111)
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')

        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_xlabel('x /m')
        ax.set_ylabel('z /m')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 2], color='black',
                 label='gt')

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 2], color='grey',
                         label='shoulder')

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 2], additional_traj[1].pos[index_second, 2]],
                         color='grey', linestyle='-.')

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end_imu: ', end_imu)

        # WATCH OUT: only works for same step sizes / frame rate
        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index

                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[
                    q_index]

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                ax.quiver(origin[0], origin[2], current_local_frame[1:][0, 0], current_local_frame[1:][0, 2],
                          units='xy', scale=2, width=0.03, headwidth=3., headlength=4.,
                          color=[self._x_color, self._x_color, self._x_color])
                ax.quiver(origin[0], origin[2], current_local_frame[1:][2, 0],
                          current_local_frame[1:][2, 2], units='xy', scale=2, width=0.03, headwidth=3., headlength=4.,
                          color=['r', 'r', 'r'])

        black_line = mlines.Line2D([], [], color='black', label='GT trajectory')
        grey_line = mlines.Line2D([], [], color='grey', label='Shoulders')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='Local x-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
        plt.legend(handles=[black_line, grey_line, blue_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_frames2D_xz' + '.' + self._format,
                        format=self._format)

    def plot_gt_with_aligned_local_yz(self, imu_tracker, frame_range, axis_limits, step_size=25, sensor_num=0):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with aligned local yz axis sensor ' + self._title, figsize=(10, 6))
        ax = fig.add_subplot(111)
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        # ax.set_xlim(-3,4)
        # ax.set_ylim(-1.5,1.5)
        # ax.set_xlim(-2,7)
        # ax.set_ylim(-2,3)
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

        ax.set_xlabel(r'x / m')
        ax.set_ylabel(r'y / m')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1], color='black',
                 label='gt')

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        print('start imu: ', start_imu, ', end_imu: ', end_imu)

        # WATCH OUT: only works for same step sizes / frame rate
        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:
                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], current_local_frame[1:][1, 0], current_local_frame[1:][1, 1],
                          label=['y', 'z'], units='xy', scale=2, width=0.03, headwidth=3., headlength=4.,
                          color=['g', 'g', 'g'])
                ax.quiver(origin[0], origin[1], current_local_frame[1:][2, 0],
                          current_local_frame[1:][2, 1], units='xy', scale=2, width=0.03, headwidth=3., headlength=4.,
                          color=['r', 'r', 'r'])

        black_line = mlines.Line2D([], [], color='black', label='GT trajectory')
        green_line = mlines.Line2D([], [], color=self._y_color, label='Local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='Local z-axis')
        # black_line = mlines.Line2D([], [], color='black', label='Kamera Trajektorie')
        # green_line = mlines.Line2D([], [], color=self._y_color, label='Sensor y-Achse')
        # red_line = mlines.Line2D([], [], color=self._z_color, label='Sensor z-Achse')
        plt.legend(handles=[black_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_gt_with_local_yz' + '.' + self._format, format=self._format)

    def plot_gt_with_aligned_global_frames3D(self, imu_tracker, frame_range, step_size, sensor_num=0,
                                             additional_traj=None):

        print('figure #', self._figure_counter)
        fig = plt.figure('gt with calculated global frames ')
        ax = fig.add_subplot(111, projection='3d')
        self._figure_counter += 1

        gt = imu_tracker.ground_truth_trajectory
        sample_to_frame = imu_tracker.tracking_data[sensor_num].imu_to_camera
        camera_frame_start = sample_to_frame[imu_tracker.tracking_data[sensor_num].start_tracking_index]
        camera_frame_end = sample_to_frame[imu_tracker.tracking_data[sensor_num].end_tracking_index]

        print('start frame: ', camera_frame_start, ', end: ', camera_frame_end)

        gt_start_index = gt.get_index_from_frame(frame_range[0])
        gt_end_index = gt.get_index_from_frame(frame_range[1])

        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.plot(gt.pos[gt_start_index:gt_end_index, 0], gt.pos[gt_start_index:gt_end_index, 1],
                 gt.pos[gt_start_index:gt_end_index, 2], color='black', label='gt')

        if additional_traj is not None:

            for traj in additional_traj:
                start_index = traj.get_index_from_frame(frame_range[0])
                end_index = traj.get_index_from_frame(frame_range[1])

                plt.plot(traj.pos[start_index:end_index, 0], traj.pos[start_index:end_index, 1],
                         traj.pos[start_index:end_index, 2], color='grey', label='shoulder')

            for sample in range(frame_range[0], frame_range[1], step_size):
                index_first = additional_traj[0].get_index_from_frame(sample)
                index_second = additional_traj[1].get_index_from_frame(sample)

                plt.plot([additional_traj[0].pos[index_first, 0], additional_traj[1].pos[index_second, 0]],
                         [additional_traj[0].pos[index_first, 1], additional_traj[1].pos[index_second, 1]],
                         [additional_traj[0].pos[index_first, 2], additional_traj[1].pos[index_second, 2]],
                         color='grey', linestyle='-.')

        start_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[0]]
        end_imu = imu_tracker.tracking_data[sensor_num].camera_to_imu[frame_range[1]]

        for sample in range(start_imu, end_imu, step_size):

            if imu_tracker.tracking_data[sensor_num].end_tracking_index >= sample >= imu_tracker.tracking_data[
                sensor_num].start_q_calc_index:

                q_index = sample - imu_tracker.tracking_data[sensor_num].start_q_calc_index
                current_local_frame = imu_tracker.tracking_data[sensor_num].aligned_global_imu_coordinate_axes[q_index]

                q = imu_tracker.tracking_data[sensor_num].quaternions_global[q_index]

                global_frame = np.zeros((4, 3))
                i = 0
                for axis in current_local_frame:
                    global_frame[i] = q.get_inverse().rotate_v(axis)
                    i += 1

                # long line is drawn first, than tip (left/right)
                colors = ['b', 'g', 'r', 'b', 'b', 'g', 'g', 'r', 'r']

                origin = gt.get_position(imu_tracker.tracking_data[sensor_num].imu_to_camera[sample])

                scaling = 0.2

                # fromat: <origin_x, n-times possible>, <origin_y>, <origin_z>, <dist_x, n-times>, <dist_y, n-times>, <dist_z, n-times> no end points!!
                ax.quiver(origin[0], origin[1], origin[2], global_frame[1:][:, 0] * scaling,
                          global_frame[1:][:, 1] * scaling, global_frame[1:][:, 2] * scaling, label=['x', 'y', 'z'],
                          color=colors)

        black_line = mlines.Line2D([], [], color='black', label='gt trajectory')
        blue_line = mlines.Line2D([], [], color=self._x_color, label='local x-axis')
        green_line = mlines.Line2D([], [], color=self._y_color, label='local y-axis')
        red_line = mlines.Line2D([], [], color=self._z_color, label='local z-axis')
        plt.legend(handles=[black_line, blue_line, green_line, red_line])

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_with_aligned_global_frames' + '.' + self._format,
                        format=self._format)

    def plot_adaptive_correction(self, imu_tracker, sensor_num=0):

        sns.set_style("white")

        print("figure #", self._figure_counter)
        plt.figure('Adaptive correction data ' + self._title + ' sensor #' + str(sensor_num), figsize=(10, 12))
        gs = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[3, 1, 3])

        self._figure_counter += 1

        start = imu_tracker.tracking_data[sensor_num].start_tracking_index  # - imu_tracker.start_q_calc_index
        end = imu_tracker.tracking_data[sensor_num].end_tracking_index  # - imu_tracker.start_q_calc_index
        samples = np.arange(imu_tracker.tracking_data[sensor_num].start_tracking_index,
                            imu_tracker.tracking_data[sensor_num].end_tracking_index)

        seconds = np.divide(samples, float(imu_tracker.imu_data[sensor_num].sample_rate))
        print('SECONDS: ', seconds)
        start_time = 26.5
        end_time = 28.0
        start_index = np.where(seconds == start_time)[0][0] + 3
        end_index = np.where(seconds == end_time)[0][0] - 1
        print("start index: ", start_index, ", end index: ", end_index)
        adaption_seconds = np.divide(imu_tracker.tracking_data[sensor_num].acc_adaption_corrected_samples,
                                     float(imu_tracker.imu_data[sensor_num].sample_rate))

        # shift to 0
        seconds = seconds - start_time
        adaption_seconds = adaption_seconds - start_time

        print("adaption times: ", adaption_seconds)

        # get surrounding velocity data

        camera_frames = np.array(
            [imu_tracker.tracking_data[sensor_num].imu_to_camera[x] for x in samples[start_index:end_index]])
        print('start end imu: ', samples[start_index], samples[end_index])
        print('camera_frames ', camera_frames)
        surround_traj_shift = imu_tracker.surrounding_trajectory.frames[0]
        print('surounding_shift: ', surround_traj_shift)

        surrounding_frame_index_shifted = camera_frames - surround_traj_shift

        print("frames shifted: ", surrounding_frame_index_shifted)

        print('LEN surrounding_v: ', len(imu_tracker.surrounding_trajectory._v))

        surrounding_v_norm = []
        camera_samples_surrounding = []
        surrounding_v = None
        for index in surrounding_frame_index_shifted:
            index = int(index)
            if 0 <= index < len(imu_tracker.surrounding_trajectory._v):
                surrounding_v_norm.append(LA.norm(imu_tracker.surrounding_trajectory._v[index, 0:2]))
                if surrounding_v is None:
                    surrounding_v = np.array([imu_tracker.surrounding_trajectory._v[index, 0:2]])
                else:
                    surrounding_v = np.append(surrounding_v, [imu_tracker.surrounding_trajectory._v[index, 0:2]],
                                              axis=0)
                camera_samples_surrounding.append(
                    imu_tracker.tracking_data[sensor_num].camera_to_imu[index + surround_traj_shift])

        # plot v ground truth
        gt_traj_shift = imu_tracker.ground_truth_trajectory.frames[0]
        print('gt_shift: ', gt_traj_shift)

        surrounding_frame_index_shifted = camera_frames - gt_traj_shift
        print(surrounding_frame_index_shifted)

        print('SURROUNDING V ', surrounding_v)
        print('V ', imu_tracker.surrounding_trajectory._v[0])

        start_q = start - imu_tracker.tracking_data[sensor_num].start_q_calc_index + start_index
        end_q = start_q + (end_index - start_index)
        print("start_q: ", start_q, ", end_q: ", end_q)
        linestyle_dashed = (0, (5, 1))

        diff = surrounding_v - imu_tracker.tracking_data[sensor_num].v_before_adaption[start_index:end_index, 0:2]

        OFFSET = 1.5
        SCALE = 1.5
        lw = 2

        ax1 = plt.subplot(gs[0])
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.ylabel('Velocity')
        # plt.plot(seconds[start_index:end_index],surrounding_v_norm,label='Norm surrounding')
        plt.plot(seconds[start_index:end_index],
                 surrounding_v[:, 0],
                 label=r'$v_{x}$ pusher', linewidth=lw)
        plt.plot(seconds[start_index:end_index],
                 imu_tracker.tracking_data[sensor_num].v_before_adaption[start_index:end_index, 0],
                 label=r'$v_{x}$ IMU', linewidth=lw)
        # plt.plot(seconds[start_index:end_index], imu_tracker.tracking_data[sensor_num].v_after_adaption[start_index:end_index, 0], label='corrected x')

        plt.plot(seconds[start_index:end_index],
                 surrounding_v[:, 1] + OFFSET,
                 label=r'$v_{y}$ pusher', linewidth=lw)
        plt.plot(seconds[start_index:end_index],
                 imu_tracker.tracking_data[sensor_num].v_before_adaption[start_index:end_index, 1] + OFFSET,
                 label=r'$v_{y}$ IMU', linewidth=lw)
        # plt.plot(seconds[start_index:end_index], imu_tracker.tracking_data[sensor_num].v_after_adaption[start_index:end_index, 1], label='corrected y')

        # plt.plot(seconds[start_index:end_index],
        #         [LA.norm(x) for x in imu_tracker.tracking_data[sensor_num].v_before_adaption[start_index:end_index, 0:2]],
        #         label='Norm IMU'),
        # plt.plot(seconds[start_index:end_index],diff,label='diff')

        plt.legend(loc='upper left', ncol=2)
        plt.ylim([0.8, 2.1])

        for adaption_time in adaption_seconds:
            if adaption_time > 0 and adaption_time < (end_time - start_time):
                plt.axvline(adaption_time, color='black', linewidth=lw / 2, linestyle='--')

        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.set_yticks([])
        ax2.fill_between(seconds[start_index:end_index], 0,
                         [LA.norm(x) for x in diff], facecolor="darkgray")
        ax2.set_ylabel(r'$|{v}_{\mathrm{diff}}|$')
        plt.ylim([0, 0.15])

        for adaption_time in adaption_seconds:
            if adaption_time > 0 and adaption_time < (end_time - start_time):
                plt.axvline(adaption_time, color='black', linewidth=lw / 2, linestyle='--')

        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.set_yticks([])

        plt.ylabel(r'Global acceleration')
        plt.plot(seconds[start_index:end_index],
                 imu_tracker.tracking_data[sensor_num].acc_before_adaption[start_q:end_q, 0],
                 label=r'$a_{x}$', linewidth=lw)
        plt.plot(seconds[start_index:end_index],
                 imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 0],
                 label=r'$a_{x}$ corrected', linestyle=linestyle_dashed, linewidth=lw)
        plt.plot(seconds[start_index:end_index],
                 imu_tracker.tracking_data[sensor_num].acc_before_adaption[start_q:end_q, 1] + OFFSET,
                 label=r'$a_{y}$', linewidth=lw)
        plt.plot(seconds[start_index:end_index],
                 imu_tracker.tracking_data[sensor_num].acc_global_lin[start_q:end_q, 1] + OFFSET,
                 label=r'$a_{y}$ corrected', linestyle=linestyle_dashed, linewidth=lw)
        plt.legend(loc='upper left', ncol=2)
        plt.ylim([-0.35, 2.7])

        for adaption_time in adaption_seconds:
            if adaption_time > 0 and adaption_time < (end_time - start_time):
                plt.axvline(adaption_time, color='black', linewidth=lw / 2, linestyle='--')

        # labels = [r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$"]
        # x=[0.09, 0, -0.52, -1.36]
        # x_shift=0.01
        # y = 110
        # for i in range(len(labels)):
        #     plt.axvline(x[i], color='black', linewidth=0.81, linestyle='--')
        #     plt.annotate(labels[i], xy=(x[i]+x_shift, y), rotation=0)

        # plt.ylim([-3,3])

        plt.xlabel(r'Time')
        # plt.xlabel(r'Zeit / $s$')

        if self._save_figs:
            plt.savefig(self._output_file + self._title + '_global_linear_acc_adaptive_correction' + str(
                sensor_num) + '.' + self._format, format=self._format)

        plt.close()
